"""Colab(GPU)에서 d1-2B의 요청 하나가 어디에 시간을 쓰는지, 동시에 처리하면 얼마나 빨라지는지 잰다. Jev 요청은 없다.

한 장짜리(mc_gui 64개, 그림 크기가 모두 같다)와 4장 묶음(charades 32개, 영상으로 넣기)에 대해 각각 잰다.
- 그림 파일을 읽고 푸는 시간만
- 하나씩 차례로(그림 읽기 포함), 그동안의 GPU 사용률, cProfile로 본 시간이 큰 함수
- system_one_batch로 8개, 32개씩
- 스레드 4개가 동시에
- 프로세스 2개, 3개가 동시에(모델을 그만큼 올린다)
모든 처리 속도는 그림 읽기를 포함한 "초당 항목 수"다.

실행 (Colab): python speed_probe_colab.py --pack /content/open/pack --out /content/open/results
결과: <out>/speed_probe.json
"""
import argparse
import cProfile
import json
import os
import pstats
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ONE = "The picture is one frame taken from a video. Does this frame show the following: {s}?"
VID = "The video clip is 4 consecutive frames from a video, 0.5 seconds apart. Does it show the following: {s}?"
COUNT = {"one": 64, "video": 32}


def pick(pack, kind):
    items = json.loads((pack / "manifest.json").read_text())["items"]
    part = [it for it in items if it["task"] == ("mc_gui" if kind == "one" else "charades")]
    return part[::len(part) // COUNT[kind]][:COUNT[kind]]


def build(item, pack, kind):
    from PIL import Image
    imgs = [Image.open(pack / p).convert("RGB") for p in item["images"]]
    qs = {f"q{k}": {"type": "noul", "instructions": (ONE if kind == "one" else VID).format(s=s)} for k, s in enumerate(item["questions"].values(), 1)}
    return None, qs, imgs if kind == "one" else [{"type": "video", "video": imgs, "fps": 2}]


def load():
    import torch
    from transformers import AutoModel
    return torch, AutoModel.from_pretrained("LiquidAI/d1-2B", trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()


def gpu_util(stop, out):
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        out.append(int(r.strip().split("\n")[0]))
        time.sleep(0.3)


def inproc(pack, out_path):
    torch, m = load()
    res = {"cpus": os.cpu_count(), "gpu": torch.cuda.get_device_name(0)}
    for kind in ("one", "video"):
        items, r = pick(pack, kind), {}

        def one(it):
            s, q, media = build(it, pack, kind)
            with torch.inference_mode():
                return m.system_one(s, q, media=media)
        for it in items[:3]:
            one(it)
        t0 = time.perf_counter()
        reqs = [build(it, pack, kind) for it in items]
        r["decode_ms_per_item"] = round((time.perf_counter() - t0) / len(items) * 1000, 1)

        stop, util = threading.Event(), []
        th = threading.Thread(target=gpu_util, args=(stop, util))
        th.start()
        t0 = time.perf_counter()
        for it in items:
            one(it)
        torch.cuda.synchronize()
        r["sequential_items_per_s"] = round(len(items) / (time.perf_counter() - t0), 1)
        stop.set()
        th.join()
        r["gpu_util_percent_mean"] = round(sum(util) / len(util)) if util else None

        prof = cProfile.Profile()
        prof.enable()
        for it in items[:16]:
            one(it)
        torch.cuda.synchronize()
        prof.disable()
        rows = sorted(pstats.Stats(prof).stats.items(), key=lambda kv: -kv[1][3])
        r["profile_top_ms_per_item"] = [[f"{Path(f).name}:{name}", round(ct / 16 * 1000, 1), round(tt / 16 * 1000, 1)]
                                        for (f, _, name), (_, _, tt, ct, _) in rows[:45] if "speed_probe" not in f][:28]
        own = sorted(pstats.Stats(prof).stats.items(), key=lambda kv: -kv[1][2])
        r["profile_own_time_top_ms_per_item"] = [[f"{Path(f).name}:{name}", round(tt / 16 * 1000, 1)] for (f, _, name), (_, _, tt, _, _) in own[:12]]

        for bs in (8, 32):
            try:
                t0 = time.perf_counter()
                for n in range(0, len(items), bs):
                    batch = [build(it, pack, kind) for it in items[n:n + bs]]
                    with torch.inference_mode():
                        m.system_one_batch(batch)
                torch.cuda.synchronize()
                r[f"batch{bs}_items_per_s"] = round(len(items) / (time.perf_counter() - t0), 1)
            except Exception as err:
                r[f"batch{bs}_items_per_s"] = f"{type(err).__name__}: {err}"[:200]
        try:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(one, items))
            torch.cuda.synchronize()
            r["threads4_items_per_s"] = round(len(items) / (time.perf_counter() - t0), 1)
        except Exception as err:
            r["threads4_items_per_s"] = f"{type(err).__name__}: {err}"[:200]
        res[kind] = r
        print(kind, json.dumps(r, ensure_ascii=False)[:1500], flush=True)
    res["torch_max_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


def worker(pack, kind, tag):
    torch, m = load()
    items = pick(pack, kind)
    for it in items[:3]:
        s, q, media = build(it, pack, kind)
        with torch.inference_mode():
            m.system_one(s, q, media=media)
    Path(f"/tmp/ready_{tag}").write_text("1")
    while not Path("/tmp/go").exists():
        time.sleep(0.05)
    for it in items:
        s, q, media = build(it, pack, kind)
        with torch.inference_mode():
            m.system_one(s, q, media=media)
    torch.cuda.synchronize()


def multiproc(pack, out_path):
    res = json.loads(out_path.read_text())
    for kind in ("one", "video"):
        for n in (2, 3):
            for f in Path("/tmp").glob("ready_*"):
                f.unlink()
            Path("/tmp/go").unlink(missing_ok=True)
            procs = [subprocess.Popen([sys.executable, __file__, "--pack", str(pack), "--worker", kind, "--tag", str(k)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for k in range(n)]
            deadline = time.time() + 300
            while len(list(Path("/tmp").glob("ready_*"))) < n and time.time() < deadline and all(p.poll() is None for p in procs):
                time.sleep(0.2)
            if len(list(Path("/tmp").glob("ready_*"))) < n:
                for p in procs:
                    p.kill()
                res[kind][f"processes{n}_items_per_s"] = "준비 실패(메모리 부족일 수 있다)"
                continue
            t0 = time.perf_counter()
            Path("/tmp/go").write_text("1")
            codes = [p.wait() for p in procs]
            res[kind][f"processes{n}_items_per_s"] = round(n * COUNT[kind] / (time.perf_counter() - t0), 1) if not any(codes) else f"실패 {codes}"
            print(kind, n, res[kind][f"processes{n}_items_per_s"], flush=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--phase", default="")
    ap.add_argument("--worker", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pack, out_path = Path(args.pack), Path(args.out) / "speed_probe.json"
    if args.worker:
        return worker(pack, args.worker, args.tag)
    if args.phase == "inproc":
        return inproc(pack, out_path)
    subprocess.run([sys.executable, __file__, "--pack", args.pack, "--out", args.out, "--phase", "inproc"], check=True)  # 끝나면 GPU 메모리가 비워진다
    multiproc(pack, out_path)
    print("끝", flush=True)


if __name__ == "__main__":
    main()
