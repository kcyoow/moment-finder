"""charades_dense: Charades에서 1초를 어떤 모양으로 넣어야 하는가(Q1, Q2)와, 경로를 그린 한 장이 여러 장을 대신하는가(Q3). Modal에서 돈다.

GPU 작업(judge) 하나가 H2O-Lightning-4B 서버(serve.sh: vLLM 0.30.0 + shim)를 같은 컨테이너 안에 띄우고, 모든 모양을 묻고, 끝나면 스스로 꺼진다.
밖으로 열린 주소가 없다. 모양·문장·그림 만들기는 cd_layouts.py(이 Mac의 미리 세기와 같은 코드)에 있다.

볼륨 (모두 <modal-profile> 계정)
  mf-charades-dense (/data) : packs/<영상>.tar(0.125초 격자 프레임), packs/q3_<영상>.tar(경로 그림과 (d)의 프레임), items.json, dryrun.json.
                              답은 results/<모양>.json, 요약은 results/_summary.json. 비공개 볼륨이다(Charades는 비상업 연구용)
  mf-models (/models)        : H2O 가중치(2026-11-12에 받아 둔 것, revision 659fc412…). 이 작업은 받지 않는다. 없으면 멈춘다
  mf-vllm-cache (/cache)     : h2o_startup이 저장한 컴파일 캐시 tar(그래프 512 + 같은 VLLM_ARGS). 읽기만 한다

실행 (Phase 2에서, 관리자가 허락한 뒤에만):
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/charades_dense/cd_modal.py::judge
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume get mf-charades-dense results experiments/charades_dense/ --force
  python3 experiments/charades_dense/cd_eval.py
인자: --layouts sep4,grid8_small (일부만), --limit 40 (모양마다 고르게 40건만. 결과 이름에 _tryN), --budget 2300 (초), --out results (볼륨의 결과 폴더)
덧붙임(2026-11-14): judge_short = sep4와 grid8_wide_small만, 함수 최대 760초, 결과는 볼륨의 wide/
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/charades_dense/cd_modal.py::judge_short
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume get mf-charades-dense wide experiments/charades_dense/results/

시간 지키기: 모양을 시작하기 전에 (미리 센 시간 × 앞 모양들에서 잰 배율)로 끝날 시각을 어림해 budget(함수 시작부터 초)을 넘으면 건너뛴다.
모양마다 처음 48건을 따로 재고, 잰 속도로 어림한 끝이 budget을 넘으면 그 모양을 멈추고(답은 partial로 저장) 다음으로 간다.
모양은 cd_layouts.LAYOUTS 순서(중요한 것부터)로 돈다. 함수의 timeout(2,600초)이 마지막 안전장치다.
"""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cd_layouts as L  # noqa: E402

app = modal.App("moment-finder-charades-dense")
data = modal.Volume.from_name("mf-charades-dense")
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # mc_dense·h2o_startup·prompt_study 기준 답과 같은 판
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH = 512                                                 # CUDA_GRAPH_MAX_TOKENS (h2o_startup: 준비 시간 단축, 속도 그대로)
TIMEOUT = 2600                                              # 함수 최대 시간(초). $0.00101784/초 → 최대 약 $2.65
BUDGET = 2300                                               # 이 시각(함수 시작부터 초)을 넘길 모양은 시작하지 않는다
PORT = 8741

gpu_image = (   # mc_dense·h2o_startup과 같은 이미지(Modal에 이미 만들어져 있다) + 이 폴더의 cd_layouts.py
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
    .add_local_python_source("cd_layouts")
)

FRAMES = {}   # 영상 → {파일 이름: 바이트}


def _get(video, name):
    return FRAMES[video][name]


def _build(unit):
    """(본문 JSON 바이트, 질문 키 표, 읽기 수, 그림 화소 합). 프로세스 풀에서 돈다(fork라 FRAMES를 물려받는다)."""
    import io
    import json
    from PIL import Image
    pics = L.pictures(unit, _get)
    payload, keys = L.body(unit, pics)
    px = sum(w * h for w, h in (Image.open(io.BytesIO(p)).size for p in pics))
    return json.dumps(payload).encode(), keys, len(payload["questions"]), px


VOLS = {"/data": data, "/models": models, "/cache": cache_vol}


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes=VOLS, timeout=TIMEOUT, cpu=8, memory=32768)
def judge(layouts: str = "", limit: int = 0, budget: int = BUDGET, out: str = "results"):
    return _judge(layouts, limit, budget, out)


# 덧붙임(2026-11-14): 모양 둘만 다시 묻는 짧은 실행. 함수 최대 760초(약 $0.77)
SHORT_TIMEOUT, SHORT_BUDGET = 760, 720


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes=VOLS, timeout=SHORT_TIMEOUT, cpu=8, memory=32768)
def judge_short(layouts: str = "sep4,grid8_wide_small", budget: int = SHORT_BUDGET, out: str = "wide"):
    return _judge(layouts, 0, budget, out)


def _judge(layouts, limit, budget, out):
    import concurrent.futures as cf
    import hashlib
    import json
    import multiprocessing as mp
    import os
    import subprocess
    import tarfile
    import time
    import urllib.error
    import urllib.request
    import PIL

    t_start = time.time()

    def el():
        return time.time() - t_start

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(gpu, flush=True)
    if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
        raise RuntimeError("mf-models에 H2O 가중치(정한 판)가 없다. 이 작업은 받지 않는다")

    # 1) 서버 띄우기 (컴파일 캐시를 먼저 푼다)
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]
    cache_tar = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(cache_tar)
    if restored:
        subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": str(L.CAP), "SHIM_MAX_PIXELS": str(L.CAP), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
    log = open("/tmp/serve.log", "w")
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 {'되살림' if restored else '없음'}: {cache_tar})", flush=True)

    # 2) 서버가 뜨는 동안: 묶음 파일을 메모리에 올리고 모든 본문을 미리 만든다
    meta = json.load(open("/data/items.json"))
    plan = {r["layout"]: r for r in json.load(open("/data/dryrun.json"))["layouts"]}
    names = [x for x in layouts.split(",") if x] or list(L.LAYOUTS)
    t0 = time.time()

    def load(path):
        with tarfile.open(path) as tar:
            return path, {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    packs = sorted(p for p in os.listdir("/data/packs") if p.endswith(".tar"))
    with cf.ThreadPoolExecutor(8) as ex:
        for path, files in ex.map(load, [f"/data/packs/{p}" for p in packs]):
            video = os.path.basename(path)[:-4].removeprefix("q3_")
            FRAMES.setdefault(video, {}).update(files)
    print(f"묶음 {len(packs)}개 올림 {time.time() - t0:.0f}초, {sum(len(b) for f in FRAMES.values() for b in f.values()) / 1e9:.2f}GB", flush=True)

    units = {}
    for name in names:
        src = meta["q3"] if L.LAYOUTS[name]["kind"] in L.Q3_KINDS else meta["items"]
        us = L.units(src, name)
        if limit:
            us = us[::max(1, len(us) // limit)][:limit]
        units[name] = us
    t0 = time.time()
    built = {}
    with cf.ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        for name in names:
            built[name] = list(ex.map(_build, units[name], chunksize=4))
    print(f"본문 {sum(len(v) for v in built.values())}개 미리 만듦 {time.time() - t0:.0f}초, "
          f"{sum(len(b[0]) for v in built.values() for b in v) / 1e9:.2f}GB", flush=True)

    # 3) 준비 확인: serve.sh가 제 점검을 마쳐야 /health가 200이다(그 전에는 503 warming)
    health, last = None, ""
    while health is None:
        if server.poll() is not None or el() > 20 * 60:
            print(open("/tmp/serve.log").read()[-5000:])
            raise RuntimeError(f"서버가 뜨지 못함: {last}")
        try:
            health = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=30).read().decode()
        except urllib.error.HTTPError as err:
            last = f"HTTP {err.code}"
            time.sleep(3)
        except Exception as err:
            last = f"{type(err).__name__}: {err}"[:200]
            time.sleep(3)
    t_ready = el()
    print(f"서버 준비 {t_ready:.0f}초 | {health[:500]}", flush=True)

    def post(body, timeout):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def ask(name, i, timeout=900):
        unit, (body, keys, reads, px) = units[name][i], built[name][i]
        msg = ""
        for attempt in range(4):
            try:
                t = time.time()
                out = post(body, timeout)
                rows, choices = L.parse_answers(keys, out)
                u = out.get("usage") or {}
                return {"unit": unit["id"], "set": unit["set"], "video": unit["video"], "answers": rows, "choices": choices, "reads": reads, "pixels": px,
                        "tokens": u.get("input_tokens"), "processed": u.get("processed_input_tokens"), "images": u.get("images"),
                        "ms": u.get("latency_ms"), "seconds": round(time.time() - t, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:200]
            time.sleep(2 ** attempt)
        return {"unit": unit["id"], "set": unit["set"], "video": unit["video"], "error": msg}

    # 4) 형식 확인: 모양마다 2건. sep4가 실패하면 멈추고, 다른 모양은 그 모양만 건너뛴다
    ok, checks = [], {}
    for name in names:
        got = [ask(name, i, 300) for i in range(min(2, len(units[name])))]
        bad = [g.get("error") for g in got if "error" in g]
        checks[name] = {"errors": bad, "tokens": [g.get("tokens") for g in got], "images": [g.get("images") for g in got],
                        "plan_tokens_per_request": plan.get(name, {}).get("input_tokens_per_request")}
        print("형식 확인", name, json.dumps(checks[name], ensure_ascii=False), json.dumps(got[0], ensure_ascii=False)[:300], flush=True)
        if bad:
            if name == "sep4":
                print(open("/tmp/serve.log").read()[-3000:])
                raise RuntimeError(f"형식 확인 실패: sep4: {bad}")
            continue
        ok.append(name)

    # 5) 묻기
    os.makedirs(f"/data/{out}", exist_ok=True)
    summary, ratio = {"_checks": checks}, 1.0   # ratio: 잰 시간 ÷ 미리 센 시간(앞 모양들 중 큰 값, 1 이상)
    for name in ok:
        est = plan.get(name, {}).get("est_seconds", 60) * (len(units[name]) / max(1, plan.get(name, {}).get("requests", len(units[name]))))
        if el() + est * ratio > budget:
            summary[name] = {"skipped": f"시간: 지금 {el():.0f}초 + 어림 {est * ratio:.0f}초 > {budget}초"}
            print("건너뜀", name, summary[name], flush=True)
            continue
        n_workers = L.WORKERS.get(name, 16)
        recs, t0, stopped = [None] * len(units[name]), time.time(), None
        with cf.ThreadPoolExecutor(n_workers) as ex:
            futs = {ex.submit(ask, name, i): i for i in range(len(units[name]))}
            done = 0
            for fut in cf.as_completed(futs):
                if fut.cancelled():
                    continue
                recs[futs[fut]] = fut.result()
                done += 1
                if done == min(48, len(futs)) or done % 300 == 0:
                    rate = done / (time.time() - t0)
                    left = (len(futs) - done) / rate
                    print(f"  {name} {done}/{len(futs)} 초당 {rate:.1f}건, 남은 어림 {left:.0f}초, 지금 {el():.0f}초", flush=True)
                    if el() + left > budget + 120 and stopped is None:
                        stopped = f"{done}건 뒤 멈춤: 잰 속도로 {el() + left:.0f}초에 끝남"
                        for f in futs:
                            f.cancel()
        wall = time.time() - t0
        recs = [r for r in recs if r is not None]
        good = [r for r in recs if "error" not in r]
        ratio = max(ratio, (wall / max(1, len(recs))) * len(units[name]) / max(1.0, est))
        toks = sorted(r["tokens"] or 0 for r in good)
        meta_out = {"layout": name, "spec": L.LAYOUTS[name], "requests": len(units[name]), "answered": len(good), "errors": len(recs) - len(good),
                    "first_error": next((r["error"] for r in recs if "error" in r), None), "partial": stopped, "wall_seconds": round(wall, 1),
                    "workers": n_workers, "limit": limit, "tokens_total": sum(toks), "processed_total": sum(r["processed"] or 0 for r in good),
                    "tokens_median": toks[len(toks) // 2] if toks else None, "planned_seconds": round(est), "server": health[:600],
                    "settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "CUDA_GRAPH_MAX_TOKENS", "MAX_PIXELS", "SHIM_MAX_PIXELS")},
                    "gpu": gpu, "h2o_revision": H2O_REVISION, "pillow": PIL.__version__}
        suffix = f"_try{limit}" if limit else ""
        with open(f"/data/{out}/{name}{suffix}.json", "w") as f:
            json.dump({"meta": meta_out, "records": recs}, f)
        data.commit()
        summary[name] = {k: meta_out[k] for k in ("requests", "answered", "errors", "partial", "wall_seconds", "planned_seconds", "tokens_median", "first_error")}
        print("끝", name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    server.terminate()
    summary["_times"] = {"ready_seconds": round(t_ready), "total_seconds": round(el()), "compile_cache_restored": restored,
                         "usd_estimate": round(el() * (0.000842 + 8 * 0.0000131 + 32 * 0.00000222), 2)}
    with open(f"/data/{out}/_summary{'_try' + str(limit) if limit else ''}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    data.commit()
    print(json.dumps(summary, ensure_ascii=False))
    return summary
