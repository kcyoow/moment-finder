"""highlight_h2o: 첫 하이라이트 실험의 요청 B를 판정 모델(H2O)에 다시 묻는다. Modal에서 돈다. 준비 단계(2026-11-14)에서는 돌리지 않았다.

GPU 작업 하나가 H2O-Lightning-4B 서버(serve.sh: vLLM 0.30.0 + shim)를 같은 컨테이너 안에 띄우고, 정한 묶음을 묻고, 끝나면 스스로 꺼진다.
밖으로 열린 주소가 없다. 요청 본문은 hh_prepare.py(이 Mac의 미리 세기와 같은 코드)가 만든다. 구조는 charades_dense/cd_modal.py를 따랐다.

볼륨 (모두 <modal-profile> 계정)
  mf-highlight-h2o (/data)       : packs/<영상>.tar(data/packs 그대로), items.json, dryrun.json, frames_summary.json. 답은 results/<묶음>.json  ← Phase 2에서 새로 만든다
  mf-mc-dense (/mcd, 읽기만)     : 정상 확인 200곳의 프레임(packs/mc_08.tar, packs/ch_04.tar). 받아 둔 답은 이 Mac의 mc_dense/results/sep4_half.json과 견준다
  mf-models (/models)            : H2O 가중치(2026-11-12에 받아 둔 것, revision 659fc412…). 이 작업은 받지 않는다. 없으면 멈춘다
  mf-vllm-cache (/cache, 읽기만) : h2o_startup이 저장한 컴파일 캐시 tar(그래프 512 + 같은 VLLM_ARGS)

실행 (Phase 2, 관리자가 비용을 허락한 뒤에만):
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume create mf-highlight-h2o
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume put mf-highlight-h2o experiments/highlight_h2o/data/packs /packs
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume put mf-highlight-h2o experiments/highlight_h2o/data/items.json /items.json
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume put mf-highlight-h2o experiments/highlight_h2o/data/dryrun.json /dryrun.json
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume put mf-highlight-h2o experiments/highlight_h2o/data/frames_summary.json /frames_summary.json
  주 실험만(같은 434곳 + 정상 확인):     MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/highlight_h2o/hh_modal.py::judge_same
  덧붙인 실험까지(+ 영상 전체 1초마다):  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/highlight_h2o/hh_modal.py::judge_all
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal volume get mf-highlight-h2o results experiments/highlight_h2o/ --force
  uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_h2o/hh_eval.py h2o

시간 지키기: 묶음을 시작하기 전에 (미리 센 시간 × 앞 묶음에서 잰 배율)로 끝날 시각을 어림해 budget(함수 시작부터 초)을 넘으면 건너뛴다.
묶음마다 처음 48건을 재고, 잰 속도로 어림한 끝이 budget을 넘거나 속도가 예상(data/dryrun.json)의 절반에 못 미치면 그 묶음을 멈추고(답은 partial로 저장) 다음으로 간다.
묶음 순서: same434(주 실험) → sanity(정상 확인) → full(덧붙인 실험). 함수의 timeout이 마지막 안전장치다.
"""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hh_prepare as P  # noqa: E402

app = modal.App("moment-finder-highlight-h2o")
data = modal.Volume.from_name("mf-highlight-h2o")
mcd = modal.Volume.from_name("mf-mc-dense").read_only()
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # mc_dense·h2o_startup·charades_dense와 같은 판
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH = 512                                                 # CUDA_GRAPH_MAX_TOKENS (h2o_startup: 준비 시간 단축, 속도 그대로)
MAX_PIXELS = 2_000_000                                      # mc_dense·h2o_startup(정상 확인의 기준 답, 컴파일 캐시를 만든 실행)과 같은 값. 640×360(230,400화소)은 어느 한도에도 걸리지 않는다
PORT = 8741
WORKERS = 16                                                # mc_dense·h2o_startup이 속도를 잰 동시 요청 수
PRICE = P.PRICE                                             # $0.00101816/초 (RTX PRO 6000 + 8코어 + 32GiB, h2o_startup README)
# 함수 최대 시간(초)과 budget. 상한 비용 = timeout × PRICE
SAME_TIMEOUT, SAME_BUDGET = 600, 540                        # judge_same: 최대 약 $0.61
ALL_TIMEOUT, ALL_BUDGET = 900, 840                          # judge_all : 최대 약 $0.92

gpu_image = (   # mc_dense·h2o_startup·charades_dense와 같은 이미지(Modal에 이미 만들어져 있다) + 이 폴더의 hh_prepare.py
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
    .add_local_python_source("hh_prepare")
)
VOLS = {"/data": data, "/mcd": mcd, "/models": models, "/cache": cache_vol}

FRAMES = {}   # 영상 → {파일 이름: 바이트}. 정상 확인 영상(mc_08, ch_04)은 mf-mc-dense의 묶음에서
LAST = {}     # 정상 확인 영상 → 마지막 틱(_done.json)


def _get(video, name):
    return FRAMES[video][name]


def _build(job):
    """(묶음, 위치) → (본문 JSON 바이트, 그림 수). 프로세스 풀에서 돈다(fork라 FRAMES를 물려받는다)."""
    import json
    arm, it = job
    if arm == "sanity":
        payload = P.sanity_body(_get, it["video"], it["k"], LAST[it["video"]])
    else:
        payload = P.body(it, _get)
    return json.dumps(payload).encode(), len(payload["state"]["frames"])


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes=VOLS, timeout=SAME_TIMEOUT, cpu=8, memory=32768)
def judge_same(budget: int = SAME_BUDGET):
    return _judge("same434,sanity", budget, "judge_same")


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes=VOLS, timeout=ALL_TIMEOUT, cpu=8, memory=32768)
def judge_all(budget: int = ALL_BUDGET):
    return _judge("same434,sanity,full", budget, "judge_all")


def _judge(arms, budget, fn_name):
    import concurrent.futures as cf
    import hashlib
    import io
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

    # 1) 서버 띄우기 (컴파일 캐시를 먼저 푼다). charades_dense와 같은 설정, 그림 한도만 mc_dense·h2o_startup 값
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]   # = c32bae6555f8
    cache_tar = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(cache_tar)
    if restored:
        subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
    log = open("/tmp/serve.log", "w")
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 {'되살림' if restored else '없음'}: {cache_tar})", flush=True)

    # 2) 서버가 뜨는 동안: 묶음 파일을 메모리에 올리고(이 Mac에서 만든 tar와 같은지 해시로 확인) 모든 본문을 미리 만든다
    items = json.load(open("/data/items.json"))
    plan = json.load(open("/data/dryrun.json"))["arms"]
    expect = json.load(open("/data/frames_summary.json"))["packs"]
    names = [x for x in arms.split(",") if x]
    t0 = time.time()

    def load(path):
        raw = open(path, "rb").read()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            return path, hashlib.sha256(raw).hexdigest()[:16], {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    paths = [f"/data/packs/{v}.tar" for v in sorted(expect)]
    if "sanity" in names:
        paths += [f"/mcd/packs/{v}.tar" for v in P.SANITY_VIDEOS]
    bad_hash = []
    with cf.ThreadPoolExecutor(8) as ex:
        for path, digest, files in ex.map(load, paths):
            video = os.path.basename(path)[:-4]
            FRAMES[video] = files
            if path.startswith("/data/") and digest != expect[video]["sha256"]:
                bad_hash.append(video)
    for v in P.SANITY_VIDEOS:
        if v in FRAMES:
            LAST[v] = json.loads(FRAMES[v]["_done.json"])["last"]
    print(f"묶음 {len(paths)}개 올림 {time.time() - t0:.0f}초, {sum(len(b) for f in FRAMES.values() for b in f.values()) / 1e9:.2f}GB", flush=True)
    if bad_hash:
        server.terminate()
        raise RuntimeError(f"볼륨의 묶음이 이 Mac의 것과 다르다: {bad_hash}")

    units = {name: items[name] for name in names}
    t0 = time.time()
    built = {}
    with cf.ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        for name in names:
            built[name] = list(ex.map(_build, [(name, it) for it in units[name]], chunksize=8))
    print(f"본문 {sum(len(v) for v in built.values())}개 미리 만듦 {time.time() - t0:.0f}초, "
          f"{sum(len(b[0]) for v in built.values() for b in v) / 1e9:.2f}GB", flush=True)

    # 3) 준비 확인: serve.sh가 제 점검을 마쳐야 /health가 200이다(그 전에는 503 warming)
    health, last = None, ""
    while health is None:
        if server.poll() is not None or el() > min(budget - 60, 15 * 60):
            print(open("/tmp/serve.log").read()[-5000:])
            server.terminate()
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

    def ask(name, i, timeout=300):
        it, (body, n_img) = units[name][i], built[name][i]
        head = {"id": it["id"], "video": it["video"]}
        head.update({"k": it["k"]} if name == "sanity" else {"t": it["t"], "role": it["role"]})
        msg = ""
        for attempt in range(4):
            try:
                t = time.time()
                out = post(body, timeout)
                ans = {"probs": P.parse_sanity(out)} if name == "sanity" else P.parse(out)
                u = out.get("usage") or {}
                return {**head, **ans, "tokens": u.get("input_tokens"), "processed": u.get("processed_input_tokens"), "images": u.get("images"),
                        "ms": u.get("latency_ms"), "seconds": round(time.time() - t, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:200]
            time.sleep(2 ** attempt)
        return {**head, "error": msg}

    # 4) 형식 확인: 묶음마다 2건. same434가 실패하면 멈추고, 다른 묶음은 그 묶음만 건너뛴다
    ok, checks = [], {}
    for name in names:
        got = [ask(name, i) for i in range(2)]
        bad = [g.get("error") for g in got if "error" in g]
        checks[name] = {"errors": bad, "tokens": [g.get("tokens") for g in got], "images": [g.get("images") for g in got],
                        "plan_tokens_per_request": plan[name]["input_tokens_est_calibrated_median"]}
        print("형식 확인", name, json.dumps(checks[name], ensure_ascii=False), json.dumps(got[0], ensure_ascii=False)[:400], flush=True)
        if bad:
            if name == "same434":
                print(open("/tmp/serve.log").read()[-3000:])
                server.terminate()
                raise RuntimeError(f"형식 확인 실패: same434: {bad}")
            continue
        ok.append(name)

    # 5) 묻기
    os.makedirs("/data/results", exist_ok=True)
    summary, ratio = {"_checks": checks}, 1.0   # ratio: 잰 시간 ÷ 미리 센 시간(앞 묶음들 중 큰 값, 1 이상)
    for name in ok:
        est = plan[name]["est_seconds_conservative"]
        if el() + est * ratio > budget:
            summary[name] = {"skipped": f"시간: 지금 {el():.0f}초 + 어림 {est * ratio:.0f}초 > {budget}초"}
            print("건너뜀", name, summary[name], flush=True)
            continue
        recs, t0, stopped = [None] * len(units[name]), time.time(), None
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(ask, name, i): i for i in range(len(units[name]))}
            done = 0
            for fut in cf.as_completed(futs):
                if fut.cancelled():
                    continue
                recs[futs[fut]] = fut.result()
                done += 1
                if done == min(48, len(futs)) or done % 500 == 0:
                    rate = done / (time.time() - t0)
                    left = (len(futs) - done) / rate
                    print(f"  {name} {done}/{len(futs)} 초당 {rate:.1f}건, 남은 어림 {left:.0f}초, 지금 {el():.0f}초", flush=True)
                    slow = rate < 0.5 * len(futs) / est   # tasks/lessons.md(2026-11-12): 새 요청 모양이 예상 속도의 절반에 못 미치면 멈추고 원인부터 본다
                    if (el() + left > budget or slow) and stopped is None:
                        stopped = (f"{done}건 뒤 멈춤: " + (f"초당 {rate:.1f}건 < 예상의 절반" if slow else f"잰 속도로 {el() + left:.0f}초에 끝남"))
                        for f in futs:
                            f.cancel()
        wall = time.time() - t0
        recs = [r for r in recs if r is not None]
        good = [r for r in recs if "error" not in r]
        ratio = max(ratio, (wall / max(1, len(recs))) * len(units[name]) / max(1.0, est))
        toks = sorted(r["tokens"] or 0 for r in good)
        meta = {"arm": name, "function": fn_name, "requests": len(units[name]), "answered": len(good), "errors": len(recs) - len(good),
                "first_error": next((r["error"] for r in recs if "error" in r), None), "partial": stopped, "wall_seconds": round(wall, 1),
                "rate_per_second": round(len(recs) / max(wall, 1e-9), 2), "planned_seconds": est, "workers": WORKERS,
                "tokens_total": sum(toks), "tokens_median": toks[len(toks) // 2] if toks else None,
                "processed_total": sum(r["processed"] or 0 for r in good), "server": health[:600],
                "settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "CUDA_GRAPH_MAX_TOKENS", "MAX_PIXELS", "SHIM_MAX_PIXELS", "SHIM_TIMEOUT")},
                "compile_cache": cache_tar, "compile_cache_restored": restored, "gpu": gpu, "h2o_revision": H2O_REVISION, "pillow": PIL.__version__}
        with open(f"/data/results/{name}.json", "w") as f:
            json.dump({"meta": meta, "records": recs}, f)
        data.commit()
        summary[name] = {k: meta[k] for k in ("requests", "answered", "errors", "partial", "wall_seconds", "rate_per_second", "planned_seconds", "tokens_median", "first_error")}
        print("끝", name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    server.terminate()
    summary["_times"] = {"ready_seconds": round(t_ready), "total_seconds": round(el()), "compile_cache_restored": restored,
                         "usd_estimate": round(el() * PRICE, 3)}
    with open(f"/data/results/_summary_{fn_name}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    data.commit()
    print(json.dumps(summary, ensure_ascii=False))
    return summary
