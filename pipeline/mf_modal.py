"""pipeline: 준비한 실행(pipeline/runs/<이름>)을 판정 모델(H2O-Lightning-4B)에 묻는다. Modal(<modal-profile> 계정)에서 돈다.

GPU 작업 하나가 H2O 서버(serve.sh: vLLM + shim)를 같은 컨테이너 안에 띄우고, 실행들을 차례로 묻고, 끝나면 스스로 꺼진다.
서버 설정과 묻는 고리는 experiments/highlight_auto/ha_modal.py와 같다(한 방향 읽기, vLLM 패치, FP8, 그래프 512, 동시 16건).
2026-11-16에 로봇 실습 영상 두 편으로 한 번 돌렸다(1,489건, 오류 0, 319초. pipeline/README.md 11절).

볼륨 (모두 <modal-profile> 계정)
  mf-pipeline (/data)             : <이름>/pack.tar, <이름>/requests.json (mf.py prepare가 만든 것). 답은 <이름>/results.json
  mf-models (/models)             : H2O 가중치(revision 659fc412…). 이 작업은 받지 않는다. 없으면 멈춘다
  mf-vllm-cache (/cache, 읽기만)  : 컴파일 캐시 tar(그래프 512 + 같은 VLLM_ARGS)

실행 (비용을 말하고 승인받은 뒤. 이 Mac에서):
  MODAL_PROFILE=<modal-profile> modal volume create mf-pipeline                                  # 처음 한 번
  MODAL_PROFILE=<modal-profile> modal volume put mf-pipeline pipeline/runs/N/pack.tar /N/pack.tar
  MODAL_PROFILE=<modal-profile> modal volume put mf-pipeline pipeline/runs/N/requests.json /N/requests.json
  MODAL_PROFILE=<modal-profile> modal run pipeline/mf_modal.py::judge --names N --limit 48       # 먼저 48곳만(tasks/lessons.md 2026-11-12)
  MODAL_PROFILE=<modal-profile> modal run pipeline/mf_modal.py::judge --names N                  # 전체. 여러 실행은 --names A,B
  MODAL_PROFILE=<modal-profile> modal volume get mf-pipeline /N/results.json pipeline/runs/N/results.json
  (받은 뒤) MODAL_PROFILE=<modal-profile> modal volume rm -r mf-pipeline /N                      # 프레임을 남기지 않는다

값: RTX PRO 6000 + 8코어 + 32GiB = 초당 $0.001018 (modal.com/pricing, 2026-11-16 확인). 함수의 상한은 1,800초 = $1.83.
"""
import modal

app = modal.App("moment-finder-pipeline")
data = modal.Volume.from_name("mf-pipeline", create_if_missing=True)
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # 실험들과 같은 판
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH = 512
MAX_PIXELS = 2_000_000
PORT = 8741
WORKERS = 16
PRICE = 0.000842 + 8 * 0.0000131 + 32 * 0.00000222
TIMEOUT, BUDGET = 1800, 1700

gpu_image = (   # experiments/highlight_auto/ha_modal.py와 같은 이미지(로컬 모듈은 얹지 않는다)
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
)


def parse(out, questions):
    """서버 답 → {질문: {"noul": p} 또는 {"score": s, "probabilities": {...}} 또는 (고르기) {"probabilities": {선택지: p}}}. 형식이 이상하면 오류로 센다(ha_modal.parse와 같은 검사).
    고르기(choice)는 2026-11-23에 더했다(experiments/highlight_probe/hp_modal.py의 parse와 같은 검사). 다시 확인 단계가 쓴다."""
    ans, rec = out["answers"], {}
    for key, q in questions.items():
        a = ans[key]
        if q["type"] == "noul":
            p = float(a["noul"])
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{key}: 확률 {p}")
            rec[key] = {"noul": p}
        elif q["type"] == "choice":
            probs = {str(k): float(v) for k, v in a["probabilities"].items()}
            if sorted(probs) != sorted(q["criteria"]) or not abs(sum(probs.values()) - 1) <= 0.01:
                raise ValueError(f"{key}: 선택지 확률이 이상하다 {probs}")
            rec[key] = {"probabilities": {o: probs[o] for o in q["criteria"]}}
        else:
            probs = {str(k): float(v) for k, v in a["probabilities"].items()}
            levels = [str(i) for i in range(len(q["criteria"]))]
            if sorted(probs) != sorted(levels) or abs(sum(probs.values()) - 1) > 0.01:
                raise ValueError(f"{key}: 단계 확률이 이상하다 {probs}")
            expect = sum(int(k) * v for k, v in probs.items())
            if abs(expect - float(a["score"])) > 0.01:
                raise ValueError(f"{key}: score {a['score']}가 확률의 기댓값 {expect:.3f}과 다르다")
            rec[key] = {"score": float(a["score"]), "probabilities": probs}
    return rec


def ask_all(name, r, post, el, budget, workers=WORKERS, log=print):
    """실행 하나를 묻는다. r: {"req": 요청 글, "pos": 위치들, "bodies": 본문 바이트들}. post(본문, timeout) → 서버의 답.
    el() → 함수가 시작한 뒤 흐른 초. 처음 2건으로 형식을 확인하고 나머지를 동시 workers건으로 묻는다.
    돌려주는 것: (요약, 답들). 형식 확인이 실패하면 ({"skipped": …}, None).
    """
    import concurrent.futures as cf
    import json
    import time
    import urllib.error

    def ask(i, timeout=300):
        head, msg = {"k": r["pos"][i]["k"]}, ""
        for attempt in range(4):
            try:
                t = time.time()
                out = post(r["bodies"][i], timeout)
                u = out.get("usage") or {}
                return {**head, "answers": parse(out, r["req"]["questions"]), "tokens": u.get("input_tokens"), "images": u.get("images"), "seconds": round(time.time() - t, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:200]
            time.sleep(2 ** attempt)
        return {**head, "error": msg}

    n = len(r["pos"])
    first = [ask(i) for i in range(min(2, n))]
    bad = [g["error"] for g in first if "error" in g]
    log("형식 확인", name, json.dumps(first[0], ensure_ascii=False)[:300], flush=True)
    if bad:
        return {"skipped": f"형식 확인 실패: {bad[0]}"}, None
    recs, t0, stopped = [None] * n, time.time(), None
    for i, g in enumerate(first):
        recs[i] = g
    est = r["req"]["plan"]["est_seconds"] * (n / max(1, len(r["req"]["positions"])))
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(ask, i): i for i in range(len(first), n)}
        done = 0
        for fut in cf.as_completed(futs):
            if fut.cancelled():
                continue
            recs[futs[fut]] = fut.result()
            done += 1
            if done == min(48, len(futs)) or done % 500 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(futs) - done) / rate
                log(f"  {name} {done}/{len(futs)} 초당 {rate:.1f}건, 남은 어림 {left:.0f}초, 지금 {el():.0f}초", flush=True)
                slow = est > 30 and rate < 0.5 * n / est                    # 예상 속도의 절반에 못 미치면 멈춘다 (tasks/lessons.md 2026-11-12)
                if (el() + left > budget or slow) and stopped is None:
                    stopped = f"{done}건 뒤 멈춤: " + (f"초당 {rate:.1f}건 < 예상의 절반" if slow else f"잰 속도로 {el() + left:.0f}초에 끝남")
                    for f in futs:
                        f.cancel()
    wall = time.time() - t0
    recs = [x for x in recs if x is not None]
    good = [x for x in recs if "error" not in x]
    toks = sorted(x["tokens"] or 0 for x in good)
    meta = {"name": name, "requests": n, "answered": len(good), "errors": len(recs) - len(good), "first_error": next((x["error"] for x in recs if "error" in x), None),
            "partial": stopped, "wall_seconds": round(wall, 1), "rate_per_second": round(len(recs) / max(wall, 1e-9), 2),
            "tokens_median": toks[len(toks) // 2] if toks else None, "workers": workers}
    return meta, recs


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models, "/cache": cache_vol}, timeout=TIMEOUT, cpu=8, memory=32768)
def judge(names: str, limit: int = 0, budget: int = BUDGET):
    """names: 쉼표로 나눈 실행 이름. limit > 0이면 실행마다 앞의 그만큼만 묻는다(먼저 돌려 보기. 답은 results_limit.json에)."""
    import base64
    import hashlib
    import io
    import json
    import os
    import subprocess
    import tarfile
    import time
    import urllib.error
    import urllib.request

    t_start = time.time()

    def el():
        return time.time() - t_start

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(gpu, flush=True)
    if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
        raise RuntimeError("mf-models에 H2O 가중치(정한 판)가 없다. 이 작업은 받지 않는다")

    # 1) 서버 띄우기
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]
    cache_tar = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(cache_tar)
    if restored:
        subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=open("/tmp/serve.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 {'되살림' if restored else '없음'})", flush=True)

    # 2) 서버가 뜨는 동안: 요청 글과 묶음을 읽고 본문을 미리 만든다
    runs = {}
    for name in [x for x in names.split(",") if x]:
        req = json.load(open(f"/data/{name}/requests.json"))
        raw = open(f"/data/{name}/pack.tar", "rb").read()
        if hashlib.sha256(raw).hexdigest() != req["pack_sha256"]:
            server.terminate()
            raise RuntimeError(f"{name}: 볼륨의 묶음이 요청 글의 해시와 다르다")
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        pos = req["positions"][:limit] if limit else req["positions"]
        bodies = []
        for q in pos:
            frames = [{"seconds": s, "frame": "data:image/jpeg;base64," + base64.b64encode(files[n]).decode()} for n, s in zip(q["frames"], q["seconds"])]
            bodies.append(json.dumps({"state": {**req["state"], "frames": frames}, "questions": req["questions"]}).encode())
        runs[name] = {"req": req, "pos": pos, "bodies": bodies}
        print(f"{name}: 위치 {len(pos)}곳, 본문 {sum(len(b) for b in bodies) / 1e6:.0f}MB ({el():.0f}초)", flush=True)

    # 3) 준비 확인
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
    print(f"서버 준비 {t_ready:.0f}초 | {health[:300]}", flush=True)

    def post(body, timeout):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # 4) 묻기
    summary = {}
    extra = {"gpu": gpu, "h2o_revision": H2O_REVISION, "compile_cache_restored": restored, "ready_seconds": round(t_ready), "limit": limit,
             "settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "CUDA_GRAPH_MAX_TOKENS", "MAX_PIXELS", "SHIM_MAX_PIXELS")}}
    for name, r in runs.items():
        meta, recs = ask_all(name, r, post, el, budget)
        if recs is None:
            summary[name] = meta
            print(open("/tmp/serve.log").read()[-2000:])
            continue
        with open(f"/data/{name}/{'results_limit' if limit else 'results'}.json", "w") as f:
            json.dump({"meta": {**meta, **extra}, "records": recs}, f)
        data.commit()
        summary[name] = {k: meta[k] for k in ("requests", "answered", "errors", "partial", "wall_seconds", "rate_per_second", "tokens_median", "first_error")}
        print("끝", name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    server.terminate()
    summary["_times"] = {"ready_seconds": round(t_ready), "total_seconds": round(el()), "usd_estimate": round(el() * PRICE, 3)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary
