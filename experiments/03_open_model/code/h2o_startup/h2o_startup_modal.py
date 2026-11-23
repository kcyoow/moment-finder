"""H2O-Lightning-4B 서버(vLLM + shim)가 준비되는 시간을 줄이는 방법을 하나씩 잰다. Modal에서 돈다.

재는 것
- 준비 시간: serve.sh를 띄운 때부터 /health가 200을 줄 때까지(그 사이 shim이 뜬 때, vLLM이 뜬 때도 적는다)
- 답이 그대로인가: 마인크래프트 위치 200곳(지금 방법의 모양, 질문 4개)을 다시 물어 2026-11-12에 받아 둔 답과 견준다(확률 차이, 0.5를 넘나든 수)
- 처리 속도: 같은 모양 480건을 동시 16개로

바꿔 보는 것 (start_test의 인자)
- graph : CUDA_GRAPH_MAX_TOKENS. 한 방향 읽기의 기본은 2048이고 H2O 카드는 512면 더 빨리 뜬다고 적었다(L40S에서 391초 대 195초)
- cache : vLLM의 컴파일 캐시를 볼륨에 두었다가 되살린다. "save"면 뜬 뒤 저장만, "use"면 시작할 때 풀어 쓰고 끝나면 다시 저장
          볼륨은 작은 파일 읽기가 느려서 캐시 폴더를 tar 하나로 묶어 둔다
- eager : --enforce-eager (컴파일과 그래프를 모두 끔)

기준: 2026-11-12 mc_dense 실행들의 준비 시간 262초, 286초, 360초(graph 2048, 캐시 없음, 작업 시작부터 잰 값)

실행: MODAL_PROFILE=<modal-profile> modal run experiments/h2o_startup/h2o_startup_modal.py::start_test --graph 512 --cache save --tag g512_cold
읽는 볼륨: mf-models(H2O 가중치), mf-mc-dense(프레임 묶음과 받아 둔 답). 쓰는 볼륨: mf-vllm-cache(컴파일 캐시 tar, 결과 json)
"""
import modal

app = modal.App("moment-finder-h2o-startup")
data = modal.Volume.from_name("mf-mc-dense")
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache", create_if_missing=True)

TICKS = 20
CHECK_VIDEOS = ("mc_08", "ch_04")          # 창 구간이 많은 영상 둘
SENT = {
    "inv": "the player's inventory screen is open, the one that shows the player's character next to the armor slots",
    "chest": "a chest's storage screen is open, the one titled Chest or Large Chest that shows the chest's rows of item slots above the player's inventory rows",
}
HEAD = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). "
ASK = {"show": "Do these frames show the following: {s}?", "any": "Does at least one of these frames show the following: {s}?"}

gpu_image = (   # mc_dense와 같은 이미지
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
)


def _frames(vids):
    import tarfile
    out = {}
    for vid in vids:
        with tarfile.open(f"/data/packs/{vid}.tar") as tar:
            out[vid] = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    return out


def _body(frames, vid, k):
    """mc_dense의 sep4_half와 같은 요청: 0.5초 간격 4장을 따로, 질문 4개."""
    import base64
    import json
    last = json.loads(frames[vid]["_done.json"])["last"]
    ticks = [min(max(TICKS * k + t, 0), last) for t in (-5, 5, 15, 25)]
    uris = ["data:image/jpeg;base64," + base64.b64encode(frames[vid][f"{n:05d}.jpg"]).decode() for n in ticks]
    state = {"frames": [{"seconds": round(n / TICKS, 2), "frame": u} for n, u in zip(ticks, uris)]}
    qs = {f"{c}_{a}": {"type": "noul", "instructions": HEAD + ask.format(s=s)} for c, s in SENT.items() for a, ask in ASK.items()}
    return {"state": state, "questions": qs}


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models, "/cache": cache_vol}, timeout=20 * 60, cpu=8, memory=32768)
def start_test(graph: int = 2048, cache: str = "off", eager: bool = False, tag: str = "", extra: str = "", compare: str = "", sleep: bool = False, compile_threads: int = 0):
    import concurrent.futures as cf
    import hashlib
    import json
    import os
    import subprocess
    import time
    import urllib.error
    import urllib.request
    t_fn = time.time()
    vllm_args = "--max-num-batched-tokens 8192 --quantization fp8" + (" --enforce-eager" if eager else "") + (" " + extra if extra else "")
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{graph}|{vllm_args}".encode()).hexdigest()[:12]
    tar_path = f"/cache/vllm_cache_{key}.tar"
    restored = None
    if cache == "use" and os.path.exists(tar_path):
        t = time.time()
        subprocess.run(["tar", "-xf", tar_path, "-C", home], check=True)
        restored = {"seconds": round(time.time() - t, 1), "bytes": os.path.getsize(tar_path)}
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": vllm_args, "CUDA_GRAPH_MAX_TOKENS": str(graph),
           "MAX_PIXELS": "2000000", "SHIM_MAX_PIXELS": "2000000", "SHIM_QUESTION_THREADS": "512", "PORT": "8741"}
    if sleep:   # 잠자기·깨우기 주소를 연다(extra에 --enable-sleep-mode도 넣어야 한다)
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if compile_threads:   # Modal의 스냅샷 예제가 권하는 설정. 스냅샷 앱에서 느려진 까닭인지 보려고 따로 켠다
        env["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    log = open("/tmp/serve.log", "w")
    t0 = time.time()
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=log, stderr=subprocess.STDOUT, env=env)

    def code(url):
        try:
            return urllib.request.urlopen(url, timeout=10).status
        except urllib.error.HTTPError as err:
            return err.code
        except Exception:
            return 0

    marks, health = {}, None
    while health is None:
        if server.poll() is not None or time.time() - t0 > 15 * 60:
            print(open("/tmp/serve.log").read()[-5000:])
            raise RuntimeError("서버가 뜨지 못함")
        if "vllm_up" not in marks and code("http://127.0.0.1:8000/v1/models") == 200:
            marks["vllm_up"] = round(time.time() - t0, 1)
        if "shim_up" not in marks and code("http://127.0.0.1:8741/health?live=1") == 200:
            marks["shim_up"] = round(time.time() - t0, 1)
        if code("http://127.0.0.1:8741/health") == 200:
            marks["ready"] = round(time.time() - t0, 1)
            health = urllib.request.urlopen("http://127.0.0.1:8741/health", timeout=10).read().decode()
        else:
            time.sleep(1)
    print(f"준비 {marks} (작업 시작부터 {time.time() - t_fn:.0f}초) graph={graph} cache={cache} eager={eager}", flush=True)
    stages = [ln.strip()[:220] for ln in open("/tmp/serve.log") if any(w in ln for w in ("took", "Graph capturing", "compile", "Compiling", "warm", "ready on", "Loading weights", "cache"))][:40]

    # 캐시 저장 (볼륨에는 tar 하나로)
    saved = None
    if cache in ("save", "use"):
        root = os.path.join(home, ".cache", "vllm")
        if os.path.isdir(root):
            n_files = sum(len(fs) for _, _, fs in os.walk(root))
            subprocess.run(["tar", "-cf", tar_path + ".part", "-C", home, ".cache/vllm"], check=True)
            os.replace(tar_path + ".part", tar_path)
            cache_vol.commit()
            saved = {"files": n_files, "bytes": os.path.getsize(tar_path)}

    # 답이 그대로인가 + 처리 속도
    frames = _frames(CHECK_VIDEOS)
    old = {}
    for r in json.load(open("/data/results/sep4_half.json"))["records"]:
        if r["vid"] in CHECK_VIDEOS and "error" not in r:
            old[(r["vid"], r["k"])] = r["probs"]
    items = [(v, k) for v in CHECK_VIDEOS for k in range(100)]

    def ask(it):
        req = urllib.request.Request("http://127.0.0.1:8741/v1/systemone", data=json.dumps(_body(frames, *it)).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        return it, {q: float(a["noul"]) for q, a in out["answers"].items()}, (out.get("usage") or {}).get("latency_ms")

    one = [ask(it) for it in items[:20]]                          # 한 건씩: 서버 안 시간
    with cf.ThreadPoolExecutor(16) as ex:
        got = dict((it, p) for it, p, _ in ex.map(ask, items))    # 예열을 겸한 200건
        t = time.time()
        load = list(ex.map(ask, [(v, k) for v in CHECK_VIDEOS for k in range(60, 300)]))   # 480건
        rate = len(load) / (time.time() - t)
    after = None
    if sleep:   # 재웠다 깨운 뒤 같은 것을 다시 잰다: 속도가 떨어지는 까닭이 설정인지, 재웠다 깨운 것인지 가리려는 것
        def call(u):
            try:
                return urllib.request.urlopen(urllib.request.Request(u, method="POST", data=b""), timeout=120).status
            except urllib.error.HTTPError as err:
                return err.code
        s_code = call("http://127.0.0.1:8000/sleep?level=1")
        time.sleep(3)
        w_code = call("http://127.0.0.1:8000/wake_up")
        while code("http://127.0.0.1:8741/health") != 200:
            time.sleep(0.5)
        one2 = [ask(it) for it in items[:20]]
        lat2 = sorted(ms for _, _, ms in one2 if ms)
        rates = []
        with cf.ThreadPoolExecutor(16) as ex:
            for _ in range(2):
                t = time.time()
                n = len(list(ex.map(ask, [(v, k) for v in CHECK_VIDEOS for k in range(60, 300)])))
                rates.append(round(n / (time.time() - t), 1))
        after = {"sleep_code": s_code, "wake_code": w_code, "single_ms_median": lat2[len(lat2) // 2] if lat2 else None, "requests_per_s_16": rates}
    diffs = [abs(got[it][q] - old[it][q]) for it in items for q in got[it]]
    flips = sum((got[it][q] >= 0.5) != (old[it][q] >= 0.5) for it in items for q in got[it])
    answers = {f"{it[0]}|{it[1]}|{q}": round(p, 5) for it in items for q, p in got[it].items()}
    worst = sorted(((abs(got[it][q] - old[it][q]), it[0], it[1], q, round(old[it][q], 3), round(got[it][q], 3)) for it in items for q in got[it]), reverse=True)[:6]
    versus = None
    if compare and os.path.exists(f"/cache/results/{compare}.json"):   # 다른 실행의 답과 견주기(같은 설정이면 다시 물을 때의 흔들림)
        prev = json.load(open(f"/cache/results/{compare}.json")).get("answers") or {}
        both = [k for k in answers if k in prev]
        if both:
            d = [abs(answers[k] - prev[k]) for k in both]
            versus = {"run": compare, "answers": len(both), "max_diff": round(max(d), 4), "mean_diff": round(sum(d) / len(d), 4),
                      "flips": sum((answers[k] >= 0.5) != (prev[k] >= 0.5) for k in both)}
    lat = sorted(ms for _, _, ms in one if ms)
    out = {"tag": tag, "graph": graph, "cache": cache, "eager": eager, "vllm_args": vllm_args, "marks": marks, "function_to_ready": round(marks["ready"] + (t0 - t_fn), 1),
           "restored": restored, "saved": saved, "check": {"answers": len(diffs), "max_diff": round(max(diffs), 4), "mean_diff": round(sum(diffs) / len(diffs), 4), "flips": flips},
           "worst_vs_stored": worst, "versus": versus, "after_sleep_wake": after,
           "single_ms_median": lat[len(lat) // 2] if lat else None, "requests_per_s_16": round(rate, 1), "stages": stages, "health": health[:300], "answers": answers}
    server.terminate()
    os.makedirs("/cache/results", exist_ok=True)
    with open(f"/cache/results/{tag or key}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    cache_vol.commit()
    print(json.dumps({k: v for k, v in out.items() if k not in ("stages", "health", "answers")}, ensure_ascii=False))
    print("단계:", *stages[:25], sep="\n  ")
    return out
