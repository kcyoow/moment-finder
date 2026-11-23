"""같은 후보와 같은 질문을 판정 모델 H2O-Lightning-4B에 묻는다. README.md 10절. Modal(<modal-profile> 계정)에서 돈다.

서버 설정은 pipeline/mf_modal.py와 같다(한 방향 읽기, vLLM 패치, FP8, 그래프 512, 동시 16건). 다른 점: MAX_IMAGES=8, 컴파일 캐시 없음.
입력 두 가지: h2o8(8장을 한 요청에), h2o4(연속 4장 요청 + 고르게 4장 요청).

볼륨 (<modal-profile> 계정. 끝나면 둘 다 지운다)
  mf-da-models (/models) : H2O 가중치(revision 659fc412…). fetch_model이 받는다
  mf-da-data (/data)     : pack.tar(프레임), requests.json(후보와 질문). main이 올린다

실행 (이 Mac에서, experiments/decision_api 폴더에서):
  MODAL_PROFILE=<modal-profile> modal run da_h2o_modal.py::fetch_model          # 가중치 받기 (CPU)
  MODAL_PROFILE=<modal-profile> modal run da_h2o_modal.py                       # 올리고, 묻고, results/h2o.json에 적는다
  MODAL_PROFILE=<modal-profile> modal volume delete mf-da-data --yes ; MODAL_PROFILE=<modal-profile> modal volume delete mf-da-models --yes

값: RTX PRO 6000 + 8코어 + 32GiB = 초당 $0.001018. 함수의 상한 1,500초 = $1.53.
"""
import modal

app = modal.App("moment-finder-decision-api")
models = modal.Volume.from_name("mf-da-models", create_if_missing=True)
data = modal.Volume.from_name("mf-da-data", create_if_missing=True)

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH, MAX_PIXELS, PORT, WORKERS = 512, 2_000_000, 8741, 16
PRICE = 0.000842 + 8 * 0.0000131 + 32 * 0.00000222
TIMEOUT, BUDGET = 1500, 1380
LEAD8 = ("These 8 frames come from one candidate segment (about {n} seconds long) of a longer video. "
         "Frames 1 to 4 are consecutive frames, 0.5 seconds apart, from one second of the segment. "
         "Frames 5 to 8 are spread evenly across the whole segment, in time order.")
LEAD_A = "These 4 frames are consecutive frames, 0.5 seconds apart, from one second of one candidate segment (about {n} seconds long) of a longer video."
LEAD_B = "These 4 frames are spread evenly across one candidate segment (about {n} seconds long) of a longer video, in time order."
PLAIN = "Does this segment show what the viewer asked for: {vr}?"
CHOICE = "Which option best describes what happens in this segment?"
NONE = "None of the other options is visible in these frames."

gpu_image = (   # pipeline/mf_modal.py와 같은 이미지
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
)
hf_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")


@app.function(image=hf_image, volumes={"/models": models}, timeout=3600, cpu=4)
def fetch_model():
    """H2O 가중치를 볼륨에 받아 둔다(GPU가 떠 있는 동안 받지 않으려는 것). 이미 있으면 아무것도 하지 않는다."""
    import os
    import time
    from huggingface_hub import snapshot_download
    if os.path.exists("/models/h2o/serve.sh") and os.path.exists("/models/h2o/_complete"):
        print("이미 있음")
        return "있음"
    t0 = time.time()
    for attempt in range(5):
        try:
            snapshot_download("h2oai/h2o-lightning-4b", local_dir="/models/h2o", revision=H2O_REVISION, max_workers=16)
            break
        except Exception as err:
            print(f"모델 받기 실패 {attempt + 1}/5: {str(err)[:200]}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError("모델을 받지 못함")
    open("/models/h2o/_complete", "w").write(H2O_REVISION)
    models.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("/models/h2o") for f in fs)
    print(f"받음: {size / 1e9:.1f}GB, {time.time() - t0:.0f}초")
    return f"{size / 1e9:.1f}GB"


def questions_of(spec):
    opts = [(spec["target"]["key"], spec["target"]["is"])] + [(c["key"], c["is"]) for c in spec["confusers"]] + [("none_of_these", NONE)]
    return {"noul_plain": {"type": "noul", "instructions": PLAIN.format(vr=spec["viewer_request"])},
            "noul_def": {"type": "noul", "instructions": spec["noul"]["instructions"], "criteria": spec["noul"]["criteria"]},
            "choice_fwd": {"type": "choice", "instructions": CHOICE, "criteria": dict(opts)},
            "choice_rev": {"type": "choice", "instructions": CHOICE, "criteria": dict(reversed(opts))}}


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models}, timeout=TIMEOUT, cpu=8, memory=32768)
def judge(limit: int = 0):
    import base64
    import concurrent.futures as cf
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
    el = lambda: time.time() - t_start
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(gpu, flush=True)
    if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
        raise RuntimeError("mf-da-models에 H2O 가중치(정한 판)가 없다. fetch_model을 먼저 돌린다")

    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH), "MAX_IMAGES": "8",
           "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=open("/tmp/serve.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 없음)", flush=True)

    req = json.load(open("/data/requests.json"))
    raw = open("/data/pack.tar", "rb").read()
    if hashlib.sha256(raw).hexdigest() != req["pack_sha256"]:
        server.terminate()
        raise RuntimeError("볼륨의 묶음이 요청 글의 해시와 다르다")
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    cands = req["candidates"][::max(1, len(req["candidates"]) // limit)][:limit] if limit else req["candidates"]
    url = lambda name: "data:image/jpeg;base64," + base64.b64encode(files[name]).decode()
    jobs = []   # (uid, arm, target, 본문)
    for c in cands:
        spec = req["specs"][c["spec"]]
        qs, n = questions_of(spec), round(c["length"])
        for arm, lead, names in (("8", LEAD8, c["frames"]), ("a", LEAD_A, c["frames"][:4]), ("b", LEAD_B, c["frames"][4:])):
            state = {"about": lead.format(n=n), "frames": [{"n": i, "frame": url(f)} for i, f in enumerate(names, 1)]}
            jobs.append((c["uid"], arm, spec["target"]["key"], json.dumps({"state": state, "questions": qs}).encode()))
    print(f"후보 {len(cands)}개, 요청 {len(jobs)}건, 본문 {sum(len(j[3]) for j in jobs) / 1e6:.0f}MB ({el():.0f}초)", flush=True)

    health, last = None, ""
    while health is None:
        if server.poll() is not None or el() > 15 * 60:
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

    def ask(job):
        uid, arm, target, body = job
        msg = ""
        for attempt in range(4):
            try:
                t = time.time()
                r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(r, timeout=300) as resp:
                    out = json.loads(resp.read())
                p, picks = {}, {}
                for q, a in out["answers"].items():
                    if "noul" in a:
                        p[q] = float(a["noul"])
                    else:
                        probs = {str(k): float(v) for k, v in a["probabilities"].items()}
                        if abs(sum(probs.values()) - 1) > 0.01 or target not in probs:
                            raise ValueError(f"{q}: 선택지 확률이 이상하다 {probs}")
                        p[q], picks[q] = probs[target], {k: round(v, 4) for k, v in probs.items()}
                if set(p) != {"noul_plain", "noul_def", "choice_fwd", "choice_rev"} or not all(0.0 <= v <= 1.0 for v in p.values()):
                    raise ValueError(f"답이 이상하다 {p}")
                u = out.get("usage") or {}
                return {"uid": uid, "arm": arm, "p": p, "picks": picks, "tokens": u.get("input_tokens"), "images": u.get("images"), "seconds": round(time.time() - t, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:300].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:300]
            time.sleep(2 ** attempt)
        return {"uid": uid, "arm": arm, "error": msg}

    # 형식 확인: 입력 꼴마다 처음 한 건. 8장이 안 받아지면 그 꼴은 건너뛴다
    skipped, recs = [], []
    for arm in ("8", "a", "b"):
        first = ask(next(j for j in jobs if j[1] == arm))
        print("형식 확인", arm, json.dumps(first, ensure_ascii=False)[:400], flush=True)
        if "error" in first:
            skipped.append({"arm": arm, "error": first["error"]})
            jobs = [j for j in jobs if j[1] != arm]
        else:
            recs.append(first)
            jobs = [j for j in jobs if not (j[0] == first["uid"] and j[1] == arm)]
    if skipped:
        print(open("/tmp/serve.log").read()[-1500:])
    t0, stopped = time.time(), None
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(ask, j) for j in jobs]
        for done, fut in enumerate(cf.as_completed(futs), 1):
            if fut.cancelled():
                continue
            recs.append(fut.result())
            if done == min(48, len(futs)) or done % 200 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(futs) - done) / rate
                print(f"  {done}/{len(futs)} 초당 {rate:.1f}건, 남은 어림 {left:.0f}초, 지금 {el():.0f}초", flush=True)
                if el() + left > BUDGET and stopped is None:
                    stopped = f"{done}건 뒤 멈춤: 잰 속도로 {el() + left:.0f}초에 끝남"
                    for f in futs:
                        f.cancel()
    server.terminate()
    good = [r for r in recs if "error" not in r]
    toks = {arm: sorted(r["tokens"] or 0 for r in good if r["arm"] == arm) for arm in ("8", "a", "b")}
    meta = {"model": "h2oai/h2o-lightning-4b", "h2o_revision": H2O_REVISION, "gpu": gpu, "requests": len(recs), "answered": len(good), "errors": len(recs) - len(good),
            "first_error": next((r["error"] for r in recs if "error" in r), None), "skipped_arms": skipped, "partial": stopped,
            "ready_seconds": round(t_ready), "ask_seconds": round(time.time() - t0, 1), "total_seconds": round(el()), "usd_estimate": round(el() * PRICE, 3),
            "tokens_median": {arm: (t[len(t) // 2] if t else None) for arm, t in toks.items()}, "workers": WORKERS,
            "settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "CUDA_GRAPH_MAX_TOKENS", "MAX_IMAGES", "MAX_PIXELS", "SHIM_MAX_PIXELS")},
            "leads": {"8": LEAD8, "a": LEAD_A, "b": LEAD_B}}
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return {"meta": meta, "records": recs}


@app.local_entrypoint()
def main(limit: int = 0):
    import hashlib
    import io
    import json
    import tarfile
    from pathlib import Path
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    specs = {}
    for p in sorted((here / "specs").glob("*.json")):
        if "_response_" not in p.name:
            s = json.loads(p.read_text())
            specs[s["request"]] = s
    buf, cands = io.BytesIO(), []
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for d in ("verify_runs", "verify_rr"):
            for ip in sorted((root / "pipeline" / d).glob("*/item.json")):
                item = json.loads(ip.read_text())
                for c in item["candidates"]:
                    names = []
                    for f in c["frames"]:
                        name = f"{d}__{item['name']}__{f}"
                        if name not in tar.getnames():
                            tar.add(ip.parent / "frames" / f, arcname=name)
                        names.append(name)
                    cands.append({"uid": f"{d}/{item['name']}#{c['id']}", "spec": specs[item["request"]]["name"], "length": c["length"], "frames": names})
    raw = buf.getvalue()
    req = {"pack_sha256": hashlib.sha256(raw).hexdigest(), "candidates": cands, "specs": {s["name"]: s for s in specs.values()}}
    print(f"후보 {len(cands)}개, 묶음 {len(raw) / 1e6:.0f}MB → 볼륨 mf-da-data")
    with data.batch_upload(force=True) as up:
        up.put_file(io.BytesIO(raw), "/pack.tar")
        up.put_file(io.BytesIO(json.dumps(req).encode()), "/requests.json")
    out = judge.remote(limit)
    path = here / "results" / (f"h2o_try{limit}.json" if limit else "h2o.json")
    path.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    m = out["meta"]
    print(f"끝: 답 {m['answered']}/{m['requests']}, 건너뛴 꼴 {m['skipped_arms']}, 준비 {m['ready_seconds']}초, 전체 {m['total_seconds']}초, 어림 ${m['usd_estimate']} → {path.name}")
