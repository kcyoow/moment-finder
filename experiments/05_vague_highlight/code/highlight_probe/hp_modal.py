"""highlight_probe A 단계(훑어보기): 판정 모델(H2O)에 고정 질문 9개를 묻는다. Modal에서 돈다. 2026-11-15 무료 준비에서는 돌리지 않았다.

GPU 작업 하나가 H2O-Lightning-4B 서버(serve.sh: vLLM 0.30.0 + shim)를 같은 컨테이너 안에 띄우고, 정한 묶음을 묻고, 끝나면 스스로 꺼진다.
틀은 ../highlight_auto/ha_modal.py와 같다(서버 설정, 컴파일 캐시, 묶음을 메모리로, 동시 16, 묶음마다 처음 48건으로 속도 재기). 다른 곳은 README 13절.

본문: 상태에는 about과 frames(그림 4장)뿐이다. 제목, 범주, 게임 이름, viewer_request는 없다.
hp_prepare.probe_body(about, sent_questions(spec), seconds, jpegs)와 JSON 글자까지 같다(hp_selftest.py가 견준다).

볼륨 (모두 <modal-profile> 계정)
  mf-highlight-probe (/data)       : packs/<TVSum 영상>.tar (data/probe_packs 그대로), probe_requests.json (data/probe_requests.json). 답은 results/<묶음>.json
                                     ← 올리기는 소유자의 승인 뒤에만(README 13절). 실행 뒤 packs/를 바로 지우고 목록으로 확인한다
  mf-highlight-h2o (/hh, 읽기만)   : 마인크래프트 묶음 packs/<영상>.tar, frames_summary.json (이미 올라가 있음. 다시 올리지 않음)
  mf-mc-dense (/mcd, 읽기만)       : 정상 확인 200곳의 프레임(packs/mc_08.tar, packs/ch_04.tar)
  mf-models (/models)              : H2O 가중치(revision 659fc412…). 이 작업은 받지 않는다. 없으면 멈춘다
  mf-vllm-cache (/cache, 읽기만)   : 컴파일 캐시 tar(그래프 512 + 같은 VLLM_ARGS)

실행 (소유자 승인 뒤, 조정자가 돌림): README 13절
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/highlight_probe/hp_modal.py::probe_all   # 상한 900초 = $0.92

시간 지키기: 묶음을 시작하기 전에 (probe_requests.json의 plan × 앞 묶음에서 잰 배율)로 끝날 시각을 어림해 budget을 넘으면 건너뛴다.
묶음마다 처음 48건을 재고, 잰 속도로 어림한 끝이 budget을 넘거나 속도가 예상의 절반에 못 미치면 그 묶음을 멈추고(답은 partial로 저장) 다음으로 간다.
묶음 순서: sanity(정상 확인 200건) → tvsum(훑어보기 742건) → minecraft(훑어보기 360건). 함수의 timeout이 마지막 안전장치다.
"""
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True          # 다른 실험 폴더에 __pycache__를 만들지 않는다
sys.path.insert(0, str(HERE.parent / "highlight_auto"))
sys.path.insert(0, str(HERE.parent / "highlight_h2o"))
import ha_prepare as P  # noqa: E402  (읽기만: 그림 data URI, 초당 값)
import hh_prepare as HP  # noqa: E402  (읽기만: 정상 확인 본문과 답 읽기)

app = modal.App("moment-finder-highlight-probe")
data = modal.Volume.from_name("mf-highlight-probe")
hh = modal.Volume.from_name("mf-highlight-h2o").read_only()
mcd = modal.Volume.from_name("mf-mc-dense").read_only()
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # highlight_h2o, highlight_auto와 같은 판
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH = 512
MAX_PIXELS = 2_000_000                                      # highlight_auto와 같다. TVSum 그림(가장 큰 640×360)과 마인크래프트 그림은 줄지 않는다
PORT = 8741
WORKERS = 16
FIRST = 48                                                  # 묶음마다 처음 48건으로 속도를 잰다(tasks/lessons.md 2026-11-12)
RETRY_SLEEP = 1.0                                           # 다시 묻기 전 1, 2, 4, 8초 (ha_modal과 같음. 자체 시험만 줄인다)
PRICE = P.PRICE
ALL_TIMEOUT, ALL_BUDGET = 900, 840                          # 상한 900 × $0.00101816 = $0.92 (README 9절). 예상 362~531초
ARMS = "sanity,tvsum,minecraft"

gpu_image = (   # highlight_auto와 같은 이미지 + highlight_auto의 ha_prepare.py, highlight_h2o의 hh_prepare.py
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
    .add_local_python_source("ha_prepare", "hh_prepare")
)
VOLS = {"/data": data, "/hh": hh, "/mcd": mcd, "/models": models, "/cache": cache_vol}
ROOTS = {"data": "/data", "hh": "/hh", "mcd": "/mcd"}       # 자체 시험은 임시 폴더로 바꿔 끼운다

FRAMES = {}   # 묶음 이름 → {파일 이름: 바이트}
LAST = {}
REQ = {}


def parse(out, questions):
    """서버 답 → {질문: 답}. 모양은 hp_prepare.profile_text가 읽는 그대로다:
    noul {"noul": p}, choice {"probabilities": {선택지: p}}, score {"score": s, "probabilities": {"0": p, …}}.
    어긋나면 ValueError(그 위치를 오류로 센다): noul은 0~1, choice는 선택지 이름이 질문과 같고 합이 1에서 0.01 안,
    score는 단계 수가 맞고 합이 1에서 0.01 안이며 확률의 기댓값이 score와 0.01 안. `not (… <= 0.01)` 꼴이라 NaN도 오류다."""
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
            if sorted(probs) != sorted(levels) or not abs(sum(probs.values()) - 1) <= 0.01:
                raise ValueError(f"{key}: 단계 확률이 이상하다 {probs}")
            expect = sum(int(k) * v for k, v in probs.items())
            if not abs(expect - float(a["score"])) <= 0.01:
                raise ValueError(f"{key}: score {a['score']}가 확률의 기댓값 {expect:.3f}과 다르다")
            rec[key] = {"score": float(a["score"]), "probabilities": {lv: probs[lv] for lv in levels}}
    return rec


def probe_body(about, questions, seconds, jpegs):
    """훑어보기 본문. hp_prepare.probe_body와 칸 이름·순서·값이 같다(hp_selftest.py가 JSON 글자까지 견준다)."""
    return {"state": {"about": about, "frames": [{"seconds": s, "frame": P.uri(b)} for s, b in zip(seconds, jpegs)]}, "questions": questions}


def _units(name):
    if name == "sanity":
        return [{"id": f"{v}@{k}", "video": v, "k": k} for v in HP.SANITY_VIDEOS for k in range(HP.SANITY_K)]
    if name == "tvsum":
        return [{"id": f"{v}@{p['k']}", "video": v, "k": p["k"], "pos": p} for v, x in REQ["tvsum"]["videos"].items() for p in x["positions"]]
    return [{"id": p["id"], "video": v, "t": p["t"], "pos": p} for v, x in REQ["minecraft"]["videos"].items() for p in x["positions"]]


def _build(job):
    """(묶음, 단위) → (본문 JSON 바이트, 그림 수). 프로세스 풀에서 돈다(fork라 FRAMES를 물려받는다)."""
    import json
    arm, it = job
    if arm == "sanity":
        payload = HP.sanity_body(lambda v, n: FRAMES[v][n], it["video"], it["k"], LAST[it["video"]])   # highlight_auto 10절과 같은 본문
    else:
        pos = it["pos"]
        payload = probe_body(REQ["about"], REQ["questions"], pos["seconds"], [FRAMES[it["video"]][n] for n in pos["frames"]])
    return json.dumps(payload).encode(), len(payload["state"]["frames"])


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes=VOLS, timeout=ALL_TIMEOUT, cpu=8, memory=32768)
def probe_all(budget: int = ALL_BUDGET):
    return _probe(ARMS, budget, "probe_all")


def _start_h2o(port):
    """GPU 컨테이너에서 H2O 서버를 띄운다(ha_modal과 같은 설정). → (서버 프로세스, 로그 경로, 결과에 적을 정보)."""
    import hashlib
    import os
    import subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(gpu, flush=True)
    if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
        raise RuntimeError("mf-models에 H2O 가중치(정한 판)가 없다. 이 작업은 받지 않는다")
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]   # = c32bae6555f8
    cache_tar = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(cache_tar)
    if restored:
        subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(port)}
    log = open("/tmp/serve.log", "w")
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=log, stderr=subprocess.STDOUT, env=env)
    info = {"settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "CUDA_GRAPH_MAX_TOKENS", "MAX_PIXELS", "SHIM_MAX_PIXELS", "SHIM_TIMEOUT")},
            "compile_cache": cache_tar, "compile_cache_restored": restored, "gpu": gpu, "h2o_revision": H2O_REVISION}
    return server, "/tmp/serve.log", info


def _probe(arms, budget, fn_name, roots=None, start=None, commit=None, port=PORT):
    """ha_modal._judge와 같은 흐름. roots·start·commit·port는 이 Mac의 자체 시험(hp_selftest.py)이 임시 폴더와 가짜 서버로 바꿔 끼우는 자리다.
    Modal에서는 기본값(볼륨 경로, H2O 서버, 볼륨 commit)을 쓴다."""
    import concurrent.futures as cf
    import hashlib
    import io
    import json
    import multiprocessing as mp
    import os
    import tarfile
    import time
    import urllib.error
    import urllib.request
    import PIL

    roots = roots or ROOTS
    start = start or _start_h2o
    commit = commit or data.commit
    t_start = time.time()

    def el():
        return time.time() - t_start

    # 1) 서버 띄우기 (highlight_auto와 같은 설정)
    server, log_path, info = start(port)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 {'되살림' if info.get('compile_cache_restored') else '없음'}: {info.get('compile_cache')})", flush=True)

    # 2) 서버가 뜨는 동안: 요청 글과 묶음을 메모리에 올리고(해시 확인) 모든 본문을 미리 만든다
    REQ.update(json.load(open(f"{roots['data']}/probe_requests.json")))
    mc_expect = json.load(open(f"{roots['hh']}/frames_summary.json"))["packs"]
    names = [x for x in arms.split(",") if x]
    t0 = time.time()

    def load(path):
        raw = open(path, "rb").read()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            return hashlib.sha256(raw).hexdigest(), {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    jobs = []   # (어느 볼륨, 경로)
    if "tvsum" in names:
        jobs += [("data", f"{roots['data']}/packs/{v}.tar") for v in sorted(REQ["packs"])]
    if "minecraft" in names:
        jobs += [("hh", f"{roots['hh']}/packs/{v}.tar") for v in sorted(REQ["minecraft"]["videos"])]   # 훑어볼 영상의 묶음만
    if "sanity" in names:
        jobs += [("mcd", f"{roots['mcd']}/packs/{v}.tar") for v in HP.SANITY_VIDEOS]
    bad_hash = []
    with cf.ThreadPoolExecutor(8) as ex:
        for (kind, path), (digest, files) in zip(jobs, ex.map(load, [p for _, p in jobs])):
            video = os.path.basename(path)[:-4]
            FRAMES[video] = files
            if kind == "data" and digest != REQ["packs"][video]:
                bad_hash.append(video)
            if kind == "hh" and digest[:16] != mc_expect[video]["sha256"]:
                bad_hash.append(video)
    for v in HP.SANITY_VIDEOS:
        if v in FRAMES:
            LAST[v] = json.loads(FRAMES[v]["_done.json"])["last"]
    print(f"묶음 {len(jobs)}개 올림 {time.time() - t0:.0f}초, {sum(len(b) for f in FRAMES.values() for b in f.values()) / 1e9:.2f}GB", flush=True)
    if bad_hash:
        server.terminate()
        raise RuntimeError(f"볼륨의 묶음이 이 Mac의 것과 다르다: {bad_hash}")

    units = {name: _units(name) for name in names}
    t0 = time.time()
    built = {}
    with cf.ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as ex:
        for name in names:
            built[name] = list(ex.map(_build, [(name, it) for it in units[name]], chunksize=8))
    print(f"본문 {sum(len(v) for v in built.values())}개 미리 만듦 {time.time() - t0:.0f}초, "
          f"{sum(len(b[0]) for v in built.values() for b in v) / 1e9:.2f}GB", flush=True)

    # 3) 준비 확인
    health, last = None, ""
    while health is None:
        if server.poll() is not None or el() > min(budget - 60, 15 * 60):
            print(open(log_path).read()[-5000:])
            server.terminate()
            raise RuntimeError(f"서버가 뜨지 못함: {last}")
        try:
            health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30).read().decode()
        except urllib.error.HTTPError as err:
            last = f"HTTP {err.code}"
            time.sleep(3)
        except Exception as err:
            last = f"{type(err).__name__}: {err}"[:200]
            time.sleep(3)
    t_ready = el()
    print(f"서버 준비 {t_ready:.0f}초 | {health[:500]}", flush=True)

    def post(body, timeout):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def ask(name, i, timeout=300):
        it, (body, n_img) = units[name][i], built[name][i]
        head = {"id": it["id"], "video": it["video"]}
        head.update({"t": it["t"]} if name == "minecraft" else {"k": it["k"]})
        msg = ""
        for attempt in range(4):
            try:
                t = time.time()
                out = post(body, timeout)
                ans = {"probs": HP.parse_sanity(out)} if name == "sanity" else {"answers": parse(out, REQ["questions"])}
                u = out.get("usage") or {}
                return {**head, **ans, "tokens": u.get("input_tokens"), "processed": u.get("processed_input_tokens"), "images": u.get("images"),
                        "ms": u.get("latency_ms"), "seconds": round(time.time() - t, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:200]
            time.sleep(RETRY_SLEEP * 2 ** attempt)
        return {**head, "error": msg}

    # 4) 형식 확인: 묶음마다 2건. 실패한 묶음은 건너뛴다(tvsum이 실패하면 멈춘다)
    ok, checks = [], {}
    for name in names:
        got = [ask(name, i) for i in range(2)]
        bad = [g.get("error") for g in got if "error" in g]
        checks[name] = {"errors": bad, "tokens": [g.get("tokens") for g in got], "images": [g.get("images") for g in got]}
        print("형식 확인", name, json.dumps(checks[name], ensure_ascii=False), json.dumps(got[0], ensure_ascii=False)[:400], flush=True)
        if bad:
            if name == "tvsum":
                print(open(log_path).read()[-3000:])
                server.terminate()
                raise RuntimeError(f"형식 확인 실패: tvsum: {bad}")
            continue
        ok.append(name)

    # 5) 묻기
    os.makedirs(f"{roots['data']}/results", exist_ok=True)
    summary, ratio = {"_checks": checks}, 1.0
    for name in ok:
        est = REQ["plan"][name]["est_seconds"]
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
                if done == min(FIRST, len(futs)) or done % 500 == 0:
                    rate = done / (time.time() - t0)
                    left = (len(futs) - done) / rate
                    print(f"  {name} {done}/{len(futs)} 초당 {rate:.1f}건, 남은 어림 {left:.0f}초, 지금 {el():.0f}초", flush=True)
                    slow = rate < 0.5 * len(futs) / est   # tasks/lessons.md 2026-11-12
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
                "questions_file_sha256": REQ.get("questions_file_sha256"), **info, "pillow": PIL.__version__}
        with open(f"{roots['data']}/results/{name}.json", "w") as f:
            json.dump({"meta": meta, "records": recs}, f)
        commit()
        summary[name] = {k: meta[k] for k in ("requests", "answered", "errors", "partial", "wall_seconds", "rate_per_second", "planned_seconds", "tokens_median", "first_error")}
        print("끝", name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    server.terminate()
    summary["_times"] = {"ready_seconds": round(t_ready), "total_seconds": round(el()), "compile_cache_restored": info.get("compile_cache_restored"),
                         "usd_estimate": round(el() * PRICE, 3)}
    with open(f"{roots['data']}/results/_summary_{fn_name}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    commit()
    print(json.dumps(summary, ensure_ascii=False))
    return summary
