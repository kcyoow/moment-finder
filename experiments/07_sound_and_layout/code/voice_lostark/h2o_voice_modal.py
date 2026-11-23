"""판정 모델(H2O-Lightning-4B)에 위치마다 다른 기록(state)을 묻는다. 목소리 실험(pipeline/README.md 25절)의 2·3단계에 쓴다.

사이트의 pipeline/mf_modal.py::judge는 모든 위치가 같은 state(+그 위치의 프레임)를 쓴다. 여기서는 위치마다 목소리 기록이 달라서
요청 묶음(spec)을 미리 만들어 둔다. 서버 띄우는 법과 묻는 고리는 mf_modal.judge와 같고, 답 검사는 choice도 읽는
experiments/highlight_probe/hp_modal.parse와 같다.

spec(볼륨 mf-voice의 /h2o/<이름>.json, voice_prep.py가 만든다):
  {"name": ..., "questions": {...}, "base_state": {...},
   "items": [{"k": ..., "state": {...}, "frames": [격자 번호, ...], "frames_first": true|false}], "est_seconds": ...}
  frames가 있으면 /v/frames_grid.tar(격자 번호 = 파일 이름 %06d.jpg - 1)에서 읽어 state에 "frames"로 넣는다.
  frames_first가 true면 state의 맨 앞에, false면 맨 뒤에 둔다(받아쓰기를 넣는 자리 시험).
  spec에 "chain": {"chunks", "key", "note"}가 있으면 사슬로 묻는다: 항목을 chunks토막으로 나눠 토막 안은 차례로, 토막끼리는 동시에.
  각 항목의 state[key]에 같은 실행이 바로 앞 항목에 낸 답(noul은 확률, score는 기댓값, choice는 선택지별 확률. note와 함께)을 넣는다.
  토막 첫 항목과 앞이 실패한 항목은 앞 답 없이. "bounds"가 있으면 그 경계로 토막을 나눈다(첫 토막은 3개 이상).

돌리는 법(소유자 승인 뒤): MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/voice_lostark/h2o_voice_modal.py --specs a,b,c
값: RTX PRO 6000 + 8코어 + 32GiB = 초당 약 $0.001018. 서버 준비 약 150초.
"""
import json

import modal

app = modal.App("mf-h2o-voice")
vol = modal.Volume.from_name("mf-voice")
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()

H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # 사이트·실험과 같은 판
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8"
GRAPH, MAX_PIXELS, PORT, WORKERS = 512, 2_000_000, 8741, 16
PRICE = 0.000842 + 8 * 0.0000131 + 32 * 0.00000222
TIMEOUT, BUDGET = 3000, 2900   # 25-7절: 한 컨테이너가 묶음 넷(약 30분)을 돌린다
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12").entrypoint([])
             .apt_install("curl", "patch").uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow"))


def parse(out, questions):
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


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/v": vol, "/models": models, "/cache": cache_vol}, timeout=TIMEOUT, cpu=8, memory=32768)
def judge(specs: str, limit: int = 0):
    import base64, concurrent.futures as cf, hashlib, os, subprocess, tarfile, time, urllib.error, urllib.request

    t0 = time.time()
    el = lambda: time.time() - t0
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(gpu, flush=True)
    if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
        raise RuntimeError("mf-models에 H2O 가중치(정한 판)가 없다")
    home = "/tmp/h2ohome"; os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]
    cache_tar = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(cache_tar)
    if restored:
        subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=open("/tmp/serve.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(f"서버 시작 {el():.0f}초 (컴파일 캐시 {'되살림' if restored else '없음'})", flush=True)

    # 서버가 뜨는 동안 본문을 만든다
    jpgs, named = {}, {}
    import glob
    for tp in sorted(glob.glob("/v/*.tar")):        # frames_grid.tar(격자 번호로) + 타일 묶음들(파일 이름으로, frame_names)
        with tarfile.open(tp) as tar:
            for m in tar.getmembers():
                b = os.path.basename(m.name)
                if b.endswith(".jpg") and not b.startswith("._"):   # macOS tar가 넣는 ._ 파일은 건너뛴다
                    data_ = tar.extractfile(m).read()
                    named[b] = data_
                    if tp.endswith("frames_grid.tar"):
                        jpgs[int(b[:-4]) - 1] = data_
    runs = []
    for name in [x for x in specs.split(",") if x]:
        sp = json.load(open(f"/v/h2o/{name}.json"))
        items = sp["items"][:limit] if limit else sp["items"]
        bodies = []
        for it in items:
            st = {**sp.get("base_state", {}), **it["state"]}
            if it.get("frames") or it.get("frame_names"):
                raws = [named[x] for x in it["frame_names"]] if it.get("frame_names") else [jpgs[g] for g in it["frames"]]
                fr = [{"seconds": s, "frame": "data:image/jpeg;base64," + base64.b64encode(r).decode()} for r, s in zip(raws, it["seconds"])]
                st = {"frames": fr, **st} if it.get("frames_first", True) else {**st, "frames": fr}
            bodies.append(json.dumps({"state": st, "questions": sp["questions"]}, ensure_ascii=False).encode())
        runs.append((name, sp, items, bodies))
        print(f"{name}: {len(items)}건, 본문 {sum(len(b) for b in bodies) / 1e6:.1f}MB ({el():.0f}초)", flush=True)

    health, last = None, ""
    while health is None:
        if server.poll() is not None or el() > 15 * 60:
            print(open("/tmp/serve.log").read()[-5000:]); server.terminate()
            raise RuntimeError(f"서버가 뜨지 못함: {last}")
        try:
            health = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=30).read().decode()
        except Exception as err:
            last = f"{type(err).__name__}: {err}"[:200]; time.sleep(3)
    t_ready = el(); print(f"서버 준비 {t_ready:.0f}초", flush=True)

    def post(body, timeout=300):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    summary = {}
    for name, sp, items, bodies in runs:
        def ask(i, body=None):
            msg = ""
            for attempt in range(4):
                try:
                    t = time.time(); out = post(body or bodies[i]); u = out.get("usage") or {}
                    return {"k": items[i]["k"], "answers": parse(out, sp["questions"]), "tokens": u.get("input_tokens"), "seconds": round(time.time() - t, 3)}
                except urllib.error.HTTPError as err:
                    msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
                except Exception as err:
                    msg = f"{type(err).__name__}: {err}"[:200]
                time.sleep(2 ** attempt)
            return {"k": items[i]["k"], "error": msg}
        ch = sp.get("chain")
        if ch:   # 사슬: 바로 앞 초에 이 실행이 낸 답을 state에 넣고 묻는다. 토막 안은 차례로, 토막끼리는 동시에
            def with_prev(i, prev):
                if prev is None:
                    return bodies[i], None
                def val(a):    # 예/아니오는 "예"의 확률, 점수는 기댓값(0부터의 단계), 고르기는 선택지마다의 확률
                    return round(a["noul"], 2) if "noul" in a else round(a["score"], 2) if "score" in a else {o: round(x, 2) for o, x in a["probabilities"].items()}
                shown = {"note": ch["note"], **{q: val(prev[q]) for q in sp["questions"]}}
                b = json.loads(bodies[i]); b["state"][ch["key"]] = shown
                return json.dumps(b, ensure_ascii=False).encode(), shown

            def ask_chain(i, prev):
                b, shown = with_prev(i, prev)
                r = ask(i, b); r["prev_shown"] = shown; return r
            recs = [None] * len(items)
            recs[0] = ask_chain(0, None)
            if len(items) > 1:
                recs[1] = ask_chain(1, recs[0].get("answers"))
            print("형식 확인", name, json.dumps(recs[min(1, len(items) - 1)], ensure_ascii=False)[:400], flush=True)
            if any("error" in g for g in recs[:2] if g):
                summary[name] = {"skipped": recs[0].get("error") or recs[1].get("error")}
                print(open("/tmp/serve.log").read()[-2000:]); continue
            if ch.get("bounds"):     # 토막 경계를 spec이 정한다(영상 여럿을 한 묶음에 넣을 때 영상 경계를 넘지 않게)
                bounds = ch["bounds"]; nch = len(bounds) - 1
            else:
                nch = ch.get("chunks", WORKERS); bounds = [round(c * len(items) / nch) for c in range(nch + 1)]
            done, t1 = [0], time.time()

            def run_chunk(c):
                lo, hi = bounds[c], bounds[c + 1]
                prev = recs[1].get("answers") if c == 0 else None
                for i in range(max(lo, 2) if c == 0 else lo, hi):
                    recs[i] = ask_chain(i, prev); prev = recs[i].get("answers")   # 실패하면 다음 초는 앞 답 없이
                    done[0] += 1
                    if done[0] % 500 == 0:
                        print(f"  {name} {done[0]}/{len(items) - 2} 초당 {done[0] / (time.time() - t1):.1f}건, 지금 {el():.0f}초", flush=True)
            with cf.ThreadPoolExecutor(nch) as ex:
                list(ex.map(run_chunk, range(nch)))
        else:
            first = [ask(i) for i in range(min(2, len(items)))]
            print("형식 확인", name, json.dumps(first[0], ensure_ascii=False)[:300], flush=True)
            if any("error" in g for g in first):
                summary[name] = {"skipped": first[0].get("error") or first[-1].get("error")}
                print(open("/tmp/serve.log").read()[-2000:]); continue
            recs = first + [None] * (len(items) - len(first)); t1 = time.time()
            with cf.ThreadPoolExecutor(WORKERS) as ex:
                futs = {ex.submit(ask, i): i for i in range(len(first), len(items))}
                for n_done, fut in enumerate(cf.as_completed(futs), 1):
                    recs[futs[fut]] = fut.result()
                    if n_done in (48,) or n_done % 500 == 0:
                        print(f"  {name} {n_done}/{len(futs)} 초당 {n_done / (time.time() - t1):.1f}건, 지금 {el():.0f}초", flush=True)
        good = [r for r in recs if r and "error" not in r]
        toks = sorted(r["tokens"] or 0 for r in good)
        meta = {"name": name, "requests": len(items), "answered": len(good), "errors": len(items) - len(good), "wall_seconds": round(time.time() - t1, 1),
                "tokens_median": toks[len(toks) // 2] if toks else None, "gpu": gpu, "ready_seconds": round(t_ready), "h2o_revision": H2O_REVISION}
        os.makedirs("/v/h2o_out", exist_ok=True)
        json.dump({"meta": meta, "records": recs}, open(f"/v/h2o_out/{name}{'_limit' if limit else ''}.json", "w"), ensure_ascii=False)
        vol.commit(); summary[name] = meta; print("끝", name, json.dumps(meta, ensure_ascii=False), flush=True)
    server.terminate()
    summary["_times"] = {"ready_seconds": round(t_ready), "total_seconds": round(el()), "usd_estimate": round(el() * PRICE, 3)}
    print(json.dumps(summary, ensure_ascii=False)); return summary


@app.local_entrypoint()
def main(specs: str, limit: int = 0):
    print(json.dumps(judge.remote(specs, limit), ensure_ascii=False, indent=1))
