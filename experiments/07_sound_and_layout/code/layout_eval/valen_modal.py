"""판정 모델 비교: Valen-Decision-1.5-4B·9B를 H2O-Lightning-4B(A)와 같은 요청으로 마인크래프트 12편에 묻는다(pipeline/README.md 25-8절).

요청은 A의 것(pipeline/runs/gem_mc_*/requests.json)을 그대로 쓴다: 같은 프레임 4장(0.5초 간격, 640×360), 같은 state 글, 같은 질문 9개.
Valen은 그림을 메시지의 image_url로 받으므로(저장소 decisions.py) state를 [그림 4장, 글 하나]의 메시지로 옮긴다. 글은 H2O shim이 만드는 것과
같은 꼴(프레임 자리에 "[image N]", 압축 JSON)이다. 답은 Valen의 model.predict(SystemOne 꼴, 온도 1.0, 질문 여럿을 한 번에)로 받는다.

볼륨(<modal-profile> 워크스페이스) mf-valen-eval: /mc_grid.tar(영상 이름_격자 번호.jpg), /mc_specs.json(영상마다 state·질문·위치)
결과: 같은 볼륨의 /out/lay_V4_mc.json, /out/lay_V9_mc.json ({"meta", "records": [{"k": 영상 번호 × 10000 + k, "answers"}]})
돌리는 법(소유자 승인 뒤): MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/layout_eval/valen_modal.py --models 4B,9B [--limit 9] [--batch 8]
  --limit n: 영상마다 앞 n개 위치만(속도 재기). 위치 여럿을 묶어(forward_state_batch) 묻고, 처음 4건은 한 건씩 낸 답과 견준다
"""
import json

import modal

app = modal.App("mf-valen-eval")
vol = modal.Volume.from_name("mf-valen-eval")
REPOS = {"4B": "Valen-Team/Valen-Decision-1.5-4B", "9B": "Valen-Team/Valen-Decision-1.5-9B"}
# Qwen3.5의 Gated DeltaNet 층은 fla·causal_conv1d가 없으면 느린 PyTorch 기본 구현으로 돈다(transformers 문서). 25-8절 첫 시도가 이 때문에 너무 느렸다
# 25-8절 진단: torch 2.9.1·Triton 3.5·CUDA 12.8 환경에서는 RTX PRO 6000에서도 한 건 약 23초였고(L40S도 비슷하게 느렸음),
# 공식이 잰 판(transformers 5.20.2 + torch 2.14.1, cookbook/versions.json)과 Triton 3.8·CUDA 13으로 바꾸자 한 건 0.125초가 됐다(valen_diag.py)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")      # fla의 Triton 커널은 처음 쓸 때 C 컴파일러가 필요하다
         .uv_pip_install("torch==2.14.1", "torchvision", "transformers==5.20.2", "accelerate", "pillow", "av", "huggingface_hub", "numpy",
                         "flash-linear-attention==0.6.0")
         .run_commands("pip install 'causal-conv1d==1.7.0+cu.13.0.torch.2.14' --extra-index-url https://wheels.astral.sh/simple/cu130/ || echo NO_CAUSAL_CONV1D"))


@app.function(image=image, gpu="RTX-PRO-6000", volumes={"/v": vol}, timeout=3600, cpu=8, memory=32768)
def judge(models: str, limit: int = 0, batch: int = 1, profile: int = 0):
    import base64, io, os, statistics, tarfile, time
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel

    t0 = time.time()
    specs = json.load(open("/v/mc_specs.json"))
    with tarfile.open("/v/mc_grid.tar") as tar:
        grid = {os.path.basename(m.name): tar.extractfile(m).read() for m in tar.getmembers() if m.isfile() and not os.path.basename(m.name).startswith("._")}
    print(f"프레임 {len(grid)}장, 영상 {len(specs)}편 ({time.time() - t0:.0f}초)", flush=True)
    summary = {}
    for size in [m for m in models.split(",") if m]:
        repo = REPOS[size]
        rev = HfApi().model_info(repo).sha
        t1 = time.time()
        path = snapshot_download(repo, revision=rev)
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype="auto", attn_implementation="sdpa").to("cuda").eval()
        torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False   # 모델 카드의 설정
        print(f"[{size}] {repo}@{rev[:10]} 올림 {time.time() - t1:.0f}초, GPU {torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)
        import importlib
        mod = importlib.import_module(type(model).__module__)
        answer, local_images = mod.answer, mod.local_images
        compiler = model.make_compiler()
        try:
            import causal_conv1d, fla  # noqa: F401
            kern = f"causal_conv1d {getattr(causal_conv1d, '__version__', '?')}, fla {getattr(fla, '__version__', '?')}"
        except Exception as e:  # noqa: BLE001
            kern = f"빠른 커널 없음: {type(e).__name__}: {e}"
        print(f"[{size}] {kern}", flush=True)

        def request_of(sp, p):
            state = {"frames": [{"seconds": s, "frame": f"[image {i + 1}]"} for i, s in enumerate(p["seconds"])], **sp["state"]}
            content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(grid[f"{sp['name']}_{f}"]).decode()}} for f in p["frames"]]
            content.append({"type": "text", "text": json.dumps(state, ensure_ascii=False, separators=(",", ":"))})
            return {"state": {"messages": [{"role": "user", "content": content}]}, "questions": sp["questions"]}

        work = [(vi, sp, p) for vi, sp in enumerate(specs) for p in (sp["positions"][:limit] if limit else sp["positions"])]
        # 묶어 낸 답이 한 건씩 낸 답(model.predict)과 같은지 처음 4건으로 본다(채움 토큰이 답을 바꾸지 않는지)
        with torch.inference_mode():
            t_s = time.time()
            single = [model.predict(request_of(sp, p), temperature=1.0)["answers"] for _, sp, p in work[:4]]
            print(f"[{size}] 한 건씩 4건(첫 실행 준비 포함) {time.time() - t_s:.1f}초", flush=True)
        recs, secs, toks, t2 = [], [], [], time.time()
        for b0 in range(0, len(work), batch):
            chunk = work[b0:b0 + batch]
            ts = time.time()
            try:
                states, tb = [], {}
                t_a = time.time()
                reqs = [request_of(sp, p) for _, sp, p in chunk]
                tb["build"] = time.time() - t_a; t_a = time.time()
                for rq in reqs:
                    with local_images(rq) as prepared:
                        states.append(compiler.compile({"request": prepared}))
                tb["compile"] = time.time() - t_a
                torch.cuda.synchronize(); t_a = time.time()
                with torch.inference_mode():
                    logits = model.forward_state_batch(states)
                torch.cuda.synchronize(); tb["gpu"] = time.time() - t_a; t_a = time.time()
                for (vi, sp, p), st, lg in zip(chunk, states, logits):
                    ans = {q.qid: answer(q, z, 1.0) for q, z in zip(st.questions, lg)}
                    recs.append({"k": vi * 10000 + p["k"], "answers": ans, "tokens": st.logical_tokens}); toks.append(st.logical_tokens)
                tb["answer"] = time.time() - t_a
                if profile or b0 // batch < 5:
                    print(f"  [{size}] 묶음 {b0 // batch}: " + ", ".join(f"{k} {v:.2f}초" for k, v in tb.items()) + f" (토큰 {[st.logical_tokens for st in states][:3]}…, 계산 {[st.compute_tokens for st in states][:3]}…)", flush=True)
            except Exception as e:  # noqa: BLE001
                for vi, sp, p in chunk:
                    recs.append({"k": vi * 10000 + p["k"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
            secs.append((time.time() - ts) / len(chunk))
            if b0 == 0:
                def val(a):
                    return a.get("noul", a.get("score"))
                try:
                    diff = f"{max(abs(val(single[i][q]) - val(recs[i]['answers'][q])) for i in range(min(4, len(chunk))) for q in single[i]):.4f}"
                except Exception as e:  # noqa: BLE001
                    diff = f"잴 수 없음({type(e).__name__})"
                print(f"[{size}] 형식 확인(묶음 {batch}) 한 건씩과 최대 차이 {diff}", json.dumps(recs[0], ensure_ascii=False)[:300], flush=True)
            if (b0 // batch) % max(1, 480 // batch) == 0 and b0:
                print(f"  [{size}] {len(recs)}건, 한 건 {statistics.median(secs):.3f}초, 지금 {time.time() - t0:.0f}초", flush=True)
            if len(recs) % 960 < batch:      # 중간 저장(멈춰도 받은 답은 남는다)
                os.makedirs("/v/out", exist_ok=True)
                json.dump({"partial": True, "records": recs}, open(f"/v/out/lay_V{size[:-1]}_mc_partial.json", "w"), ensure_ascii=False); vol.commit()
        good = [r for r in recs if "error" not in r]
        meta = {"name": f"lay_V{size[:-1]}_mc", "model": repo, "revision": rev, "requests": len(recs), "answered": len(good), "errors": len(recs) - len(good),
                "wall_seconds": round(time.time() - t2, 1), "seconds_median": round(statistics.median(secs), 3) if secs else None, "batch": batch, "kernels": kern,
                "tokens_median": int(statistics.median(toks)) if toks else None, "gpu": torch.cuda.get_device_name(0), "temperature": 1.0,
                "first_error": next((r["error"] for r in recs if "error" in r), None)}
        os.makedirs("/v/out", exist_ok=True)
        json.dump({"meta": meta, "records": recs}, open(f"/v/out/{meta['name']}{'_limit' if limit else ''}.json", "w"), ensure_ascii=False)
        vol.commit(); summary[size] = meta; print("끝", json.dumps(meta, ensure_ascii=False), flush=True)
        del model; torch.cuda.empty_cache()
    summary["_total_seconds"] = round(time.time() - t0)
    return summary


@app.local_entrypoint()
def main(models: str = "4B,9B", limit: int = 0, batch: int = 1, profile: int = 0):
    print(json.dumps(judge.remote(models, limit, batch, profile), ensure_ascii=False, indent=1))
