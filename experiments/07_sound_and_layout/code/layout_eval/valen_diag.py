"""Valen-Decision-1.5-4B가 GPU에서 한 건에 약 23초 걸린 원인 진단(pipeline/README.md 25-8절). <modal-profile> 워크스페이스, RTX PRO 6000 한 번.

공식 문서가 잰 판(transformers 5.20.2, torch 2.14.1 — cookbook/versions.json, 다만 공식은 CPU)으로 올리고, 마인크래프트 한 위치(A와 같은 요청)를 두고 잰다:
  1. 처음 한 번(커널 준비 포함), 이어서 두 번(한 건 시간)
  2. 층 종류별 시간: 그림 인코더, 선형 주의(Gated DeltaNet) 층, 전체 주의 층, 판정 머리 — 각 모듈 앞뒤에서 cuda 동기화로 잰다
  3. torch 프로파일러: GPU 시간이 큰 연산 20개, CPU 시간이 큰 연산 15개, GPU 합계와 벽시계 시간
  4. TF32 켜기(공식 설정은 끔)로 한 번
  5. SDPA를 memory-efficient 경로로만 묶어 한 번(안 되면 그 이유)
  6. 한 건이 빠르면 8건 묶음도 한 번
transformers가 fla·causal_conv1d 대신 느린 기본 구현을 썼는지(경고)도 잡아 둔다.
실행: MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/layout_eval/valen_diag.py
"""
import json

import modal

app = modal.App("mf-valen-diag")
vol = modal.Volume.from_name("mf-valen-eval")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")
         .uv_pip_install("torch==2.14.1", "torchvision", "transformers==5.20.2", "accelerate", "pillow", "av", "huggingface_hub", "numpy",
                         "flash-linear-attention==0.6.0")
         # torch 2.14.1은 CUDA 13 휠이다. causal-conv1d의 미리 빌드된 판이 없으면 건너뛴다(그때는 transformers 경고로 드러난다)
         .run_commands("pip install 'causal-conv1d==1.7.0+cu.13.0.torch.2.14' --extra-index-url https://wheels.astral.sh/simple/cu130/ || echo NO_CAUSAL_CONV1D"))


@app.function(image=image, gpu="RTX-PRO-6000", volumes={"/v": vol}, timeout=1800, cpu=8, memory=32768)
def diag(size: str = "4B"):
    import base64, importlib, io, logging, os, tarfile, time
    import torch
    import transformers
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel

    warned = []

    class Catch(logging.Handler):
        def emit(self, record):
            m = record.getMessage()
            if "falling back" in m or "reference PyTorch" in m:
                warned.append(m[:160])
    logging.getLogger("transformers").addHandler(Catch())

    def ver(name):
        try:
            return importlib.import_module(name).__version__
        except Exception as e:  # noqa: BLE001
            return f"없음({type(e).__name__})"
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "triton": ver("triton"),
           "fla": ver("fla"), "causal_conv1d": ver("causal_conv1d"), "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}
    print("환경", json.dumps(env, ensure_ascii=False), flush=True)

    sp = json.load(open("/v/mc_specs.json"))[0]
    p = sp["positions"][100]
    with tarfile.open("/v/mc_grid.tar") as tar:
        want = {f"{sp['name']}_{f}" for f in p["frames"]}
        grid = {os.path.basename(m.name): tar.extractfile(m).read() for m in tar.getmembers() if os.path.basename(m.name) in want}

    repo = {"4B": "Valen-Team/Valen-Decision-1.5-4B", "9B": "Valen-Team/Valen-Decision-1.5-9B"}[size]
    rev = HfApi().model_info(repo).sha
    t0 = time.time()
    model = AutoModel.from_pretrained(snapshot_download(repo, revision=rev), trust_remote_code=True, dtype="auto", attn_implementation="sdpa").to("cuda").eval()
    torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False      # 공식 설정
    print(f"{repo}@{rev[:10]} 올림 {time.time() - t0:.0f}초, GPU {torch.cuda.memory_allocated() / 1e9:.1f}GB, "
          f"본체 dtype {next(model.backbone.parameters()).dtype}", flush=True)

    mod = importlib.import_module(type(model).__module__)
    compiler = model.make_compiler()
    state = {"frames": [{"seconds": s, "frame": f"[image {i + 1}]"} for i, s in enumerate(p["seconds"])], **sp["state"]}
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(grid[f"{sp['name']}_{f}"]).decode()}} for f in p["frames"]]
    content.append({"type": "text", "text": json.dumps(state, ensure_ascii=False, separators=(",", ":"))})
    req = {"state": {"messages": [{"role": "user", "content": content}]}, "questions": sp["questions"]}
    with mod.local_images(req) as prepared:
        st = compiler.compile({"request": prepared})
    print(f"토큰 {st.logical_tokens}, 계산 토큰 {st.compute_tokens}, 입력 {[(k, tuple(v.shape)) for k, v in st.inputs.items() if hasattr(v, 'shape')]}", flush=True)

    def fwd(states):
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            model.forward_state_batch(states)
        torch.cuda.synchronize()
        return round(time.time() - t, 3)

    out = {"env": env}
    out["first"] = fwd([st]); print("처음 한 번(준비 포함)", out["first"], flush=True)
    out["steady"] = [fwd([st]) for _ in range(2)]; print("이어서 두 번", out["steady"], flush=True)

    # 모듈별 시간
    bb = model.backbone
    cats, hooks, stamp = {}, [], {}

    def add(module, cat):
        def pre(m, a, kw=None):
            torch.cuda.synchronize(); stamp[id(m)] = time.time()
        def post(m, a, o, kw=None):
            torch.cuda.synchronize(); cats[cat] = cats.get(cat, 0.0) + time.time() - stamp.pop(id(m))
        hooks.append(module.register_forward_pre_hook(pre)); hooks.append(module.register_forward_hook(post))
    add(bb.visual, "그림 인코더")
    for layer in bb.language_model.layers:
        if hasattr(layer, "linear_attn"):
            add(layer.linear_attn, "선형 주의 층(GDN)")
        if hasattr(layer, "self_attn"):
            add(layer.self_attn, "전체 주의 층(SDPA)")
        if hasattr(layer, "mlp"):
            add(layer.mlp, "MLP")
    if getattr(model, "head", None) is not None:
        add(model.head, "판정 머리")
    out["hooked_total"] = fwd([st])
    for h in hooks:
        h.remove()
    out["by_module"] = {k: round(v, 3) for k, v in sorted(cats.items(), key=lambda x: -x[1])}
    print("모듈별(동기화 포함 합계", out["hooked_total"], "초)", json.dumps(out["by_module"], ensure_ascii=False), flush=True)

    # 프로파일러
    from torch.profiler import ProfilerActivity, profile
    t = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fwd([st])
    wall = time.time() - t
    ka = prof.key_averages()
    dev = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
    gpu_total = sum(getattr(e, dev) for e in ka) / 1e6
    out["profile"] = {"wall": round(wall, 3), "gpu_kernel_seconds": round(gpu_total, 3)}
    print(f"프로파일: 벽시계 {wall:.2f}초, GPU 커널 합계 {gpu_total:.2f}초", flush=True)
    for key, n in ((dev, 20), ("self_cpu_time_total", 15)):
        try:
            print(ka.table(sort_by=key, row_limit=n, max_name_column_width=70), flush=True)
        except Exception as e:  # noqa: BLE001
            print("표를 못 만듦", key, type(e).__name__, e, flush=True)

    # TF32
    torch.set_float32_matmul_precision("high"); torch.backends.cudnn.allow_tf32 = True
    out["tf32_on"] = fwd([st]); print("TF32 켬", out["tf32_on"], flush=True)
    torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False

    # SDPA 경로
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for name, be in (("efficient만", [SDPBackend.EFFICIENT_ATTENTION]), ("math만", [SDPBackend.MATH])):
        try:
            with sdpa_kernel(be):
                out[f"sdpa_{name}"] = fwd([st])
        except Exception as e:  # noqa: BLE001
            out[f"sdpa_{name}"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"SDPA {name}", out[f"sdpa_{name}"], flush=True)

    if min(out["steady"]) < 3:
        out["batch8"] = fwd([st] * 8); print("8건 묶음", out["batch8"], flush=True)
    out["fallback_warnings"] = warned
    print("느린 기본 구현 경고", warned, flush=True)
    return out


@app.local_entrypoint()
def main(size: str = "4B"):
    print(json.dumps(diag.remote(size), ensure_ascii=False, indent=1))
