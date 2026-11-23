# @title EmbeddingGemma 2 speed on the GPU the H2O-Lightning-4B run used (L4, then A100)
"""Run on the same session after h2o_run.py. Stops the vLLM server first to free the GPU.
Writes /content/eg/results/speed_l4.json on an L4, speed_<gpu>.json otherwise (speed_a100.json on an A100).
Timing covers preprocessing + forward on in-memory 1024x576 frames.
"""
import signal

if globals().get("H2O_PROC") is not None and H2O_PROC.poll() is None:
    os.killpg(os.getpgid(H2O_PROC.pid), signal.SIGTERM)
    try:
        H2O_PROC.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(H2O_PROC.pid), signal.SIGKILL)
    time.sleep(5)
print("GPU after stopping vLLM:", subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
                                                 shell=True, capture_output=True, text=True).stdout.strip())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers[image]>=6.1.0", "transformers"],
               check=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

eg = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"audio_config": None},
                         model_kwargs={"torch_dtype": torch.float32}, device="cuda")
imgs = [PILImage.open(p).convert("RGB") for _, _, _, p in frames[:64]]


def enc(images, budget, batch_size):
    return eg.encode([{"image": im} for im in images], batch_size=batch_size, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False,
                     processing_kwargs={"image": {"max_soft_tokens": budget}})


l4 = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "runs": []}
ref = {}
for label, dtype in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
    eg.to(dtype)
    for budget, bs in ((280, 16), (70, 64)):
        try:
            enc(imgs[:bs], budget, bs)  # warm-up
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = enc(imgs, budget, bs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            run = {"dtype": label, "budget": budget, "batch": bs, "n": len(imgs), "frames_per_s": round(len(imgs) / dt, 2),
                   "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            if label == "float32":
                ref[budget] = out
            else:
                cos = (out.astype("float32") * ref[budget]).sum(axis=1)
                run.update(cos_vs_float32_min=float(np.nanmin(cos)), nan=bool(np.isnan(out).any()))
        except Exception as err:
            run = {"dtype": label, "budget": budget, "error": f"{type(err).__name__}: {str(err)[:160]}"}
        l4["runs"].append(run)
        print(run)
eg.to(torch.float32)
out_name = "speed_l4.json" if "L4" in l4["gpu"] else "speed_" + l4["gpu"].split()[-1].split("-")[0].lower() + ".json"
(RES / out_name).write_text(json.dumps(l4, indent=1))
print("wrote", RES / out_name)
