# @title speed: frames per second for image embedding on this VM (GPU and CPU), and where the time goes
"""Run after embed_frames.py in the same kernel (reuses model, frame_images, encode_images, EMB).
Writes /content/eg/results/speed.json.

Timing covers preprocessing + model forward on in-memory 1024x576 PIL images (JPEG decode excluded).
The 1120-token budget is not repeated here; embed_frames.py already timed it on all 243 frames.
"""
import os

speed = {"cpu_count": os.cpu_count(), "runs": [], "full_run_timing": timing}


def bench(device, budget, batch_size, n, label="float32"):
    imgs = frame_images[:n]
    encode_images(imgs[:batch_size], budget, batch_size)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = encode_images(imgs, budget, batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    run = {"device": device, "dtype": label, "budget": budget, "batch": batch_size, "n": n,
           "frames_per_s": round(n / dt, 2),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
    speed["runs"].append(run)
    print(run)
    return out


if DEVICE == "cuda":
    for budget, bs in ((280, 32), (280, 8), (70, 64)):
        bench("cuda", budget, bs, 64)

    # CPU-side preprocessing alone (resize + patching), to see how much of the time is not the GPU.
    try:
        processor = model[0].processor
    except AttributeError:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    for budget in (280, 70):
        t0 = time.perf_counter()
        feats = processor(images=frame_images[:16], max_soft_tokens=budget, return_tensors="pt")
        dt = time.perf_counter() - t0
        shapes = {k: tuple(v.shape) for k, v in feats.items() if hasattr(v, "shape")}
        speed.setdefault("preprocess", []).append({"budget": budget, "seconds_per_frame": round(dt / 16, 4), "shapes": shapes})
        print("preprocess only", budget, f"{dt / 16 * 1000:.0f} ms/frame", shapes)

    # Half precision: faster on a T4, but only usable if the embeddings stay the same.
    ref = EMB[280][:64]
    for label, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        if label == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("bfloat16 not supported on this GPU")
            continue
        try:
            model.to(dtype)
            out = bench("cuda", 280, 32, 64, label=label)
            cos = (out.astype("float32") * ref).sum(axis=1)
            agree = {"dtype": label, "nan": bool(np.isnan(out).any()), "cos_vs_float32_mean": float(np.nanmean(cos)),
                     "cos_vs_float32_min": float(np.nanmin(cos))}
            speed.setdefault("half_precision_agreement", []).append(agree)
            print(agree)
        except Exception as err:  # keep going; this is an optional variant
            print(label, "failed:", type(err).__name__, str(err)[:200])
        finally:
            model.to(torch.float32)

model.to("cpu")
for budget, bs, n in ((70, 4, 8), (280, 4, 8)):
    bench("cpu", budget, bs, n)
model.to(DEVICE)

(RES / "speed.json").write_text(json.dumps(speed, indent=1))
print("wrote", RES / "speed.json")
