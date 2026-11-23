# @title H2O-Lightning-4B throughput at several concurrency levels, on frames the server has not seen
"""Run after h2o_run.py in the same kernel (reuses ask, frames, URIS, QUESTIONS).
Writes /content/eg/results/h2o_speed_<mode>.json, where mode is the server mode h2o_serve.py started ("stock" or "patched").

vLLM caches the prefix of a prompt it has already read, so re-sending the same frames looks faster than real
use. Each run below therefore uses its own altered copies of the frames (mirrored and slightly brightened or
darkened), which the server has never seen. GPU and CPU load are sampled during each run.
"""
import io
import threading

import psutil
from PIL import Image as PILImage, ImageEnhance, ImageOps


def monitored(label, fn):
    samples, stop = [], threading.Event()

    def mon():
        while not stop.is_set():
            g = subprocess.run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True,
                               capture_output=True, text=True).stdout.strip()
            samples.append((float(g or 0), psutil.cpu_percent(interval=0.4)))

    th = threading.Thread(target=mon)
    th.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    ok = [r for r in out if not r["error"]]
    shim_ms = sorted(r["usage"]["latency_ms"] for r in ok)
    client_s = sorted(r["latency"] for r in ok)
    res = {"label": label, "n": len(out), "errors": len(out) - len(ok), "frames_per_s": round(len(out) / wall, 2),
           "gpu_util_mean": round(sum(s[0] for s in samples) / len(samples)),
           "cpu_util_mean": round(sum(s[1] for s in samples) / len(samples)),
           "shim_latency_ms_p50": shim_ms[len(shim_ms) // 2] if shim_ms else None,
           "client_latency_s_p50": client_s[len(client_s) // 2] if client_s else None,
           "client_latency_s_p95": client_s[min(len(client_s) - 1, int(0.95 * len(client_s)))] if client_s else None}
    if len(ok) < len(out):
        res["first_error"] = next(r["error"] for r in out if r["error"])[:200]
    print(res)
    return res


def unseen(tag, subset, brightness):
    out = []
    for v, idx, sec, p in subset:
        img = ImageEnhance.Brightness(ImageOps.mirror(PILImage.open(p).convert("RGB"))).enhance(brightness)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        URIS[(f"{v}_{tag}", idx)] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        out.append((f"{v}_{tag}", idx, sec, p))
    return out


def pool_run(jobs, k):
    with cf.ThreadPoolExecutor(k) as pool:
        return list(pool.map(ask, jobs))


one = {"fh_crit": QUESTIONS["fh_crit"]}
mode = globals().get("H2O_MODE", "stock")
speed_runs = {"gpu": H2O_META["gpu"], "cpu_count": os.cpu_count(), "mode": mode, "serve_env": H2O_META.get("serve_env"),
              "runs": []}
plan = [("u1", 1.00, 1, QUESTIONS, 60, "1 at a time, 3 questions"),
        ("u8", 1.03, 8, QUESTIONS, 243, "8 in flight, 3 questions"),
        ("u16", 1.09, 16, QUESTIONS, 243, "16 in flight, 3 questions"),
        ("u32", 0.97, 32, QUESTIONS, 243, "32 in flight, 3 questions"),
        ("u64", 1.06, 64, QUESTIONS, 243, "64 in flight, 3 questions"),
        ("u128", 0.91, 128, QUESTIONS, 243, "128 in flight, 3 questions"),
        ("o1", 1.12, 1, one, 60, "1 at a time, 1 question"),
        ("o32", 0.94, 32, one, 243, "32 in flight, 1 question"),
        ("o64", 0.88, 64, one, 243, "64 in flight, 1 question")]
for tag, brightness, k, qs, n, label in plan:
    sub = unseen(tag, frames[:n], brightness)
    jobs = [(f, qs, tag) for f in sub]
    run = monitored(label, (lambda: [ask(j) for j in jobs]) if k == 1 else (lambda: pool_run(jobs, k)))
    speed_runs["runs"].append(run)
    for key in [k2 for k2 in URIS if k2[0].endswith("_" + tag)]:
        del URIS[key]

(RES / f"h2o_speed_{mode}.json").write_text(json.dumps(speed_runs, indent=1))
print("wrote", RES / f"h2o_speed_{mode}.json")
