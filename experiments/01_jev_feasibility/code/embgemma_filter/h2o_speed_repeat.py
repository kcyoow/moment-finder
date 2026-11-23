# diagnostic: is the H2O-Lightning run limited by the GPU or by the VM's CPU?
import threading, psutil
def monitored(label, fn):
    samples, stop = [], threading.Event()
    def mon():
        while not stop.is_set():
            g = subprocess.run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True,
                               capture_output=True, text=True).stdout.strip()
            samples.append((float(g or 0), psutil.cpu_percent(interval=0.4)))
    th = threading.Thread(target=mon); th.start()
    t0 = time.perf_counter(); out = fn(); wall = time.perf_counter() - t0
    stop.set(); th.join()
    gpu = sum(s[0] for s in samples) / len(samples); cpu = sum(s[1] for s in samples) / len(samples)
    shim_ms = sorted(r["usage"]["latency_ms"] for r in out if not r["error"])
    res = {"label": label, "n": len(out), "frames_per_s": round(len(out) / wall, 2), "gpu_util_mean": round(gpu), "cpu_util_mean": round(cpu),
           "shim_latency_ms_p50": shim_ms[len(shim_ms) // 2]}
    print(res); return res
sub = frames[:96]
diag = {"cpu_count": os.cpu_count(), "runs": []}
diag["runs"].append(monitored("sequential", lambda: [ask((f, QUESTIONS, "diag1")) for f in sub[:48]]))
for k in (4, 16):
    def run(k=k):
        with cf.ThreadPoolExecutor(k) as pool:
            return list(pool.map(ask, [(f, QUESTIONS, f"diag{k}") for f in sub]))
    diag["runs"].append(monitored(f"{k} in flight", run))
print("cpu_count", diag["cpu_count"])
(RES / "h2o_diag.json").write_text(json.dumps(diag, indent=1))
