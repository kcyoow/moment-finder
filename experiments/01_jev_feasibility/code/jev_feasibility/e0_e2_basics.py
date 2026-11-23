# @title E0-E2: environment, text call, cookbook frames, image size limits, tokens, throughput
import concurrent.futures as cf
import platform
import statistics
import subprocess
import urllib.request
import zipfile
from importlib.metadata import version

out = {}

# E0 environment + text call ---------------------------------------------------
ffmpeg_line = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
out["env"] = {"python": platform.python_version(), "platform": platform.platform(), "ffmpeg": ffmpeg_line,
              "typesafe_sdk": version("typesafe-sdk"), "pydantic": version("pydantic"), "cpus": os.cpu_count()}
print("E0 env:", out["env"])

urgent_q = {"is_urgent": Noul(instructions="Does this convey urgency?",
                              criteria=NoulCriteria(true="Explicitly time-sensitive", false="No urgency expressed"))}
rec = call("e0_text", "Help! My payouts have been failing for 3 days.", urgent_q, images=False)
out["text_call"] = rec
print("E0 text call: noul", rec.get("nouls"), "(docs example 0.95 on jev-1.13.0) | tokens",
      rec.get("input_tokens"), "| latency", rec["latency"], "| model", rec.get("model"), "| error", rec["error"])

# E1 cookbook frames ------------------------------------------------------------
EXAMPLES = "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/cookbooks/feeder_watch"
frames_dir = WORK / "usfws_frames"
if not frames_dir.exists():
    urllib.request.urlretrieve(f"{EXAMPLES}/usfws_frames.zip", WORK / "usfws_frames.zip")
    zipfile.ZipFile(WORK / "usfws_frames.zip").extractall(WORK)
BIRD_Q = {"bird_present": Noul(instructions="Is there at least one bird in `frame`?")}
out["cookbook"] = []
for name, expected in [("u010.jpg", 0.98), ("u090.jpg", 0.95), ("u097.jpg", 0.07)]:
    path = frames_dir / name
    size = PILImage.open(path).size
    rec = call("e1_cookbook", {"frame": Image(path)}, BIRD_Q, meta={"file": name, "size": size, "expected": expected})
    out["cookbook"].append(rec)
    print(f"E1 {name} {size}: noul {rec.get('nouls', {}).get('bird_present')} (cookbook {expected}) | tokens {rec.get('input_tokens')} | {rec['latency']}s | err {rec['error']}")

# E2a image sizes vs tokens (and the documented 280px minimum) -----------------
src = frames_dir / "u010.jpg"
out["sizes"] = []
for long_side in [1024, 768, 640, 512, 448]:
    data = prep_image(src, long_side)
    size = PILImage.open(io.BytesIO(data)).size
    rec = call("e2_size", {"frame": Image(data)}, BIRD_Q, meta={"long_side": long_side, "size": size})
    out["sizes"].append(rec)
    print(f"E2 size {size}: tokens {rec.get('input_tokens')} | noul {rec.get('nouls', {}).get('bird_present')} | err {rec['error']}")

# E2b latency: sequential vs concurrent ----------------------------------------
data = prep_image(src, 1024)
def one(i, tag):
    return call(tag, {"frame": Image(data)}, BIRD_Q, meta={"i": i})

seq = [one(i, "e2_seq") for i in range(6)]
seq_lat = [r["latency"] for r in seq]
print(f"E2 sequential x6: median {statistics.median(seq_lat):.2f}s, min {min(seq_lat):.2f}, max {max(seq_lat):.2f}")
out["throughput"] = {"sequential_latency": seq_lat}
for workers in [8, 16]:
    n = workers * 3
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(workers) as pool:
        recs = list(pool.map(lambda i: one(i, f"e2_par{workers}"), range(n)))
    wall = time.perf_counter() - t0
    lat = sorted(r["latency"] for r in recs)
    errs = [r["error"] for r in recs if r["error"]]
    nouls = [r["nouls"]["bird_present"] for r in recs if not r["error"]]
    out["throughput"][f"par{workers}"] = {"n": n, "wall": round(wall, 2), "req_per_s": round(n / wall, 2),
                                          "p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1],
                                          "errors": errs, "noul_values": sorted(set(nouls))}
    print(f"E2 concurrent {workers} workers x{n}: wall {wall:.1f}s -> {n / wall:.1f} req/s | p50 {lat[len(lat) // 2]:.2f}s p95 {lat[int(len(lat) * 0.95) - 1]:.2f}s | errors {len(errs)} | distinct noul {sorted(set(nouls))}")

(RES / "e0_e2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("saved", RES / "e0_e2.json")
