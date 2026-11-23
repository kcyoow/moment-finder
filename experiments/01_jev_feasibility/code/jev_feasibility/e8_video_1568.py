# @title E8: same 1 fps frames at the maximum accepted size (long side 1568) -> does more resolution help small hands?
import concurrent.futures as cf
import csv
import re
import subprocess

videos = json.loads((WORK / "videos.json").read_text())
VF = ("select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,1)',showinfo,"
      "scale=1568:1568:force_original_aspect_ratio=decrease:force_divisible_by=2")
jobs = []
for v in videos:
    out = WORK / "videos" / f"{v['id']}_1568"
    out.mkdir(exist_ok=True)
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(WORK / "videos" / f"{v['id']}.webm"), "-vf", VF,
                           "-fps_mode", "passthrough", "-q:v", "2", str(out / "%06d.jpg")], capture_output=True, text=True)
    times = [float(t) for t in re.findall(r"pts_time:\s*([0-9.]+)", proc.stderr)]
    files = sorted(out.glob("*.jpg"))
    ref = list(csv.DictReader(open(WORK / "videos" / v["id"] / "frames.csv")))
    assert len(files) == len(times) == len(ref), (v["id"], len(files), len(times), len(ref))
    assert all(abs(t - float(r["seconds"])) < 1e-3 for t, r in zip(times, ref)), "frame times differ from the 1024 run"
    for idx, (p, t) in enumerate(zip(files, times), 1):
        jobs.append((v["id"], idx, t, p))
questions = video_questions()


def run8(job):
    vid, idx, sec, path = job
    data = path.read_bytes()
    return call("e8_video_1568", {"frame": Image(data)}, questions,
                meta={"video": vid, "idx": idx, "seconds": sec, "long_side": 1568,
                      "size": list(PILImage.open(io.BytesIO(data)).size)})


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as pool:
    recs8 = list(pool.map(run8, jobs))
wall = time.perf_counter() - t0
old = json.loads((RES / "e7_video_preds.json").read_text())
(RES / "e7_video_preds.json").write_text(json.dumps([r for r in old if r["long_side"] != 1568] + recs8, ensure_ascii=False))
errs = [r for r in recs8 if r["error"]]
tok = [r["input_tokens"] for r in recs8 if not r["error"]]
print(f"E8: {len(recs8)} requests in {wall:.1f}s | errors {len(errs)} | size {recs8[0]['size']} | mean tokens {sum(tok) / max(1, len(tok)):.0f}")
if errs:
    print("  first error:", errs[0]["error"][:300])
