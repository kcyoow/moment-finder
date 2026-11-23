# @title E7 run: every 1 fps frame of the 4 test videos -> Jev (3 criteria in one request), at 1024 and 768
import concurrent.futures as cf
import csv

videos = json.loads((WORK / "videos.json").read_text())
jobs = []
for v in videos:
    rows = list(csv.DictReader(open(WORK / "videos" / v["id"] / "frames.csv")))
    for idx, row in enumerate(rows, 1):
        for long_side in (1024, 768):
            jobs.append((v["id"], idx, float(row["seconds"]), WORK / "videos" / v["id"] / row["file"], long_side))
questions = video_questions()


def run(job):
    vid, idx, sec, path, long_side = job
    data = path.read_bytes() if long_side == 1024 else prep_image(path, long_side)
    return call("e7_video", {"frame": Image(data)}, questions,
                meta={"video": vid, "idx": idx, "seconds": sec, "long_side": long_side,
                      "size": list(PILImage.open(io.BytesIO(data)).size)})


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as pool:
    recs = list(pool.map(run, jobs))
wall = time.perf_counter() - t0
errors = [r for r in recs if r["error"]]
(RES / "e7_video_preds.json").write_text(json.dumps(recs, ensure_ascii=False))
tok = {ls: [r["input_tokens"] for r in recs if r["long_side"] == ls and not r["error"]] for ls in (1024, 768)}
print(f"E7: {len(recs)} requests in {wall:.1f}s ({len(recs) / wall:.1f} req/s), errors {len(errors)}")
for ls, t in tok.items():
    print(f"  long side {ls}: mean input tokens {sum(t) / len(t):.0f} (3 questions/request)")
if errors:
    print("  first error:", errors[0]["error"])
