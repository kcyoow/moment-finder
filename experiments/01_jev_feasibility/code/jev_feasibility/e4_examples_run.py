# @title E4 rerun: finger-heart criteria with example photos (after replacing the undersized example)
"""Run after e3_images_run.py in the same kernel (reuses img_bytes, size_ok, labels).

Example photos are shrunk to long side 512 but never below 280px on the short side,
because one out-of-range image makes the whole request fail (400 Invalid image dimensions).
"""
import concurrent.futures as cf
import math

meta = json.loads((WORK / "image_labels.json").read_text())
ex_ids = meta["examples"]["pos"] + meta["examples"]["neg"]


def example_bytes(img_id):
    w, h = PILImage.open(IMGS / f"{img_id}.jpg").size
    long_side = max(512, math.ceil(280 * max(w, h) / min(w, h)))
    data = img_bytes(img_id, long_side)
    ok, size = size_ok(data)
    assert ok, (img_id, size)
    return data


fh_examples_q = {"fh_examples": Noul(
    instructions=FINGER_HEART_TEXT,
    criteria=NoulCriteria(
        true={"description": FINGER_HEART_TRUE, "examples": [Image(example_bytes(i)) for i in meta["examples"]["pos"]]},
        false={"description": FINGER_HEART_FALSE, "examples": [Image(example_bytes(i)) for i in meta["examples"]["neg"]]}))}

jobs = [("e4_examples", i, 1024, fh_examples_q) for i in sorted(labels) if i not in ex_ids]
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as pool:
    recs4 = list(pool.map(run, jobs))
wall = time.perf_counter() - t0
old = json.loads((RES / "e3_e5_image_preds.json").read_text())
merged = [r for r in old if r["tag"] != "e4_examples"] + recs4
(RES / "e3_e5_image_preds.json").write_text(json.dumps(merged, ensure_ascii=False))
errs = [r for r in recs4 if r.get("error")]
tok = [r["input_tokens"] for r in recs4 if r.get("input_tokens")]
print(f"E4: {len(recs4)} jobs in {wall:.1f}s | errors {len(errs)} | skipped {sum(1 for r in recs4 if r.get('skipped'))} | mean tokens {sum(tok) / max(1, len(tok)):.0f}")
if errs:
    print("  first error:", errs[0]["error"][:300])
