# @title E3-E5: labeled Commons photo set -> Jev (by resolution), criteria with example images, repeat stability
"""Needs /content/jev/imgs/*.jpg and /content/jev/image_labels.json.

E3: every labeled image x long side {1024, 768, 512} with the 3 video questions in one request.
E4: images not used as examples, 1024 only: finger-heart question whose criteria carry example photos.
E5: 40 images x 3 extra repeats of the 1024 request (stability).
"""
import concurrent.futures as cf

meta = json.loads((WORK / "image_labels.json").read_text())
labels = {k: v for k, v in meta["images"].items() if v["label"] in ("pos", "neg")}
ex_ids = meta["examples"]["pos"] + meta["examples"]["neg"]
IMGS = WORK / "imgs"
_cache = {}


def img_bytes(img_id, long_side):
    key = (img_id, long_side)
    if key not in _cache:
        _cache[key] = prep_image(IMGS / f"{img_id}.jpg", long_side)
    return _cache[key]


def size_ok(data):
    w, h = PILImage.open(io.BytesIO(data)).size
    return min(w, h) >= 280 and max(w, h) <= 1568, [w, h]


def ex_image(img_id):
    return Image(img_bytes(img_id, 512))


fh_examples_q = {"fh_examples": Noul(
    instructions=FINGER_HEART_TEXT,
    criteria=NoulCriteria(
        true={"description": FINGER_HEART_TRUE, "examples": [ex_image(i) for i in meta["examples"]["pos"]]},
        false={"description": FINGER_HEART_FALSE, "examples": [ex_image(i) for i in meta["examples"]["neg"]]}))}
qs = video_questions()

jobs = []
for img_id in sorted(labels):
    for ls in (1024, 768, 512):
        jobs.append(("e3_images", img_id, ls, qs))
    if img_id not in ex_ids:
        jobs.append(("e4_examples", img_id, 1024, fh_examples_q))
rep_ids = sorted(i for i in labels if labels[i]["label"] == "pos")[:20] + sorted(i for i in labels if labels[i]["label"] == "neg")[:20]
for rep in range(3):
    for img_id in rep_ids:
        jobs.append((f"e5_repeat{rep}", img_id, 1024, qs))


def run(job):
    tag, img_id, ls, questions = job
    data = img_bytes(img_id, ls)
    ok, size = size_ok(data)
    m = {"id": img_id, "long_side": ls, "size": size, "label": labels[img_id]["label"], "group": labels[img_id]["group"]}
    if not ok:
        return {"tag": tag, "skipped": "size", **m}
    return call(tag, {"frame": Image(data)}, questions, meta=m)


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as pool:
    recs = list(pool.map(run, jobs))
wall = time.perf_counter() - t0
(RES / "e3_e5_image_preds.json").write_text(json.dumps(recs, ensure_ascii=False))
errs = [r for r in recs if r.get("error")]
skips = [r for r in recs if r.get("skipped")]
print(f"E3-E5: {len(recs)} jobs in {wall:.1f}s | errors {len(errs)} | skipped {len(skips)}")
for tag in ("e3_images", "e4_examples", "e5_repeat0"):
    t = [r["input_tokens"] for r in recs if r["tag"] == tag and r.get("input_tokens")]
    if t:
        print(f"  {tag}: mean input tokens {sum(t) / len(t):.0f} over {len(t)} requests")
if errs:
    print("  first error:", errs[0]["error"][:300])
