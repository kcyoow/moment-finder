# speed on frames the server has not seen (mirrored copies), so vLLM's prefix cache cannot help
import io
from PIL import Image as PILImage, ImageOps
def mirrored(tag, subset):
    out = []
    for v, idx, sec, p in subset:
        buf = io.BytesIO(); ImageOps.mirror(PILImage.open(p).convert("RGB")).save(buf, format="JPEG", quality=92)
        key = (f"{v}_{tag}", idx)
        URIS[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        out.append((key[0], idx, sec, p))
    return out
a, b, c = mirrored("m1", frames[0:60]), mirrored("m2", frames[60:160]), mirrored("m3", frames[160:240])
fresh = {"runs": []}
fresh["runs"].append(monitored("unseen, sequential", lambda: [ask((f, QUESTIONS, "fresh1")) for f in a]))
def run8():
    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(ask, [(f, QUESTIONS, "fresh8") for f in b]))
fresh["runs"].append(monitored("unseen, 8 in flight", run8))
one = {"fh_crit": QUESTIONS["fh_crit"]}
def run8one():
    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(ask, [(f, one, "fresh8one") for f in c]))
fresh["runs"].append(monitored("unseen, 8 in flight, 1 question", run8one))
diag["fresh"] = fresh["runs"]
(RES / "h2o_diag.json").write_text(json.dumps(diag, indent=1))
