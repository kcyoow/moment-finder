# @title H2O-Lightning-4B run: the same 243 frames and the same three questions as the Jev experiment
"""Run after h2o_serve.py in the same kernel. Writes /content/eg/results/h2o_preds.json.

Request shape is the TypeSafe /v1/systemone shape the shim serves: state = {"frame": <data URI>}, three Noul
questions in one request (texts copied from experiments/jev_feasibility/questions.py).
Passes: (1) one request at a time -> latency; (2) 16 in flight -> throughput; (3) one question per request -> per-decision latency.
"""
import base64
import concurrent.futures as cf
import csv
import statistics
import urllib.request

URL = "http://127.0.0.1:8741/v1/systemone"
FINGER_HEART_TEXT = "Is anyone in `frame` making a finger heart gesture?"
FINGER_HEART_TRUE = ("At least one hand shows the Korean finger heart: the thumb and index finger are crossed "
                     "so their tips form a small heart. One hand is enough; both hands doing it also counts.")
FINGER_HEART_FALSE = ("No hand shows that gesture. Other gestures do not count, including a big heart made with "
                      "both hands together, waving, a V sign, a thumbs up, an OK sign, a fist, or hands that are "
                      "not visible.")
TWO_HAND_HEART_TEXT = "Is anyone in `frame` forming one big heart shape with both hands together?"
QUESTIONS = {
    "fh_plain": {"type": "noul", "instructions": FINGER_HEART_TEXT},
    "fh_crit": {"type": "noul", "instructions": FINGER_HEART_TEXT,
                "criteria": {"true": FINGER_HEART_TRUE, "false": FINGER_HEART_FALSE}},
    "two_hand_heart": {"type": "noul", "instructions": TWO_HAND_HEART_TEXT},
}
VIDEOS = ["v1_chaeunwoo", "v2_goahsung", "v3_jungkyungho", "v4_choiyoojung"]
frames = []
for v in VIDEOS:
    rows = list(csv.DictReader(open(WORK / "frames" / v / "frames.csv")))
    for idx, row in enumerate(rows, 1):
        frames.append((v, idx, float(row["seconds"]), WORK / "frames" / v / row["file"]))
URIS = {(v, idx): "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() for v, idx, _, p in frames}


def ask(job):
    (v, idx, sec, _), questions, tag = job
    body = json.dumps({"state": {"frame": URIS[(v, idx)]}, "questions": questions}).encode()
    rec = {"tag": tag, "video": v, "idx": idx, "seconds": sec, "long_side": 1024, "error": None}
    t0 = time.perf_counter()
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=300)
        out = json.load(r)
        rec.update(nouls={k: a["noul"] for k, a in out["answers"].items()}, usage=out.get("usage"),
                   model=out.get("model"), attention_mode=out.get("attention_mode"))
    except Exception as err:
        detail = err.read().decode(errors="replace")[:300] if hasattr(err, "read") else str(err)[:300]
        rec["error"] = f"{type(err).__name__}: {detail}"
    rec["latency"] = round(time.perf_counter() - t0, 4)
    return rec


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


summary, records = {}, []
ask((frames[0], QUESTIONS, "warm"))  # first image request pays one-time costs

t0 = time.perf_counter()
seq = [ask((f, QUESTIONS, "seq3")) for f in frames]
wall = time.perf_counter() - t0
lat = [r["latency"] for r in seq if not r["error"]]
summary["seq3"] = {"n": len(seq), "errors": sum(bool(r["error"]) for r in seq), "wall_s": round(wall, 1),
                   "frames_per_s": round(len(seq) / wall, 2), "latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95),
                   "mean_input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in seq if not r["error"])}
records += seq
print("sequential, 3 questions/request:", summary["seq3"])
if seq[0]["error"]:
    print("first error:", seq[0]["error"])

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(16) as pool:
    par = list(pool.map(ask, [(f, QUESTIONS, "par16") for f in frames]))
wall = time.perf_counter() - t0
summary["par16"] = {"n": len(par), "errors": sum(bool(r["error"]) for r in par), "wall_s": round(wall, 1),
                    "frames_per_s": round(len(par) / wall, 2)}
records += par
print("16 in flight, 3 questions/request:", summary["par16"])
diff = [abs(a["nouls"]["fh_crit"] - b["nouls"]["fh_crit"]) for a, b in zip(seq, par) if not a["error"] and not b["error"]]
summary["seq_vs_par_max_diff"] = max(diff) if diff else None

one = {"fh_crit": QUESTIONS["fh_crit"]}
t0 = time.perf_counter()
single = [ask((f, one, "seq1")) for f in frames[:80]]
wall = time.perf_counter() - t0
lat = [r["latency"] for r in single if not r["error"]]
summary["seq1"] = {"n": len(single), "errors": sum(bool(r["error"]) for r in single),
                   "frames_per_s": round(len(single) / wall, 2), "latency_p50": pct(lat, 0.5), "latency_p95": pct(lat, 0.95)}
records += single
print("sequential, 1 question/request:", summary["seq1"])
print("max |seq - par| on fh_crit:", summary["seq_vs_par_max_diff"])
print("sample:", {k: seq[23][k] for k in ("video", "idx", "nouls", "usage", "attention_mode")})

mode = globals().get("H2O_MODE", "stock")  # set by h2o_serve.py; the L4 run predates it and was "stock"
preds_path = RES / ("h2o_preds.json" if mode == "stock" else f"h2o_preds_{mode}.json")
preds_path.write_text(json.dumps({"meta": H2O_META, "summary": summary, "records": records}, ensure_ascii=False))
print("wrote", preds_path, len(records), "records")
