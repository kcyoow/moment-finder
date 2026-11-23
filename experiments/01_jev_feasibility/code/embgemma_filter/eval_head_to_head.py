"""Head-to-head: EmbeddingGemma 2 alone vs Jev alone on the same 243 labelled frames (pure Python, run locally).

Usage: python3 eval_head_to_head.py [out.json]

Each model gives one score per frame for "is someone making a finger heart?":
  Jev               probability from a Noul question (e7_video_preds.json)
  EmbeddingGemma 2  cosine similarity between a text (or example-photo) query and the frame (frame_scores.json)
A frame counts as detected when score >= threshold. Ambiguous frames are excluded from frame metrics.

Thresholds
  best_f1   threshold that maximises F1 on these same frames (optimistic for both models)
  natural   Jev only: 0.5
  held_out  threshold that maximises F1 on the other three videos, applied to the fourth (no peeking)
  all_pos   highest threshold that still catches every finger-heart frame
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEV = HERE.parent / "jev_feasibility"
sys.path.insert(0, str(JEV))
from eval_video import auc, runs, segment_eval  # noqa: E402

labels = json.load(open(JEV / "data" / "video_labels.json"))
VIDEOS = [v for v in labels if not v.startswith("_")]
POS = {(v, i) for v in VIDEOS for i in labels[v]["pos"]}
AMB = {(v, i) for v in VIDEOS for i in labels[v]["ambiguous"]}
TWO = {(v, i) for v in VIDEOS for i in labels[v]["hard_neg"].get("two_hand_heart", [])}


def counts(score, thr, keys):
    tp = sum(1 for k in keys if score[k] >= thr and k in POS)
    fp = sum(1 for k in keys if score[k] >= thr and k not in POS)
    fn = sum(1 for k in keys if score[k] < thr and k in POS)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def best_threshold(score, keys):
    best = None
    for t in sorted({score[k] for k in keys}, reverse=True):
        m = prf(*counts(score, t, keys))
        if best is None or m["f1"] > best[1]["f1"]:
            best = (t, m)
    return best


def scenes(score, thr_by_video, keys):
    """Scene-level result: a run of consecutive finger-heart frames is one scene."""
    tot = {"events": 0, "detected": 0, "false_segments": 0}
    for v in VIDEOS:
        pos = sorted(i for (vv, i) in POS if vv == v and (vv, i) in keys)
        amb = sorted(i for (vv, i) in AMB if vv == v)
        pred = [i for (vv, i) in keys if vv == v and score[(vv, i)] >= thr_by_video[v]]
        e = segment_eval(pred, pos, amb)
        tot["events"] += e["gt_events"]
        tot["detected"] += e["detected"]
        tot["false_segments"] += e["false_segments"]
    return tot


def evaluate(score, natural=None):
    keys = [k for k in score if k not in AMB]
    pos = [score[k] for k in keys if k in POS]
    neg = [score[k] for k in keys if k not in POS]
    two = [score[k] for k in keys if k in TWO]
    other = [score[k] for k in keys if k not in POS and k not in TWO]
    out = {"n_pos": len(pos), "n_neg": len(neg), "auc": auc(pos, neg), "auc_vs_two_hand": auc(pos, two),
           "auc_vs_other": auc(pos, other), "pos_range": [min(pos), max(pos)], "neg_range": [min(neg), max(neg)]}

    t, m = best_threshold(score, keys)
    out["best_f1"] = dict(m, threshold=t, **scenes(score, {v: t for v in VIDEOS}, keys))
    if natural is not None:
        out["natural"] = dict(prf(*counts(score, natural, keys)), threshold=natural,
                              **scenes(score, {v: natural for v in VIDEOS}, keys))
    t_all = min(pos)
    out["all_pos"] = dict(prf(*counts(score, t_all, keys)), threshold=t_all,
                          two_hand_flagged=sum(s >= t_all for s in two), n_two_hand=len(two),
                          **scenes(score, {v: t_all for v in VIDEOS}, keys))

    tp = fp = fn = 0
    thr_by_video = {}
    for held in VIDEOS:
        train = [k for k in keys if k[0] != held]
        t_held, _ = best_threshold(score, train)
        thr_by_video[held] = t_held
        a, b, c = counts(score, t_held, [k for k in keys if k[0] == held])
        tp, fp, fn = tp + a, fp + b, fn + c
    out["held_out"] = dict(prf(tp, fp, fn), thresholds=thr_by_video, **scenes(score, thr_by_video, keys))
    return out


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def line(name, r, key):
    m = r[key]
    return (f"  {name:32s} P {fmt(m['precision'])} R {fmt(m['recall'])} F1 {m['f1']:.2f} | scenes {m['detected']}/{m['events']}"
            f" | false segments {m['false_segments']}")


def main():
    jev_recs = json.load(open(JEV / "results" / "e7_video_preds.json"))
    emb = json.load(open(HERE / "results" / "frame_scores.json"))
    systems = {}
    for q, side in (("fh_crit", 1024), ("fh_plain", 1024), ("fh_crit", 768)):
        systems[f"Jev {q}@{side}px"] = ({(r["video"], r["idx"]): r["nouls"][q] for r in jev_recs
                                         if r["long_side"] == side and not r["error"]}, 0.5)
    hpath = HERE / "results" / "h2o_preds.json"
    if hpath.exists():  # H2O-Lightning-4B: same frames, same question texts as Jev (h2o_run.py, sequential pass)
        h2o_recs = [r for r in json.load(open(hpath))["records"] if r["tag"] == "seq3" and not r["error"]]
        for q in ("fh_crit", "fh_plain"):
            systems[f"H2O-Lightning {q}@1024px"] = ({(r["video"], r["idx"]): r["nouls"][q] for r in h2o_recs}, 0.5)
    for name, budget in (("en_detail", 280), ("en_plain", 280), ("ko_plain", 280), ("ex_pos_max", 280),
                         ("en_detail", 70), ("en_detail", 1120), ("en_plain", 1120)):
        systems[f"EmbGemma {name}@{budget}tok"] = ({(r["video"], r["idx"]): r["scores"][name] for r in emb["records"]
                                                    if r["budget"] == budget}, None)

    report = {"meta": emb["meta"], "systems": {}}
    for name, (score, natural) in systems.items():
        report["systems"][name] = evaluate(score, natural)

    print("Ranking quality (AUC = chance that a finger-heart frame outscores a non-finger-heart frame)")
    for name, r in report["systems"].items():
        print(f"  {name:32s} AUC {r['auc']:.3f} | vs two-hand heart {r['auc_vs_two_hand']:.3f} | vs other {r['auc_vs_other']:.3f}"
              f" | scores: finger heart {r['pos_range'][0]:.2f}..{r['pos_range'][1]:.2f}, others {r['neg_range'][0]:.2f}..{r['neg_range'][1]:.2f}")
    print("\nBest-F1 threshold chosen on these same frames (optimistic)")
    for name, r in report["systems"].items():
        print(line(name, r, "best_f1") + f" | thr {r['best_f1']['threshold']:.3f}")
    print("\nThreshold chosen without peeking (Jev: 0.5; held-out = tuned on the other three videos)")
    for name, r in report["systems"].items():
        if "natural" in r:
            print(line(name + " [0.5]", r, "natural"))
        print(line(name + " [held-out]", r, "held_out"))
    print("\nThreshold low enough to catch every finger-heart frame")
    for name, r in report["systems"].items():
        m = r["all_pos"]
        print(line(name, r, "all_pos") + f" | wrong frames flagged {m['fp']}/{r['n_neg']} (two-hand heart {m['two_hand_flagged']}/{m['n_two_hand']})")

    wpath = HERE / "results" / "window_scores.json"
    if wpath.exists():
        win = json.load(open(wpath))
        centre_keys = {(r["video"], r["idx"]) for r in win["records"]}
        print(f"\nVideo input (3-second clips, {len(centre_keys)} centre frames; query en_detail)")
        report["windows"] = {}
        wsys = {f"EmbGemma {mode}": {(r["video"], r["idx"]): r["scores"][mode]["en_detail"] for r in win["records"]}
                for mode in ("video140", "video280", "image_mean3", "image_centre")}
        jev_c = systems["Jev fh_crit@1024px"][0]
        wsys["Jev fh_crit@1024px (same frames)"] = {k: jev_c[k] for k in centre_keys}
        for name, score in wsys.items():
            r = evaluate(score)
            report["windows"][name] = r
            print(f"  {name:34s} AUC {r['auc']:.3f} | vs two-hand heart {r['auc_vs_two_hand']:.3f} | best F1 {r['best_f1']['f1']:.2f}"
                  f" (P {fmt(r['best_f1']['precision'])} R {fmt(r['best_f1']['recall'])}) | held-out F1 {r['held_out']['f1']:.2f}")

    out = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "head_to_head.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
