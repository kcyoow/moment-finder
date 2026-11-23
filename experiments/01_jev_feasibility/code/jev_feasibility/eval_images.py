"""Score E3-E5 photo predictions (pure Python, run locally).

Usage: python3 eval_images.py results/e3_e5_image_preds.json [out.json]
"""
import json
import statistics
import sys
from collections import defaultdict

from eval_video import auc, prf


def summarize(recs, q):
    scores = [r["nouls"][q] for r in recs]
    ys = [r["label"] == "pos" for r in recs]
    pos = [s for s, y in zip(scores, ys) if y]
    neg = [s for s, y in zip(scores, ys) if not y]
    best = max((prf(scores, ys, t / 100) for t in range(5, 100, 5)), key=lambda d: d["f1"])
    by_group = defaultdict(list)
    for r in recs:
        by_group[r["group"]].append(r["nouls"][q])
    return {"n_pos": len(pos), "n_neg": len(neg), "auc": auc(pos, neg), "at_0.5": prf(scores, ys, 0.5),
            "at_0.8": prf(scores, ys, 0.8), "best_f1": best,
            "group_mean": {g: round(statistics.mean(v), 3) for g, v in sorted(by_group.items())},
            "group_rate_ge_0.5": {g: f"{sum(s >= 0.5 for s in v)}/{len(v)}" for g, v in sorted(by_group.items())}}


def main():
    recs = [r for r in json.load(open(sys.argv[1])) if not r.get("skipped") and not r.get("error")]
    skipped = [r for r in json.load(open(sys.argv[1])) if r.get("skipped") or r.get("error")]
    report = {"skipped_or_error": [(r["tag"], r["id"], r["long_side"], r.get("skipped") or r.get("error")) for r in skipped]}
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    e3 = [r for r in recs if r["tag"] == "e3_images"]
    e4 = [r for r in recs if r["tag"] == "e4_examples"]
    e4_ids = {r["id"] for r in e4}
    rows = []
    for q in ("fh_plain", "fh_crit"):
        for ls in (1024, 768, 512):
            rows.append((f"E3 {q}@{ls}", summarize([r for r in e3 if r["long_side"] == ls], q)))
    rows.append(("E4 fh_crit@1024 (same images as E4)", summarize([r for r in e3 if r["long_side"] == 1024 and r["id"] in e4_ids], "fh_crit")))
    rows.append(("E4 fh_examples@1024", summarize(e4, "fh_examples")))
    for name, s in rows:
        report[name] = s
        a5, a8, b = s["at_0.5"], s["at_0.8"], s["best_f1"]
        print(f"{name:38s} pos {s['n_pos']:3d} neg {s['n_neg']:3d} | AUC {fmt(s['auc'])} | @0.5 P {fmt(a5['precision'])} R {fmt(a5['recall'])} "
              f"| @0.8 P {fmt(a8['precision'])} R {fmt(a8['recall'])} | best F1 {b['f1']:.2f}@{b['thr']:.2f}")
        print(f"{'':38s} share >= 0.5 by group: {s['group_rate_ge_0.5']}")
    # tokens by long side
    for ls in (1024, 768, 512):
        t = [r["input_tokens"] for r in e3 if r["long_side"] == ls]
        print(f"tokens @{ls}: mean {statistics.mean(t):.0f} (3 questions)")
    t4 = [r["input_tokens"] for r in e4]
    print(f"tokens E4 (1 question + 6 example images @512): mean {statistics.mean(t4):.0f}")
    # E5 repeat stability: same request 4 times (e3 1024 + 3 repeats)
    base = {r["id"]: r for r in e3 if r["long_side"] == 1024}
    reps = defaultdict(list)
    for r in recs:
        if r["tag"].startswith("e5_repeat"):
            reps[r["id"]].append(r)
    diffs, flips, n = defaultdict(list), defaultdict(int), 0
    for img_id, rr in reps.items():
        allr = [base[img_id]] + rr
        n += 1
        for q in ("fh_plain", "fh_crit", "two_hand_heart"):
            vals = [x["nouls"][q] for x in allr]
            diffs[q].append(max(vals) - min(vals))
            flips[q] += len({v >= 0.5 for v in vals}) > 1
    report["E5"] = {q: {"images": n, "mean_range": round(statistics.mean(d), 4), "max_range": max(d),
                        "decision_flips_at_0.5": flips[q]} for q, d in diffs.items()}
    for q, s in report["E5"].items():
        print(f"E5 {q}: {s['images']} images x 4 runs | mean max-min {s['mean_range']:.3f} | worst {s['max_range']:.2f} | flips at 0.5: {s['decision_flips_at_0.5']}")
    print("skipped/error:", report["skipped_or_error"][:10])
    if len(sys.argv) > 2:
        json.dump(report, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
