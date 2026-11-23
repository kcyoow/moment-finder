"""Score E7 video predictions against data/video_labels.json (pure Python, run locally).

Usage: python3 eval_video.py results/e7_video_preds.json data/video_labels.json [out.json]
"""
import json
import sys
from collections import defaultdict


def auc(pos, neg):
    """Probability a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(scores, labels, thr):
    tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= thr and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s < thr and y)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def runs(idxs, gap=1):
    """Group sorted frame indices into [start, end] runs allowing `gap` missing frames."""
    out = []
    for i in sorted(idxs):
        if out and i - out[-1][1] <= gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def segment_eval(pred_idx, gt_idx, dontcare, gap=1):
    pred, gt = runs(pred_idx, gap), runs(gt_idx, gap)
    def overlaps(seg, idxs):
        return any(seg[0] <= i <= seg[1] for i in idxs)
    hit = sum(1 for g in gt if any(p[0] <= g[1] and g[0] <= p[1] for p in pred))
    false = [p for p in pred if not overlaps(p, set(gt_idx) | set(dontcare))]
    return {"gt_events": len(gt), "detected": hit, "pred_segments": len(pred), "false_segments": len(false),
            "false_list": false}


def main():
    preds = json.load(open(sys.argv[1]))
    labels = json.load(open(sys.argv[2]))
    by = defaultdict(dict)  # (video, long_side) -> idx -> rec
    for r in preds:
        by[(r["video"], r["long_side"])][r["idx"]] = r
    videos = [v for v in labels if not v.startswith("_")]
    targets = {
        "fh_plain": lambda lab: (set(lab["pos"]), set(lab["ambiguous"])),
        "fh_crit": lambda lab: (set(lab["pos"]), set(lab["ambiguous"])),
        "two_hand_heart": lambda lab: (set(lab["hard_neg"].get("two_hand_heart", [])), set()),
    }
    report = {}
    for q, get_gt in targets.items():
        for ls in sorted({r["long_side"] for r in preds}, reverse=True):
            scores, ys, groups = [], [], defaultdict(list)
            seg = {"gt_events": 0, "detected": 0, "pred_segments": 0, "false_segments": 0, "false_list": []}
            seg5 = dict(seg, false_list=[])
            for v in videos:
                lab = labels[v]
                pos, amb = get_gt(lab)
                frames = by[(v, ls)]
                for idx, r in sorted(frames.items()):
                    s = r["nouls"][q]
                    if idx in amb:
                        groups["ambiguous"].append(s)
                        continue
                    scores.append(s)
                    ys.append(idx in pos)
                    if q.startswith("fh"):
                        for name, ids in lab["hard_neg"].items():
                            if idx in ids:
                                groups[f"hardneg_{name}"].append(s)
                    if idx in pos:
                        groups["pos"].append(s)
                    elif not any(idx in ids for ids in lab["hard_neg"].values()):
                        groups["other_neg"].append(s)
                for thr, acc in ((0.8, seg), (0.5, seg5)):
                    e = segment_eval([i for i, r in frames.items() if r["nouls"][q] >= thr and i not in amb],
                                     sorted(pos), sorted(amb))
                    for k in ("gt_events", "detected", "pred_segments", "false_segments"):
                        acc[k] += e[k]
                    acc["false_list"] += [(v, f) for f in e["false_list"]]
            pos_s = [s for s, y in zip(scores, ys) if y]
            neg_s = [s for s, y in zip(scores, ys) if not y]
            best = max((prf(scores, ys, t / 100) for t in range(5, 100, 5)), key=lambda d: d["f1"])
            report[f"{q}@{ls}"] = {
                "n_pos": len(pos_s), "n_neg": len(neg_s), "auc": auc(pos_s, neg_s),
                "at_0.5": prf(scores, ys, 0.5), "at_0.8": prf(scores, ys, 0.8), "best_f1": best,
                "mean_by_group": {g: round(sum(v) / len(v), 3) for g, v in groups.items() if v},
                "max_by_group": {g: max(v) for g, v in groups.items() if v},
                "segments_at_0.8": seg, "segments_at_0.5": seg5,
            }
    if len(sys.argv) > 3:
        json.dump(report, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    for key, r in report.items():
        a5, a8, b = r["at_0.5"], r["at_0.8"], r["best_f1"]
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{key:22s} pos {r['n_pos']:3d} neg {r['n_neg']:3d} | AUC {fmt(r['auc'])} | "
              f"@0.5 P {fmt(a5['precision'])} R {fmt(a5['recall'])} | @0.8 P {fmt(a8['precision'])} R {fmt(a8['recall'])} | "
              f"best F1 {b['f1']:.2f} @ {b['thr']:.2f}")
        print(f"{'':22s} mean noul by group {r['mean_by_group']}")
        print(f"{'':22s} max  noul by group {r['max_by_group']}")
        s8, s5 = r["segments_at_0.8"], r["segments_at_0.5"]
        print(f"{'':22s} segments@0.8: events {s8['detected']}/{s8['gt_events']} detected, false segs {s8['false_segments']} {s8['false_list']}")
        print(f"{'':22s} segments@0.5: events {s5['detected']}/{s5['gt_events']} detected, false segs {s5['false_segments']} {s5['false_list']}")


if __name__ == "__main__":
    main()
