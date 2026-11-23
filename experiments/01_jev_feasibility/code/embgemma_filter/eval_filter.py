"""Compare "Jev only" with "EmbeddingGemma 2 filter -> Jev" on the 243 labelled frames (pure Python, run locally).

Usage: python3 eval_filter.py [out.json]

Inputs
  results/frame_scores.json                      EmbeddingGemma 2 cosine scores per frame (embed_frames.py)
  ../jev_feasibility/results/e7_video_preds.json Jev probabilities per frame (no new Jev calls)
  ../jev_feasibility/data/video_labels.json      ground truth, labelled before any model ran

The filter keeps a frame when its EmbeddingGemma score >= t (one threshold for all videos); with
neighbours=1 the frames right before and after a kept frame are kept too. Jev then judges the kept
frames only (fh_crit @1024px, positive when >= 0.5).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEV = HERE.parent / "jev_feasibility"
sys.path.insert(0, str(JEV))
from eval_video import auc, runs, segment_eval  # noqa: E402

JEV_Q, JEV_SIDE, JEV_THR = "fh_crit", 1024, 0.5
PRIMARY = (280, "en_detail")  # fixed before looking at results


def load():
    labels = json.load(open(JEV / "data" / "video_labels.json"))
    videos = [v for v in labels if not v.startswith("_")]
    jev = {(r["video"], r["idx"]): r["nouls"][JEV_Q]
           for r in json.load(open(JEV / "results" / "e7_video_preds.json"))
           if r["long_side"] == JEV_SIDE and not r["error"]}
    emb = json.load(open(HERE / "results" / "frame_scores.json"))
    scores = defaultdict(dict)  # (budget, name) -> (video, idx) -> score
    for r in emb["records"]:
        for name, s in r["scores"].items():
            scores[(r["budget"], name)][(r["video"], r["idx"])] = s
    return labels, videos, jev, emb, scores


def kept_set(score, t, videos, n_frames, neighbours):
    kept = {k for k, s in score.items() if s >= t}
    if neighbours:
        for v, i in list(kept):
            for j in range(i - neighbours, i + neighbours + 1):
                if 1 <= j <= n_frames[v]:
                    kept.add((v, j))
    return kept


def pipeline(kept, labels, videos, jev, n_frames):
    """Frame and scene metrics for 'Jev on the kept frames only'. kept=None means Jev on every frame."""
    tp = fp = fn = 0
    events = detected = false_segs = 0
    for v in videos:
        pos, amb = set(labels[v]["pos"]), set(labels[v]["ambiguous"])
        pred = []
        for i in range(1, n_frames[v] + 1):
            sent = kept is None or (v, i) in kept
            hit = sent and jev[(v, i)] >= JEV_THR
            if i in amb:
                continue
            if hit:
                pred.append(i)
            tp += hit and i in pos
            fp += hit and i not in pos
            fn += (not hit) and i in pos
        e = segment_eval(pred, sorted(pos), sorted(amb))
        events += e["gt_events"]
        detected += e["detected"]
        false_segs += e["false_segments"]
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "events_detected": detected, "events": events, "false_segments": false_segs}


def filter_stats(kept, labels, videos, n_frames):
    total = sum(n_frames.values())
    n_pos = n_neg = pos_kept = neg_kept = scenes = scenes_kept = 0
    per_video = {}
    for v in videos:
        pos, amb = set(labels[v]["pos"]), set(labels[v]["ambiguous"])
        neg = [i for i in range(1, n_frames[v] + 1) if i not in pos and i not in amb]
        n_pos += len(pos)
        n_neg += len(neg)
        pos_kept += sum((v, i) in kept for i in pos)
        nk = sum((v, i) in kept for i in neg)
        neg_kept += nk
        for a, b in runs(sorted(pos)):
            scenes += 1
            scenes_kept += any((v, i) in kept for i in range(a, b + 1))
        per_video[v] = round(sum((v, i) in kept for i in range(1, n_frames[v] + 1)) / n_frames[v], 3)
    return {"kept_frames": len(kept), "kept_frac": len(kept) / total, "pos_kept": pos_kept, "n_pos": n_pos,
            "pos_recall": pos_kept / n_pos, "neg_pass_rate": neg_kept / n_neg, "scenes_kept": scenes_kept,
            "scenes": scenes, "kept_frac_by_video": per_video}


def sweep(score, labels, videos, jev, n_frames, neighbours):
    """Every distinct threshold, highest first -> list of operating points."""
    points = []
    for t in sorted(set(score.values()), reverse=True):
        kept = kept_set(score, t, videos, n_frames, neighbours)
        st = filter_stats(kept, labels, videos, n_frames)
        st.update(threshold=t, pipeline=pipeline(kept, labels, videos, jev, n_frames))
        points.append(st)
    return points


def first(points, cond):
    """Cheapest operating point (fewest kept frames) that satisfies cond."""
    ok = [p for p in points if cond(p)]
    return min(ok, key=lambda p: p["kept_frames"]) if ok else None


def at_budget(points, neg_rate):
    """Best operating point that lets at most `neg_rate` of the negative frames through."""
    ok = [p for p in points if p["neg_pass_rate"] <= neg_rate]
    return max(ok, key=lambda p: (p["pos_recall"], -p["kept_frames"])) if ok else None


def main():
    labels, videos, jev, emb, scores = load()
    n_frames = {v: max(i for (vv, i) in jev if vv == v) for v in videos}
    pos_keys = {(v, i) for v in videos for i in labels[v]["pos"]}
    amb_keys = {(v, i) for v in videos for i in labels[v]["ambiguous"]}
    two_keys = {(v, i) for v in videos for i in labels[v]["hard_neg"].get("two_hand_heart", [])}

    report = {"meta": emb["meta"], "timing": emb["timing"], "primary": list(PRIMARY),
              "jev_only": pipeline(None, labels, videos, jev, n_frames), "ranking": {}, "filter": {}}
    jv = report["jev_only"]
    jev_auc = auc([s for k, s in jev.items() if k in pos_keys],
                  [s for k, s in jev.items() if k not in pos_keys and k not in amb_keys])
    report["jev_only"]["auc"] = jev_auc
    print(f"Jev only ({JEV_Q}@{JEV_SIDE}, >= {JEV_THR}): AUC {jev_auc:.3f} | P {jv['precision']:.2f} R {jv['recall']:.2f} "
          f"| scenes {jv['events_detected']}/{jv['events']} | false segments {jv['false_segments']} | Jev calls 243/243\n")

    print("EmbeddingGemma 2 alone: ranking quality (AUC, finger heart vs everything else; ambiguous frames excluded)")
    for key in sorted(scores):
        sc = scores[key]
        p = [s for k, s in sc.items() if k in pos_keys]
        n = [s for k, s in sc.items() if k not in pos_keys and k not in amb_keys]
        two = [s for k, s in sc.items() if k in two_keys]
        other = [s for k, s in sc.items() if k not in pos_keys and k not in amb_keys and k not in two_keys]
        r = {"auc": auc(p, n), "auc_vs_two_hand": auc(p, two), "auc_vs_other_neg": auc(p, other),
             "mean_pos": sum(p) / len(p), "mean_two_hand": sum(two) / len(two), "mean_other_neg": sum(other) / len(other),
             "min_pos": min(p), "max_neg": max(n)}
        report["ranking"][f"{key[1]}@{key[0]}"] = r
        star = " *" if key == PRIMARY else ""
        print(f"  {key[1]:26s}@{key[0]:<5d} AUC {r['auc']:.3f} (vs two-hand heart {r['auc_vs_two_hand']:.3f}, "
              f"vs other {r['auc_vs_other_neg']:.3f}) | mean pos {r['mean_pos']:.3f} two-hand {r['mean_two_hand']:.3f} "
              f"other {r['mean_other_neg']:.3f}{star}")

    print("\nFilter -> Jev. 'neg pass' = share of non-finger-heart frames the filter lets through (what drives Jev cost on long videos)")
    for key in sorted(scores):
        for nb in (0, 1):
            pts = sweep(scores[key], labels, videos, jev, n_frames, nb)
            name = f"{key[1]}@{key[0]}+nb{nb}"
            ops = {
                "keep_all_pos": first(pts, lambda p: p["pos_recall"] == 1.0),
                "keep_95pct_pos": first(pts, lambda p: p["pos_recall"] >= 0.95),
                "keep_all_scenes": first(pts, lambda p: p["scenes_kept"] == p["scenes"]),
                "same_as_jev_only": first(pts, lambda p: p["pipeline"]["recall"] == jv["recall"]
                                          and p["pipeline"]["events_detected"] == jv["events_detected"]),
                **{f"neg_pass_le_{int(b * 100)}": at_budget(pts, b) for b in (0.1, 0.2, 0.3, 0.5)},
            }
            report["filter"][name] = ops
            if key[1] not in ("en_plain", "en_detail", "ko_plain", "en_detail_minus_two_hand", "ex_pos_max"):
                continue
            print(f"  {name}{' *' if key == PRIMARY else ''}")
            for label, p in ops.items():
                if p is None:
                    print(f"    {label:18s} -")
                    continue
                pl = p["pipeline"]
                prec = "-" if pl["precision"] is None else f"{pl['precision']:.2f}"
                print(f"    {label:18s} keep {p['kept_frac']:5.1%} of frames (neg pass {p['neg_pass_rate']:5.1%}) | "
                      f"finger hearts kept {p['pos_kept']}/{p['n_pos']}, scenes {p['scenes_kept']}/{p['scenes']} | "
                      f"then Jev: P {prec} R {pl['recall']:.2f}, scenes {pl['events_detected']}/{pl['events']}, "
                      f"false segs {pl['false_segments']}")

    # What gets through at the cheapest threshold that keeps every finger-heart frame (primary score, no neighbours).
    sc = scores[PRIMARY]
    t_all = min(s for k, s in sc.items() if k in pos_keys)
    kept = {k for k, s in sc.items() if s >= t_all}
    neg_all = [k for k in sc if k not in pos_keys and k not in amb_keys]
    other = [k for k in neg_all if k not in two_keys]
    detail = {"threshold": t_all,
              "two_hand_passed": sum(k in kept for k in two_keys), "two_hand_total": len(two_keys),
              "other_neg_passed": sum(k in kept for k in other), "other_neg_total": len(other),
              "kept_frac_by_video": {v: round(sum((v, i) in kept for i in range(1, n_frames[v] + 1)) / n_frames[v], 3)
                                     for v in videos}}
    report["primary_detail"] = detail
    print(f"\nPrimary score {PRIMARY[1]}@{PRIMARY[0]}, lowest finger-heart score = threshold {t_all:.4f}:")
    print(f"  two-hand heart frames passed {detail['two_hand_passed']}/{detail['two_hand_total']}, "
          f"other negatives passed {detail['other_neg_passed']}/{detail['other_neg_total']}")
    print(f"  share of each video sent to Jev: {detail['kept_frac_by_video']}")

    # Non-oracle check: pick the threshold from the other videos' finger hearts, apply it to the held-out video.
    print("\nLeave-one-video-out threshold (threshold = lowest finger-heart score in the OTHER videos):")
    report["leave_one_video_out"] = {}
    for key in (PRIMARY, (1120, "en_detail"), (280, "en_plain"), (280, "ex_pos_max")):
        sc = scores[key]
        rows, tot_pos, tot_kept_pos, tot_neg, tot_kept_neg = [], 0, 0, 0, 0
        for held in videos:
            train = [s for (v, i), s in sc.items() if v != held and (v, i) in pos_keys]
            t = min(train)
            pos = [s for (v, i), s in sc.items() if v == held and (v, i) in pos_keys]
            neg = [s for (v, i), s in sc.items() if v == held and (v, i) not in pos_keys and (v, i) not in amb_keys]
            kp, kn = sum(s >= t for s in pos), sum(s >= t for s in neg)
            rows.append({"held_out": held, "threshold": t, "pos_kept": kp, "n_pos": len(pos), "neg_kept": kn, "n_neg": len(neg)})
            tot_pos += len(pos); tot_kept_pos += kp; tot_neg += len(neg); tot_kept_neg += kn
        summary = {"pos_kept": tot_kept_pos, "n_pos": tot_pos, "neg_pass_rate": tot_kept_neg / tot_neg, "by_video": rows}
        report["leave_one_video_out"][f"{key[1]}@{key[0]}"] = summary
        per = ", ".join(f"{r['held_out'].split('_')[0]} {r['pos_kept']}/{r['n_pos']} pos, {r['neg_kept']}/{r['n_neg']} neg" for r in rows)
        print(f"  {key[1]}@{key[0]}: finger hearts kept {tot_kept_pos}/{tot_pos}, neg pass {tot_kept_neg / tot_neg:.1%} | {per}")

    out = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "filter_eval.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
