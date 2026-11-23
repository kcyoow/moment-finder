"""charades_dense 채점. 규칙은 결과를 보기 전에 정했다(README의 "채점"과 같다). 표준 라이브러리만 쓴다.

지표는 prompt_study/prompt_eval.py의 함수를 그대로 쓴다: AUC, AP, 0.5 기준의 잡은 수·정밀도·재현율, 기준(이번 실행의 sep4)의 재현율에서의 정밀도,
한 곳 찾기(질문마다 가장 높은 위치가 정답 안), 먼 헛짚음. 차이의 95% 구간은 영상째 다시 뽑기(2,000번, 씨앗 20261112). dev가 정하고 hold가 확인한다.

실행: python3 experiments/charades_dense/cd_eval.py [--try N] [--stored-as-sep4]
  --stored-as-sep4 : 받아 둔 기준 답(prompt_study h2o__ts.json의 base)을 sep4 자리에 넣어 채점 코드만 확인한다(prompt_eval과 같은 수가 나와야 한다)
결과: results/cd_eval.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(HERE), str(EXP / "prompt_study"), str(EXP / "open_model")]
import cd_layouts as L  # noqa: E402
from open_eval import auc  # noqa: E402
from prompt_eval import ap, at_cut, stats  # noqa: E402

SEED, BOOT = 20261112, 2000
SHORT = 2.0          # 짧은 동작: 정답 구간 길이 2초 이하
DIFFS = ("auc", "top1", "precision", "precision_at_base_recall")


def rnd(x, n=3):
    return None if x is None else round(x, n)


def load(name, try_n, base=None):
    p = (base or HERE / "results") / (f"{name}_try{try_n}.json" if try_n else f"{name}.json")
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- 점수 모으기: {(위치 id, 질문 id): 확률}
def position_scores(meta, try_n, stored_as_sep4=False, base=None):
    items = meta["items"]
    by_vt = {(it["video"], it["t"]): it["id"] for it in items}
    got, tokens, extra = {}, {}, {}
    if stored_as_sep4:
        rec = json.loads((EXP / "prompt_study" / "results" / "h2o__ts.json").read_text())["records"]
        got["sep4"] = {(r["id"], q): p for r in rec if not r["error"] for q, p in r["probs"]["base"].items()}
    for layout in L.LAYOUTS:
        if L.LAYOUTS[layout]["kind"] in L.Q3_KINDS or (stored_as_sep4 and layout == "sep4"):
            continue
        res = load(layout, try_n, base)
        if res is None:
            continue
        units = {u["id"]: u for u in L.units(items, layout)}
        good = [r for r in res["records"] if "error" not in r]
        tokens[layout] = {"requests": len(res["records"]), "answered": len(good), "partial": res["meta"].get("partial"),
                          "tokens_per_request": rnd(sum(r["tokens"] or 0 for r in good) / max(1, len(good)), 1),
                          "tokens_per_video_second": rnd(sum(r["tokens"] or 0 for r in good) / max(1, sum(len(units[r["unit"]]["items"]) for r in good)), 1),
                          "processed_per_video_second": rnd(sum(r["processed"] or 0 for r in good) / max(1, sum(len(units[r["unit"]]["items"]) for r in good)), 1),
                          "wall_seconds": res["meta"].get("wall_seconds")}
        if layout == "sheet32":
            row, win = {}, {}
            for r in good:
                u = units[r["unit"]]
                for qid, what, x, p in r["answers"]:
                    if what == "row":
                        row[(by_vt[(u["video"], x)], qid)] = p
                    else:
                        for it in u["items"]:
                            win[(it["id"], qid)] = p
                extra.setdefault("sheet32_windows", []).append(r)
            got["sheet32_row"], got["sheet32_win"] = row, win
        elif layout == "sheet60_jr":
            sc, shown = {}, []
            for r in good:
                u = units[r["unit"]]
                tt = L.tile_time(u)
                ps = {qid: p for qid, what, x, p in r["answers"] if what == "shown"}
                for qid, probs in r["choices"].items():
                    shown.append(ps[qid])
                    for tid, t in tt.items():
                        sc[(by_vt[(u["video"], t)], qid)] = ps[qid] * probs.get(tid, 0.0)
            got["sheet60_jr"] = sc
            extra["sheet60_jr_shown_mean"] = rnd(sum(shown) / max(1, len(shown)))
        else:
            got[layout] = {(by_vt[(units[r["unit"]]["video"], x)], qid): p for r in good for qid, what, x, p in r["answers"]}
    return got, tokens, extra


# ---------------------------------------------------------------- 채점 (prompt_eval.main의 방식을 위치 모양에 맞게)
def score_set(meta, got, which, base="sep4"):
    items = [it for it in meta["items"] if it["set"] == which]
    win = meta["windows"]
    pos_t = {}
    for it in items:
        for qid, y in it["gt"].items():
            if y:
                pos_t.setdefault((it["video"], qid), []).append(it["t"])
    keys = [(it["id"], qid) for it in items for qid in it["questions"]]
    byid = {it["id"]: it for it in items}

    def recs(name, vids=None):
        out = {} if vids else []
        for iid, qid in keys:
            if (iid, qid) not in got[name]:
                continue
            it = byid[iid]
            near = pos_t.get((it["video"], qid))
            r = ((it["video"], qid), it["t"], it["gt"][qid], got[name][(iid, qid)], min(abs(it["t"] - t) for t in near) if near else None)
            (out.setdefault(it["video"], []) if vids else out).append(r)
        return out
    if base not in got:
        return None
    b_recs = recs(base)
    if len(b_recs) != len(keys):
        print(f"[{which}] {base} 답이 {len(b_recs)}/{len(keys)}뿐이다")
    b = stats(b_recs)
    b_vid = recs(base, vids=True)
    videos = sorted(b_vid)
    res = {"pairs": len(keys), "positives": sum(r[2] for r in b_recs), "videos": len(videos), "layouts": {}}
    for name in got:
        rs = recs(name)
        if len(rs) != len(b_recs):
            res["layouts"][name] = {"missing": len(b_recs) - len(rs)}
            continue
        s = stats(rs, target=b["recall"])
        row = {k: (rnd(v) if isinstance(v, float) else v) for k, v in s.items()}
        vid = recs(name, vids=True)
        rng = random.Random(SEED)
        diffs, aucs = {k: [] for k in DIFFS}, []
        for _ in range(BOOT):
            pick = rng.choices(videos, k=len(videos))
            bb = stats([r for v in pick for r in b_vid[v]])
            mm = stats([r for v in pick for r in vid[v]], target=bb["recall"])
            aucs.append(mm["auc"])
            for k in DIFFS:
                x, y = (mm[k], bb[k]) if k != "top1" else (mm[k] / mm["questions"], bb[k] / bb["questions"])
                if x is not None and y is not None:
                    diffs[k].append(x - y)
        aucs.sort()
        row["auc_ci"] = [rnd(aucs[int(0.025 * BOOT)]), rnd(aucs[int(0.975 * BOOT) - 1])]
        if name != base:
            for k, vals in diffs.items():
                vals.sort()
                if s[k] is None or b[k] is None:
                    row[f"{k}_diff"] = None
                    continue
                point = (s[k] - b[k]) if k != "top1" else (s[k] - b[k]) / b["questions"]
                row[f"{k}_diff"] = [rnd(point), rnd(vals[int(0.025 * len(vals))]), rnd(vals[int(0.975 * len(vals)) - 1])] if vals and point is not None else None
        # 짧은 동작(정답 구간 2초 이하): 한 곳 찾기와 "찾음"(구간 안 위치 중 하나라도 0.5 이상)
        short_q = {(it["video"], q) for it in items for q in it["questions"] if all(e - s <= SHORT for s, e in win[q])}
        best, found = {}, {}
        for key, t, y, p, _ in sorted(rs, key=lambda r: (r[0], r[1])):
            if key in short_q:
                if key not in best or p > best[key][0]:
                    best[key] = (p, y)
                found[key] = found.get(key, False) or (y == 1 and p >= 0.5)
        row["short"] = {"questions": len(short_q), "top1": sum(y for _, y in best.values()), "found_at_0.5": sum(found.values())}
        res["layouts"][name] = row
    return res


def verdicts(dev, hold, base="sep4"):
    """미리 정한 판정. guard: AUC ≥ 기준 − 0.005이고 한 곳 찾기 ≥ 기준 − 2."""
    out = {}
    for name, r in dev["layouts"].items():
        if name in (base, "stored_base") or name.startswith("prev_") or "auc" not in r:
            continue
        b = dev["layouts"][base]
        guard = r["auc"] >= b["auc"] - 0.005 and r["top1"] >= b["top1"] - 2
        d = r.get("precision_at_base_recall_diff")
        if not guard:
            v = "지금 방법보다 나쁨"
        elif d and d[1] is not None and d[1] > 0:
            v = "지금 방법보다 나음(같은 재현율의 정밀도 구간이 0 위)"
        else:
            v = "지금 방법과 비슷함"
        h, hb = (hold or {}).get("layouts", {}).get(name), (hold or {}).get("layouts", {}).get(base)
        conf = None
        if h and hb and "auc" in h:
            hg = h["auc"] >= hb["auc"] - 0.005 and h["top1"] >= hb["top1"] - 2
            hd = h.get("precision_at_base_recall_diff")
            conf = ("확인됨" if hg and (v.startswith("지금 방법과") or (hd and hd[0] is not None and hd[0] > 0)) else "확인 안 됨") if v != "지금 방법보다 나쁨" else ("hold도 나쁨" if not hg else "hold에서는 guard 통과")
        out[name] = {"dev": v, "hold": conf, "reads_it": r["auc_ci"][0] is not None and r["auc_ci"][0] > 0.5}
    return out


# ---------------------------------------------------------------- sheet32 창 단위(1단계로 쓸 때)
def coarse(meta, extra, got, which):
    if "sheet32_windows" not in extra:
        return None
    items = {it["id"]: it for it in meta["items"] if it["set"] == which}
    units = {u["id"]: u for u in L.units(list(items.values()), "sheet32")}
    rows = []   # (창 id, 질문 id, 창 확률, 창 정답, 창 안 정답 위치 수, 영상)
    for r in extra["sheet32_windows"]:
        u = units.get(r["unit"])
        if u is None:
            continue
        for qid, what, x, p in r["answers"]:
            if what == "window":
                ys = [it["gt"][qid] for it in u["items"]]
                rows.append((u["id"], qid, p, int(any(ys)), sum(ys), u["video"], [it["id"] for it in u["items"]]))
    if not rows:
        return None
    out = {"windows_x_questions": len(rows), "positive_windows": sum(r[3] for r in rows), "window_auc": rnd(auc([(r[2], r[3]) for r in rows]))}
    if "sep4" in got:   # 참고: 같은 창의 sep4 최댓값(토큰 약 3.4배)
        out["window_auc_sep4_max"] = rnd(auc([(max(got["sep4"][(i, r[1])] for i in r[6]), r[3]) for r in rows]))
    # sep4가 맞힌 한 곳이 남는가
    best = {}
    for (iid, qid), p in got.get("sep4", {}).items():
        if iid in items and (qid not in best or p > best[qid][0]):
            best[qid] = (p, iid, items[iid]["gt"][qid])
    total_pos = sum(r[4] for r in rows)
    for cut in (0.3, 0.5, 0.7):
        kept = [r for r in rows if r[2] >= cut]
        kept_ids = {(r[1], i) for r in kept for i in r[6]}
        right = [(q, iid) for q, (p, iid, y) in best.items() if y]
        out[f"cut_{cut}"] = {"windows_kept": rnd(len(kept) / len(rows)), "positives_kept": rnd(sum(r[4] for r in kept) / max(1, total_pos)),
                             "sep4_top1_kept": rnd(sum((q, i) in kept_ids for q, i in right) / max(1, len(right)))}
    c = out["cut_0.5"]
    out["useful_first_stage"] = c["windows_kept"] <= 0.6 and c["positives_kept"] >= 0.9 and c["sep4_top1_kept"] >= 0.95
    return out


# ---------------------------------------------------------------- Q3
def q3(meta, try_n):
    segs = {s["id"]: s for s in meta["q3"]}
    names = {"a": "q3_plain", "b": "q3_path", "c": "q3_comp", "d": "q3_sep4"}
    got = {}
    for k, layout in names.items():
        res = load(layout, try_n)
        if res is None:
            continue
        units = {u["id"]: u for u in L.units(meta["q3"], layout)}
        got[k] = {(units[r["unit"]]["items"][0]["id"], qid): p for r in res["records"] if "error" not in r for qid, what, x, p in r["answers"]}
    if not got:
        return None
    keys = [(sid, q) for sid, s in segs.items() for q in s["questions"]]
    subsets = {"all": set(segs), "drawn_b": {i for i, s in segs.items() if s["paths_path"] > 0},
               "drawn_c": {i for i, s in segs.items() if s["paths_comp"] > 0},
               "drawn_both": {i for i, s in segs.items() if s["paths_path"] > 0 and s["paths_comp"] > 0}}
    out = {"segments": len(segs), "pairs": len(keys), "positives": sum(sum(s["gt"].values()) for s in segs.values()),
           "subset_sizes": {k: len(v) for k, v in subsets.items()}, "by_subset": {}}

    def rows(k, sub, label="gt"):
        return [(segs[s]["video"], got[k][(s, q)], segs[s][label][q]) for s, q in keys if s in sub and (s, q) in got[k]]

    def topk(k, sub):
        best = {}
        for s, q in keys:
            if s in sub and (s, q) in got[k]:
                key = (segs[s]["video"], q)
                p = got[k][(s, q)]
                if key not in best or p > best[key][0]:
                    best[key] = (p, segs[s]["gt"][q])
        return sum(y for _, y in best.values()), len(best)
    for sname, sub in subsets.items():
        tab = {}
        for k in got:
            rs = rows(k, sub)
            tab[k] = {"auc": rnd(auc([(p, y) for _, p, y in rs])), "ap": rnd(ap([(p, y) for _, p, y in rs])), "top1": topk(k, sub),
                      "auc_end_label": rnd(auc([(p, y) for _, p, y in rows(k, sub, "gt_end")]))}
        comps = {}
        for x, y in (("b", "a"), ("c", "a"), ("a", "d"), ("b", "d"), ("c", "d")):
            if x not in got or y not in got:
                continue
            vids = sorted({segs[s]["video"] for s in sub})
            rx, ry = rows(x, sub), rows(y, sub)
            bx = {v: [(p, t) for vv, p, t in rx if vv == v] for v in vids}
            by = {v: [(p, t) for vv, p, t in ry if vv == v] for v in vids}
            rng, d_auc, d_ap = random.Random(SEED), [], []
            for _ in range(BOOT):
                pick = rng.choices(vids, k=len(vids))
                px, py = [r for v in pick for r in bx[v]], [r for v in pick for r in by[v]]
                ax, ay = auc(px), auc(py)
                if ax is not None and ay is not None:
                    d_auc.append(ax - ay)
                apx, apy = ap(px), ap(py)
                if apx is not None and apy is not None:
                    d_ap.append(apx - apy)
            d_auc.sort()
            d_ap.sort()
            pa = tab[x]["auc"] - tab[y]["auc"] if tab[x]["auc"] is not None and tab[y]["auc"] is not None else None
            pp = tab[x]["ap"] - tab[y]["ap"] if tab[x]["ap"] is not None and tab[y]["ap"] is not None else None
            comps[f"{x}-{y}"] = {"auc": [rnd(pa), rnd(d_auc[int(0.025 * len(d_auc))]), rnd(d_auc[int(0.975 * len(d_auc)) - 1])] if d_auc else None,
                                 "ap": [rnd(pp), rnd(d_ap[int(0.025 * len(d_ap))]), rnd(d_ap[int(0.975 * len(d_ap)) - 1])] if d_ap else None}
        out["by_subset"][sname] = {"layouts": tab, "diffs": comps}
    # 미리 정한 판정: 경로가 도움(그려진 구간에서 x−a의 AUC 구간이 0 위), 4장을 대신함(x−d의 AUC가 −0.01 이상이고 구간 아래끝 −0.03 이상)
    v = {}
    for x, sub in (("b", "drawn_b"), ("c", "drawn_c")):
        d1 = out["by_subset"][sub]["diffs"].get(f"{x}-a")
        d2 = out["by_subset"]["all"]["diffs"].get(f"{x}-d")
        v[x] = {"helps_over_plain": bool(d1 and d1["auc"] and d1["auc"][1] is not None and d1["auc"][1] > 0),
                "stands_in_for_4": bool(d2 and d2["auc"] and d2["auc"][0] is not None and d2["auc"][0] >= -0.01 and d2["auc"][1] >= -0.03)}
    out["verdicts"] = v
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--try", dest="try_n", type=int, default=0)
    ap_.add_argument("--stored-as-sep4", action="store_true")
    ap_.add_argument("--dir", default="", help="results/<dir>/의 답을 채점한다(덧붙임 실행). 첫 실행의 sep4·grid8_small을 prev_로 함께 넣는다")
    args = ap_.parse_args()
    meta = json.loads((HERE / "data" / "items.json").read_text())
    if args.dir:
        return main_dir(meta, args.dir)
    got, tokens, extra = position_scores(meta, args.try_n, args.stored_as_sep4)
    out = {"tokens": tokens, "sets": {}, "coarse": {}}
    # 정상 확인: 이번 sep4 대 받아 둔 기준 답
    stored = {(r["id"], q): p for r in json.loads((EXP / "prompt_study" / "results" / "h2o__ts.json").read_text())["records"]
              if not r["error"] for q, p in r["probs"]["base"].items()}
    if "sep4" in got and not args.stored_as_sep4:
        got["stored_base"] = stored
        both = [k for k in got["sep4"] if k in stored]
        d = [abs(got["sep4"][k] - stored[k]) for k in both]
        out["sanity"] = {"pairs": len(both), "mean_abs_diff": rnd(sum(d) / max(1, len(d)), 4), "max_abs_diff": rnd(max(d) if d else None, 4),
                         "flips_at_0.5": sum((got["sep4"][k] >= 0.5) != (stored[k] >= 0.5) for k in both)}
    for which in ("dev", "hold"):
        res = score_set(meta, got, which)
        if res:
            out["sets"][which] = res
        c = coarse(meta, extra, got, which)
        if c:
            out["coarse"][which] = c
    if "dev" in out["sets"]:
        out["verdicts"] = verdicts(out["sets"]["dev"], out["sets"].get("hold"))
        if "sanity" in out and "stored_base" in out["sets"]["dev"]["layouts"]:
            ok = all(abs(out["sets"][w]["layouts"]["stored_base"]["auc"] - out["sets"][w]["layouts"]["sep4"]["auc"]) <= 0.01 and
                     abs(out["sets"][w]["layouts"]["stored_base"]["top1"] - out["sets"][w]["layouts"]["sep4"]["top1"]) <= 3 for w in out["sets"])
            out["sanity"]["pass"] = ok and out["sanity"]["mean_abs_diff"] <= 0.02
    out["sheet60_jr_shown_mean"] = extra.get("sheet60_jr_shown_mean")
    out["q3"] = q3(meta, args.try_n)
    (HERE / "results").mkdir(exist_ok=True)
    name = "cd_eval_stored_check.json" if args.stored_as_sep4 else f"cd_eval{'_try' + str(args.try_n) if args.try_n else ''}.json"
    (HERE / "results" / name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for which, res in out["sets"].items():
        print(f"\n[{which}] 위치×질문 {res['pairs']}, 정답 안 {res['positives']}, 영상 {res['videos']}")
        print(f"{'모양':16s}{'AUC':>7s}{'AUC 구간':>16s}{'AP':>7s}{'정밀':>7s}{'재현':>7s}{'같은재현 정밀':>10s}{'한곳':>9s}{'짧은 한곳/찾음':>14s}   같은재현 정밀 차이 [95%]")
        for n, r in res["layouts"].items():
            if "auc" not in r:
                print(f"{n:16s} 답 빠짐 {r}")
                continue
            print(f"{n:16s}{r['auc']:7.3f}{str(r['auc_ci']):>16s}{r['ap']:7.3f}{(r['precision'] or 0):7.3f}{(r['recall'] or 0):7.3f}{(r['precision_at_base_recall'] or 0):10.3f}"
                  f"{r['top1']:5d}/{r['questions']:<3d}{r['short']['top1']:6d}/{r['short']['found_at_0.5']}/{r['short']['questions']:<5d}  {r.get('precision_at_base_recall_diff') or ''}")
    for k in ("sanity", "verdicts", "coarse", "tokens", "sheet60_jr_shown_mean", "q3"):
        if out.get(k):
            print(f"\n{k}:", json.dumps(out[k], ensure_ascii=False)[:2500])


def main_dir(meta, sub):
    """덧붙임 실행(2026-11-14): 이 실행의 sep4가 기준. 정상 확인은 첫 실행의 sep4와, 판정 없는 비교는 첫 실행의 grid8_small과(실행이 다름)."""
    got, tokens, _ = position_scores(meta, 0, base=HERE / "results" / sub)
    prev, prev_tokens, _ = position_scores(meta, 0)
    got["prev_sep4"], got["prev_grid8_small"] = prev["sep4"], prev["grid8_small"]
    out = {"tokens": tokens, "prev_tokens": {k: prev_tokens[k] for k in ("sep4", "grid8_small")}, "sets": {}, "cross_run_vs_prev_grid8_small": {}}
    both = [k for k in got["sep4"] if k in got["prev_sep4"]]
    d = [abs(got["sep4"][k] - got["prev_sep4"][k]) for k in both]
    out["sanity"] = {"pairs": len(both), "mean_abs_diff": rnd(sum(d) / max(1, len(d)), 4), "max_abs_diff": rnd(max(d), 4),
                     "flips_at_0.5": sum((got["sep4"][k] >= 0.5) != (got["prev_sep4"][k] >= 0.5) for k in both)}
    for which in ("dev", "hold"):
        out["sets"][which] = score_set(meta, got, which)
        cross = score_set(meta, {k: got[k] for k in ("prev_grid8_small", "grid8_wide_small") if k in got}, which, base="prev_grid8_small")
        out["cross_run_vs_prev_grid8_small"][which] = cross["layouts"].get("grid8_wide_small")
    out["verdicts"] = verdicts(out["sets"]["dev"], out["sets"]["hold"])
    out["sanity"]["pass"] = out["sanity"]["mean_abs_diff"] <= 0.02 and all(
        abs(out["sets"][w]["layouts"]["prev_sep4"]["auc"] - out["sets"][w]["layouts"]["sep4"]["auc"]) <= 0.01 and
        abs(out["sets"][w]["layouts"]["prev_sep4"]["top1"] - out["sets"][w]["layouts"]["sep4"]["top1"]) <= 3 for w in out["sets"])
    (HERE / "results" / sub / f"cd_eval_{sub}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for which, res in out["sets"].items():
        print(f"\n[{which}]")
        for n, r in res["layouts"].items():
            print(f"{n:18s} AUC {r['auc']:.3f} {r['auc_ci']} AP {r['ap']:.3f} 한곳 {r['top1']}/{r['questions']} 같은재현 정밀 {r['precision_at_base_recall']:.3f} "
                  f"AUC차 {r.get('auc_diff')} 정밀차 {r.get('precision_at_base_recall_diff')} 한곳차 {r.get('top1_diff')}")
        c = out["cross_run_vs_prev_grid8_small"][which]
        print("  실행이 다른 비교 grid8_wide_small − 첫 실행 grid8_small:", {k: c.get(k) for k in ("auc_diff", "top1_diff", "precision_at_base_recall_diff")})
    for k in ("sanity", "verdicts", "tokens"):
        print(k, json.dumps(out[k], ensure_ascii=False))


if __name__ == "__main__":
    main()
