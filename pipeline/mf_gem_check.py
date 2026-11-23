"""중간 LLM(Gemini 4 Hydrogen)이 쓴 질문으로 받은 답을 견준다 (이 Mac). 방법은 pipeline/README.md 12-3절, 12-4b절.

로봇: 벌(R0, Ra, Rb)마다 11-3절과 같은 것(AUC, 정답 구간 다섯을 담은 클립과 순위)을 내고 11절(조정자의 질문)과 나란히 놓는다.
마인크래프트: 벌(Ma, Mb)마다 큰 사건 56곳의 AUC(기준만, 막연한 질문과 합친 것)와 사건 종류별 AUC를 내고, 받아 둔 손 기준·이전 글쓴이·막연한 질문과 나란히 놓는다.
쓰는 것: pipeline/eval/gem_check.json

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_gem_check.py
"""
import sys

sys.dont_write_bytecode = True
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf_combine_check as CC  # noqa: E402  (받아 둔 마인크래프트 답 읽기와 지표. ha_eval, hh_eval을 불러온다)
import mf_core as C  # noqa: E402
import mf_robot_check as R  # noqa: E402

HE, HP, E = CC.HE, CC.HP, CC.E
RUNS = HERE / "runs"
r4 = E.r4


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_run(name):
    d = RUNS / name
    pos, qj, res = read(d / "positions.json"), read(d / "questions.json"), read(d / "results.json")
    recs = {r["k"]: r.get("answers") for r in res["records"] if "error" not in r}
    answers = [recs.get(q["k"]) for q in pos]
    levels = {k: len(q["criteria"]) for k, q in qj["questions"].items() if q.get("type") == "score"}
    return pos, qj, answers, levels, res["meta"]


def sets_of(qj):
    out = {}
    for k in qj["criteria_keys"]:
        out.setdefault(k.split("_")[0], []).append(k)
    return out


# =====================================================================  로봇
def robot():
    trials = {}
    with open(R.SRC / "clip_list.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = "_".join(row["clip"].split("/")[-1].split("_")[:2])
            if row["source"] in R.RUN_OF:
                trials[name] = {"run": R.RUN_OF[row["source"]], "start": float(row["start"])}
    out = {}
    for old, new in (("robot_green", "gem_robot_green"), ("robot_red", "gem_robot_red")):
        pos, qj, answers, levels, meta = load_run(new)
        t0, t1 = np.array([q["t0"] for q in pos]), np.array([q["t1"] for q in pos])
        center = 0.5 * (t0 + t1)
        duration = read(RUNS / new / "run.json")["duration"]
        events = [{"trial": k, "result": lab, "start": round(trials[k]["start"] + a, 1), "end": round(trials[k]["start"] + b, 1)}
                  for k, (a, b, lab) in R.HELD.items() if trials[k]["run"] == old]
        y = np.zeros(len(center), bool)
        for e in events:
            y |= (center >= e["start"]) & (center <= e["end"])
        keep = np.ones(len(center), bool)
        if old in R.SKIP:
            keep &= ~((center >= R.SKIP[old][0]) & (center <= R.SKIP[old][1]))
        res = {"positions": len(pos), "answered": meta["answered"], "errors": meta["errors"], "rate_per_second": meta["rate_per_second"], "sets": {}}

        def one(answers_, keys, levels_, label):
            comb = C.combine(answers_, keys, "worth", levels=levels_)
            crit_only = C.combine(answers_, keys, None, levels=levels_)["final"]
            clips = C.make_clips(comb["final"], t0, t1, duration)
            ev = []
            for e in events:
                hit = [c for c in clips if min(c["end"], e["end"]) - max(c["start"], e["start"]) >= 1.0]
                ev.append({"trial": e["trial"], "result": e["result"], "clip_ranks": [c["rank"] for c in hit], "clips": [(c["start"], c["end"]) for c in hit]})
            return {"label": label, "questions": len(keys), "auc_final": r4(R.auc(comb["final"][keep], y[keep])), "auc_criteria_only": r4(R.auc(crit_only[keep], y[keep])),
                    "auc_per_question": {k: r4(R.auc(comb["raw"][k][keep], y[keep])) for k in keys},
                    "clips": len(clips), "clips_seconds": round(sum(c["length"] for c in clips), 1), "events_in_clips": sum(bool(e["clip_ranks"]) for e in ev),
                    "events": ev, "clip_list": [{"rank": c["rank"], "start": c["start"], "end": c["end"], "peak": c["peak"]} for c in clips]}, comb
        for sname, keys in sets_of(qj).items():
            res["sets"][sname], comb_new = one(answers, keys, levels, f"Gemini {sname}")
        # 11절: 조정자의 질문(같은 식으로 다시 낸다)
        pos_o, qj_o, answers_o, levels_o, _ = load_run(old)
        res["sets"]["coordinator"], comb_old = one(answers_o, qj_o["criteria_keys"], levels_o, "조정자 (11절, 기록을 본 뒤에 씀)")
        w_new, w_old = comb_new["raw"]["worth"], comb_old["raw"]["worth"]
        res["vague_only_auc"] = {"this_run": r4(R.auc(w_new[keep], y[keep])), "section11": r4(R.auc(w_old[keep], y[keep]))}
        res["vague_repeat"] = {"pearson": r4(float(np.corrcoef(w_new, w_old)[0, 1])), "mean_abs_diff": r4(float(np.abs(w_new - w_old).mean())),
                               "crossed_0.5": int(((w_new >= 0.5) != (w_old >= 0.5)).sum())}
        out[old] = res
    return out


# =====================================================================  마인크래프트
def minecraft():
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted((r for r in HP.read(CC.HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    stored = {"hand": {r["id"]: {"action": r["action"], "progress": r["progress"]} for r in recs},
              "P12": {r["id"]: r["answers"] for r in read(CC.HP_ / "results" / "main_minecraft__mc_plain.json")["records"] if "error" not in r}}
    arms, info = [], {"runs": 0, "errors": 0, "answered": 0, "rates": []}
    for v, rs in byv.items():
        name = "gem_" + "_".join(v.split("_")[:2])
        pos, qj, answers, levels, meta = load_run(name)
        info["runs"] += 1
        info["errors"] += meta["errors"]
        info["answered"] += meta["answered"]
        info["rates"].append(meta["rate_per_second"])
        by_t = {q["center"]: a for q, a in zip(pos, answers)}          # 영상 길이가 300초에 조금 못 미치면 마지막 위치가 없다
        answers = [by_t.get(r["t"]) for r in rs]
        info["missing"] = info.get("missing", 0) + sum(a is None for a in answers)
        cols = {}
        w_new = np.array([C.answer_value((a or {}).get("worth")) for a in answers])
        w_old = np.array([r["worth"] if not isinstance(r["worth"], dict) else r["worth"]["noul"] for r in rs], float)
        cols["vague_new"], cols["vague_stored"] = C.pct_rank(w_new), C.pct_rank(w_old)
        for sname, keys in sets_of(qj).items():
            cols[sname] = C.pct_rank(C.combine(answers, keys, None, levels=levels)["final"])
            cols[f"{sname}+vague"] = C.pct_rank(C.combine(answers, keys, "worth", levels=levels)["final"])
        for sname, by in stored.items():
            ans = [by.get(r["id"]) for r in rs]
            keys = sorted({q for a in ans if a for q in a})
            s = CC.scores(ans, keys)["value_mean"]
            cols[sname] = C.pct_rank(s)
            cols[f"{sname}+vague"] = C.pct_rank(0.5 * C.pct_rank(w_old) + 0.5 * C.pct_rank(s))
        arms = list(cols)
        for key in qj["questions"]:                                   # 질문 하나씩의 값 (결과를 본 뒤에 더한 풀이. README 12-7절)
            if key != "worth":
                top = (levels or {}).get(key) or 2
                cols[f"q_{key}"] = np.array([C.answer_value((a or {}).get(key)) for a in answers], float) / (top - 1)
        for i, r in enumerate(rs):
            for k, c in cols.items():
                r[f"s_{k}"] = c[i]
            r["w_new"], r["w_old"] = w_new[i], w_old[i]

    def auc(key, label="big"):
        return lambda xs: HE.auc([x[key] for x in xs], [x[label] for x in xs])

    def diff(a, b):
        return lambda xs: (lambda p, q: None if p is None or q is None else p - q)(auc(f"s_{a}")(xs), auc(f"s_{b}")(xs))
    out = {"big_events": int(sum(r["big"] for r in recs)), "fight_events": int(sum(r["fight"] for r in recs)), "progress_events": int(sum(r["prog"] for r in recs)),
           "run_info": {"runs": info["runs"], "answered": info["answered"], "errors": info["errors"], "positions_without_answer": info.get("missing", 0), "rate_per_second_mean": r4(float(np.mean(info["rates"])))},
           "arms": {}, "differences": {}}
    for a in arms:
        out["arms"][a] = {"auc_big": r4(auc(f"s_{a}")(recs)), "ci": HE.boot(recs, auc(f"s_{a}"))[:2],
                          "auc_fight": r4(auc(f"s_{a}", "fight")(recs)), "auc_progress": r4(auc(f"s_{a}", "prog")(recs))}
    for a, b in (("ma", "hand"), ("mb", "hand"), ("ma", "P12"), ("ma+vague", "hand+vague"), ("ma+vague", "vague_stored"), ("mb+vague", "vague_stored"), ("ma", "mb")):
        lo, hi, _ = HE.boot(recs, diff(a, b))
        out["differences"][f"{a}_minus_{b}"] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi]}
    qkeys = sorted(k[4:] for k in recs[0] if k.startswith("s_q_"))
    ok_ = lambda k: [r for r in recs if r[k] == r[k]]                # 답이 없는 한 곳(NaN)은 뺀다
    out["per_question"] = {q: {"auc_big": r4(auc(f"s_q_{q}")(ok_(f"s_q_{q}"))), "auc_fight": r4(auc(f"s_q_{q}", "fight")(ok_(f"s_q_{q}"))),
                               "auc_progress": r4(auc(f"s_q_{q}", "prog")(ok_(f"s_q_{q}"))),
                               "mean": r4(float(np.mean([r[f"s_q_{q}"] for r in ok_(f"s_q_{q}")]))),
                               "seconds_at_or_above_half": int(sum(r[f"s_q_{q}"] >= 0.5 for r in ok_(f"s_q_{q}")))} for q in qkeys}
    kinds = {}
    for r in recs:
        if r["big"]:
            kinds.setdefault(str(r.get("kind_label") or r.get("event") or ("fight" if r["fight"] else "progress")), []).append(r)
    out["rank_of_events"] = {k: {"n": len(v), **{a: r4(float(np.mean([x[f"s_{a}"] for x in v]))) for a in ("ma", "mb", "hand", "P12", "vague_new")}} for k, v in kinds.items()}
    keep = [r for r in recs if r["w_new"] == r["w_new"]]
    wn, wo = np.array([r["w_new"] for r in keep]), np.array([r["w_old"] for r in keep])
    out["vague_repeat"] = {"pearson": r4(float(np.corrcoef(wn, wo)[0, 1])), "mean_abs_diff": r4(float(np.abs(wn - wo).mean())), "crossed_0.5": int(((wn >= 0.5) != (wo >= 0.5)).sum())}
    return out


def main():
    out = {"label": "중간 LLM(Gemini 4 Hydrogen)이 요청만 보고 쓴 질문의 결과 (pipeline/README.md 12절)", "robot": robot(), "minecraft": minecraft()}
    with open(HERE / "eval" / "gem_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for run, r in out["robot"].items():
        print(f"== {run}: 위치 {r['positions']}, 답 {r['answered']}, 오류 {r['errors']}, 초당 {r['rate_per_second']} | 막연한 질문만 AUC {r['vague_only_auc']} | 다시 물은 막연한 질문 {r['vague_repeat']}")
        for sname, s in r["sets"].items():
            print(f"  [{sname}] {s['label']}: 질문 {s['questions']} | AUC 합친 것 {s['auc_final']}, 기준만 {s['auc_criteria_only']} | 클립 {s['clips']}개 {s['clips_seconds']}초 | 정답 구간 {s['events_in_clips']}/{len(s['events'])}"
                  f" 순위 {[(e['trial'], e['clip_ranks']) for e in s['events']]}")
            print(f"       질문별 {s['auc_per_question']}")
    m = out["minecraft"]
    print(f"== 마인크래프트: 큰 사건 {m['big_events']} (싸움·피해 {m['fight_events']}, 제작·다이아 {m['progress_events']}) {m['run_info']} | 다시 물은 막연한 질문 {m['vague_repeat']}")
    for a, v in m["arms"].items():
        print(f"  {a:14s} 큰 사건 AUC {v['auc_big']} {v['ci']} | 싸움·피해 {v['auc_fight']} | 제작·다이아 {v['auc_progress']}")
    for k, d in m["differences"].items():
        print(f"  {k}: {d['point']:+.4f} {d['ci']}")


if __name__ == "__main__":
    main()
