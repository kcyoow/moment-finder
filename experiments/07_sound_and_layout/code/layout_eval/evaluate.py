"""모양 A·B·C·D·E를 마인크래프트(큰 사건 56곳)·로봇(정답 구간)의 정답과 견준다(pipeline/README.md 25-7절). 돈은 들지 않는다.

A는 pipeline/runs의 받아 둔 답, B~E는 results/h2o_out/lay_<모양>_<mc|robot>.json(Modal에서 받은 것)이다.
V4·V9는 판정 모델을 Valen-Decision-1.5 4B·9B로 바꾼 것(A와 같은 요청, 마인크래프트만. valen_modal.py)이다.
모양마다 pipeline/runs와 같은 꼴의 폴더(results/runs_<모양>/)를 만들고 pipeline/mf_gem_check.py의 계산을 그대로 쓴다.
마인크래프트는 같은 위치에서 모양끼리 짝지어 AUC 차이의 부트스트랩 구간도 낸다(mf_gem_check와 같은 HE.boot).
쓰는 것: results/layout_eval.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/layout_eval/evaluate.py
"""
import json, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parents[1] / "pipeline"
sys.path.insert(0, str(PIPE))
import mf_core as C  # noqa: E402
import mf_gem_check as G  # noqa: E402

RES = ROOT / "results"
index = json.load(open(ROOT / "specs/index.json"))
ARMS = ["A"] + [a for a in ("B", "C", "D", "E", "V4", "V9") if (RES / f"h2o_out/lay_{a}_mc.json").exists()]   # V4·V9: Valen-Decision-1.5 4B·9B(마인크래프트만)
has = lambda a, s: a == "A" or (RES / f"h2o_out/lay_{a}_{s}.json").exists()


def arm_runs(a):
    """모양 a의 답을 pipeline/runs와 같은 꼴로 둔 폴더."""
    if a == "A":
        return PIPE / "runs"
    d = RES / f"runs_{a}"
    for s, ix in index.items():
        if not has(a, s):
            continue
        j = json.load(open(RES / f"h2o_out/lay_{a}_{s}.json"))
        by = {}
        for r in j["records"]:
            if r:
                by.setdefault(r["k"] // 10000, []).append({**r, "k": r["k"] % 10000})
        for vi, nm in enumerate(ix["names"]):
            o = d / nm; o.mkdir(parents=True, exist_ok=True)
            for f in ("positions.json", "questions.json", "run.json"):
                shutil.copy(PIPE / "runs" / nm / f, o / f)
            recs = sorted(by.get(vi, []), key=lambda r: r["k"])
            good = [r for r in recs if "error" not in r]
            meta = {**j["meta"], "name": nm, "requests": len(recs), "answered": len(good), "errors": len(recs) - len(good),
                    "rate_per_second": round(j["meta"]["requests"] / max(j["meta"]["wall_seconds"], 1e-9), 2)}
            json.dump({"meta": meta, "records": recs}, open(o / "results.json", "w"), ensure_ascii=False)
    for old in ("robot_green", "robot_red"):            # mf_gem_check.robot()이 11절의 조정자 답도 읽는다
        lk = d / old
        if not lk.exists():
            lk.symlink_to(PIPE / "runs" / old)
    return d


out = {"arms": ARMS, "robot": {}, "minecraft": {}}
# ---------------------------------------------------------------- 로봇: 모양마다 mf_gem_check.robot() 그대로
for a in [x for x in ARMS if has(x, "robot")]:
    G.RUNS = arm_runs(a)
    rb = G.robot()
    out["robot"][a] = {run: {s: {"auc_final": v["auc_final"], "auc_criteria_only": v["auc_criteria_only"], "events_in_clips": f"{v['events_in_clips']}/{len(v['events'])}",
                                  "ranks": [e["clip_ranks"] for e in v["events"]], "clips": v["clips"]}
                             for s, v in r["sets"].items() if s != "coordinator"} | {"vague_only_auc": r["vague_only_auc"]["this_run"], "errors": r["errors"]}
                       for run, r in rb.items()}

# ---------------------------------------------------------------- 마인크래프트: mf_gem_check.minecraft()의 앞부분을 모양 여럿에 펼친다
CC, HE, HP = G.CC, G.HE, G.HP
items = HP.load_items()["full"]
want = {it["id"]: it for it in items}
order = {i: k for k, i in enumerate(want)}
recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
        for r in sorted((r for r in HP.read(CC.HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
HE.finish(recs, "full", items)
byv = {}
for r in recs:
    byv.setdefault(r["video"], []).append(r)
cols_all = set()
for a in ARMS:
    G.RUNS = arm_runs(a)
    for v, rs in byv.items():
        name = "gem_" + "_".join(v.split("_")[:2])
        pos, qj, answers, levels, meta = G.load_run(name)
        by_t = {q["center"]: x for q, x in zip(pos, answers)}
        ans = [by_t.get(r["t"]) for r in rs]
        cols = {"vague": C.pct_rank(np.array([C.answer_value((x or {}).get("worth")) for x in ans]))}
        for sname, keys in G.sets_of(qj).items():
            cols[sname] = C.pct_rank(C.combine(ans, keys, None, levels=levels)["final"])
            cols[f"{sname}+vague"] = C.pct_rank(C.combine(ans, keys, "worth", levels=levels)["final"])
        for key in qj["questions"]:
            if key != "worth":
                top = (levels or {}).get(key) or 2
                cols[f"q_{key}"] = np.array([C.answer_value((x or {}).get(key)) for x in ans], float) / (top - 1)
        for i, r in enumerate(rs):
            for k, c in cols.items():
                r[f"{a}:{k}"] = c[i]; cols_all.add(k)


def auc(key, label="big"):
    return lambda xs: (lambda ys: HE.auc([x[key] for x in ys], [x[label] for x in ys]))([x for x in xs if x[key] == x[key]])


mc = {"big_events": int(sum(r["big"] for r in recs)), "positions": len(recs), "arms": {}, "vs_A": {}}
main_cols = ["ma", "mb", "ma+vague", "mb+vague", "vague"]
for a in ARMS:
    mc["arms"][a] = {k: {"auc_big": G.r4(auc(f"{a}:{k}")(recs)), "ci": HE.boot(recs, auc(f"{a}:{k}"))[:2],
                         "auc_fight": G.r4(auc(f"{a}:{k}", "fight")(recs)), "auc_progress": G.r4(auc(f"{a}:{k}", "prog")(recs))} for k in main_cols}
    mc["arms"][a]["per_question"] = {k[2:]: {"auc_big": G.r4(auc(f"{a}:{k}")(recs)),
                                             "seconds_at_or_above_half": int(sum(r[f"{a}:{k}"] >= 0.5 for r in recs if r[f"{a}:{k}"] == r[f"{a}:{k}"]))}
                                     for k in sorted(cols_all) if k.startswith("q_")}
    if a != "A":
        mc["vs_A"][a] = {}
        for k in main_cols:
            f = lambda xs, k=k: (lambda p, q: None if p is None or q is None else p - q)(auc(f"{a}:{k}")(xs), auc(f"A:{k}")(xs))
            lo, hi, _ = HE.boot(recs, f)
            mc["vs_A"][a][k] = {"point": G.r4(mc["arms"][a][k]["auc_big"] - mc["arms"]["A"][k]["auc_big"]), "ci": [lo, hi]}
out["minecraft"] = mc
json.dump(out, open(RES / "layout_eval.json", "w"), ensure_ascii=False, indent=1)

print("모양", ARMS, "| 마인크래프트 큰 사건", mc["big_events"], "위치", mc["positions"])
for k in main_cols:
    print(f"  [{k:9s}] " + "  ".join(f"{a} {mc['arms'][a][k]['auc_big']:.3f}" for a in ARMS)
          + " | A와의 차이 " + "  ".join(f"{a} {mc['vs_A'][a][k]['point']:+.3f} [{mc['vs_A'][a][k]['ci'][0]:+.3f},{mc['vs_A'][a][k]['ci'][1]:+.3f}]" for a in ARMS if a != "A"))
for k in ("ma", "mb"):
    print(f"  싸움·피해 / 제작·다이아 [{k}] " + "  ".join(f"{a} {mc['arms'][a][k]['auc_fight']:.3f}/{mc['arms'][a][k]['auc_progress']:.3f}" for a in ARMS))
print("  질문별 AUC(예 초)")
for q in mc["arms"]["A"]["per_question"]:
    print(f"    {q:28s} " + "  ".join(f"{a} {mc['arms'][a]['per_question'][q]['auc_big']:.3f}({mc['arms'][a]['per_question'][q]['seconds_at_or_above_half']})" for a in ARMS))
for run in ("robot_green", "robot_red"):
    print(f"  로봇 {run}")
    for s in ("r0", "ra", "rb"):
        print(f"    [{s}] " + "  ".join(f"{a} AUC {out['robot'][a][run][s]['auc_final']:.3f} 정답 {out['robot'][a][run][s]['events_in_clips']} 순위 {out['robot'][a][run][s]['ranks']}" for a in out["robot"]))
