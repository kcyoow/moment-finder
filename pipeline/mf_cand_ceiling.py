"""후보를 목록 여럿(합친 점수, 질문 하나하나, 막연한 질문)에서 돌아가며 뽑으면 완벽한 확인자 기준으로 얼마나 나아지는지 본다 (돈 없음). pipeline/README.md 19절.

쓰는 것: pipeline/eval/cand_ceiling.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_cand_ceiling.py
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf  # noqa: E402
import mf_combine2_check as K  # noqa: E402
import mf_core as C  # noqa: E402
import mf_verify as M  # noqa: E402
import mf_views_check as V  # noqa: E402

BUDGET = C.DEFAULTS["budget"]


round_robin = C.round_robin          # 2026-11-16: 웹사이트가 쓰게 되어 mf_core.py로 옮겼다(같은 함수)


def perfect(cands, has, n_in, dur):
    kept, used = [], 0.0
    for c in cands:
        if has(c) and used < BUDGET * dur:
            kept.append(c)
            used += c["length"]
    return {"events": n_in(kept), "kept": len(kept), "looked": len(cands), "with_event": sum(map(has, cands)), "seconds": round(used, 1)}


def main():
    recs, byv, sets, levels = V.load()
    out = {"label": "후보를 목록 여럿에서 돌아가며 뽑았을 때의 상한 (pipeline/README.md 19절)", "arms": {}}
    for arm, keys in M.MC_KEYS.items():
        res = {"combined_only": {}, "round_robin": {}}
        for v, rs in byv.items():
            t = np.array([r["t"] for r in rs])
            t0, t1, dur = list(t - 0.5), list(t + 0.5), float(t.max() + 0.5)
            big = np.array([bool(r["big"]) for r in rs])
            has = lambda c: bool((big & (t >= c["start"]) & (t < c["end"])).any())
            n_in = lambda cs: int(sum((big & (t >= c["start"]) & (t < c["end"])).sum() for c in cs)) if cs else 0
            vr = C.pct_rank(np.array([r["w"] for r in rs]))
            ser = [V.series({v: rs}, sets, levels, arm, k)[v] for k in keys]
            final = C.pct_rank(0.5 * vr + 0.5 * C.pct_rank(np.mean(ser, axis=0)))
            clips = lambda x: C.make_clips(x, t0, t1, dur, budget=0.9)
            lists = [clips(final)] + [clips(C.pct_rank(x)) for x in ser] + [clips(vr)]
            for n in (5, 6, 8, 10):
                for name, cands in (("combined_only", lists[0][:n]), ("round_robin", round_robin(lists, n))):
                    r = perfect(cands, has, n_in, dur)
                    agg = res[name].setdefault(str(n), {"events": 0, "kept": 0, "looked": 0, "with_event": 0, "seconds": 0.0})
                    for k_, x in r.items():
                        agg[k_] = round(agg[k_] + x, 1)
        out["arms"][arm] = res
    # 13-5절의 한 편
    run = mf.RUNS / "web_1116_043326_3"
    pos, qj, info = mf.read(run / "positions.json"), mf.read(run / "questions.json"), mf.read(run / "run.json")
    recs_ = {r["k"]: r.get("answers") for r in mf.read(run / "results.json")["records"] if "error" not in r}
    answers = [recs_.get(q["k"]) for q in pos]
    lv = mf.levels_of(qj["questions"])
    keys = [k for k in qj["questions"] if k != "worth"]
    t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
    comb = C.combine(answers, keys, "worth", levels=lv)
    clips = lambda x: C.make_clips(x, t0, t1, info["duration"], budget=0.9)
    ser = [np.nan_to_num(np.array([K.unit((a or {}).get(k), lv, k) for a in answers])) for k in keys]
    vr = C.pct_rank(np.nan_to_num(np.array([K.unit((a or {}).get("worth")) for a in answers])))
    lists = [clips(comb["final"])] + [clips(C.pct_rank(x)) for x in ser] + [clips(vr)]
    names = ["합친 점수"] + keys + ["막연한 질문"]
    fight = [79.5, 82.5, 84.5, 86.5, 87.5, 91.5, 94.5, 97.5, 139.5, 141.5, 142.5]
    one = {}
    for n in (5, 6, 8):
        cands = round_robin(lists, n)
        one[str(n)] = [{"start": c["start"], "end": c["end"], "from": names[c["source"]], "fight_events": sum(c["start"] <= x < c["end"] for x in fight)} for c in cands]
    out["mc12_candidates"] = one
    with open(HERE / "eval" / "cand_ceiling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for arm, res in out["arms"].items():
        for name in ("combined_only", "round_robin"):
            print(f"{arm} {name:14s} " + " | ".join(f"후보 {n}: 사건 {r['events']:.0f} (맞는 후보 {r['with_event']:.0f}/{r['looked']:.0f}, 남긴 클립 {r['kept']:.0f}, {r['seconds']:.0f}초)" for n, r in res[name].items()))
    for n, cs in one.items():
        print(f"mc_12 후보 {n}개:", [(c["start"], c["end"], c["from"], c["fight_events"]) for c in cs])


if __name__ == "__main__":
    main()
