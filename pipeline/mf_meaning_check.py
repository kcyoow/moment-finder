"""무엇을 하이라이트로 치는가: 표시를 종류로 쪼개, 정하는 쪽마다 어느 종류를 높이 치는지 본다 (돈 없음). pipeline/README.md 21절.

쓰는 것: pipeline/eval/meaning_check.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_meaning_check.py
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf  # noqa: E402
import mf_core as C  # noqa: E402
import mf_verify as M  # noqa: E402
import mf_views_check as V  # noqa: E402

HE = V.HE
RAW = HERE.parent / "experiments" / "frame_limit" / "data" / "mc_events.json"
KINDS = ["diamond", "hostile", "animal", "hurt", "craft"]            # 한 위치에 여럿이면 앞선 것 하나
KO = {"diamond": "다이아몬드", "hostile": "적과 싸움", "animal": "동물 사냥", "hurt": "피해를 입음", "craft": "제작 2단계 이상", "none": "큰 사건 없음"}
OWNER = ("diamond", "hostile", "hurt")                               # 소유자의 말에 가장 가까운 것 (21-3절 라)
ANIMAL_WINDOW = 10.0


def kind_of(evs, animal_keys):
    has = lambda k, lv=None: any(e["kind"] == k and (lv is None or e.get("level", 0) >= lv) for e in evs)
    fights = [e for e in evs if e["kind"] == "fight"]
    if has("find"):
        return "diamond"
    if any((e["video"], e["t"]) not in animal_keys for e in fights):
        return "hostile"
    if fights:
        return "animal"
    if has("hurt"):
        return "hurt"
    if has("crafting", 2):
        return "craft"
    return "none"


def pooled_auc(byv, per_video, pos, neg):
    s, y = [], []
    for v, x in per_video.items():
        r = C.pct_rank(np.asarray(x, float))
        for i, rec in enumerate(byv[v]):
            if pos(rec):
                s.append(r[i]); y.append(True)
            elif neg(rec):
                s.append(r[i]); y.append(False)
    return (round(float(HE.auc(s, y)), 3) if any(y) else None), int(sum(y))


def main():
    recs, byv, sets, levels = V.load()
    raw = json.loads(RAW.read_text())["events"]
    kills = [e for e in raw if e["kind"] == "kill"]
    animal_keys = {(e["video"], e["t"]) for e in raw if e["kind"] == "fight"
                   and any(k["video"] == e["video"] and 0 <= k["t"] - e["t"] <= ANIMAL_WINDOW for k in kills)}
    for r in recs:
        r["k"] = kind_of(r["events"], animal_keys)
        assert (r["k"] != "none") == bool(r["big"]), (r["id"], r["k"], r["big"])
    out = {"label": "무엇을 하이라이트로 치는가 (pipeline/README.md 21절)", "kinds": {}, "raw_events": {}}
    out["raw_events"] = {"damage_dealt": sum(e["kind"] == "fight" for e in raw), "damage_dealt_on_animals": len(animal_keys),
                         "kills": {a: sum(k["item"] == a for k in kills) for a in sorted({k["item"] for k in kills})}}
    for k in KINDS + ["none"]:
        rs = [r for r in recs if r["k"] == k]
        out["kinds"][k] = {"name": KO[k], "positions": len(rs), "videos": sorted({r["video"][:5] for r in rs}) if k != "none" else None}
    print("종류별 위치:", {KO[k]: (v["positions"], len(v["videos"] or [])) for k, v in out["kinds"].items()}, "| 큰 사건 합", sum(bool(r["big"]) for r in recs))
    print("원자료:", out["raw_events"])

    # ---------- (나) 정하는 쪽마다 종류별 AUC
    scores = {"막연한 질문": {v: [r["w"] for r in rs] for v, rs in byv.items()},
              "손 기준: 싸움·위험": {v: [r["action"]["score"] for r in rs] for v, rs in byv.items()},
              "손 기준: 진행": {v: [r["progress"]["score"] for r in rs] for v, rs in byv.items()},
              "손 기준: 합친 것": {v: [r["final"] for r in rs] for v, rs in byv.items()}}
    half = lambda crit: {v: C.pct_rank(0.5 * C.pct_rank(np.array([r["w"] for r in rs])) + 0.5 * C.pct_rank(np.asarray(crit[v], float))) for v, rs in byv.items()}
    scores["손 기준: 막연한 질문과 반반"] = half(scores["손 기준: 합친 것"])
    for arm, keys in M.MC_KEYS.items():
        ser = [V.series(byv, sets, levels, arm, k) for k in keys]
        scores[f"{arm}: 질문만"] = {v: np.mean([s[v] for s in ser], axis=0) for v in byv}
        scores[f"{arm}: 합친 점수"] = half(scores[f"{arm}: 질문만"])
        for k, s in zip(keys, ser):
            scores[f"{arm}: {k}"] = s
    neg = lambda r: not r["big"]
    out["auc_by_kind"] = {}
    for name, per in scores.items():
        row = {}
        for k in KINDS:
            row[k], _ = pooled_auc(byv, per, lambda r, k=k: r["k"] == k, neg)
        row["owner"], n_owner = pooled_auc(byv, per, lambda r: r["k"] in OWNER, neg)
        row["rest_of_label"], _ = pooled_auc(byv, per, lambda r: r["k"] in ("animal", "craft"), neg)
        row["big"], _ = pooled_auc(byv, per, lambda r: bool(r["big"]), neg)
        out["auc_by_kind"][name] = row
        print(f"  {name:28s}", {KO.get(k, k): v for k, v in row.items()})
    out["owner_positions"] = n_owner

    # 영상째 다시 뽑기(2,000번): 소유자의 말에 맞춘 표시(13곳)와 지금의 표시(56곳)에서, 정하는 쪽의 AUC와 그 차이
    rng = np.random.default_rng(20261116)
    vids = list(byv)
    def boot(per_a, per_b, pos):
        vals = []
        for _ in range(2000):
            pick = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
            sa, sb, y = [], [], []
            for v in pick:
                ra, rb = C.pct_rank(np.asarray(per_a[v], float)), C.pct_rank(np.asarray(per_b[v], float))
                for i, rec in enumerate(byv[v]):
                    if pos(rec) or not rec["big"]:
                        sa.append(ra[i]); sb.append(rb[i]); y.append(bool(pos(rec)))
            if any(y):
                vals.append((HE.auc(sa, y), HE.auc(sb, y)))
        a = np.array(vals)
        ci = lambda x: [round(float(q), 3) for q in np.percentile(x, [2.5, 97.5])]
        return {"a": ci(a[:, 0]), "b": ci(a[:, 1]), "a_minus_b": ci(a[:, 0] - a[:, 1]), "resamples_used": len(vals)}
    out["boot"] = {}
    for lab, pos in (("owner", lambda r: r["k"] in OWNER), ("big", lambda r: bool(r["big"])), ("craft", lambda r: r["k"] == "craft")):
        for a in ("Ma: 합친 점수", "Mb: 합친 점수", "막연한 질문"):
            out["boot"][f"{lab}: {a} − 손 기준"] = boot(scores[a], scores["손 기준: 막연한 질문과 반반"], pos)
            b = out["boot"][f"{lab}: {a} − 손 기준"]
            print(f"  다시 뽑기 [{lab}] {a} {b['a']} / 손 기준 {b['b']} / 차이 {b['a_minus_b']} (쓴 횟수 {b['resamples_used']})")

    # ---------- (나) Gemini의 확인: 후보 단위 / (다) 높이 친 것 / (라) 소유자의 말에 맞춘 표시
    vid = {v[:5]: v for v in byv}
    out["verify"] = {}
    listing = []
    for folder, sec in (("verify_runs", "18절"), ("verify_rr", "20절")):
        for arm in ("Ma", "Mb"):
            cands, agg = [], {lab: {k: {"clips": 0, "with_event": 0, "events": 0} for k in ("now", "checked", "perfect")} for lab in ("big", "owner")}
            total = {"big": 0, "owner": 0}
            none_owner = 0
            for d in sorted((HERE / folder).glob(f"mc_{arm}_mc_*/verdict.json")):
                item, verdict = mf.read(d.parent / "item.json"), mf.read(d)
                rs = byv[vid[item["name"][6:11]]]
                t = np.array([r["t"] for r in rs]); kind = np.array([r["k"] for r in rs])
                p = {c["id"]: c for c in verdict["candidates"]}
                kept = M.keep_clips(item, verdict)
                now = [{"start": a, "end": b} for a, b in item["now_clips"]]
                inside = lambda c: (t >= c["start"]) & (t < c["end"])
                for lab, ok in (("big", kind != "none"), ("owner", np.isin(kind, OWNER))):
                    total[lab] += int(ok.sum())
                    perfect = [c for c in item["candidates"] if (ok & inside(c)).any()]
                    for name, cs in (("now", now), ("checked", kept), ("perfect", perfect)):
                        a = agg[lab][name]
                        a["clips"] += len(cs); a["with_event"] += sum(bool((ok & inside(c)).any()) for c in cs)
                        a["events"] += int((ok & np.any([inside(c) for c in cs], axis=0)).sum()) if cs else 0
                none_owner += not kept
                for c in item["candidates"]:
                    ks = [k for k in KINDS if (kind[inside(c)] == k).any()]
                    row = {"item": item["name"], "sec": sec, "arm": arm, "start": c["start"], "end": c["end"], "p": p[c["id"]]["p"], "see": p[c["id"]]["see"],
                           "kind": ks[0] if ks else "none", "kinds": ks, "kept": c in kept, "from": c.get("from")}
                    cands.append(row); listing.append(row)
            by_kind = {}
            for k in KINDS + ["none"]:
                ps = [c["p"] for c in cands if c["kind"] == k]
                by_kind[k] = {"n": len(ps), "median_p": round(float(np.median(ps)), 2) if ps else None, "mean_p": round(float(np.mean(ps)), 2) if ps else None,
                              "kept": sum(c["kept"] for c in cands if c["kind"] == k), "p>=0.5": sum(x >= 0.5 for x in ps)}
            y = np.array([c["kind"] in OWNER for c in cands]); yb = np.array([c["kind"] != "none" for c in cands]); s = np.array([c["p"] for c in cands])
            out["verify"][f"{sec} {arm}"] = {"candidates": len(cands), "by_kind": by_kind, "events_total": total, "rescored": agg, "videos_none_returned": int(none_owner),
                                           "auc_of_p": {"big": round(float(HE.auc(list(s), list(yb))), 3), "owner": round(float(HE.auc(list(s), list(y))), 3)}}
            print(f"== {sec} {arm}: 후보 {len(cands)} | p의 가름: 큰 사건 {out['verify'][f'{sec} {arm}']['auc_of_p']['big']}, 소유자의 말 {out['verify'][f'{sec} {arm}']['auc_of_p']['owner']}")
            print("   종류별 [후보 수, p 가운데 값, 남긴 것]:", {KO[k]: (v["n"], v["median_p"], v["kept"]) for k, v in by_kind.items()})
            for lab in ("big", "owner"):
                print(f"   {lab} (모두 {total[lab]}곳):", {k: (v["clips"], v["with_event"], v["events"]) for k, v in agg[lab].items()})
    out["high_p"] = sorted([r for r in listing if r["p"] >= 0.3], key=lambda r: (r["item"][6:11], r["start"], r["sec"], r["arm"]))
    mf.write(HERE / "eval" / "meaning_check.json", out)
    print(f"\np ≥ 0.3인 후보 {len(out['high_p'])}개 (전체 {len(listing)}개)")
    for r in out["high_p"]:
        print(f"  {r['item'][6:11]} {r['sec']} {r['arm']} {M.mmss(r['start'])}~{M.mmss(r['end'])} p={r['p']:.2f} [{KO[r['kind']]}] {r['see'][:105]}")


if __name__ == "__main__":
    main()
