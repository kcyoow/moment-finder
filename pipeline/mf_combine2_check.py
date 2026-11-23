"""기준 질문들을 합치는 식 여섯을 받아 둔 답으로 견준다 (이 Mac, 돈 없음). pipeline/README.md 14절.

물음: 질문들이 서로 다른 종류의 장면을 묻는 요청("A거나 B거나")에서 평균 대신 "가장 높은 것" 쪽의 식이 더 나은가,
      그 식이 다른 꼴의 요청(한 가지를 여러 면으로 묻는 것)을 해치지 않는가.

식 (모두 영상 안에서, 학습한 가중치 없이. 값은 noul의 "예" 확률, score의 기댓값 ÷ (단계 수 − 1))
  value_mean : 값의 평균 (지금 기본값)
  pmean3     : (값³의 평균)의 세제곱근
  noisy_or   : 1 − Π(1 − 값)
  value_max  : 값의 최댓값
  rank_mean, rank_max : 7절의 것 (영상 안 백분위 순위의 평균 / 최댓값)
식마다 기준만, 그리고 막연한 질문과 반반(0.5·순위(막연) + 0.5·순위(기준))을 낸다.

자료: 마인크래프트 일곱 벌(손 기준, 나, G, GP, P12, Gemini Ma, Mb), TVSum 넷(나, T, TP, P), 로봇 넷(조정자, R0, Ra, Rb).
지표: 마인크래프트 큰 사건 AUC와 클립에 든 큰 사건 수, TVSum τ와 상위 15% 올림 값, 로봇 영상마다 AUC.
구간: 영상째 다시 뽑기 2,000번, 씨앗 20261112 (각 실험의 함수 그대로). 후보(pmean3, noisy_or, value_max) − value_mean에만 낸다.

쓰는 것: pipeline/eval/combine_rule2.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_combine2_check.py
"""
import sys

sys.dont_write_bytecode = True
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf_combine_check as CC  # noqa: E402  (받아 둔 답 읽기. ha_eval, hh_eval, hh_prepare를 불러온다)
import mf_core as C  # noqa: E402
import mf_gem_check as G  # noqa: E402
import mf_robot_check as R  # noqa: E402

E, HE, HP = CC.E, CC.HE, CC.HP
RULES = ("value_mean", "pmean3", "noisy_or", "value_max", "rank_mean", "rank_max")
CAND = ("pmean3", "noisy_or", "value_max")
LATER = ("rank_max",)          # 후보가 아니었다. 결과를 본 뒤에 값 평균과의 차이의 구간을 더 냈다(14-5절)
FORMS = ("", "+vague")
r4 = E.r4
CTX = {}                       # 지금 보는 영상의 막연한 질문 순위. 바꿔 끼운 합치는 함수가 읽을 수 있게 둔다(이 파일의 계산에는 쓰지 않는다)


def unit(a, levels=None, key=None):
    """답 → 0~1 값. score의 단계 수는 답의 probabilities에서, 없으면 질문의 단계 수에서 읽는다."""
    if a is None:
        return np.nan
    if "noul" in a:
        return float(a["noul"])
    n = len(a["probabilities"]) if "probabilities" in a else (levels or {})[key]
    return float(a["score"]) / (n - 1)


def scores(answers, keys, levels=None):
    """한 영상의 위치별 답(없으면 None)과 기준 질문 → {식: 위치별 점수}. 답이 없는 곳은 그 질문의 평균으로 채운다(7절과 같다)."""
    val = [np.array([unit((a or {}).get(k), levels, k) for a in answers]) for k in keys]
    rank = np.array([C.pct_rank(v) for v in val])
    v = np.clip(np.array([np.where(np.isnan(x), np.nanmean(x) if not np.all(np.isnan(x)) else 0.0, x) for x in val]), 0.0, 1.0)
    return {"value_mean": v.mean(0), "pmean3": np.cbrt((v ** 3).mean(0)), "noisy_or": 1.0 - np.prod(1.0 - v, axis=0), "value_max": v.max(0),
            "rank_mean": rank.mean(0), "rank_max": rank.max(0)}


def forms(s, vague_rank):
    return {"": C.pct_rank(s), "+vague": C.pct_rank(0.5 * vague_rank + 0.5 * C.pct_rank(s))}


# =====================================================================  마인크래프트
def minecraft():
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted((r for r in HP.read(CC.HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    sets = {"hand": {r["id"]: {"action": r["action"], "progress": r["progress"]} for r in recs}}

    def add(name, path, prefix=None):
        sets[name] = {r["id"]: {k[len(prefix):] if prefix else k: v for k, v in r["answers"].items() if (k.startswith(prefix) if prefix else True)}
                      for r in CC.read(path)["records"] if "error" not in r}
    add("na", CC.HA / "results" / "minecraft.json")
    add("G", CC.HP_ / "results" / "main_minecraft__mc_game.json", "g__")
    add("GP", CC.HP_ / "results" / "main_minecraft__mc_game.json", "gp__")
    add("P12", CC.HP_ / "results" / "main_minecraft__mc_plain.json")
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    levels_g = {}
    sets["Ma"], sets["Mb"] = {}, {}
    for v, rs in byv.items():                                       # Gemini의 질문(12절). 같은 시각끼리 맞춘다
        pos, qj, answers, levels, _ = G.load_run("gem_" + "_".join(v.split("_")[:2]))
        by_t = {q["center"]: a for q, a in zip(pos, answers)}
        for r in rs:
            a = by_t.get(r["t"])
            for name, pre in (("Ma", "ma_"), ("Mb", "mb_")):
                if a:
                    sets[name][r["id"]] = {k[len(pre):]: x for k, x in a.items() if k.startswith(pre)}
        levels_g.update({k[3:]: n for k, n in levels.items()})
    arms, clip_hits, share = [], {}, {}
    for name, by in sets.items():
        for v, rs in byv.items():
            answers = [by.get(r["id"]) for r in rs]
            keys = sorted({q for a in answers if a for q in a if q != "worth"})
            vr = C.pct_rank(np.array([r["worth"] if not isinstance(r["worth"], dict) else r["worth"]["noul"] for r in rs], float))
            CTX["vr"] = vr
            t0, t1 = [r["t"] - 0.5 for r in rs], [r["t"] + 0.5 for r in rs]
            big = np.array([bool(r["big"]) for r in rs])
            for rule, s in scores(answers, keys, levels_g).items():
                for form, x in forms(s, vr).items():
                    arm = f"{name}:{rule}{form}"
                    for i, r in enumerate(rs):
                        r[f"s_{arm}"] = x[i]
                    clips = C.make_clips(x, t0, t1, max(t1))
                    inside = np.array([any(c["start"] <= r["t"] < c["end"] for c in clips) for r in rs])
                    clip_hits[arm] = clip_hits.get(arm, 0) + int((inside & big).sum())
                    share[arm] = share.get(arm, 0) + int(inside.sum())
        arms += [f"{name}:{rule}{form}" for rule in RULES for form in FORMS]

    def auc(key):
        return lambda rs: HE.auc([x[key] for x in rs], [x["big"] for x in rs])

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(auc(f"s_{a}")(rs), auc(f"s_{b}")(rs))
    n_big = int(sum(r["big"] for r in recs))
    out = {"big_events": n_big, "positions": len(recs), "sets": list(sets), "arms": {}, "differences": {}}
    for a in arms:
        out["arms"][a] = {"auc_big": r4(auc(f"s_{a}")(recs)), "big_in_clips": clip_hits[a], "clip_share": r4(share[a] / len(recs))}
    for name in sets:
        for form in FORMS:
            for rule in CAND:
                a, b = f"{name}:{rule}{form}", f"{name}:value_mean{form}"
                lo, hi, _ = HE.boot(recs, diff(a, b))
                out["differences"][f"{a}_minus_value_mean"] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi]}
    out["differences_added_after"] = {}
    for name in sets:
        for form in FORMS:
            for rule in LATER:
                a, b = f"{name}:{rule}{form}", f"{name}:value_mean{form}"
                lo, hi, _ = HE.boot(recs, diff(a, b))
                out["differences_added_after"][f"{a}_minus_value_mean"] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi]}
    return out


# =====================================================================  TVSum
def tvsum():
    tv = E.TV()
    sets, worth = CC.tvsum_sets()
    n = {v: len(tv.items[v]["positions"]) for v in tv.videos}
    out = {}
    for name, by in sets.items():
        per = {f"{rule}{form}": {"tau": {}, "lift": {}} for rule in RULES for form in FORMS}
        for v in tv.videos:
            answers = [by.get(v, {}).get(k) for k in range(n[v])]
            keys = sorted({q for a in answers if a for q in a})
            vr = C.pct_rank(np.array([worth.get(v, {}).get(k, np.nan) for k in range(n[v])]))
            CTX["vr"] = vr
            for rule, s in scores(answers, keys).items():
                for form, x in forms(s, vr).items():
                    m = E.video_metrics(tv, v, x)
                    per[rule + form]["tau"][v], per[rule + form]["lift"][v] = m["tau"], m["lift"]
        res = {"arms": {a: {m: r4(np.mean(list(per[a][m].values()))) for m in ("tau", "lift")} for a in per}, "differences": {}}
        for form in FORMS:
            for rule in CAND:
                for m in ("tau", "lift"):
                    d = {v: per[rule + form][m][v] - per["value_mean" + form][m][v] for v in tv.videos}
                    res["differences"][f"{rule}{form}_minus_value_mean_{m}"] = {"point": r4(np.mean(list(d.values()))), "ci": E.boot_ci({"d": d}, tv.videos)}
        res["differences_added_after"] = {}
        for form in FORMS:
            for rule in LATER:
                for m in ("tau", "lift"):
                    d = {v: per[rule + form][m][v] - per["value_mean" + form][m][v] for v in tv.videos}
                    res["differences_added_after"][f"{rule}{form}_minus_value_mean_{m}"] = {"point": r4(np.mean(list(d.values()))), "ci": E.boot_ci({"d": d}, tv.videos)}
        out[name] = res
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
        res = {}
        for run, pick in ((old, {"coordinator": None}), (new, {"r0": "r0_", "ra": "ra_", "rb": "rb_"})):
            pos, qj, answers, levels, _ = G.load_run(run)
            center = np.array([0.5 * (q["t0"] + q["t1"]) for q in pos])
            y = np.zeros(len(center), bool)
            for k, (a, b, _lab) in R.HELD.items():
                if trials[k]["run"] == old:
                    y |= (center >= trials[k]["start"] + a) & (center <= trials[k]["start"] + b)
            keep = np.ones(len(center), bool)
            if old in R.SKIP:
                keep &= ~((center >= R.SKIP[old][0]) & (center <= R.SKIP[old][1]))
            vr = C.pct_rank(np.array([unit((a or {}).get("worth")) for a in answers]))
            CTX["vr"] = vr
            for sname, pre in pick.items():
                keys = [k for k in qj["criteria_keys"] if pre is None or k.startswith(pre)]
                res[sname] = {f"{rule}{form}": r4(R.auc(x[keep], y[keep])) for rule, s in scores(answers, keys, levels).items() for form, x in forms(s, vr).items()}
        out[old] = res
    return out


def main():
    out = {"label": "기준 질문들을 합치는 식 여섯의 비교 (받아 둔 답. pipeline/README.md 14절. 일부 결과를 본 뒤의 비교다)", "rules": list(RULES), "candidates": list(CAND),
           "minecraft": minecraft(), "tvsum": tvsum(), "robot": robot()}
    with open(HERE / "eval" / "combine_rule2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    m = out["minecraft"]
    print(f"마인크래프트: 큰 사건 {m['big_events']}곳, 위치 {m['positions']}. 칸은 AUC / 클립에 든 큰 사건 수")
    for name in m["sets"]:
        for form in FORMS:
            print(f"  {name:4s}{form:7s} " + " | ".join(f"{rule} {m['arms'][f'{name}:{rule}{form}']['auc_big']:.3f}/{m['arms'][f'{name}:{rule}{form}']['big_in_clips']}" for rule in RULES))
    for k, d in list(m["differences"].items()) + [("(뒤에 더함) " + k, d) for k, d in m["differences_added_after"].items()]:
        print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    print("TVSum (τ / 상위 15% 올림 값)")
    for name, res in out["tvsum"].items():
        for form in FORMS:
            print(f"  {name:3s}{form:7s} " + " | ".join(f"{rule} {res['arms'][rule + form]['tau']:.4f}/{res['arms'][rule + form]['lift']:.3f}" for rule in RULES))
        for k, d in list(res["differences"].items()) + [("(뒤에 더함) " + k, d) for k, d in res["differences_added_after"].items()]:
            if k.endswith("_tau"):
                print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    print("로봇 (영상마다 AUC)")
    for v, res in out["robot"].items():
        for sname, arms in res.items():
            for form in FORMS:
                print(f"  {v:12s} {sname:11s}{form:7s} " + " | ".join(f"{rule} {arms[rule + form]:.3f}" for rule in RULES))


if __name__ == "__main__":
    main()
