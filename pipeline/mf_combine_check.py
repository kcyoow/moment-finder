"""기준 질문들을 하나의 점수로 합치는 방식 셋을 받아 둔 답으로 견준다 (이 Mac, 돈 없음). pipeline/README.md 7절.

까닭: 마인크래프트의 손 기준(action, progress)을 highlight_auto 5절의 식(질문별 순위의 평균)으로 합치면 큰 사건의 AUC가 0.790이고,
highlight_h2o가 쓴 식(값을 0~1로 맞춰 평균)으로 합치면 0.847이다. 같은 답인데 합치는 식에 따라 달라진다.

견주는 식 (모두 영상 안에서, 학습한 가중치 없이)
  rank_mean  : 질문마다 영상 안 백분위 순위 → 평균               (highlight_auto 5절. 지금까지의 식)
  value_mean : noul은 "예" 확률, score는 기댓값 ÷ (단계 수 − 1) → 평균   (highlight_h2o의 최종 점수와 같은 꼴)
  rank_max   : 질문마다 영상 안 백분위 순위 → 가장 큰 것              ("어느 하나라도 강하게 맞으면")
기준 묶음: TVSum 넷((나), T, TP, P), 마인크래프트 다섯(손 기준, (나), G, GP, P12). 답은 experiments/의 결과 파일 그대로다.
지표: TVSum은 τ와 상위 15% 올림 값, 마인크래프트는 큰 사건의 AUC. 막연한 질문과 반반 합친 값도 함께 낸다.

쓰는 것: pipeline/eval/combine_rule.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_combine_check.py
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
HA, HH, HP_ = EXP / "highlight_auto", EXP / "highlight_h2o", EXP / "highlight_probe"
for p in (HERE, HA, HH):
    sys.path.insert(0, str(p))
import ha_eval as E  # noqa: E402
import hh_eval as HE  # noqa: E402
import hh_prepare as HP  # noqa: E402
import mf_core as C  # noqa: E402

RULES = ("rank_mean", "value_mean", "rank_max")
r4 = E.r4


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def unit_value(a):
    """답 → 0~1 값. noul은 확률, score는 기댓값 ÷ (단계 수 − 1)."""
    if a is None:
        return np.nan
    if "noul" in a:
        return float(a["noul"])
    return float(a["score"]) / (len(a["probabilities"]) - 1)


def scores(answers, keys):
    """한 영상의 위치별 답(없으면 None)과 기준 질문 → {식: 위치별 점수}."""
    val = {k: np.array([unit_value((a or {}).get(k)) for a in answers]) for k in keys}
    rank = {k: C.pct_rank(v) for k, v in val.items()}
    filled = {k: np.where(np.isnan(v), np.nanmean(v) if not np.all(np.isnan(v)) else 0.0, v) for k, v in val.items()}
    return {"rank_mean": np.mean([rank[k] for k in keys], axis=0), "value_mean": np.mean([filled[k] for k in keys], axis=0),
            "rank_max": np.max([rank[k] for k in keys], axis=0)}


def with_vague(s, vague_rank):
    return 0.5 * vague_rank + 0.5 * C.pct_rank(s)


# =====================================================================  TVSum
def tvsum_sets():
    """{묶음: {영상: {k: 답}}}와 그 묶음의 질문 키."""
    sets = {}

    def add(name, path, prefix=None, drop=("worth",)):
        by = {}
        for r in read(path)["records"]:
            if "error" in r:
                continue
            a = {k[len(prefix):] if prefix else k: v for k, v in r["answers"].items() if (k.startswith(prefix) if prefix else k not in drop)}
            by.setdefault(r["video"], {})[r["k"]] = a
        sets[name] = by
    add("na", HA / "results" / "tvsum.json")
    add("T", HP_ / "results" / "main_tvsum__tv_titled.json", "t__")
    add("TP", HP_ / "results" / "main_tvsum__tv_titled.json", "tp__")
    add("P", HP_ / "results" / "main_tvsum__tv_plain.json")
    worth = {}
    for r in read(HA / "results" / "tvsum.json")["records"]:
        if "error" not in r:
            worth.setdefault(r["video"], {})[r["k"]] = r["answers"]["worth"]["noul"]
    return sets, worth


def eval_tvsum():
    tv = E.TV()
    sets, worth = tvsum_sets()
    n = {v: len(tv.items[v]["positions"]) for v in tv.videos}
    out = {}
    for name, by in sets.items():
        per = {f"{rule}{suffix}": {"tau": {}, "lift": {}} for rule in RULES for suffix in ("", "+vague")}
        for v in tv.videos:
            answers = [by.get(v, {}).get(k) for k in range(n[v])]
            keys = sorted({q for a in answers if a for q in a})
            vr = C.pct_rank(np.array([worth.get(v, {}).get(k, np.nan) for k in range(n[v])]))
            for rule, s in scores(answers, keys).items():
                for suffix, x in (("", s), ("+vague", with_vague(s, vr))):
                    m = E.video_metrics(tv, v, x)
                    per[rule + suffix]["tau"][v], per[rule + suffix]["lift"][v] = m["tau"], m["lift"]
        res = {"arms": {a: {m: {"mean": r4(np.mean(list(per[a][m].values()))), "ci": E.boot_ci({"x": per[a][m]}, tv.videos)} for m in ("tau", "lift")} for a in per},
               "differences": {}}
        for suffix in ("", "+vague"):
            for rule in ("value_mean", "rank_max"):
                for m in ("tau", "lift"):
                    d = {v: per[rule + suffix][m][v] - per["rank_mean" + suffix][m][v] for v in tv.videos}
                    res["differences"][f"{rule}{suffix}_minus_rank_mean{suffix}_{m}"] = {"point": r4(np.mean(list(d.values()))), "ci": E.boot_ci({"d": d}, tv.videos)}
        out[name] = res
    return out


# =====================================================================  마인크래프트
def eval_minecraft():
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted((r for r in HP.read(HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    sets = {"hand": {r["id"]: {"action": r["action"], "progress": r["progress"]} for r in recs}}

    def add(name, path, prefix=None):
        sets[name] = {r["id"]: {k[len(prefix):] if prefix else k: v for k, v in r["answers"].items() if (k.startswith(prefix) if prefix else True)}
                      for r in read(path)["records"] if "error" not in r}
    add("na", HA / "results" / "minecraft.json")
    add("G", HP_ / "results" / "main_minecraft__mc_game.json", "g__")
    add("GP", HP_ / "results" / "main_minecraft__mc_game.json", "gp__")
    add("P12", HP_ / "results" / "main_minecraft__mc_plain.json")
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    arms = []
    for name, by in sets.items():
        for v, rs in byv.items():
            answers = [by.get(r["id"]) for r in rs]
            keys = sorted({q for a in answers if a for q in a})
            vr = C.pct_rank(np.array([r["worth"] if not isinstance(r["worth"], dict) else r["worth"]["noul"] for r in rs], float))
            for rule, s in scores(answers, keys).items():
                for suffix, x in (("", C.pct_rank(s)), ("+vague", C.pct_rank(with_vague(s, vr)))):
                    for i, r in enumerate(rs):
                        r[f"s_{name}:{rule}{suffix}"] = x[i]
        arms += [f"{name}:{rule}{suffix}" for rule in RULES for suffix in ("", "+vague")]

    def auc(key):
        return lambda rs: HE.auc([x[key] for x in rs], [x["big"] for x in rs])

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(auc(f"s_{a}")(rs), auc(f"s_{b}")(rs))
    out = {"big_events": int(sum(r["big"] for r in recs)), "arms": {}, "differences": {}}
    for a in arms:
        out["arms"][a] = {"auc_big": r4(auc(f"s_{a}")(recs)), "ci": HE.boot(recs, auc(f"s_{a}"))[:2]}
    for name in sets:
        for suffix in ("", "+vague"):
            for rule in ("value_mean", "rank_max"):
                a, b = f"{name}:{rule}{suffix}", f"{name}:rank_mean{suffix}"
                lo, hi, _ = HE.boot(recs, diff(a, b))
                out["differences"][f"{a}_minus_rank_mean"] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi]}
    return out


def main():
    out = {"label": "기준 질문들을 합치는 식 셋의 비교 (받아 둔 답. 결과를 본 뒤에 한 확인이다)", "rules": list(RULES),
           "tvsum": eval_tvsum(), "minecraft": eval_minecraft()}
    (HERE / "eval").mkdir(exist_ok=True)
    with open(HERE / "eval" / "combine_rule.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("TVSum (τ / 상위 15% 올림 값)")
    for name, res in out["tvsum"].items():
        for suffix in ("", "+vague"):
            print(f"  {name:3s}{suffix:7s} " + " | ".join(f"{rule} {res['arms'][rule + suffix]['tau']['mean']:.4f} / {res['arms'][rule + suffix]['lift']['mean']:.4f}" for rule in RULES))
        for k, d in res["differences"].items():
            print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    print("마인크래프트 (큰 사건 AUC)")
    m = out["minecraft"]
    for name in ("hand", "na", "G", "GP", "P12"):
        for suffix in ("", "+vague"):
            print(f"  {name:4s}{suffix:7s} " + " | ".join(f"{rule} {m['arms'][f'{name}:{rule}{suffix}']['auc_big']:.4f}" for rule in RULES))
    for k, d in m["differences"].items():
        print(f"      {k}: {d['point']:+.4f} {d['ci']}")


if __name__ == "__main__":
    main()
