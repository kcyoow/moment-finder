"""질문마다 "주변의 평소 값"을 빼고 남는 것만 쓰면 헛짚음이 줄어드는지 받아 둔 답으로 본다 (이 Mac, 돈 없음). pipeline/README.md 15절.

방식 (질문마다, 영상 안에서. 값은 0~1. 뺀 뒤 음수는 0)
  value_mean : 그대로 (지금 기본값)
  med10      : 값 − 앞뒤 10초의 중앙값
  med30      : 값 − 앞뒤 30초의 중앙값
  onset10    : 값 − 바로 앞 10초의 평균
  global     : 값 − 영상 전체의 중앙값
그다음은 지금과 같다: 질문 평균 → (막연한 질문과 반반) → 영상 안 순위. 막연한 질문은 손대지 않는다.
자료·지표·구간은 mf_combine2_check.py(14절)와 같고, 그 파일의 함수를 그대로 쓴다(합치는 함수만 바꿔 끼운다).

쓰는 것: pipeline/eval/context_rule.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_context_check.py
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

RULES = ("value_mean", "med10", "med30", "onset10", "global")
CAND = RULES[1:]
FORMS = K.FORMS


def contrast(v, kind):
    """한 질문의 영상 안 값 → 주변의 평소 값을 뺀 것(음수는 0)."""
    if kind == "global":
        return np.clip(v - np.median(v), 0.0, None)
    out = np.zeros(len(v))
    for t in range(len(v)):
        if kind == "onset10":
            w = v[max(0, t - 10): t]
            base = w.mean() if len(w) else v[t]
        else:
            m = 10 if kind == "med10" else 30
            base = np.median(v[max(0, t - m): t + m + 1])
        out[t] = max(0.0, v[t] - base)
    return out


def scores(answers, keys, levels=None):
    val = [np.array([K.unit((a or {}).get(k), levels, k) for a in answers]) for k in keys]
    v = np.clip(np.array([np.where(np.isnan(x), np.nanmean(x) if not np.all(np.isnan(x)) else 0.0, x) for x in val]), 0.0, 1.0)
    out = {"value_mean": v.mean(0)}
    for kind in CAND:
        out[kind] = np.mean([contrast(x, kind) for x in v], axis=0)
    return out


def one_video():
    """13-5절의 한 편(mc_12, 웹사이트 실행): 방식마다 클립과 클립에 든 표시된 사건 수."""
    run = mf.RUNS / "web_1116_043326_3"
    pos, qj, info = mf.read(run / "positions.json"), mf.read(run / "questions.json"), mf.read(run / "run.json")
    recs = {r["k"]: r.get("answers") for r in mf.read(run / "results.json")["records"] if "error" not in r}
    answers = [recs.get(q["k"]) for q in pos]
    levels = mf.levels_of(qj["questions"])
    keys = [k for k in qj["questions"] if k != "worth"]
    t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
    vr = C.pct_rank(np.array([K.unit((a or {}).get("worth")) for a in answers]))
    fight = [79.5, 82.5, 84.5, 86.5, 87.5, 91.5, 94.5, 97.5, 139.5, 141.5, 142.5]
    prog = [45.5, 52.5, 69.5, 71.5, 253.5]
    out = {}
    for rule, s in scores(answers, keys, levels).items():
        for form, x in K.forms(s, vr).items():
            clips = C.make_clips(x, t0, t1, info["duration"])
            inside = lambda t: any(c["start"] <= t < c["end"] for c in clips)
            out[rule + form] = {"clips": [[c["start"], c["end"]] for c in clips], "fight_in_clips": sum(map(inside, fight)), "progress_in_clips": sum(map(inside, prog))}
    return out


def main():
    K.RULES, K.CAND, K.LATER, K.scores = RULES, CAND, (), scores          # 14절의 함수에 합치는 함수만 바꿔 끼운다
    out = {"label": "질문마다 주변의 평소 값을 뺀 방식 넷의 비교 (받아 둔 답. pipeline/README.md 15절)", "rules": list(RULES),
           "minecraft": K.minecraft(), "tvsum": K.tvsum(), "robot": K.robot(), "one_video_mc12": one_video()}
    with open(HERE / "eval" / "context_rule.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    m = out["minecraft"]
    print("마인크래프트: 칸은 AUC / 클립에 든 큰 사건 수")
    for name in m["sets"]:
        for form in FORMS:
            print(f"  {name:4s}{form:7s} " + " | ".join(f"{rule} {m['arms'][f'{name}:{rule}{form}']['auc_big']:.3f}/{m['arms'][f'{name}:{rule}{form}']['big_in_clips']}" for rule in RULES))
    for k, d in m["differences"].items():
        print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    print("TVSum (τ / 상위 15% 올림 값)")
    for name, res in out["tvsum"].items():
        for form in FORMS:
            print(f"  {name:3s}{form:7s} " + " | ".join(f"{rule} {res['arms'][rule + form]['tau']:.4f}/{res['arms'][rule + form]['lift']:.3f}" for rule in RULES))
        for k, d in res["differences"].items():
            if k.endswith("_tau"):
                print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    print("로봇 (영상마다 AUC)")
    for v, res in out["robot"].items():
        for sname, arms in res.items():
            for form in FORMS:
                print(f"  {v:12s} {sname:11s}{form:7s} " + " | ".join(f"{rule} {arms[rule + form]:.3f}" for rule in RULES))
    print("한 편(mc_12, 웹사이트 실행): 싸움·피해 11곳 / 제작·다이아 5곳 가운데 클립에 든 수")
    for k, v in out["one_video_mc12"].items():
        print(f"  {k:18s} 싸움 {v['fight_in_clips']:2d} 제작 {v['progress_in_clips']} | {v['clips']}")


if __name__ == "__main__":
    main()
