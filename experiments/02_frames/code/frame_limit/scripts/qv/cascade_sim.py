"""4장 판정의 비용을 줄이는 방법을 이미 받은 확률로 계산한다. 새 요청을 보내지 않는다.

비교하는 방식
- 항상 한 장 / 항상 4장 / 요청 종류로 나눔(motion만 4장)
- 한 장으로 먼저 훑고, 확률이 tau 이상인 위치와 그 앞뒤 한 칸만 4장으로 다시 봄

위치당 토큰은 새 Charades 실행의 평균(한 장 1,295, 4장 4,378)을 쓴다. 임계값 규칙과 끊김 잇기는
recipe_eval.py와 같다. tau는 결과를 보면서 비교하는 값이라 낙관적으로 읽어야 한다.

실행: python3 experiments/frame_limit/scripts/qv/cascade_sim.py
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

from qv_eval import auc, prf
from recipe_eval import NAME, R, catch, fill_gaps, load

ONE, WIN = 1295, 4378


def mix(item, tau, pad=1):
    hit = [p >= tau for p in item["one"]]
    chosen = [any(hit[max(0, i - pad): i + pad + 1]) for i in range(len(hit))]
    return [w if c else o for o, w, c in zip(item["one"], item["win4"], chosen)], sum(chosen)


def report(items, name, series_fn, tokens_fn):
    aucs, tp, fp, fn, tokens, n = [], 0, 0, 0, 0, 0
    for it in items:
        ps, extra = series_fn(it)
        a = auc(ps, it["y"])
        if a is not None:
            aucs.append(a)
        caught = fill_gaps(catch(ps, R))
        tp += sum(c and y for c, y in zip(caught, it["y"]))
        fp += sum(c and not y for c, y in zip(caught, it["y"]))
        fn += sum(not c and y for c, y in zip(caught, it["y"]))
        tokens += tokens_fn(it, extra)
        n += len(ps)
    prec, rec, f1 = prf(tp, fp, fn)
    print(f"  {name:32s} 평균 AUC {sum(aucs) / len(aucs):.3f}  F1 {f1:.3f} (정밀도 {prec:.2f} 재현율 {rec:.2f})"
          f"  위치당 토큰 {tokens / n:,.0f} ({tokens / n / ONE:.2f}배)")


def main():
    for tag in ("sta", "loc"):
        items, _ = load(tag)
        print(f"\n{NAME[tag]}: 항목 {len(items)}개 중 motion {sum(it['needs'] == 'motion' for it in items)}개")
        report(items, "항상 한 장", lambda it: (it["one"], 0), lambda it, e: ONE * len(it["one"]))
        report(items, "항상 4장", lambda it: (it["win4"], 0), lambda it, e: WIN * len(it["one"]))
        report(items, "종류로 나눔 (motion만 4장)",
               lambda it: (it["win4"] if it["needs"] == "motion" else it["one"], 0),
               lambda it, e: (WIN if it["needs"] == "motion" else ONE) * len(it["one"]))
        for tau in (0.05, 0.1, 0.2, 0.3):
            report(items, f"한 장 먼저, {tau} 이상 주변만 4장", lambda it, tau=tau: mix(it, tau),
                   lambda it, e: ONE * len(it["one"]) + WIN * e)


if __name__ == "__main__":
    main()
