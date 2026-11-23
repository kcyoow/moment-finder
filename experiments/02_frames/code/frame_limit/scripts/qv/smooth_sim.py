"""위치별 확률을 섞는 방법과 자르는 기준을 이미 받은 확률로 비교한다. 새 요청을 보내지 않는다.
표준 라이브러리만 쓴다.

섞는 방법 (위치 단위. Charades는 위치 간격 1초, QVHighlights는 2초)
- none  : 섞지 않음
- avg3  : 앞 0.25 + 자기 0.5 + 뒤 0.25  (지금까지 쓴 것)
- uni3  : 앞·자기·뒤를 똑같이 1/3씩
- tri5  : 앞뒤 두 칸까지, 가까울수록 크게 (1, 2, 3, 2, 1)
- uni5  : 앞뒤 두 칸까지 똑같이
- gauss7: 앞뒤 세 칸까지 종 모양 가중치 (표준편차 1.5칸)
- med3, med5: 평균 대신 가운데 값(중앙값)
- max3  : 앞·자기·뒤 중 가장 큰 값

자르는 기준 (섞은 뒤의 확률에 적용)
- abs0.5   : 0.5 이상
- rel_floor: max(0.3, 0.5 × 최고 확률) 이상  (지금까지 쓴 것)
- rel_gate : 섞기 전 최고 확률이 0.3 미만이면 "없음", 아니면 0.5 × 섞은 뒤 최고 확률 이상
- mean     : 그 영상 전체 확률의 평균 이상 (섞기 전 최고 확률 0.3 미만이면 "없음")
- mean+std : 평균 + 표준편차 이상 (같은 "없음" 조건)
- otsu     : 확률들을 낮은 무리와 높은 무리로 가장 뚜렷하게 가르는 경계 이상 (같은 "없음" 조건)

섞는 방법과 자르는 기준은 앞서 본 영상(loc, qv)에서 고르고, 새 영상(sta, qv2)의 수치로 확인한다.

실행: python3 experiments/frame_limit/scripts/qv/smooth_sim.py
결과: 표 출력, results/smooth_sim.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import math
import statistics
from pathlib import Path

from qv_eval import auc, prf
from recipe_eval import NAME, fill_gaps, load

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
DEV, NEW = ["loc", "qv"], ["sta", "qv2"]
GATE = 0.3


def weighted(ps, weights):
    half = len(weights) // 2
    padded = [ps[0]] * half + list(ps) + [ps[-1]] * half
    total = sum(weights)
    return [sum(w * padded[i + k] for k, w in enumerate(weights)) / total for i in range(len(ps))]


def windowed(ps, size, fn):
    half = size // 2
    padded = [ps[0]] * half + list(ps) + [ps[-1]] * half
    return [fn(padded[i:i + size]) for i in range(len(ps))]


KERNELS = {
    "none": lambda ps: list(ps),
    "avg3": lambda ps: weighted(ps, [1, 2, 1]),
    "uni3": lambda ps: weighted(ps, [1, 1, 1]),
    "tri5": lambda ps: weighted(ps, [1, 2, 3, 2, 1]),
    "uni5": lambda ps: weighted(ps, [1] * 5),
    "gauss7": lambda ps: weighted(ps, [math.exp(-d * d / (2 * 1.5 ** 2)) for d in range(-3, 4)]),
    "med3": lambda ps: windowed(ps, 3, statistics.median),
    "med5": lambda ps: windowed(ps, 5, statistics.median),
    "max3": lambda ps: windowed(ps, 3, max),
}


def cut(raw, ps, rule):
    """잡힌 위치 목록. raw는 섞기 전, ps는 섞은 뒤의 확률."""
    if rule == "abs0.5":
        thr = 0.5
    elif rule == "rel_floor":
        thr = max(GATE, 0.5 * max(ps))
    elif max(raw) < GATE:
        return [False] * len(ps)
    elif rule == "rel_gate":
        thr = 0.5 * max(ps)
    elif rule == "mean":
        thr = statistics.fmean(ps)
    elif rule == "otsu":
        thr = otsu(ps)
    else:  # mean+std
        thr = statistics.fmean(ps) + statistics.pstdev(ps)
    return [p >= thr for p in ps]


def otsu(ps):
    """값들을 낮은 무리와 높은 무리로 가장 뚜렷하게 가르는 경계 (Otsu 방법)."""
    values = sorted(ps)
    best, thr = -1.0, values[-1]
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            continue
        low, high = values[:i], values[i:]
        between = len(low) * len(high) * (statistics.fmean(low) - statistics.fmean(high)) ** 2
        if between > best:
            best, thr = between, (values[i - 1] + values[i]) / 2
    return thr


RULES = ["abs0.5", "rel_floor", "rel_gate", "mean", "mean+std", "otsu"]


def mean_auc(items, frames, kernel):
    vals = [auc(KERNELS[kernel](it[frames]), it["y"]) for it in items if it[frames]]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def f1(items, frames, kernel, rule):
    tp = fp = fn = 0
    for it in items:
        if not it[frames]:
            continue
        caught = fill_gaps(cut(it[frames], KERNELS[kernel](it[frames]), rule))
        tp += sum(c and y for c, y in zip(caught, it["y"]))
        fp += sum(c and not y for c, y in zip(caught, it["y"]))
        fn += sum(not c and y for c, y in zip(caught, it["y"]))
    prec, rec, score = prf(tp, fp, fn)
    return {"precision": prec, "recall": rec, "f1": score}


def main():
    sets = {tag: load(tag)[0] for tag in DEV + NEW}
    cells = [(tag, frames) for tag in DEV + NEW for frames in ("one", "win4")
             if any(it[frames] for it in sets[tag])]
    label = {(t, f): f"{NAME[t]} {'한 장' if f == 'one' else '4장'}" for t, f in cells}
    result = {"auc": {}, "f1": {}}

    print("1. 섞는 방법별 평균 AUC")
    print(f"{'방법':8s}" + "".join(f"{label[c]:>22s}" for c in cells) + f"{'본 영상 평균':>10s}")
    dev_score = {}
    for name in KERNELS:
        row = {c: mean_auc(sets[c[0]], c[1], name) for c in cells}
        dev_score[name] = sum(v for c, v in row.items() if c[0] in DEV) / sum(c[0] in DEV for c in cells)
        result["auc"][name] = {label[c]: round(v, 3) for c, v in row.items()}
        print(f"{name:8s}" + "".join(f"{row[c]:>22.3f}" for c in cells) + f"{dev_score[name]:>10.3f}")
    best_kernel = max(KERNELS, key=lambda k: dev_score[k])
    print(f"본 영상에서 가장 높은 방법: {best_kernel}")

    for kernel in dict.fromkeys(["none", "avg3", best_kernel]):
        print(f"\n2. 자르는 기준별 F1 (섞는 방법 {kernel}, 한 칸 끊김 잇기 적용)")
        print(f"{'기준':10s}" + "".join(f"{label[c]:>22s}" for c in cells) + f"{'본 영상 평균':>10s}")
        result["f1"][kernel] = {}
        for rule in RULES:
            row = {c: f1(sets[c[0]], c[1], kernel, rule) for c in cells}
            dev = sum(v["f1"] for c, v in row.items() if c[0] in DEV) / sum(c[0] in DEV for c in cells)
            result["f1"][kernel][rule] = {label[c]: {k: round(x, 3) for k, x in v.items()} for c, v in row.items()}
            print(f"{rule:10s}" + "".join(f"{v['f1']:>10.3f} ({v['precision']:.2f}/{v['recall']:.2f})"
                                          for v in row.values()) + f"{dev:>10.3f}")
    result["best_kernel_on_dev"] = best_kernel
    (HERE / "results" / "smooth_sim.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
