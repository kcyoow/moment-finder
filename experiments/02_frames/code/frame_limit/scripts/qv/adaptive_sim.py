"""Jev에 묻는 위치를 줄였을 때의 결과를 저장된 확률로 흉내 낸다. 새 요청 없음. 표준 라이브러리만 쓴다.

방법
- 전부 묻기: 지금까지의 방식
- 고르게 n칸마다: 일정한 간격으로만 묻고 사이는 직선으로 잇는다
- 굵게 묻고 달라지는 곳만 다시 묻기: 먼저 start칸 간격으로 묻고, 이웃한 두 답의 차이가 delta를 넘으면
  그 가운데를 더 묻는다. 차이가 없어질 때까지 반복한다
- 무료 후보: SigLIP 2 작은 모델(한 장, 5칸 평균)의 점수가 높은 위치를 처음 묻는 목록에 더한다.
  굵게 물을 때 사이에 끼어 놓치는 장면을 줄이려는 것이다

채점은 siglip_eval.py와 같다(5칸 평균 → Otsu → 한 칸 끊김 잇기, 항목별 AUC 평균).
설정(start, delta, 후보 비율)은 본 영상(qv)에서 고르고 새 영상(qv2, sta)에서 확인한다.

주의: 묻지 않은 위치의 답은 저장된 답을 쓰지 않고 이웃 값으로 채운다. 한 칸은 qv·qv2에서 2초, sta에서 1초다.

실행: python3 experiments/frame_limit/scripts/qv/adaptive_sim.py
결과: 표 출력, results/adaptive_sim.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import math

import siglip_eval as E
from qv_eval import auc, prf

SETTINGS = [(start, delta, frac) for start in (4, 8) for delta in (0.1, 0.2, 0.3) for frac in (0.0, 0.05, 0.10)]


def interpolate(n, known):
    idx = sorted(known)
    out = []
    for k in range(n):
        if k in known:
            out.append(known[k])
            continue
        lo = max((i for i in idx if i < k), default=None)
        hi = min((i for i in idx if i > k), default=None)
        if lo is None:
            out.append(known[hi])
        elif hi is None:
            out.append(known[lo])
        else:
            w = (k - lo) / (hi - lo)
            out.append(known[lo] * (1 - w) + known[hi] * w)
    return out


def uniform(ps, stride):
    return {k: ps[k] for k in range(0, len(ps), stride)} | {len(ps) - 1: ps[-1]}


def adaptive(ps, start, delta, seeds=()):
    n = len(ps)
    known = {k: ps[k] for k in list(range(0, n, start)) + [n - 1] + list(seeds)}
    changed = True
    while changed:
        changed = False
        idx = sorted(known)
        for a, b in zip(idx, idx[1:]):
            if b - a > 1 and abs(known[a] - known[b]) > delta:
                mid = (a + b) // 2
                known[mid] = ps[mid]
                changed = True
    return known


def free_candidates(scores, frac):
    """무료 모델 점수가 높은 위치를 서로 붙지 않게 고른다."""
    if frac <= 0:
        return []
    sm = E.smooth(scores)
    want = max(1, math.ceil(frac * len(sm)))
    picked = []
    for k in sorted(range(len(sm)), key=lambda i: -sm[i]):
        if all(abs(k - p) > 1 for p in picked):
            picked.append(k)
        if len(picked) >= want:
            break
    return picked


def run(items, cond, make):
    total, asked, positions, aucs = [0, 0, 0], 0, 0, []
    for it in items:
        if not it[cond]:
            continue
        known = make(it, it[cond])
        est = interpolate(len(it[cond]), known)
        asked += len(known)
        positions += len(it[cond])
        a = auc(E.smooth(est), it["y"])
        if a is not None:
            aucs.append(a)
        total = [x + c for x, c in zip(total, E.counts(E.cut_same(est), it["y"]))]
    return {"asked": round(asked / positions, 3), "auc": round(sum(aucs) / len(aucs), 3), "f1": round(prf(*total)[2], 3)}


def main():
    sets = {t: E.bundle(t) for t in ("qv", "qv2", "sta")}
    base = E.Scores.from_file(E.HERE / "results" / "siglip_scores_base.json")
    rows = [("전부 묻기", lambda it, ps: dict(enumerate(ps)))]
    for stride in (2, 4, 8):
        rows.append((f"고르게 {stride}칸마다", lambda it, ps, s=stride: uniform(ps, s)))
    for start, delta, frac in SETTINGS:
        name = f"굵게 {start}칸, 차이 {delta}" + (f", 무료 후보 {int(frac * 100)}%" if frac else "")
        rows.append((name, lambda it, ps, s=start, d=delta, f=frac:
                     adaptive(ps, s, d, free_candidates(E.series(base, it, "one"), f))))
    result = {}
    for title, tag, cond in [("QVHighlights 본 10개(한 장), 설정을 고르는 묶음", "qv", "one"),
                             ("QVHighlights 새 10개(한 장)", "qv2", "one"),
                             ("Charades 새 12개(Jev 4장)", "sta", "win4")]:
        print(f"\n{title}")
        print(f"  {'방법':34s}{'물은 비율':>8s}{'AUC':>8s}{'F1':>8s}")
        result[title] = {}
        for name, make in rows:
            r = run(sets[tag], cond, make)
            result[title][name] = r
            print(f"  {name:34s}{r['asked']:>9.0%}{r['auc']:>9.3f}{r['f1']:>8.3f}")
    (E.HERE / "results" / "adaptive_sim.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
