"""moment-finder 실전 흐름의 순수 계산: 답을 영상 안 순위로 합치고, 위치별 점수를 클립으로 묶는다. numpy만 쓴다.

합치는 규칙은 experiments/highlight_auto/README.md 5절과 같다(값이 아니라 영상 안 백분위 순위로, 학습한 가중치 없이).
클립으로 묶는 규칙은 pipeline/README.md 3절에 있다.
"""
import numpy as np

SMOOTH = (0.25, 0.5, 0.25)                      # 안내서 4절: 앞뒤 위치와 평균
DEFAULTS = {
    "budget": 0.15,      # 클립 길이의 합이 영상 길이의 이 비율에 닿으면 멈춘다 (TVSum·SumMe의 관례 15%)
    "grow_q": 0.70,      # 정점에서 클립을 넓힐 때, 점수가 영상 안 위 30%에 드는 위치까지만 잇는다
    "bridge": 1,         # 한 칸 끊긴 곳은 잇는다 (안내서 4절)
    "pre": 2.0,          # 앞에 붙이는 시간(초). 순간의 앞 맥락
    "post": 1.0,         # 뒤에 붙이는 시간(초)
    "min_len": 4.0,      # 가장 짧은 클립(초). 사람이 고른 순간은 4~6초 (조사 보고서 2.6절)
    "max_len": 30.0,     # 가까운 클립을 이을 때의 가장 긴 길이(초)
    "hard_max": 60.0,    # 한 대목(점수가 이어져 높은 구간)은 이 길이까지 자르지 않는다(초). 넘으면 정점에서 먼 쪽부터 줄인다
    "min_gap": 5.0,      # 두 클립 사이가 이보다 가까우면(초) 하나로 잇고, 이으면 max_len을 넘을 때는 새 클립을 만들지 않는다
    "max_clips": 0,      # 0이면 예산에 닿을 때까지
    "smooth": True,
}


def pct_rank(x):
    """영상 안 백분위 순위 (중간 순위 − 0.5) ÷ n. 같은 값은 순위의 평균. 값이 없는 곳(nan)은 0.5."""
    x = np.asarray(x, float)
    out = np.full(len(x), 0.5)
    ok = ~np.isnan(x)
    v = x[ok]
    if len(v) == 0:
        return out
    order = np.argsort(v, kind="stable")
    xs = v[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(v))
    for a, b in zip(starts, ends):
        r[order[a:b]] = (a + 1 + b) / 2
    out[ok] = (r - 0.5) / len(v)
    return out


def answer_value(a):
    """한 질문의 답 → 값. noul은 "예" 확률, score는 단계의 기댓값."""
    if a is None:
        return np.nan
    return float(a["noul"]) if "noul" in a else float(a["score"])


COMBINE_RULES = ("value_mean", "rank_mean", "rank_max")


def combine(answers, criteria_keys, vague_key="worth", rule="value_mean", levels=None):
    """answers: 위치마다 {질문: 답} (답이 없는 위치는 None) → 질문별 순위, 기준 점수, 막연한 질문의 순위, 합친 점수.

    기준 질문들을 합치는 식 (pipeline/README.md 7절):
      value_mean : noul은 "예" 확률, score는 기댓값 ÷ (단계 수 − 1)로 0~1에 맞춰 평균. 기본값
      rank_mean  : 질문마다 영상 안 백분위 순위를 내어 평균 (experiments/highlight_auto/README.md 5절의 식)
      rank_max   : 질문마다 영상 안 백분위 순위 가운데 가장 큰 것
    final = 0.5 × 순위(막연한 질문) + 0.5 × 순위(기준 점수). 한쪽만 있으면 그쪽의 순위.
    levels: {score 질문: 단계 수}. value_mean에서 score 질문을 0~1로 맞추는 데 쓴다(없으면 답의 probabilities 길이).
    """
    if rule not in COMBINE_RULES:
        raise ValueError(f"모르는 식: {rule}")
    n = len(answers)
    keys = list(criteria_keys) + ([vague_key] if vague_key else [])
    raw = {k: np.array([answer_value((a or {}).get(k)) for a in answers]) for k in keys}
    ranks = {k: pct_rank(v) for k, v in raw.items()}
    out = {"raw": raw, "rank": ranks, "n": n, "rule": rule}
    crit = None
    if criteria_keys:
        if rule == "value_mean":
            cols = []
            for k in criteria_keys:
                top = 1.0
                first = next(((a or {}).get(k) for a in answers if (a or {}).get(k) is not None), None)
                if first is not None and "score" in first:
                    top = float((levels or {}).get(k) or len(first.get("probabilities", {})) or 0) - 1
                    if top <= 0:
                        raise ValueError(f"{k}: score 질문의 단계 수를 알 수 없다")
                v = raw[k] / top
                cols.append(np.where(np.isnan(v), np.nanmean(v) if not np.all(np.isnan(v)) else 0.0, v))
            crit = np.mean(cols, axis=0)
        elif rule == "rank_mean":
            crit = np.mean([ranks[k] for k in criteria_keys], axis=0)
        else:
            crit = np.max([ranks[k] for k in criteria_keys], axis=0)
        out["criteria"] = crit
    vague = ranks[vague_key] if vague_key else None
    if vague is not None:
        out["vague"] = vague
    if crit is not None and vague is not None:
        out["final"] = 0.5 * vague + 0.5 * pct_rank(crit)
    else:
        out["final"] = pct_rank(crit) if crit is not None else vague
    return out


def smooth(s):
    s = np.asarray(s, float)
    if len(s) < 3:
        return s.copy()
    a, b, c = SMOOTH
    out = b * s
    out[1:] += a * s[:-1]
    out[:-1] += c * s[1:]
    out[0] /= (b + c)
    out[-1] /= (a + b)
    return out


def make_clips(score, t0, t1, duration, **kw):
    """위치별 점수(클수록 요청에 맞음), 위치의 시작·끝 시각(초), 영상 길이 → 클립 목록(순위 순).

    1) 앞뒤 평균. 2) 점수가 높은 위치부터 정점으로 삼는다. 3) 정점에서 양옆으로, 점수가 영상 안 위 (1 − grow_q)에 드는 위치를 따라 넓힌다
    (bridge칸 이하로 끊긴 곳은 잇는다). 4) 앞에 pre초, 뒤에 post초를 붙인다. min_len보다 짧으면 늘리고, 한 대목이 hard_max를 넘으면 줄인다.
    5) 이미 있는 클립과 min_gap초보다 가까우면 잇는다. 이으면 max_len을 넘을 때는 같은 대목으로 보고 새 클립을 만들지 않는다.
    6) 길이의 합이 budget × 영상 길이에 닿거나 max_clips에 닿으면 멈춘다.
    """
    p = {**DEFAULTS, **kw}
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    s = smooth(score) if p["smooth"] else np.asarray(score, float).copy()
    n = len(s)
    thr = float(np.quantile(s, p["grow_q"]))
    warm = s >= thr
    filled = warm.copy()
    i = 0
    while i < n:                                    # 끊긴 곳 잇기
        if warm[i]:
            i += 1
            continue
        j = i
        while j < n and not warm[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= p["bridge"]:
            filled[i:j] = True
        i = j
    center = 0.5 * (t0 + t1)
    taken = np.zeros(n, bool)
    clips = []
    budget = p["budget"] * duration
    for pk in np.argsort(-s, kind="stable"):
        if s[pk] < thr or (p["max_clips"] and len(clips) >= p["max_clips"]) or sum(c["end"] - c["start"] for c in clips) >= budget:
            break
        if taken[pk]:
            continue
        a = b = int(pk)
        while a > 0 and filled[a - 1] and not taken[a - 1]:
            a -= 1
        while b < n - 1 and filled[b + 1] and not taken[b + 1]:
            b += 1
        core_max = max(p["hard_max"] - p["pre"] - p["post"], t1[pk] - t0[pk])
        while t1[b] - t0[a] > core_max:             # 너무 길면 정점에서 먼 쪽부터 줄인다
            if (pk - a) >= (b - pk):
                a += 1
            else:
                b -= 1
        start, end = t0[a] - p["pre"], t1[b] + p["post"]
        short = p["min_len"] - (end - start)
        if short > 0:
            start, end = start - short / 2, end + short / 2
        new = {"start": max(0.0, float(start)), "end": min(float(duration), float(end)), "peak": float(s[pk]), "peak_t": float(center[pk]), "a": a, "b": b}
        skip, again, merged = False, True, []
        while again and not skip:                   # 가까운 클립과는 잇는다. 이을 수 없으면 같은 대목이므로 새로 만들지 않는다
            again = False
            for c in clips:
                if max(c["start"] - new["end"], new["start"] - c["end"]) <= p["min_gap"]:
                    u0, u1 = min(c["start"], new["start"]), max(c["end"], new["end"])
                    if u1 - u0 <= p["max_len"]:
                        top = c if c["peak"] >= new["peak"] else new
                        new = {"start": u0, "end": u1, "peak": top["peak"], "peak_t": top["peak_t"], "a": min(c["a"], new["a"]), "b": max(c["b"], new["b"])}
                        clips.remove(c)
                        merged.append(c)
                        again = True
                    else:
                        skip = True
                    break
        taken[a:b + 1] = True
        if skip:
            clips.extend(merged)                    # 2026-11-16 고침: 한 클립과 이은 뒤 다른 클립과 잇지 못해 그만둘 때, 이미 있던 클립까지 사라지던 것을 되돌린다
            continue
        clips.append(new)
        taken |= (center >= new["start"]) & (center <= new["end"])
    clips.sort(key=lambda c: -c["peak"])
    for r, c in enumerate(clips, 1):
        inside = (center >= c["start"]) & (center <= c["end"])
        c.update({"rank": r, "start": round(c["start"], 2), "end": round(c["end"], 2), "length": round(c["end"] - c["start"], 2),
                  "peak": round(c["peak"], 4), "peak_t": round(c["peak_t"], 2), "mean": round(float(s[inside].mean()), 4) if inside.any() else None})
    return clips


def round_robin(lists, n):
    """클립 목록 여럿에서 차례로 하나씩 n개. 이미 뽑은 것과 절반 넘게 겹치면 그 목록의 다음 것으로 넘어간다 (README 19·20·22절).
    뽑힌 클립에는 source(몇 번째 목록에서 왔는지)를 단다."""
    out, idx = [], [0] * len(lists)
    while len(out) < n and any(i < len(l) for i, l in zip(idx, lists)):
        for li, l in enumerate(lists):
            while idx[li] < len(l):
                c = l[idx[li]]
                idx[li] += 1
                if all(min(c["end"], o["end"]) - max(c["start"], o["start"]) <= 0.5 * min(c["length"], o["length"]) for o in out):
                    out.append(dict(c, source=li))
                    break
            if len(out) >= n:
                break
    return out


def covered(clips, t0, t1):
    """클립에 가운데 시각이 든 위치의 표시."""
    center = 0.5 * (np.asarray(t0, float) + np.asarray(t1, float))
    m = np.zeros(len(center), bool)
    for c in clips:
        m |= (center >= c["start"]) & (center <= c["end"])
    return m


def why(clip, ranks, t0, t1, top=2):
    """그 클립 안에서 순위 평균이 가장 높은 질문 top개 → [(질문, 클립 안 순위 평균)]."""
    center = 0.5 * (np.asarray(t0, float) + np.asarray(t1, float))
    inside = (center >= clip["start"]) & (center <= clip["end"])
    if not inside.any():
        return []
    got = sorted(((k, float(r[inside].mean())) for k, r in ranks.items()), key=lambda x: -x[1])
    return [(k, round(v, 3)) for k, v in got[:top]]
