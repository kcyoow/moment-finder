"""highlight_auto 따로 계산 (이 Mac, CPU). results/eval.json의 수를 다른 코드로 다시 내서 견준다.

ha_eval.py, ha_prepare.py, hh_eval.py, hh_prepare.py를 불러 쓰지 않는다. scipy도 쓰지 않는다.
순위, Kendall tau-b(분할표에서 셈), Spearman, 평균 정밀도, AUC(짝을 직접 셈), 다시 뽑기를 여기서 직접 짰다.
다시 뽑기는 같은 씨앗·같은 순서(영상 이름 순, numpy default_rng의 integers)를 써서 구간까지 같은 값이 나와야 한다.

읽는 것: data/tvsum_src의 anno.tsv(메모리에서만), data/tvsum_items.json, data/requests.json, data/tvsum_baseline_scores.json,
        results/*.json, ../highlight_h2o의 data/items.json·results/full.json, ../frame_limit/data/mc_events.json,
        ../mc_dense/results/sep4_half.json
쓰는 것: 없음. 표준 출력에 지표만 찍는다. 사람 점수 값은 찍지 않는다.

실행: uv run --offline --no-project --with numpy python -I experiments/highlight_auto/ha_verify.py
"""
import json
import math
from pathlib import Path

import numpy as np

HA = Path(__file__).resolve().parent
ROOT = HA.parents[1]
HH = ROOT / "experiments" / "highlight_h2o"
SEED, B = 20261112, 2000
TOP_METRICS = ("lift", "top_mean", "hit1", "hit5", "ap15")


def J(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 직접 짠 통계
def midrank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(x))
    for s, e in zip(starts, ends):
        r[order[s:e]] = (s + 1 + e) / 2.0
    return r


def pct(x):
    return (midrank(x) - 0.5) / len(x)


def tau_b(x, y):
    """분할표에서 바로 세는 Kendall tau-b."""
    xu, xi = np.unique(x, return_inverse=True)
    yu, yi = np.unique(y, return_inverse=True)
    T = np.zeros((len(xu), len(yu)), dtype=np.int64)
    np.add.at(T, (xi, yi), 1)
    n = int(T.sum())
    below = np.cumsum(T[::-1], axis=0)[::-1] - T          # x가 더 큰 줄들의 합
    cs = np.cumsum(below, axis=1)
    less = cs - below                                       # y가 더 작은 칸
    greater = cs[:, -1:] - cs                               # y가 더 큰 칸
    C, D = int((T * greater).sum()), int((T * less).sum())
    n0 = n * (n - 1) // 2
    rs, cs_ = T.sum(1), T.sum(0)
    n1 = int((rs * (rs - 1) // 2).sum())
    n2 = int((cs_ * (cs_ - 1) // 2).sum())
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return (C - D) / den if den > 0 else float("nan")


def spearman(x, y):
    rx, ry = midrank(x), midrank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def topk_mean(score, value, k):
    """점수 위 k곳의 value 평균. 경계의 같은 점수는 무작위로 고르는 기댓값."""
    score, value = np.asarray(score, float), np.asarray(value, float)
    kth = np.sort(score)[::-1][k - 1]
    above, tie = score > kth, score == kth
    return float((value[above].sum() + value[tie].mean() * (k - int(above.sum()))) / k)


def avg_prec(score, y):
    """평균 정밀도. 같은 점수는 한 묶음으로 보고 묶음 끝에서 잰다."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order].astype(float)
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(yy)[ends]
    rec, prec = tp / yy.sum(), tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def k15(n):
    return (15 * n + 99) // 100                    # ⌈0.15 × n⌉을 정수로


def top_all(score, value):
    score, value = np.asarray(score, float), np.asarray(value, float)
    k = k15(len(value))
    H = value >= np.sort(value)[::-1][k - 1]
    top = topk_mean(score, value, k)
    return {"lift": top - float(value.mean()), "top_mean": top, "hit1": topk_mean(score, H.astype(float), 1),
            "hit5": topk_mean(score, H.astype(float), 5), "ap15": avg_prec(score, H)}


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.shape[0] * neg.shape[1]))


def boot(rows, combine=None, seed=SEED, b=B):
    """영상째 다시 뽑기. rows: {이름: 영상 순서의 값 배열}."""
    rng = np.random.default_rng(seed)
    names = list(rows)
    M = np.array([rows[n] for n in names], float)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, M.shape[1], M.shape[1])
        means = dict(zip(names, M[:, pick].mean(1)))
        vals.append(combine(means) if combine else means[names[0]])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


CHECKS = {"same": 0, "diff": 0}


def show(name, mine, theirs, tol=6e-5, quiet=False):
    ok = all(abs(a - b) <= tol for a, b in zip(np.ravel(mine), np.ravel(theirs)))
    CHECKS["same" if ok else "diff"] += 1
    if not ok or not quiet:
        print(f"  {'같음' if ok else '다름 !!'}  {name}: 따로 {np.round(np.ravel(mine), 4).tolist()} / eval {np.ravel(theirs).tolist()}")
    return ok


ev = J(HA / "results" / "eval.json")

# ================================================================== 정상 확인
print("== 정상 확인 ==")
got = {r["id"]: r["probs"] for r in J(HA / "results" / "sanity.json")["records"] if "error" not in r}
old = {f"{r['vid']}@{r['k']}": r["probs"] for r in J(ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"] if "error" not in r}
pairs = [(got[i][q], old[i][q]) for i in got if i in old for q in got[i]]
d = np.array([abs(a - b) for a, b in pairs])
cross = sum((a >= 0.5) != (b >= 0.5) for a, b in pairs)
print(f"  쌍 {len(pairs)}, 평균 차이 {d.mean():.4f}, 가장 큰 차이 {d.max():.4f}, 0.5를 넘나든 것 {cross}")
s = ev["sanity"]
show("정상 확인 (쌍, 평균 차이, 넘나든 것)", [len(pairs), round(float(d.mean()), 4), cross], [s["pairs"], s["mean_abs_diff"], s["crossed_0_5"]])

# ================================================================== TVSum
print("== TVSum 자료 ==")
anno_path = next((HA / "data" / "tvsum_src").rglob("ydata-tvsum50-anno.tsv"))
anno = {}
for line in open(anno_path, encoding="utf-8"):
    if not line.strip():
        continue
    vid, _cat, sc = line.rstrip("\n").split("\t")
    anno.setdefault(vid, []).append(np.array(sc.split(","), dtype=np.int16))
items = J(HA / "data" / "tvsum_items.json")["videos"]
videos = sorted(items)
req = J(HA / "data" / "requests.json")
Y, LENS, A, GT = {}, {}, {}, {}
n_pos = varied = 0
for v in videos:
    y = np.stack(anno[v])
    pos = items[v]["positions"]
    assert y.shape[0] == 20, (v, y.shape)
    assert pos[0]["start"] == 0 and all(a["end"] == b["start"] for a, b in zip(pos, pos[1:])) and pos[-1]["end"] == y.shape[1], v
    lens = np.array([p["end"] - p["start"] for p in pos])
    a = np.stack([[y[i, p["start"]:p["end"]].mean() for p in pos] for i in range(20)])
    varied += sum(int(y[i, p["start"]:p["end"]].min() != y[i, p["start"]:p["end"]].max()) for p in pos for i in range(20))
    Y[v], LENS[v], A[v], GT[v] = y, lens, a, a.mean(0)
    n_pos += len(pos)
print(f"  영상 {len(videos)}, 위치 {n_pos}, 평가자 20명씩. 한 위치 안에서 점수가 바뀌는 (평가자, 위치) 쌍 {varied} / {n_pos * 20}")
k_float = [v for v in videos if math.ceil(0.15 * len(items[v]["positions"])) != k15(len(items[v]["positions"]))]
print(f"  ha_eval.py의 math.ceil(0.15 * n)이 정수 셈과 다른 영상: {len(k_float)}개")

recs = J(HA / "results" / "tvsum.json")["records"]
errs = [r for r in recs if "error" in r]
keys = {v: [k for k in req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in videos}
types = {v: {k: q["type"] for k, q in req["tvsum"]["videos"][v]["questions"].items()} for v in videos}
vals = {v: {} for v in videos}
bad_score = 0
for r in recs:
    if "error" in r:
        continue
    v, out = r["video"], {}
    for k in ["worth"] + keys[v]:
        a = r["answers"][k]
        if types[v][k] == "noul":
            out[k] = a["noul"]
        else:
            pr = a["probabilities"]
            exp = sum(int(lv) * p for lv, p in pr.items())
            if abs(exp - a["score"]) > 0.01 or abs(sum(pr.values()) - 1) > 0.01:
                bad_score += 1
            out[k] = a["score"]
    vals[v][r["k"]] = out
missing = sum(len(items[v]["positions"]) - len(vals[v]) for v in videos)
print(f"  답 {len(recs)}건, 오류 {len(errs)}, 빠진 위치 {missing}, 기댓값이 확률과 어긋난 score 답 {bad_score}, 글쓴이 질문 {sum(len(k) for k in keys.values())}개")

base = J(HA / "data" / "tvsum_baseline_scores.json")
ARMS = ("ga", "na", "da", "position", "change", "loudness")
ALLM = ("tau", "rho", *TOP_METRICS)
PV = {a_: {m: [] for m in ALLM} for a_ in ARMS}
QT = {}
for v in videos:
    n = len(items[v]["positions"])
    col = lambda k: np.array([vals[v][i][k] for i in range(n)], float)  # noqa: E731
    qr = {k: pct(col(k)) for k in keys[v]}
    ga = pct(col("worth"))
    na = np.mean([qr[k] for k in keys[v]], axis=0)
    da = 0.5 * ga + 0.5 * pct(na)
    sc = {"ga": ga, "na": na, "da": da, **{b_: np.array(base[b_][v], float) for b_ in ("position", "change", "loudness")}}
    for a_ in ARMS:
        x = np.repeat(sc[a_], LENS[v])
        PV[a_]["tau"].append(np.mean([tau_b(x, Y[v][i]) for i in range(20)]))
        PV[a_]["rho"].append(np.mean([spearman(x, Y[v][i]) for i in range(20)]))
        t = top_all(sc[a_], GT[v])
        for m in TOP_METRICS:
            PV[a_][m].append(t[m])
    QT[v] = {k: float(np.mean([tau_b(np.repeat(qr[k], LENS[v]), Y[v][i]) for i in range(20)])) for k in keys[v]}

print("== TVSum 팔: 지표 일곱 가지의 평균과 구간 ==")
for a_ in ARMS:
    e = ev["tvsum"]["arms"][a_]
    for m in ALLM:
        show(f"{a_} {m}", [np.mean(PV[a_][m]), *boot({m: PV[a_][m]})], [e[m]["mean"], *e[m]["ci"]], quiet=m not in ("tau", "lift"))

print("== TVSum 범주별 평균 (τ, ρ, 올림 값), 질문 하나하나의 τ ==")
cats = sorted({items[v]["category"] for v in videos})
for a_ in ARMS:
    for c in cats:
        idx = [i for i, v in enumerate(videos) if items[v]["category"] == c]
        show(f"{a_} {c}", [np.mean([PV[a_][m][i] for i in idx]) for m in ("tau", "rho", "lift")],
             [ev["tvsum"]["arms"][a_]["by_category"][c][m] for m in ("tau", "rho", "lift")], tol=6e-4, quiet=True)
qd = [abs(QT[v][k] - ev["tvsum"]["per_question_tau"][v][k]) for v in videos for k in keys[v]]
qall = [QT[v][k] for v in videos for k in keys[v]]
print(f"  질문 {len(qd)}개의 τ: eval과의 가장 큰 차이 {max(qd):.5f} (eval은 소수 셋째 자리). 평균 {np.mean(qall):.4f}, 가운데 값 {np.median(qall):.4f}, 0보다 작은 것 {sum(x < 0 for x in qall)}개")
CHECKS["same" if max(qd) <= 6e-4 else "diff"] += 1

print("== TVSum 비교 1, 2, 3 ==")
for m in ("tau", "lift"):
    g = {a_: np.array(PV[a_][m]) for a_ in ARMS}
    c = ev["tvsum"]["comparisons"]
    show(f"1 (나)-(가) {m}", [np.mean(g["na"] - g["ga"]), *boot({"d": g["na"] - g["ga"]})], [c[f"1_na_minus_ga_{m}"]["point"], *c[f"1_na_minus_ga_{m}"]["ci"]])
    comb = lambda mm: mm["da"] - max(mm["ga"], mm["na"])  # noqa: E731
    show(f"2 (다)-나은 것 {m}", [g["da"].mean() - max(g["ga"].mean(), g["na"].mean()), *boot({"da": g["da"], "ga": g["ga"], "na": g["na"]}, comb)],
         [c[f"2_da_minus_better_{m}"]["point"], *c[f"2_da_minus_better_{m}"]["ci"]])
    for a_ in ("ga", "na", "da"):
        for b_ in ("position", "change", "loudness"):
            show(f"3 {a_}-{b_} {m}", [np.mean(g[a_] - g[b_]), *boot({"d": g[a_] - g[b_]})],
                 [c[f"3_{a_}_minus_{b_}_{m}"]["point"], *c[f"3_{a_}_minus_{b_}_{m}"]["ci"]], quiet=True)

print("== 씨앗을 바꿔도 주 비교의 갈래가 같은가 (참고. 판정은 씨앗 20261112, 2,000번) ==")
for m in ("tau", "lift"):
    g = {a_: np.array(PV[a_][m]) for a_ in ARMS}
    for sd in (1, 2, 3):
        l1 = boot({"d": g["na"] - g["ga"]}, seed=sd, b=10000)
        l2 = boot({"da": g["da"], "ga": g["ga"], "na": g["na"]}, lambda mm: mm["da"] - max(mm["ga"], mm["na"]), seed=sd, b=10000)
        print(f"  {m} 씨앗 {sd} (10,000번): 비교 1 [{l1[0]:.4f}, {l1[1]:.4f}]  비교 2 [{l2[0]:.4f}, {l2[1]:.4f}]")

print("== 사람 기준값 ==")
ht, hr = [], []
HT = {m: [] for m in TOP_METRICS}
OR = {m: [] for m in TOP_METRICS}
for v in videos:
    y = Y[v]
    T = np.full((20, 20), np.nan)
    R = np.full((20, 20), np.nan)
    for i in range(20):
        for j in range(i + 1, 20):
            T[i, j] = T[j, i] = tau_b(y[i], y[j])
            R[i, j] = R[j, i] = spearman(y[i], y[j])
    ht.append(np.mean([np.nanmean(T[i]) for i in range(20)]))
    hr.append(np.mean([np.nanmean(R[i]) for i in range(20)]))
    per = [top_all(A[v][i], np.delete(A[v], i, axis=0).mean(0)) for i in range(20)]
    orc = top_all(GT[v], GT[v])
    for m in TOP_METRICS:
        HT[m].append(np.mean([p[m] for p in per]))
        OR[m].append(orc[m])
hrf = ev["tvsum"]["human_reference"]
show("사람 tau, rho (평가자 짝마다의 상관 평균)", [np.mean(ht), np.mean(hr)], [hrf["human_tau"], hrf["human_rho"]])
for m in TOP_METRICS:
    show(f"사람 {m} (한 명으로 고르고 다른 19명으로 잼)", [np.mean(HT[m]), *boot({m: HT[m]})], [hrf["human_top"][m]["mean"], *hrf["human_top"][m]["ci"]], quiet=m != "lift")
    show(f"정답으로 고른 상한 {m}", [np.mean(OR[m])], [hrf["oracle_top"][m]], quiet=m != "lift")

# ================================================================== 마인크래프트
print("== 마인크래프트 ==")
mitems = J(HH / "data" / "items.json")["full"]
events = J(ROOT / "experiments" / "frame_limit" / "data" / "mc_events.json")["events"]
byv = {}
for it in mitems:
    byv.setdefault(it["video"], []).append(it)
att = {it["id"]: [] for it in mitems}
unmatched = 0
for e in events:
    if e["kind"] in ("kill", "tool_break"):
        continue
    best, bd = None, None
    for it in byv.get(e["video"], []):
        dd = abs(it["t"] - e["t"])
        if dd <= 1.5 and (bd is None or dd < bd):
            best, bd = it, dd
    if best is None:
        unmatched += 1
    else:
        att[best["id"]].append(e)
lab = {}
for i, evs in att.items():
    fight = any(e["kind"] in ("fight", "hurt") for e in evs)
    prog = any((e["kind"] == "crafting" and e.get("level") in (2, 3)) or e["kind"] == "find" for e in evs)
    lab[i] = {"fight": fight, "prog": prog, "big": fight or prog}
print(f"  위치 {len(mitems)}, 영상 {len(byv)}, 큰 사건 {sum(x['big'] for x in lab.values())}곳 "
      f"(싸움·피해 {sum(x['fight'] for x in lab.values())}, 제작 2단계 이상·다이아몬드 {sum(x['prog'] for x in lab.values())}), 붙지 못한 사건 {unmatched}")

stored = {r["id"]: r for r in J(HH / "results" / "full.json")["records"] if "error" not in r}
new = {r["id"]: r["answers"] for r in J(HA / "results" / "minecraft.json")["records"] if "error" not in r}
mkeys = list(req["minecraft"]["questions"])
mtypes = {k: q["type"] for k, q in req["minecraft"]["questions"].items()}
print(f"  받아 둔 답 {len(stored)}, 새 답 {len(new)}, 글쓴이 질문 {mkeys}")
MARMS = ("hand", "vague", "na", "da")
S = {a_: {} for a_ in (*MARMS, *[f"q_{k}" for k in mkeys])}
RAW = {"final": {}, "worth": {}}
for v, its in byv.items():
    ids = [it["id"] for it in its]
    final = np.array([0.5 * stored[i]["action"]["score"] / 3 + 0.5 * stored[i]["progress"]["score"] / 3 for i in ids])
    worth = np.array([stored[i]["worth"] for i in ids], float)
    qr = {k: pct(np.array([new[i][k]["noul"] if mtypes[k] == "noul" else new[i][k]["score"] for i in ids], float)) for k in mkeys}
    na = pct(np.mean([qr[k] for k in mkeys], axis=0))
    rw = pct(worth)
    for n_, arr in (("hand", pct(final)), ("vague", rw), ("na", na), ("da", pct(0.5 * rw + 0.5 * na)), *[(f"q_{k}", qr[k]) for k in mkeys]):
        S[n_][v] = arr
    RAW["final"][v], RAW["worth"][v] = final, worth
mv = sorted(byv)
LAB = {kind: {v: np.array([lab[it["id"]][kind] for it in byv[v]], bool) for v in mv} for kind in ("big", "fight", "prog")}
BIG = LAB["big"]


def pooled(table, kind="big"):
    s = np.concatenate([table[v] for v in mv])
    y = np.concatenate([LAB[kind][v] for v in mv])
    return s, y


def pooled_auc(table, kind="big"):
    s, y = pooled(table, kind)
    return auc_pair(s[y], s[~y])


me = ev["minecraft"]
show("받아 둔 원래 값의 AUC (final, worth)", [pooled_auc(RAW["final"]), pooled_auc(RAW["worth"])], [me["raw_stored_auc"]["final"], me["raw_stored_auc"]["worth"]])
AUC = {a_: pooled_auc(S[a_]) for a_ in MARMS}
for a_ in MARMS:
    s, y = pooled(S[a_])
    hit = picked_n = 0
    for v in mv:                                   # 영상별 상위 3곳, 같은 3초 안은 한 번만
        order = np.argsort(-S[a_][v], kind="stable")
        picked = []
        for i in order:
            if all(abs(byv[v][i]["t"] - byv[v][p]["t"]) > 3.0 for p in picked):
                picked.append(i)
            if len(picked) == 3:
                break
        hit += int(BIG[v][picked].sum())
        picked_n += len(picked)
    a = me["arms"][a_]
    show(f"{a_}: AUC, AP, 영상별 상위 3곳의 큰 사건(고른 수), 위 15곳, 위 36곳",
         [AUC[a_], avg_prec(s, y), hit, picked_n, topk_mean(s, y.astype(float), 15) * 15, topk_mean(s, y.astype(float), 36) * 36],
         [a["auc_big"], a["ap_big"], a["top3"]["big_in_top"], a["top3"]["picked"], a["pooled_top15_big"], a["pooled_top36_big"]], tol=6e-4)
for k in mkeys:
    show(f"질문 {k} AUC (큰 사건, 싸움·피해, 제작·다이아몬드)", [pooled_auc(S[f"q_{k}"], kind) for kind in ("big", "fight", "prog")],
         [me["per_question_auc"][k][kind] for kind in ("big", "fight", "prog")], tol=6e-4)

# 영상째 다시 뽑기: 영상 짝마다 이긴 수를 미리 세고, 뽑힌 횟수로 합친다(ha_eval과 다른 길)
W = {a_: np.array([[(lambda p, q: (p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum())(S[a_][vi][BIG[vi]], S[a_][vj][~BIG[vj]])
                    for vj in mv] for vi in mv], float) for a_ in MARMS}
NP = np.array([BIG[v].sum() for v in mv], float)
NN = np.array([(~BIG[v]).sum() for v in mv], float)
rng = np.random.default_rng(SEED)
D = {"M1": [], "M2": [], "M3": [], **{a_: [] for a_ in MARMS}}
for _ in range(B):
    pick = rng.integers(0, len(mv), len(mv))
    m = np.bincount(pick, minlength=len(mv)).astype(float)
    den = (m @ NP) * (m @ NN)
    if den == 0:
        continue
    a = {a_: float(m @ W[a_] @ m) / den for a_ in MARMS}
    for a_ in MARMS:
        D[a_].append(a[a_])
    D["M1"].append(a["na"] - a["hand"])
    D["M2"].append(a["na"] - a["vague"])
    D["M3"].append(a["da"] - max(a["vague"], a["na"]))
for a_ in MARMS:
    show(f"{a_} AUC의 구간", np.percentile(D[a_], [2.5, 97.5]), me["arms"][a_]["auc_big_ci"], tol=6e-4)
pts = {"M1": AUC["na"] - AUC["hand"], "M2": AUC["na"] - AUC["vague"], "M3": AUC["da"] - max(AUC["vague"], AUC["na"])}
names = {"M1": "M1_na_minus_hand", "M2": "M2_na_minus_vague", "M3": "M3_da_minus_better"}
for k in ("M1", "M2", "M3"):
    lo, hi = np.percentile(D[k], [2.5, 97.5])
    c = me["comparisons"][names[k]]
    # eval의 점 추정은 소수 넷째 자리로 줄인 AUC끼리의 차라서 넷째 자리가 1 다를 수 있다
    show(k, [pts[k], lo, hi], [c["point"], *c["ci"]], tol=6e-4)

print(f"\n견준 묶음 {CHECKS['same'] + CHECKS['diff']}개 중 같음 {CHECKS['same']}, 다름 {CHECKS['diff']}")
