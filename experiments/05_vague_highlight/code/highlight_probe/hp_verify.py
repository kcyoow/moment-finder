"""highlight_probe 따로 계산 (이 Mac, CPU). results/eval.json의 수를 다른 코드로 다시 내서 견준다.

hp_eval.py, ha_eval.py, ha_prepare.py, hh_eval.py, hh_prepare.py를 불러 쓰지 않는다. scipy도 쓰지 않는다.
순위, Kendall tau-b(분할표에서 셈), 맨 위 지표, AUC(짝을 직접 셈), 다시 뽑기를 여기서 직접 짰다(../highlight_auto/ha_verify.py와 같은 방식).
다시 뽑기는 같은 씨앗·같은 순서를 써서 구간까지 같은 값이 나와야 한다.

쓰는 것: 없음. 표준 출력에 지표만 찍는다. 사람 점수 값은 찍지 않는다.
실행: uv run --offline --no-project --with numpy python -I experiments/highlight_probe/hp_verify.py
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HA = ROOT / "experiments" / "highlight_auto"
HH = ROOT / "experiments" / "highlight_h2o"
SEED, B = 20261112, 2000
CHECKS = {"same": 0, "diff": 0}


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


def pct_missing(c):
    """빠진 값(nan)은 0.5, 나머지는 답이 있는 것끼리의 백분위 순위."""
    c = np.asarray(c, float)
    r = np.full(len(c), 0.5)
    ok = ~np.isnan(c)
    if ok.any():
        r[ok] = pct(c[ok])
    return r


def tau_b(x, y):
    xu, xi = np.unique(x, return_inverse=True)
    yu, yi = np.unique(y, return_inverse=True)
    T = np.zeros((len(xu), len(yu)), dtype=np.int64)
    np.add.at(T, (xi, yi), 1)
    n = int(T.sum())
    below = np.cumsum(T[::-1], axis=0)[::-1] - T
    cs = np.cumsum(below, axis=1)
    less = cs - below
    greater = cs[:, -1:] - cs
    C, D = int((T * greater).sum()), int((T * less).sum())
    n0 = n * (n - 1) // 2
    rs, cs_ = T.sum(1), T.sum(0)
    n1 = int((rs * (rs - 1) // 2).sum())
    n2 = int((cs_ * (cs_ - 1) // 2).sum())
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return (C - D) / den if den > 0 else float("nan")


def topk_mean(score, value, k):
    score, value = np.asarray(score, float), np.asarray(value, float)
    kth = np.sort(score)[::-1][k - 1]
    above, tie = score > kth, score == kth
    return float((value[above].sum() + value[tie].mean() * (k - int(above.sum()))) / k)


def lift15(score, value):
    value = np.asarray(value, float)
    k = (15 * len(value) + 99) // 100
    return topk_mean(score, value, k) - float(value.mean())


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.shape[0] * neg.shape[1]))


def boot(rows, combine=None):
    rng = np.random.default_rng(SEED)
    names = list(rows)
    M = np.array([rows[n] for n in names], float)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, M.shape[1], M.shape[1])
        means = dict(zip(names, M[:, pick].mean(1)))
        vals.append(combine(means) if combine else means[names[0]])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def show(name, mine, theirs, tol=6e-5, quiet=False):
    ok = all(abs(a - b) <= tol for a, b in zip(np.ravel(mine), np.ravel(theirs)))
    CHECKS["same" if ok else "diff"] += 1
    if not ok or not quiet:
        print(f"  {'같음' if ok else '다름 !!'}  {name}: 따로 {np.round(np.ravel(mine), 4).tolist()} / eval {np.ravel(theirs).tolist()}")
    return ok


# ------------------------------------------------------------------ 자료
def load_tvsum():
    anno_path = next((HA / "data" / "tvsum_src").rglob("ydata-tvsum50-anno.tsv"))
    anno = {}
    for line in open(anno_path, encoding="utf-8"):
        if line.strip():
            vid, _cat, sc = line.rstrip("\n").split("\t")
            anno.setdefault(vid, []).append(np.array(sc.split(","), dtype=np.int16))
    items = J(HA / "data" / "tvsum_items.json")["videos"]
    videos = sorted(items)
    Y, LENS, GT = {}, {}, {}
    for v in videos:
        y = np.stack(anno[v])
        pos = items[v]["positions"]
        assert y.shape[0] == 20 and pos[-1]["end"] == y.shape[1], v
        Y[v] = y
        LENS[v] = np.array([p["end"] - p["start"] for p in pos])
        GT[v] = np.stack([[y[i, p["start"]:p["end"]].mean() for p in pos] for i in range(20)]).mean(0)
    return videos, items, Y, LENS, GT


def tv_metrics(score, videos, Y, LENS, GT):
    """{영상: 위치별 점수} → 영상 순서의 τ와 올림 값."""
    tau, lift = [], []
    for v in videos:
        x = np.repeat(score[v], LENS[v])
        tau.append(np.mean([tau_b(x, Y[v][i]) for i in range(20)]))
        lift.append(lift15(score[v], GT[v]))
    return {"tau": tau, "lift": lift}


def value_of(a):
    return a["noul"] if "noul" in a else a["score"]


def arm_score(records, keys_by_video, n_by_video, id_of):
    """답 → {영상: 질문마다의 순위(빠진 곳 0.5)의 같은 무게 평균}."""
    got = {}
    for r in records:
        if "error" not in r:
            got.setdefault(r["video"], {})[id_of(r)] = r["answers"]
    out = {}
    for v, n in n_by_video.items():
        cols = []
        for k in keys_by_video[v]:
            c = np.full(n, np.nan)
            for idx, ans in got.get(v, {}).items():
                c[idx] = value_of(ans[k])
            cols.append(pct_missing(c))
        out[v] = np.mean(cols, axis=0)
    return out


def load_minecraft():
    mitems = J(HH / "data" / "items.json")["full"]
    events = J(ROOT / "experiments" / "frame_limit" / "data" / "mc_events.json")["events"]
    byv = {}
    for it in mitems:
        byv.setdefault(it["video"], []).append(it)
    att = {it["id"]: [] for it in mitems}
    for e in events:
        if e["kind"] in ("kill", "tool_break"):
            continue
        best, bd = None, None
        for it in byv.get(e["video"], []):
            dd = abs(it["t"] - e["t"])
            if dd <= 1.5 and (bd is None or dd < bd):
                best, bd = it, dd
        if best is not None:
            att[best["id"]].append(e)
    big = {}
    for i, evs in att.items():
        fight = any(e["kind"] in ("fight", "hurt") for e in evs)
        prog = any((e["kind"] == "crafting" and e.get("level") in (2, 3)) or e["kind"] == "find" for e in evs)
        big[i] = fight or prog
    mv = sorted(byv)
    BIG = {v: np.array([big[it["id"]] for it in byv[v]], bool) for v in mv}
    stored = {r["id"]: r for r in J(HH / "results" / "full.json")["records"] if "error" not in r}
    return mv, byv, BIG, stored


def mc_stats(S, mv, BIG, diffs):
    """S: {팔: {영상: 순위}} → (AUC, AUC 구간, 차이의 구간). 영상 짝마다 이긴 수를 미리 세는 길(hh_eval.boot와 다른 길, 같은 씨앗)."""
    arms = list(S)

    def pooled(a):
        s = np.concatenate([S[a][v] for v in mv])
        y = np.concatenate([BIG[v] for v in mv])
        return auc_pair(s[y], s[~y])
    AUC = {a: pooled(a) for a in arms}
    W = {a: np.array([[(lambda p, q: (p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum())(S[a][vi][BIG[vi]], S[a][vj][~BIG[vj]])
                       for vj in mv] for vi in mv], float) for a in arms}
    NP = np.array([BIG[v].sum() for v in mv], float)
    NN = np.array([(~BIG[v]).sum() for v in mv], float)
    rng = np.random.default_rng(SEED)
    D = {**{a: [] for a in arms}, **{k: [] for k in diffs}}
    for _ in range(B):
        pick = rng.integers(0, len(mv), len(mv))
        m = np.bincount(pick, minlength=len(mv)).astype(float)
        den = (m @ NP) * (m @ NN)
        if den == 0:
            continue
        a_ = {a: float(m @ W[a] @ m) / den for a in arms}
        for a in arms:
            D[a].append(a_[a])
        for k, (x, z) in diffs.items():
            D[k].append(a_[x] - a_[z])
    CI = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in D.items()}
    return AUC, CI


def main():
    ev = J(HERE / "results" / "eval.json")
    req = J(HERE / "data" / "main_requests.json")
    old_req = J(HA / "data" / "requests.json")

    print("== 정상 확인 ==")
    old = {f"{r['vid']}@{r['k']}": r["probs"] for r in J(ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"] if "error" not in r}
    for fn in ("main_tvsum", "main_minecraft"):
        got = {r["id"]: r["probs"] for r in J(HERE / "results" / f"{fn}__sanity.json")["records"] if "error" not in r}
        pairs = [(got[i][q], old[i][q]) for i in got if i in old for q in got[i]]
        d = np.array([abs(a - b) for a, b in pairs])
        cross = sum((a >= 0.5) != (b >= 0.5) for a, b in pairs)
        s = ev["sanity"][fn]
        show(f"{fn} (쌍, 평균 차이, 넘나든 것)", [len(pairs), round(float(d.mean()), 4), cross], [s["pairs"], s["mean_abs_diff"], s["crossed_0_5"]])

    print("== TVSum ==")
    videos, items, Y, LENS, GT = load_tvsum()
    n = {v: len(items[v]["positions"]) for v in videos}
    files = {b: J(HERE / "results" / f"main_tvsum__{b}.json")["records"] for b in ("tv_titled", "tv_plain")}
    for b, recs in files.items():
        e = ev["tvsum"]["bundles"][b]
        bad = sum("error" in r for r in recs)
        show(f"{b} (답, 오류)", [len(recs) - bad, bad + sum(n.values()) - len(recs)], [e["answered"], e["errors"]])
    score = {}
    for arm, (b, prefix) in {"T": ("tv_titled", "t__"), "TP": ("tv_titled", "tp__"), "P": ("tv_plain", "")}.items():
        keys = {v: [k for k in req["arms"][b]["videos"][v]["questions"] if k.startswith(prefix)] for v in videos}
        score[arm] = arm_score(files[b], keys, n, lambda r: r["k"])
    old_recs = J(HA / "results" / "tvsum.json")["records"]
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in videos}
    score["na"] = arm_score(old_recs, keys_old, n, lambda r: r["k"])
    score["ga"] = arm_score(old_recs, {v: ["worth"] for v in videos}, n, lambda r: r["k"])
    M = {arm: tv_metrics(score[arm], videos, Y, LENS, GT) for arm in score}
    for arm in ("ga", "na", "T", "TP", "P"):
        e = ev["tvsum"]["arms"][arm]
        for m in ("tau", "lift"):
            show(f"{arm} {m} (평균, 구간)", [np.mean(M[arm][m]), *boot({m: M[arm][m]})], [e[m]["mean"], *e[m]["ci"]])
    for name, (a, b) in {"H1_TP_minus_T": ("TP", "T"), "H2_P_minus_T": ("P", "T"), "H3_P_minus_ga": ("P", "ga"), "note_T_minus_na": ("T", "na")}.items():
        for m in ("tau", "lift"):
            d = np.array(M[a][m]) - np.array(M[b][m])
            c = ev["tvsum"]["comparisons"][f"{name}_{m}"]
            show(f"{name} {m} (차이, 구간)", [d.mean(), *boot({"d": d})], [c["point"], *c["ci"]])

    print("== 마인크래프트 ==")
    mv, byv, BIG, stored = load_minecraft()
    print(f"  위치 {sum(len(x) for x in byv.values())}, 영상 {len(mv)}, 큰 사건 {int(sum(b.sum() for b in BIG.values()))}곳")
    idx = {it["id"]: (it["video"], i) for v in mv for i, it in enumerate(byv[v])}
    n_mc = {v: len(byv[v]) for v in mv}
    mfiles = {b: J(HERE / "results" / f"main_minecraft__{b}.json")["records"] for b in ("mc_game", "mc_plain")}
    for b, recs in mfiles.items():
        e = ev["minecraft"]["bundles"][b]
        bad = sum("error" in r for r in recs)
        show(f"{b} (답, 오류)", [len(recs) - bad, bad + 3600 - len(recs)], [e["answered"], e["errors"]])
    S = {}
    for arm, (b, prefix) in {"G": ("mc_game", "g__"), "GP": ("mc_game", "gp__"), "P": ("mc_plain", "")}.items():
        keys = {v: [k for k in req["arms"][b]["videos"][v]["questions"] if k.startswith(prefix)] for v in mv}
        raw = arm_score(mfiles[b], keys, n_mc, lambda r: idx[r["id"]][1])
        S[arm] = {v: pct(raw[v]) for v in mv}
    na_raw = arm_score(J(HA / "results" / "minecraft.json")["records"], {v: list(old_req["minecraft"]["questions"]) for v in mv}, n_mc, lambda r: idx[r["id"]][1])
    S["na"] = {v: pct(na_raw[v]) for v in mv}
    S["hand"] = {v: pct(np.array([0.5 * stored[it["id"]]["action"]["score"] / 3 + 0.5 * stored[it["id"]]["progress"]["score"] / 3 for it in byv[v]])) for v in mv}
    diffs = {"M4_GP_minus_na": ("GP", "na"), "M5_P_minus_na": ("P", "na"), "M6_GP_minus_hand": ("GP", "hand"), "note_G_minus_na": ("G", "na")}
    AUC, CI = mc_stats(S, mv, BIG, diffs)
    for arm in ("hand", "na", "G", "GP", "P"):
        e = ev["minecraft"]["arms"][arm]
        show(f"{arm} AUC, 구간", [AUC[arm], *CI[arm]], [e["auc_big"], *e["auc_big_ci"]], tol=6e-4)
    for k, (x, z) in diffs.items():
        c = ev["minecraft"]["comparisons"][k]
        show(f"{k} (차이, 구간)", [AUC[x] - AUC[z], *CI[k]], [c["point"], *c["ci"]], tol=6e-4)
    print(f"\n견준 묶음 {CHECKS['same'] + CHECKS['diff']}개 중 같음 {CHECKS['same']}, 다름 {CHECKS['diff']}")


if __name__ == "__main__":
    main()
