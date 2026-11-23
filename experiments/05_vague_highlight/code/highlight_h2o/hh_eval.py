"""highlight_h2o 채점. 판정 모델 요청은 없다(이 Mac, CPU만). 규칙은 README.md "읽는 규칙"과 같고, 결과를 보기 전에 정했다.

방식
  jev        : 받아 둔 TypeSafe Jev 답(frame_limit/results/hl_mc_score.json)을 같은 코드로 채점한다.
               원래 보고서의 AUC 넷(0.995, 0.886, 0.822, 0.593)이 나와야 한다. 덧붙여 영상째 다시 뽑기 구간과
               한 영상 빼고 로지스틱 회귀를 낸다("TypeSafe Jev로 잰 값") → results/jev_eval.json
  baselines  : 판정 모델 없이 공짜로 세는 기준선(영상 안의 위치, 프레임 사이 변화). 두 묶음 모두 → results/baselines.json
  h2o        : (GPU 실행 뒤) 판정 모델(H2O)의 답 results/same434.json, sanity.json, full.json을 채점 → results/h2o_eval.json

정답(hl_eval.py 그대로): 사건을 앞뒤 1.5초 안의 가장 가까운 위치에 붙인다(kill·tool_break는 빼고). 같은 434곳에서는 후보에만 붙고 견줄 곳 48곳은
모두 "나머지"다. 영상 전체 1초마다에서는 모든 위치에 붙는다.
  action의 정답 = 싸움·피해, progress의 정답 = 제작 2단계 이상·다이아몬드, 최종 점수와 막연한 질문의 정답 = 큰 사건(둘 중 하나)
  최종 점수 = 0.5 × action/3 + 0.5 × progress/3 (원래 식 그대로)

실행: uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_h2o/hh_eval.py jev
      uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_h2o/hh_eval.py baselines
      uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_h2o/hh_eval.py h2o
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hh_prepare as P  # noqa: E402

JEV_REPORTED = {"action": 0.995, "progress": 0.886, "final": 0.822, "worth": 0.593}   # REPORT_highlight.md 2절의 표
SEED, RESAMPLES = 20261112, 2000
CUTS = {"action": 1.5, "progress": 1.5, "final": 0.5, "worth": 0.5}   # 가운데 값. 결과를 보기 전에 정함
TOP, DEDUP = 3, 3.0                                                     # 영상별 상위 3곳, 같은 3초 안은 한 번만(원래 그대로)
SANITY_MAX_MEAN, SANITY_MAX_CROSS = 0.010, 8                            # h2o_startup README: 같은 설정 다시 묻기 800개 중 3~7개, 평균 0.007; 가장 큰 값 8개, 0.009
MAX_ERROR_SHARE = 0.02
TARGET = {"action": "fight", "progress": "prog", "final": "big", "worth": "big"}
KIND_KEYS = ["fight", "hurt", "crafting", "mining", "moving"]           # 회귀에서 other는 뺀다(합이 1이라서)


# ---- 지표
def midranks(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(scores, labels):
    """맞는 쪽이 이기면 1, 같으면 0.5(hl_eval.py의 auc와 같은 값)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = midranks(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(scores, labels):
    """평균 정밀도. 같은 점수는 한 기준으로 묶는다."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.sum() == 0:
        return None
    total, prev_r, out = y.sum(), 0.0, 0.0
    for thr in np.unique(s)[::-1]:
        m = s >= thr
        tp = (y & m).sum()
        rec, prec = tp / total, tp / m.sum()
        out += (rec - prev_r) * prec
        prev_r = rec
    return float(out)


def r3(x):
    return None if x is None else round(x, 3)


# ---- 자료
def events():
    return P.read(P.FL / "data" / "mc_events.json")["events"]


def finish(recs, arm, all_items=None):
    """recs: 답이 있는 위치들(같은 묶음 순서). 사건을 붙이고 정답과 최종 점수를 단다.
    all_items를 주면 사건은 답이 없는(오류난) 위치까지 포함한 묶음 전체에 붙인다(오류가 정답을 옮기지 않게)."""
    eligible = (lambda r: r["role"] != "control") if arm == "same434" else (lambda r: True)
    att, unmatched = P.attach(all_items or recs, events(), eligible)
    for r in recs:
        r["events"] = att[r["id"]]
        r.update(P.label(r["events"]))
        r["final"] = 0.5 * r["action"]["score"] / 3 + 0.5 * r["progress"]["score"] / 3
    return unmatched


def jev_records():
    out = []
    for r in P.read(P.FL / "results" / "hl_mc_score.json")["records"]:
        if r["error"]:
            continue
        out.append({"id": f"{r['video'][:5]}@{r['t']:g}", "video": r["video"], "t": r["t"], "role": r["role"],
                    "action": r["action"], "progress": r["progress"], "kind": r["kind"], "worth": r["worth"]})
    return out


def signal(r, name):
    return r["final"] if name == "final" else r["worth"] if name == "worth" else r[name]["score"]


# ---- 표들
def four_aucs(recs):
    return {name: auc([signal(r, name) for r in recs], [r[TARGET[name]] for r in recs]) for name in TARGET}


def groups(recs):
    cand = [r for r in recs if r["role"] != "control"]
    has = P.has
    big = lambda r: r["big"]
    g = {"싸움 (준 피해)": [r for r in cand if has(r["events"], "fight")], "피해 입음": [r for r in cand if has(r["events"], "hurt")],
         "제작 3단계 (다이아몬드 도구)": [r for r in cand if has(r["events"], "crafting", 3)],
         "제작 2단계 (작업대·화로·도구)": [r for r in cand if has(r["events"], "crafting", 2) and not has(r["events"], "crafting", 3)],
         "제작 1단계 (판자·막대 등)": [r for r in cand if has(r["events"], "crafting", 1) and not big(r)],
         "다이아몬드 주움": [r for r in cand if has(r["events"], "find")],
         "광석 캠": [r for r in cand if has(r["events"], "ore") and not big(r)],
         "사건이 안 붙은 후보": [r for r in cand if not r["events"]],
         "견줄 곳 (아무 일 없는 곳)": [r for r in recs if r["role"] == "control"]}
    m = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    return {k: {"n": len(rs), "action": m([r["action"]["score"] for r in rs]), "progress": m([r["progress"]["score"] for r in rs]),
                "final": m([r["final"] for r in rs]), "worth": m([r["worth"] for r in rs])} for k, rs in g.items()}


def top3(recs, key):
    """원래 규칙: 영상마다 점수 순으로, 이미 고른 곳과 3초 넘게 떨어진 곳만 3곳. 견줄 곳은 고르지 않는다."""
    cand = [r for r in recs if r["role"] != "control"]
    hit = n = 0
    per = {}
    for video in sorted({r["video"] for r in cand}):
        rs = sorted((r for r in cand if r["video"] == video), key=lambda r: -key(r))
        picked = []
        for r in rs:
            if all(abs(r["t"] - p["t"]) > DEDUP for p in picked):
                picked.append(r)
            if len(picked) == TOP:
                break
        hit += sum(r["big"] for r in picked)
        n += len(picked)
        per[video] = [[r["t"], round(key(r), 3), bool(r["big"])] for r in picked]
    base = sum(r["big"] for r in cand) / len(cand)
    return {"big_in_top": hit, "picked": n, "share": round(hit / n, 3), "base_share": round(base, 3), "per_video": per}


def thresholds(recs):
    out = {}
    for name, cut in CUTS.items():
        s = np.array([signal(r, name) for r in recs])
        y = np.array([r[TARGET[name]] for r in recs])
        m = s >= cut
        out[name] = {"cut": cut, "label": TARGET[name], "flagged": int(m.sum()), "hits": int((m & y).sum()), "positives": int(y.sum()),
                     "precision": r3((m & y).sum() / m.sum()) if m.sum() else None, "recall": r3((m & y).sum() / y.sum()) if y.sum() else None}
    n = out["worth"]["flagged"]
    if n:
        order = sorted(recs, key=lambda r: -r["final"])[:n]
        out["final_top_n_at_worth_count"] = {"n": n, "hits": sum(r["big"] for r in order), "precision": r3(sum(r["big"] for r in order) / n)}
    return out


def by_video(recs, fn):
    vids = sorted({r["video"] for r in recs})
    return vids, {v: [r for r in recs if r["video"] == v] for v in vids}


def boot(recs, stat):
    """영상째 다시 뽑기(2,000번, 씨앗 20261112). stat(위치 목록) → 값 또는 None. (2.5%, 97.5%, 버린 횟수)."""
    rng = np.random.default_rng(SEED)
    vids, groups_ = by_video(recs, None)
    vals, skipped = [], 0
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(vids), len(vids))
        sample = [r for i in pick for r in groups_[vids[i]]]
        v = stat(sample)
        if v is None:
            skipped += 1
            continue
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [r3(float(lo)), r3(float(hi)), skipped]


def diff_stat(a, b, label="big"):
    def f(rs):
        y = [r[label] for r in rs]
        x, z = auc([a(r) for r in rs], y), auc([b(r) for r in rs], y)
        return None if x is None or z is None else x - z
    return f


def verdict(lo, hi):
    if lo > 0:
        return "재현됨: 구간이 0 위"
    if hi < 0:
        return "뒤집힘: 구간이 0 아래"
    return "판정 못 함: 구간이 0을 걸침(이 크기의 자료로는 차이를 우연과 가르지 못함)"


# ---- 한 영상 빼고 로지스틱 회귀
def features(r):
    a, p = r["action"]["probabilities"], r["progress"]["probabilities"]
    return [a["1"], a["2"], a["3"], p["1"], p["2"], p["3"]] + [r["kind"][k] for k in KIND_KEYS] + [r["worth"]]


FEATURE_NAMES = ["action_p1", "action_p2", "action_p3", "progress_p1", "progress_p2", "progress_p3"] + [f"kind_{k}" for k in KIND_KEYS] + ["worth"]


def fit_lr(X, y, lam=1.0):
    """L2 로지스틱 회귀(뉴턴법). X는 표준화된 값, 절편은 벌점 없음. 결과를 보기 전에 lam = 1로 정함."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + R @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def lovo(recs):
    X = np.array([features(r) for r in recs], float)
    y = np.array([r["big"] for r in recs], float)
    vids = np.array([r["video"] for r in recs])
    pred = np.zeros(len(recs))
    for v in sorted(set(vids)):
        tr, te = vids != v, vids == v
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        w = fit_lr((X[tr] - mu) / sd, y[tr])
        pred[te] = 1 / (1 + np.exp(-(w[0] + ((X[te] - mu) / sd) @ w[1:])))
    for r, p in zip(recs, pred):
        r["lovo"] = float(p)
    singles = {"action": lambda r: r["action"]["score"], "progress": lambda r: r["progress"]["score"], "worth": lambda r: r["worth"],
               **{f"kind_{k}": (lambda k: lambda r: r["kind"][k])(k) for k in list(P.KINDS)}}
    single_auc = {k: r3(auc([f(r) for r in recs], y.astype(bool))) for k, f in singles.items()}
    best = max(single_auc, key=lambda k: single_auc[k])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    w_all = fit_lr((X - mu) / sd, y)
    return {"what": "한 영상을 빼고 나머지 11개로 맞춘 회귀로 그 영상을 매김(12번). 특징 12개(질문마다의 확률, 합이 1인 묶음은 하나씩 뺌), L2 벌점 1, 표준화",
            "features": FEATURE_NAMES, "auc_lovo": r3(auc(pred, y.astype(bool))), "ap_lovo": r3(ap(pred, y.astype(bool))),
            "auc_fixed_formula": r3(auc([r["final"] for r in recs], y.astype(bool))), "ap_fixed_formula": r3(ap([r["final"] for r in recs], y.astype(bool))),
            "single_question_auc": single_auc, "best_single": best, "best_single_note": "가장 좋은 하나는 결과를 보고 고른 것이라 그쪽에 유리하다",
            "diff_lovo_minus_formula_ci": boot(recs, diff_stat(lambda r: r["lovo"], lambda r: r["final"])),
            "diff_lovo_minus_best_single_ci": boot(recs, diff_stat(lambda r: r["lovo"], singles[best])),
            "coef_all_data_standardized": {n: round(float(c), 3) for n, c in zip(["intercept"] + FEATURE_NAMES, w_all)}}


def analyse(recs, arm, all_items=None):
    unmatched = finish(recs, arm, all_items)
    aucs = four_aucs(recs)
    out = {"positions": len(recs), "unmatched_events": unmatched,
           "positives": {k: int(sum(r[k] for r in recs)) for k in ("fight", "prog", "big")},
           "auc": {k: r3(v) for k, v in aucs.items()},
           "auc_ci": {k: boot(recs, (lambda name: lambda rs: auc([signal(r, name) for r in rs], [r[TARGET[name]] for r in rs]))(k)) for k in TARGET},
           "ap_big": {k: r3(ap([signal(r, k) for r in recs], [r["big"] for r in recs])) for k in ("final", "worth")},
           "final_minus_worth": {"auc_diff": r3(aucs["final"] - aucs["worth"]),
                                 "ci95": boot(recs, diff_stat(lambda r: r["final"], lambda r: r["worth"]))},
           "thresholds": thresholds(recs), "top3_final": top3(recs, lambda r: r["final"]), "top3_worth": top3(recs, lambda r: r["worth"])}
    lo, hi, _ = out["final_minus_worth"]["ci95"]
    out["final_minus_worth"]["reading"] = verdict(lo, hi)
    if arm == "same434":
        out["groups"] = groups(recs)
        cand = [r for r in recs if r["role"] != "control"]
        for r in cand:
            if P.has(r["events"], "crafting", 3):
                rs = sorted((x for x in cand if x["video"] == r["video"]), key=lambda x: -x["final"])
                out["diamond_pickaxe"] = {"video": r["video"], "t": r["t"], "rank": rs.index(r) + 1, "of": len(rs)}
    out["lovo_logistic"] = lovo(recs)
    return out


# ---- 방식
def run_jev(args):
    recs = jev_records()
    out = analyse(recs, "same434")
    out["label"] = "TypeSafe Jev로 잰 값 (jev-1.14.0, 2026-11-11에 받아 둔 답. 새 요청 없음)"
    out["reproduce"] = {k: {"reported": JEV_REPORTED[k], "recomputed": out["auc"][k], "same": out["auc"][k] == JEV_REPORTED[k]} for k in JEV_REPORTED}
    stored = P.read(P.FL / "results" / "hl_mc_eval.json")["rank"]
    out["reproduce_extra"] = {"top3_big_share": [out["top3_final"]["big_in_top"], out["top3_final"]["picked"], stored["top_big_share"]],
                              "groups_same_as_hl_mc_eval": out["groups"] == stored["groups"],
                              "diamond_pickaxe": [out.get("diamond_pickaxe"), stored.get("diamond_pickaxe")]}
    P.write(HERE / "results" / "jev_eval.json", out)
    show(out, "TypeSafe Jev로 잰 값")
    print("다시 계산:", json.dumps(out["reproduce"], ensure_ascii=False))
    print("덧붙여 맞춰 본 것:", json.dumps(out["reproduce_extra"], ensure_ascii=False)[:600])
    if not all(v["same"] for v in out["reproduce"].values()):
        raise SystemExit("원래 보고서의 AUC와 다르다")


def show(out, title):
    print(f"===== {title} =====")
    print(f"위치 {out['positions']}, 정답 수 {out['positives']}, 붙지 못한 사건 {out['unmatched_events']}")
    for k in TARGET:
        print(f"  AUC {k:8s} {out['auc'][k]}  [{out['auc_ci'][k][0]}, {out['auc_ci'][k][1]}]")
    fw = out["final_minus_worth"]
    print(f"  최종 − 막연한 질문 AUC {fw['auc_diff']} [{fw['ci95'][0]}, {fw['ci95'][1]}] (버린 다시 뽑기 {fw['ci95'][2]}) → {fw['reading']}")
    print(f"  AP(큰 사건): {out['ap_big']}")
    print(f"  0.5 기준 등: {json.dumps(out['thresholds'], ensure_ascii=False)}")
    print(f"  상위 3곳: 최종 {out['top3_final']['big_in_top']}/{out['top3_final']['picked']}, 막연한 질문 {out['top3_worth']['big_in_top']}/{out['top3_worth']['picked']}, 바탕 {out['top3_final']['base_share']}")
    lv = out["lovo_logistic"]
    print(f"  한 영상 빼고 회귀 AUC {lv['auc_lovo']} (AP {lv['ap_lovo']}), 고정 식 {lv['auc_fixed_formula']}, 가장 좋은 하나 {lv['best_single']} {lv['single_question_auc'][lv['best_single']]}; "
          f"회귀−식 {lv['diff_lovo_minus_formula_ci']}, 회귀−하나 {lv['diff_lovo_minus_best_single_ci']}")


def run_baselines(args):
    items = P.load_items()
    frames = P.load_packs()
    videos = P.read(P.FL / "data" / "mc_items.json")["videos"]
    from PIL import Image
    import io
    diff_cache = {}

    def change(video, times):
        """프레임 4장에서 이웃한 두 장의 화소 차이 절댓값 평균(RGB 0~255)을 세 쌍에서 평균."""
        vals = []
        for a, b in zip(times, times[1:]):
            key = (video, a, b)
            if key not in diff_cache:
                fa = np.asarray(Image.open(io.BytesIO(frames[video][P.fname(a)])).convert("RGB"), np.int16)
                fb = np.asarray(Image.open(io.BytesIO(frames[video][P.fname(b)])).convert("RGB"), np.int16)
                diff_cache[key] = float(np.abs(fa - fb).mean())
            vals.append(diff_cache[key])
        return sum(vals) / len(vals)

    out = {"audio": "12개 영상 모두 소리 흐름이 없다(data/counts.json의 audio_streams). 소리 크기 기준선은 낼 수 없다"}
    for arm in ("same434", "full"):
        recs = [{"id": it["id"], "video": it["video"], "t": it["t"], "role": it["role"], "times": it["times"]} for it in items[arm]]
        eligible = (lambda r: r["role"] != "control") if arm == "same434" else (lambda r: True)
        att, _ = P.attach(recs, events(), eligible)
        for r in recs:
            r.update(P.label(att[r["id"]]))
            r["position"] = r["t"] / videos[r["video"]]["duration"]
            r["change"] = change(r["video"], r["times"])
        res = {}
        for name in ("position", "change"):
            res[name] = {}
            for lab in ("big", "fight", "prog"):
                s, y = [r[name] for r in recs], [r[lab] for r in recs]
                res[name][lab] = {"auc": r3(auc(s, y)), "ap": r3(ap(s, y)), "positives": int(sum(y)),
                                  "auc_ci": boot(recs, (lambda nm, lb: lambda rs: auc([x[nm] for x in rs], [x[lb] for x in rs]))(name, lab))}
        res["note"] = ("position = 위치 시각 ÷ 영상 길이(늦을수록 큼). change = 그 위치의 4장(같은 프레임)에서 이웃한 두 장의 화소 차이 평균(클수록 큼). "
                       "AUC가 0.5보다 작으면 반대 방향이 정답 쪽이라는 뜻이다(방향은 미리 정했고 뒤집지 않는다)")
        out[arm] = res
        print(arm, json.dumps({k: {lab: [v[lab]["auc"], v[lab]["ap"]] for lab in ("big", "fight", "prog")} for k, v in res.items() if k != "note"}, ensure_ascii=False))
    P.write(HERE / "results" / "baselines.json", out)


def run_h2o(args):
    res_dir = Path(args.results)          # 기본은 results/(Modal 볼륨에서 받아 온 답). 코드 시험 때만 다른 폴더를 준다
    items = P.load_items()
    out = {"label": "판정 모델(H2O)로 잰 값 (H2O-Lightning-4B, 판 659fc412)"}
    # 1) 정상 확인
    sp = res_dir / "sanity.json"
    if sp.exists():
        got = {r["id"]: r["probs"] for r in P.read(sp)["records"] if "error" not in r}
        old = {f"{r['vid']}@{r['k']}": r["probs"] for r in P.read(P.MCD / "results" / "sep4_half.json")["records"] if "error" not in r}
        pairs = [(got[i][q], old[i][q]) for i in got if i in old for q in got[i]]
        d = [abs(a - b) for a, b in pairs]
        cross = sum((a >= 0.5) != (b >= 0.5) for a, b in pairs)
        ok = len(pairs) == 800 and sum(d) / len(d) <= SANITY_MAX_MEAN and cross <= SANITY_MAX_CROSS
        out["sanity"] = {"pairs": len(pairs), "mean_abs_diff": round(sum(d) / len(d), 4), "max_abs_diff": round(max(d), 4), "crossed_0_5": cross,
                         "rule": f"평균 ≤ {SANITY_MAX_MEAN}, 0.5를 넘나든 것 ≤ {SANITY_MAX_CROSS}/800, 800쌍이 모두 있어야 함", "pass": ok}
        print("정상 확인:", json.dumps(out["sanity"], ensure_ascii=False))
    jev_path = HERE / "results" / "jev_eval.json"
    jev = P.read(jev_path) if jev_path.exists() else None
    # 2) 같은 434곳, 3) 영상 전체 1초마다
    for arm in ("same434", "full"):
        p = res_dir / f"{arm}.json"
        if not p.exists():
            continue
        data = P.read(p)
        good = [r for r in data["records"] if "error" not in r]
        bad = len(data["records"]) - len(good)
        want = {it["id"]: it for it in items[arm]}
        order = {i: n for n, i in enumerate(want)}
        recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
                for r in sorted(good, key=lambda r: order[r["id"]])]
        res = analyse(recs, arm, items[arm])
        res["errors"] = bad
        res["missing"] = len(want) - len(good)
        res["judge"] = bad / max(1, len(want)) <= MAX_ERROR_SHARE and out.get("sanity", {}).get("pass", False)
        res["meta"] = data.get("meta")
        if arm == "same434" and jev:
            res["vs_typesafe_jev"] = {k: {"typesafe_jev": jev["auc"][k], "h2o": res["auc"][k]} for k in TARGET}
        out[arm] = res
        show(res, f"판정 모델(H2O) {arm}")
    # 4) 같은 434곳을 두 번 물은 셈(영상 전체 1초마다에 같은 입력이 들어 있다). 판정에 쓰지 않음
    if "same434" in out and "full" in out:
        a = {r["id"]: r for r in P.read(res_dir / "same434.json")["records"] if "error" not in r}
        b = {r["id"]: r for r in P.read(res_dir / "full.json")["records"] if "error" not in r and r["id"] in a}
        pairs = [(a[i]["worth"], b[i]["worth"]) for i in b] + [(a[i]["kind"][k], b[i]["kind"][k]) for i in b for k in P.KINDS] + \
                [(a[i][n]["probabilities"][j], b[i][n]["probabilities"][j]) for i in b for n in ("action", "progress") for j in "0123"]
        d = [abs(x - y) for x, y in pairs]
        out["repeat_434"] = {"positions": len(b), "probabilities": len(d), "mean_abs_diff": round(sum(d) / len(d), 4),
                             "crossed_0_5": sum((x >= 0.5) != (y >= 0.5) for x, y in pairs)}
    P.write(res_dir / "h2o_eval.json", out)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("mode", choices=["jev", "baselines", "h2o"])
    ap_.add_argument("--results", default=str(HERE / "results"))
    args = ap_.parse_args()
    {"jev": run_jev, "baselines": run_baselines, "h2o": run_h2o}[args.mode](args)


if __name__ == "__main__":
    main()
