"""highlight_auto 채점 (이 Mac, CPU). 판정 모델 요청은 없다. 규칙은 README 5·7·8·9절이고 결과를 보기 전에 정했다.

TVSum의 점수(anno.tsv)는 data/tvsum_src/ 안의 파일을 경로로 읽어 메모리에서만 쓴다. 점수 자체는 어떤 파일에도 쓰지 않는다.
results/에는 지표(영상마다의 값과 평균, 구간)만 쓴다. results/는 이 Mac에만 둔다.

단계
  human     : 사람 기준값(Otani 방식 τ, ρ)과 무작위 확인, 맨 위 지표의 사람 기준값·상한 → results/human.json
  baselines : 공짜 기준선 셋(위치, 화면 변화, 소리 크기)의 7절 지표와 구간 → results/baselines_tvsum.json
  h2o       : (GPU 실행 뒤) <results>/tvsum.json, minecraft.json, sanity.json 채점 → <results>/eval.json
  synth     : 가짜 답을 --out 폴더(results 밖)에 만들고 h2o 채점을 끝까지 돌려 본다

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/highlight_auto/ha_eval.py <단계>
"""
import sys

sys.dont_write_bytecode = True          # 다른 실험 폴더(highlight_h2o, charades_dense)에 __pycache__를 만들지 않는다
import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ha_prepare as P  # noqa: E402

SEED, RESAMPLES = 20261112, 2000
TOP_SHARE = 0.15
MAX_ERROR_SHARE = 0.02
SANITY_MAX_MEAN, SANITY_MAX_CROSS = 0.010, 8
HUMAN_PUBLISHED = {"tau": 0.177, "rho": 0.204}          # Otani 등 CVPR 2019 표 2, 공식 노트북의 출력
RESULTS = HERE / "results"


def r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# =====================================================================  TVSum 자료 (메모리에서만)
class TV:
    """영상마다: 평가자 20명의 프레임별 점수 Y(20×N), 위치(샷)의 프레임 수, 위치별 평가자 점수 A(20×P), gt(P)."""

    def __init__(self):
        anno, _ = P.load_anno()
        items = P.read(HERE / "data" / "tvsum_items.json")["videos"]
        self.videos = sorted(items)                       # 다시 뽑기의 영상 순서(이름 순, hh_eval과 같은 방식)
        self.cat = {v: items[v]["category"] for v in self.videos}
        self.items = items
        self.Y, self.lens, self.A, self.gt = {}, {}, {}, {}
        for v in self.videos:
            Y = np.asarray(anno[v], dtype=np.int16)
            pos = items[v]["positions"]
            lens = np.array([p["end"] - p["start"] for p in pos])
            if lens.sum() != Y.shape[1]:
                raise SystemExit(f"{v}: 위치의 프레임 합 {lens.sum()} ≠ 점수 길이 {Y.shape[1]}")
            edges = np.concatenate([[0], np.cumsum(lens)])
            A = np.stack([np.add.reduceat(Y[a].astype(float), edges[:-1]) / lens for a in range(Y.shape[0])])
            self.Y[v], self.lens[v], self.A[v], self.gt[v] = Y, lens, A, A.mean(0)


# =====================================================================  지표
def tau_rho(x, Y):
    """Otani 등의 공식 노트북과 같은 계산: τ = kendalltau(rankdata(-x), rankdata(-y)), ρ = spearmanr(x, y). 평가자 평균."""
    from scipy.stats import kendalltau, rankdata, spearmanr
    rx = rankdata(-x)
    t = [kendalltau(rx, rankdata(-y)).statistic for y in Y]
    r = [spearmanr(x, y).statistic for y in Y]
    return float(np.mean(t)), float(np.mean(r))


def topk_expect(score, value, k):
    """점수 위 k곳의 value 평균. 경계에서 같은 점수가 여럿이면 무작위로 고르는 기댓값."""
    score, value = np.asarray(score, float), np.asarray(value, float)
    total, need = 0.0, k
    for lvl in np.unique(score)[::-1]:
        m = score == lvl
        c = int(m.sum())
        if c <= need:
            total += value[m].sum()
            need -= c
        else:
            total += value[m].mean() * need
            need = 0
        if need == 0:
            break
    return total / k


def ap(score, labels):
    """평균 정밀도(highlight_h2o의 ap와 같다: 같은 점수는 한 기준으로 묶음)."""
    s, y = np.asarray(score, float), np.asarray(labels, bool)
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


def top_metrics(score, value):
    """맨 위 지표(README 7-2): 상위 15% 올림 값, 고른 곳의 평균, HIT@1, HIT@5, mAP@15. value = 위치별 정답 점수(gt 등)."""
    n = len(value)
    K = math.ceil(TOP_SHARE * n)
    srt = np.sort(value)[::-1]
    H = (np.asarray(value) >= srt[K - 1]).astype(float)
    top = topk_expect(score, value, K)
    return {"lift": top - float(np.mean(value)), "top_mean": top, "hit1": topk_expect(score, H, 1), "hit5": topk_expect(score, H, 5),
            "ap15": ap(score, H.astype(bool)), "H_share": float(H.mean())}


def video_metrics(tv, v, s):
    """위치별 점수 s(길이 P)로 한 영상의 7절 지표."""
    s = np.asarray(s, float)
    x = np.repeat(s, tv.lens[v])
    t, r = tau_rho(x, tv.Y[v])
    return {"tau": t, "rho": r, **top_metrics(s, tv.gt[v])}


def pct_rank(x):
    """영상 안 백분위 순위 (중간 순위 − 0.5) ÷ n. README 5절."""
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return (rankdata(x) - 0.5) / len(x)


def boot_ci(per_video, videos, combine=None):
    """영상째 다시 뽑기(2,000번, 씨앗 20261112). per_video: {이름: {영상: 값}}. combine(평균들 dict) → 값. 기본은 이름 하나의 평균."""
    rng = np.random.default_rng(SEED)
    names = list(per_video)
    M = np.array([[per_video[nm][v] for v in videos] for nm in names], float)
    vals = []
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(videos), len(videos))
        means = dict(zip(names, M[:, pick].mean(1)))
        vals.append(combine(means) if combine else means[names[0]])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [r4(lo), r4(hi)]


def verdict(lo, hi, x_name, y_name, where, metric, n_label="영상 50개로는"):
    """README 8절의 세 갈래. 구간이 0을 걸치면 '같다'나 '차이가 없다'라고 쓰지 않는다."""
    if lo > 0:
        return f"{y_name}보다 {x_name} 쪽이 낫다 ({where}, {metric})"
    if hi < 0:
        return f"{x_name}보다 {y_name} 쪽이 낫다 ({where}, {metric})"
    return f"판정 못 함: {n_label} 이 차이를 우연과 가르지 못했다 ({where}, {metric})"


METRICS = ("tau", "rho", "lift", "top_mean", "hit1", "hit5", "ap15")


def summarize(tv, per_video_metrics):
    """{영상: 지표 dict} → 지표마다 평균, 구간, 범주별 평균."""
    out = {}
    for m in METRICS:
        pv = {v: per_video_metrics[v][m] for v in tv.videos}
        out[m] = {"mean": r4(np.mean(list(pv.values()))), "ci": boot_ci({m: pv}, tv.videos)}
    cats = sorted(set(tv.cat.values()))
    out["by_category"] = {c: {m: r3(np.mean([per_video_metrics[v][m] for v in tv.videos if tv.cat[v] == c])) for m in ("tau", "rho", "lift")} for c in cats}
    return out


# =====================================================================  human
def _human_video(args):
    v, Y, A, lens, gt, n_rand = args
    from scipy.stats import kendalltau, rankdata, spearmanr
    na = Y.shape[0]
    ranks = [rankdata(-y) for y in Y]
    T = np.full((na, na), np.nan)
    R = np.full((na, na), np.nan)
    for i in range(na):
        for j in range(i + 1, na):
            T[i, j] = T[j, i] = kendalltau(ranks[i], ranks[j]).statistic
            R[i, j] = R[j, i] = spearmanr(Y[i], Y[j]).statistic
    tau = float(np.mean([np.nanmean(T[i]) for i in range(na)]))     # 공식 노트북: 평가자마다 다른 19명과의 평균, 그것의 평균
    rho = float(np.mean([np.nanmean(R[i]) for i in range(na)]))
    # 맨 위 지표의 사람 기준값: 평가자 i의 점수로 고르고 다른 19명 평균으로 잰다
    hum = {k: [] for k in ("lift", "top_mean", "hit1", "hit5", "ap15")}
    for i in range(na):
        others = np.delete(A, i, axis=0).mean(0)
        m = top_metrics(A[i], others)
        for k in hum:
            hum[k].append(m[k])
    human_top = {k: float(np.mean(x)) for k, x in hum.items()}
    oracle = top_metrics(gt, gt)
    # 무작위: (1) 위치(샷)마다 하나의 난수(우리 팔과 같은 모양) (2) 프레임마다 난수(Otani 등의 방식)
    import zlib
    rng = np.random.default_rng([SEED, zlib.crc32(v.encode())])
    rs, rf = [], []
    for _ in range(n_rand):
        s = rng.random(len(lens))
        x = np.repeat(s, lens)
        rx = rankdata(-x)
        rs.append((np.mean([kendalltau(rx, r).statistic for r in ranks]), np.mean([spearmanr(x, y).statistic for y in Y]),
                   top_metrics(s, gt)["lift"]))
        f = rng.random(Y.shape[1])
        rff = rankdata(-f)
        rf.append((np.mean([kendalltau(rff, r).statistic for r in ranks]), np.mean([spearmanr(f, y).statistic for y in Y])))
    return v, {"tau": tau, "rho": rho, "pair_tau_min": float(np.nanmin(T)), "pair_tau_max": float(np.nanmax(T)),
               "human_top": human_top, "oracle": oracle,
               "random_shot": [float(np.mean([r[k] for r in rs])) for k in range(3)],
               "random_frame": [float(np.mean([r[k] for r in rf])) for k in range(2)]}


def run_human(args):
    import concurrent.futures as cf
    import time
    tv = TV()
    t0 = time.time()
    jobs = [(v, tv.Y[v], tv.A[v], tv.lens[v], tv.gt[v], args.random) for v in tv.videos]
    per = {}
    with cf.ProcessPoolExecutor(8) as ex:
        for v, res in ex.map(_human_video, jobs):
            per[v] = res
    vids = tv.videos
    tau = float(np.mean([per[v]["tau"] for v in vids]))
    rho = float(np.mean([per[v]["rho"] for v in vids]))
    ok = round(tau, 3) == HUMAN_PUBLISHED["tau"] and round(rho, 3) == HUMAN_PUBLISHED["rho"]
    out = {
        "what": "사람 기준값(Otani 등 CVPR 2019, 공식 노트북 rank order statistics.ipynb와 같은 계산)을 data/tvsum_src의 anno.tsv로 다시 낸 것",
        "human_tau": r4(tau), "human_rho": r4(rho), "published": HUMAN_PUBLISHED, "reproduced_3dp": ok,
        "human_tau_ci": boot_ci({"x": {v: per[v]["tau"] for v in vids}}, vids),
        "human_rho_ci": boot_ci({"x": {v: per[v]["rho"] for v in vids}}, vids),
        "random_draws_per_video": args.random,
        "random_shot_level": {"tau": r4(np.mean([per[v]["random_shot"][0] for v in vids])), "rho": r4(np.mean([per[v]["random_shot"][1] for v in vids])),
                              "lift": r4(np.mean([per[v]["random_shot"][2] for v in vids]))},
        "random_frame_level": {"tau": r4(np.mean([per[v]["random_frame"][0] for v in vids])), "rho": r4(np.mean([per[v]["random_frame"][1] for v in vids]))},
        "random_check_pass": all(abs(np.mean([per[v][k][i] for v in vids])) <= 0.005 for k, i in (("random_shot", 0), ("random_shot", 1), ("random_frame", 0), ("random_frame", 1))),
        "human_top": {k: {"mean": r4(np.mean([per[v]["human_top"][k] for v in vids])), "ci": boot_ci({"x": {v: per[v]["human_top"][k] for v in vids}}, vids)}
                      for k in ("lift", "top_mean", "hit1", "hit5", "ap15")},
        "oracle_top": {k: r4(np.mean([per[v]["oracle"][k] for v in vids])) for k in ("lift", "top_mean", "hit1", "hit5", "ap15")},
        "mean_gt_all": r4(np.mean([tv.gt[v].mean() for v in vids])),
        "per_video": {v: {"tau": r4(per[v]["tau"]), "rho": r4(per[v]["rho"]), "pair_tau_min": r3(per[v]["pair_tau_min"]), "pair_tau_max": r3(per[v]["pair_tau_max"])} for v in vids},
        "seconds": round(time.time() - t0, 1),
    }
    P.write(RESULTS / "human.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"}, ensure_ascii=False, indent=1))


# =====================================================================  baselines
BASE_NOTE = {"position": "위치(샷) 가운데 시각 ÷ 영상 길이. 늦을수록 큼",
             "change": "그 위치의 그림 4장에서 이웃한 두 장의 RGB 차이 절댓값 평균을 세 쌍에서 평균. 클수록 큼",
             "loudness": "샷 구간의 소리 RMS(dBFS). 클수록 큼"}


def baseline_scores(tv):
    """위치별 기준선 점수 {이름: {영상: 배열}}. 점수(정답)와 무관한 값이다."""
    from PIL import Image
    loud = P.read(HERE / "data" / "tvsum_loudness.json")["videos"]
    packs = P.load_packs()
    out = {"position": {}, "change": {}, "loudness": {}}
    for v in tv.videos:
        it = tv.items[v]
        dur = it["frames_anno"] / it["fps"]
        out["position"][v] = np.array([p["center"] / dur for p in it["positions"]])
        cache = {}

        def arr(i):
            if i not in cache:
                cache[i] = np.asarray(Image.open(io.BytesIO(packs[v][P.fname(i)])).convert("RGB"), np.int16)
            return cache[i]
        ch = []
        for p in it["positions"]:
            fr = p["frames"]
            ch.append(np.mean([float(np.abs(arr(a) - arr(b)).mean()) for a, b in zip(fr, fr[1:])]))
        out["change"][v] = np.array(ch)
        out["loudness"][v] = np.array(loud[v]["db"])
    return out


def run_baselines(args):
    tv = TV()
    sc = baseline_scores(tv)
    out = {"note": BASE_NOTE, "direction": "미리 정함, 뒤집지 않는다(README 6절)", "baselines": {}}
    for name, per in sc.items():
        pvm = {v: video_metrics(tv, v, per[v]) for v in tv.videos}
        out["baselines"][name] = {**summarize(tv, pvm), "per_video": {v: {m: r4(pvm[v][m]) for m in ("tau", "rho", "lift")} for v in tv.videos}}
        print(name, {m: (out["baselines"][name][m]["mean"], out["baselines"][name][m]["ci"]) for m in ("tau", "rho", "lift", "hit1", "hit5", "ap15")}, flush=True)
    # 판정 모델 실행 뒤 비교 3에 쓰려고 위치별 기준선 점수도 남긴다(정답이 아니라 영상에서 잰 값)
    P.write(HERE / "data" / "tvsum_baseline_scores.json", {n: {v: [round(float(x), 5) for x in per[v]] for v in tv.videos} for n, per in sc.items()})
    P.write(RESULTS / "baselines_tvsum.json", out)


# =====================================================================  h2o (판정 모델 답 채점)
def _values(rec, keys):
    """한 위치의 답 → {질문: 값}. noul은 "예" 확률, score는 기댓값."""
    out = {}
    for k in keys:
        a = rec["answers"][k]
        out[k] = a["noul"] if "noul" in a else a["score"]
    return out


def _arm_scores(recs_by_video, positions_by_video, keys_by_video, worth_key="worth"):
    """영상마다 (가), (나), (다)와 질문별 값. 답이 없는 위치(오류)는 그 질문의 백분위 순위 0.5로 둔다(README 8절의 오류 조건 안에서만 판정)."""
    out = {}
    for v, n in positions_by_video.items():
        got = recs_by_video.get(v, {})
        keys = keys_by_video[v]
        cols = {k: np.full(n, np.nan) for k in [worth_key] + keys if k is not None}
        for idx, vals in got.items():
            for k in cols:
                cols[k][idx] = vals[k]
        ranks = {}
        for k, c in cols.items():
            r = np.full(n, 0.5)
            ok = ~np.isnan(c)
            if ok.any():
                r[ok] = pct_rank(c[ok])
            ranks[k] = r
        na_raw = np.mean([ranks[k] for k in keys], axis=0)
        res = {"na": na_raw, "q": {k: ranks[k] for k in keys}, "missing": int(n - len(got))}
        if worth_key is not None:
            ga = ranks[worth_key]
            res["ga"] = ga
            res["da"] = 0.5 * ga + 0.5 * pct_rank(na_raw)
        out[v] = res
    return out


def sanity_check(res_dir):
    sp = Path(res_dir) / "sanity.json"
    if not sp.exists():
        return {"pass": False, "note": "sanity.json 없음"}
    got = {r["id"]: r["probs"] for r in P.read(sp)["records"] if "error" not in r}
    old = {f"{r['vid']}@{r['k']}": r["probs"] for r in P.read(P.ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"] if "error" not in r}
    pairs = [(got[i][q], old[i][q]) for i in got if i in old for q in got[i]]
    d = [abs(a - b) for a, b in pairs]
    cross = sum((a >= 0.5) != (b >= 0.5) for a, b in pairs)
    ok = len(pairs) == 800 and sum(d) / len(d) <= SANITY_MAX_MEAN and cross <= SANITY_MAX_CROSS
    return {"pairs": len(pairs), "mean_abs_diff": r4(sum(d) / max(1, len(d))), "max_abs_diff": r4(max(d) if d else None), "crossed_0_5": cross,
            "rule": f"800쌍 모두, 평균 ≤ {SANITY_MAX_MEAN}, 0.5를 넘나든 것 ≤ {SANITY_MAX_CROSS}", "pass": ok}


def eval_tvsum(res_dir, req, human):
    tv = TV()
    data = P.read(Path(res_dir) / "tvsum.json")
    recs = data["records"]
    good = [r for r in recs if "error" not in r]
    errors = len(recs) - len(good)
    want = sum(len(tv.items[v]["positions"]) for v in tv.videos)
    keys = {v: [k for k in req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in tv.videos}
    by_v = {}
    for r in good:
        by_v.setdefault(r["video"], {})[r["k"]] = _values(r, ["worth"] + keys[r["video"]])
    arms = _arm_scores(by_v, {v: len(tv.items[v]["positions"]) for v in tv.videos}, keys)
    pvm = {name: {v: video_metrics(tv, v, arms[v][key]) for v in tv.videos} for name, key in (("ga", "ga"), ("na", "na"), ("da", "da"))}
    base_sc = P.read(HERE / "data" / "tvsum_baseline_scores.json")
    for b in ("position", "change", "loudness"):
        pvm[b] = {v: video_metrics(tv, v, np.array(base_sc[b][v])) for v in tv.videos}
    label = {"ga": "막연한 질문(가)", "na": "글쓴이 기준(나)", "da": "합친 것(다)", "position": "위치 기준선", "change": "화면 변화 기준선", "loudness": "소리 크기 기준선"}
    out = {"requests": want, "answered": len(good), "errors": errors, "error_share": r4(errors / max(1, want)),
           "arms": {n: summarize(tv, pvm[n]) for n in pvm}}
    # 질문 하나하나(보조, 판정하지 않음)
    out["per_question_tau"] = {v: {k: r3(tau_rho(np.repeat(arms[v]["q"][k], tv.lens[v]), tv.Y[v])[0]) for k in keys[v]} for v in tv.videos}
    comps = {}
    vids = tv.videos
    for metric, mname in (("tau", "전체 순서(τ)"), ("lift", "맨 위(상위 15% 올림 값)")):
        g = {n: {v: pvm[n][v][metric] for v in vids} for n in pvm}
        d = {v: g["na"][v] - g["ga"][v] for v in vids}
        lo, hi = boot_ci({"d": d}, vids)
        comps[f"1_na_minus_ga_{metric}"] = {"point": r4(np.mean(list(d.values()))), "ci": [lo, hi], "verdict": verdict(lo, hi, label["na"], label["ga"], "TVSum", mname)}
        pt = np.mean(list(g["da"].values())) - max(np.mean(list(g["ga"].values())), np.mean(list(g["na"].values())))
        lo, hi = boot_ci({"da": g["da"], "ga": g["ga"], "na": g["na"]}, vids, combine=lambda m: m["da"] - max(m["ga"], m["na"]))
        better = "na" if np.mean(list(g["na"].values())) >= np.mean(list(g["ga"].values())) else "ga"
        comps[f"2_da_minus_better_{metric}"] = {"point": r4(pt), "ci": [lo, hi], "better_on_full_data": better,
                                                 "verdict": verdict(lo, hi, label["da"], "둘 중 나은 것", "TVSum", mname)}
        for a in ("ga", "na", "da"):
            for b in ("position", "change", "loudness"):
                d = {v: g[a][v] - g[b][v] for v in vids}
                lo, hi = boot_ci({"d": d}, vids)
                comps[f"3_{a}_minus_{b}_{metric}"] = {"point": r4(np.mean(list(d.values()))), "ci": [lo, hi], "verdict": verdict(lo, hi, label[a], label[b], "TVSum", mname)}
    out["comparisons"] = comps
    out["human_reference"] = {k: human.get(k) for k in ("human_tau", "human_rho", "human_top", "oracle_top")}
    out["meta"] = data.get("meta")
    return out


def eval_minecraft(res_dir, req):
    sys.path.insert(0, str(P.HH))
    import hh_prepare as HP   # noqa: E402  (highlight_h2o, 읽기만)
    import hh_eval as HE      # noqa: E402
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: n for n, i in enumerate(want)}
    stored = [r for r in HP.read(P.HH / "results" / "full.json")["records"] if "error" not in r]
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted(stored, key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    keys = list(req["minecraft"]["questions"])
    data = P.read(Path(res_dir) / "minecraft.json")
    good = {r["id"]: _values(r, keys) for r in data["records"] if "error" not in r}
    errors = len(data["records"]) - len(good)
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    for v, rs in byv.items():
        def col(f):
            return np.array([f(r) for r in rs], float)
        ranks = {}
        for k in keys:
            c = col(lambda r: good.get(r["id"], {}).get(k, np.nan))
            rk = np.full(len(rs), 0.5)
            ok = ~np.isnan(c)
            if ok.any():
                rk[ok] = pct_rank(c[ok])
            ranks[k] = rk
        na_rank = pct_rank(np.mean([ranks[k] for k in keys], axis=0))
        r_worth = pct_rank(col(lambda r: r["worth"]))
        da_rank = pct_rank(0.5 * r_worth + 0.5 * na_rank)
        hand = pct_rank(col(lambda r: r["final"]))
        for i, r in enumerate(rs):
            r["s_hand"], r["s_vague"], r["s_na"], r["s_da"] = hand[i], r_worth[i], na_rank[i], da_rank[i]
            for k in keys:
                r[f"q_{k}"] = ranks[k][i]
    arms = {"hand": "손 기준(받아 둔 final)", "vague": "막연한 질문(받아 둔 worth)", "na": "글쓴이 기준(나)", "da": "합친 것(다)"}
    out = {"requests": len(want), "answered": len(good), "errors": errors, "error_share": r4(errors / len(want)), "arms": {}}

    def pooled_top(key, k):
        s = np.array([r[key] for r in recs])
        y = np.array([r["big"] for r in recs], float)
        return topk_expect(s, y, k) * k
    for a in arms:
        key = f"s_{a}"
        y = [r["big"] for r in recs]
        out["arms"][a] = {"label": arms[a], "auc_big": r4(HE.auc([r[key] for r in recs], y)),
                          "auc_big_ci": HE.boot(recs, (lambda kk: lambda rs: HE.auc([x[kk] for x in rs], [x["big"] for x in rs]))(key))[:2],
                          "ap_big": r4(HE.ap([r[key] for r in recs], y)), "top3": {k: v for k, v in HE.top3(recs, lambda r: r[key]).items() if k != "per_video"},
                          "pooled_top15_big": r3(pooled_top(key, 15)), "pooled_top36_big": r3(pooled_top(key, 36))}
    out["raw_stored_auc"] = {"final": r4(HE.auc([r["final"] for r in recs], [r["big"] for r in recs])),
                             "worth": r4(HE.auc([r["worth"] for r in recs], [r["big"] for r in recs])),
                             "expected_from_highlight_h2o": {"final": 0.855, "worth": 0.719}}
    out["per_question_auc"] = {k: {lab: r3(HE.auc([r[f"q_{k}"] for r in recs], [r[lab] for r in recs])) for lab in ("big", "fight", "prog")} for k in keys}

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(HE.auc([r[f"s_{a}"] for r in rs], [r["big"] for r in rs]),
                                                                                   HE.auc([r[f"s_{b}"] for r in rs], [r["big"] for r in rs]))

    def m3(rs):
        y = [r["big"] for r in rs]
        a = [HE.auc([r[f"s_{k}"] for r in rs], y) for k in ("da", "vague", "na")]
        return None if None in a else a[0] - max(a[1], a[2])
    comps = {}
    for name, (a, b) in {"M1_na_minus_hand": ("na", "hand"), "M2_na_minus_vague": ("na", "vague")}.items():
        lo, hi, _ = HE.boot(recs, diff(a, b))
        pt = out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]
        comps[name] = {"point": r4(pt), "ci": [lo, hi], "verdict": verdict(lo, hi, arms[a], arms[b], "마인크래프트", "AUC", "영상 12개, 큰 사건 56곳으로는")}
    lo, hi, _ = HE.boot(recs, m3)
    pt = out["arms"]["da"]["auc_big"] - max(out["arms"]["vague"]["auc_big"], out["arms"]["na"]["auc_big"])
    comps["M3_da_minus_better"] = {"point": r4(pt), "ci": [lo, hi], "verdict": verdict(lo, hi, arms["da"], "둘 중 나은 것", "마인크래프트", "AUC", "영상 12개, 큰 사건 56곳으로는")}
    out["comparisons"] = comps
    out["baselines_from_highlight_h2o"] = P.read(P.HH / "results" / "baselines.json").get("full")
    out["meta"] = data.get("meta")
    return out


def run_h2o(args):
    res_dir = Path(args.results)
    req = P.read(args.requests)
    human = P.read(RESULTS / "human.json")
    out = {"label": "판정 모델(H2O)로 잰 값 (H2O-Lightning-4B, 판 659fc412)", "sanity": sanity_check(res_dir)}
    out["human_reproduced"] = human["reproduced_3dp"]
    if (res_dir / "tvsum.json").exists():
        out["tvsum"] = eval_tvsum(res_dir, req, human)
    if (res_dir / "minecraft.json").exists():
        out["minecraft"] = eval_minecraft(res_dir, req)
    ok = out["sanity"]["pass"] and out["human_reproduced"]
    for part in ("tvsum", "minecraft"):
        if part in out:
            out[part]["judge"] = ok and out[part]["error_share"] <= MAX_ERROR_SHARE
    P.write(res_dir / "eval.json", out)
    print(json.dumps({"sanity": out["sanity"], **{p: {"judge": out[p]["judge"], "errors": out[p]["errors"],
                                                     "comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in out[p]["comparisons"].items()
                                                                     if not k.startswith("3_")}} for p in ("tvsum", "minecraft") if p in out}},
                     ensure_ascii=False, indent=1))


# =====================================================================  synth (가짜 답으로 끝까지)
def run_synth(args):
    """결과 폴더 밖(--out)에 가짜 기준·요청·답을 만들고 h2o 채점을 돌린다. 코드가 도는지만 본다.
    가짜 TVSum 답: worth는 난수, q_pos는 위치(늦을수록 큼)를 흉내, q_score는 난수 score. 가짜 마인크래프트 답: 받아 둔 action·progress 값을 그대로
    쓰는 질문 둘(그래서 (나)의 AUC는 손 기준에 가깝게 나와야 한다)과 난수 하나. 정상 확인은 받아 둔 sep4_half를 그대로 복사(통과해야 함)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    items = P.read(HERE / "data" / "tvsum_items.json")["videos"]
    crit = {"writer_model": "synthetic", "written_at": "2026-11-14", "videos": {}}
    for v in items:
        crit["videos"][v] = {"kind": "synthetic", "limits": "synthetic", "questions": {
            "q_pos": {"type": "noul", "instructions": "Do these frames in `frames` show the following: synthetic A?", "criteria": {"true": "a", "false": "b"}},
            "q_score": {"type": "score", "instructions": "How much synthetic B do `frames` show?", "criteria": ["none", "some", "a lot"]},
            "q_rand": {"type": "noul", "instructions": "Do `frames` show synthetic C?", "criteria": {"true": "c", "false": "d"}}}}
    crit["videos"]["minecraft"] = {"kind": "synthetic", "limits": "synthetic", "questions": {
        "copy_action": {"type": "score", "instructions": "How much A in `frames`?", "criteria": ["0", "1", "2", "3"]},
        "copy_progress": {"type": "score", "instructions": "How much B in `frames`?", "criteria": ["0", "1", "2", "3"]},
        "rand": {"type": "noul", "instructions": "Do `frames` show C?", "criteria": {"true": "c", "false": "d"}}}}
    P.write(out / "criteria.json", crit)
    req = P.build_requests(crit, write_to=out / "requests.json")
    tv_recs = []
    for v, it in items.items():
        n = len(it["positions"])
        for p in it["positions"]:
            if rng.random() < 0.003:
                tv_recs.append({"id": f"{v}@{p['k']}", "video": v, "k": p["k"], "error": "synthetic error"})
                continue
            probs = rng.dirichlet([1, 1, 1])
            tv_recs.append({"id": f"{v}@{p['k']}", "video": v, "k": p["k"], "answers": {
                "worth": {"noul": float(rng.random())}, "q_pos": {"noul": p["k"] / n},
                "q_score": {"score": float(probs[1] + 2 * probs[2]), "probabilities": {str(i): float(x) for i, x in enumerate(probs)}},
                "q_rand": {"noul": float(rng.random())}}})
    P.write(out / "tvsum.json", {"meta": {"synthetic": True}, "records": tv_recs})
    stored = {r["id"]: r for r in P.read(P.HH / "results" / "full.json")["records"] if "error" not in r}
    mc = []
    for i, r in stored.items():
        mc.append({"id": i, "video": r["video"], "t": r["t"], "answers": {"copy_action": {"score": r["action"]["score"], "probabilities": r["action"]["probabilities"]},
                                                                          "copy_progress": {"score": r["progress"]["score"], "probabilities": r["progress"]["probabilities"]},
                                                                          "rand": {"noul": float(rng.random())}}})
    P.write(out / "minecraft.json", {"meta": {"synthetic": True}, "records": mc})
    san = P.read(P.ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"]
    P.write(out / "sanity.json", {"meta": {"synthetic": True}, "records": [{"id": f"{r['vid']}@{r['k']}", "video": r["vid"], "k": r["k"], "probs": r["probs"]}
                                                                          for r in san if "error" not in r and r["vid"] in ("mc_08", "ch_04") and r["k"] < 100]})
    args.results, args.requests = str(out), str(out / "requests.json")
    run_h2o(args)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("stage", choices=["human", "baselines", "h2o", "synth"])
    ap_.add_argument("--random", type=int, default=100, help="human: 영상마다 무작위 점수 벌 수")
    ap_.add_argument("--results", default=str(RESULTS))
    ap_.add_argument("--requests", default=str(HERE / "data" / "requests.json"))
    ap_.add_argument("--out", default=None)
    args = ap_.parse_args()
    {"human": run_human, "baselines": run_baselines, "h2o": run_h2o, "synth": run_synth}[args.stage](args)


if __name__ == "__main__":
    main()
