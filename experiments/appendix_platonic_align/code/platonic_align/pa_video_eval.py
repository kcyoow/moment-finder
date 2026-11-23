"""platonic_align 12절 채점 (이 Mac, CPU). 규칙은 README 12-3·12-4절이고 MSR-VTT 임베딩을 받기 전에 썼다.

회전을 사진(Flickr) 대신 MSR-VTT의 움직이는 클립과 캡션으로 맞춘 뒤
  (1) MSR-VTT 1k-A 시험 1,000쌍의 표와 비교 K1~K5,
  (2) 그 회전을 우리 자료(TVSum, 마인크래프트)의 받아 둔 임베딩에 쓴 팔 Lv·Lsv·Dv와 비교 V1~V4
를 낸다. 질문 점수, 순위, 지표, 다시 뽑기는 pa_eval.py와 ../highlight_auto/ha_eval.py, ../highlight_h2o/hh_eval.py의 함수를 그대로 쓴다.
TVSum의 점수 파일은 그쪽 폴더에서 읽어 메모리에서만 쓴다. results/에는 지표만 쓴다.

단계
  eval  : results/video.npz, results/video.json, results/ours.npz → results/video_eval.json
  null  : 무작위 회전 기준(README 12-4절의 덧붙임) → results/video_null.json
  synth : 가짜 MSR-VTT 임베딩을 --out 폴더(results 밖)에 만들고 끝까지 돌려 본다(우리 자료 쪽은 받아 둔 ours.npz를 읽기만 한다)

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/platonic_align/pa_video_eval.py eval
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pa_eval as A  # noqa: E402

E, P, HA, HH = A.E, A.P, A.HA, A.HH
RES = HERE / "results"
r4 = E.r4
MIN_R1 = 0.01                                   # README 12-3절의 조건
INPUTS = {"L": "lev", "Ls": "lev_static", "D": "dino"}                    # 이름 → video.npz와 ours.npz의 임베딩 키
PHOTO = {"L": "levjepa", "Ls": "levjepa", "D": "dinov2"}                  # 이름 → ours.npz의 사진 회전
NEW = {"Lv": "L", "Lsv": "Ls", "Dv": "D"}                                 # 새 팔 → 어느 입력의 영상 회전인가
TOL_MODAL = 0.002                               # Modal 안의 표와 이 Mac의 표가 같다고 보는 차이
N_RANDOM, SEED_RANDOM = 100, 20261116           # 무작위 회전의 수와 씨앗(12-4절의 덧붙임)


# =====================================================================  회전과 찾기
def procrustes(a, b):
    """평균을 뺀 짝 (n, d_a), (n, d_b) → 직교 사상 (d_a, d_b). 저장소 prh/align.py의 procrustes와 같은 식."""
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vt


def fit(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    return {"R": procrustes(a - mu_a, b - mu_b), "mu_a": mu_a, "mu_b": mu_b}


def ranks(v, t):
    """v: 회전한 클립, t: 문장. 같은 줄이 짝이다. 논문 evaluate.py video_align과 같은 순위(1부터, 동점은 틀린 쪽으로)."""
    s = A.unit(t) @ A.unit(v).T
    d = np.diag(s)[:, None]
    return {"t2v": (s >= d).sum(1), "v2t": (s.T >= d).sum(1)}


def mutual_knn(a, b, k=10):
    """저장소 prh/metrics.py의 mutual_knn과 같은 정의: 코사인 k-이웃이 겹치는 비율의 평균."""
    k = min(k, len(a) - 1)

    def mask(x):
        s = A.unit(x) @ A.unit(x).T
        np.fill_diagonal(s, -np.inf)
        m = np.zeros(s.shape, bool)
        np.put_along_axis(m, np.argpartition(-s, k - 1, axis=1)[:, :k], True, axis=1)
        return m
    return float((mask(a) & mask(b)).sum(1).mean() / k)


def verdict3(lo, hi, pos, neg, n_label):
    if lo > 0:
        return pos
    if hi < 0:
        return neg
    return f"판정 못 함: {n_label} 이 차이를 우연과 가르지 못했다"


# =====================================================================  (1) MSR-VTT 1k-A
def video_rotations(vz):
    """입력마다, 맞추는 쪽 클립과 그 캡션 20개 평균으로 맞춘 회전."""
    te = vz["is_test"].astype(bool)
    b_tr = vz["text_mean"][~te].astype(np.float64)
    return {k: fit(vz[key][~te].astype(np.float64), b_tr) for k, key in INPUTS.items()}


def msrvtt_part(vz, oz):
    ids = [str(x) for x in vz["ids"]]
    te = vz["is_test"].astype(bool)
    if [ids[i] for i in np.flatnonzero(te)] != [str(x) for x in vz["js_ids"]]:
        raise SystemExit("시험 클립의 순서가 시험 문장의 순서와 다르다")
    b_tr, b_te = vz["text_mean"][~te].astype(np.float64), vz["js_text"].astype(np.float64)
    X = {k: vz[key].astype(np.float64) for k, key in INPUTS.items()}
    rot_v = video_rotations(vz)
    rot_p = {k: {p: oz[f"rot/{PHOTO[k]}/{p}"].astype(np.float64) for p in ("R", "mu_a", "mu_b")} for k in X}
    got = {}

    def row(name, a, m_a, R, m_b, what):
        got[name] = {**ranks((a - m_a) @ R, b_te - m_b), "what": what}
    for k in X:
        own = X[k][~te].mean(0)
        row(f"{k}|video", X[k][te], rot_v[k]["mu_a"], rot_v[k]["R"], rot_v[k]["mu_b"], "영상 회전(제 입력으로 맞춘 것), MSR-VTT 맞추는 쪽 평균")
        row(f"{k}|photo", X[k][te], own, rot_p[k]["R"], rot_p[k]["mu_b"], "사진 회전, MSR-VTT 클립 평균과 Flickr 캡션 평균(11절 방식)")
        row(f"{k}|photo_flickr_means", X[k][te], rot_p[k]["mu_a"], rot_p[k]["R"], rot_p[k]["mu_b"], "사진 회전, Flickr 평균 그대로(논문 표 11 방식)")
    row("L|video_static_fit", X["L"][te], X["L"][~te].mean(0), rot_v["Ls"]["R"], rot_v["Ls"]["mu_b"], "정지 클립으로 맞춘 영상 회전을 움직이는 클립에, 평균은 움직이는 클립 것")
    table = {}
    for name, g in got.items():
        k = name.split("|")[0]
        table[name] = {"what": g["what"], **{f"{d} R@{n}": r4(float((g[d] <= n).mean())) for d in ("t2v", "v2t") for n in (1, 5, 10)}}
        table[name]["mknn"] = r4(mutual_knn(X[k][te] - X[k][~te].mean(0), b_te - b_tr.mean(0)))
    hit = {name: (g["t2v"] <= 1).astype(float) for name, g in got.items()}
    n = int(te.sum())
    n_label = f"시험 문장 {n:,}개로는"
    idx = list(range(n))

    def comp(d, pos, neg):
        lo, hi = E.boot_ci({"d": dict(enumerate(d))}, idx)
        return {"point": r4(float(d.mean())), "ci": [lo, hi], "verdict": verdict3(lo, hi, pos, neg, n_label)}
    k1 = hit["L|video"] - hit["L|photo"]
    kd = hit["D|video"] - hit["D|photo"]
    comps = {
        "K1_L_video_minus_L_photo": comp(k1, "영상으로 맞춘 회전 쪽이 낫다 (LeVJEPA)", "사진으로 맞춘 회전 쪽이 낫다 (LeVJEPA)"),
        "K2_gain_L_minus_gain_D": comp(k1 - kd, "LeVJEPA의 이득이 DINOv2의 이득보다 크다", "DINOv2의 이득이 LeVJEPA의 이득보다 크다"),
        "K3_L_fit_moving_minus_fit_static": comp(hit["L|video"] - hit["L|video_static_fit"], "움직이는 클립으로 맞춘 회전 쪽이 낫다", "정지 클립으로 맞춘 회전 쪽이 낫다"),
        "K4_L_moving_minus_Ls_static": comp(hit["L|video"] - hit["Ls|video"], "움직이는 클립 쪽이 낫다", "정지 클립 쪽이 낫다"),
        "K5_L_minus_D": comp(hit["L|video"] - hit["D|video"], "LeVJEPA 쪽이 낫다", "DINOv2 쪽이 낫다"),
        "note_gain_D_video_minus_photo": comp(kd, "영상으로 맞춘 회전 쪽이 낫다 (DINOv2)", "사진으로 맞춘 회전 쪽이 낫다 (DINOv2)"),
    }
    r1 = table["L|video"]["t2v R@1"]
    cond = {"L_video_t2v_R1": r1, "rule": f"≥ {MIN_R1} (우연 {1 / n:.4f})", "aligned": bool(r1 >= MIN_R1)}
    return {"n_fit": int((~te).sum()), "n_test": n, "table": table, "comparisons": comps, "condition": cond}, rot_v


# =====================================================================  (2) 우리 자료
class SimV(A.Sim):
    """A.Sim(11절의 유사도)에 영상 회전을 쓴 팔을 더한다. 식과 평균 빼기는 같다(m은 그 자료 위치 임베딩의 평균)."""

    def __init__(self, npz, positions, rot_v):
        super().__init__(npz, positions)
        self.rot_v = rot_v

    def table(self, ds, vid, enc, center="own"):
        if enc in A.ENC:
            return super().table(ds, vid, enc, center)
        key, rot = INPUTS[NEW[enc]], self.rot_v[NEW[enc]]
        x = self.z[f"{ds}/{vid}/{key}"].astype(np.float64)
        return A.unit((x - self.pool[(ds, key)]) @ rot["R"]) @ A.unit(self.q - rot["mu_b"]).T


TAGS = ("L", "Ls", "D", "Lv", "Lsv", "Dv")


def tvsum_part(sim, rows):
    tv = E.TV()
    old_req = P.read(HA / "data" / "requests.json")
    n = {v: len(tv.items[v]["positions"]) for v in tv.videos}
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in tv.videos}
    by = {}
    for r in P.read(HA / "results" / "tvsum.json")["records"]:
        if "error" not in r:
            by.setdefault(r["video"], {})[r["k"]] = E._values(r, ["worth"] + keys_old[r["video"]])
    a_old = E._arm_scores(by, n, keys_old)
    score = {"h2o_na": {v: a_old[v]["na"] for v in tv.videos}}
    tv_rows = {}
    for r in rows:
        if r["dataset"] == "tvsum":
            tv_rows.setdefault((r["video"], r["arm"]), []).append(r)
    for tag in TAGS:
        for arm in ("writer", "title"):
            score[f"sim_{arm}:{tag}"] = {v: A.mean_rank(A.question_scores(sim.table("tvsum", v, tag, "own"), tv_rows[(v, arm)])) for v in tv.videos}
        score[f"fuse:{tag}"] = {v: 0.5 * E.pct_rank(score["h2o_na"][v]) + 0.5 * E.pct_rank(score[f"sim_writer:{tag}"][v]) for v in tv.videos}

    def label(a):
        return {"h2o_na": "판정 모델 (나)"}.get(a, a)
    pvm = {arm: {v: E.video_metrics(tv, v, score[arm][v]) for v in tv.videos} for arm in score}
    out = {"arms": {arm: {m: {"mean": s[m]["mean"], "ci": s[m]["ci"]} for m in ("tau", "lift")} for arm, s in ((a, E.summarize(tv, pvm[a])) for a in pvm)}}
    pairs = {"V1_sim_writer_Lv_minus_L": ("sim_writer:Lv", "sim_writer:L"), "V2_fuse_Lv_minus_h2o_na": ("fuse:Lv", "h2o_na"),
             "note_fuse_Lv_minus_fuse_L": ("fuse:Lv", "fuse:L"), "note_sim_writer_Dv_minus_D": ("sim_writer:Dv", "sim_writer:D"),
             "note_fuse_Dv_minus_h2o_na": ("fuse:Dv", "h2o_na"), "note_sim_writer_Lsv_minus_Ls": ("sim_writer:Lsv", "sim_writer:Ls"),
             "note_fuse_Lsv_minus_h2o_na": ("fuse:Lsv", "h2o_na"), "note_sim_title_Lv_minus_L": ("sim_title:Lv", "sim_title:L"),
             "note_sim_writer_Lv_minus_Dv": ("sim_writer:Lv", "sim_writer:Dv")}
    comps = {}
    for metric, mname in (("tau", "전체 순서(τ)"), ("lift", "맨 위(상위 15% 올림 값)")):
        g = {arm: {v: pvm[arm][v][metric] for v in tv.videos} for arm in pvm}
        for name, (a, b) in pairs.items():
            comps[f"{name}_{metric}"] = A.compare(g, tv.videos, a, b, label, "TVSum", mname)
    out["comparisons"] = comps
    return out


def mc_recs():
    """마인크래프트의 받아 둔 답(손 기준)과 큰 사건 표시. pa_eval.eval_minecraft와 같은 길."""
    import hh_eval as HE      # noqa: E402
    import hh_prepare as HP   # noqa: E402
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    stored = [r for r in HP.read(HH / "results" / "full.json")["records"] if "error" not in r]
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted(stored, key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    return recs, byv


def minecraft_part(sim, rows):
    import hh_eval as HE      # noqa: E402
    recs, byv = mc_recs()
    mc_rows = {}
    for r in rows:
        if r["dataset"] == "minecraft":
            mc_rows.setdefault(r["arm"], []).append(r)
    names = []
    for v, rs in byv.items():
        if [r["t"] for r in rs] != sim.pos["minecraft"][v]:
            raise SystemExit(f"{v}: 받아 둔 답의 위치와 임베딩의 위치가 다르다")
        hand = E.pct_rank(np.array([r["final"] for r in rs], float))
        cols = {"h2o_hand": hand}
        for tag in TAGS:
            S = sim.table("minecraft", v, tag, "own")
            for arm in ("hand", "writer"):
                cols[f"sim_{arm}:{tag}"] = E.pct_rank(A.mean_rank(A.question_scores(S, mc_rows[arm])))
            cols[f"fuse_hand:{tag}"] = E.pct_rank(0.5 * hand + 0.5 * cols[f"sim_hand:{tag}"])
        names = list(cols)
        for i, r in enumerate(rs):
            for k, c in cols.items():
                r[f"s_{k}"] = c[i]
    y = [r["big"] for r in recs]
    out = {"big_events": int(sum(y)), "arms": {}}
    for a in names:
        key = f"s_{a}"
        out["arms"][a] = {"auc_big": r4(HE.auc([r[key] for r in recs], y)),
                          "auc_big_ci": HE.boot(recs, (lambda kk: lambda rs: HE.auc([x[kk] for x in rs], [x["big"] for x in rs]))(key))[:2]}

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(HE.auc([r[f"s_{a}"] for r in rs], [r["big"] for r in rs]),
                                                                                   HE.auc([r[f"s_{b}"] for r in rs], [r["big"] for r in rs]))

    def label(a):
        return {"h2o_hand": "판정 모델 손 기준"}.get(a, a)
    comps = {}
    for name, (a, b) in {"V3_sim_hand_Lv_minus_L": ("sim_hand:Lv", "sim_hand:L"), "V4_fuse_hand_Lv_minus_h2o_hand": ("fuse_hand:Lv", "h2o_hand"),
                         "note_fuse_hand_Lv_minus_fuse_hand_L": ("fuse_hand:Lv", "fuse_hand:L"), "note_sim_hand_Dv_minus_D": ("sim_hand:Dv", "sim_hand:D"),
                         "note_fuse_hand_Dv_minus_h2o_hand": ("fuse_hand:Dv", "h2o_hand"), "note_sim_hand_Lsv_minus_Ls": ("sim_hand:Lsv", "sim_hand:Ls"),
                         "note_fuse_hand_Lsv_minus_h2o_hand": ("fuse_hand:Lsv", "h2o_hand"), "note_sim_hand_Lv_minus_Dv": ("sim_hand:Lv", "sim_hand:Dv"),
                         "note_sim_writer_Lv_minus_L": ("sim_writer:Lv", "sim_writer:L")}.items():
        lo, hi, _ = HE.boot(recs, diff(a, b))
        comps[name] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi],
                       "verdict": E.verdict(lo, hi, label(a), label(b), "마인크래프트", "AUC", "영상 12개, 큰 사건 56곳으로는")}
    out["comparisons"] = comps
    return out


def regression(out, old):
    """11절의 팔을 이 코드가 results/eval.json과 같게 다시 내는가."""
    diffs, n = [], 0
    for arm, e in out["tvsum"]["arms"].items():
        if arm in old["tvsum"]["arms"]:
            for m in ("tau", "lift"):
                n += 1
                if e[m] != {k: old["tvsum"]["arms"][arm][m][k] for k in ("mean", "ci")}:
                    diffs.append(f"tvsum {arm} {m}")
    for arm, e in out["minecraft"]["arms"].items():
        if arm in old["minecraft"]["arms"]:
            n += 1
            if e != {k: old["minecraft"]["arms"][arm][k] for k in ("auc_big", "auc_big_ci")}:
                diffs.append(f"minecraft {arm}")
    return {"checked": n, "different": diffs}


def modal_check(table, video_json):
    """Modal 안에서 저장소의 함수로 낸 영상 회전 행과 이 Mac의 값."""
    t = video_json.get("tables", {})
    if "error" in t or not t:
        return {"compared": 0, "note": f"Modal 쪽 표가 없다: {t.get('error')}"}
    worst, n = 0.0, 0
    for k in INPUTS:
        for key in ("mknn", "t2v R@1", "t2v R@5", "t2v R@10", "v2t R@1"):
            worst, n = max(worst, abs(t[k][key] - table[f"{k}|video"][key])), n + 1
    return {"compared": n, "largest_difference": r4(worst), "tolerance": TOL_MODAL, "same": bool(worst <= TOL_MODAL)}


# =====================================================================  무작위 회전 기준 (12-4절의 덧붙임)
_W = {}


def _null_init(rots):
    """작업 프로세스마다 한 번: 자료를 읽어 둔다."""
    oz = np.load(RES / "ours.npz")
    positions = P.read(HERE / "data" / "positions.json")
    q = P.read(HERE / "data" / "queries.json")
    tv = E.TV()
    tv_rows, mc_rows = {}, []
    for r in q["rows"]:
        if r["dataset"] == "tvsum" and r["arm"] == "writer":
            tv_rows.setdefault(r["video"], []).append(r)
        if r["dataset"] == "minecraft" and r["arm"] == "hand":
            mc_rows.append(r)
    _, byv = mc_recs()
    x, pool = {}, {}
    for ds in ("tvsum", "minecraft"):
        for key in INPUTS.values():
            for v in positions[ds]:
                x[(ds, v, key)] = oz[f"{ds}/{v}/{key}"].astype(np.float64)
            pool[(ds, key)] = np.concatenate([x[(ds, v, key)] for v in positions[ds]]).mean(0)
    for v, rs in byv.items():
        if [r["t"] for r in rs] != positions["minecraft"][v]:
            raise SystemExit(f"{v}: 받아 둔 답의 위치와 임베딩의 위치가 다르다")
    _W.update(q=oz["q"].astype(np.float64), tv=tv, tv_rows=tv_rows, mc_rows=mc_rows, byv=byv, x=x, pool=pool, rots=rots)


def arm_values(key, R, mu_b):
    """회전 하나 → (TVSum sim-writer의 평균 τ, 마인크래프트 sim-hand의 큰 사건 AUC). 식과 평균 빼기는 12-4절과 같다."""
    import hh_eval as HE      # noqa: E402
    tv, qc = _W["tv"], A.unit(_W["q"] - mu_b)

    def table(ds, v):
        return A.unit((_W["x"][(ds, v, key)] - _W["pool"][(ds, key)]) @ R) @ qc.T
    tau = np.mean([E.video_metrics(tv, v, A.mean_rank(A.question_scores(table("tvsum", v), _W["tv_rows"][v])))["tau"] for v in tv.videos])
    s, y = [], []
    for v, rs in _W["byv"].items():
        s.extend(E.pct_rank(A.mean_rank(A.question_scores(table("minecraft", v), _W["mc_rows"]))))
        y.extend(r["big"] for r in rs)
    return float(tau), float(HE.auc(s, y))


def _null_task(job):
    kind, name, seed = job
    rot = _W["rots"][(kind, name)]
    R = rot["R"]
    if seed is not None:                              # 같은 모양의 무작위 직교 사상
        u, _, vt = np.linalg.svd(np.random.default_rng(SEED_RANDOM * 1000 + seed).normal(size=R.shape), full_matrices=False)
        R = u @ vt
    return (kind, name, seed, *arm_values(INPUTS[name], R, rot["mu_b"]))


def run_null(src, write_to=None, n_random=N_RANDOM, workers=8):
    from multiprocessing import Pool
    vz, oz = np.load(src / "video.npz"), np.load(RES / "ours.npz")
    rot_v = video_rotations(vz)
    rots = {}
    for k in INPUTS:
        rots[("flickr", k)] = {p: oz[f"rot/{PHOTO[k]}/{p}"].astype(np.float64) for p in ("R", "mu_b")}
        rots[("msrvtt", k)] = {p: rot_v[k][p] for p in ("R", "mu_b")}
    jobs = [(kind, k, seed) for kind in ("flickr", "msrvtt") for k in INPUTS for seed in [None, *range(n_random)]]
    with Pool(workers, initializer=_null_init, initargs=(rots,)) as pool:
        done = pool.map(_null_task, jobs, chunksize=4)
    old = P.read(RES / "eval.json")
    arm_name = {("flickr", k): k for k in INPUTS} | {("msrvtt", k): n for n, k in NEW.items()}
    out = {"label": "무작위 회전 기준: 같은 평균 빼기에 무작위 직교 사상을 넣었을 때의 값", "n_random": n_random, "seed": SEED_RANDOM, "arms": {}}
    for kind in ("flickr", "msrvtt"):
        for k in INPUTS:
            fitted = next(d for d in done if d[:3] == (kind, k, None))
            rnd = np.array([d[3:] for d in done if d[:2] == (kind, k) and d[2] is not None])
            e = {"input": k, "text_mean": kind, "fitted": {"tvsum_tau": r4(fitted[3]), "minecraft_auc": r4(fitted[4])}, "random": {}, "share_random_at_or_above": {}, "above_97.5": {}}
            for j, m in enumerate(("tvsum_tau", "minecraft_auc")):
                lo, mid, hi = np.percentile(rnd[:, j], [2.5, 50, 97.5])
                e["random"][m] = {"mean": r4(rnd[:, j].mean()), "p2.5": r4(lo), "p50": r4(mid), "p97.5": r4(hi), "min": r4(rnd[:, j].min()), "max": r4(rnd[:, j].max())}
                e["share_random_at_or_above"][m] = r4(float((rnd[:, j] >= fitted[3 + j]).mean()))
                e["above_97.5"][m] = bool(fitted[3 + j] > hi)
            out["arms"][arm_name[(kind, k)]] = e
            if kind == "flickr":                      # 11절의 값과 같아야 한다
                want = (old["tvsum"]["arms"][f"sim_writer:{k}"]["tau"]["mean"], old["minecraft"]["arms"][f"sim_hand:{k}"]["auc_big"])
                if (e["fitted"]["tvsum_tau"], e["fitted"]["minecraft_auc"]) != want:
                    raise SystemExit(f"11절의 값이 다시 나오지 않았다: {k} {e['fitted']} ≠ {want}")
    if write_to:
        P.write(write_to, out)
    import json
    print(json.dumps(out["arms"], ensure_ascii=False, indent=1))
    return out


def run_eval(src, write_to=None, synthetic=False):
    vz, oz = np.load(src / "video.npz"), np.load(RES / "ours.npz")
    positions = P.read(HERE / "data" / "positions.json")
    q = P.read(HERE / "data" / "queries.json")
    ms, rot_v = msrvtt_part(vz, oz)
    sim = SimV(oz, positions, rot_v)
    out = {"label": "platonic_align 12절: 회전을 MSR-VTT의 움직이는 클립으로 맞춘 것", "synthetic": synthetic, "msrvtt": ms,
           "tvsum": tvsum_part(sim, q["rows"]), "minecraft": minecraft_part(sim, q["rows"])}
    out["judge"] = {"Lv": ms["condition"]["aligned"]}
    out["regression"] = regression(out, P.read(RES / "eval.json"))
    out["modal_check"] = modal_check(ms["table"], P.read(src / "video.json"))
    if write_to:
        P.write(write_to, out)
    brief = {"condition": ms["condition"], "judge": out["judge"], "regression": out["regression"], "modal_check": out["modal_check"],
             "msrvtt_table": {k: {m: v[m] for m in ("t2v R@1", "t2v R@5", "t2v R@10", "v2t R@1", "mknn")} for k, v in ms["table"].items()},
             "msrvtt_comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in ms["comparisons"].items()},
             "tvsum_arms": {a: [e["tau"]["mean"], e["lift"]["mean"]] for a, e in out["tvsum"]["arms"].items()},
             "tvsum_comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in out["tvsum"]["comparisons"].items()},
             "minecraft_arms": {a: e["auc_big"] for a, e in out["minecraft"]["arms"].items()},
             "minecraft_comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in out["minecraft"]["comparisons"].items()}}
    import json
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    if out["regression"]["different"]:
        raise SystemExit(f"11절의 팔이 다시 나오지 않았다: {out['regression']['different']}")
    return out


def run_synth(args):
    out = Path(args.out)
    if out.resolve() == RES.resolve() or RES.resolve() in out.resolve().parents:
        raise SystemExit("--out은 results 밖이어야 한다")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n, n_test, d = 1200, 300, 384
    z = rng.normal(size=(n, d))                                     # 장면

    def side(width, noise):                                         # 장면을 width차원에 옮기고 잡음과 치우침을 더한다
        u, _, vt = np.linalg.svd(rng.normal(size=(width, d)), full_matrices=False)
        return (z @ (u @ vt).T + noise * rng.normal(size=(n, width)) + rng.normal(size=width)).astype(np.float32)
    is_test = np.arange(n) >= n - n_test
    lev = side(1024, 0.6)
    arr = {"ids": np.array([f"video{i}" for i in range(n)]), "is_test": is_test, "js_ids": np.array([f"video{i}" for i in range(n - n_test, n)]),
           "n_frames": np.full(n, 150), "lev": lev, "lev_static": (lev + 0.8 * rng.normal(size=lev.shape)).astype(np.float32), "dino": side(768, 0.4),
           "text_mean": (z + 0.5 * rng.normal(size=(n, d)) + 1.0).astype(np.float32)}
    arr["text_first"] = (arr["text_mean"] + rng.normal(size=(n, d))).astype(np.float32)
    arr["js_text"] = (arr["text_mean"][is_test] + rng.normal(size=(n_test, d))).astype(np.float32)
    np.savez_compressed(out / "video.npz", **arr)
    P.write(out / "video.json", {"synthetic": True, "tables": {}})
    run_eval(out, write_to=out / "video_eval.json", synthetic=True)
    run_null(out, write_to=out / "video_null.json", n_random=args.n_random or 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "null", "synth"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-random", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "synth":
        if not args.out:
            raise SystemExit("synth에는 --out(results 밖의 폴더)이 필요하다")
        run_synth(args)
    elif args.stage == "null":
        run_null(RES, write_to=RES / "video_null.json", n_random=args.n_random or N_RANDOM)
    else:
        run_eval(RES, write_to=RES / "video_eval.json")


if __name__ == "__main__":
    main()
