"""platonic_align 12절 따로 계산 (이 Mac, CPU). results/video_eval.json의 주요 수를 다른 코드로 다시 내서 견준다.

pa_video_eval.py, pa_eval.py, ha_eval.py, hh_eval.py를 불러 쓰지 않는다. scipy도 쓰지 않는다.
통계와 자료 읽기는 ../highlight_probe/hp_verify.py의 직접 짠 함수(순위, tau-b, 올림 값, AUC, 다시 뽑기)를,
질문 점수와 팔의 점수는 pa_verify.py의 arm_from_rows를 쓴다. 회전 맞추기와 찾기 순위는 여기서 다시 짰다.

쓰는 것: 없음. 실행: uv run --offline --no-project --with numpy python -I experiments/platonic_align/pa_video_verify.py [결과 폴더]
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "highlight_probe"))
import hp_verify as V  # noqa: E402
import pa_verify as PV  # noqa: E402

HA = V.HA
KEYS = {"L": "lev", "Ls": "lev_static", "D": "dino"}
PHOTO = {"L": "levjepa", "Ls": "levjepa", "D": "dinov2"}
NEW = {"Lv": "L", "Lsv": "Ls", "Dv": "D"}


def rows_unit(x):
    return x / np.sqrt((x * x).sum(1, keepdims=True))


def rotation(a, b):
    """평균을 뺀 짝으로 tr(Rᵀ aᵀ b)를 가장 크게 하는 직교 사상. 극분해의 직교 인자를 Newton–Schulz 반복으로 구한다(특잇값 분해를 쓰지 않는 길).

    처음에는 m (mᵀm)^(-1/2)을 고윳값 분해로 구했는데, aᵀb의 특잇값이 10자리 넘게 벌어져 있어 mᵀm에서 정밀도가 무너졌다
    (음수 고윳값이 나왔다). 2026-11-16에 이 반복으로 바꿨다. README 12-12절.
    """
    x = a.T @ b                                    # (d_a, d_b), d_a ≥ d_b
    x = x / np.sqrt((x * x).sum())                 # 특잇값이 모두 1 이하가 되게
    eye = np.eye(x.shape[1])
    for _ in range(500):
        nxt = 0.5 * x @ (3 * eye - x.T @ x)
        done = np.abs(nxt - x).max() < 1e-13
        x = nxt
        if done:
            return x
    raise SystemExit("회전을 구하는 반복이 끝나지 않았다")


def own_rank(v, t):
    """문장마다 제 클립의 순위(1부터). 제 것보다 높거나 같은 클립의 수(제 것 포함)."""
    s = rows_unit(t) @ rows_unit(v).T
    return np.array([(s[i] >= s[i, i]).sum() for i in range(len(s))])


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    ev = V.J(src / "video_eval.json")
    vz, oz = np.load(src / "video.npz"), np.load(HERE / "results" / "ours.npz")
    pos = V.J(HERE / "data" / "positions.json")
    qj = V.J(HERE / "data" / "queries.json")
    q = oz["q"].astype(np.float64)

    print("== MSR-VTT 1k-A ==")
    te = vz["is_test"].astype(bool)
    bt, bj = vz["text_mean"][~te].astype(np.float64), vz["js_text"].astype(np.float64)
    X = {k: vz[key].astype(np.float64) for k, key in KEYS.items()}
    fitv = {}
    for k in X:
        ma, mb = X[k][~te].mean(0), bt.mean(0)
        fitv[k] = {"R": rotation(X[k][~te] - ma, bt - mb), "mu_a": ma, "mu_b": mb}
    ph = {k: {p: oz[f"rot/{PHOTO[k]}/{p}"].astype(np.float64) for p in ("R", "mu_a", "mu_b")} for k in X}
    hit = {}
    for k in X:
        own = X[k][~te].mean(0)
        hit[f"{k}|video"] = own_rank((X[k][te] - fitv[k]["mu_a"]) @ fitv[k]["R"], bj - fitv[k]["mu_b"]) <= 1
        hit[f"{k}|photo"] = own_rank((X[k][te] - own) @ ph[k]["R"], bj - ph[k]["mu_b"]) <= 1
        hit[f"{k}|photo_flickr_means"] = own_rank((X[k][te] - ph[k]["mu_a"]) @ ph[k]["R"], bj - ph[k]["mu_b"]) <= 1
    hit["L|video_static_fit"] = own_rank((X["L"][te] - X["L"][~te].mean(0)) @ fitv["Ls"]["R"], bj - fitv["Ls"]["mu_b"]) <= 1
    for name, h in hit.items():
        V.show(f"{name} 문장→클립 1위 정답률", [h.mean()], [ev["msrvtt"]["table"][name]["t2v R@1"]])
    f = {k: h.astype(float) for k, h in hit.items()}
    k1, kd = f["L|video"] - f["L|photo"], f["D|video"] - f["D|photo"]
    for name, d in {"K1_L_video_minus_L_photo": k1, "K2_gain_L_minus_gain_D": k1 - kd, "K3_L_fit_moving_minus_fit_static": f["L|video"] - f["L|video_static_fit"],
                    "K4_L_moving_minus_Ls_static": f["L|video"] - f["Ls|video"], "K5_L_minus_D": f["L|video"] - f["D|video"]}.items():
        c = ev["msrvtt"]["comparisons"][name]
        V.show(f"{name} (차이, 구간)", [d.mean(), *V.boot({"d": d})], [c["point"], *c["ci"]])

    pool = {(ds, key): np.concatenate([oz[f"{ds}/{v}/{key}"] for v in pos[ds]]).astype(np.float64).mean(0)
            for ds in ("tvsum", "minecraft") for key in KEYS.values()}

    def table(ds, vid, tag):
        if tag in NEW:
            k = NEW[tag]
            R, mb = fitv[k]["R"], fitv[k]["mu_b"]
        else:
            k = tag
            R, mb = ph[k]["R"], ph[k]["mu_b"]
        x = oz[f"{ds}/{vid}/{KEYS[k]}"].astype(np.float64) - pool[(ds, KEYS[k])]
        return rows_unit(x @ R) @ rows_unit(q - mb).T
    rows = {}
    for r in qj["rows"]:
        rows.setdefault((r["dataset"], r["video"], r["arm"]), []).append(r)

    print("== TVSum ==")
    videos, items, Y, LENS, GT = V.load_tvsum()
    n = {v: len(items[v]["positions"]) for v in videos}
    old_req = V.J(HA / "data" / "requests.json")
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in videos}
    score = {"h2o_na": V.arm_score(V.J(HA / "results" / "tvsum.json")["records"], keys_old, n, lambda r: r["k"])}
    for tag in ("L", "Lv", "Dv", "Lsv"):
        score[f"sim_writer:{tag}"] = {v: PV.arm_from_rows(table("tvsum", v, tag), rows[("tvsum", v, "writer")]) for v in videos}
        score[f"fuse:{tag}"] = {v: 0.5 * V.pct(score["h2o_na"][v]) + 0.5 * V.pct(score[f"sim_writer:{tag}"][v]) for v in videos}
    M = {a: V.tv_metrics(score[a], videos, Y, LENS, GT) for a in score}
    for a in score:
        e = ev["tvsum"]["arms"][a]
        for m in ("tau", "lift"):
            V.show(f"{a} {m} (평균, 구간)", [np.mean(M[a][m]), *V.boot({m: M[a][m]})], [e[m]["mean"], *e[m]["ci"]])
    for name, (a, b) in {"V1_sim_writer_Lv_minus_L": ("sim_writer:Lv", "sim_writer:L"), "V2_fuse_Lv_minus_h2o_na": ("fuse:Lv", "h2o_na"),
                         "note_fuse_Lv_minus_fuse_L": ("fuse:Lv", "fuse:L"), "note_fuse_Dv_minus_h2o_na": ("fuse:Dv", "h2o_na")}.items():
        for m in ("tau", "lift"):
            d = np.array(M[a][m]) - np.array(M[b][m])
            c = ev["tvsum"]["comparisons"][f"{name}_{m}"]
            V.show(f"{name} {m} (차이, 구간)", [d.mean(), *V.boot({"d": d})], [c["point"], *c["ci"]])

    print("== 마인크래프트 ==")
    mv, byv, BIG, stored = V.load_minecraft()
    S = {"h2o_hand": {v: V.pct(np.array([0.5 * stored[it["id"]]["action"]["score"] / 3 + 0.5 * stored[it["id"]]["progress"]["score"] / 3 for it in byv[v]])) for v in mv}}
    for tag in ("L", "Lv", "Dv", "Lsv"):
        S[f"sim_hand:{tag}"] = {v: V.pct(PV.arm_from_rows(table("minecraft", v, tag), rows[("minecraft", None, "hand")])) for v in mv}
        S[f"fuse_hand:{tag}"] = {v: V.pct(0.5 * S["h2o_hand"][v] + 0.5 * S[f"sim_hand:{tag}"][v]) for v in mv}
    diffs = {"V3_sim_hand_Lv_minus_L": ("sim_hand:Lv", "sim_hand:L"), "V4_fuse_hand_Lv_minus_h2o_hand": ("fuse_hand:Lv", "h2o_hand"),
             "note_fuse_hand_Lv_minus_fuse_hand_L": ("fuse_hand:Lv", "fuse_hand:L"), "note_fuse_hand_Dv_minus_h2o_hand": ("fuse_hand:Dv", "h2o_hand")}
    AUC, CI = V.mc_stats(S, mv, BIG, diffs)
    for a in S:
        e = ev["minecraft"]["arms"][a]
        V.show(f"{a} AUC, 구간", [AUC[a], *CI[a]], [e["auc_big"], *e["auc_big_ci"]], tol=6e-4)
    for k, (x, y) in diffs.items():
        c = ev["minecraft"]["comparisons"][k]
        V.show(f"{k} (차이, 구간)", [AUC[x] - AUC[y], *CI[k]], [c["point"], *c["ci"]], tol=6e-4)
    print(f"\n견준 묶음 {V.CHECKS['same'] + V.CHECKS['diff']}개 중 같음 {V.CHECKS['same']}, 다름 {V.CHECKS['diff']}")


if __name__ == "__main__":
    main()
