"""platonic_align 따로 계산 (이 Mac, CPU). results/eval.json의 주요 수를 다른 코드로 다시 내서 견준다.

pa_eval.py, ha_eval.py, hh_eval.py를 불러 쓰지 않는다. scipy도 쓰지 않는다.
통계와 자료 읽기는 ../highlight_probe/hp_verify.py의 직접 짠 함수(순위, tau-b, 올림 값, AUC, 다시 뽑기)를 쓴다.
유사도와 질문 점수, 팔의 점수는 여기서 다시 짰다(README 5절의 식).

쓰는 것: 없음. 실행: uv run --offline --no-project --with numpy python -I experiments/platonic_align/pa_verify.py
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highlight_probe"))
import hp_verify as V  # noqa: E402

HA, HH = V.HA, V.HH
KEY = {"D": ("dino", "dinov2"), "L": ("lev", "levjepa"), "Ls": ("lev_static", "levjepa")}


def cosine_table(z, q, pool, ds, vid, enc):
    key, rot = KEY[enc]
    R = z[f"rot/{rot}/R"].astype(np.float64)
    mu_b = z[f"rot/{rot}/mu_b"].astype(np.float64)
    a = (z[f"{ds}/{vid}/{key}"].astype(np.float64) - pool[(ds, key)]) @ R
    b = q - mu_b
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def arm_from_rows(S, rows):
    """한 팔의 줄들 → 질문마다의 점수 → 순위의 같은 무게 평균."""
    by = {}
    for r in rows:
        by.setdefault(r["question"], {})[r["part"]] = r["text_index"]
        by[r["question"]]["_type"] = r["type"]
    ranks = []
    for parts in by.values():
        t = parts["_type"]
        if t == "title":
            s = S[:, parts["text"]]
        elif t == "noul":
            s = S[:, parts["true"]] - S[:, parts["false"]]
        else:
            n = sum(k.startswith("level") for k in parts)
            s = np.zeros(len(S))
            for i in range(n):
                s = s + (i - (n - 1) / 2) * S[:, parts[f"level{i}"]]
        ranks.append(V.pct(s))
    return np.mean(ranks, axis=0)


def main():
    ev = V.J(HERE / "results" / "eval.json")
    z = np.load(HERE / "results" / "ours.npz")
    pos = V.J(HERE / "data" / "positions.json")
    qj = V.J(HERE / "data" / "queries.json")
    q = z["q"].astype(np.float64)
    pool = {(ds, key): np.concatenate([z[f"{ds}/{v}/{key}"] for v in pos[ds]]).astype(np.float64).mean(0)
            for ds in ("tvsum", "minecraft") for key in ("dino", "lev", "lev_static")}
    old_req = V.J(HA / "data" / "requests.json")

    print("== TVSum ==")
    videos, items, Y, LENS, GT = V.load_tvsum()
    n = {v: len(items[v]["positions"]) for v in videos}
    old_recs = V.J(HA / "results" / "tvsum.json")["records"]
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in videos}
    score = {"h2o_na": V.arm_score(old_recs, keys_old, n, lambda r: r["k"])}
    rows = {}
    for r in qj["rows"]:
        rows.setdefault((r["dataset"], r["video"], r["arm"]), []).append(r)
    for enc in ("D", "L"):
        score[f"sim_writer:{enc}"] = {v: arm_from_rows(cosine_table(z, q, pool, "tvsum", v, enc), rows[("tvsum", v, "writer")]) for v in videos}
        score[f"fuse:{enc}"] = {v: 0.5 * V.pct(score["h2o_na"][v]) + 0.5 * V.pct(score[f"sim_writer:{enc}"][v]) for v in videos}
    M = {a: V.tv_metrics(score[a], videos, Y, LENS, GT) for a in score}
    for a in score:
        e = ev["tvsum"]["arms"][a]
        for m in ("tau", "lift"):
            V.show(f"{a} {m} (평균, 구간)", [np.mean(M[a][m]), *V.boot({m: M[a][m]})], [e[m]["mean"], *e[m]["ci"]])
    for name, (a, b) in {"A1_sim_writer_L_minus_h2o_na": ("sim_writer:L", "h2o_na"), "A2_fuse_L_minus_h2o_na": ("fuse:L", "h2o_na"),
                         "A3_sim_writer_L_minus_D": ("sim_writer:L", "sim_writer:D"), "note_fuse_D_minus_h2o_na": ("fuse:D", "h2o_na")}.items():
        for m in ("tau", "lift"):
            d = np.array(M[a][m]) - np.array(M[b][m])
            c = ev["tvsum"]["comparisons"][f"{name}_{m}"]
            V.show(f"{name} {m} (차이, 구간)", [d.mean(), *V.boot({"d": d})], [c["point"], *c["ci"]])

    print("== 마인크래프트 ==")
    mv, byv, BIG, stored = V.load_minecraft()
    S = {"h2o_hand": {v: V.pct(np.array([0.5 * stored[it["id"]]["action"]["score"] / 3 + 0.5 * stored[it["id"]]["progress"]["score"] / 3 for it in byv[v]])) for v in mv}}
    for enc in ("D", "L"):
        S[f"sim_hand:{enc}"] = {v: V.pct(arm_from_rows(cosine_table(z, q, pool, "minecraft", v, enc), rows[("minecraft", None, "hand")])) for v in mv}
        S[f"fuse_hand:{enc}"] = {v: V.pct(0.5 * S["h2o_hand"][v] + 0.5 * S[f"sim_hand:{enc}"][v]) for v in mv}
    diffs = {"B1_sim_hand_L_minus_h2o_hand": ("sim_hand:L", "h2o_hand"), "B2_fuse_hand_L_minus_h2o_hand": ("fuse_hand:L", "h2o_hand"),
             "note_fuse_hand_D_minus_h2o_hand": ("fuse_hand:D", "h2o_hand")}
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
