"""platonic_align 채점 (이 Mac, CPU). 규칙은 README 5·6절이고 임베딩을 받기 전에 썼다.

지표와 다시 뽑기는 ../highlight_auto/ha_eval.py, ../highlight_h2o/hh_eval.py의 함수를 그대로 쓴다.
TVSum의 점수(anno.tsv)는 그쪽 폴더의 파일을 경로로 읽어 메모리에서만 쓴다. results/에는 지표만 쓴다.

단계
  eval  : results/ours.npz(임베딩과 회전), results/embed.json(Flickr 정렬 표) 채점 → results/eval.json
  synth : 가짜 임베딩을 --out 폴더(results 밖)에 만들고 채점을 끝까지 돌려 본다

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/platonic_align/pa_eval.py eval
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
HA = HERE.parent / "highlight_auto"
HH = HERE.parent / "highlight_h2o"
for p in (HA, HH):
    sys.path.insert(0, str(p))
import ha_eval as E  # noqa: E402
import ha_prepare as P  # noqa: E402

RES = HERE / "results"
r4 = E.r4
ENC = {"D": ("dino", "dinov2"), "L": ("lev", "levjepa"), "Ls": ("lev_static", "levjepa")}     # 팔 이름 → (임베딩 키, 회전 이름)
MIN_R1_D, MIN_R1_L = 0.15, 0.01          # README 6절의 판정하는 조건


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Sim:
    """위치 임베딩과 문장 임베딩의 유사도 s = cos((x − m) R, q − μ_b). m은 그 자료의 위치 평균(주) 또는 Flickr 그림 평균(보조)."""

    def __init__(self, npz, positions):
        self.z = npz
        self.q = npz["q"].astype(np.float64)
        self.pos = positions
        self.pool = {}
        for ds in ("tvsum", "minecraft"):
            for key in ("dino", "lev", "lev_static"):
                self.pool[(ds, key)] = np.concatenate([npz[f"{ds}/{v}/{key}"] for v in positions[ds]]).astype(np.float64).mean(0)

    def table(self, ds, vid, enc, center):
        key, rot = ENC[enc]
        R, mu_a, mu_b = (self.z[f"rot/{rot}/{k}"].astype(np.float64) for k in ("R", "mu_a", "mu_b"))
        x = self.z[f"{ds}/{vid}/{key}"].astype(np.float64)
        m = self.pool[(ds, key)] if center == "own" else mu_a
        return unit((x - m) @ R) @ unit(self.q - mu_b).T          # (위치, 문장)


def question_scores(S, rows):
    """rows: 한 영상(또는 자료)의 한 팔에 속한 줄들 → {질문: 위치별 점수}. README 5절의 식."""
    by = {}
    for r in rows:
        by.setdefault(r["question"], []).append(r)
    out = {}
    for qn, rs in by.items():
        t = rs[0]["type"]
        part = {r["part"]: r["text_index"] for r in rs}
        if t == "title":
            out[qn] = S[:, part["text"]]
        elif t == "noul":
            out[qn] = S[:, part["true"]] - S[:, part["false"]]
        else:
            L = len(part)
            out[qn] = sum((i - (L - 1) / 2) * S[:, part[f"level{i}"]] for i in range(L))
    return out


def mean_rank(qs):
    return np.mean([E.pct_rank(v) for v in qs.values()], axis=0)


def compare(per_video, vids, a, b, label, where, metric_name):
    d = {v: per_video[a][v] - per_video[b][v] for v in vids}
    lo, hi = E.boot_ci({"d": d}, vids)
    return {"point": r4(np.mean(list(d.values()))), "ci": [lo, hi], "verdict": E.verdict(lo, hi, label(a), label(b), where, metric_name)}


# =====================================================================  TVSum
def eval_tvsum(sim, rows):
    tv = E.TV()
    old_req = P.read(HA / "data" / "requests.json")
    n = {v: len(tv.items[v]["positions"]) for v in tv.videos}
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in tv.videos}
    by = {}
    for r in P.read(HA / "results" / "tvsum.json")["records"]:
        if "error" not in r:
            by.setdefault(r["video"], {})[r["k"]] = E._values(r, ["worth"] + keys_old[r["video"]])
    a_old = E._arm_scores(by, n, keys_old)
    score = {"h2o_ga": {v: a_old[v]["ga"] for v in tv.videos}, "h2o_na": {v: a_old[v]["na"] for v in tv.videos}}
    tv_rows = {}
    for r in rows:
        if r["dataset"] == "tvsum":
            tv_rows.setdefault((r["video"], r["arm"]), []).append(r)
    for enc in ENC:
        for center in ("own", "flickr"):
            tag = enc if center == "own" else f"{enc}@flickr"
            for arm in ("writer", "title"):
                name = f"sim_{arm}:{tag}"
                score[name] = {}
                for v in tv.videos:
                    S = sim.table("tvsum", v, enc, center)
                    if len(S) != n[v]:
                        raise SystemExit(f"{v}: 임베딩 {len(S)}곳 ≠ 위치 {n[v]}곳")
                    score[name][v] = mean_rank(question_scores(S, tv_rows[(v, arm)]))
            score[f"fuse:{tag}"] = {v: 0.5 * E.pct_rank(score["h2o_na"][v]) + 0.5 * E.pct_rank(score[f"sim_writer:{tag}"][v]) for v in tv.videos}

    def label(a):
        return {"h2o_ga": "판정 모델 막연한 질문(가)", "h2o_na": "판정 모델 (나)"}.get(a, a)
    pvm = {arm: {v: E.video_metrics(tv, v, score[arm][v]) for v in tv.videos} for arm in score}
    out = {"arms": {arm: {m: {"mean": s[m]["mean"], "ci": s[m]["ci"]} for m in ("tau", "rho", "lift", "hit5", "ap15")}
                    for arm, s in ((a, E.summarize(tv, pvm[a])) for a in pvm)}}
    comps = {}
    for metric, mname in (("tau", "전체 순서(τ)"), ("lift", "맨 위(상위 15% 올림 값)")):
        g = {arm: {v: pvm[arm][v][metric] for v in tv.videos} for arm in pvm}
        pairs = {"A1_sim_writer_L_minus_h2o_na": ("sim_writer:L", "h2o_na"), "A2_fuse_L_minus_h2o_na": ("fuse:L", "h2o_na"),
                 "A3_sim_writer_L_minus_D": ("sim_writer:L", "sim_writer:D"),
                 "note_sim_writer_D_minus_h2o_na": ("sim_writer:D", "h2o_na"), "note_fuse_D_minus_h2o_na": ("fuse:D", "h2o_na"),
                 "note_fuse_Ls_minus_h2o_na": ("fuse:Ls", "h2o_na"), "note_sim_writer_Ls_minus_L": ("sim_writer:Ls", "sim_writer:L"),
                 "note_sim_title_L_minus_h2o_ga": ("sim_title:L", "h2o_ga"), "note_sim_title_D_minus_h2o_ga": ("sim_title:D", "h2o_ga")}
        for name, (a, b) in pairs.items():
            comps[f"{name}_{metric}"] = compare(g, tv.videos, a, b, label, "TVSum", mname)
    out["comparisons"] = comps
    out["stored_check"] = {"h2o_ga_tau": out["arms"]["h2o_ga"]["tau"]["mean"], "h2o_na_tau": out["arms"]["h2o_na"]["tau"]["mean"],
                           "expected_from_highlight_auto": {"ga_tau": 0.1658, "na_tau": 0.2122}}
    return out


# =====================================================================  마인크래프트
def eval_minecraft(sim, rows):
    import hh_eval as HE      # noqa: E402
    import hh_prepare as HP   # noqa: E402
    old_req = P.read(HA / "data" / "requests.json")
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    stored = [r for r in HP.read(HH / "results" / "full.json")["records"] if "error" not in r]
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted(stored, key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    old_keys = list(old_req["minecraft"]["questions"])
    na = {r["id"]: E._values(r, old_keys) for r in P.read(HA / "results" / "minecraft.json")["records"] if "error" not in r}
    mc_rows = {}
    for r in rows:
        if r["dataset"] == "minecraft":
            mc_rows.setdefault(r["arm"], []).append(r)
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    names = []
    for v, rs in byv.items():
        centers = [r["t"] for r in rs]
        if centers != sim.pos["minecraft"][v]:
            raise SystemExit(f"{v}: 받아 둔 답의 위치와 임베딩의 위치가 다르다")

        def col(f):
            return np.array([f(r) for r in rs], float)
        hand = E.pct_rank(col(lambda r: r["final"]))
        ranks = []
        for k in old_keys:
            c = col(lambda r: na.get(r["id"], {}).get(k, np.nan))
            rk = np.full(len(rs), 0.5)
            ok = ~np.isnan(c)
            if ok.any():
                rk[ok] = E.pct_rank(c[ok])
            ranks.append(rk)
        s_na = E.pct_rank(np.mean(ranks, axis=0))
        cols = {"h2o_hand": hand, "h2o_vague": E.pct_rank(col(lambda r: r["worth"])), "h2o_na": s_na}
        for enc in ENC:
            for center in ("own", "flickr"):
                tag = enc if center == "own" else f"{enc}@flickr"
                S = sim.table("minecraft", v, enc, center)
                for arm in ("hand", "writer"):
                    cols[f"sim_{arm}:{tag}"] = E.pct_rank(mean_rank(question_scores(S, mc_rows[arm])))
                cols[f"fuse_hand:{tag}"] = E.pct_rank(0.5 * hand + 0.5 * cols[f"sim_hand:{tag}"])
                cols[f"fuse_writer:{tag}"] = E.pct_rank(0.5 * s_na + 0.5 * cols[f"sim_writer:{tag}"])
        names = list(cols)
        for i, r in enumerate(rs):
            for k, c in cols.items():
                r[f"s_{k}"] = c[i]
    y = [r["big"] for r in recs]
    out = {"big_events": int(sum(y)), "arms": {}}
    for a in names:
        key = f"s_{a}"
        out["arms"][a] = {"auc_big": r4(HE.auc([r[key] for r in recs], y)),
                          "auc_big_ci": HE.boot(recs, (lambda kk: lambda rs: HE.auc([x[kk] for x in rs], [x["big"] for x in rs]))(key))[:2],
                          "auc_fight": E.r3(HE.auc([r[key] for r in recs], [r["fight"] for r in recs])),
                          "auc_prog": E.r3(HE.auc([r[key] for r in recs], [r["prog"] for r in recs]))}

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(HE.auc([r[f"s_{a}"] for r in rs], [r["big"] for r in rs]),
                                                                                   HE.auc([r[f"s_{b}"] for r in rs], [r["big"] for r in rs]))

    def label(a):
        return {"h2o_hand": "판정 모델 손 기준", "h2o_vague": "판정 모델 막연한 질문", "h2o_na": "판정 모델 (나)"}.get(a, a)
    comps = {}
    for name, (a, b) in {"B1_sim_hand_L_minus_h2o_hand": ("sim_hand:L", "h2o_hand"), "B2_fuse_hand_L_minus_h2o_hand": ("fuse_hand:L", "h2o_hand"),
                         "note_sim_hand_D_minus_h2o_hand": ("sim_hand:D", "h2o_hand"), "note_fuse_hand_D_minus_h2o_hand": ("fuse_hand:D", "h2o_hand"),
                         "note_fuse_hand_Ls_minus_h2o_hand": ("fuse_hand:Ls", "h2o_hand"), "note_sim_hand_L_minus_D": ("sim_hand:L", "sim_hand:D"),
                         "note_sim_writer_L_minus_h2o_na": ("sim_writer:L", "h2o_na"), "note_fuse_writer_L_minus_h2o_na": ("fuse_writer:L", "h2o_na")}.items():
        lo, hi, _ = HE.boot(recs, diff(a, b))
        comps[name] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi],
                       "verdict": E.verdict(lo, hi, label(a), label(b), "마인크래프트", "AUC", "영상 12개, 큰 사건 56곳으로는")}
    out["comparisons"] = comps
    out["stored_check"] = {"hand": out["arms"]["h2o_hand"]["auc_big"], "na": out["arms"]["h2o_na"]["auc_big"],
                           "expected_from_highlight_auto": {"hand": 0.8468, "na": 0.6619}}
    return out


def run_eval(res, write_to=None):
    npz = np.load(res / "ours.npz")
    emb = P.read(res / "embed.json")
    positions = P.read(HERE / "data" / "positions.json")
    q = P.read(HERE / "data" / "queries.json")
    if npz["q"].shape[0] != len(q["texts"]):
        raise SystemExit("문장 임베딩의 수가 queries.json과 다르다")
    t = emb["tables"]
    cond = {"dinov2_R1_fit_1k_val": r4(t["dinov2"]["fit_1k_val"]["R@1"]), "levjepa_R1_fit_1k_val": r4(t["levjepa"]["fit_1k_val"]["R@1"]),
            "rule": f"DINOv2 ≥ {MIN_R1_D} (저장소 기대값 약 0.25), LeVJEPA ≥ {MIN_R1_L} (우연 0.001)"}
    cond["reproduced"] = bool(t["dinov2"]["fit_1k_val"]["R@1"] >= MIN_R1_D)
    cond["levjepa_aligned"] = bool(t["levjepa"]["fit_1k_val"]["R@1"] >= MIN_R1_L)
    sim = Sim(npz, positions)
    out = {"label": "platonic_align 채점. 글 JEPA(BERT-S, Flickr 캡션) + DINOv2 ViT-B/14, LeVJEPA-1.5-Large + 회전",
           "conditions": cond, "flickr_tables": t, "tvsum": eval_tvsum(sim, q["rows"]), "minecraft": eval_minecraft(sim, q["rows"])}
    out["judge"] = {"D": cond["reproduced"], "L": cond["reproduced"] and cond["levjepa_aligned"]}
    if write_to:
        P.write(write_to, out)
    brief = {"conditions": cond, "judge": out["judge"],
             "flickr": {k: {f: {m: r4(x) for m, x in v.items()} for f, v in t[k].items()} for k in ("dinov2", "dinov2_on_levjepa_rows", "levjepa")}}
    for part in ("tvsum", "minecraft"):
        o = out[part]
        show = [a for a in o["arms"] if "@flickr" not in a]
        brief[part] = {"stored_check": o["stored_check"],
                       "arms": {a: ({"tau": o["arms"][a]["tau"]["mean"], "lift": o["arms"][a]["lift"]["mean"]} if part == "tvsum"
                                    else {"auc": o["arms"][a]["auc_big"], "fight": o["arms"][a]["auc_fight"], "prog": o["arms"][a]["auc_prog"]}) for a in show},
                       "comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in o["comparisons"].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return out


def run_synth(args):
    out = Path(args.out)
    if out.resolve() == RES.resolve() or RES.resolve() in out.resolve().parents:
        raise SystemExit("--out은 results 밖이어야 한다")
    out.mkdir(parents=True, exist_ok=True)
    pos = P.read(HERE / "data" / "positions.json")
    q = P.read(HERE / "data" / "queries.json")
    rng = np.random.default_rng(0)
    arr = {"q": rng.normal(size=(len(q["texts"]), 384)).astype(np.float32)}
    for name, d in (("dinov2", 768), ("levjepa", 1024)):
        u, _, vt = np.linalg.svd(rng.normal(size=(d, 384)), full_matrices=False)
        arr[f"rot/{name}/R"] = (u @ vt).astype(np.float32)
        arr[f"rot/{name}/mu_a"] = rng.normal(size=d).astype(np.float32)
        arr[f"rot/{name}/mu_b"] = rng.normal(size=384).astype(np.float32)
    for ds in ("tvsum", "minecraft"):
        for v, cs in pos[ds].items():
            arr[f"{ds}/{v}/dino"] = rng.normal(size=(len(cs), 768)).astype(np.float32)
            arr[f"{ds}/{v}/lev"] = rng.normal(size=(len(cs), 1024)).astype(np.float32)
            arr[f"{ds}/{v}/lev_static"] = rng.normal(size=(len(cs), 1024)).astype(np.float32)
    np.savez_compressed(out / "ours.npz", **arr)
    fake = {"mknn": 0.05, "R@1": 0.2, "R@5": 0.5, "wR@1": 0.2, "wR@5": 0.5, "cos": 0.3}
    P.write(out / "embed.json", {"synthetic": True, "tables": {k: {"fit_all": fake, "fit_1k_val": fake} for k in ("dinov2", "dinov2_on_levjepa_rows", "levjepa")}})
    run_eval(out, write_to=out / "eval.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "synth"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.stage == "synth":
        if not args.out:
            raise SystemExit("synth에는 --out(results 밖의 폴더)이 필요하다")
        run_synth(args)
    else:
        run_eval(RES, write_to=RES / "eval.json")


if __name__ == "__main__":
    main()
