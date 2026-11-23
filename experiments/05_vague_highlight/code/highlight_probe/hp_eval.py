"""highlight_probe D 단계: 채점 (이 Mac, CPU). 판정 모델 요청은 없다. 규칙은 README 7·8절이고 결과를 보기 전에 정했다.

지표와 다시 뽑기는 ../highlight_auto/ha_eval.py의 함수를 그대로 쓴다(τ·ρ, 맨 위 지표, 백분위 순위, 영상째 다시 뽑기, 판정 글).
TVSum의 점수(anno.tsv)는 그쪽 폴더의 파일을 경로로 읽어 메모리에서만 쓴다. results/에는 지표만 쓴다.

단계
  eval  : results/main_tvsum__*.json, main_minecraft__*.json 채점 → results/eval.json
  synth : 가짜 답을 --out 폴더(results 밖)에 만들고 채점을 끝까지 돌려 본다(코드가 도는지만 본다)

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/highlight_probe/hp_eval.py eval

팔
  TVSum       : T(t__), TP(tp__) ← main_tvsum__tv_titled.json / P ← main_tvsum__tv_plain.json / 받아 둔 (가), (나) ← ../highlight_auto/results/tvsum.json
  마인크래프트 : G(g__), GP(gp__) ← main_minecraft__mc_game.json / P ← main_minecraft__mc_plain.json / 받아 둔 손 기준·막연한 질문·(나)
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
REQ = HERE / "data" / "main_requests.json"
r4 = E.r4


def sanity(path):
    """ha_eval.sanity_check와 같은 규칙. 파일 이름만 받는다."""
    if not Path(path).exists():
        return {"pass": False, "note": f"{Path(path).name} 없음"}
    got = {r["id"]: r["probs"] for r in P.read(path)["records"] if "error" not in r}
    old = {f"{r['vid']}@{r['k']}": r["probs"] for r in P.read(P.ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"] if "error" not in r}
    pairs = [(got[i][q], old[i][q]) for i in got if i in old for q in got[i]]
    d = [abs(a - b) for a, b in pairs]
    cross = sum((a >= 0.5) != (b >= 0.5) for a, b in pairs)
    ok = len(pairs) == 800 and sum(d) / len(d) <= E.SANITY_MAX_MEAN and cross <= E.SANITY_MAX_CROSS
    return {"pairs": len(pairs), "mean_abs_diff": r4(sum(d) / max(1, len(d))), "max_abs_diff": r4(max(d) if d else None), "crossed_0_5": cross,
            "rule": f"800쌍 모두, 평균 ≤ {E.SANITY_MAX_MEAN}, 0.5를 넘나든 것 ≤ {E.SANITY_MAX_CROSS}", "pass": ok}


def share(data):
    recs = data["records"]
    bad = sum("error" in r for r in recs)
    want = data["meta"]["requests"]
    return {"requests": want, "answered": len(recs) - bad, "errors": bad + (want - len(recs)), "error_share": r4((bad + want - len(recs)) / max(1, want)),
            "partial": data["meta"].get("partial")}


def compare(per_video, vids, a, b, label, where, metric_name, n_label="영상 50개로는"):
    d = {v: per_video[a][v] - per_video[b][v] for v in vids}
    lo, hi = E.boot_ci({"d": d}, vids)
    return {"point": r4(np.mean(list(d.values()))), "ci": [lo, hi], "verdict": E.verdict(lo, hi, label[a], label[b], where, metric_name, n_label)}


# =====================================================================  TVSum
def eval_tvsum(res):
    tv = E.TV()
    req, old_req = P.read(REQ), P.read(HA / "data" / "requests.json")
    n = {v: len(tv.items[v]["positions"]) for v in tv.videos}
    old = P.read(HA / "results" / "tvsum.json")
    keys_old = {v: [k for k in old_req["tvsum"]["videos"][v]["questions"] if k != "worth"] for v in tv.videos}
    by = {}
    for r in old["records"]:
        if "error" not in r:
            by.setdefault(r["video"], {})[r["k"]] = E._values(r, ["worth"] + keys_old[r["video"]])
    a_old = E._arm_scores(by, n, keys_old)
    score = {"ga": {v: a_old[v]["ga"] for v in tv.videos}, "na": {v: a_old[v]["na"] for v in tv.videos}}
    files = {"tv_titled": P.read(res / "main_tvsum__tv_titled.json"), "tv_plain": P.read(res / "main_tvsum__tv_plain.json")}
    n_q = {}
    for arm, (bundle, prefix) in {"T": ("tv_titled", "t__"), "TP": ("tv_titled", "tp__"), "P": ("tv_plain", "")}.items():
        keys = {v: [k for k in req["arms"][bundle]["videos"][v]["questions"] if k.startswith(prefix)] for v in tv.videos}
        by = {}
        for r in files[bundle]["records"]:
            if "error" not in r:
                by.setdefault(r["video"], {})[r["k"]] = E._values(r, keys[r["video"]])
        a = E._arm_scores(by, n, keys, worth_key=None)
        score[arm] = {v: a[v]["na"] for v in tv.videos}
        n_q[arm] = sum(len(k) for k in keys.values())
    label = {"ga": "막연한 질문(가)", "na": "지난 기준(나)", "T": "새로 쓴 제목 기준(T)", "TP": "제목+요약 기준(TP)", "P": "요약만 본 기준(P)"}
    pvm = {arm: {v: E.video_metrics(tv, v, score[arm][v]) for v in tv.videos} for arm in score}
    out = {"bundles": {b: share(d) for b, d in files.items()}, "questions": n_q, "arms": {arm: {"label": label[arm], **E.summarize(tv, pvm[arm])} for arm in pvm}}
    comps = {}
    for metric, mname in (("tau", "전체 순서(τ)"), ("lift", "맨 위(상위 15% 올림 값)")):
        g = {arm: {v: pvm[arm][v][metric] for v in tv.videos} for arm in pvm}
        for name, (a, b) in {"H1_TP_minus_T": ("TP", "T"), "H2_P_minus_T": ("P", "T"), "H3_P_minus_ga": ("P", "ga"),
                             "note_T_minus_na": ("T", "na"), "note_TP_minus_na": ("TP", "na"), "note_P_minus_na": ("P", "na"),
                             "note_T_minus_ga": ("T", "ga"), "note_TP_minus_ga": ("TP", "ga")}.items():
            comps[f"{name}_{metric}"] = compare(g, tv.videos, a, b, label, "TVSum", mname)
    out["comparisons"] = comps
    out["stored_check"] = {"ga_tau": out["arms"]["ga"]["tau"]["mean"], "na_tau": out["arms"]["na"]["tau"]["mean"], "expected_from_highlight_auto": {"ga_tau": 0.1658, "na_tau": 0.2122}}
    out["meta"] = {b: d.get("meta") for b, d in files.items()}
    return out, score


# =====================================================================  마인크래프트
def eval_minecraft(res):
    import hh_eval as HE      # noqa: E402
    import hh_prepare as HP   # noqa: E402
    req, old_req = P.read(REQ), P.read(HA / "data" / "requests.json")
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    stored = [r for r in HP.read(HH / "results" / "full.json")["records"] if "error" not in r]
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted(stored, key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    files = {"mc_game": P.read(res / "main_minecraft__mc_game.json"), "mc_plain": P.read(res / "main_minecraft__mc_plain.json")}
    old_keys = list(old_req["minecraft"]["questions"])
    answers = {"na": {r["id"]: E._values(r, old_keys) for r in P.read(HA / "results" / "minecraft.json")["records"] if "error" not in r}}
    keys = {"na": lambda v: old_keys}
    for arm, (bundle, prefix) in {"G": ("mc_game", "g__"), "GP": ("mc_game", "gp__"), "P": ("mc_plain", "")}.items():
        kv = {v: [k for k in e["questions"] if k.startswith(prefix)] for v, e in req["arms"][bundle]["videos"].items()}
        keys[arm] = (lambda kv_: lambda v: kv_[v])(kv)
        answers[arm] = {r["id"]: E._values(r, kv[r["video"]]) for r in files[bundle]["records"] if "error" not in r}
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    for v, rs in byv.items():
        def col(f):
            return np.array([f(r) for r in rs], float)
        hand, vague = E.pct_rank(col(lambda r: r["final"])), E.pct_rank(col(lambda r: r["worth"]))
        for i, r in enumerate(rs):
            r["s_hand"], r["s_vague"] = hand[i], vague[i]
        for arm in ("na", "G", "GP", "P"):
            ranks = []
            for k in keys[arm](v):
                c = col(lambda r: answers[arm].get(r["id"], {}).get(k, np.nan))
                rk = np.full(len(rs), 0.5)
                ok = ~np.isnan(c)
                if ok.any():
                    rk[ok] = E.pct_rank(c[ok])
                ranks.append(rk)
            s = E.pct_rank(np.mean(ranks, axis=0))
            for i, r in enumerate(rs):
                r[f"s_{arm}"] = s[i]
    label = {"hand": "손 기준(받아 둔 final)", "vague": "막연한 질문(받아 둔 worth)", "na": "지난 기준(나)", "G": "새로 쓴 게임 기준(G)",
             "GP": "게임+요약 기준(GP)", "P": "요약만 본 기준(P)"}
    y = [r["big"] for r in recs]
    out = {"bundles": {b: share(d) for b, d in files.items()}, "big_events": int(sum(y)), "arms": {}}
    for a in label:
        key = f"s_{a}"
        out["arms"][a] = {"label": label[a], "auc_big": r4(HE.auc([r[key] for r in recs], y)),
                          "auc_big_ci": HE.boot(recs, (lambda kk: lambda rs: HE.auc([x[kk] for x in rs], [x["big"] for x in rs]))(key))[:2],
                          "ap_big": r4(HE.ap([r[key] for r in recs], y)),
                          "auc_fight": E.r3(HE.auc([r[key] for r in recs], [r["fight"] for r in recs])),
                          "auc_prog": E.r3(HE.auc([r[key] for r in recs], [r["prog"] for r in recs]))}

    def diff(a, b):
        return lambda rs: (lambda x, z: None if x is None or z is None else x - z)(HE.auc([r[f"s_{a}"] for r in rs], [r["big"] for r in rs]),
                                                                                   HE.auc([r[f"s_{b}"] for r in rs], [r["big"] for r in rs]))
    comps = {}
    for name, (a, b) in {"M4_GP_minus_na": ("GP", "na"), "M5_P_minus_na": ("P", "na"), "M6_GP_minus_hand": ("GP", "hand"),
                         "note_G_minus_na": ("G", "na"), "note_GP_minus_G": ("GP", "G"), "note_P_minus_hand": ("P", "hand"), "note_P_minus_vague": ("P", "vague")}.items():
        lo, hi, _ = HE.boot(recs, diff(a, b))
        comps[name] = {"point": r4(out["arms"][a]["auc_big"] - out["arms"][b]["auc_big"]), "ci": [lo, hi],
                       "verdict": E.verdict(lo, hi, label[a], label[b], "마인크래프트", "AUC", "영상 12개, 큰 사건 56곳으로는")}
    out["comparisons"] = comps
    out["stored_check"] = {"hand": out["arms"]["hand"]["auc_big"], "vague": out["arms"]["vague"]["auc_big"], "na": out["arms"]["na"]["auc_big"],
                           "expected_from_highlight_auto": {"hand": 0.8468, "vague": 0.7203, "na": 0.6619}}
    out["meta"] = {b: d.get("meta") for b, d in files.items()}
    return out, recs


def run_eval(res, write_to=None):
    human = P.read(HA / "results" / "human.json")
    out = {"label": "highlight_probe 본 실행의 채점. 판정 모델(H2O) H2O-Lightning-4B 판 659fc412",
           "sanity": {"main_tvsum": sanity(res / "main_tvsum__sanity.json"), "main_minecraft": sanity(res / "main_minecraft__sanity.json")},
           "human_reproduced": human["reproduced_3dp"]}
    out["tvsum"], _ = eval_tvsum(res)
    out["minecraft"], _ = eval_minecraft(res)
    out["tvsum"]["judge"] = bool(out["sanity"]["main_tvsum"]["pass"] and out["human_reproduced"]
                                 and all(b["error_share"] <= E.MAX_ERROR_SHARE and not b["partial"] for b in out["tvsum"]["bundles"].values()))
    out["minecraft"]["judge"] = bool(out["sanity"]["main_minecraft"]["pass"]
                                     and all(b["error_share"] <= E.MAX_ERROR_SHARE and not b["partial"] for b in out["minecraft"]["bundles"].values()))
    if write_to:
        P.write(write_to, out)
    brief = {"sanity": out["sanity"], "human_reproduced": out["human_reproduced"]}
    for part in ("tvsum", "minecraft"):
        o = out[part]
        brief[part] = {"judge": o["judge"], "bundles": o["bundles"], "stored_check": o["stored_check"],
                       "arms": {a: ({"tau": v["tau"]["mean"], "tau_ci": v["tau"]["ci"], "lift": v["lift"]["mean"], "lift_ci": v["lift"]["ci"]} if part == "tvsum"
                                    else {"auc_big": v["auc_big"], "ci": v["auc_big_ci"], "fight": v["auc_fight"], "prog": v["auc_prog"]}) for a, v in o["arms"].items()},
                       "comparisons": {k: [v["point"], v["ci"], v["verdict"]] for k, v in o["comparisons"].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return out


def run_synth(args):
    """가짜 답(무작위)을 results 밖에 만들고 채점을 끝까지 돌린다. 값에는 뜻이 없다."""
    sys.path.insert(0, str(HH))
    import hh_prepare as HP   # noqa: E402
    out = Path(args.out)
    if out.resolve() == RES.resolve() or RES.resolve() in out.resolve().parents:
        raise SystemExit("--out은 results 밖이어야 한다")
    out.mkdir(parents=True, exist_ok=True)
    req = P.read(REQ)
    rng = np.random.default_rng(0)

    def ans(qs):
        a = {}
        for k, q in qs.items():
            if q["type"] == "noul":
                a[k] = {"noul": float(rng.random())}
            else:
                p = rng.dirichlet(np.ones(len(q["criteria"])))
                a[k] = {"score": float(sum(i * x for i, x in enumerate(p))), "probabilities": {str(i): float(x) for i, x in enumerate(p)}}
        return a
    for bundle in ("tv_titled", "tv_plain"):
        recs = [{"id": f"{v}@{p['k']}", "video": v, "k": p["k"], "answers": ans(e["questions"])} for v, e in req["arms"][bundle]["videos"].items() for p in req["positions"][v]]
        recs[3] = {"id": recs[3]["id"], "video": recs[3]["video"], "k": recs[3]["k"], "error": "synthetic"}
        P.write(out / f"main_tvsum__{bundle}.json", {"meta": {"requests": len(recs), "partial": None, "synthetic": True}, "records": recs})
    items = HP.load_items()["full"]
    for bundle in ("mc_game", "mc_plain"):
        recs = [{"id": it["id"], "video": it["video"], "t": it["t"], "answers": ans(req["arms"][bundle]["videos"][it["video"]]["questions"])} for it in items]
        P.write(out / f"main_minecraft__{bundle}.json", {"meta": {"requests": len(recs), "partial": None, "synthetic": True}, "records": recs})
    old = P.read(P.ROOT / "experiments" / "mc_dense" / "results" / "sep4_half.json")["records"]
    keep = [r for r in old if "error" not in r and r["vid"] in HP.SANITY_VIDEOS and r["k"] < HP.SANITY_K]
    srec = [{"id": f"{r['vid']}@{r['k']}", "video": r["vid"], "k": r["k"], "probs": r["probs"]} for r in keep]
    for fn in ("main_tvsum", "main_minecraft"):
        P.write(out / f"{fn}__sanity.json", {"meta": {"requests": len(srec), "synthetic": True}, "records": srec})
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
