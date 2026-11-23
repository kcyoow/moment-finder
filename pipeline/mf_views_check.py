"""헛짚는 질문을 다른 자리에서 다루는 생각 넷을 받아 둔 답으로 본다 (이 Mac, 돈 없음). pipeline/README.md 16절.

(가) 질문 하나를 끄면: mc_12(웹사이트 실행)에서 질문을 하나씩 뺀 클립, 12편의 Gemini Mb에서 다이아몬드 질문 둘을 뺀 AUC
(나) 같은 것을 여러 말로: 일곱 벌의 "싸움" 질문 여섯, "피해·위험" 질문 넷을 하나씩 / 둘·셋 평균 / 모두 평균으로. 정답은 싸움·피해 20곳
(다) 정답 없이 질문을 의심할 신호: 질문마다 0.5 이상인 초의 비율, 막연한 질문과의 순위 상관, 같은 벌의 다른 질문 평균과의 순위 상관,
     바로 다음 초와의 상관. 이 신호와 그 질문의 큰 사건 AUC 사이의 순위 상관
(라) 막연한 질문과 맞는 만큼 무게 주기: 14·15절과 같은 자료와 구간으로 값 평균과 견준다(mf_combine2_check.py의 함수에 합치는 함수만 바꿔 끼운다)

쓰는 것: pipeline/eval/views_check.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_views_check.py
"""
import sys

sys.dont_write_bytecode = True
import itertools  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf  # noqa: E402
import mf_combine2_check as K  # noqa: E402
import mf_core as C  # noqa: E402

CC, G, HE, HP = K.CC, K.G, K.HE, K.HP
r3 = lambda x: None if x is None else round(float(x), 3)
FIGHT = [("hand", "action"), ("na", "mob_fight"), ("G", "fight_with_hostile_creature"), ("GP", "combat_intensity"), ("P12", "fight_intensity"), ("Ma", "combat_action")]
HURT = [("na", "player_hurt"), ("GP", "player_harm"), ("Ma", "crisis_or_damage"), ("G", "dangerous_event")]


def load():
    """마인크래프트 12편: 위치 목록(정답 포함), 영상별 묶음, 벌마다 {위치 id: {질문: 답}}, Gemini 질문의 단계 수."""
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted((r for r in HP.read(CC.HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    sets = {"hand": {r["id"]: {"action": r["action"], "progress": r["progress"]} for r in recs}}

    def add(name, path, prefix=None):
        sets[name] = {r["id"]: {k[len(prefix):] if prefix else k: v for k, v in r["answers"].items() if (k.startswith(prefix) if prefix else True)}
                      for r in CC.read(path)["records"] if "error" not in r}
    add("na", CC.HA / "results" / "minecraft.json")
    add("G", CC.HP_ / "results" / "main_minecraft__mc_game.json", "g__")
    add("GP", CC.HP_ / "results" / "main_minecraft__mc_game.json", "gp__")
    add("P12", CC.HP_ / "results" / "main_minecraft__mc_plain.json")
    byv = {}
    for r in recs:
        byv.setdefault(r["video"], []).append(r)
    levels = {}
    sets["Ma"], sets["Mb"] = {}, {}
    for v, rs in byv.items():
        pos, qj, answers, lv, _ = G.load_run("gem_" + "_".join(v.split("_")[:2]))
        by_t = {q["center"]: a for q, a in zip(pos, answers)}
        for r in rs:
            a = by_t.get(r["t"])
            if a:
                sets["Ma"][r["id"]] = {k[3:]: x for k, x in a.items() if k.startswith("ma_")}
                sets["Mb"][r["id"]] = {k[3:]: x for k, x in a.items() if k.startswith("mb_")}
        levels.update({k[3:]: n for k, n in lv.items()})
    for rs in byv.values():                       # 막연한 질문의 값
        for r in rs:
            r["w"] = r["worth"] if not isinstance(r["worth"], dict) else r["worth"]["noul"]
    return recs, byv, sets, levels


def series(byv, sets, levels, name, key):
    """{영상: 그 질문의 위치별 값(답이 없는 곳은 평균)}. 그 영상에 질문이 없으면 뺀다."""
    out = {}
    for v, rs in byv.items():
        x = np.array([K.unit((sets[name].get(r["id"]) or {}).get(key), levels, key) for r in rs])
        if not np.all(np.isnan(x)):
            out[v] = np.clip(np.where(np.isnan(x), np.nanmean(x), x), 0.0, 1.0)
    return out


def pooled_auc(byv, per_video, label):
    """영상 안 순위로 바꿔 모은 AUC. per_video: {영상: 위치별 점수}."""
    s, y = [], []
    for v, x in per_video.items():
        s += list(C.pct_rank(x))
        y += [bool(r[label]) for r in byv[v]]
    return HE.auc(s, y)


# =====================================================================  (가)
def view_toggle(byv, sets, levels):
    run = mf.RUNS / "web_1116_043326_3"
    pos, qj, info = mf.read(run / "positions.json"), mf.read(run / "questions.json"), mf.read(run / "run.json")
    recs = {r["k"]: r.get("answers") for r in mf.read(run / "results.json")["records"] if "error" not in r}
    answers = [recs.get(q["k"]) for q in pos]
    lv = mf.levels_of(qj["questions"])
    keys = [k for k in qj["questions"] if k != "worth"]
    t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
    fight = [79.5, 82.5, 84.5, 86.5, 87.5, 91.5, 94.5, 97.5, 139.5, 141.5, 142.5]
    prog = [45.5, 52.5, 69.5, 71.5, 253.5]
    one = {}
    for off in [None] + keys:
        use = [k for k in keys if k != off]
        final = C.combine(answers, use, "worth", levels=lv)["final"]
        clips = C.make_clips(final, t0, t1, info["duration"])
        inside = lambda t: any(c["start"] <= t < c["end"] for c in clips)
        one[off or "(모두 씀)"] = {"clips": [[c["start"], c["end"]] for c in clips], "fight_in_clips": sum(map(inside, fight)), "progress_in_clips": sum(map(inside, prog))}
    mb = {}
    for label, drop in (("넷 모두", ()), ("다이아몬드 질문 둘을 뺌", ("diamond_ore_visible", "mining_diamond_ore"))):
        for vague in (False, True):
            per = {}
            for v, rs in byv.items():
                ks = [k for k in ("diamond_ore_visible", "mining_diamond_ore", "lava_hazard_close", "hostile_mob_visible") if k not in drop]
                x = np.mean([series({v: rs}, sets, levels, "Mb", k)[v] for k in ks], axis=0)
                per[v] = 0.5 * C.pct_rank(np.array([r["w"] for r in rs])) + 0.5 * C.pct_rank(x) if vague else x
            mb[label + (" + 막연한 질문" if vague else "")] = {lab: r3(pooled_auc(byv, per, lab)) for lab in ("big", "fight", "prog")}
    return {"mc12_one_question_off": one, "Mb_12_videos": mb}


# =====================================================================  (나)
def view_paraphrase(byv, sets, levels):
    out = {}
    for fam, members in (("싸움", FIGHT), ("피해·위험", HURT)):
        ser = {f"{n}.{k}": series(byv, sets, levels, n, k) for n, k in members}
        vids = [v for v in byv if all(v in s for s in ser.values())]
        single = {q: r3(pooled_auc(byv, {v: s[v] for v in vids}, "fight")) for q, s in ser.items()}
        res = {"videos": len(vids), "single": single, "single_mean": r3(np.mean(list(single.values()))), "single_min": min(single.values()), "single_max": max(single.values())}
        for k in range(2, len(ser) + 1):
            aucs = [pooled_auc(byv, {v: np.mean([ser[q][v] for q in combo], axis=0) for v in vids}, "fight") for combo in itertools.combinations(ser, k)]
            res[f"mean_of_{k}"] = {"combos": len(aucs), "auc_mean": r3(np.mean(aucs)), "auc_min": r3(min(aucs)), "auc_max": r3(max(aucs))}
        out[fam] = res
    return out


# =====================================================================  (다)
def view_signals(byv, sets, levels):
    rows = []
    for name, by in sets.items():
        keys = sorted({q for a in by.values() for q in a if q != "worth"})
        ser = {k: series(byv, sets, levels, name, k) for k in keys}
        for k in keys:
            vids = list(ser[k])
            if len(vids) < 6:
                continue
            yes, agree_v, agree_o, smooth = [], [], [], []
            for v in vids:
                x = ser[k][v]
                yes.append(float((x >= 0.5).mean()))
                w = np.array([r["w"] for r in byv[v]])
                agree_v.append(spearmanr(x, w)[0] if x.std() > 0 else 0.0)
                others = [ser[o][v] for o in keys if o != k and v in ser[o]]
                if others and x.std() > 0 and np.mean(others, axis=0).std() > 0:
                    agree_o.append(spearmanr(x, np.mean(others, axis=0))[0])
                smooth.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 else 0.0)
            rows.append({"set": name, "question": k, "videos": len(vids), "auc_big": r3(pooled_auc(byv, {v: ser[k][v] for v in vids}, "big")),
                         "yes_rate": r3(np.mean(yes)), "agree_vague": r3(np.nanmean(agree_v)), "agree_others": r3(np.nanmean(agree_o)) if agree_o else None,
                         "next_second": r3(np.nanmean(smooth))})
    auc = np.array([r["auc_big"] for r in rows])
    rho = {}
    for sig in ("yes_rate", "agree_vague", "agree_others", "next_second"):
        ok = [i for i, r in enumerate(rows) if r[sig] is not None]
        c, p = spearmanr([rows[i][sig] for i in ok], auc[ok])
        rho[sig] = {"spearman_with_auc": r3(c), "p": r3(p), "questions": len(ok)}
    return {"questions": rows, "signal_vs_usefulness": rho}


# =====================================================================  (라)
def scores_weighted(answers, keys, levels=None):
    val = [np.array([K.unit((a or {}).get(k), levels, k) for a in answers]) for k in keys]
    v = np.clip(np.array([np.where(np.isnan(x), np.nanmean(x) if not np.all(np.isnan(x)) else 0.0, x) for x in val]), 0.0, 1.0)
    vr = np.nan_to_num(K.CTX["vr"], nan=0.5)
    w = np.array([max(0.0, float(np.corrcoef(C.pct_rank(x), vr)[0, 1])) if x.std() > 0 else 0.0 for x in v])
    return {"value_mean": v.mean(0), "w_agree": (w[:, None] * v).sum(0) / w.sum() if w.sum() > 0 else v.mean(0)}


def view_weighted():
    K.RULES, K.CAND, K.LATER, K.scores = ("value_mean", "w_agree"), ("w_agree",), (), scores_weighted
    return {"minecraft": K.minecraft(), "tvsum": K.tvsum(), "robot": K.robot()}


def main():
    recs, byv, sets, levels = load()
    out = {"label": "헛짚는 질문을 다른 자리에서 다루는 생각 넷 (받아 둔 답. pipeline/README.md 16절)",
           "toggle": view_toggle(byv, sets, levels), "paraphrase": view_paraphrase(byv, sets, levels), "signals": view_signals(byv, sets, levels)}
    print("(가) mc_12에서 질문 하나를 끄면 (싸움·피해 11 / 제작·다이아 5 가운데 클립에 든 수)")
    for k, v in out["toggle"]["mc12_one_question_off"].items():
        print(f"  끈 질문 {k:26s} 싸움 {v['fight_in_clips']:2d} 제작 {v['progress_in_clips']} | {v['clips']}")
    for k, v in out["toggle"]["Mb_12_videos"].items():
        print(f"  Mb 12편 {k:32s} {v}")
    print("(나) 같은 것을 여러 말로 (싸움·피해 20곳의 AUC)")
    for fam, r in out["paraphrase"].items():
        print(f"  {fam}: 하나씩 {r['single']} | 평균 {r['single_mean']} (가장 낮은 {r['single_min']}, 가장 높은 {r['single_max']})")
        for k, v in r.items():
            if k.startswith("mean_of_"):
                print(f"     {k}: {v}")
    print("(다) 정답 없는 신호와 질문의 쓸모(큰 사건 AUC)")
    for r in sorted(out["signals"]["questions"], key=lambda r: -r["auc_big"]):
        print(f"  {r['set']:4s} {r['question']:28s} AUC {r['auc_big']:.3f} | 예 비율 {r['yes_rate']:.3f} | 막연한 질문과 {r['agree_vague']:+.3f} | 다른 질문과 {r['agree_others'] if r['agree_others'] is None else format(r['agree_others'], '+.3f')} | 다음 초와 {r['next_second']:.3f}")
    print("  ", out["signals"]["signal_vs_usefulness"])
    out["weighted"] = view_weighted()
    with open(HERE / "eval" / "views_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    m = out["weighted"]["minecraft"]
    print("(라) 막연한 질문과 맞는 만큼 무게 주기: 마인크래프트 AUC / 클립에 든 큰 사건")
    for name in m["sets"]:
        for form in K.FORMS:
            print(f"  {name:4s}{form:7s} " + " | ".join(f"{rule} {m['arms'][f'{name}:{rule}{form}']['auc_big']:.3f}/{m['arms'][f'{name}:{rule}{form}']['big_in_clips']}" for rule in K.RULES))
    for k, d in m["differences"].items():
        print(f"      {k}: {d['point']:+.4f} {d['ci']}")
    for name, res in out["weighted"]["tvsum"].items():
        for k, d in res["differences"].items():
            if k.endswith("_tau"):
                print(f"  TVSum {name:3s} {k}: {d['point']:+.4f} {d['ci']}")
    for v, res in out["weighted"]["robot"].items():
        for sname, arms in res.items():
            print(f"  {v:12s} {sname:11s} " + " | ".join(f"{rule}{form} {arms[rule + form]:.3f}" for form in K.FORMS for rule in K.RULES))


if __name__ == "__main__":
    main()
