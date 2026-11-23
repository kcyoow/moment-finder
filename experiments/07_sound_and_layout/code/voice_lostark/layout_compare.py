"""프레임 넣는 모양 A·B·C·D 비교(25-5절). 돈은 들지 않는다. 있는 답만 읽는다.

  1. 종류마다 "예"(>0.5) 초 수, A와의 상관(초마다 확률)
  2. 모아 둔 다시 확인 (후보 순위는 types_combine.py와 같은 섞기로 전체를 다시 만들어 본다): 네 모양이 낸 후보를 Gemini가 본 결과(types_verify_r2·_LB·_LC·_LD)를 한데 모아
     p≥0.5 구간을 "Gemini가 맞다고 본 구간", 나머지를 "아니라고 본 구간"으로 둔다(사람 판단 아님).
     모양마다 그 구간 안의 최대 확률로 둘을 가르는 AUC, 맞다고 본 구간이 그 모양의 후보 3개에 든 수
  3. D의 끌림: D−A가 (직전 3초 A 평균 − A) 쪽으로 움직이는가. B·C를 대조로 같은 값을 잰다
  4. E(사슬)의 끌림: E−A가 (E가 받은 바로 앞 답 − A) 쪽으로 움직이는가, "예"가 이어지는 길이(사슬이 한 번 "예"면 계속 "예"로 붙는가)
  주의: B·C·D 요청의 GAME 문장은 A보다 짧았다(카멘·한국어 대화가 빠짐). E는 A 묶음을 그대로 복사해 앞 답만 다르다
실행: uv run --no-project --with numpy python -I experiments/voice_lostark/layout_compare.py
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT.parents[1] / "pipeline"))
import mf_core as C  # noqa: E402
from mf import positions_for  # noqa: E402

P = json.load(open(ROOT / "types/proposal_r2.json")); keys = [t["key"] for t in P["types"]]
names = {t["key"]: t["name_ko"] for t in P["types"]}
n = json.load(open(RES / "out/per_second.json"))["seconds"]
pos_ = positions_for(n); T0 = np.array([p["t0"] for p in pos_]); T1 = np.array([p["t1"] for p in pos_])
W8 = {"not_needed": (1.0, 0.0), "helpful": (0.6, 0.4), "essential": (0.3, 0.7)}
_g = {r["k"]: r["answers"] for r in json.load(open(RES / "h2o_out/types_gate.json"))["records"] if r and "answers" in r}
_v = {r["k"]: r["answers"] for r in json.load(open(RES / "h2o_out/types_voice.json"))["records"] if r and "answers" in r}
ROLE = {}
for i, t in enumerate(P["types"]):
    pr = _g[i]["voice_role"]["probabilities"]; ROLE[t["key"]] = W8[max(pr, key=pr.get)] if t.get("voice_question") else (1.0, 0.0)
VOICE = {q: np.array([C.answer_value(_v[s][q]) if (s in _v and q in _v[s]) else 0.0 for s in range(n)]) for q in keys}
ARMS = {"A": ("types_frames_r2", "_r2"), "B": ("layout_B", "_LB"), "C": ("layout_C", "_LC"), "D": ("layout_D", "_LD"), "E": ("layout_E", "_LE")}


def probs(spec):
    f = RES / f"h2o_out/{spec}.json"
    if not f.exists():
        return None, None
    j = json.load(open(f)); a = {r["k"]: r["answers"] for r in j["records"] if r and "answers" in r}
    return {q: np.array([C.answer_value(a[s][q]) if s in a else np.nan for s in range(n)]) for q in keys}, j["meta"]


S, meta = {}, {}
for arm, (spec, _) in ARMS.items():
    s, m = probs(spec)
    if s is not None:
        S[arm], meta[arm] = s, m
arms = list(S)
out = {"arms": arms, "meta": {a: {k: meta[a][k] for k in ("requests", "errors", "wall_seconds", "tokens_median")} for a in arms}, "types": {}}

# 모아 둔 다시 확인
pool = {q: [] for q in keys}
for arm, (_, suf) in ARMS.items():
    for q in keys:
        d = RES / f"types_verify{suf}" / q
        if (d / "verdict.json").exists():
            it = json.load(open(d / "item.json")); v = json.load(open(d / "verdict.json"))
            for c in v["candidates"]:
                w = it["candidates"][c["id"] - 1]
                pool[q].append({"from": arm, "start": w["start"], "end": w["end"], "p": c["p"]})


def dedupe(ws):          # 같은 구간을 두 모양이 냈으면 하나로(겹침이 짧은 쪽의 절반 넘으면 같은 것, p는 평균)
    res = []
    for w in sorted(ws, key=lambda w: w["start"]):
        for r in res:
            ov = min(r["end"], w["end"]) - max(r["start"], w["start"])
            if ov > 0.5 * min(r["end"] - r["start"], w["end"] - w["start"]):
                r["ps"].append(w["p"]); r["from"] += w["from"]; r["start"] = min(r["start"], w["start"]); r["end"] = max(r["end"], w["end"]); break
        else:
            res.append({**w, "ps": [w["p"]]})
    for r in res:
        r["p"] = float(np.mean(r["ps"])); r["agree"] = (min(r["ps"]) >= 0.5) == (max(r["ps"]) >= 0.5)
    return res


def auc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))


all_pos, all_neg = {a: [] for a in arms}, {a: [] for a in arms}
for q in keys:
    W = dedupe(pool[q])
    row = {"name_ko": names[q], "windows": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items() if k != "ps"} for w in W], "arms": {}}
    for a in arms:
        sc = S[a][q]; sc0 = np.nan_to_num(sc, nan=0.0)
        summ = json.load(open(RES / f"types_summary{ARMS[a][1]}.json")) if (RES / f"types_summary{ARMS[a][1]}.json").exists() else None
        wf, wv = ROLE[q]   # types_combine.py와 같은 섞기로 후보 전체 순위를 다시 만든다
        cand = C.make_clips(wf * sc0 + wv * VOICE[q], T0, T1, n, budget=0.9)
        ws = [float(sc0[int(w["start"]):int(np.ceil(w["end"]))].max()) for w in W]
        pos = [x for x, w in zip(ws, W) if w["p"] >= 0.5]; neg = [x for x, w in zip(ws, W) if w["p"] < 0.5]
        all_pos[a] += pos; all_neg[a] += neg
        def rank(w):
            r = [i + 1 for i, c in enumerate(cand) if min(c["end"], w["end"]) - max(c["start"], w["start"]) > 0]
            return r[0] if r else None
        ranks = [rank(w) for w in W if w["p"] >= 0.5]
        pk = next((t.get("pick") for t in summ["types"] if t["key"] == q), None) if summ else None
        row["arms"][a] = {"yes_seconds": int((sc > 0.5).sum()), "corr_with_A": round(float(np.corrcoef(sc0, np.nan_to_num(S["A"][q], nan=0.0))[0, 1]), 3) if a != "A" else 1.0,
                          "auc_pool": auc(pos, neg), "pool_pos_ranks": ranks, "in_top3": f"{sum(1 for r in ranks if r and r <= 3)}/{len(ranks)}", "in_top10": f"{sum(1 for r in ranks if r and r <= 10)}/{len(ranks)}", "pick_p": pk["p"] if pk else None}
    out["types"][q] = row

out["pooled_auc_all_types"] = {a: auc(all_pos[a], all_neg[a]) for a in arms}
out["found_types"] = {}
for a in arms:
    f = RES / f"types_summary{ARMS[a][1]}.json"
    if f.exists():
        out["found_types"][a] = [t["key"] for t in json.load(open(f))["types"] if (t.get("pick") or {}).get("p", 0) >= 0.5]

# D의 끌림(직전 값 쪽으로 가는가). B·C는 대조: 직전 값을 보지 않았으니 같은 계산이 0 근처여야 한다
A = S["A"]; out["pull"] = {}
for a in [x for x in arms if x != "A"]:
    dd, mm, flip, toward = [], [], 0, 0
    for q in keys:
        for k in range(3, n):
            prev = A[q][k - 3:k]
            if np.isnan(A[q][k]) or np.isnan(S[a][q][k]) or np.isnan(prev).any():
                continue
            m = prev.mean() - A[q][k]; d = S[a][q][k] - A[q][k]
            dd.append(d); mm.append(m)
            if (A[q][k] >= 0.5) != (S[a][q][k] >= 0.5):
                flip += 1; toward += (np.sign(d) == np.sign(m))
    dd, mm = np.array(dd), np.array(mm)
    out["pull"][a] = {"corr(arm-A, prevmean-A)": round(float(np.corrcoef(dd, mm)[0, 1]), 3), "slope": round(float(np.polyfit(mm, dd, 1)[0]), 3),
                      "flip_rate": round(flip / len(dd), 4), "flips_toward_prev": f"{toward}/{flip}"}

# E(사슬): 보여 준 값 = 같은 실행이 바로 앞 초에 낸 답(prev_shown). 끌림과 "예"가 이어지는 길이
if "E" in S:
    recs = {r["k"]: r for r in json.load(open(RES / "h2o_out/layout_E.json"))["records"] if r and "answers" in r}
    dd, mm, flip, toward, nshown = [], [], 0, 0, 0
    for q in keys:
        for k, r in recs.items():
            sh = r.get("prev_shown")
            if not sh or np.isnan(A[q][k]):
                continue
            nshown += 1
            m = sh[q] - A[q][k]; d = S["E"][q][k] - A[q][k]; dd.append(d); mm.append(m)
            if (A[q][k] >= 0.5) != (S["E"][q][k] >= 0.5):
                flip += 1; toward += (np.sign(d) == np.sign(m))
    dd, mm = np.array(dd), np.array(mm)
    out["pull"]["E(own prev)"] = {"corr(E-A, shown-A)": round(float(np.corrcoef(dd, mm)[0, 1]), 3), "slope": round(float(np.polyfit(mm, dd, 1)[0]), 3),
                                  "flip_rate": round(flip / len(dd), 4), "flips_toward_shown": f"{toward}/{flip}", "items_with_prev": nshown // len(keys)}

    def runs(x):
        b = np.nan_to_num(x, nan=0.0) > 0.5; L, c = [], 0
        for v in b:
            if v:
                c += 1
            elif c:
                L.append(c); c = 0
        if c:
            L.append(c)
        return L
    out["stick"] = {}
    for a in arms:
        lag1 = [float(np.corrcoef(np.nan_to_num(S[a][q][:-1]), np.nan_to_num(S[a][q][1:]))[0, 1]) for q in keys]
        rl = [x for q in keys for x in runs(S[a][q])]
        out["stick"][a] = {"lag1_autocorr_mean": round(float(np.mean(lag1)), 3), "yes_runs": len(rl), "yes_run_mean_len": round(float(np.mean(rl)), 2) if rl else 0, "yes_run_max": max(rl) if rl else 0}

json.dump(out, open(RES / "layout_compare.json", "w"), ensure_ascii=False, indent=1)
print("모양", arms, {a: out["meta"][a] for a in arms})
print("확실히 찾은 종류", out["found_types"])
print("모아 둔 다시 확인 AUC(모든 종류)", {a: (round(v, 3) if v is not None else None) for a, v in out["pooled_auc_all_types"].items()})
for q, r in out["types"].items():
    pos = [f"{int(w['start'])//60}:{int(w['start'])%60:02d}({w['from']},{w['p']:.2f})" for w in r["windows"] if w["p"] >= 0.5]
    print(f"[{q}] {r['name_ko']} 맞다고 본 구간 {pos}")
    for a, x in r["arms"].items():
        print(f"   {a}: 예 {x['yes_seconds']:4d}초  A상관 {x['corr_with_A']:.2f}  AUC {x['auc_pool'] if x['auc_pool'] is None else round(x['auc_pool'], 2)}  맞는 구간 순위 {x['pool_pos_ranks']} (3위 안 {x['in_top3']}, 10위 안 {x['in_top10']})  대표 p {x['pick_p']}")
print("끌림", json.dumps(out["pull"], ensure_ascii=False))
if "stick" in out:
    print("이어짐", json.dumps(out["stick"], ensure_ascii=False))
