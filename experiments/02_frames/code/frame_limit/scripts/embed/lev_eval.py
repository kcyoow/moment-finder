"""LeVJEPA 임베딩이 Jev 판정을 얼마나 돕는지 계산한다. 새 Jev 요청은 보내지 않는다.

쓰는 것: results/lev_emb.npz (lev_embed.py), 이미 받은 Jev 확률(한 장, 4장), 요청 종류(needs).

1. 퍼뜨리기: 한 영상 안에서 LeVJEPA 임베딩이 비슷한 위치끼리 Jev 한 장 확률을 나눠 갖게 한다.
   새 확률 = Σ_j w_ij × 확률_j,  w_ij = softmax(코사인 유사도 / TAU)
   비교 대상: 그대로(한 장), 앞뒤 위치와 평균(시간 평균), 16×16 픽셀 임베딩으로 같은 계산, Jev 4장
2. 4장 위치 고르기: 움직임 신호가 큰 위치 q%만 Jev 4장 확률을 쓰고 나머지는 한 장 확률을 쓴다.
   신호: LeVJEPA 프레임 간 변화, 픽셀 차이(싼 기준선), 무작위(200번 평균)
3. 합치기: 퍼뜨린 확률과 4장 확률을 평균

TAU는 앞서 본 영상(loc)에서 고르고 새 영상(sta)에 적용한다.

실행: <venv>/bin/python experiments/frame_limit/scripts/embed/lev_eval.py    (numpy 필요)
결과: 표 출력, results/lev_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import random
from pathlib import Path

import numpy as np

from qv_eval import auc, prf
from recipe_eval import NAME, R, catch, fill_gaps, load

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
TAUS = [0.01, 0.02, 0.05, 0.1]
TAUS_CENTERED = [0.05, 0.1, 0.2, 0.5]
ONE_TOKENS, WIN_TOKENS = 1295, 4378


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def spread(emb, scores, tau, center=False):
    """임베딩이 비슷한 위치끼리 확률을 나눈다. 자기 자신도 포함한다.

    center=True면 그 영상의 평균 임베딩을 먼저 뺀다. 임베딩들이 서로 너무 비슷하게 나올 때
    (코사인 유사도가 0.9 이상에 몰릴 때) 차이를 키우는 흔한 처리다.
    """
    if center:
        emb = emb - emb.mean(0, keepdims=True)
    sim = unit(emb) @ unit(emb).T
    w = np.exp((sim - sim.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return list(w @ np.array(scores))


def time_average(scores):
    s = np.array(scores, dtype=float)
    padded = np.concatenate([s[:1], s, s[-1:]])
    return list(0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:])


def summarize(items, series_of):
    """series_of(item) → 확률 목록. 평균 AUC(전체, motion, single)와 임계값 규칙을 적용한 F1."""
    aucs = {"all": [], "motion": [], "single": []}
    tp = fp = fn = 0
    for it in items:
        ps = series_of(it)
        a = auc(ps, it["y"])
        if a is not None:
            aucs["all"].append(a)
            aucs[it["needs"]].append(a)
        caught = fill_gaps(catch(ps, R))
        tp += sum(c and y for c, y in zip(caught, it["y"]))
        fp += sum(c and not y for c, y in zip(caught, it["y"]))
        fn += sum(not c and y for c, y in zip(caught, it["y"]))
    prec, rec, f1 = prf(tp, fp, fn)
    return {"auc": float(np.mean(aucs["all"])), "auc_motion": float(np.mean(aucs["motion"])),
            "auc_single": float(np.mean(aucs["single"])), "precision": prec, "recall": rec, "f1": f1}


def line(name, s, cost=None):
    print(f"  {name:34s} AUC {s['auc']:.3f} (motion {s['auc_motion']:.3f}, single {s['auc_single']:.3f})"
          f"  F1 {s['f1']:.3f} (정밀도 {s['precision']:.2f} 재현율 {s['recall']:.2f})"
          + (f"  Jev 토큰 {cost:.2f}배" if cost is not None else ""))


def main():
    z = np.load(HERE / "results" / "lev_emb.npz")
    where = {}
    for i, (vid, t) in enumerate(zip(z["vid"], z["t"])):
        where[(str(vid), float(t))] = i
    sets = {}
    for tag in ("loc", "sta"):
        items, _ = load(tag)
        meta = json.loads((HERE / "data" / f"{tag}_items.json").read_text())
        video_of = {it["id"]: it["video"] for it in meta["items"]}
        for it in items:
            rows = [where[(video_of[it["id"]], float(t))] for t in it["ts"]]
            it["cls"], it["patch"], it["pix16"] = z["cls"][rows], z["patch_mean"][rows], z["pix16"][rows]
            it["motion_lev"], it["motion_pix"] = z["motion_lev"][rows], z["motion_pix"][rows]
        sets[tag] = items

    # TAU 고르기: 앞서 본 영상에서 퍼뜨린 뒤의 평균 AUC
    print("TAU 고르기 (앞서 본 Charades 5개, 퍼뜨린 뒤 평균 AUC)")
    pick = {}
    for emb in ("cls", "patch"):
        scores = {tau: summarize(sets["loc"], lambda it: spread(it[emb], it["one"], tau))["auc"] for tau in TAUS}
        pick[emb] = max(TAUS, key=lambda tau: scores[tau])
        print(f"  {emb:6s} " + ", ".join(f"TAU {tau}: {scores[tau]:.3f}" for tau in TAUS) + f"  → {pick[emb]}")
    scores = {tau: summarize(sets["loc"], lambda it: spread(it["pix16"], it["one"], tau))["auc"] for tau in TAUS}
    pick["pix16"] = max(TAUS, key=lambda tau: scores[tau])
    print(f"  pix16  " + ", ".join(f"TAU {tau}: {scores[tau]:.3f}" for tau in TAUS) + f"  → {pick['pix16']}")
    for emb in ("cls", "pix16"):  # 영상별 평균을 뺀 경우. 유사도 범위가 넓어지므로 TAU 후보도 넓게 잡는다
        scores = {tau: summarize(sets["loc"], lambda it: spread(it[emb], it["one"], tau, center=True))["auc"]
                  for tau in TAUS_CENTERED}
        pick[emb + "_c"] = max(TAUS_CENTERED, key=lambda tau: scores[tau])
        print(f"  {emb} (평균 뺌) " + ", ".join(f"TAU {tau}: {scores[tau]:.3f}" for tau in TAUS_CENTERED)
              + f"  → {pick[emb + '_c']}")

    result = {"tau": pick, "sets": {}}
    rng = random.Random(0)
    for tag in ("sta", "loc"):
        items = sets[tag]
        print(f"\n{NAME[tag]} (항목 {len(items)}개, motion {sum(it['needs'] == 'motion' for it in items)}개)")
        block = {}

        def add(name, fn, cost=None):
            block[name] = {**summarize(items, fn), "jev_tokens": cost}
            line(name, block[name], cost)

        print(" 기준")
        add("Jev 한 장", lambda it: it["one"], 1.0)
        add("Jev 4장", lambda it: it["win4"], WIN_TOKENS / ONE_TOKENS)
        print(" 1. Jev 한 장 확률을 퍼뜨리기 (Jev 추가 비용 없음)")
        add("시간 평균 (앞뒤 한 칸)", lambda it: time_average(it["one"]), 1.0)
        add("픽셀 임베딩으로 퍼뜨리기", lambda it: spread(it["pix16"], it["one"], pick["pix16"]), 1.0)
        add("LeVJEPA cls로 퍼뜨리기", lambda it: spread(it["cls"], it["one"], pick["cls"]), 1.0)
        add("LeVJEPA 패치 평균으로 퍼뜨리기", lambda it: spread(it["patch"], it["one"], pick["patch"]), 1.0)
        add("픽셀 임베딩(평균 뺌)으로 퍼뜨리기",
            lambda it: spread(it["pix16"], it["one"], pick["pix16_c"], center=True), 1.0)
        add("LeVJEPA cls(평균 뺌)로 퍼뜨리기",
            lambda it: spread(it["cls"], it["one"], pick["cls_c"], center=True), 1.0)
        print(" 2. 4장을 쓸 위치만 고르기 (고른 위치는 4장 확률, 나머지는 한 장 확률)")
        for q in (0.25, 0.5):
            cost = (1 - q) + q * WIN_TOKENS / ONE_TOKENS

            def chosen(it, signal):
                k = max(1, round(q * len(signal)))
                top = set(np.argsort(-np.asarray(signal))[:k].tolist())
                return [w if i in top else o for i, (o, w) in enumerate(zip(it["one"], it["win4"]))]

            add(f"LeVJEPA 변화가 큰 {q:.0%}", lambda it: chosen(it, it["motion_lev"]), cost)
            add(f"픽셀 차이가 큰 {q:.0%}", lambda it: chosen(it, it["motion_pix"]), cost)
            draws = [summarize(items, lambda it: chosen(it, [rng.random() for _ in it["one"]])) for _ in range(200)]
            block[f"무작위 {q:.0%}"] = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]} | {"jev_tokens": cost}
            line(f"무작위 {q:.0%} (200번 평균)", block[f"무작위 {q:.0%}"], cost)
        print(" 3. 4장 확률에 더하기")
        add("Jev 4장을 시간 평균", lambda it: time_average(it["win4"]), WIN_TOKENS / ONE_TOKENS)
        add("Jev 4장을 LeVJEPA cls로 퍼뜨리기", lambda it: spread(it["cls"], it["win4"], pick["cls"]),
            WIN_TOKENS / ONE_TOKENS)
        result["sets"][tag] = block

    # 움직임 신호 자체가 정답 구간을 가리키는가 (요청 내용과 무관한 신호)
    print("\n움직임 신호만으로 정답 구간을 가리는 정도 (새 Charades, 평균 AUC. 0.5면 무관)")
    for key, name in (("motion_lev", "LeVJEPA 프레임 간 변화"), ("motion_pix", "픽셀 차이")):
        vals = [auc(list(it[key]), it["y"]) for it in sets["sta"]]
        vals = [v for v in vals if v is not None]
        result.setdefault("signal_auc", {})[key] = float(np.mean(vals))
        print(f"  {name}: {np.mean(vals):.3f}")
    # 곁가지로 본 것: 시간 평균(앞뒤 한 칸, 가중치 0.25/0.5/0.25)은 다른 묶음에서도 도움이 되는가
    print("\n시간 평균의 효과 (평균 AUC, 그대로 → 시간 평균)")
    result["time_average"] = {}
    for tag in ("loc", "sta", "qv", "qv2"):
        items = sets[tag] if tag in sets else load(tag)[0]
        row = {}
        for frames in ("one", "win4"):
            sel = [it for it in items if it[frames]]
            if not sel:
                continue
            raw = [auc(it[frames], it["y"]) for it in sel]
            avg = [auc(time_average(it[frames]), it["y"]) for it in sel]
            pairs = [(a, b) for a, b in zip(raw, avg) if a is not None]
            row[frames] = {"n": len(pairs), "raw": float(np.mean([p[0] for p in pairs])),
                           "averaged": float(np.mean([p[1] for p in pairs])),
                           "improved": int(sum(p[1] > p[0] for p in pairs)),
                           "worse": int(sum(p[1] < p[0] for p in pairs))}
        result["time_average"][tag] = row
        print(f"  {NAME[tag]}: " + ", ".join(
            f"{'한 장' if k == 'one' else '4장'} {v['raw']:.3f} → {v['averaged']:.3f} (오른 항목 {v['improved']}, 내린 항목 {v['worse']}, 전체 {v['n']})"
            for k, v in row.items()))
    timing = json.loads((HERE / "results" / "lev_timing.json").read_text())
    result["timing"] = timing
    print("\n속도:", json.dumps(timing, ensure_ascii=False))
    (HERE / "results" / "lev_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
