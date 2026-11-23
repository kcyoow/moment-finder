"""LeVJEPA 프로브 실험의 결과를 Jev·SigLIP과 같은 기준으로 견준다. 표준 라이브러리만 쓴다.

자료
- results/lev_probe_result.json: Colab에서 lev_probe_colab.py가 만든 것 (변형별 평가 수치, 우리 12개 영상의 위치별 점수)
- results/sta_preds.json: Jev 확률 (이미 있는 것)
- results/siglip_scores_*.json: 학습 없는 SigLIP 2 점수 (작업 10)

보는 것
1. 공식 평가 쪽 영상(학습에 쓰지 않은 것)에서 변형별 항목 AUC, 변형 사이의 차이와 95% 구간(영상 단위로 다시 뽑음)
2. 우리 12개 영상에서 Jev 한 장·4장, 학습 없는 SigLIP, 학습한 머리들을 같은 표로

실행: python3 experiments/frame_limit/scripts/embed/lev_probe_eval.py
결과: 표 출력, results/lev_probe_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import random
from pathlib import Path

import siglip_eval as E
from qv_eval import auc

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
PAIRS = [("LeVJEPA cls+패치", "SigLIP 이미지 (4장 평균)"),
         ("LeVJEPA cls+패치 + SigLIP 4장", "SigLIP 이미지 (4장 평균)"),
         ("LeVJEPA cls+패치", "학습 없는 SigLIP (4장 평균)"),
         ("SigLIP 이미지 (4장 평균)", "학습 없는 SigLIP (4장 평균)"),
         ("LeVJEPA cls+패치", "글 없이 LeVJEPA cls+패치"),
         ("SigLIP 이미지 (4장 평균)", "글 없이 SigLIP 4장")]


def boot(values_a, values_b, groups, rounds=5000, seed=20261111):
    """항목별 값의 차이(a − b) 평균과, 영상 단위로 다시 뽑은 95% 구간."""
    rng = random.Random(seed)
    rows = [(g, a - b) for g, a, b in zip(groups, values_a, values_b) if a is not None and b is not None]
    ids = sorted({g for g, _ in rows})
    by = {g: [d for gg, d in rows if gg == g] for g in ids}
    mean = sum(d for _, d in rows) / len(rows)
    out = []
    for _ in range(rounds):
        ds = [d for g in (rng.choice(ids) for _ in ids) for d in by[g]]
        out.append(sum(ds) / len(ds))
    out.sort()
    return round(mean, 3), round(out[int(0.025 * rounds)], 3), round(out[int(0.975 * rounds)], 3), len(rows), len(ids)


def main():
    res = json.loads((HERE / "results" / "lev_probe_result.json").read_text())
    info, timing = res["info"], res["timing"]
    print(f"GPU {info['gpu']}, 영상 {info['videos']}개, 클립 {info['clips']}개, LeVJEPA 초당 {info['lev_clips_per_second']}클립 "
          f"(전체 {info['seconds_total']:.0f}초), 학습 영상 {res['train_videos']}개, 문장 {res['counts']}")
    out = {"info": info, "timing": timing, "test": {}, "ours": {}}

    print("\n===== 1. 공식 평가 쪽 영상 (학습에 쓰지 않음) =====")
    print(f"{'변형':34s}{'항목':>6s}{'AUC':>8s}{'5칸 평균 뒤':>10s}{'최고점 적중':>10s}{'씨앗별 AUC':>24s}")
    for name, v in res["variants"].items():
        t = v["test"]
        seeds = " ".join(f"{s['test']:.3f}" for s in v.get("per_seed", []))
        out["test"][name] = {k: t[k] for k in ("items", "auc_raw", "auc_smooth", "peak_hit")}
        print(f"{name:34s}{t['items']:>6d}{t['auc_raw']:>8.3f}{t['auc_smooth']:>10.3f}{t['peak_hit']:>7d}/{t['items']}{seeds:>24s}")
    groups = res["test_item_video"]
    print("\n차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
    out["test_diff"] = {}
    for a, b in PAIRS:
        m, lo, hi, n, g = boot(res["variants"][a]["test"]["per_item"], res["variants"][b]["test"]["per_item"], groups)
        out["test_diff"][f"{a} − {b}"] = [m, lo, hi]
        print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (항목 {n}, 영상 {g})")

    print("\n===== 2. 우리 12개 영상 (요청문 32개) =====")
    items = E.bundle("sta")
    by_id = {it["id"]: it for it in items}
    rows = {"Jev 한 장": [(it["one"], it["y"]) for it in items], "Jev 4장": [(it["win4"], it["y"]) for it in items]}
    for label, fname in E.MODEL_FILES.items():
        path = HERE / "results" / fname
        if path.exists():
            sc = E.Scores.from_file(path)
            for cond, tag in (("one", "한 장"), ("win4", "4장 평균")):
                rows[f"{label} 학습 없이, 원본 프레임 {tag}"] = [(E.series(sc, it, cond), it["y"]) for it in items]
    for name, scores in res["ours_scores"].items():
        assert all(len(scores[it["id"]]) == len(it["y"]) for it in items), f"{name}: 위치 수가 다르다"
        rows[name] = [(scores[it["id"]], it["y"]) for it in items]
    print(f"{'방식':44s}{'AUC':>7s}{'5칸 평균 뒤':>10s}{'최고점':>8s}{'F1':>7s}{'F1 상한':>8s}")
    per_item = {}
    for name, pairs in rows.items():
        r = E.evaluate(pairs)
        per_item[name] = [auc(ps, y) for ps, y in pairs]
        if name in res["ours_scores"]:
            # 저장된 점수는 소수 5자리로 줄인 확률이라 0이나 1에 붙은 값끼리 동점이 생긴다.
            # 순서로 내는 값(AUC, 최고점)은 Colab에서 줄이기 전 점수로 계산한 것을 쓴다. F1만 저장된 확률로 낸다.
            exact = res["variants"][name]["ours"]
            r.update(auc_raw=exact["auc_raw"], auc_smooth=exact["auc_smooth"], peak_hit=exact["peak_hit"])
            per_item[name] = exact["per_item"]
        out["ours"][name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"{name:44s}{r['auc_raw']:>7.3f}{r['auc_smooth']:>10.3f}{r['peak_hit']:>5d}/{r['items']}{r['f1_same']:>7.3f}{r['f1_best']:>8.3f}")
    videos = [it["video"] for it in items]
    print("\n차이와 95% 구간 (항목 AUC, 영상 12개를 다시 뽑음)")
    out["ours_diff"] = {}
    for a, b in [("LeVJEPA cls+패치", "Jev 4장"), ("LeVJEPA cls+패치", "Jev 한 장"), ("SigLIP 이미지 (4장 평균)", "Jev 4장"),
                 ("LeVJEPA cls+패치 + SigLIP 4장", "Jev 4장"), ("LeVJEPA cls+패치", "SigLIP 이미지 (4장 평균)")]:
        m, lo, hi, n, g = boot(per_item[a], per_item[b], videos)
        out["ours_diff"][f"{a} − {b}"] = [m, lo, hi]
        print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # 줄인 확률로 다시 낸 AUC가 Colab의 값과 얼마나 다른지 적어 둔다 (동점 때문에 생기는 차이)
    gaps = {name: abs(res["variants"][name]["ours"]["auc_raw"] -
                      sum(a for a in (auc(ps, y) for ps, y in rows[name]) if a is not None) / len(rows[name]))
            for name in res["ours_scores"]}
    out["rounding_gap_max"] = round(max(gaps.values()), 4)
    print(f"\n참고: 소수 5자리로 줄인 확률로 AUC를 다시 내면 최대 {max(gaps.values()):.3f} 달라진다. 위 표의 AUC·최고점은 줄이기 전 값이다.")
    (HERE / "results" / "lev_probe_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
