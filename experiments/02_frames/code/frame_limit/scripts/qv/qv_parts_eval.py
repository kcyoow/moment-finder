"""요청을 조건별 질문으로 나눠 물은 결과(results/qv_parts_preds.json)를 채점한다. 표준 라이브러리만 쓴다.

비교하는 방식 (모두 한 장 조건, 같은 프레임)
- whole_alone : 요청 전체를 한 질문으로 물음. 앞선 실행(results/qv_preds.json)의 값
- whole_joint : 같은 질문을 조건별 질문과 한 요청에 함께 물음. 다른 질문이 답을 바꾸는지 확인하는 용도
- min         : 조건별 확률의 최솟값. 임계값으로 자르면 "모든 조건이 임계값 이상"과 같다
- prod        : 조건별 확률의 곱
나누지 않은 요청(조건 목록이 빈 것)은 min, prod에 whole_joint 값을 쓴다.

채점 규칙은 qv_eval.py와 같다 (위치의 시각이 정답 구간 안이면 양성, 0.5 이상이면 "잡았다").

실행: python3 experiments/frame_limit/scripts/qv/qv_parts_eval.py
결과: 표 출력, results/qv_parts_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from math import prod
from pathlib import Path

from qv_eval import GRID, THR, auc, confusion, fmt, prf

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
METHODS = ["whole_alone", "whole_joint", "min", "prod"]
LABEL = {"whole_alone": "한 질문(앞선 실행)", "whole_joint": "한 질문(함께 물음)", "min": "조건별 최솟값", "prod": "조건별 곱"}


def main():
    items = json.loads((HERE / "data" / "qv_items.json").read_text())["items"]
    parts = json.loads((HERE / "data" / "qv_parts.json").read_text())["parts"]
    base = json.loads((HERE / "results" / "qv_preds.json").read_text())
    new = json.loads((HERE / "results" / "qv_parts_preds.json").read_text())
    if new["meta"]["saved"] != new["meta"]["jobs"] or new["meta"]["errors"]:
        print(f"주의: 요청 {new['meta']['jobs']}건 중 저장 {new['meta']['saved']}건, 오류 {new['meta']['errors']}건")
    alone = {(r["item"], r["t"]): r["noul"] for r in base["records"]
             if r["frames"] == "one" and r["prompt"] == "plain" and not r["error"]}
    joint = {(r["item"], r["t"]): r["nouls"] for r in new["records"] if not r["error"]}
    result = {"meta": new["meta"], "threshold": THR, "items": {}, "summary": {}}
    rows_by_method = {m: [] for m in METHODS}
    for it in items:
        wins, conds = it["windows"], parts[it["id"]]
        ts = sorted(t for (q, t) in joint if q == it["id"])
        ys = [any(s < t < e for s, e in wins) for t in ts]
        series = {"whole_alone": [alone[(it["id"], t)] for t in ts],
                  "whole_joint": [joint[(it["id"], t)]["whole"] for t in ts]}
        cvals = [[joint[(it["id"], t)][f"c{i}"] for t in ts] for i in range(1, len(conds) + 1)]
        series["min"] = [min(v) for v in zip(*cvals)] if conds else series["whole_joint"]
        series["prod"] = [round(prod(v), 4) for v in zip(*cvals)] if conds else series["whole_joint"]
        print(f"\n[{it['id']}] \"{it['query']}\"  정답 구간 {wins}")
        entry = {"query": it["query"], "windows": wins, "parts": conds, "methods": {}, "conditions": []}
        for i, (text, vals) in enumerate(zip(conds, cvals), 1):
            inside = [p for p, y in zip(vals, ys) if y]
            outside = [p for p, y in zip(vals, ys) if not y]
            stats = {"text": text, "auc": round(auc(vals, ys), 3), "mean_inside": round(sum(inside) / len(inside), 3),
                     "mean_outside": round(sum(outside) / len(outside), 3), "p": vals}
            entry["conditions"].append(stats)
            print(f"  조건 {i} {''.join(str(min(9, int(p * 10))) for p in vals)}  {text}"
                  f"  (안 {stats['mean_inside']:.2f} / 밖 {stats['mean_outside']:.2f})")
        for m in METHODS:
            ps = series[m]
            tp, fp, fn = confusion(ps, ys, THR)
            peak = max(range(len(ps)), key=lambda k: ps[k])
            row = {"auc": round(auc(ps, ys), 3), "tp": tp, "fp": fp, "fn": fn, "peak_t": ts[peak],
                   "peak_hit": ys[peak], "by_threshold": {str(th): confusion(ps, ys, th) for th in GRID},
                   "p": ps, "y": ys, "decomposed": bool(conds)}
            entry["methods"][m] = row
            rows_by_method[m].append(row)
            print(f"  {LABEL[m]:11s} {''.join(str(min(9, int(p * 10))) for p in ps)}  AUC {row['auc']:.2f}"
                  f"  맞게 {tp} / 엉뚱하게 {fp} / 놓침 {fn}  최고점 {ts[peak]:.0f}초{' 적중' if ys[peak] else ' 빗나감'}")
        print(f"  {'정답':11s} {''.join('^' if y else '.' for y in ys)}")
        entry["t"] = ts
        result["items"][it["id"]] = entry

    def summarize(rs):
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        prec, rec, f1 = prf(tp, fp, fn)
        sweep = {}
        for th in GRID:
            a, b, d = (sum(r["by_threshold"][str(th)][k] for r in rs) for k in range(3))
            sweep[str(th)] = {"tp": a, "fp": b, "fn": d,
                              **dict(zip(("precision", "recall", "f1"),
                                         [None if x is None else round(x, 3) for x in prf(a, b, d)]))}
        best = max(GRID, key=lambda th: sweep[str(th)]["f1"] or 0)
        return {"n": len(rs), "mean_auc": round(sum(r["auc"] for r in rs) / len(rs), 3),
                "pooled_auc": round(auc([p for r in rs for p in r["p"]], [y for r in rs for y in r["y"]]), 3),
                "peak_hits": sum(r["peak_hit"] for r in rs), "tp": tp, "fp": fp, "fn": fn,
                "precision": None if prec is None else round(prec, 3), "recall": round(rec, 3),
                "f1": round(f1, 3), "videos_fp_le_2": sum(r["fp"] <= 2 for r in rs),
                "best_threshold": best, "sweep": sweep}

    for title, keep in (("전체 10개 영상", lambda r: True), ("조건을 나눈 8개 영상", lambda r: r["decomposed"])):
        print(f"\n요약: {title}")
        print(f"{'방식':14s}{'평균 AUC':>8s}{'모은 AUC':>8s}{'최고점':>6s}{'맞게':>6s}{'엉뚱':>6s}{'놓침':>6s}"
              f"{'정밀도':>7s}{'재현율':>7s}{'F1':>6s}   0.7 기준 정밀도/재현율   F1 최고 임계값")
        block = {}
        for m in METHODS:
            s = summarize([r for r in rows_by_method[m] if keep(r)])
            block[m] = s
            s7, sb = s["sweep"]["0.7"], s["sweep"][str(s["best_threshold"])]
            print(f"{LABEL[m]:14s}{s['mean_auc']:>8.2f}{s['pooled_auc']:>8.2f}{s['peak_hits']:>4d}/{s['n']:<2d}"
                  f"{s['tp']:>6d}{s['fp']:>6d}{s['fn']:>6d}{fmt(s['precision']):>7s}{fmt(s['recall']):>7s}"
                  f"{fmt(s['f1']):>6s}   {fmt(s7['precision'])} / {fmt(s7['recall'])}"
                  f"              {s['best_threshold']} ({fmt(sb['precision'])} / {fmt(sb['recall'])}, F1 {fmt(sb['f1'])})")
        result["summary"][title] = block

    # 한 요청에 다른 질문을 함께 넣으면 같은 질문의 답이 달라지는가
    diffs = [abs(alone[k] - joint[k]["whole"]) for k in joint]
    flips = sum((alone[k] >= THR) != (joint[k]["whole"] >= THR) for k in joint)
    with_parts = [abs(alone[k] - joint[k]["whole"]) for k in joint if parts[k[0]]]
    result["independence"] = {"n": len(diffs), "mean_abs_diff": round(sum(diffs) / len(diffs), 4),
                              "max_abs_diff": round(max(diffs), 3), "flips_at_0.5": flips,
                              "mean_abs_diff_with_parts": round(sum(with_parts) / len(with_parts), 4)}
    print(f"\n같은 질문을 혼자 물었을 때와 함께 물었을 때의 차이: 평균 {result['independence']['mean_abs_diff']:.3f}, "
          f"최대 {result['independence']['max_abs_diff']:.2f}, 0.5 기준 판정이 바뀐 곳 {flips}/{len(diffs)}")
    (HERE / "results" / "qv_parts_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    m = new["meta"]
    print(f"요청 {m['saved']}건, 오류 {m['errors']}건, 입력 토큰 {m['input_tokens']:,}, 약 ${m['cost_usd']}")


if __name__ == "__main__":
    main()
