"""loc_run.py 결과를 채점한다. 표준 라이브러리만 쓴다.

판정 규칙 (실행 전에 정함)
- 위치(1초 간격)의 시각이 정답 구간 안이면 양성
- AUC: 정답 구간 안의 위치가 밖의 위치보다 확률이 높은 정도. 0.5면 구분 못 함
- 최고점: 확률이 가장 높은 위치가 정답 구간 안인가
- 예측 구간: 확률이 임계값 이상인 연속 위치 중 최고점을 포함한 묶음. 위치 하나는 앞뒤 0.5초를 대표한다.
  최고점이 임계값보다 낮으면 "못 찾음"(IoU 0)
- 처음/중간/끝: TVBench가 보여 주는 구간 안에서, 위치의 2/3 이상이 0.5 이상이면 "내내",
  아니면 최고점이 놓인 삼등분

실행: python3 experiments/frame_limit/scripts/loc/loc_eval.py
결과: 표 출력, results/loc_eval.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CONDS = [("one", "plain"), ("one", "refined"), ("win4", "plain"), ("win4", "refined")]
LABEL = {("one", "plain"): "한 장·짧은 질문", ("one", "refined"): "한 장·다듬은 질문",
         ("win4", "plain"): "4장·짧은 질문", ("win4", "refined"): "4장·다듬은 질문"}
THR = 0.5
CELL = 1.0  # 위치 하나가 대표하는 시간(초)
GRID = [round(0.1 * k, 1) for k in range(1, 10)]


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def segment(ts, ps, thr):
    i = max(range(len(ps)), key=lambda k: ps[k])
    if ps[i] < thr:
        return None
    lo = hi = i
    while lo > 0 and ps[lo - 1] >= thr:
        lo -= 1
    while hi < len(ps) - 1 and ps[hi + 1] >= thr:
        hi += 1
    return [ts[lo] - CELL / 2, ts[hi] + CELL / 2]


def iou(a, b):
    if a is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def coarse(ts, ps, start, end):
    inside = [(t, p) for t, p in zip(ts, ps) if start <= t <= end]
    if sum(p >= THR for _, p in inside) >= len(inside) * 2 / 3:
        return "throughout"
    peak = max(inside, key=lambda x: x[1])[0]
    rel = (peak - start) / (end - start)
    return "beginning" if rel < 1 / 3 else ("middle" if rel < 2 / 3 else "end")


def main():
    data = json.loads((HERE / "results" / "loc_preds.json").read_text())
    items = json.loads((HERE / "data" / "loc_items.json").read_text())["items"]
    failed = [r for r in data["records"] if r["error"]]
    if failed or data["meta"]["saved"] != data["meta"]["jobs"]:
        print(f"주의: 요청 {data['meta']['jobs']}건 중 저장 {data['meta']['saved']}건, 실패 {len(failed)}건. 수치가 불완전합니다")
    series = {}
    for r in data["records"]:
        if not r["error"]:
            series.setdefault((r["item"], r["frames"], r["prompt"]), []).append((r["t"], r["noul"]))
    result = {"meta": data["meta"], "threshold": THR, "items": {}, "summary": {}}
    per_cond = {c: [] for c in CONDS}
    for it in items:
        s, e = it["moment"]
        print(f"\n[{it['id']}] \"{it['request_ko']}\"  정답 구간 {s}~{e}초")
        rows = {}
        for c in CONDS:
            pts = sorted(series.get((it["id"], *c), []))
            if not pts:
                continue
            ts, ps = [t for t, _ in pts], [p for _, p in pts]
            ys = [s <= t <= e for t in ts]
            peak = max(range(len(ps)), key=lambda k: ps[k])
            seg = segment(ts, ps, THR)
            inside = [p for p, y in zip(ps, ys) if y]
            outside = [p for p, y in zip(ps, ys) if not y]
            row = {"auc": round(auc(ps, ys), 3), "peak_t": ts[peak], "peak_p": ps[peak], "peak_hit": ys[peak],
                   "segment": seg, "iou": round(iou(seg, [s, e]), 3),
                   "mean_inside": round(sum(inside) / len(inside), 3),
                   "mean_outside": round(sum(outside) / len(outside), 3),
                   "iou_by_threshold": {str(th): round(iou(segment(ts, ps, th), [s, e]), 3) for th in GRID},
                   "t": ts, "p": ps}
            if it.get("tvbench"):
                tv = it["tvbench"]
                row["coarse"] = coarse(ts, ps, tv["start"], tv["end"])
                row["coarse_correct"] = row["coarse"] == tv["answer"]
            rows["/".join(c)] = row
            per_cond[c].append(row)
            curve = "".join(str(min(9, int(p * 10))) for p in ps)
            seg_txt = "못 찾음" if seg is None else f"{seg[0]:.1f}~{seg[1]:.1f}초"
            print(f"  {LABEL[c]:12s} {curve}  AUC {row['auc']:.2f}  최고점 {ts[peak]:.0f}초({ps[peak]:.2f})"
                  f"{' 적중' if ys[peak] else ' 빗나감'}  구간 {seg_txt}  IoU {row['iou']:.2f}"
                  f"  안 {row['mean_inside']:.2f} / 밖 {row['mean_outside']:.2f}"
                  + (f"  답 {row['coarse']}{'(O)' if row['coarse_correct'] else '(X)'}" if "coarse" in row else ""))
        any_ts = next(iter(rows.values()))["t"]
        print(f"  {'정답 구간':12s} " + "".join("^" if s <= t <= e else "." for t in any_ts)
              + f"  (왼쪽부터 {any_ts[0]:.0f}초, 1초 간격. 숫자는 확률의 첫째 자리)")
        result["items"][it["id"]] = {"request_ko": it["request_ko"], "moment": it["moment"], "conditions": rows}

    print(f"\n요약 (영상 {len(items)}개)")
    print(f"{'조건':14s}{'평균 AUC':>9s}{'최고점 적중':>9s}{'평균 IoU@0.5':>12s}{'IoU≥0.5':>9s}"
          f"{'가장 좋은 임계값':>12s}{'처음/중간/끝':>10s}")
    for c in CONDS:
        rs = per_cond[c]
        if not rs:
            continue
        best = max(GRID, key=lambda th: sum(r["iou_by_threshold"][str(th)] for r in rs))
        best_iou = sum(r["iou_by_threshold"][str(best)] for r in rs) / len(rs)
        co = [r for r in rs if "coarse" in r]
        summ = {"n": len(rs), "mean_auc": round(sum(r["auc"] for r in rs) / len(rs), 3),
                "peak_hits": sum(r["peak_hit"] for r in rs),
                "mean_iou": round(sum(r["iou"] for r in rs) / len(rs), 3),
                "iou_ge_0.5": sum(r["iou"] >= 0.5 for r in rs),
                "best_threshold": best, "best_threshold_mean_iou": round(best_iou, 3),
                "coarse_correct": sum(r["coarse_correct"] for r in co), "coarse_n": len(co),
                "mean_inside": round(sum(r["mean_inside"] for r in rs) / len(rs), 3),
                "mean_outside": round(sum(r["mean_outside"] for r in rs) / len(rs), 3),
                "pooled_auc": round(auc([p for r in rs for p in r["p"]],
                                        [y for r, it in zip(rs, items) for y in
                                         [it["moment"][0] <= t <= it["moment"][1] for t in r["t"]]]), 3)}
        result["summary"]["/".join(c)] = summ
        print(f"{LABEL[c]:14s}{summ['mean_auc']:>9.2f}{summ['peak_hits']:>7d}/{summ['n']}{summ['mean_iou']:>12.2f}"
              f"{summ['iou_ge_0.5']:>7d}/{summ['n']}{best:>8.1f} → {best_iou:.2f}"
              f"{summ['coarse_correct']:>8d}/{summ['coarse_n']}"
              f"   구간 안 평균 {summ['mean_inside']:.2f} / 밖 평균 {summ['mean_outside']:.2f}")
    print("(가장 좋은 임계값은 같은 5개 영상에서 고른 값이라 낙관적이다)")
    (HERE / "results" / "loc_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    m = data["meta"]
    print(f"\n요청 {m['saved']}건, 오류 {m['errors']}건, 입력 토큰 {m['input_tokens']:,}, 약 ${m['cost_usd']}")


if __name__ == "__main__":
    main()
