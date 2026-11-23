"""QVHighlights 실행 결과(results/qv_preds.json)를 채점한다. 표준 라이브러리만 쓴다.

판정 규칙 (실행 전에 정함)
- 위치(2초 간격)의 시각이 요청에 맞는 구간 안이면 양성. 위치 하나는 앞뒤 1초를 대표한다
- AUC: 구간 안의 위치가 밖의 위치보다 확률이 높은 정도. 0.5면 구분 못 함
- 0.5 기준: 확률이 0.5 이상이면 "잡았다". 맞게 잡은 위치(TP), 엉뚱하게 잡은 위치(FP), 놓친 위치(FN)
  - 정밀도 = TP / (TP + FP), 재현율 = TP / (TP + FN)
  - 엉뚱하게 잡은 위치 중 정답 구간 끝에서 4초 안에 있는 것은 "경계 근처"로 따로 센다
- 예측 구간: 0.5 이상인 연속 위치의 묶음. 정답 구간과 조금이라도 겹치면 맞는 구간으로 센다
- 최고점: 확률이 가장 높은 위치가 정답 구간 안인가

실행: python3 experiments/frame_limit/scripts/qv/qv_eval.py
결과: 표 출력, results/qv_eval.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CONDS = ["one", "win4"]
LABEL = {"one": "한 장", "win4": "4장"}
THR = 0.5
CELL = 2.0
NEAR = 4.0
GRID = [round(0.1 * k, 1) for k in range(1, 10)]


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def runs(ts, ps, thr):
    """thr 이상인 연속 위치를 구간 [시작, 끝](초)으로 묶는다."""
    out, cur = [], None
    for t, p in zip(ts, ps):
        if p >= thr:
            cur = [cur[0], t + CELL / 2] if cur else [t - CELL / 2, t + CELL / 2]
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def overlaps(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def confusion(ps, ys, thr):
    tp = sum(p >= thr and y for p, y in zip(ps, ys))
    fp = sum(p >= thr and not y for p, y in zip(ps, ys))
    fn = sum(p < thr and y for p, y in zip(ps, ys))
    return tp, fp, fn


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return prec, rec, f1


def fmt(x):
    return "  - " if x is None else f"{x:.2f}"


def main():
    data = json.loads((HERE / "results" / "qv_preds.json").read_text())
    items = json.loads((HERE / "data" / "qv_items.json").read_text())["items"]
    failed = [r for r in data["records"] if r["error"]]
    if failed or data["meta"]["saved"] != data["meta"]["jobs"]:
        print(f"주의: 요청 {data['meta']['jobs']}건 중 저장 {data['meta']['saved']}건, 실패 {len(failed)}건. 수치가 불완전합니다")
    series = {}
    for r in data["records"]:
        if not r["error"]:
            series.setdefault((r["item"], r["frames"]), []).append((r["t"], r["noul"]))
    result = {"meta": data["meta"], "threshold": THR, "items": {}, "summary": {}}
    per_cond = {c: [] for c in CONDS}
    for it in items:
        wins = it["windows"]
        print(f"\n[{it['id']}] \"{it['query']}\"  정답 구간 {wins}")
        rows = {}
        for c in CONDS:
            pts = sorted(series.get((it["id"], c), []))
            if not pts:
                continue
            ts, ps = [t for t, _ in pts], [p for _, p in pts]
            ys = [any(s < t < e for s, e in wins) for t in ts]
            tp, fp, fn = confusion(ps, ys, THR)
            prec, rec, _ = prf(tp, fp, fn)
            fp_near = sum(p >= THR and not y and any(min(abs(t - s), abs(t - e)) <= NEAR for s, e in wins)
                          for t, p, y in zip(ts, ps, ys))
            segs = runs(ts, ps, THR)
            seg_ok = sum(any(overlaps(sg, w) for w in wins) for sg in segs)
            win_hit = sum(any(overlaps(sg, w) for sg in segs) for w in wins)
            peak = max(range(len(ps)), key=lambda k: ps[k])
            row = {"auc": round(auc(ps, ys), 3), "tp": tp, "fp": fp, "fn": fn, "fp_near": fp_near,
                   "positives": sum(ys), "negatives": len(ys) - sum(ys),
                   "precision": None if prec is None else round(prec, 3),
                   "recall": None if rec is None else round(rec, 3),
                   "segments": segs, "segments_correct": seg_ok, "windows": len(wins), "windows_hit": win_hit,
                   "peak_t": ts[peak], "peak_p": ps[peak], "peak_hit": ys[peak],
                   "mean_inside": round(sum(p for p, y in zip(ps, ys) if y) / sum(ys), 3),
                   "mean_outside": round(sum(p for p, y in zip(ps, ys) if not y) / (len(ys) - sum(ys)), 3),
                   "by_threshold": {str(th): confusion(ps, ys, th) for th in GRID}, "t": ts, "p": ps, "y": ys}
            rows[c] = row
            per_cond[c].append(row)
            curve = "".join(str(min(9, int(p * 10))) for p in ps)
            print(f"  {LABEL[c]:4s} {curve}")
            print(f"       AUC {row['auc']:.2f}  최고점 {ts[peak]:.0f}초({ps[peak]:.2f}){' 적중' if ys[peak] else ' 빗나감'}"
                  f"  맞게 {tp} / 엉뚱하게 {fp}(경계 근처 {fp_near}) / 놓침 {fn}"
                  f"  정밀도 {fmt(prec)} 재현율 {fmt(rec)}  구간 {len(segs)}개 중 맞는 것 {seg_ok}"
                  f"  안 {row['mean_inside']:.2f} / 밖 {row['mean_outside']:.2f}")
        any_ts = next(iter(rows.values()))["t"]
        print(f"  정답 {''.join('^' if any(s < t < e for s, e in wins) else '.' for t in any_ts)}"
              f"  (왼쪽부터 1초, 2초 간격. 숫자는 확률의 첫째 자리)")
        result["items"][it["id"]] = {"query": it["query"], "windows": wins, "conditions": rows}

    print(f"\n요약 (영상 {len(items)}개, 위치 {sum(len(r['t']) for r in per_cond[CONDS[0]])}곳)")
    for c in CONDS:
        rs = per_cond[c]
        if not rs:
            continue
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        prec, rec, f1 = prf(tp, fp, fn)
        neg = sum(r["negatives"] for r in rs)
        sweep = {}
        for th in GRID:
            a, b, d = (sum(r["by_threshold"][str(th)][k] for r in rs) for k in range(3))
            sweep[str(th)] = dict(zip(("tp", "fp", "fn", "precision", "recall", "f1"),
                                      (a, b, d, *[None if x is None else round(x, 3) for x in prf(a, b, d)])))
        best = max(GRID, key=lambda th: sweep[str(th)]["f1"] or 0)
        summ = {"n": len(rs), "mean_auc": round(sum(r["auc"] for r in rs) / len(rs), 3),
                "pooled_auc": round(auc([p for r in rs for p in r["p"]], [y for r in rs for y in r["y"]]), 3),
                "peak_hits": sum(r["peak_hit"] for r in rs),
                "tp": tp, "fp": fp, "fn": fn, "fp_near": sum(r["fp_near"] for r in rs), "negatives": neg,
                "precision": round(prec, 3) if prec is not None else None, "recall": round(rec, 3),
                "f1": round(f1, 3), "false_positive_rate": round(fp / neg, 4),
                "videos_without_fp": sum(r["fp"] == 0 for r in rs),
                "videos_with_any_tp": sum(r["tp"] > 0 for r in rs),
                "segments": sum(len(r["segments"]) for r in rs),
                "segments_correct": sum(r["segments_correct"] for r in rs),
                "windows": sum(r["windows"] for r in rs), "windows_hit": sum(r["windows_hit"] for r in rs),
                "best_threshold": best, "sweep": sweep}
        result["summary"][c] = summ
        print(f"{LABEL[c]:4s} 평균 AUC {summ['mean_auc']:.2f} (전체를 한데 모으면 {summ['pooled_auc']:.2f})"
              f"  최고점 적중 {summ['peak_hits']}/{summ['n']}")
        print(f"     0.5 기준: 맞게 {tp} / 엉뚱하게 {fp} (경계 근처 {summ['fp_near']}) / 놓침 {fn}"
              f"  정밀도 {fmt(prec)} 재현율 {fmt(rec)} F1 {fmt(f1)}"
              f"  정답 밖 위치 {neg}곳 중 {fp}곳({fp / neg:.1%})을 잡음, 엉뚱한 곳이 없는 영상 {summ['videos_without_fp']}/{summ['n']}")
        print(f"     예측 구간 {summ['segments']}개 중 맞는 것 {summ['segments_correct']}개,"
              f" 정답 구간 {summ['windows']}개 중 찾은 것 {summ['windows_hit']}개")
        print("     임계값별 정밀도/재현율: " + "  ".join(
            f"{th}:{fmt(sweep[str(th)]['precision'])}/{fmt(sweep[str(th)]['recall'])}" for th in GRID)
              + f"  (F1 최고 {best})")
    (HERE / "results" / "qv_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    m = data["meta"]
    print(f"\n요청 {m['saved']}건, 오류 {m['errors']}건, 입력 토큰 {m['input_tokens']:,}, 약 ${m['cost_usd']}")


if __name__ == "__main__":
    main()
