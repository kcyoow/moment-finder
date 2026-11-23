"""찾는 장면이 없는 영상에서 "없음"을 가려내는지 채점한다. 표준 라이브러리만 쓴다.

자료: results/qv2_absent_preds.json. 새 QVHighlights 영상 10개의 각 위치(한 장)에 자기 요청문과
다른 영상의 요청문 19개를 함께 물은 것.
- 없는 쌍: (영상, 다른 영상의 요청문) 190개. 대체로 그 영상에 없는 장면이다
- 있는 쌍: (영상, 자기 요청문) 10개 + 앞서 본 QVHighlights 10개의 (영상, 자기 요청문) 10개

보는 것
1. 영상 하나에 대한 요약값(최고 확률 등)으로 있는 쌍과 없는 쌍이 갈리는가 (AUC)
2. "요약값이 기준 미만이면 없음" 규칙에서 기준을 바꿔 가며: 없는 쌍을 없음으로 맞게 본 비율, 있는 쌍을 살린 비율
3. 자르는 규칙별로 없는 쌍에서 무언가를 잡은 비율(오경보)과 잡은 위치 수

실행: python3 experiments/frame_limit/scripts/qv/absent_eval.py
결과: 표 출력, results/absent_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from pathlib import Path

from qv_eval import auc
from recipe_eval import fill_gaps, load
from smooth_sim import KERNELS, cut

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
SMOOTH = "gauss7"
CELL = 2.0  # 위치 하나가 대표하는 시간(초)


def top3(ps):
    return sum(sorted(ps)[-3:]) / 3


SIGNALS = {
    "최고 확률": max,
    "섞은 뒤 최고 확률": lambda ps: max(KERNELS[SMOOTH](ps)),
    "가장 높은 세 위치의 평균": top3,
}


def main():
    data = json.loads((HERE / "results" / "qv2_absent_preds.json").read_text())
    if data["meta"]["errors"] or data["meta"]["saved"] != data["meta"]["jobs"]:
        print(f"주의: 요청 {data['meta']['jobs']}건 중 저장 {data['meta']['saved']}건, 오류 {data['meta']['errors']}건")
    foreign = json.loads((HERE / "data" / "qv2_foreign.json").read_text())
    text = {it["id"]: it["query"] for tag in ("qv", "qv2")
            for it in json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]}
    by_item = {}
    for r in data["records"]:
        if not r["error"]:
            by_item.setdefault(r["item"], []).append((r["t"], r["nouls"]))
    present, absent = [], []  # (영상의 요청 id, 물은 요청 id, 시각 목록, 확률 목록)
    for item, rows in by_item.items():
        rows.sort()
        ts = [t for t, _ in rows]
        present.append((item, item, ts, [n["whole"] for _, n in rows]))
        for k, src in enumerate(foreign["source"][item], 1):
            absent.append((item, src, ts, [n[f"c{k}"] for _, n in rows]))
    for it in load("qv")[0]:  # 앞서 본 묶음의 있는 쌍 (같은 조건: 한 장, 요청 전체 질문)
        present.append((it["id"], it["id"], it["ts"], it["one"]))
    print(f"있는 쌍 {len(present)}개, 없는 쌍 {len(absent)}개")
    result = {"meta": data["meta"], "present": len(present), "absent": len(absent)}

    print("\n1. 요약값으로 있는 쌍과 없는 쌍을 가르는 정도 (AUC. 1이면 완전히 갈림)")
    result["signal_auc"] = {}
    for name, fn in SIGNALS.items():
        scores = [fn(p[3]) for p in present] + [fn(a[3]) for a in absent]
        labels = [True] * len(present) + [False] * len(absent)
        result["signal_auc"][name] = round(auc(scores, labels), 3)
        ps, ab = sorted(fn(p[3]) for p in present), sorted(fn(a[3]) for a in absent)
        print(f"  {name}: AUC {result['signal_auc'][name]:.3f}   있는 쌍 최소 {ps[0]:.2f}·중앙 {ps[len(ps) // 2]:.2f}"
              f"   없는 쌍 중앙 {ab[len(ab) // 2]:.2f}·상위 10% {ab[int(len(ab) * 0.9)]:.2f}·최대 {ab[-1]:.2f}")

    print("\n2. \"기준 미만이면 없음\" 규칙 (없는 쌍을 맞게 없음으로 본 비율 / 있는 쌍을 살린 비율)")
    result["gate"] = {}
    for name, fn in SIGNALS.items():
        row = {}
        for g in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            rejected = sum(fn(a[3]) < g for a in absent) / len(absent)
            kept = sum(fn(p[3]) >= g for p in present) / len(present)
            row[str(g)] = {"absent_rejected": round(rejected, 3), "present_kept": round(kept, 3)}
        result["gate"][name] = row
        print(f"  {name}: " + "  ".join(f"{g}: {v['absent_rejected']:.0%}/{v['present_kept']:.0%}" for g, v in row.items()))

    print("\n3. 자르는 규칙별 결과")
    print(f"  {'규칙':40s}{'없는 쌍에서 무언가 잡음':>14s}{'없는 쌍 평균 잡은 시간':>14s}{'있는 쌍을 없음으로 봄':>14s}")
    result["rules"] = {}
    rules = [("섞지 않음, 0.5 고정", "none", "abs0.5", None)]
    for gate in (0.3, 0.5, 0.7):
        for rule, label in (("rel_gate", "최고 확률의 절반"), ("mean+std", "평균 + 표준편차"), ("otsu", "Otsu")):
            rules.append((f"넓게 섞음, {label}, 최고 확률 {gate} 미만이면 없음", SMOOTH, rule, gate))
    for label, kernel, rule, gate in rules:
        def caught(ps):
            if gate is not None and max(ps) < gate:
                return [False] * len(ps)
            if rule == "abs0.5":
                return fill_gaps([p >= 0.5 for p in KERNELS[kernel](ps)])
            # cut()의 "없음" 조건(0.3)은 gate가 0.3 이상이면 항상 통과한다
            return fill_gaps(cut(ps, KERNELS[kernel](ps), rule))
        alarms = [sum(caught(a[3])) for a in absent]
        missed = sum(not any(caught(p[3])) for p in present)
        row = {"absent_with_any": round(sum(x > 0 for x in alarms) / len(absent), 3),
               "absent_mean_seconds": round(sum(alarms) * CELL / len(absent), 2),
               "present_called_absent": missed}
        result["rules"][label] = row
        print(f"  {label:40s}{row['absent_with_any']:>13.0%}{row['absent_mean_seconds']:>13.1f}초"
              f"{missed:>10d}/{len(present)}")

    print("\n4. 확률이 가장 높게 나온 없는 쌍 (프레임을 직접 확인할 대상)")
    ranked = sorted(absent, key=lambda a: -max(a[3]))[:12]
    result["top_absent"] = []
    for video, src, ts, ps in ranked:
        peak = max(range(len(ps)), key=lambda k: ps[k])
        high = sum(p >= 0.5 for p in ps)
        result["top_absent"].append({"video": video, "asked": src, "query": text[src], "peak_t": ts[peak],
                                     "peak": ps[peak], "positions_ge_0.5": high})
        print(f"  영상 {video:6s} ← \"{text[src][:62]}\"  최고 {ps[peak]:.2f} ({ts[peak]:.0f}초), 0.5 이상 {high}곳")

    print("\n5. 어떤 요청문이 다른 영상에서 자주 높게 나오나 (최고 확률 0.5 이상인 영상 수)")
    per_query = {}
    for video, src, ts, ps in absent:
        per_query.setdefault(src, []).append(max(ps))
    result["per_query"] = {}
    for src, peaks in sorted(per_query.items(), key=lambda kv: -sum(p >= 0.5 for p in kv[1])):
        n = sum(p >= 0.5 for p in peaks)
        result["per_query"][src] = {"query": text[src], "videos": len(peaks), "peak_ge_0.5": n}
        if n:
            print(f"  {n}/{len(peaks)}  \"{text[src]}\"")
    (HERE / "results" / "absent_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    m = data["meta"]
    print(f"\n요청 {m['saved']}건, 오류 {m['errors']}건, 입력 토큰 {m['input_tokens']:,}, 약 ${m['cost_usd']}")


if __name__ == "__main__":
    main()
