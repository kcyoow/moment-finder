"""위치별 확률을 겹치지 않는 묶음으로 평균 내는 방식을 미끄러지는 평균과 비교한다.
이미 받은 확률로 계산하고 새 요청을 보내지 않는다. 표준 라이브러리만 쓴다.

- 묶음 평균: 위치를 앞에서부터 n개씩 끊어 묶고, 묶음 안의 평균을 그 묶음의 값으로 쓴다.
  어디서부터 끊느냐(시작점)에 따라 결과가 달라지므로 가능한 시작점 n가지를 모두 계산해
  평균과 최소~최대를 낸다
- 미끄러지는 평균: 위치마다 자기와 앞뒤 위치를 섞는다 (smooth_sim.py의 uni3, uni5, gauss7)

자르는 기준은 smooth_sim.py와 같다(최고 확률의 절반, 평균 + 표준편차, Otsu. 한 칸 끊김 잇기 적용).
수치는 새 영상(sta, qv2)의 것이다.

실행: python3 experiments/frame_limit/scripts/qv/block_sim.py
결과: 표 출력, results/block_sim.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from pathlib import Path

from qv_eval import auc, prf
from recipe_eval import NAME, fill_gaps, load
from smooth_sim import KERNELS, cut

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
RULES = [("rel_gate", "최고 확률의 절반"), ("mean+std", "평균 + 표준편차"), ("otsu", "Otsu")]
CELLS = [("sta", "one"), ("sta", "win4"), ("qv2", "one")]


def block_average(ps, n, offset):
    """offset번째 위치부터 n개씩 묶는다. 앞에 남는 위치들은 따로 한 묶음이 된다."""
    out, start = [], 0
    edges = ([offset] if offset else []) + list(range(offset + n, len(ps) + n, n))
    for end in edges:
        end = min(end, len(ps))
        if end > start:
            out.extend([sum(ps[start:end]) / (end - start)] * (end - start))
        start = end
    return out


def evaluate(items, frames, transform):
    aucs, counts = [], {rule: [0, 0, 0] for rule, _ in RULES}
    for it in items:
        if not it[frames]:
            continue
        ps = transform(it[frames])
        a = auc(ps, it["y"])
        if a is not None:
            aucs.append(a)
        for rule, _ in RULES:
            caught = fill_gaps(cut(it[frames], ps, rule))
            counts[rule][0] += sum(c and y for c, y in zip(caught, it["y"]))
            counts[rule][1] += sum(c and not y for c, y in zip(caught, it["y"]))
            counts[rule][2] += sum(not c and y for c, y in zip(caught, it["y"]))
    return {"auc": sum(aucs) / len(aucs), **{rule: prf(*counts[rule])[2] for rule, _ in RULES}}


def main():
    sets = {tag: load(tag)[0] for tag in ("sta", "qv2")}
    label = {(t, f): f"{NAME[t]} {'한 장' if f == 'one' else '4장'}" for t, f in CELLS}
    methods = [("섞지 않음", [KERNELS["none"]]),
               ("미끄러지는 평균 3칸", [KERNELS["uni3"]]),
               ("미끄러지는 평균 5칸", [KERNELS["uni5"]]),
               ("미끄러지는 종 모양 7칸", [KERNELS["gauss7"]])]
    for n in (2, 3, 4, 5, 6):
        methods.append((f"묶음 평균 {n}칸", [lambda ps, n=n, o=o: block_average(ps, n, o) for o in range(n)]))
    result = {}
    for metric, title in [("auc", "평균 AUC")] + [(rule, f"F1, 자르는 기준: {name}") for rule, name in RULES]:
        print(f"\n{title}  (묶음 평균은 시작점을 바꿔 가며 낸 평균, 괄호는 최소~최대)")
        print(f"{'방식':18s}" + "".join(f"{label[c]:>30s}" for c in CELLS))
        for name, transforms in methods:
            cells = []
            for tag, frames in CELLS:
                vals = [evaluate(sets[tag], frames, t)[metric] for t in transforms]
                mean = sum(vals) / len(vals)
                result.setdefault(name, {}).setdefault(label[(tag, frames)], {})[metric] = {
                    "mean": round(mean, 3), "min": round(min(vals), 3), "max": round(max(vals), 3)}
                cells.append(f"{mean:.3f}" + (f" ({min(vals):.2f}~{max(vals):.2f})" if len(vals) > 1 else ""))
            print(f"{name:18s}" + "".join(f"{c:>30s}" for c in cells))

    # 찾는 장면이 없는 영상 가리기: 묶음 평균의 최고값으로도 갈리는가
    absent_file = HERE / "results" / "qv2_absent_preds.json"
    if absent_file.exists():
        data = json.loads(absent_file.read_text())
        rows = {}
        for r in data["records"]:
            rows.setdefault(r["item"], []).append((r["t"], r["nouls"]))
        present, absent = [], []
        for item, recs in rows.items():
            recs.sort()
            present.append([n["whole"] for _, n in recs])
            absent.extend([[n[k] for _, n in recs] for k in recs[0][1] if k != "whole"])
        present += [it["one"] for it in load("qv")[0]]
        print("\n있는 쌍과 없는 쌍을 최고값으로 가르는 정도 (AUC)")
        result["presence_auc"] = {}
        for name, transforms in methods:
            vals = [auc([max(t(p)) for p in present] + [max(t(a)) for a in absent],
                        [True] * len(present) + [False] * len(absent)) for t in transforms]
            result["presence_auc"][name] = round(sum(vals) / len(vals), 3)
            print(f"  {name:18s} {sum(vals) / len(vals):.3f}")
    (HERE / "results" / "block_sim.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
