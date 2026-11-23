"""SigLIP 2의 유사도와 Jev의 확률을 같은 프레임·같은 요청문·같은 정답으로 견준다. 표준 라이브러리만 쓴다.

자료
- results/siglip_scores_<모델>.json: 모든 프레임 × 모든 요청문의 코사인 유사도 (siglip_colab.py가 Colab에서 만든다)
- results/<묶음>_preds.json: Jev 확률 (이미 있는 것. 새 요청 없음)

보는 것
1. 순서의 질: 정답 구간 안의 위치가 밖의 위치보다 점수가 높은 정도(AUC). 섞기 전과 5칸 평균 뒤
2. 가장 점수가 높은 위치가 정답 구간 안인 항목 수
3. 같은 규칙으로 잘랐을 때의 F1(5칸 평균 → Otsu → 한 칸 끊김 잇기), 영상마다 최선의 기준을 골랐을 때의 F1(상한)
4. 찾는 장면이 없는 영상 가리기: 영상의 최고 점수로 있는 쌍과 없는 쌍이 갈리는 정도(AUC)
5. SigLIP으로 후보를 먼저 추리고 Jev는 후보만 볼 때: 남는 정답 위치 비율과 최종 F1

SigLIP의 4장 조건은 네 프레임 유사도의 평균이다. 자르는 규칙은 두 모델에 똑같이 적용한다
(Jev 쪽의 "최고 확률 0.3 미만이면 없음" 조건은 유사도에 쓸 수 없어서 여기서는 뺀다. 그래서 Jev의 F1이
REPORT_recipe.md의 값과 조금 다를 수 있다).

실행: python3 experiments/frame_limit/scripts/embed/siglip_eval.py            (채점)
      python3 experiments/frame_limit/scripts/embed/siglip_eval.py --selfcheck (Jev 확률을 SigLIP 자리에 넣어 계산 경로 점검)
결과: 표 출력, results/siglip_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import math
import sys
from pathlib import Path

from qv_eval import auc, prf
from recipe_eval import NAME, fill_gaps, load
from smooth_sim import KERNELS, otsu

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
SMOOTH = "uni5"
NEW = [("sta", "one"), ("sta", "win4"), ("qv2", "one")]
DEV = [("loc", "one"), ("loc", "win4"), ("qv", "one"), ("qv", "win4")]
KEEP = [0.1, 0.2, 0.3, 0.5]
MODEL_FILES = {"SigLIP 2 base": "siglip_scores_base.json", "SigLIP 2 so400m": "siglip_scores_so400m.json"}


class Scores:
    """(프레임 경로, 요청문) → 점수."""

    def __init__(self, table):
        self.table = table

    @classmethod
    def from_file(cls, path):
        d = json.loads(path.read_text())
        col = {q: j for j, q in enumerate(d["queries"])}
        rows = {p: row for p, row in zip(d["frames"], d["cos"])}
        obj = cls(None)
        obj.rows, obj.col, obj.meta = rows, col, {k: d[k] for k in d if k not in ("frames", "queries", "cos")}
        return obj

    def get(self, path, query):
        if self.table is not None:
            return self.table.get((path, query))
        row = self.rows.get(path)
        return None if row is None or query not in self.col else row[self.col[query]]


def bundle(tag):
    """묶음의 항목에 영상과 프레임 경로를 붙인다."""
    items = load(tag)[0]
    meta = {it["id"]: it for it in json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]}
    index = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
    for it in items:
        v = index[meta[it["id"]]["video"]]
        it["video"] = meta[it["id"]]["video"]
        it["query"] = meta[it["id"]]["sentence"]  # Jev 질문에 넣은 문장 그대로
        it["path_one"] = [v["frames"][f"{t:.2f}"] for t in it["ts"]]
        it["path_win4"] = [[v["frames"].get(f"{round(t + o, 2):.2f}") for o in v["window_offsets"]] for t in it["ts"]]
    return items


def series(scores, it, cond, query=None):
    """항목의 위치별 점수. 값이 하나라도 없으면 None."""
    q = query or it["query"]
    if cond == "one":
        vals = [scores.get(p, q) for p in it["path_one"]]
    else:
        vals = []
        for paths in it["path_win4"]:
            got = [scores.get(p, q) if p else None for p in paths]
            vals.append(None if any(g is None for g in got) else sum(got) / len(got))
    return None if any(v is None for v in vals) else vals


def smooth(ps):
    return KERNELS[SMOOTH](ps)


def cut_same(ps):
    """두 모델에 똑같이 쓰는 규칙: 5칸 평균 → Otsu → 한 칸 끊김 잇기."""
    sm = smooth(ps)
    thr = otsu(sm)
    return fill_gaps([p >= thr for p in sm])


def counts(caught, y):
    return (sum(c and t for c, t in zip(caught, y)), sum(c and not t for c, t in zip(caught, y)),
            sum(not c and t for c, t in zip(caught, y)))


def best_counts(ps, y):
    """정답을 보고 고른 가장 좋은 기준에서의 개수 (상한)."""
    sm, best = smooth(ps), None
    for thr in sorted(set(sm)):
        c = counts([p >= thr for p in sm], y)
        f = prf(*c)[2] or 0.0
        if best is None or f > best[0]:
            best = (f, c)
    return best[1]


def evaluate(pairs):
    """pairs: [(위치별 점수, 정답 여부)]. 항목별 AUC 평균, 최고점 적중, F1 두 가지."""
    raw, sm, hit = [], [], 0
    same, top = [0, 0, 0], [0, 0, 0]
    for ps, y in pairs:
        a, b = auc(ps, y), auc(smooth(ps), y)
        if a is not None:
            raw.append(a)
            sm.append(b)
        hit += y[max(range(len(ps)), key=lambda k: ps[k])]
        same = [s + c for s, c in zip(same, counts(cut_same(ps), y))]
        top = [s + c for s, c in zip(top, best_counts(ps, y))]
    return {"items": len(pairs), "auc_raw": sum(raw) / len(raw), "auc_smooth": sum(sm) / len(sm), "peak_hit": hit,
            "f1_same": prf(*same)[2], "precision_same": prf(*same)[0], "recall_same": prf(*same)[1],
            "f1_best": prf(*top)[2]}


def first_pass(items, cond, scores, keep):
    """SigLIP(한 장, 5칸 평균)의 상위 keep 비율만 Jev가 본다고 할 때. 보지 않은 위치의 Jev 확률은 0으로 둔다."""
    total = [0, 0, 0]
    kept_gt = all_gt = 0
    for it in items:
        if not it[cond]:
            continue
        s = series(scores, it, "one")
        if s is None:
            return None
        sm = smooth(s)
        n = max(1, math.ceil(keep * len(sm)))
        chosen = set(sorted(range(len(sm)), key=lambda k: -sm[k])[:n])
        kept_gt += sum(1 for k, t in enumerate(it["y"]) if t and k in chosen)
        all_gt += sum(it["y"])
        masked = [p if k in chosen else 0.0 for k, p in enumerate(it[cond])]
        total = [a + c for a, c in zip(total, counts(cut_same(masked), it["y"]))]
    return {"gt_kept": kept_gt / all_gt, "f1": prf(*total)[2]}


def presence(sets, scores):
    """qv2 영상 × QVHighlights 요청문 20개. 자기 요청문은 있는 쌍, 나머지 19개는 없는 쌍. qv의 자기 요청문 쌍도 있는 쌍에 더한다."""
    queries = {it["id"]: it["query"] for tag in ("qv", "qv2") for it in sets[tag]}
    present, absent = [], []
    for it in sets["qv2"]:
        for qid, q in queries.items():
            s = series(scores, it, "one", q)
            if s is None:
                return None
            (present if qid == it["id"] else absent).append(max(s))
    for it in sets["qv"]:
        s = series(scores, it, "one")
        if s is None:
            return None
        present.append(max(s))
    return {"auc": auc(present + absent, [True] * len(present) + [False] * len(absent)),
            "present": len(present), "absent": len(absent)}


def jev_as_scores(sets, absent_run):
    """점검용: Jev 확률을 (프레임, 요청문) 표로 옮긴다. 이 표로 채점하면 Jev와 같은 숫자가 나와야 한다.
    absent_run이 참이면 '없는 영상' 실행의 값도 넣는다(그 실행은 qv2의 자기 요청문도 다시 물었다)."""
    table = {}
    for items in sets.values():
        for it in items:
            for p, v in zip(it["path_one"], it["one"]):
                table[(p, it["query"])] = v
    if not absent_run:
        return Scores(table)
    absent = json.loads((HERE / "results" / "qv2_absent_preds.json").read_text())
    foreign = json.loads((HERE / "data" / "qv2_foreign.json").read_text())
    by_id = {it["id"]: it for it in sets["qv2"]}
    path_at = {it["id"]: dict(zip(it["ts"], it["path_one"])) for it in sets["qv2"]}
    for r in absent["records"]:
        if r["error"]:
            continue
        for k, text in enumerate(foreign["parts"][r["item"]], 1):
            table[(path_at[r["item"]][r["t"]], text)] = r["nouls"][f"c{k}"]
        table[(path_at[r["item"]][r["t"]], by_id[r["item"]]["query"])] = r["nouls"]["whole"]
    return Scores(table)


def jev_presence(sets):
    data = json.loads((HERE / "results" / "qv2_absent_preds.json").read_text())
    rows = {}
    for r in data["records"]:
        if not r["error"]:
            rows.setdefault(r["item"], []).append(r["nouls"])
    present = [max(n["whole"] for n in ns) for ns in rows.values()] + [max(it["one"]) for it in sets["qv"]]
    absent = [max(n[k] for n in ns) for ns in rows.values() for k in ns[0] if k != "whole"]
    return {"auc": auc(present + absent, [True] * len(present) + [False] * len(absent)),
            "present": len(present), "absent": len(absent)}


def fmt(x, digits=3):
    return "   -  " if x is None else f"{x:.{digits}f}"


def main():
    selfcheck = "--selfcheck" in sys.argv
    sets = {tag: bundle(tag) for tag in ("loc", "qv", "qv2", "sta")}
    label = {(t, c): f"{NAME[t]} {'한 장' if c == 'one' else '4장'}" for t, c in NEW + DEV}
    models = {}
    if selfcheck:
        models["Jev 확률을 넣은 점검"] = jev_as_scores(sets, absent_run=False)
    else:
        for name, fname in MODEL_FILES.items():
            path = HERE / "results" / fname
            if path.exists():
                models[name] = Scores.from_file(path)
        if not models:
            raise SystemExit("results/siglip_scores_*.json 이 없다. Colab에서 siglip_colab.py를 먼저 돌린다.")

    result = {"smooth": SMOOTH, "cells": {}, "models": {n: getattr(s, "meta", {}) for n, s in models.items()}}
    rows = {"Jev": {}}
    for cell in NEW + DEV:
        tag, cond = cell
        rows["Jev"][cell] = evaluate([(it[cond], it["y"]) for it in sets[tag] if it[cond]])
        for name, scores in models.items():
            pairs = [(series(scores, it, cond), it["y"]) for it in sets[tag] if it[cond]]
            rows.setdefault(name, {})[cell] = None if any(p is None for p, _ in pairs) else evaluate(pairs)
    for name in rows:
        result["cells"][name] = {label[c]: v for c, v in rows[name].items()}

    def table(title, key, cells, digits=3, count=False):
        print(f"\n{title}")
        print(f"{'':22s}" + "".join(f"{label[c]:>24s}" for c in cells))
        for name in rows:
            vals = []
            for c in cells:
                r = rows[name][c]
                vals.append("   -  " if r is None else (f"{r[key]}/{r['items']}" if count else fmt(r[key], digits)))
            print(f"{name:22s}" + "".join(f"{v:>24s}" for v in vals))

    for group, cells in (("새 영상", NEW), ("규칙을 정할 때 본 영상", DEV)):
        print(f"\n===== {group} =====")
        table("1. 순서의 질 (평균 AUC, 섞기 전)", "auc_raw", cells)
        table("   순서의 질 (평균 AUC, 5칸 평균 뒤)", "auc_smooth", cells)
        table("2. 가장 점수가 높은 위치가 정답 구간 안인 항목", "peak_hit", cells, count=True)
        table("3. 같은 규칙으로 자른 F1 (5칸 평균 → Otsu → 끊김 잇기)", "f1_same", cells)
        table("   영상마다 최선의 기준을 골랐을 때의 F1 (상한)", "f1_best", cells)

    print("\n===== 요청 종류별 (Charades 새 12개, 평균 AUC, 섞기 전) =====")
    result["by_needs"] = {}
    for needs, title in (("motion", "동작이 바뀌는 요청"), ("single", "한 장으로 보이는 요청")):
        group = [it for it in sets["sta"] if it["needs"] == needs]
        line = {}
        for cond in ("one", "win4"):
            line[f"Jev {cond}"] = evaluate([(it[cond], it["y"]) for it in group])["auc_raw"]
            for name, scores in models.items():
                pairs = [(series(scores, it, cond), it["y"]) for it in group]
                line[f"{name} {cond}"] = None if any(p is None for p, _ in pairs) else evaluate(pairs)["auc_raw"]
        result["by_needs"][title] = line
        print(f"  {title} ({len(group)}개): " + ", ".join(f"{k} {fmt(v)}" for k, v in line.items()))

    print("\n===== 4. 찾는 장면이 없는 영상 가리기 (영상의 최고 점수로 가르는 AUC) =====")
    result["presence"] = {"Jev": jev_presence(sets)}
    for name, scores in models.items():
        result["presence"][name] = presence(sets, jev_as_scores(sets, absent_run=True) if selfcheck else scores)
    for name, r in result["presence"].items():
        print(f"  {name:22s} " + ("   -  " if r is None else f"{r['auc']:.3f}  (있는 쌍 {r['present']}, 없는 쌍 {r['absent']})"))

    print("\n===== 5. SigLIP으로 후보를 추리고 Jev는 후보만 볼 때 (새 영상) =====")
    print("  남긴 비율 = Jev에 보내는 위치의 비율. 정답 유지 = 정답 위치 중 후보에 남은 비율. F1 = 같은 규칙으로 자른 결과")
    result["first_pass"] = {}
    for name, scores in models.items():
        for cell in NEW:
            tag, cond = cell
            base = rows["Jev"][cell]["f1_same"]
            line = {}
            for keep in KEEP:
                line[str(keep)] = first_pass(sets[tag], cond, scores, keep)
            result["first_pass"].setdefault(name, {})[label[cell]] = {"jev_all": base, **line}
            print(f"  {name}, Jev {label[cell]} (전부 볼 때 F1 {fmt(base)}): " + "  ".join(
                f"{int(k * 100)}%: 정답 유지 {fmt(v['gt_kept'], 2)}, F1 {fmt(v['f1'])}" if v else f"{int(k * 100)}%: -"
                for k, v in zip(KEEP, line.values())))

    for name, scores in models.items():
        m = getattr(scores, "meta", None)
        if m:
            print(f"\n{name}: {m['model']}, 매개변수 {m['params_m']}M, {m['device']}에서 초당 {m['frames_per_second']}장")
    if selfcheck:
        ok = all(abs(rows["Jev"][c][k] - rows["Jev 확률을 넣은 점검"][c][k]) < 1e-12
                 for c in NEW + DEV if c[1] == "one" for k in ("auc_raw", "auc_smooth", "f1_same", "f1_best"))
        same_presence = abs(result["presence"]["Jev"]["auc"] - result["presence"]["Jev 확률을 넣은 점검"]["auc"]) < 1e-12
        print(f"\n점검: 한 장 조건의 값이 Jev와 같은가 → {'같음' if ok else '다름'}, 없는 영상 가리기 → {'같음' if same_presence else '다름'}")
        if not (ok and same_presence):
            raise SystemExit(1)
    else:
        (HERE / "results" / "siglip_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
