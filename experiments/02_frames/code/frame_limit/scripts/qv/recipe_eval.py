"""적용 순서(규칙 묶음)를 단계별로 채점한다. 표준 라이브러리만 쓴다.

영상 묶음
- 규칙을 정하는 데 쓴 것: qv(QVHighlights 10개), loc(Charades 5개)
- 처음 보는 것: qv2(QVHighlights 다음 10개), sta(Charades 새 12개, 문장 32개)

단계 (위에서부터 하나씩 더한다)
0. 기준선     : 한 장, 요청 전체를 한 질문으로, 확률 0.5 이상이면 잡음
1. 임계값     : 기준마다 임계값 = max(FLOOR, R × 그 기준의 최고 확률). 최고 확률이 FLOOR보다 낮으면 "없음"
2. 여러 장    : 요청 종류가 motion이면 4장 조건의 확률을 쓴다
3. 분명한 조건: 요청 전체 질문과 분명한 조건 질문들의 최솟값을 쓴다
4. 끊김 잇기  : 잡힌 위치 사이에 한 칸만 비면 잇는다
5. 넓은 요청  : 위치의 절반 넘게 잡히면 "요청이 넓음"으로 표시하고 따로 센다

R은 규칙을 정하는 영상에서만 고른다(후보 0.5, 0.6, 0.7, 0.8). 고른 값은 아래 상수에 적어 고정한다.

실행: python3 experiments/frame_limit/scripts/qv/recipe_eval.py --dev-only   (R을 고를 때. 새 영상 결과는 읽지 않는다)
      python3 experiments/frame_limit/scripts/qv/recipe_eval.py              (전체)
결과: 표 출력, results/recipe_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import sys
from pathlib import Path

from qv_eval import auc, prf

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
FLOOR = 0.3
R_CANDIDATES = [0.5, 0.6, 0.7, 0.8]
R = 0.5  # 규칙을 정하는 영상에서 고른 값(후보 중 F1 평균이 가장 높음). 새 영상 결과를 보기 전에 고정했다
BROAD = 0.5
DEV, NEW = ["qv", "loc"], ["qv2", "sta"]
NAME = {"qv": "QVHighlights 10개", "loc": "Charades 5개", "qv2": "QVHighlights 새 10개", "sta": "Charades 새 12개"}
STAGES = ["기준선", "+임계값", "+여러 장", "+분명한 조건", "+끊김 잇기", "+넓은 요청 표시"]


def load(tag):
    """묶음 하나를 항목 목록으로 읽는다. 항목: id, query, windows, needs, ts, y, one, win4, conds."""
    data = HERE / "data"
    meta = json.loads((data / f"{tag}_items.json").read_text())
    preds = json.loads((HERE / "results" / f"{tag}_preds.json").read_text())
    whole, conds = {}, {}
    for r in preds["records"]:
        if r["error"]:
            continue
        key = (r["item"], r["frames"], r["t"])
        if "nouls" in r:
            whole[key] = r["nouls"]["whole"]
            conds[key] = [v for k, v in sorted(r["nouls"].items()) if k != "whole"]
        elif r["prompt"] == "plain":
            whole[key] = r["noul"]
    if tag in DEV:
        dev = json.loads((data / "recipe_dev.json").read_text())[tag]
        needs, clear = dev["needs"], dev.get("clear", {})
        if tag == "qv":  # 조건별 확률은 따로 돌린 실행에 있다
            for r in json.loads((HERE / "results" / "qv_parts_preds.json").read_text())["records"]:
                conds[(r["item"], r["frames"], r["t"])] = [r["nouls"][f"c{i}"] for i in clear[r["item"]]]
    else:
        needs = json.loads((data / f"{tag}_parts.json").read_text())["needs"]
    items = []
    for it in meta["items"]:
        wins = it.get("windows") or [it["moment"]]
        ts = sorted(t for (q, f, t) in whole if q == it["id"] and f == "one")
        series = {f: [whole[(it["id"], f, t)] for t in ts] if (it["id"], f, ts[0]) in whole else None
                  for f in ("one", "win4")}
        cvals = [conds.get((it["id"], "one", t), []) for t in ts]
        items.append({"id": it["id"], "query": it.get("query") or it.get("sentence"), "windows": wins,
                      "needs": needs[it["id"]], "ts": ts, "y": [any(s <= t <= e for s, e in wins) for t in ts],
                      "one": series["one"], "win4": series["win4"], "conds": [list(c) for c in zip(*cvals)]})
    tokens = {}
    for r in preds["records"]:
        if not r["error"]:
            tokens.setdefault(r["frames"], []).append(r["input_tokens"])
    return items, {f: sum(v) / len(v) for f, v in tokens.items()}


def series(item, routing, conditions):
    ps = item["win4"] if routing and item["needs"] == "motion" and item["win4"] else item["one"]
    if conditions and item["conds"]:
        ps = [min(v) for v in zip(ps, *item["conds"])]
    return ps


def catch(ps, r):
    """r이 None이면 0.5로 자르고, 아니면 그 기준의 최고 확률에 맞춘 임계값으로 자른다."""
    thr = 0.5 if r is None else max(FLOOR, r * max(ps))
    return [p >= thr for p in ps]


def fill_gaps(caught):
    out = list(caught)
    for i in range(1, len(out) - 1):
        if not caught[i] and caught[i - 1] and caught[i + 1]:
            out[i] = True
    return out


def segments(caught, ps):
    """잡힌 연속 위치의 묶음. (시작 번호, 끝 번호, 묶음 안 최고 확률)"""
    out, start = [], None
    for i, c in enumerate(caught + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            out.append((start, i - 1, max(ps[start:i])))
            start = None
    return sorted(out, key=lambda s: -s[2])


def run_stage(items, stage, r):
    """단계 번호까지 적용한 결과. 항목별 (확률, 잡힌 위치, 넓음 표시)"""
    out = []
    for it in items:
        ps = series(it, routing=stage >= 2, conditions=stage >= 3)
        caught = catch(ps, r if stage >= 1 else None)
        if stage >= 4:
            caught = fill_gaps(caught)
        broad = stage >= 5 and sum(caught) > BROAD * len(caught)
        out.append((ps, caught, broad))
    return out


def score(items, results):
    tp = fp = fn = top1 = top3_ok = top3_n = found = 0
    aucs, peaks, broad = [], 0, []
    for it, (ps, caught, is_broad) in zip(items, results):
        a = auc(ps, it["y"])
        if a is not None:
            aucs.append(a)
        peaks += it["y"][max(range(len(ps)), key=lambda k: ps[k])]
        if is_broad:
            broad.append(it["id"])
            continue
        tp += sum(c and y for c, y in zip(caught, it["y"]))
        fp += sum(c and not y for c, y in zip(caught, it["y"]))
        fn += sum(not c and y for c, y in zip(caught, it["y"]))
        segs = segments(caught, ps)
        ok = [any(it["y"][a:b + 1]) for a, b, _ in segs]
        found += bool(segs)
        top1 += bool(ok and ok[0])
        top3_ok += sum(ok[:3])
        top3_n += len(ok[:3])
    prec, rec, f1 = prf(tp, fp, fn)
    return {"n": len(items), "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "mean_auc": sum(aucs) / len(aucs), "peak_hits": peaks, "found": found, "top1": top1,
            "top3_ok": top3_ok, "top3_n": top3_n, "broad": broad}


def f(x):
    return "  - " if x is None else f"{x:.2f}"


def table(tag, items, r):
    print(f"\n{NAME[tag]} (항목 {len(items)}개)")
    print(f"{'단계':14s}{'맞게':>6s}{'엉뚱':>6s}{'놓침':>6s}{'정밀도':>7s}{'재현율':>7s}{'F1':>6s}"
          f"{'평균 AUC':>8s}{'1순위 구간이 정답':>12s}{'상위 3개 구간':>10s}{'넓음':>5s}")
    rows = {}
    for k, name in enumerate(STAGES):
        s = score(items, run_stage(items, k, r))
        rows[name] = s
        print(f"{name:14s}{s['tp']:>6d}{s['fp']:>6d}{s['fn']:>6d}{f(s['precision']):>7s}{f(s['recall']):>7s}"
              f"{f(s['f1']):>6s}{s['mean_auc']:>8.2f}{s['top1']:>9d}/{s['n'] - len(s['broad']):<3d}"
              f"{s['top3_ok']:>6d}/{s['top3_n']:<4d}{len(s['broad']):>4d}")
    return rows


def main():
    dev_only = "--dev-only" in sys.argv
    sets = {tag: load(tag) for tag in (DEV if dev_only else DEV + NEW)}
    print("R 고르기 (규칙을 정하는 영상, 끊김 잇기까지 적용한 F1)")
    choice = {}
    for r in R_CANDIDATES:
        f1s = [score(sets[t][0], run_stage(sets[t][0], 4, r))["f1"] for t in DEV]
        choice[r] = sum(f1s) / len(f1s)
        print(f"  R={r}: " + ", ".join(f"{NAME[t]} F1 {x:.3f}" for t, x in zip(DEV, f1s)) + f"  평균 {choice[r]:.3f}")
    best = max(R_CANDIDATES, key=lambda r: choice[r])
    print(f"  가장 좋은 값 {best}. 스크립트에 고정된 값 R={R}" + ("" if best == R else "  ← 다름. 상수를 확인할 것"))
    if dev_only:
        return
    result = {"R": R, "floor": FLOOR, "broad": BROAD, "r_selection": choice, "sets": {}}
    for tag, (items, tokens) in sets.items():
        rows = table(tag, items, R)
        result["sets"][tag] = {"stages": rows, "tokens_per_request": tokens}

    print("\n요청 종류를 나눈 것이 맞았나 (평균 AUC, 한 장 → 4장)")
    for tag in ("loc", "sta"):
        items = sets[tag][0]
        tokens = sets[tag][1]
        block = {}
        for kind in ("motion", "single"):
            sel = [it for it in items if it["needs"] == kind]
            pairs = [(auc(it["one"], it["y"]), auc(it["win4"], it["y"])) for it in sel]
            pairs = [p for p in pairs if None not in p]
            one, win = (sum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
            better = sum(p[1] > p[0] for p in pairs)
            block[kind] = {"n": len(pairs), "one": one, "win4": win, "win4_better": better}
            print(f"  {NAME[tag]} {kind:6s} {len(pairs):2d}개: {one:.2f} → {win:.2f}  (4장이 더 높은 항목 {better}개)")
        every = [(auc(it["one"], it["y"]), auc(it["win4"], it["y"]),
                  auc(it["win4"] if it["needs"] == "motion" else it["one"], it["y"])) for it in items]
        every = [e for e in every if None not in e]
        means = [sum(e[i] for e in every) / len(every) for i in range(3)]
        share = sum(it["needs"] == "motion" for it in items) / len(items)
        cost = {"one": tokens["one"], "win4": tokens["win4"],
                "routed": share * tokens["win4"] + (1 - share) * tokens["one"]}
        block["all"] = {"always_one": means[0], "always_win4": means[1], "routed": means[2], "tokens": cost}
        print(f"  {NAME[tag]} 전체: 항상 한 장 {means[0]:.2f} / 항상 4장 {means[1]:.2f} / 나눠서 {means[2]:.2f}"
              f"   요청당 토큰 {cost['one']:,.0f} / {cost['win4']:,.0f} / {cost['routed']:,.0f}")
        result["sets"][tag]["routing"] = block

    print("\n세 갈래로 나누면 (최종 확률, 0.7 이상 = 확실, 0.3~0.7 = 확인 필요, 0.3 미만 = 버림)")
    for tag, (items, _) in sets.items():
        sure, review, drop = [0, 0], [0, 0], [0, 0]
        for it in items:
            for p, y in zip(series(it, True, True), it["y"]):
                bucket = sure if p >= 0.7 else (review if p >= 0.3 else drop)
                bucket[0] += 1
                bucket[1] += y
        total_pos = sure[1] + review[1] + drop[1]
        result["sets"][tag]["buckets"] = {"sure": sure, "review": review, "drop": drop}
        print(f"  {NAME[tag]}: 확실 {sure[0]}곳 중 정답 {sure[1]} ({sure[1] / max(1, sure[0]):.0%}),"
              f" 확인 필요 {review[0]}곳 중 정답 {review[1]} ({review[1] / max(1, review[0]):.0%}),"
              f" 버림 {drop[0]}곳 중 정답 {drop[1]} (전체 정답 {total_pos}곳의 {drop[1] / total_pos:.0%})")

    print("\n넓은 요청으로 표시된 항목")
    for tag, (items, _) in sets.items():
        for it, (ps, caught, broad) in zip(items, run_stage(items, 5, R)):
            if broad:
                tp = sum(c and y for c, y in zip(caught, it["y"]))
                print(f"  [{NAME[tag]}] {it['id']} \"{it['query']}\"  잡힌 {sum(caught)}/{len(caught)}곳 중 정답 {tp}곳,"
                      f" 정답 구간은 {sum(it['y'])}곳")
    (HERE / "results" / "recipe_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
