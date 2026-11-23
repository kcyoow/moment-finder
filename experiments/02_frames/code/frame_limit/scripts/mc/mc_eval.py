"""마인크래프트 기록에서 "창이 열려 있다"를 찾은 결과를 채점한다. 표준 라이브러리만 쓴다.

정답은 행동 기록의 값 그대로다(틱마다 isGuiOpen, isGuiInventory). 위치는 1초 간격, 영상마다 300개.

방식
- Jev 한 장            : 위치마다 프레임 한 장 (results/mc_preds.json)
- 시트 60칸 고르기·칸마다: 60장을 한 이미지에 (results/mc_sheet60_preds.json)
- 규칙 (무료)          : 가운데 절반에서 창 바탕색에 가까운 화소의 비율 (results/mc_rule.json). 어떤 창인지는 가리지 못한다
- 위치만               : 내용을 보지 않고 앞·뒤 순서로만 낸 점수

보는 것 (실행 전에 정한 것)
1. 영상×질문별 AUC, 5칸 평균 뒤 AUC, 최고점 적중, 같은 규칙으로 자른 F1, 확률 0.5로 자른 정밀도·재현율
2. 방식 사이의 차이와 95% 구간(영상 단위로 다시 뽑음)
3. 구간 단위: 창이 열린 구간마다 1초 간격 표본에 걸리는가, 걸렸다면 점수 상위 10%·20%·30% 위치 안에 하나라도 드는가. 길이별
4. 시트의 "있는가" 답이 정답이 든 시트와 아닌 시트를 가르는 정도(이번에는 시트가 모두 60칸으로 꽉 차 있다)

실행: python3 experiments/frame_limit/scripts/mc/mc_eval.py
결과: 표 출력, results/mc_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import math

import siglip_eval as E
from blend_eval import read, sheet_scores
from lev_probe_eval import boot
from qv_eval import auc, prf

HERE = E.HERE
KINDS = [("gui", "창이 열려 있다 (어떤 창이든)"), ("inv", "인벤토리 창이 열려 있다")]
BINS = [("1초 미만", 0, 1), ("1~3초", 1, 3), ("3~10초", 3, 10), ("10초 이상", 10, 10 ** 9)]
KEEPS = [0.1, 0.2, 0.3]
TICK = 0.05


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def main():
    items = read("data/mc_items.json")["items"]
    frames = read("data/mc_frames.json")
    run = read("results/mc_preds.json")
    assert run["meta"]["errors"] == 0, "한 장 요청에 오류가 남아 있다"
    table = {}
    for r in run["records"]:
        for item_id, p in r["nouls"].items():
            table[(item_id, r["t"])] = p
    rule = read("results/mc_rule.json")["scores"]
    for it in items:
        it["ts"] = frames[it["video"]]["centers"]
        it["y"] = [any(s <= t <= e for s, e in it["windows"]) for t in it["ts"]]
        assert sum(it["y"]) == it["positive_positions"], it["id"]
        it["one"] = [table[(it["id"], t)] for t in it["ts"]]
        it["rule"] = rule[it["video"]]
        it["order"] = [-k for k in range(len(it["ts"]))]
    sheets = sheet_scores(items, "mc", "data/sheets60.json", "results/mc_sheet60_preds.json")
    sheet_meta = read("results/mc_sheet60_preds.json")
    for it in items:
        assert sheets[it["id"]]["pos"] == list(range(len(it["ts"])))
        it["pick"], it["tiles"] = sheets[it["id"]]["pick"], sheets[it["id"]]["tiles"]
    npos = sum(len(frames[v]["centers"]) for v in frames)
    one_tokens = run["meta"]["input_tokens"] / len(run["records"])
    by_kind_tokens = {}
    for r in sheet_meta["records"]:
        by_kind_tokens[r["kind"]] = by_kind_tokens.get(r["kind"], 0) + r["input_tokens"]
    per_query = len(KINDS) * npos
    print(f"Jev 한 장 요청 {len(run['records'])}건, {run['meta']['input_tokens']:,}토큰, ${run['meta']['cost_usd']} / "
          f"시트 요청 {sheet_meta['meta']['requests']}건, {sheet_meta['meta']['input_tokens']:,}토큰, ${sheet_meta['meta']['cost_usd']}")
    print(f"위치 하나의 입력 토큰: 한 장 {one_tokens:,.0f}(두 질문이 함께 씀), 시트 고르기 {by_kind_tokens['pick'] / per_query:,.0f}, "
          f"칸마다 {by_kind_tokens['tiles'] / per_query:,.0f}(질문 하나당)")
    out = {"meta": {"frames_run": run["meta"], "sheets_run": sheet_meta["meta"], "positions": npos,
                    "tokens_per_position": {"one": round(one_tokens), "pick": round(by_kind_tokens["pick"] / per_query),
                                            "tiles": round(by_kind_tokens["tiles"] / per_query)}}, "kinds": {}}
    methods = [("Jev 한 장", "one"), ("시트 60칸: 고르기", "pick"), ("시트 60칸: 칸마다", "tiles"), ("규칙 (무료)", "rule"), ("위치만", "order")]

    for kind, title in KINDS:
        group = [it for it in items if it["kind"] == kind]
        present = [it for it in group if any(it["y"])]
        absent = [it for it in group if not any(it["y"])]
        videos = [it["video"] for it in present]
        share = sum(sum(it["y"]) for it in group) / sum(len(it["y"]) for it in group)
        print(f"\n===== {title}: 영상 {len(group)}개 중 정답이 있는 것 {len(present)}개, 정답 위치 {share:.1%} =====")
        print(f"{'방식':20s}{'AUC':>7s}{'5칸 평균 뒤':>9s}{'최고점':>8s}{'F1':>7s}{'정밀도':>7s}{'재현율':>7s}   0.5로 자르면 (정밀도 / 재현율 / F1)")
        res, per_item = {}, {}
        for name, k in methods:
            pairs = [(it[k], it["y"]) for it in present]
            if k == "order":  # 앞에서부터와 뒤에서부터 중 나은 쪽
                a = [auc(ps, y) for ps, y in pairs]
                best = max(sum(a) / len(a), 1 - sum(a) / len(a))
                res[name] = {"auc_raw": round(best, 3)}
                print(f"{name:20s}{best:>7.3f}")
                continue
            r = E.evaluate(pairs)
            per_item[name] = [auc(ps, y) for ps, y in pairs]
            row = {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()}
            cut = ""
            if k in ("one", "tiles"):  # 확률을 내는 방식만
                c = [0, 0, 0]
                for it in group:  # 정답이 없는 영상도 넣는다(잘못 잡으면 정밀도가 내려간다)
                    c = [x + y for x, y in zip(c, E.counts([p >= 0.5 for p in it[k]], it["y"]))]
                p_, r_, f_ = prf(*c)
                row["cut_0.5"] = [None if v is None else round(v, 3) for v in (p_, r_, f_)]
                cut = f"   {fmt(p_)} / {fmt(r_)} / {fmt(f_)}"
            res[name] = row
            print(f"{name:20s}{r['auc_raw']:>7.3f}{r['auc_smooth']:>9.3f}{r['peak_hit']:>5d}/{r['items']}{fmt(r['f1_same']):>7s}"
                  f"{fmt(r['precision_same']):>7s}{fmt(r['recall_same']):>7s}{cut}")
        diffs = {}
        print("차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
        for a, b in (("시트 60칸: 칸마다", "Jev 한 장"), ("시트 60칸: 고르기", "Jev 한 장"), ("시트 60칸: 칸마다", "시트 60칸: 고르기"),
                     ("Jev 한 장", "규칙 (무료)"), ("시트 60칸: 칸마다", "규칙 (무료)")):
            m, lo, hi, n, g = boot(per_item[a], per_item[b], videos)
            diffs[f"{a} − {b}"] = [m, lo, hi]
            print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for it in absent:
            print(f"정답이 없는 영상 {it['video'][:5]}의 최고 점수: Jev 한 장 {max(it['one']):.2f}, 시트 칸마다 {max(it['tiles']):.2f}, "
                  f"시트 고르기 {max(it['pick']):.2f}")

        # 3. 구간 단위
        events = []
        for it in present:
            for s, e in it["windows"]:
                inside = [k for k, t in enumerate(it["ts"]) if s <= t <= e]
                ev = {"video": it["video"], "length": round(e - s + TICK, 2), "inside": inside, "ticks": (round(s / TICK), round(e / TICK))}
                n = len(it["ts"])
                for name, k in methods[:4]:
                    ranked = sorted(range(n), key=lambda j: -it[k][j])
                    for keep in KEEPS:
                        chosen = set(ranked[:max(1, math.ceil(keep * n))])
                        ev[(name, keep)] = any(j in chosen for j in inside)
                    if k in ("one", "tiles"):
                        ev[(name, "0.5")] = any(it[k][j] >= 0.5 for j in inside)
                events.append(ev)
        print(f"구간 단위: 정답 구간 {len(events)}개")
        print(f"  {'길이':8s}{'구간':>5s}{'표본에 걸림':>8s}   걸린 구간 중 상위 20% 위치 안에 든 비율 (한 장 / 고르기 / 칸마다 / 규칙)   0.5 이상인 위치가 있는 비율 (한 장 / 칸마다)")
        ev_out = {}
        for label, lo, hi in BINS + [("전체", 0, 10 ** 9)]:
            evs = [e for e in events if lo <= e["length"] < hi]
            seen = [e for e in evs if e["inside"]]
            cell = {"events": len(evs), "sampled": len(seen)}
            if seen:
                for name, _ in methods[:4]:
                    for keep in KEEPS:
                        cell[f"{name}@{keep}"] = round(sum(e[(name, keep)] for e in seen) / len(seen), 3)
                for name in ("Jev 한 장", "시트 60칸: 칸마다"):
                    cell[f"{name}@p0.5"] = round(sum(e[(name, "0.5")] for e in seen) / len(seen), 3)
                top = " / ".join(f"{cell[f'{name}@0.2']:.2f}" for name, _ in methods[:4])
                thr = " / ".join(f"{cell[f'{name}@p0.5']:.2f}" for name in ("Jev 한 장", "시트 60칸: 칸마다"))
                print(f"  {label:8s}{len(evs):>5d}{len(seen):>8d}   {top}   {thr}")
            else:
                print(f"  {label:8s}{len(evs):>5d}{len(seen):>8d}")
            ev_out[label] = cell
        # 표본 간격을 줄이면 걸리는 구간이 얼마나 느는가 (정답만으로 계산. 요청 없음)
        rates = {}
        for label, period in (("1초 간격", 20), ("0.5초 간격", 10), ("0.25초 간격", 5)):
            hit = lambda ev: any((t - period // 2) % period == 0 for t in range(ev["ticks"][0], ev["ticks"][1] + 1))
            short = [ev for ev in events if ev["length"] < 1]
            rates[label] = {"all": round(sum(map(hit, events)) / len(events), 3),
                            "under_1s": round(sum(map(hit, short)) / len(short), 3) if short else None}
        print("  표본 간격별로 걸리는 구간의 비율(전체 / 1초 미만): "
              + ", ".join(f"{k} {v['all']:.2f} / {fmt(v['under_1s'], 2)}" for k, v in rates.items()))
        ev_out["sampling"] = rates

        if kind == "inv":  # 인벤토리 / 다른 창 / 창 없음으로 나눠 본 평균 확률
            gui = {it["video"]: it["y"] for it in items if it["kind"] == "gui"}
            split = {}
            for name, k in (("Jev 한 장", "one"), ("시트 60칸: 칸마다", "tiles")):
                cells = {"인벤토리": [], "다른 창": [], "창 없음": []}
                for it in group:
                    for p, y, g in zip(it[k], it["y"], gui[it["video"]]):
                        cells["인벤토리" if y else "다른 창" if g else "창 없음"].append(p)
                split[name] = {c: {"n": len(v), "mean": round(sum(v) / len(v), 3), "at_least_0.5": round(sum(x >= 0.5 for x in v) / len(v), 3)}
                               for c, v in cells.items()}
                print(f"{name}의 평균 확률: " + ", ".join(f"{c} {d['mean']:.2f}(0.5 이상 {d['at_least_0.5']:.0%}, {d['n']}칸)" for c, d in split[name].items()))
            res["by_screen"] = split
        # 4. 시트의 "있는가" 답
        recs = {(r["item"], r["sheet"]): r for r in sheet_meta["records"] if r["kind"] == "pick"}
        shown, has = [], []
        for it in group:
            for s in range(len(it["ts"]) // 60):
                shown.append(recs[(it["id"], s)]["nouls"]["shown"])
                has.append(any(it["y"][s * 60:(s + 1) * 60]))
        a_sheet = auc(shown, has)
        pos, neg = [p for p, h in zip(shown, has) if h], [p for p, h in zip(shown, has) if not h]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"시트의 '있는가' 답: 정답이 든 시트 {len(pos)}장 평균 {mean(pos):.2f}, 아닌 시트 {len(neg)}장 평균 {mean(neg):.2f}, "
              f"가르는 정도(AUC) {fmt(a_sheet)}. 정답이 든 시트 중 0.5 미만 {sum(p < 0.5 for p in pos)}장, 아닌 시트 중 0.5 이상 {sum(p >= 0.5 for p in neg)}장")
        out["kinds"][kind] = {"title": title, "videos": len(group), "with_gt": len(present), "gt_share": round(share, 4),
                              "rows": res, "diff": diffs, "events": ev_out,
                              "sheet_shown": {"auc": a_sheet, "with_gt": len(pos), "without_gt": len(neg),
                                              "mean_with": round(mean(pos), 3), "mean_without": round(mean(neg), 3),
                                              "with_below_0.5": sum(p < 0.5 for p in pos), "without_at_least_0.5": sum(p >= 0.5 for p in neg)}}
    (HERE / "results" / "mc_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
