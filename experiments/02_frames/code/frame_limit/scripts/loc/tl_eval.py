"""TimeLens-Bench의 고친 질문·고친 정답으로 잰 결과를 채점한다. 표준 라이브러리만 쓴다.

자료
- data/tl_<묶음>_items.json: 고친 질문과 구간 (tl_prepare.py)
- results/tl_preds.json: Jev 한 장·4장 확률 (tl_run.py. 같은 영상의 질문을 한 요청에 묶어 보낸 것)
- results/tl_sheet60_preds.json: 시트 60칸 고르기·칸마다 (sheet_run.py --prefix tl_)
- 원래 질문으로 잰 것: results/<묶음>_preds.json, results/sheet60_preds.json, results/siglip_scores_*.json

보는 것 (실행 전에 정한 것. 채점 함수는 siglip_eval.py·sheet_eval.py와 같다)
1. 고친 질문 전체: 방식별 AUC, 5칸 평균 뒤 AUC, 최고점 적중, 같은 규칙으로 자른 F1, 방식 사이의 차이와 95% 구간
2. 시트를 1차로 쓰고 상위 30%만 원본 프레임으로 확인할 때 남는 정답 비율과 F1
3. 구간 길이별(2초 이하 / 3~5초 / 6초 이상) AUC
4. 문장이 글자까지 같은 질문: 같은 확률을 원래 정답과 고친 정답으로 각각 채점 (정답만 바꾼 효과). 무료 모델 SigLIP 2도 여기서만 잰다
5. 같은 문장을 다시 물었을 때 답이 지난 실행과 얼마나 다른가 (질문을 묶은 효과 + 다시 보낸 효과)

실행: python3 experiments/frame_limit/scripts/loc/tl_eval.py
결과: 표 출력, results/tl_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import math

import sheet_eval as S
import siglip_eval as E
from lev_probe_eval import boot
from qv_eval import auc, prf

HERE = E.HERE
SETS = [("Charades 12개 영상 (1초 간격)", "sta"), ("QVHighlights 본 10개 영상 (2초 간격)", "qv"),
        ("QVHighlights 새 10개 영상 (2초 간격)", "qv2")]
KEEP = 0.3
BINS = [("2초 이하", 0, 2), ("3~5초", 3, 5), ("6초 이상", 6, 10 ** 9)]


def labels(ts, windows):
    return [any(s <= t <= e for s, e in windows) for t in ts]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def first_pass(first, verify, ys, keep):
    """1차 점수(5칸 평균)의 상위 keep만 확인 쪽이 본다고 할 때. siglip_eval.first_pass와 같은 계산."""
    total, kept, every = [0, 0, 0], 0, 0
    for f, v, y in zip(first, verify, ys):
        sm = E.smooth(f)
        n = max(1, math.ceil(keep * len(sm)))
        chosen = set(sorted(range(len(sm)), key=lambda k: -sm[k])[:n])
        kept += sum(1 for k, t in enumerate(y) if t and k in chosen)
        every += sum(y)
        masked = [p if k in chosen else 0.0 for k, p in enumerate(v)]
        total = [a + c for a, c in zip(total, E.counts(E.cut_same(masked), y))]
    return {"gt_kept": round(kept / every, 3), "f1": round(prf(*total)[2] or 0.0, 3)}


def load(tag, index, sheet_recs, old_sheet_recs, free):
    """고친 항목마다 위치별 점수를 모은다. 문장이 같은 항목에는 지난 실행의 점수도 붙인다."""
    items = json.loads((HERE / "data" / f"tl_{tag}_items.json").read_text())["items"]
    frames = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
    table, tokens = {}, {}
    for r in json.loads((HERE / "results" / "tl_preds.json").read_text())["records"]:
        if r["tag"] != f"tl_{tag}" or r["error"]:
            continue
        tokens.setdefault(r["frames"], []).append(r["input_tokens"])
        for item_id, p in r["nouls"].items():
            table[(item_id, r["frames"], r["t"])] = p
    for it in items:
        it["ts"] = frames[it["video"]]["centers"]
        it["y"] = labels(it["ts"], it["windows"])
        it["length"] = it["windows"][0][1] - it["windows"][0][0]
        for cond in ("one", "win4"):
            vals = [table.get((it["id"], cond, t)) for t in it["ts"]]
            it[cond] = None if any(v is None for v in vals) else vals
        assert it["one"] is not None, f"{it['id']}: 한 장 결과가 빠졌다"
    series = S.sheet_series(items, tag, index, sheet_recs)
    for it in items:
        it["pick"], it["tiles"] = series[it["id"]]["pick"], series[it["id"]]["tiles"]
        it["sheet_tokens"] = series[it["id"]]["tokens"]
    # 문장이 같은 항목: 지난 실행의 확률, 지난 시트 결과, 무료 모델 점수
    old = {o["id"]: o for o in E.bundle(tag)}
    old_series = S.sheet_series(list(old.values()), tag, index, old_sheet_recs)
    for it in items:
        o = old.get(it["same_as"])
        if o is None:
            continue
        assert o["ts"] == it["ts"], f"{it['id']}: 위치가 다르다"
        it["old"] = {"y": o["y"], "one": o["one"], "win4": o["win4"],
                     "pick": old_series[o["id"]]["pick"], "tiles": old_series[o["id"]]["tiles"]}
        for label, sc in free.items():
            for cond in ("one", "win4"):
                it["old"][f"{label} {cond}"] = E.series(sc, o, cond)
    return items, {k: sum(v) / len(v) for k, v in tokens.items()}


def main():
    index = json.loads((HERE / "data" / "sheets60.json").read_text())
    new_sheets = json.loads((HERE / "results" / "tl_sheet60_preds.json").read_text())
    old_sheets = json.loads((HERE / "results" / "sheet60_preds.json").read_text())
    assert new_sheets["meta"]["errors"] == 0, "시트 요청에 오류가 남아 있다"
    key = lambda r: (r["tag"], r["item"], r["sheet"], r["kind"])
    sheet_recs = {key(r): r for r in new_sheets["records"]}
    old_sheet_recs = {key(r): r for r in old_sheets["records"]}
    free = {label: E.Scores.from_file(HERE / "results" / f) for label, f in E.MODEL_FILES.items()
            if (HERE / "results" / f).exists()}
    before = json.loads((HERE / "results" / "sheet60_eval.json").read_text())["sets"]
    run_meta = json.loads((HERE / "results" / "tl_preds.json").read_text())["meta"]
    print(f"Jev 프레임 요청 {run_meta['saved']}건(오류 {run_meta['errors']}건), 입력 {run_meta['input_tokens']:,}토큰, "
          f"${run_meta['cost_usd']} / 시트 요청 {new_sheets['meta']['requests']}건, {new_sheets['meta']['input_tokens']:,}토큰, "
          f"${new_sheets['meta']['cost_usd']}")
    out = {"meta": {"frames_run": run_meta, "sheets_run": new_sheets["meta"]}, "sets": {}}
    pooled_bins, pooled_rows, drift_all = {}, {}, {}

    for title, tag in SETS:
        items, tokens = load(tag, index, sheet_recs, old_sheet_recs, free)
        videos = [it["video"] for it in items]
        npos = sum(len(it["ts"]) for it in items)
        same = [it for it in items if "old" in it]
        defined = sum(auc(it["one"], it["y"]) is not None for it in items)
        print(f"\n===== {title}: 고친 질문 {len(items)}개(문장이 같은 것 {len(same)}개), 영상 {len(set(videos))}개 =====")
        if defined < len(items):
            print(f"(정답이 모든 위치를 덮어 AUC를 낼 수 없는 항목 {len(items) - defined}개는 AUC 평균에서 빠진다)")

        # 1. 고친 질문 전체
        methods = {"Jev 한 장": "one"}
        if all(it["win4"] for it in items):
            methods["Jev 4장"] = "win4"
        methods.update({"시트 60칸: 고르기": "pick", "시트 60칸: 칸마다": "tiles"})
        res, per_item = {}, {}
        print(f"{'방식':22s}{'AUC':>7s}{'5칸 평균 뒤':>9s}{'최고점':>8s}{'F1':>7s}{'정밀도':>7s}{'재현율':>7s}{'F1 상한':>8s}   지난 결과(원래 질문·원래 정답)의 AUC")
        for name, k in methods.items():
            pairs = [(it[k], it["y"]) for it in items]
            r = E.evaluate(pairs)
            per_item[name] = [auc(ps, y) for ps, y in pairs]
            res[name] = {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()}
            was = next((v["auc_raw"] for n, v in before[tag]["rows"].items() if n.startswith(name)), None)
            print(f"{name:22s}{r['auc_raw']:>7.3f}{r['auc_smooth']:>9.3f}{r['peak_hit']:>5d}/{r['items']}{fmt(r['f1_same']):>7s}"
                  f"{fmt(r['precision_same']):>7s}{fmt(r['recall_same']):>7s}{fmt(r['f1_best']):>8s}   {fmt(was)}")
        pairs_to_compare = [("시트 60칸: 칸마다", "Jev 한 장"), ("시트 60칸: 칸마다", "Jev 4장"), ("Jev 4장", "Jev 한 장"),
                            ("시트 60칸: 칸마다", "시트 60칸: 고르기"), ("시트 60칸: 고르기", "Jev 한 장")]
        diffs = {}
        print("차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
        for a, b in pairs_to_compare:
            if a in per_item and b in per_item:
                m, lo, hi, n, g = boot(per_item[a], per_item[b], videos)
                diffs[f"{a} − {b}"] = [m, lo, hi]
                print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        tok_sheet = {k: sum(it["sheet_tokens"][k] for it in items) / npos for k in ("pick", "tiles")}
        print(f"토큰: 프레임 요청 하나에 한 장 {tokens.get('one', 0):,.0f}"
              + (f", 4장 {tokens['win4']:,.0f}" if "win4" in tokens else "")
              + f" (영상의 질문 {len(items) / len(set(videos)):.1f}개가 한 요청을 나눠 씀). 시트는 질문·위치당 고르기 {tok_sheet['pick']:,.0f}, 칸마다 {tok_sheet['tiles']:,.0f}")

        # 2. 시트로 먼저 거르고 원본 프레임으로 확인
        verify = "win4" if "Jev 4장" in methods and tag == "sta" else "one"
        ys = [it["y"] for it in items]
        passes = {name: first_pass([it[k] for it in items], [it[verify] for it in items], ys, KEEP)
                  for name, k in (("시트 고르기", "pick"), ("시트 칸마다", "tiles"))}
        base_f1 = res["Jev 4장" if verify == "win4" else "Jev 한 장"]["f1_same"]
        print(f"시트 상위 {KEEP:.0%}만 Jev {'4장' if verify == 'win4' else '한 장'}으로 확인 (전부 물으면 F1 {fmt(base_f1)}): "
              + ", ".join(f"{n} 남는 정답 {p['gt_kept']:.3f}·F1 {p['f1']:.3f}" for n, p in passes.items()))

        # 3. 구간 길이별
        bins = {}
        print("구간 길이별 AUC 평균 (항목 수)")
        for label, lo, hi in BINS:
            group = [it for it in items if lo <= it["length"] <= hi]
            row = {name: mean([auc(it[k], it["y"]) for it in group]) for name, k in methods.items()}
            bins[label] = {"items": len(group), **{n: (None if v is None else round(v, 3)) for n, v in row.items()}}
            print(f"  {label:8s}({len(group):2d}개) " + ", ".join(f"{n} {fmt(v)}" for n, v in row.items()))
            for name, k in methods.items():
                pooled_bins.setdefault(label, {}).setdefault(name, []).extend(auc(it[k], it["y"]) for it in group)
            for it in group:
                pooled_rows.setdefault(label, []).append(
                    {"video": it["video"], **{k: (auc(it[k], it["y"]) if it[k] else None) for k in ("one", "win4", "pick", "tiles")}})

        # 4. 문장이 같은 질문: 같은 확률을 원래 정답과 고친 정답으로 채점
        same_res, apart = {}, []
        if same:
            vids = [it["video"] for it in same]
            print(f"문장이 같은 질문 {len(same)}개: 지난 실행의 확률을 두 정답으로 채점 (AUC. 정답 위치 수 원래 "
                  f"{sum(sum(it['old']['y']) for it in same)} → 고친 것 {sum(sum(it['y']) for it in same)})")
            rows = [("Jev 한 장", "one"), ("Jev 4장", "win4"), ("시트 고르기", "pick"), ("시트 칸마다", "tiles")]
            rows += [(f"{label} {'한 장' if c == 'one' else '4장 평균'} (무료)", f"{label} {c}") for label in free for c in ("one", "win4")]
            for name, k in rows:
                if any(it["old"].get(k) is None for it in same):
                    continue
                both = [(auc(it["old"][k], it["old"]["y"]), auc(it["old"][k], it["y"]), it["video"]) for it in same]
                both = [b for b in both if b[0] is not None and b[1] is not None]  # 두 정답 모두에서 AUC가 정해지는 항목만
                a_old, a_new = [b[0] for b in both], [b[1] for b in both]
                m, lo, hi, n, g = boot(a_new, a_old, [b[2] for b in both])
                same_res[name] = {"items": len(both), "orig_gt": round(mean(a_old), 3), "refined_gt": round(mean(a_new), 3),
                                  "diff": [m, lo, hi]}
                print(f"  {name:32s} 원래 정답 {mean(a_old):.3f} → 고친 정답 {mean(a_new):.3f}  ({m:+.3f} [{lo:+.3f}, {hi:+.3f}], {len(both)}개)")
            # 두 구간이 전혀 겹치지 않는 항목: 방식마다 어느 정답과 더 맞는지
            for it in same:
                if not any(a and b for a, b in zip(it["y"], it["old"]["y"])):
                    cells = ", ".join(f"{name.split(' (')[0]} {fmt(auc(it['old'][k], it['old']['y']), 2)}→{fmt(auc(it['old'][k], it['y']), 2)}"
                                      for name, k in rows if it["old"].get(k) is not None)
                    apart.append({"id": it["id"], "sentence": it["sentence"], "orig": it["orig_windows"], "refined": it["windows"]})
                    print(f"  겹치지 않는 항목 [{it['sentence'][:50]}] 원래 {it['orig_windows']} / 고친 것 {it['windows']}: {cells}")
            # 고친 정답에서 Jev와 무료 모델, 시트의 차이
            for a, b in (("tiles", "SigLIP 2 so400m one"), ("one", "SigLIP 2 so400m one"), ("win4", "SigLIP 2 so400m win4"),
                         ("tiles", "one"), ("tiles", "win4")):
                if any(it["old"].get(a) is None or it["old"].get(b) is None for it in same):
                    continue
                m, lo, hi, n, g = boot([auc(it["old"][a], it["y"]) for it in same], [auc(it["old"][b], it["y"]) for it in same], vids)
                same_res[f"refined_gt: {a} − {b}"] = [m, lo, hi]
                print(f"  고친 정답에서 {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        # 5. 같은 문장을 다시 물었을 때의 차이
        drift = {}
        for k in ("one", "win4", "tiles", "pick"):
            d = [abs(a - b) for it in same if it.get(k) and it["old"].get(k) for a, b in zip(it[k], it["old"][k])]
            if d:
                drift[k] = {"pairs": len(d), "mean_abs": round(sum(d) / len(d), 4), "over_0.1": round(sum(x > 0.1 for x in d) / len(d), 3)}
                drift_all.setdefault(k, []).extend(d)
        if drift:
            print("같은 문장을 다시 물었을 때 지난 실행과의 차이: "
                  + ", ".join(f"{k} 평균 {v['mean_abs']:.3f}(0.1 넘는 비율 {v['over_0.1']:.1%}, {v['pairs']}쌍)" for k, v in drift.items()))
        out["sets"][tag] = {"title": title, "items": len(items), "same_sentence": len(same), "rows": res, "diff": diffs,
                            "first_pass": passes, "by_length": bins, "same_sentence_rows": same_res, "no_overlap": apart, "drift": drift,
                            "tokens_per_frame_request": {k: round(v) for k, v in tokens.items()},
                            "sheet_tokens_per_query_position": {k: round(v) for k, v in tok_sheet.items()}}

    print("\n===== 세 묶음을 합친 구간 길이별 AUC 평균 =====")
    out["by_length_all"] = {}
    for label, _, _ in BINS:
        row = {n: mean(v) for n, v in pooled_bins.get(label, {}).items()}
        count = max((len([x for x in v if x is not None]) for v in pooled_bins.get(label, {}).values()), default=0)
        out["by_length_all"][label] = {"items": count, **{n: (None if v is None else round(v, 3)) for n, v in row.items()}}
        print(f"  {label:8s}({count:2d}개) " + ", ".join(f"{n} {fmt(v)}" for n, v in row.items()))
    print("구간 길이별 차이와 95% 구간 (세 묶음 합, 영상 단위로 다시 뽑음. 결과를 본 뒤 더한 계산)")
    out["by_length_diff"] = {}
    for label, _, _ in BINS:
        rows = pooled_rows.get(label, [])
        cells = {}
        for a, b in (("tiles", "pick"), ("tiles", "one"), ("win4", "tiles"), ("win4", "one")):
            use = [r for r in rows if r[a] is not None and r[b] is not None]
            if len({r["video"] for r in use}) >= 3:
                m, lo, hi, n, g = boot([r[a] for r in use], [r[b] for r in use], [r["video"] for r in use])
                cells[f"{a} − {b}"] = [m, lo, hi, n]
        out["by_length_diff"][label] = cells
        print(f"  {label:8s} " + ", ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] ({v[3]}개)" for k, v in cells.items()))
    out["drift_all"] = {k: {"pairs": len(d), "mean_abs": round(sum(d) / len(d), 4)} for k, d in drift_all.items()}
    print("같은 문장을 다시 물었을 때의 차이(세 묶음 합): "
          + ", ".join(f"{k} {v['mean_abs']:.3f}({v['pairs']}쌍)" for k, v in out["drift_all"].items()))
    (HERE / "results" / "tl_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
