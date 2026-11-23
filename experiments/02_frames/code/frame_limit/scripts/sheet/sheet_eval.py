"""60장 묶음(접촉 시트) 결과를 한 장·4장·무료 모델과 같은 프레임·같은 정답으로 견준다. 표준 라이브러리만 쓴다.

자료
- results/sheet<tiles>_preds.json, data/sheets<tiles>.json: sheet_run.py가 만든 것
- results/<묶음>_preds.json: Jev 한 장·4장 확률 (이미 있는 것)
- results/siglip_scores_*.json: 무료 모델 SigLIP 2 점수 (작업 10)

위치별 점수 (실행 전에 정한 것)
- 시트(고르기)   : 시트의 "있는가" 확률 × 그 칸의 Choice 확률. jevreel이 쓰는 두 답을 곱한 것
- 시트(칸마다)   : 그 칸의 예 확률 그대로
- 참고로 Choice 확률만 쓴 값도 함께 적는다 (시트가 한 장인 영상에서는 위와 순서가 같다)

보는 것
1. 항목별 AUC, 최고점이 정답 안인 항목 수, 같은 규칙으로 자른 F1 (siglip_eval.py의 함수 그대로)
2. 위치 하나에 든 입력 토큰
3. 한 장·4장·무료 모델과의 차이와 95% 구간 (영상 단위로 다시 뽑음)
4. 시트를 1차 거르기로 쓰고 남은 위치만 원본 프레임으로 물을 때: 남는 정답 비율과 최종 F1 (새 요청 없이 계산)
5. 시트의 "있는가" 확률이 정답이 든 시트와 아닌 시트를 가르는 정도, Choice가 앞쪽 칸으로 쏠리는지
6. 고르기와 칸마다를 직접 견줌: 5칸 평균 뒤 AUC·F1의 차이와 구간, 칸마다 확률에 0.5 기준을 썼을 때 (결과를 본 뒤 더한 계산)

실행: python3 experiments/frame_limit/scripts/sheet/sheet_eval.py [--tiles 60]
결과: 표 출력, results/sheet<tiles>_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import random
import sys

import siglip_eval as E
from lev_probe_eval import boot
from qv_eval import auc, prf

HERE = E.HERE
SETS = [("Charades 새 12개 영상 (동작 요청, 1초 간격)", "sta"), ("QVHighlights 본 10개 영상 (2초 간격)", "qv"),
        ("QVHighlights 새 10개 영상 (2초 간격)", "qv2")]
KEEPS = [0.2, 0.3, 0.5]


def tokens_per_request(tag):
    """이미 있는 한 장·4장 실행에서 요청 하나에 든 입력 토큰 평균."""
    path = HERE / "results" / f"{tag}_preds.json"
    out = {}
    if path.exists():
        for r in json.loads(path.read_text())["records"]:
            if not r.get("error") and r.get("input_tokens"):
                out.setdefault(r.get("frames"), []).append(r["input_tokens"])
    return {k: sum(v) / len(v) for k, v in out.items()}


def sheet_series(items, tag, index, recs):
    """항목마다 위치별 점수 세 가지와 든 토큰을 만든다."""
    by_video = {}
    for sh in index["sheets"]:
        if sh["tag"] == tag:
            by_video.setdefault(sh["video"], []).append(sh)
    out = {}
    for it in items:
        sheets = sorted(by_video[it["video"]], key=lambda s: s["sheet"])
        n = len(it["ts"])
        assert sum(len(s["tiles"]) for s in sheets) == n, f"{it['id']}: 위치 수가 다르다"
        pick, only, tiles = [None] * n, [None] * n, [None] * n
        tok = {"pick": 0, "tiles": 0}
        rows = []
        for sh in sheets:
            a, b = recs[(tag, it["id"], sh["sheet"], "pick")], recs[(tag, it["id"], sh["sheet"], "tiles")]
            tok["pick"] += a["input_tokens"]
            tok["tiles"] += b["input_tokens"]
            shown, probs = a["nouls"]["shown"], a["choice"]["probabilities"]
            for t in sh["tiles"]:
                assert abs(t["t"] - it["ts"][t["pos"]]) < 1e-6, f"{it['id']}: 시각이 다르다"
                pick[t["pos"]] = shown * probs[t["id"]]
                only[t["pos"]] = probs[t["id"]]
                tiles[t["pos"]] = b["nouls"][t["id"]]
            rows.append({"shown": shown, "none": probs["none"], "has_gt": any(it["y"][t["pos"]] for t in sh["tiles"]),
                         "probs": [probs[t["id"]] for t in sh["tiles"]], "y": [it["y"][t["pos"]] for t in sh["tiles"]]})
        assert None not in pick and None not in tiles
        out[it["id"]] = {"pick": pick, "only": only, "tiles": tiles, "tokens": tok, "sheets": rows}
    return out


def f1_boot(counts_a, counts_b, groups, rounds=5000, seed=20261111):
    """F1의 차이(a − b)와, 영상 단위로 다시 뽑은 95% 구간. F1은 뽑힌 항목들의 개수를 합쳐서 낸다."""
    rng = random.Random(seed)
    ids = sorted(set(groups))
    by = {g: [k for k, gg in enumerate(groups) if gg == g] for g in ids}

    def f1(cnts, ks):
        return prf(*[sum(cnts[k][j] for k in ks) for j in range(3)])[2] or 0.0

    every = range(len(groups))
    out = []
    for _ in range(rounds):
        ks = [k for g in (rng.choice(ids) for _ in ids) for k in by[g]]
        out.append(f1(counts_a, ks) - f1(counts_b, ks))
    out.sort()
    return [round(f1(counts_a, every) - f1(counts_b, every), 3), round(out[int(0.025 * rounds)], 3),
            round(out[int(0.975 * rounds)], 3)]


def as_scores(items, series, kind):
    """시트 점수를 (프레임 경로, 요청문) 표로 옮겨 siglip_eval의 first_pass에 넣을 수 있게 한다."""
    return E.Scores({(p, it["query"]): s for it in items for p, s in zip(it["path_one"], series[it["id"]][kind])})


def main():
    tiles = int(sys.argv[sys.argv.index("--tiles") + 1]) if "--tiles" in sys.argv else 60
    index = json.loads((HERE / "data" / f"sheets{tiles}.json").read_text())
    data = json.loads((HERE / "results" / f"sheet{tiles}_preds.json").read_text())
    assert data["meta"]["errors"] == 0, "오류가 남은 요청이 있다"
    recs = {(r["tag"], r["item"], r["sheet"], r["kind"]): r for r in data["records"]}
    free = {label: E.Scores.from_file(HERE / "results" / f) for label, f in E.MODEL_FILES.items()
            if (HERE / "results" / f).exists()}
    print(f"시트 실행: 요청 {data['meta']['requests']}건, 입력 {data['meta']['input_tokens']:,}토큰, ${data['meta']['cost_usd']}")
    out = {"meta": data["meta"], "sets": {}}
    all_sheets = []
    for title, tag in SETS:
        items = E.bundle(tag)
        ser = sheet_series(items, tag, index, recs)
        per_req = tokens_per_request(tag)
        npos = sum(len(it["ts"]) for it in items)
        rows = {"Jev 한 장": ([(it["one"], it["y"]) for it in items], per_req.get("one"))}
        if all(it.get("win4") for it in items):
            rows["Jev 4장"] = ([(it["win4"], it["y"]) for it in items], per_req.get("win4"))
        for label, sc in free.items():
            for cond, name in (("one", "한 장"), ("win4", "4장 평균")):
                s = [E.series(sc, it, cond) for it in items]
                if all(x is not None for x in s):
                    rows[f"{label} {name} (무료)"] = ([(x, it["y"]) for x, it in zip(s, items)], 0)
        for kind, name in (("pick", f"시트 {tiles}칸: 고르기 (jevreel 방식)"), ("only", "  참고: Choice 확률만"),
                           ("tiles", f"시트 {tiles}칸: 칸마다 예/아니오")):
            tok = sum(ser[it["id"]]["tokens"]["tiles" if kind == "tiles" else "pick"] for it in items) / npos
            rows[name] = ([(ser[it["id"]][kind], it["y"]) for it in items], tok)
        print(f"\n===== {title}: 요청문 {len(items)}개, 위치 {npos}개 =====")
        print(f"{'방식':38s}{'토큰/위치':>9s}{'AUC':>7s}{'5칸 평균 뒤':>9s}{'최고점':>8s}{'F1':>7s}{'F1 상한':>8s}")
        res, per_item = {}, {}
        for name, (pairs, tok) in rows.items():
            r = E.evaluate(pairs)
            per_item[name] = [auc(ps, y) for ps, y in pairs]
            res[name] = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                         "tokens_per_position": None if tok is None else round(tok)}
            t = "-" if tok is None else f"{tok:,.0f}"
            print(f"{name:38s}{t:>9s}{r['auc_raw']:>7.3f}{r['auc_smooth']:>9.3f}{r['peak_hit']:>5d}/{r['items']}"
                  f"{r['f1_same']:>7.3f}{r['f1_best']:>8.3f}")
        videos = [it["video"] for it in items]
        names = list(rows)
        sheet_names = [n for n in names if n.startswith("시트")]
        others = [n for n in names if n.startswith("Jev") or "so400m" in n]
        diffs = {}
        print("차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
        for a in sheet_names:
            for b in others:
                m, lo, hi, n, g = boot(per_item[a], per_item[b], videos)
                diffs[f"{a} − {b}"] = [m, lo, hi]
                print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        m, lo, hi, n, g = boot(per_item[sheet_names[1]], per_item[sheet_names[0]], videos)
        diffs[f"{sheet_names[1]} − {sheet_names[0]}"] = [m, lo, hi]
        print(f"  {sheet_names[1]} − {sheet_names[0]}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        # 4. 시트를 1차 거르기로 쓰고, 남은 위치만 원본 프레임(한 장 또는 4장)으로 물을 때
        cond = "win4" if "Jev 4장" in rows and tag == "sta" else "one"
        full_tok = per_req.get(cond) or 0
        base_f1 = res["Jev 4장" if cond == "win4" else "Jev 한 장"]["f1_same"]
        print(f"1차로 거른 뒤 Jev {'4장' if cond == 'win4' else '한 장'}으로 확인 (전부 물으면 F1 {base_f1:.3f}, 위치당 {full_tok:,.0f}토큰)")
        print(f"  {'1차':28s}{'남기는 비율':>8s}{'남는 정답':>9s}{'F1':>7s}{'토큰/위치':>10s}")
        passes = {}
        firsts = [("SigLIP 2 base 한 장 (무료)", free.get("SigLIP 2 base"), 0.0),
                  (f"시트 {tiles}칸 고르기", as_scores(items, ser, "pick"), rows[sheet_names[0]][1]),
                  (f"시트 {tiles}칸 칸마다", as_scores(items, ser, "tiles"), rows[sheet_names[1]][1])]
        for label, sc, first_tok in firsts:
            if sc is None:
                continue
            for keep in KEEPS:
                r = E.first_pass(items, cond, sc, keep)
                if r is None:
                    continue
                total = first_tok + keep * full_tok
                passes[f"{label} {int(keep * 100)}%"] = {"gt_kept": round(r["gt_kept"], 3), "f1": round(r["f1"], 3),
                                                        "tokens_per_position": round(total)}
                print(f"  {label:28s}{keep:>9.0%}{r['gt_kept']:>10.3f}{r['f1']:>7.3f}{total:>10,.0f}")
        # 점검: 내용을 보지 않고 위치만으로 낸 점수, 모든 항목을 한데 모은 AUC(기준 하나로 가를 수 있는 정도)
        early = [a for a in (auc([-k for k in range(len(it["y"]))], it["y"]) for it in items) if a is not None]
        early = sum(early) / len(early)
        ys = [t for it in items for t in it["y"]]
        pooled = {name: round(auc([p for ps, _ in pairs for p in ps], ys), 3)
                  for name, (pairs, _) in rows.items() if name.startswith(("Jev", "시트"))}
        print(f"점검: 위치만으로 낸 AUC {max(early, 1 - early):.3f} (앞·뒤 중 나은 쪽). 한데 모은 AUC: "
              + ", ".join(f"{k} {v:.3f}" for k, v in pooled.items()))
        # 6. 고르기와 칸마다를 직접 견줌 (결과를 본 뒤 더한 계산)
        pt, pp = rows[sheet_names[1]][0], rows[sheet_names[0]][0]
        sm = list(boot([auc(E.smooth(ps), y) for ps, y in pt], [auc(E.smooth(ps), y) for ps, y in pp], videos)[:3])
        f1d = f1_boot([E.counts(E.cut_same(ps), y) for ps, y in pt], [E.counts(E.cut_same(ps), y) for ps, y in pp], videos)
        gt = [p for ps, y in pt for p, t in zip(ps, y) if t]
        ng = [p for ps, y in pt for p, t in zip(ps, y) if not t]
        half = prf(sum(p >= 0.5 for p in gt), sum(p >= 0.5 for p in ng), sum(p < 0.5 for p in gt))
        zero = sum(1 for ps, y in pp for p, t in zip(ps, y) if t and p <= 0.005) / len(gt)
        print(f"칸마다 − 고르기: 5칸 평균 뒤 AUC {sm[0]:+.3f} [{sm[1]:+.3f}, {sm[2]:+.3f}], "
              f"F1 {f1d[0]:+.3f} [{f1d[1]:+.3f}, {f1d[2]:+.3f}]")
        print(f"  고르기 점수가 사실상 0인 정답 위치 {zero:.0%}. 칸마다 확률 평균: 정답 위치 {sum(gt) / len(gt):.2f}, "
              f"아닌 위치 {sum(ng) / len(ng):.2f}. 0.5 기준으로 자르면 정밀도 {half[0]:.3f}, 재현율 {half[1]:.3f}, F1 {half[2]:.3f}")
        versus = {"auc_smooth_diff": sm, "f1_diff": f1d, "pick_zero_at_gt": round(zero, 3),
                  "tiles_mean_gt": round(sum(gt) / len(gt), 3), "tiles_mean_not_gt": round(sum(ng) / len(ng), 3),
                  "tiles_cut_0.5": [round(x, 3) for x in half]}
        for it in items:
            all_sheets += [dict(s, tag=tag) for s in ser[it["id"]]["sheets"]]
        out["sets"][tag] = {"title": title, "items": len(items), "positions": npos, "rows": res, "diff": diffs,
                            "first_pass": passes, "position_only_auc": round(max(early, 1 - early), 3), "pooled_auc": pooled,
                            "tiles_vs_pick": versus}

    # 5. 시트 단위의 답
    print("\n===== 시트 단위 =====")
    multi = [s for s in all_sheets if s["tag"] != "sta"]  # Charades는 영상마다 시트가 한 장이고 모두 정답이 들어 있다
    pos, neg = [s["shown"] for s in multi if s["has_gt"]], [s["shown"] for s in multi if not s["has_gt"]]
    a = auc(pos + neg, [True] * len(pos) + [False] * len(neg))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"QVHighlights 시트 {len(multi)}장: 정답이 든 시트 {len(pos)}장의 '있는가' 평균 {mean(pos):.2f}, "
          f"아닌 시트 {len(neg)}장 평균 {mean(neg):.2f}, 가르는 정도(AUC) {a if a is None else round(a, 3)}")
    small = sum(1 for s in multi if not s["has_gt"] and len(s["probs"]) < tiles)
    print(f"  주의: 정답이 없는 시트 {len(neg)}장 중 {small}장이 칸이 덜 찬 마지막 시트다. 시트 크기와 섞여 있어 이 값은 참고만 한다")
    sta = [s["shown"] for s in all_sheets if s["tag"] == "sta"]
    print(f"Charades 시트 {len(sta)}건(모두 정답이 듦): '있는가' 평균 {mean(sta):.2f}, 0.7 이상 {sum(x >= 0.7 for x in sta)}건, "
          f"0.3 미만 {sum(x < 0.3 for x in sta)}건")
    full = [s for s in all_sheets if len(s["probs"]) >= 20 and s["has_gt"]]
    third = lambda xs: sum(xs[: len(xs) // 3])
    mass = sum(third(s["probs"]) for s in full) / max(1e-9, sum(sum(s["probs"]) for s in full))
    gt = sum(third(s["y"]) for s in full) / sum(sum(s["y"]) for s in full)
    print(f"앞쪽 3분의 1 칸: Choice 확률의 {mass:.0%}, 정답 위치의 {gt:.0%} (칸 20개 이상이고 정답이 든 시트 {len(full)}건)")
    out["sheets"] = {"qv_auc_shown": a, "qv_with_gt": len(pos), "qv_without_gt": len(neg), "qv_mean_shown_gt": round(mean(pos), 3),
                     "qv_mean_shown_no_gt": round(mean(neg), 3), "sta_mean_shown": round(mean(sta), 3),
                     "sta_shown_ge_0.7": sum(x >= 0.7 for x in sta), "sta_shown_lt_0.3": sum(x < 0.3 for x in sta),
                     "first_third_choice_mass": round(mass, 3), "first_third_gt_share": round(gt, 3)}
    (HERE / "results" / f"sheet{tiles}_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
