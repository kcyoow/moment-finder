"""같은 프레임 4장을 "따로 4장"으로 넣은 답과 "2×2 한 그림"으로 넣은 답을 견준다. 표준 라이브러리만 쓴다.

채점은 open_model/open_eval.py의 score_model 그대로다(Charades 30개 영상, 위치 959곳, 질문 100개, 엄격한 한 곳 찾기와 AUC).
두 방식을 같은 서버에서 받았으므로 차이는 넣는 모양에서만 온다.
- 한 곳 찾기 차이: 같은 질문끼리 짝지어 빼고 영상째 다시 뽑은 부트스트랩 2,000회의 95% 구간
- AUC 차이: 같은 위치·질문의 확률 쌍으로, 같은 방식의 구간
- 두 방식의 확률이 얼마나 같은가: 순위 상관, 평균 차이, 0.5 기준 같은 쪽 비율
- 그림 모양별(가로 768×432, 세로, 4:3)로도 나눠 본다

실행: python3 experiments/grid4/grid4_eval.py [--name h2o-rtx]
결과: results/grid4_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN = HERE.parent / "open_model"
sys.path.insert(0, str(OPEN))
import open_eval as oe  # noqa: E402

REFERENCE = "h2o-4b__full_fp8_oneway"  # Colab L4에서 같은 설정(FP8, 한 방향)으로 따로 4장을 넣었던 답


def load(path):
    return {r["id"]: r for r in json.loads(path.read_text())["records"] if not r["error"]} if path.exists() else None


def top1(items, got):
    best = {}
    for it in items:
        if it["id"] not in got:
            continue
        for q in it["questions"]:
            p = got[it["id"]]["probs"][q]
            if (it["video"], q) not in best or p > best[(it["video"], q)][0]:
                best[(it["video"], q)] = (p, it["gt"][q])
    return {k: hit for k, (_, hit) in best.items()}


def paired_top1(a, b):
    by = {}
    for (video, q), hit in a.items():
        if (video, q) in b:
            by.setdefault(video, []).append(hit - b[(video, q)])
    rows = [d for ds in by.values() for d in ds]
    mean = lambda rs: sum(rs) / len(rs) if rs else None
    return {"questions": len(rows), "a_hits": sum(a.values()), "b_hits": sum(b.values()), "diff": oe.rnd(mean(rows)), "interval": oe.interval(by, mean)}


def auc_pair(items, a, b):
    by = {}
    for it in items:
        if it["id"] in a and it["id"] in b:
            for q in it["questions"]:
                by.setdefault(it["video"], []).append((a[it["id"]]["probs"][q], b[it["id"]]["probs"][q], it["gt"][q]))

    def diff(rs):
        x, y = oe.auc([(p, t) for p, _, t in rs]), oe.auc([(p, t) for _, p, t in rs])
        return None if x is None or y is None else x - y
    rows = [r for rs in by.values() for r in rs]
    pa, pb = [r[0] for r in rows], [r[1] for r in rows]
    return {"pairs": len(rows), "auc_a": oe.rnd(oe.auc([(p, t) for p, _, t in rows])), "auc_b": oe.rnd(oe.auc([(p, t) for _, p, t in rows])),
            "diff": oe.rnd(diff(rows)), "interval": oe.interval(by, diff),
            "spearman": oe.rnd(oe.spearman(pa, pb)), "mean_abs_diff": oe.rnd(sum(abs(x - y) for x, y in zip(pa, pb)) / len(rows)),
            "same_side": oe.rnd(sum((x >= oe.CUT) == (y >= oe.CUT) for x, y in zip(pa, pb)) / len(rows)),
            "mean_a": oe.rnd(sum(pa) / len(pa)), "mean_b": oe.rnd(sum(pb) / len(pb))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="h2o-rtx")
    args = ap.parse_args()
    items = [it for it in json.loads((OPEN / "pack" / "manifest.json").read_text())["items"] if it["task"] == "charades"]
    shape = {it["id"]: tuple(it["frame_size"]) for it in json.loads((HERE / "pack" / "manifest.json").read_text())["items"]}
    sep, grid = load(HERE / "results" / f"{args.name}__sep.json"), load(HERE / "results" / f"{args.name}__grid.json")
    if not sep or not grid:
        raise SystemExit("results/에 __sep.json과 __grid.json이 둘 다 있어야 한다")
    both = [it for it in items if it["id"] in sep and it["id"] in grid]
    out = {"_note": "grid4_eval.py의 결과. a = 2×2 한 그림(grid), b = 따로 4장(sep). 같은 서버에서 받은 답", "items": len(both),
           "missing": {"sep": [it["id"] for it in items if it["id"] not in sep][:10], "grid": [it["id"] for it in items if it["id"] not in grid][:10]}}

    for label, name in (("sep", f"{args.name}__sep"), ("grid", f"{args.name}__grid")):
        m = oe.score_model(name, items, HERE / "results")
        c = m["tasks"]["charades"]
        out[label] = {"answered": m["answered"], "top1": c["top1"], "auc": c["all"], "seconds_median": c["seconds_median"],
                      "jev_spearman": c["jev_spearman"], "jev_same_side": c["jev_same_side"]}
        t = c["top1"]
        print(f"{label:5s} 답 {m['answered']}/{len(items)} | 한 곳 찾기 {t['hits']}/{t['questions']} (받아 둔 Jev {t['hits_jev']}, 차이 {t['diff']} {t['diff_interval']}) | "
              f"AUC {c['all']['auc']} (Jev {c['all']['auc_jev']}) | 0.5 정확도 {c['all']['accuracy']} | 왕복 {c['seconds_median']}초")

    sub = lambda got: {i: r for i, r in got.items() if i in sep and i in grid}
    out["top1_grid_vs_sep"] = paired_top1(top1(both, sub(grid)), top1(both, sub(sep)))
    out["auc_grid_vs_sep"] = auc_pair(both, grid, sep)
    p, a = out["top1_grid_vs_sep"], out["auc_grid_vs_sep"]
    print(f"\n한 곳 찾기: 2×2 {p['a_hits']} 대 따로 {p['b_hits']} (질문 {p['questions']}개), 차이 {p['diff']} {p['interval']}")
    print(f"AUC: 2×2 {a['auc_a']} 대 따로 {a['auc_b']}, 차이 {a['diff']} {a['interval']}")
    print(f"확률: 순위 상관 {a['spearman']}, 평균 차이 {a['mean_abs_diff']}, 0.5 기준 같은 쪽 {a['same_side']}, 평균 확률 2×2 {a['mean_a']} / 따로 {a['mean_b']}")

    out["by_shape"] = {}
    for sz in sorted(set(shape.values())):
        part = [it for it in both if shape[it["id"]] == sz]
        videos = len({it["video"] for it in part})
        if videos >= 2:
            r = auc_pair(part, grid, sep)
            t = paired_top1(top1(part, sub(grid)), top1(part, sub(sep)))
            out["by_shape"][f"{sz[0]}x{sz[1]}"] = {"videos": videos, "positions": len(part), "auc": r, "top1": t}
            print(f"  프레임 {sz[0]}×{sz[1]} (영상 {videos}개, 위치 {len(part)}곳): AUC {r['auc_a']} 대 {r['auc_b']} (차이 {r['diff']} {r['interval']}), "
                  f"한 곳 찾기 {t['a_hits']} 대 {t['b_hits']} / {t['questions']}")

    ref = load(OPEN / "results" / f"{REFERENCE}.json")
    if ref:
        r = auc_pair(both, sep, ref)
        t = paired_top1(top1(both, sub(sep)), top1(both, {i: x for i, x in ref.items() if i in sep and i in grid}))
        out["sep_vs_colab_l4"] = {"auc": r, "top1": t}
        print(f"\n확인(같은 설정, 다른 GPU): 따로 4장 Cloud Run {t['a_hits']} 대 Colab L4 {t['b_hits']}, 확률 평균 차이 {r['mean_abs_diff']}, 순위 상관 {r['spearman']}")

    for label, got in (("sep", sep), ("grid", grid)):
        recs = [got[it["id"]] for it in both]
        med = lambda k: sorted(x[k] for x in recs if x.get(k) is not None)[len(recs) // 2] if any(x.get(k) is not None for x in recs) else None
        out[label].update(tokens_median=med("tokens"), shim_ms_median=med("shim_ms"), tokens_sum=sum(x.get("tokens") or 0 for x in recs))
        print(f"{label:5s} 요청당 토큰 {out[label]['tokens_median']}, 서버 안 시간 {out[label]['shim_ms_median']}ms")
    (HERE / "results" / "grid4_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
