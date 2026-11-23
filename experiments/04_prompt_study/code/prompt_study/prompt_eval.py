"""prompt_run.py가 받은 답을 변형별로 채점한다. 표준 라이브러리만 쓴다.

무엇을 재나 (dev와 hold 각각, 위치×질문을 모두 모아서)
- AUC: 정답 구간 안의 위치를 밖의 위치보다 높게 매기는 정도(기준 0.5와 무관). ../open_model/open_eval.py의 auc 그대로
- AP: 확률이 높은 순으로 볼 때의 평균 정밀도
- 0.5 기준: 잡은 수, 정밀도(잡은 것 중 정답 구간 안), 재현율(정답 구간 안의 위치 중 잡은 것)
- 같은 재현율에서의 정밀도: 기준 문장(base)이 0.5에서 낸 재현율만큼 잡도록 기준을 옮겼을 때의 정밀도.
  문장을 엄하게 쓰면 확률이 전체로 내려가 0.5 정밀도는 저절로 오른다. 그것이 기준만 올린 것과 다른지 보려는 값이다
- 한 곳 찾기: 질문마다 확률이 가장 높은 위치가 정답 구간 안인가(open_eval과 같은 규칙)
- 먼 헛짚음: 0.5 이상인데 정답 구간에서 2초 넘게 떨어진 위치의 수
- base와의 차이는 영상째 다시 뽑아(2,000번) 95% 구간을 낸다

고르는 규칙(결과를 보기 전에 정함): dev에서 AUC가 base보다 0.005 넘게 낮지 않고 한 곳 찾기가 2문항 넘게 낮지 않은 것 가운데
"같은 재현율에서의 정밀도"가 가장 높은 것. 고른 것은 hold의 값으로 확인한다. 변형 전부의 값을 함께 보고한다.

실행: python3 prompt_eval.py [--try N]  → results/prompt_eval.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "open_model"))
from open_eval import auc  # noqa: E402
from prompt_run import load_items  # noqa: E402
from variants import BASE, COMBOS, VARIANTS, kind  # noqa: E402

SEED, BOOT, CUT, FAR = 20261112, 2000, 0.5, 2.0


def ap(rows):
    """rows: (p, y). 확률이 같은 것은 한 덩어리로 본다."""
    pos = sum(y for _, y in rows)
    if not pos:
        return None
    rows = sorted(rows, key=lambda r: -r[0])
    total, tp, i = 0.0, 0, 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        hit = sum(y for _, y in rows[i:j + 1])
        tp += hit
        total += hit * tp / (j + 1)
        i = j + 1
    return total / pos


def at_cut(rows, cut=CUT):
    flagged = [y for p, y in rows if p >= cut]
    pos = sum(y for _, y in rows)
    tp = sum(flagged)
    return len(flagged), (tp / len(flagged) if flagged else None), (tp / pos if pos else None)


def precision_at_recall(rows, target):
    """재현율이 target 이상이 되는 가장 높은 기준에서의 (정밀도, 기준). 확률이 같은 것은 함께 넘는다."""
    pos = sum(y for _, y in rows)
    if not pos or target is None:
        return None, None
    rows = sorted(rows, key=lambda r: -r[0])
    tp, i = 0, 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        tp += sum(y for _, y in rows[i:j + 1])
        if tp / pos >= target - 1e-12:
            return tp / (j + 1), rows[i][0]
        i = j + 1
    return tp / len(rows), rows[-1][0]


def top1(recs):
    """recs: (질문 키, t, y, p) 목록 → (맞은 수, 질문 수). 가장 높은 확률이 여럿이면 앞선 위치."""
    best = {}
    for key, t, y, p in sorted(recs, key=lambda r: (r[0], r[1])):
        if key not in best or p > best[key][0]:
            best[key] = (p, y)
    return sum(y for _, y in best.values()), len(best)


def stats(recs, target=None):
    """recs: (질문 키, t, y, p, 정답 구간까지의 거리). target: 맞출 재현율(없으면 자기 것)."""
    rows = [(p, y) for _, _, y, p, _ in recs]
    n, prec, rec = at_cut(rows)
    hits, nq = top1([r[:4] for r in recs])
    p_at, cut = precision_at_recall(rows, rec if target is None else target)
    return {"auc": auc(rows), "ap": ap(rows), "flagged": n, "precision": prec, "recall": rec, "precision_at_base_recall": p_at, "cut_at_base_recall": cut,
            "top1": hits, "questions": nq, "far_false": sum(1 for _, _, y, p, d in recs if p >= CUT and not y and d is not None and d > FAR + 0.01),
            "mean_p": sum(p for p, _ in rows) / len(rows)}


def rnd(x, n=3):
    return None if x is None else round(x, n)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--try", dest="try_n", type=int, default=0, help="…_tryN.json 결과를 읽는다(형식 확인용)")
    ap_.add_argument("--name", default="h2o")
    ap_.add_argument("--boot", type=int, default=BOOT)
    args = ap_.parse_args()

    items = {it["id"]: it for it in load_items()}
    got = {}  # 변형 → {(항목 id, 질문 id): 확률}
    for layout in sorted({v["layout"] for v in VARIANTS.values()}):
        path = HERE / "results" / (f"{args.name}__{layout}_try{args.try_n}.json" if args.try_n else f"{args.name}__{layout}.json")
        if not path.exists():
            print("없음:", path.name)
            continue
        for r in json.loads(path.read_text())["records"]:
            if not r["error"]:
                for variant, probs in r["probs"].items():
                    for qid, p in probs.items():
                        got.setdefault(variant, {})[(r["id"], qid)] = p
    for name, combo in COMBOS.items():
        parts = [got[v] for v in combo["of"] if v in got]
        if len(parts) != len(combo["of"]):
            continue
        keys = set(parts[0]).intersection(*parts[1:])
        fn = {"mean": lambda xs: sum(xs) / len(xs), "min": min}[combo["how"]]
        got[name] = {k: fn([p[k] for p in parts]) for k in keys}
    if BASE not in got:
        raise SystemExit("기준 문장의 답이 없다")

    # 질문마다 정답 구간까지의 거리(초)
    pos_t = {}
    for it in items.values():
        for qid, y in it["gt"].items():
            if y:
                pos_t.setdefault((it["video"], qid), []).append(it["t"])
    text = {(it["video"], qid): sentence for it in items.values() for qid, sentence in it["questions"].items()}
    out = {"base": BASE, "boot": args.boot, "sets": {}}
    for which in ("dev", "hold"):
        mine = [it for it in items.values() if it["set"] == which]
        keys = [(it["id"], qid) for it in mine for qid in it["questions"] if (it["id"], qid) in got[BASE]]
        if not keys:
            continue

        def recs_of(variant, by_video=False):
            g = got[variant]
            rows = {} if by_video else []
            for iid, qid in keys:
                if (iid, qid) not in g:
                    continue
                it = items[iid]
                near = pos_t.get((it["video"], qid))
                r = ((it["video"], qid), it["t"], it["gt"][qid], g[(iid, qid)], min(abs(it["t"] - t) for t in near) if near else None)
                (rows.setdefault(it["video"], []) if by_video else rows).append(r)
            return rows

        base_recs = recs_of(BASE)
        base = stats(base_recs)
        base_by_video = recs_of(BASE, by_video=True)
        videos = sorted(base_by_video)
        res = {"videos": len(videos), "positions": len(mine), "pairs": len(keys), "positives": sum(r[2] for r in base_recs),
               "questions_with_answer": len({r[0] for r in base_recs if r[2]}), "variants": {},
               "base_at_other_cuts": {str(c): dict(zip(("flagged", "precision", "recall"), [rnd(x) for x in at_cut([(p, y) for _, _, y, p, _ in base_recs], c)]))
                                      for c in (0.6, 0.7, 0.8)}}
        for variant in [v for v in list(VARIANTS) + list(COMBOS) if v in got]:
            recs = recs_of(variant)
            if len(recs) != len(base_recs):
                print(f"{which}/{variant}: 답 {len(recs)}/{len(base_recs)}개뿐이라 건너뜀")
                continue
            s = stats(recs, target=base["recall"])
            row = {k: (rnd(v) if isinstance(v, float) else v) for k, v in s.items()}
            if variant != BASE and args.boot:
                by_video = recs_of(variant, by_video=True)
                rng, diffs = random.Random(SEED), {"auc": [], "top1": [], "precision": [], "precision_at_base_recall": []}
                for _ in range(args.boot):
                    pick = rng.choices(videos, k=len(videos))
                    b = stats([r for v in pick for r in base_by_video[v]])
                    m = stats([r for v in pick for r in by_video[v]], target=b["recall"])
                    for k in diffs:
                        x, y = (m[k], b[k]) if k != "top1" else (m[k] / m["questions"], b[k] / b["questions"])
                        if x is not None and y is not None:
                            diffs[k].append(x - y)
                for k, vals in diffs.items():
                    vals.sort()
                    point = (s[k] - base[k]) if k != "top1" else (s[k] - base[k]) / base["questions"]
                    row[f"{k}_diff"] = [rnd(point), rnd(vals[int(0.025 * len(vals))]), rnd(vals[int(0.975 * len(vals)) - 1])] if vals else None
            row["groups"] = {}
            for g in ("change", "other"):
                sub = [r for r in recs if kind(text[r[0]]) == g]
                bsub = [r for r in base_recs if kind(text[r[0]]) == g]
                if sub and sum(r[2] for r in sub):
                    gs = stats(sub, target=at_cut([(p, y) for _, _, y, p, _ in bsub])[2])
                    row["groups"][g] = {k: (rnd(gs[k]) if isinstance(gs[k], float) else gs[k])
                                        for k in ("auc", "flagged", "precision", "recall", "precision_at_base_recall", "top1", "questions", "far_false")}
            res["variants"][variant] = row
        out["sets"][which] = res

    # 고르기 (dev만 본다)
    dev = out["sets"].get("dev", {}).get("variants", {})
    if dev:
        b = dev[BASE]
        ok = [v for v, r in dev.items() if v != BASE and r["auc"] >= b["auc"] - 0.005 and r["top1"] >= b["top1"] - 2]
        out["pick"] = max(ok, key=lambda v: dev[v]["precision_at_base_recall"]) if ok else None
        out["pick_rule"] = "dev에서 AUC ≥ base − 0.005, 한 곳 찾기 ≥ base − 2인 것 중 같은 재현율에서의 정밀도가 가장 높은 것"
    suffix = f"_try{args.try_n}" if args.try_n else ""
    (HERE / "results" / f"prompt_eval{suffix}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    for which, res in out["sets"].items():
        print(f"\n[{which}] 영상 {res['videos']}, 위치 {res['positions']}, 위치×질문 {res['pairs']}, 정답 안 {res['positives']}, 다른 기준의 base: {res['base_at_other_cuts']}")
        print(f"{'변형':22s}{'AUC':>7s}{'AP':>7s}{'잡음':>6s}{'정밀':>7s}{'재현':>7s}{'같은재현 정밀':>10s}{'한곳':>8s}{'먼헛짚':>6s}{'평균p':>7s}   같은재현 정밀 차이[95%]")
        for v, r in res["variants"].items():
            d = r.get("precision_at_base_recall_diff")
            print(f"{v:22s}{r['auc']:7.3f}{r['ap']:7.3f}{r['flagged']:6d}{(r['precision'] or 0):7.3f}{(r['recall'] or 0):7.3f}{(r['precision_at_base_recall'] or 0):10.3f}"
                  f"{r['top1']:5d}/{r['questions']:<3d}{r['far_false']:5d}{r['mean_p']:7.3f}   {d if d else ''}")
        for g in ("change", "other"):
            print(f"  [{g}] " + "  ".join(f"{v}: {r['groups'][g]['precision']}/{r['groups'][g]['recall']}/{r['groups'][g]['precision_at_base_recall']}"
                                          for v, r in res["variants"].items() if g in r.get("groups", {})) + "   (정밀/재현/같은재현 정밀)")
    print("\n고른 것:", out.get("pick"))


if __name__ == "__main__":
    main()
