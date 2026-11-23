"""Gemini Decision 1의 답을 정답·받아 둔 Jev 답과 견주고, 같은 묶음에서 잰 오픈 모델과 나란히 놓는다. 표준 라이브러리만 쓴다.

채점은 ../open_model/open_eval.py의 score_model을 그대로 쓴다(같은 항목, 같은 기준: AUC, 0.5 기준 정확도, Charades 엄격한 한 곳 찾기,
싸움 사건 찾음/헛짚음, 알림 찾음/헛짚음). 견줄 모델의 답은 ../open_model/results 에서 읽기만 한다.
- 한 곳 찾기의 모델 사이 차이: 같은 질문끼리 짝지어 빼고, 영상째 다시 뽑은 부트스트랩 2,000회의 95% 구간
- 토큰과 시간: 답의 usage에서 읽은 요청당 토큰(과제별 중앙값), 왕복 시간, 전체 처리 속도

실행: python3 experiments/gemini_decision/gemini_eval.py [--name 결과이름] [--results 폴더]
결과: results/gemini_eval.json (--name을 바꾸면 gemini_eval__<이름>.json)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPEN = HERE.parent / "open_model"
sys.path.insert(0, str(OPEN))
import open_eval as oe  # noqa: E402

OTHERS = ["h2o-4b", "h2o-4b__full_fp8_oneway", "d1-2b"]  # REPORT.md 4절의 H2O 기본, H2O FP8 + 한 방향 + 8개 동시, d1-2B


def records(name, folder):
    path = folder / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def top1(items, got):
    """Charades 질문마다 확률이 가장 높은 위치가 정답 구간 안인가(같으면 앞선 위치). open_eval과 같은 규칙."""
    best = {}
    for it in items:
        if it["task"] != "charades" or it["id"] not in got:
            continue
        for q in it["questions"]:
            p = got[it["id"]][q]
            if (it["video"], q) not in best or p > best[(it["video"], q)][0]:
                best[(it["video"], q)] = (p, it["gt"][q])
    return {k: hit for k, (_, hit) in best.items()}


def paired(a, b):
    by = {}
    for (video, q), hit in a.items():
        if (video, q) in b:
            by.setdefault(video, []).append(hit - b[(video, q)])
    rows = [d for ds in by.values() for d in ds]
    mean = lambda rs: sum(rs) / len(rs) if rs else None
    return {"questions": len(rows), "diff": oe.rnd(mean(rows)), "interval": oe.interval(by, mean) if rows else None}


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def usage(run, items):
    task_of = {it["id"]: it["task"] for it in items}
    recs = [r for r in run["records"] if not r["error"]]
    out = {}
    for task in dict.fromkeys(task_of.values()):
        mine = [r for r in recs if task_of.get(r["id"]) == task]
        if mine:
            secs = sorted(r["seconds"] for r in mine)
            out[task] = {"requests": len(mine), "tokens_median": median(r.get("tokens") for r in mine),
                         "image_tokens_median": median((r.get("by_modality") or {}).get("image") for r in mine),
                         "tokens_sum": sum(r.get("tokens") or 0 for r in mine),
                         "seconds_median": secs[len(secs) // 2], "seconds_p90": secs[int(0.9 * (len(secs) - 1))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gemini-decision-1")
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    folder = Path(args.results)
    items = json.loads((OPEN / "pack" / "manifest.json").read_text())["items"]
    run = records(args.name, folder)
    if run is None:
        raise SystemExit(f"{folder / (args.name + '.json')} 이 없다. gemini_run.py를 먼저 돌린다")

    scored = {args.name: oe.score_model(args.name, items, folder)}
    answers = {args.name: {r["id"]: r["probs"] for r in run["records"] if not r["error"]}}
    for other in OTHERS:
        o = records(other, OPEN / "results")
        if o is not None and other != args.name:
            scored[other] = oe.score_model(other, items)
            answers[other] = {r["id"]: r["probs"] for r in o["records"] if not r["error"]}

    mine = scored[args.name]
    print(f"{args.name}: 답 {mine['answered']}/{mine['items']}, 오류 {mine['errors']}")
    pick = lambda m, task, key, sub=None: ((m["tasks"].get(task) or {}).get(key) or {}).get(sub) if sub else (m["tasks"].get(task) or {}).get(key)
    top = lambda m, k="hits": (lambda t: f"{t[k]}/{t['questions']}" if t else "-")(pick(m, "charades", "top1"))
    rows = [("Charades 한 곳 찾기", lambda m: top(m), lambda m: top(m, "hits_jev")),
            ("Charades AUC", lambda m: pick(m, "charades", "all", "auc"), lambda m: pick(m, "charades", "all", "auc_jev")),
            ("QVHighlights AUC", lambda m: pick(m, "qv", "all", "auc"), lambda m: pick(m, "qv", "all", "auc_jev")),
            ("창 열림 AUC", lambda m: pick(m, "mc_gui", "gui", "auc"), lambda m: pick(m, "mc_gui", "gui", "auc_jev")),
            ("창 열림 0.5 정확도", lambda m: pick(m, "mc_gui", "gui", "accuracy"), lambda m: pick(m, "mc_gui", "gui", "accuracy_jev")),
            ("인벤토리만 AUC", lambda m: pick(m, "mc_gui", "inv", "auc"), lambda m: pick(m, "mc_gui", "inv", "auc_jev")),
            ("싸움 찾음", lambda m: pick(m, "fight", "found"), lambda m: pick(m, "fight", "found_jev")),
            ("싸움 헛짚음", lambda m: pick(m, "fight", "false_alarms"), lambda m: pick(m, "fight", "false_alarms_jev")),
            ("알림 찾음", lambda m: pick(m, "toast", "found"), lambda m: "기준"),
            ("알림 헛짚음", lambda m: pick(m, "toast", "false_alarms"), lambda m: "기준"),
            ("요청당 초: 한 장", lambda m: pick(m, "mc_gui", "seconds_median"), lambda m: "-"),
            ("요청당 초: 4장 묶음", lambda m: pick(m, "charades", "seconds_median"), lambda m: "-"),
            ("전체 시간(초)", lambda m: m["meta"].get("wall_seconds"), lambda m: "-")]
    names = list(scored)
    print(f"\n{'':22s}{'Jev(받아 둔 답)':>16s}" + "".join(f"{n[-24:]:>26s}" for n in names))
    for label, cell, jev in rows:
        print(f"{label:22s}{str(jev(mine)):>16s}" + "".join(f"{str(cell(scored[n])):>26s}" for n in names))
    t = pick(mine, "charades", "top1")
    if t:
        print(f"\n한 곳 찾기, Jev와의 차이: {t['diff']} {t['diff_interval']}")
    # 모델끼리 견줄 때는 이 모델이 답한 위치만 쓴다(답을 못 받은 위치가 있으면 다른 모델도 그 위치를 빼고 고른다)
    hits = {n: top1(items, {i: p for i, p in a.items() if i in answers[args.name]}) for n, a in answers.items()}
    pairs = {other: paired(hits[args.name], hits[other]) for other in names[1:]}
    missing = [it["id"] for it in items if it["id"] not in answers[args.name]]
    if missing:
        print(f"답을 못 받은 항목 {len(missing)}개: {missing[:8]}")
    for other, p in pairs.items():
        print(f"한 곳 찾기, {other}와의 차이: {p['diff']} {p['interval']} (질문 {p['questions']}개)")

    use = usage(run, items)
    meta = run["meta"]
    print(f"\n토큰과 시간 (해상도 '{(meta.get('options') or {}).get('resolution') or '기본'}', 동시 {(meta.get('options') or {}).get('workers')}개)")
    for task, u in use.items():
        print(f"  {task:9s} 요청 {u['requests']:5d}  요청당 토큰 {u['tokens_median']} (그림 {u['image_tokens_median']})  합 {u['tokens_sum']:,}  왕복 {u['seconds_median']}초 (p90 {u['seconds_p90']})")
    if meta.get("wall_seconds") and meta.get("items_run"):
        tokens = meta.get("tokens_total")
        print(f"  전체 {meta['items_run']}건 {meta['wall_seconds']}초 = 초당 {meta['items_run'] / meta['wall_seconds']:.1f}건, "
              f"토큰 {'-' if tokens is None else format(tokens, ',')}, ${meta.get('cost_usd')}, 다시 보냄 {meta.get('retries_total')}")

    out = {"_note": "gemini_eval.py의 결과. 채점은 open_model/open_eval.py와 같다. _jev가 붙은 값은 같은 항목에서 받아 둔 Jev(jev-1.14.0) 답",
           "models": scored, "top1_vs": pairs, "missing": missing, "usage": use, "run_meta": meta}
    stem = "gemini_eval" if args.name == "gemini-decision-1" else f"gemini_eval__{args.name}"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
