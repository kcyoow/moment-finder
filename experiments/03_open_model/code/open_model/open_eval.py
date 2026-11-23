"""오픈 결정 모델의 답(results/<모델>.json)을 정답과 받아 둔 Jev 답에 견준다. 표준 라이브러리만 쓴다.

모델이 답한 항목만으로, 같은 항목에서 Jev와 나란히 잰다.
- AUC          : 정답이 1인 것을 0인 것보다 높게 매기는 비율(0.5 = 무작위, 1 = 완벽). 0.5에서 자른 정확도도 함께
- 차이의 구간  : 영상 단위로 다시 뽑은 부트스트랩 2,000회의 95% 구간(같은 영상의 프레임은 서로 비슷하므로 영상째 뽑는다)
- charades     : 한 곳 찾기. 질문마다 확률이 가장 높은 위치가 정답 구간 안이면 맞음(엄격한 기준)
- toast        : 정답이 "Jev가 켰고 눈으로 확인한 알림"이라 Jev는 정의상 전부 맞는다. 모델이 Jev를 얼마나 따라가는지만 본다
- fight        : 프레임 단위 정답이 없다. 사건(15개)마다 ±1초 안의 프레임 중 하나라도 0.5 이상이면 찾은 것, 견줄 프레임 300장 중 0.5 이상은 헛짚은 것
- Jev와의 일치 : 순위 상관(스피어만)과, 0.5를 기준으로 같은 쪽에 있는 비율
- 시간         : 요청 하나당 초(처음 5개는 데우는 것으로 보고 뺀 중앙값)

실행: python3 experiments/open_model/open_eval.py [모델 이름 …]   (없으면 results/의 전부. _tryN은 이름을 줘야 본다)
      python3 experiments/open_model/open_eval.py --accel         (가속 변형 results/<모델>__<변형>.json을 base 변형과 견준다)
결과: results/open_eval.json, results/open_eval_accel.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVENTS = HERE.parent / "frame_limit" / "data" / "mc_events.json"
CUT, BOOT, SEED = 0.5, 2000, 20261111


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def auc(pairs):
    """pairs: (확률, 정답 0/1). 한쪽이 없으면 None."""
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    r = ranks([p for p, _ in pairs])
    return (sum(ri for ri, (_, y) in zip(r, pairs) if y) - pos * (pos + 1) / 2) / (pos * neg)


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else None


def interval(by_video, stat):
    """영상째 다시 뽑아 stat(행 목록)의 95% 구간을 낸다."""
    rng, videos, vals = random.Random(SEED), sorted(by_video), []
    for _ in range(BOOT):
        rows = [r for v in rng.choices(videos, k=len(videos)) for r in by_video[v]]
        s = stat(rows)
        if s is not None:
            vals.append(s)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)] if vals else None


def rnd(x, n=3):
    return None if x is None else round(x, n)


def score_binary(rows):
    """rows: (영상, 모델 확률, Jev 확률, 정답). 정답이 있는 것만 온다."""
    by_video = {}
    for r in rows:
        by_video.setdefault(r[0], []).append(r)

    def diff(rs):
        a, b = auc([(m, y) for _, m, _, y in rs]), auc([(j, y) for _, _, j, y in rs])
        return None if a is None or b is None else a - b
    acc = lambda k: sum((r[k] >= CUT) == bool(r[3]) for r in rows) / len(rows)
    return {"n": len(rows), "positives": sum(r[3] for r in rows),
            "auc": rnd(auc([(m, y) for _, m, _, y in rows])), "auc_jev": rnd(auc([(j, y) for _, _, j, y in rows])),
            "auc_diff": rnd(diff(rows)), "auc_diff_interval": interval(by_video, diff),
            "accuracy": rnd(acc(1)), "accuracy_jev": rnd(acc(2))}


def score_model(name, items, results=None):
    """results: 답 파일이 있는 폴더(없으면 이 폴더의 results/). 다른 실험 폴더의 답을 같은 식으로 채점할 때 쓴다."""
    run = json.loads(((results or HERE / "results") / f"{name}.json").read_text())
    recs = [r for r in run["records"] if not r["error"]]
    got = {r["id"]: r["probs"] for r in recs}
    out = {"meta": {k: run["meta"].get(k) for k in ("repo", "gpu", "gpu_used_mb", "torch_max_memory_mb", "load_seconds", "wall_seconds", "items_run", "options")},
           "answered": len(got), "items": len(items), "errors": len(run["meta"].get("errors") or []), "tasks": {}}
    secs = sorted(r["seconds"] for r in recs[5:]) or [None]
    out["seconds"] = {"median": secs[len(secs) // 2], "p90": secs[int(0.9 * (len(secs) - 1))]}
    task_of = {it["id"]: it["task"] for it in items}
    for task in sorted({it["task"] for it in items}):
        mine = [it for it in items if it["task"] == task and it["id"] in got]
        if not mine:
            continue
        t = sorted(r["seconds"] for r in recs[5:] if task_of[r["id"]] == task)
        pairs = [(got[it["id"]][q], it["jev"][q]) for it in mine for q in it["questions"]]
        res = {"answered": len(mine), "seconds_median": t[len(t) // 2] if t else None,
               "jev_spearman": rnd(spearman([a for a, _ in pairs], [b for _, b in pairs])),
               "jev_same_side": rnd(sum((a >= CUT) == (b >= CUT) for a, b in pairs) / len(pairs))}
        if task in ("mc_gui", "toast"):
            for q in (["gui", "inv"] if task == "mc_gui" else ["advancement"]):
                res[q] = score_binary([(it["video"], got[it["id"]][q], it["jev"][q], it["gt"][q]) for it in mine])
            if task == "toast":
                pos = [got[it["id"]]["advancement"] for it in mine if it["gt"]["advancement"]]
                neg = [got[it["id"]]["advancement"] for it in mine if not it["gt"]["advancement"]]
                res["found"], res["false_alarms"] = f"{sum(p >= CUT for p in pos)}/{len(pos)}", f"{sum(p >= CUT for p in neg)}/{len(neg)}"
        elif task == "qv":
            res["all"] = score_binary([(it["video"], got[it["id"]][q], it["jev"][q], it["gt"][q]) for it in mine for q in it["questions"]])
        elif task == "charades":
            res["all"] = score_binary([(it["video"], got[it["id"]][q], it["jev"][q], it["gt"][q]) for it in mine for q in it["questions"]])
            best = {}  # (영상, 질문) → 가장 높은 위치의 정답 여부, 모델과 Jev 각각
            for it in mine:
                for q in it["questions"]:
                    for k, p in ((0, got[it["id"]][q]), (1, it["jev"][q])):
                        cur = best.setdefault((it["video"], q), [(-1, 0), (-1, 0)])
                        if p > cur[k][0]:
                            cur[k] = (p, it["gt"][q])
            by_video = {}
            for (video, _), (m, j) in best.items():
                by_video.setdefault(video, []).append((m[1], j[1]))
            rows = [r for rs in by_video.values() for r in rs]
            res["top1"] = {"questions": len(rows), "hits": sum(m for m, _ in rows), "hits_jev": sum(j for _, j in rows),
                           "diff": rnd(sum(m - j for m, j in rows) / len(rows)),
                           "diff_interval": interval(by_video, lambda rs: sum(m - j for m, j in rs) / len(rs))}
        elif task == "fight":
            events = [e for e in json.loads(EVENTS.read_text())["events"] if e["kind"] == "fight"]
            ev = [it for it in mine if it["role"] == "event"]
            events = [e for e in events if e["video"] in {it["video"] for it in mine}]  # 일부 영상만 돌린 실행에서는 그 영상의 사건만 센다
            ctrl = [it for it in mine if it["role"] == "control"]
            for key, pick in (("", lambda it: got[it["id"]]["fight"]), ("_jev", lambda it: it["jev"]["fight"])):
                res["found" + key] = f"{sum(any(pick(it) >= CUT for it in ev if it['video'] == e['video'] and abs(it['t'] - e['t']) <= 1.01) for e in events)}/{len(events)}"
                res["false_alarms" + key] = f"{sum(pick(it) >= CUT for it in ctrl)}/{len(ctrl)}"
        out["tasks"][task] = res
    return out


def accel(items):
    """가속 변형을 같은 항목의 base 변형과 견준다. 항목 수가 적어(영상의 1/3) 한 곳 찾기는 거칠다."""
    names = sorted(p.stem for p in (HERE / "results").glob("*__*.json") if "_try" not in p.stem)
    answers = {n: {r["id"]: r["probs"] for r in json.loads((HERE / "results" / f"{n}.json").read_text())["records"] if not r["error"]} for n in names}
    names = [n for n in names if answers[n]]  # 답이 하나도 없는 실행(서버가 죽은 경우 등)은 뺀다
    out = {"_note": "open_eval.py --accel의 결과. vs_base는 같은 항목·같은 질문에서 base 변형의 확률과 견준 것", "variants": {}}
    print(f"{'변형':24s}{'답':>5s}{'요청당초':>9s}{'초당항목':>9s}{'GPU MB':>8s}{'창AUC':>7s}{'QV AUC':>8s}{'Ch AUC':>8s}{'Ch 한곳':>9s}{'싸움':>7s}{'알림':>7s}{'평균차':>7s}{'같은쪽':>7s}")
    for name in names:
        m = score_model(name, items)
        base = answers.get(name.split("__")[0] + "__base", {})
        pairs = [(p, base[i][q]) for i, probs in answers[name].items() if i in base for q, p in probs.items()]
        m["vs_base"] = {"pairs": len(pairs), "mean_abs_diff": rnd(sum(abs(a - b) for a, b in pairs) / len(pairs)) if pairs else None,
                        "same_side": rnd(sum((a >= CUT) == (b >= CUT) for a, b in pairs) / len(pairs)) if pairs else None}
        meta, t = m["meta"], m["tasks"]
        m["items_per_second"] = rnd(meta["items_run"] / meta["wall_seconds"], 1) if meta.get("wall_seconds") else None
        out["variants"][name] = m
        cell = lambda task, key, sub=None: (t.get(task, {}).get(key) or {}).get(sub) if sub else t.get(task, {}).get(key)
        top = t.get("charades", {}).get("top1")
        print(f"{name:24s}{m['answered']:>5d}{m['seconds']['median']:>9.3f}{str(m['items_per_second']):>9s}{str(meta.get('torch_max_memory_mb') or meta.get('gpu_used_mb')):>8s}"
              f"{str(cell('mc_gui', 'gui', 'auc')):>7s}{str(cell('qv', 'all', 'auc')):>8s}{str(cell('charades', 'all', 'auc')):>8s}"
              f"{(str(top['hits']) + '/' + str(top['questions'])) if top else '-':>9s}{str(cell('fight', 'found') or '-'):>7s}{str(cell('toast', 'found') or '-'):>7s}"
              f"{str(m['vs_base']['mean_abs_diff']):>7s}{str(m['vs_base']['same_side']):>7s}")
    (HERE / "results" / "open_eval_accel.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


def main():
    items = json.loads((HERE / "pack" / "manifest.json").read_text())["items"]
    if "--accel" in sys.argv:
        return accel(items)
    names = sys.argv[1:] or sorted(p.stem for p in (HERE / "results").glob("*.json") if "_try" not in p.stem and "__" not in p.stem and p.stem not in ("open_eval", "open_eval_accel", "flow_eval", "speed_probe"))
    out = {"_note": "open_eval.py의 결과. auc_jev 등 _jev가 붙은 값은 같은 항목에서 받아 둔 Jev(jev-1.14.0) 답으로 낸 것", "models": {}}
    for name in names:
        out["models"][name] = m = score_model(name, items)
        print(f"\n== {name}: 답 {m['answered']}/{m['items']}, 오류 {m['errors']}, 요청당 {m['seconds']['median']}초(p90 {m['seconds']['p90']}), "
              f"{m['meta']['gpu']} {m['meta']['gpu_used_mb']}MB")
        for task, r in m["tasks"].items():
            line = f"  {task:9s} Jev와 상관 {r['jev_spearman']}, 같은 쪽 {r['jev_same_side']}, {r['seconds_median']}초 |"
            for q in ("gui", "inv", "advancement", "all"):
                if q in r:
                    b = r[q]
                    line += f" [{q}] AUC {b['auc']} (Jev {b['auc_jev']}, 차이 {b['auc_diff']} {b['auc_diff_interval']}), 정확도 {b['accuracy']} (Jev {b['accuracy_jev']})"
            if "top1" in r:
                t = r["top1"]
                line += f" | 한 곳 찾기 {t['hits']}/{t['questions']} (Jev {t['hits_jev']}, 차이 {t['diff']} {t['diff_interval']})"
            if "found" in r:
                line += f" | 찾음 {r['found']}" + (f" (Jev {r['found_jev']})" if "found_jev" in r else "") + f", 헛짚음 {r['false_alarms']}" + (f" (Jev {r['false_alarms_jev']})" if "false_alarms_jev" in r else "")
            print(line)
    if not sys.argv[1:]:
        (HERE / "results" / "open_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
