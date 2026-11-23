"""판정 모델의 답을 채점한다. README.md 2절(정답)과 4절(읽는 법). 표준 라이브러리만 쓴다.

  python3 da_eval.py        # results/microsoft.json, results/gemini.json → results/eval.json
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIRS = ("verify_runs", "verify_rr")
KEEP = 0.5
B = 2000
# 명확한 후보(README 2절): 1 = 남겨야 함, 0 = 빼야 함
CLEAR = {"verify_runs/web_1116_041133_3": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},      # 싸움 없는 영상에 "싸우는 장면"
         "verify_runs/web_1116_043326_2": {1: 0, 2: 0, 3: 0, 4: 0},            # 시도 없는 로봇 영상
         "verify_runs/web_1116_043326_1": {1: 1, 2: 0},                        # 59초에 컵 안 / 빈 팔
         "verify_runs/robot_Ra_green": {1: 1, 3: 0, 5: 0},                     # green_02 성공 / green_08 컵 옆에 떨굼 / green_05 접시에 다시 내려놓음
         "verify_runs/robot_Ra_red": {1: 1}}                                   # red_02 성공
CHECKS = {"싸움 없는 영상의 후보 5개를 모두 뺌": [("verify_runs/web_1116_041133_3", i, 0) for i in range(1, 6)],
          "시도 없는 로봇 영상의 후보 4개를 모두 뺌": [("verify_runs/web_1116_043326_2", i, 0) for i in range(1, 5)],
          "웹 실행의 59초가 든 후보를 남김": [("verify_runs/web_1116_043326_1", 1, 1)]}


def auc(s, y):
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def boot(rows, fn, seed=0):
    """영상을 단위로 다시 뽑은 95% 구간. fn(rows) → 값 또는 None."""
    vids = sorted({r["video"] for r in rows})
    by = {v: [r for r in rows if r["video"] == v] for v in vids}
    rng, vals = random.Random(seed), []
    for _ in range(B):
        v = fn([r for k in rng.choices(vids, k=len(vids)) for r in by[k]])
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)] if len(vals) > 100 else None


def r3(x):
    return None if x is None else round(x, 3)


def main():
    res = {m: {r["uid"]: r for r in json.loads((HERE / "results" / f"{m}.json").read_text())["records"]} for m in ("microsoft", "gemini")}
    meta = {m: json.loads((HERE / "results" / f"{m}.json").read_text())["meta"] for m in res}
    low_path = HERE / "results" / "gemini_low.json"     # README 11절: Gemini Decision을 그림 해상도 low(그림당 280토큰)로
    if low_path.exists():
        d = json.loads(low_path.read_text())
        res["gemini_low"], meta["gemini_low"] = {r["uid"]: r for r in d["records"]}, d["meta"]
    h2o_path = HERE / "results" / "h2o.json"            # README 10절: h2o8 = 8장 한 요청, h2o4 = 4장 요청 둘(연속, 고르게)의 평균
    if h2o_path.exists():
        h = json.loads(h2o_path.read_text())
        by = {}
        for r in h["records"]:
            if "error" not in r:
                by.setdefault(r["uid"], {})[r["arm"]] = r
        n_all = len(res["gemini"])
        h8 = {u: {"p": v["8"]["p"], "tokens": v["8"]["tokens"], "seconds": v["8"]["seconds"]} for u, v in by.items() if "8" in v}
        h4 = {u: {"p": {q: (v["a"]["p"][q] + v["b"]["p"][q]) / 2 for q in v["a"]["p"]}, "tokens": (v["a"]["tokens"] or 0) + (v["b"]["tokens"] or 0),
                  "seconds": v["a"]["seconds"] + v["b"]["seconds"]} for u, v in by.items() if "a" in v and "b" in v}
        for name, d in (("h2o8", h8), ("h2o4", h4)):
            if len(d) == n_all:
                res[name] = d
            else:
                print(f"{name}: 답이 {len(d)}/{n_all}개뿐이라 채점에서 뺀다")
        meta["h2o"] = h["meta"]
    rows = []
    for d in DIRS:
        for ip in sorted((ROOT / "pipeline" / d).glob("*/item.json")):
            item, verdict = json.loads(ip.read_text()), json.loads((ip.parent / "verdict.json").read_text())
            hp = {c["id"]: c for c in verdict["candidates"]}
            ev = [e["t"] for e in item["events"] if "t" in e]
            for c in item["candidates"]:
                uid = f"{d}/{item['name']}#{c['id']}"
                row = {"uid": uid, "group": item["group"], "request": item["request"], "video": Path(item["video"]).name, "start": c["start"], "end": c["end"],
                       "hydrogen": hp[c["id"]]["p"], "see": hp[c["id"]]["see"], "h2o_peak": c["peak"],
                       "event": (any(c["start"] <= t < c["end"] for t in ev) if item["group"].startswith("minecraft_") else None),
                       "clear": CLEAR.get(f"{d}/{item['name']}", {}).get(c["id"])}
                for m in res:
                    p = res[m][uid]["p"]
                    row[f"{m}.noul_plain"], row[f"{m}.noul_def"] = p["noul_plain"], p["noul_def"]
                    row[f"{m}.choice_fwd"], row[f"{m}.choice_rev"] = p["choice_fwd"], p["choice_rev"]
                    row[f"{m}.choice"] = None if None in (p["choice_fwd"], p["choice_rev"]) else (p["choice_fwd"] + p["choice_rev"]) / 2
                rows.append(row)
    arms = [f"{m}.{a}" for m in res for a in ("noul_plain", "noul_def", "choice")]
    scorers = ["h2o_peak", "hydrogen"] + arms
    out = {"n_candidates": len(rows), "keep": KEEP, "bootstrap": B, "unanswered": {a: sum(r[a] is None for r in rows) for a in arms}}

    # ---- 명확한 후보 15개
    clear = [r for r in rows if r["clear"] is not None]
    out["clear"] = {"n": len(clear), "keep": sum(r["clear"] for r in clear), "drop": sum(not r["clear"] for r in clear), "by_scorer": {}}
    for s in scorers:
        miss = [f"{r['uid']} ({'남겨야 함' if r['clear'] else '빼야 함'}, {r[s]:.2f})" for r in clear if (r[s] >= KEEP) != bool(r["clear"])]
        checks = {name: all((r[s] >= KEEP) == bool(want) for r in clear for (u, i, want) in items if r["uid"] == f"{u}#{i}") for name, items in CHECKS.items()}
        out["clear"]["by_scorer"][s] = {"correct": len(clear) - len(miss), "missed": miss, "checks": checks, "auc": r3(auc([r[s] for r in clear], [r["clear"] for r in clear]))}

    # ---- 마인크래프트: 게임 기록의 큰 사건 / Hydrogen의 답
    mc = [r for r in rows if r["event"] is not None]

    def table(rs, truth):
        t = {}
        for s in scorers:
            y, x = [truth(r) for r in rs], [r[s] for r in rs]
            t[s] = {"auc": r3(auc(x, y)), "pos_kept": sum(a and b >= KEEP for a, b in zip(y, x)), "pos_dropped": sum(a and b < KEEP for a, b in zip(y, x)),
                    "neg_kept": sum((not a) and b >= KEEP for a, b in zip(y, x)), "neg_dropped": sum((not a) and b < KEEP for a, b in zip(y, x))}
        return t

    def diff(rs, a, b, truth):
        f = lambda q: (lambda u, v: None if u is None or v is None else u - v)(auc([r[a] for r in q], [truth(r) for r in q]), auc([r[b] for r in q], [truth(r) for r in q]))
        return {"diff": r3(f(rs)), "ci": boot(rs, f)}

    ev_truth, hy_truth = (lambda r: r["event"]), (lambda r: r["hydrogen"] >= KEEP)
    out["minecraft_event"] = {"n": len(mc), "positives": sum(r["event"] for r in mc), "all": table(mc, ev_truth),
                              "by_group": {g: table([r for r in mc if r["group"] == g], ev_truth) for g in sorted({r["group"] for r in mc})}}
    out["hydrogen_truth"] = {"n": len(rows), "positives": sum(hy_truth(r) for r in rows), "all": table(rows, hy_truth),
                             "minecraft": {"n": len(mc), "positives": sum(hy_truth(r) for r in mc), **table(mc, hy_truth)}}
    out["choice_vs_noul_def"], out["vs_hydrogen_event_auc"] = {}, {}
    for m in res:
        out["choice_vs_noul_def"][m] = {"event_auc_minecraft": diff(mc, f"{m}.choice", f"{m}.noul_def", ev_truth),
                                        "hydrogen_auc_all": diff(rows, f"{m}.choice", f"{m}.noul_def", hy_truth),
                                        "noul_def_vs_plain_event": diff(mc, f"{m}.noul_def", f"{m}.noul_plain", ev_truth),
                                        "noul_def_vs_plain_hydrogen": diff(rows, f"{m}.noul_def", f"{m}.noul_plain", hy_truth)}
        for a in ("noul_plain", "noul_def", "choice"):
            out["vs_hydrogen_event_auc"][f"{m}.{a}"] = diff(mc, f"{m}.{a}", "hydrogen", ev_truth)
    out["model_pairs"] = {}                             # 같은 질문 꼴에서 모델끼리 (앞 − 뒤)
    for x, y in (("gemini", "microsoft"), ("h2o8", "gemini"), ("h2o8", "microsoft"), ("h2o4", "gemini"), ("h2o4", "h2o8"),
                 ("gemini_low", "gemini"), ("gemini_low", "h2o8"), ("gemini_low", "microsoft")):
        if x in res and y in res:
            out["model_pairs"][f"{x} - {y}"] = {a: {"event_auc_minecraft": diff(mc, f"{x}.{a}", f"{y}.{a}", ev_truth), "hydrogen_auc_all": diff(rows, f"{x}.{a}", f"{y}.{a}", hy_truth)}
                                                for a in ("noul_plain", "noul_def", "choice")}

    # ---- 순서, Hydrogen과의 순위 상관, 분포
    out["order"] = {}
    for m in res:
        d = [r[f"{m}.choice_fwd"] - r[f"{m}.choice_rev"] for r in rows]
        out["order"][m] = {"mean_abs_diff": round(sum(abs(x) for x in d) / len(d), 4), "mean_signed_fwd_minus_rev": round(sum(d) / len(d), 4), "max_abs_diff": round(max(abs(x) for x in d), 4),
                           "cross_keep": sum((r[f"{m}.choice_fwd"] >= KEEP) != (r[f"{m}.choice_rev"] >= KEEP) for r in rows), "identical": sum(x == 0 for x in d)}
    out["spearman_with_hydrogen"] = {s: r3(spearman([r[s] for r in rows], [r["hydrogen"] for r in rows])) for s in scorers if s != "hydrogen"}
    out["distribution"] = {s: {"min": round(min(r[s] for r in rows), 3), "max": round(max(r[s] for r in rows), 3), "kept": sum(r[s] >= KEEP for r in rows)} for s in scorers}
    out["usage"] = {m: {"requests": len(res[m]), "tokens_total": meta[m]["tokens_total"], "cost_usd": meta[m]["cost_usd"], "served_by": meta[m]["served_by"],
                        "tokens_median": sorted(r["tokens"] for r in res[m].values())[len(res[m]) // 2],
                        "seconds_median": sorted(r["seconds"] for r in res[m].values())[len(res[m]) // 2], "retries": sum(r["retries"] for r in res[m].values())} for m in ("microsoft", "gemini", "gemini_low") if m in meta}
    if "h2o" in meta:
        out["usage"]["h2o"] = {k: meta["h2o"].get(k) for k in ("requests", "answered", "errors", "skipped_arms", "partial", "ready_seconds", "ask_seconds", "total_seconds", "usd_estimate", "tokens_median", "gpu")}
        for name in ("h2o8", "h2o4"):
            if name in res:
                out["usage"]["h2o"][f"{name}_seconds_median"] = round(sorted(r["seconds"] for r in res[name].values())[len(res[name]) // 2], 3)
    out["robot_web"] = [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ["uid", "start", "end", "clear", "h2o_peak", "hydrogen"] + arms + ["see"]}
                        for r in rows if not r["group"].startswith("minecraft_")]
    (HERE / "results" / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    # ---- 화면에
    short = lambda s: s.replace("microsoft.", "MS ").replace("gemini_low.", "GL ").replace("gemini.", "GD ").replace("h2o8.", "H8 ").replace("h2o4.", "H4 ")
    print(f"후보 {len(rows)}개 · 답 못 받은 것 {out['unanswered']}")
    print(f"\n[명확한 후보 {len(clear)}개: 남겨야 함 {out['clear']['keep']}, 빼야 함 {out['clear']['drop']}]")
    for s in scorers:
        c = out["clear"]["by_scorer"][s]
        print(f"  {short(s):14s} 맞힘 {c['correct']:2d}/15 · 세 확인 {sum(c['checks'].values())}/3 · 틀린 것: {c['missed']}")
    for title, key, sub in (("마인크래프트 · 게임 기록의 큰 사건", "minecraft_event", "all"), ("전체 · Hydrogen의 답(p≥0.5)을 정답으로", "hydrogen_truth", "all")):
        t = out[key]
        print(f"\n[{title}: 후보 {t['n']}개, 양성 {t['positives']}개]")
        for s in scorers:
            a = t[sub][s]
            print(f"  {short(s):14s} AUC {a['auc']} · 양성 남김/뺌 {a['pos_kept']}/{a['pos_dropped']} · 음성 남김/뺌 {a['neg_kept']}/{a['neg_dropped']}")
    print("\n[마인크래프트 · 게임 기록 · 묶음별 AUC]")
    for g, t in out["minecraft_event"]["by_group"].items():
        print(f"  {g}: " + ", ".join(f"{short(s)} {t[s]['auc']}" for s in scorers))
    print("\n[choice − noul_def (AUC의 차이, 95% 구간)]")
    for m, d in out["choice_vs_noul_def"].items():
        for k, v in d.items():
            print(f"  {m:9s} {k:28s} {v['diff']} {v['ci']}")
    print("\n[게임 기록 AUC: 모델 − Hydrogen]")
    for a, v in out["vs_hydrogen_event_auc"].items():
        print(f"  {short(a):14s} {v['diff']} {v['ci']}")
    print("\n[모델끼리: 같은 질문 꼴의 AUC 차이 (게임 기록 / Hydrogen의 답)]")
    for pair, d in out["model_pairs"].items():
        for a, v in d.items():
            print(f"  {pair:20s} {a:11s} {v['event_auc_minecraft']['diff']} {v['event_auc_minecraft']['ci']} / {v['hydrogen_auc_all']['diff']} {v['hydrogen_auc_all']['ci']}")
    print("\n[순서]", out["order"])
    print("[Hydrogen의 p와의 순위 상관]", {short(k): v for k, v in out["spearman_with_hydrogen"].items()})
    print("[분포: 최소, 최대, 0.5 이상인 후보 수]", {short(k): v for k, v in out["distribution"].items()})
    print("[쓴 것]", out["usage"])
    print("\n[로봇·웹 실행의 후보]")
    for r in out["robot_web"]:
        print(f"  {r['uid'][12:]:28s} {r['start']:6.0f}~{r['end']:<6.0f} 정답 {'-' if r['clear'] is None else r['clear']} | H2O {r['h2o_peak']:.2f} Hyd {r['hydrogen']:.2f} | " +
              " ".join(f"{short(a)[:2]}{a.split('.')[1][:7]} {r[a]:.2f}" for a in arms) + f" | {r['see'][:40]}")


if __name__ == "__main__":
    main()
