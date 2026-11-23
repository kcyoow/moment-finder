"""판정 모델이 고른 후보 클립을 Gemini가 프레임으로 다시 확인한다. pipeline/README.md 18절.

흐름: 합친 점수 → 후보 클립 N개(순위대로) → 후보마다 프레임 8장(가장 높았던 1초의 연속 4장 + 구간에서 고르게 4장)
      → Gemini 4 Hydrogen에 요청 한 줄과 함께 한 번에 → 후보마다 p(찾는 장면이 담겨 있을 확률) → p ≥ 0.5만 남긴다
Gemini가 받는 것: verifier_brief.md, 사용자의 요청, 영상 길이, 후보의 시각과 프레임. 판정 모델의 질문과 답은 주지 않는다.

  python mf_verify.py prepare          # 후보와 프레임을 만든다 (돈 없음). verify_runs/<이름>/
  python mf_verify.py ask [--only 이름] [--workers 4] [--budget 8]     # Gemini를 부른다 (돈이 든다)
  python mf_verify.py eval             # 정답과 견준다 → eval/verify_check.json

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_verify.py <명령>
"""
import argparse
import base64
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import mf  # noqa: E402
import mf_core as C  # noqa: E402
import mf_writer as W  # noqa: E402

OUT = HERE / "verify_runs"
N_CAND, KEEP_P = 5, 0.5
N_RR = 6                    # 20절: 목록 여럿에서 돌아가며 뽑는 후보 수
PRICE_IN, PRICE_OUT = W.PRICE_IN, W.PRICE_OUT
MC_DIR = ROOT / "experiments" / "frame_limit" / "data" / "mc"
REQ = {"Ma": "마크 영상인데 하이라이트 좀 뽑아줘", "Mb": "마크 다이아 캐는 영상이야. 중요한 장면 뽑아줘", "Ra": "로봇이 사탕 컵에 넣는 거 성공한 장면 찾아줘"}
MC_KEYS = {"Ma": ["combat_action", "crisis_or_damage", "fast_action_movement", "rare_find_or_milestone"],
           "Mb": ["diamond_ore_visible", "mining_diamond_ore", "lava_hazard_close", "hostile_mob_visible"]}
WEB = ["web_1116_041133_3", "web_1116_043326_1", "web_1116_043326_2", "web_1116_043326_3"]


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def candidates(final, t0, t1, duration, n=N_CAND):
    """합친 점수 → 후보 클립(순위대로 n개). 클립 규칙은 그대로, 예산만 넓힌다."""
    return C.make_clips(final, t0, t1, duration, budget=0.9)[:n]


def frame_times(clip, duration):
    a = [min(max(clip["peak_t"] + d, 0.0), duration - 0.05) for d in (-0.75, -0.25, 0.25, 0.75)]
    b = [clip["start"] + (i + 0.5) * (clip["end"] - clip["start"]) / 4 for i in range(4)]
    return [round(x, 2) for x in a], [round(min(x, duration - 0.05), 2) for x in b]


def save_item(name, video, request, duration, size, clips, events=None, extra=None, d=None):
    """후보와 프레임을 verify_runs/<name>/에 적는다. d를 주면 그 폴더에 적는다."""
    d = Path(d) if d else OUT / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    width = 640 if size[0] >= size[1] else 360
    cands = []
    for i, c in enumerate(clips, 1):
        ta, tb = frame_times(c, duration)
        files = []
        for tag, ts in (("a", ta), ("b", tb)):
            for j, t in enumerate(ts, 1):
                f = d / "frames" / f"c{i}_{tag}{j}.jpg"
                if not f.exists():
                    data = mf.grab(video, t, width=width)
                    if not data:
                        raise SystemExit(f"{name}: {t}초의 프레임을 뽑지 못했다")
                    f.write_bytes(data)
                files.append(f.name)
        cands.append({"id": i, "rank": c["rank"], "start": c["start"], "end": c["end"], "peak_t": c["peak_t"], "peak": c["peak"], "length": c["length"],
                      "times_a": ta, "times_b": tb, "frames": files, **({"from": c["from"]} if "from" in c else {})})
    mf.write(d / "item.json", {"name": name, "video": str(video), "request": request, "duration": duration, "size": list(size), "candidates": cands,
                               "events": events or [], **(extra or {})})
    return len(cands)


# =====================================================================  prepare
def prepare(_args):
    import mf_views_check as V
    n_items = n_cands = 0
    recs, byv, sets, levels = V.load()
    for arm, keys in MC_KEYS.items():                                           # 마인크래프트 12편 × Ma, Mb
        for v, rs in byv.items():
            vr = C.pct_rank(np.array([r["w"] for r in rs]))
            x = np.mean([V.series({v: rs}, sets, levels, arm, k)[v] for k in keys], axis=0)
            final = C.pct_rank(0.5 * vr + 0.5 * C.pct_rank(x))
            t = np.array([r["t"] for r in rs])
            dur = float(t.max() + 0.5)
            video = next(MC_DIR.glob(v.split("_", 2)[2] + "-*.mp4"))
            clips = candidates(final, list(t - 0.5), list(t + 0.5), dur)
            now = C.make_clips(final, list(t - 0.5), list(t + 0.5), dur)
            events = [{"t": r["t"], "kind": "fight" if r["fight"] else "prog"} for r in rs if r["big"]]
            n_cands += save_item(f"mc_{arm}_{v[:5]}", video, REQ[arm], dur, (640, 360), clips, events,
                                 {"group": f"minecraft_{arm}", "now_clips": [[c["start"], c["end"]] for c in now]})
            n_items += 1
    import mf_gem_check as G                                                    # 로봇 둘 × Ra
    import mf_robot_check as R
    trials = {}
    with open(R.SRC / "clip_list.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            nm = "_".join(row["clip"].split("/")[-1].split("_")[:2])
            if row["source"] in R.RUN_OF:
                trials[nm] = {"run": R.RUN_OF[row["source"]], "start": float(row["start"])}
    for old in ("robot_green", "robot_red"):
        pos, qj, answers, lv, _ = G.load_run("gem_" + old)
        info = mf.read(mf.RUNS / ("gem_" + old) / "run.json")
        keys = [k for k in qj["criteria_keys"] if k.startswith("ra_")]
        final = C.combine(answers, keys, "worth", levels=lv)["final"]
        t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
        events = [{"trial": k, "result": lab, "start": round(trials[k]["start"] + a, 1), "end": round(trials[k]["start"] + b, 1)}
                  for k, (a, b, lab) in R.HELD.items() if trials[k]["run"] == old]
        now = C.make_clips(final, t0, t1, info["duration"])
        n_cands += save_item(f"robot_Ra_{old[6:]}", info["video"], REQ["Ra"], info["duration"], (info["width"], info["height"]),
                             candidates(final, t0, t1, info["duration"]), events, {"group": "robot_Ra", "now_clips": [[c["start"], c["end"]] for c in now]})
        n_items += 1
    for run in WEB:                                                             # 웹사이트 실행 넷
        d = mf.RUNS / run
        info, sc, w = mf.read(d / "run.json"), mf.read(d / "scores.json"), mf.read(d / "writer.json")
        now = mf.read(d / "clips.json")["clips"]
        n_cands += save_item(run, info["video"], w["request"], info["duration"], (info["width"], info["height"]),
                             candidates(np.array(sc["final"]), sc["t0"], sc["t1"], info["duration"]), None,
                             {"group": "web", "now_clips": [[c["start"], c["end"]] for c in now]})
        n_items += 1
    print(f"항목 {n_items}개, 후보 {n_cands}개, 프레임 {n_cands * 8}장 → {OUT}")


# =====================================================================  prepare-rr (20절: 후보를 목록 여럿에서 돌아가며)
def prepare_rr(_args):
    import mf_cand_ceiling as CE
    import mf_combine2_check as K
    import mf_views_check as V
    n_items = n_cands = 0
    recs, byv, sets, levels = V.load()
    for arm, keys in MC_KEYS.items():
        for v, rs in byv.items():
            t = np.array([r["t"] for r in rs])
            t0, t1, dur = list(t - 0.5), list(t + 0.5), float(t.max() + 0.5)
            vr = C.pct_rank(np.array([r["w"] for r in rs]))
            ser = [V.series({v: rs}, sets, levels, arm, k)[v] for k in keys]
            final = C.pct_rank(0.5 * vr + 0.5 * C.pct_rank(np.mean(ser, axis=0)))
            clips = lambda x: C.make_clips(x, t0, t1, dur, budget=0.9)
            names = ["합친 점수"] + keys + ["막연한 질문"]
            cands = [dict(c, **{"from": names[c["source"]]}) for c in CE.round_robin([clips(final)] + [clips(C.pct_rank(x)) for x in ser] + [clips(vr)], N_RR)]
            video = next(MC_DIR.glob(v.split("_", 2)[2] + "-*.mp4"))
            now = C.make_clips(final, t0, t1, dur)
            events = [{"t": r["t"], "kind": "fight" if r["fight"] else "prog"} for r in rs if r["big"]]
            n_cands += save_item(f"mc_{arm}_{v[:5]}", video, REQ[arm], dur, (640, 360), cands, events,
                                 {"group": f"minecraft_{arm}", "now_clips": [[c["start"], c["end"]] for c in now], "order": "p"})
            n_items += 1
    run = "web_1116_043326_3"                                                    # 13-5절의 한 편
    d = mf.RUNS / run
    pos, qj, info, w = mf.read(d / "positions.json"), mf.read(d / "questions.json"), mf.read(d / "run.json"), mf.read(d / "writer.json")
    by_k = {r["k"]: r.get("answers") for r in mf.read(d / "results.json")["records"] if "error" not in r}
    answers = [by_k.get(q["k"]) for q in pos]
    lv = mf.levels_of(qj["questions"])
    keys = [k for k in qj["questions"] if k != "worth"]
    t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
    comb = C.combine(answers, keys, "worth", levels=lv)
    clips = lambda x: C.make_clips(x, t0, t1, info["duration"], budget=0.9)
    ser = [np.nan_to_num(np.array([K.unit((a or {}).get(k), lv, k) for a in answers])) for k in keys]
    vr = C.pct_rank(np.nan_to_num(np.array([K.unit((a or {}).get("worth")) for a in answers])))
    names = ["합친 점수"] + keys + ["막연한 질문"]
    cands = [dict(c, **{"from": names[c["source"]]}) for c in CE.round_robin([clips(comb["final"])] + [clips(C.pct_rank(x)) for x in ser] + [clips(vr)], N_RR)]
    events = [{"t": x, "kind": "fight"} for x in (79.5, 82.5, 84.5, 86.5, 87.5, 91.5, 94.5, 97.5, 139.5, 141.5, 142.5)] + [{"t": x, "kind": "prog"} for x in (45.5, 52.5, 69.5, 71.5, 253.5)]
    n_cands += save_item(run, info["video"], w["request"], info["duration"], (info["width"], info["height"]), cands, events,
                         {"group": "web", "now_clips": [[c["start"], c["end"]] for c in mf.read(d / "clips.json")["clips"]], "order": "p"})
    n_items += 1
    print(f"항목 {n_items}개, 후보 {n_cands}개, 프레임 {n_cands * 8}장 → {OUT}")


# =====================================================================  ask
def build_parts(item, d):
    brief = (HERE / "verifier_brief.md").read_text(encoding="utf-8")
    head = (f"{brief}\n\n---\n\n# 이번 일\n\n## 사용자의 요청 (그대로)\n\n{item['request']}\n\n## 영상\n\n- 길이 {mmss(item['duration'])}\n"
            f"- 후보 {len(item['candidates'])}개. 아래에 차례로 준다\n")
    parts, texts = [{"text": head}], [head]
    for c in item["candidates"]:
        for tag, label, ts in (("a", "(A) 가장 확신한 1초의 연속 4장", c["times_a"]), ("b", "(B) 구간 전체에서 고르게 4장", c["times_b"])):
            t = (f"\n## 후보 {c['id']} — 영상의 {mmss(c['start'])}~{mmss(c['end'])} ({c['length']:.0f}초)\n" if tag == "a" else "") + \
                f"{label} (영상의 {', '.join(f'{x:.2f}초' for x in ts)})"
            parts.append({"text": t})
            texts.append(t)
            for j in range(1, 5):
                raw = (d / "frames" / f"c{c['id']}_{tag}{j}.jpg").read_bytes()
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(raw).decode()}})
    tail = "\n위의 '내놓을 것' 모양의 JSON 하나만 낸다."
    parts.append({"text": tail})
    texts.append(tail)
    return parts, "\n".join(texts)


def check(data, n):
    errs = []
    cs = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(cs, list) or [c.get("id") for c in cs if isinstance(c, dict)] != list(range(1, n + 1)):
        return [f"candidates must list ids 1..{n} in order"]
    for c in cs:
        if not isinstance(c.get("p"), (int, float)) or not 0 <= c["p"] <= 1:
            errs.append(f"candidate {c.get('id')}: p must be a number between 0 and 1")
        if not isinstance(c.get("see"), str) or not c["see"].strip():
            errs.append(f"candidate {c.get('id')}: see must be a short sentence")
    return errs


def ask_one(name, thinking="HIGH", d=None):
    d = Path(d) if d else OUT / name
    item = mf.read(d / "item.json")
    parts, text = build_parts(item, d)
    (d / "prompt.txt").write_text(text, encoding="utf-8")
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 16384, "thinkingConfig": {"thinkingLevel": thinking}, "responseMimeType": "application/json"}}
    usage, errs, data, resp = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}, ["(아직 부르지 않음)"], None, None
    for attempt in range(3):
        resp = W.call(body, W.PROJECT, W.MODEL)
        u = resp.get("usageMetadata", {})
        usage = {"prompt": usage["prompt"] + u.get("promptTokenCount", 0), "output": usage["output"] + u.get("candidatesTokenCount", 0),
                 "thoughts": usage["thoughts"] + u.get("thoughtsTokenCount", 0), "calls": usage["calls"] + 1}
        out_text = W.text_of(resp)
        try:
            data = W.parse_json(out_text)
            errs = check(data, len(item["candidates"]))
        except Exception as e:
            data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
        if not errs:
            break
        body["contents"] += [{"role": "model", "parts": [{"text": out_text}]},
                             {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
    slim = {k: v for k, v in resp.items() if k != "candidates"}
    slim["text"] = W.text_of(resp)
    mf.write(d / "response.json", slim)
    if errs:
        raise RuntimeError(f"{name}: 형식 오류가 남았다: {errs}")
    usd = usage["prompt"] / 1e6 * PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * PRICE_OUT
    verdict = {"name": name, "model": resp.get("modelVersion", W.MODEL), "thinking": thinking, "format_retries": usage["calls"] - 1,
               "usage": {**usage, "usd_estimate": round(usd, 4)}, "note": data.get("note"),
               "candidates": [{"id": c["id"], "p": float(c["p"]), "see": c["see"]} for c in data["candidates"]]}
    mf.write(d / "verdict.json", verdict)
    return verdict


def ask(args):
    names = sorted(p.parent.name for p in OUT.glob("*/item.json"))
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    todo = [n for n in names if not (OUT / n / "verdict.json").exists()]
    spent = sum(mf.read(OUT / n / "verdict.json")["usage"]["usd_estimate"] for n in names if (OUT / n / "verdict.json").exists())
    lock, state = threading.Lock(), {"usd": spent, "stop": False}
    print(f"부를 것 {len(todo)}개 (이미 있는 것 {len(names) - len(todo)}개, 지금까지 ${spent:.2f})", flush=True)

    def one(n):
        with lock:
            if state["stop"]:
                return n, "건너뜀(예산)"
        try:
            v = ask_one(n)
        except Exception as e:
            return n, f"실패: {type(e).__name__}: {str(e)[:200]}"
        with lock:
            state["usd"] += v["usage"]["usd_estimate"]
            if state["usd"] > args.budget:
                state["stop"] = True
        return n, f"p {[round(c['p'], 2) for c in v['candidates']]} | 입력 {v['usage']['prompt']:,} 출력+생각 {v['usage']['output'] + v['usage']['thoughts']:,} ${v['usage']['usd_estimate']}"
    with ThreadPoolExecutor(args.workers) as ex:
        for n, msg in ex.map(one, todo):
            print(f"  {n}: {msg}", flush=True)
    print(f"합계 ${state['usd']:.2f}")


# =====================================================================  eval
def keep_clips(item, verdict):
    """p ≥ KEEP_P인 후보를 순위대로, 예산 15% 안에서."""
    p = {c["id"]: c["p"] for c in verdict["candidates"]}
    kept, used = [], 0.0
    order = sorted(item["candidates"], key=lambda c: -p[c["id"]]) if item.get("order") == "p" else item["candidates"]   # 20절은 p가 높은 순
    for c in order:
        if p[c["id"]] >= KEEP_P and used < C.DEFAULTS["budget"] * item["duration"]:
            kept.append(c)
            used += c["length"]
    return kept


def evaluate(_args):
    out = {"label": "Gemini가 후보의 프레임을 보고 다시 확인한 결과 (pipeline/README.md " + ("18절" if OUT.name == "verify_runs" else "20절: 후보를 목록 여럿에서 돌아가며 뽑음") + ")",
           "keep_p": KEEP_P, "groups": {}, "items": {}}
    usd = tokens = 0
    rows = []
    for d in sorted(OUT.glob("*/verdict.json")):
        item, v = mf.read(d.parent / "item.json"), mf.read(d)
        usd += v["usage"]["usd_estimate"]
        tokens += v["usage"]["prompt"]
        p = {c["id"]: c for c in v["candidates"]}
        ev_t = [e["t"] for e in item["events"] if "t" in e]
        ev_iv = [(e["start"], e["end"], e["trial"], e["result"]) for e in item["events"] if "start" in e]
        has = lambda a, b: any(a <= t < b for t in ev_t) or any(min(b, e1) - max(a, e0) >= 1.0 for e0, e1, _, _ in ev_iv)
        n_in = lambda cs: sum(any(c["start"] <= t < c["end"] for c in cs) for t in ev_t) + sum(any(min(c["end"], e1) - max(c["start"], e0) >= 1.0 for c in cs) for e0, e1, _, _ in ev_iv)
        kept = keep_clips(item, v)
        cand = [{"id": c["id"], "start": c["start"], "end": c["end"], "p": p[c["id"]]["p"], "see": p[c["id"]]["see"], "has_event": bool(has(c["start"], c["end"])) if item["events"] else None,
                 "kept": c in kept, "from": c.get("from")} for c in item["candidates"]]
        now = [{"start": a, "end": b} for a, b in item["now_clips"]]
        rec = {"group": item["group"], "request": item["request"], "events": len(item["events"]), "candidates": cand, "note": v["note"],
               "now": {"clips": len(now), "with_event": sum(has(c["start"], c["end"]) for c in now), "events": n_in(now), "seconds": round(sum(c["end"] - c["start"] for c in now), 1)},
               "checked": {"clips": len(kept), "with_event": sum(has(c["start"], c["end"]) for c in kept), "events": n_in(kept), "seconds": round(sum(c["length"] for c in kept), 1)},
               "perfect": (lambda ks: {"clips": len(ks), "events": n_in(ks)})([c for c in item["candidates"] if has(c["start"], c["end"])]),
               "usage": v["usage"]}
        out["items"][item["name"]] = rec
        rows.append(rec)
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        cs = [c for r in rs for c in r["candidates"] if c["has_event"] is not None]
        agg = {"items": len(rs), "events": sum(r["events"] for r in rs)}
        for k in ("now", "checked", "perfect"):
            agg[k] = {m: round(sum(r[k].get(m, 0) for r in rs), 1) for m in ("clips", "with_event", "events", "seconds") if m in rs[0][k]}
        agg["none_returned"] = sum(r["checked"]["clips"] == 0 for r in rs)
        if cs:
            tp = sum(c["has_event"] and c["p"] >= KEEP_P for c in cs); fn = sum(c["has_event"] and c["p"] < KEEP_P for c in cs)
            fp = sum(not c["has_event"] and c["p"] >= KEEP_P for c in cs); tn = sum(not c["has_event"] and c["p"] < KEEP_P for c in cs)
            y = np.array([c["has_event"] for c in cs]); s = np.array([c["p"] for c in cs])
            import mf_robot_check as R
            agg["candidate_level"] = {"n": len(cs), "with_event_kept": tp, "with_event_dropped": fn, "without_event_kept": fp, "without_event_dropped": tn,
                                      "auc_of_p": round(float(R.auc(s, y)), 3) if y.any() and not y.all() else None}
        out["groups"][g] = agg
    out["cost"] = {"usd_estimate": round(usd, 2), "prompt_tokens": tokens, "calls": sum(r["usage"]["calls"] for r in rows)}
    mf.write(HERE / "eval" / ("verify_check.json" if OUT.name == "verify_runs" else f"{OUT.name}_check.json"), out)
    for g, a in out["groups"].items():
        print(f"== {g}: 항목 {a['items']}, 정답 {a['events']} | 지금 {a['now']} | 확인 뒤 {a['checked']} | 완벽한 확인 {a['perfect']} | '없음'으로 답한 항목 {a['none_returned']}")
        if "candidate_level" in a:
            print("   후보 하나하나:", a["candidate_level"])
    print("비용", out["cost"])
    for name, r in out["items"].items():
        if r["group"] in ("web", "robot_Ra"):
            print(f"-- {name} | {r['request']} | {r['note']}")
            for c in r["candidates"]:
                print(f"     후보 {c['id']} {mmss(c['start'])}~{mmss(c['end'])} p={c['p']:.2f} {'남김' if c['kept'] else '뺌  '} 정답{'있음' if c['has_event'] else ('없음' if c['has_event'] is not None else '?')} {('[' + c['from'] + '] ') if c.get('from') else ''}| {c['see'][:110]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--dir", default="verify_runs", help="항목을 두는 폴더 이름(pipeline/ 아래). 20절은 verify_rr")
    sub.add_parser("prepare")
    sub.add_parser("prepare-rr")
    a = sub.add_parser("ask")
    a.add_argument("--only", default=None)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--budget", type=float, default=8.0)
    sub.add_parser("eval")
    args = ap.parse_args()
    global OUT
    OUT = HERE / args.dir
    {"prepare": prepare, "prepare-rr": prepare_rr, "ask": ask, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
