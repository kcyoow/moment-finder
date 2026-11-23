"""하이라이트 뽑기 첫 실험(마인크래프트)의 채점. 표준 라이브러리만 쓴다. Jev 요청은 없다.

정답은 행동 기록의 사건(data/mc_events.json)과 창이 열린 구간(data/mc_items.json)이다. 사건은 하이라이트의 대리 지표다.

지표 (실행 전에 정함)
1. 찾기(요청 A): 사건마다 같은 종류의 칸이 앞뒤 1초 안에서 0.5 이상이면 찾은 것. 켜진 칸 가운데 같은 종류 사건의 앞뒤 2초 안인 수
   (사건은 점이고 장면은 더 길어서 낮게 잡히는 값). crafting은 창이 열린 구간과도 견준다
2. 순위(요청 B): 사건을 앞뒤 1.5초 안의 가장 가까운 후보 위치에 붙인다. 묶음별 평균 점수. action은 fight·hurt 대 나머지,
   progress는 제작 2단계 이상·다이아몬드 대 나머지의 AUC. 최종 점수 = 0.5 × action/3 + 0.5 × progress/3.
   영상별 상위 3곳에 "큰 사건"(fight, hurt, 제작 2단계 이상, 다이아몬드)이 붙은 비율을 후보 전체에서의 비율과 견준다

실행: python3 experiments/frame_limit/scripts/highlight/hl_eval.py
결과: 표 출력, results/hl_mc_eval.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CUT, NEAR_FIND, NEAR_TILE, NEAR_SCORE, TOP = 0.5, 1.0, 2.0, 1.5, 3
A_KINDS = {"fight": "fight", "hurt": "hurt", "crafting": "crafting", "ore": "ore"}  # 요청 A의 사건 → 정답의 종류


def read(path):
    return json.loads((HERE / path).read_text())


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    events = read("data/mc_events.json")["events"]
    frames = read("data/mc_frames.json")
    gui = {it["video"]: it["windows"] for it in read("data/mc_items.json")["items"] if it["kind"] == "gui"}
    sheets = {(s["video"], s["sheet"]): s for s in read("data/sheets60.json")["sheets"] if s["tag"] == "mc"}
    tile = {}  # (영상, 사건) → {시각: 확률}
    sheet_data = read("results/hl_mc_sheet.json")
    for r in sheet_data["records"]:
        for name, by_tile in r["nouls"].items():
            row = tile.setdefault((r["video"], name), {})
            for t in sheets[(r["video"], r["sheet"])]["tiles"]:
                row[t["t"]] = by_tile[t["id"]]
    out = {"find": {}, "rank": {}}

    print("===== 1. 찾기 (요청 A: 시트에서 칸마다) =====")
    print(f"{'사건':10s}{'기록된 사건':>8s}{'찾음':>6s}{'켜진 칸':>8s}{'사건의 2초 안':>10s}")
    for name, kind in A_KINDS.items():
        evs = [e for e in events if e["kind"] == kind]
        found = sum(any(p >= CUT and abs(t - e["t"]) <= NEAR_FIND for t, p in tile[(e["video"], name)].items()) for e in evs)
        hot = [(v, t) for (v, n), row in tile.items() if n == name for t, p in row.items() if p >= CUT]
        near = sum(any(e["video"] == v and abs(t - e["t"]) <= NEAR_TILE for e in evs) for v, t in hot)
        row = {"events": len(evs), "found": found, "flagged_tiles": len(hot), "flagged_near_event": near}
        extra = ""
        if name == "crafting":
            inside = sum(any(s - 0.5 <= t <= e + 0.5 for s, e in gui[v]) for v, t in hot)
            open_pos = sum(any(s <= t <= e for s, e in gui[v]) for v in gui for t in frames[v]["centers"])
            covered = sum(any(s <= t <= e for s, e in gui[v]) and tile[(v, name)][t] >= CUT for v in gui for t in frames[v]["centers"])
            row.update(flagged_in_open_window=inside, open_positions=open_pos, open_positions_flagged=covered)
            extra = f"   창이 열린 구간 안인 칸 {inside}/{len(hot)}, 창이 열린 위치 {open_pos}곳 중 켜진 곳 {covered}"
        out["find"][name] = row
        print(f"{name:10s}{len(evs):>10d}{found:>8d}{len(hot):>9d}{near:>12d}{extra}")
    crafts = [e for e in events if e["kind"] == "crafting"]
    for level in (1, 2, 3):
        evs = [e for e in crafts if e["level"] == level]
        found = sum(any(p >= CUT and abs(t - e["t"]) <= NEAR_FIND for t, p in tile[(e["video"], "crafting")].items()) for e in evs)
        print(f"  제작 {level}단계: {found}/{len(evs)}")
    finds = [e for e in events if e["kind"] == "find"]
    any_kind = sum(any(p >= CUT and abs(t - e["t"]) <= NEAR_FIND for name in A_KINDS for t, p in tile[(e["video"], name)].items()) for e in finds)
    print(f"  다이아몬드를 주운 순간 {len(finds)}건 중 어느 사건이든 켜진 칸이 1초 안에 있는 것 {any_kind}건")
    out["find"]["diamond_pickups_near_any_flag"] = [any_kind, len(finds)]

    print("\n===== 2. 순위 (요청 B: 후보 위치의 점수) =====")
    score_data = read("results/hl_mc_score.json")
    recs = [r for r in score_data["records"] if not r["error"]]
    for r in recs:
        r["final"] = 0.5 * r["action"]["score"] / 3 + 0.5 * r["progress"]["score"] / 3
        r["events"] = []
    cand = [r for r in recs if r["role"] == "candidate"]
    unmatched = 0
    for e in events:
        if e["kind"] in ("kill", "tool_break"):
            continue
        near = [r for r in cand if r["video"] == e["video"] and abs(r["t"] - e["t"]) <= NEAR_SCORE]
        if near:
            min(near, key=lambda r: abs(r["t"] - e["t"]))["events"].append(e)
        else:
            unmatched += 1

    def has(r, kind, level=None):
        return any(e["kind"] == kind and (level is None or e.get("level") == level) for e in r["events"])

    big = lambda r: has(r, "fight") or has(r, "hurt") or has(r, "find") or has(r, "crafting", 2) or has(r, "crafting", 3)
    groups = {"싸움 (준 피해)": [r for r in cand if has(r, "fight")], "피해 입음": [r for r in cand if has(r, "hurt")],
              "제작 3단계 (다이아몬드 도구)": [r for r in cand if has(r, "crafting", 3)],
              "제작 2단계 (작업대·화로·도구)": [r for r in cand if has(r, "crafting", 2) and not has(r, "crafting", 3)],
              "제작 1단계 (판자·막대 등)": [r for r in cand if has(r, "crafting", 1) and not big(r)],
              "다이아몬드 주움": [r for r in cand if has(r, "find")],
              "광석 캠": [r for r in cand if has(r, "ore") and not big(r)],
              "사건이 안 붙은 후보": [r for r in cand if not r["events"]],
              "견줄 곳 (아무 일 없는 곳)": [r for r in recs if r["role"] == "control"]}
    print(f"후보 {len(cand)}곳, 견줄 곳 {len(recs) - len(cand)}곳. 후보에 붙지 못한 사건 {unmatched}건")
    print(f"{'묶음':24s}{'곳':>5s}{'action':>8s}{'progress':>9s}{'최종':>7s}{'worth':>7s}")
    out["rank"]["groups"] = {}
    for name, rs in groups.items():
        row = {"n": len(rs), "action": mean([r["action"]["score"] for r in rs]), "progress": mean([r["progress"]["score"] for r in rs]),
               "final": mean([r["final"] for r in rs]), "worth": mean([r["worth"] for r in rs])}
        out["rank"]["groups"][name] = row
        if rs:
            print(f"{name:24s}{len(rs):>5d}{row['action']:>8.2f}{row['progress']:>9.2f}{row['final']:>7.2f}{row['worth']:>7.2f}")
    everything = recs
    act_pos = [r["action"]["score"] for r in cand if has(r, "fight") or has(r, "hurt")]
    act_neg = [r["action"]["score"] for r in everything if not (has(r, "fight") or has(r, "hurt"))]
    pro_pos = [r["progress"]["score"] for r in cand if has(r, "crafting", 2) or has(r, "crafting", 3) or has(r, "find")]
    pro_neg = [r["progress"]["score"] for r in everything if not (has(r, "crafting", 2) or has(r, "crafting", 3) or has(r, "find"))]
    big_pos, big_neg = [r for r in cand if big(r)], [r for r in everything if not big(r)]
    aucs = {"action: 싸움·피해 대 나머지": auc(act_pos, act_neg), "progress: 제작 2단계 이상·다이아몬드 대 나머지": auc(pro_pos, pro_neg),
            "최종 점수: 큰 사건 대 나머지": auc([r["final"] for r in big_pos], [r["final"] for r in big_neg]),
            "worth(막연한 질문): 큰 사건 대 나머지": auc([r["worth"] for r in big_pos], [r["worth"] for r in big_neg])}
    out["rank"]["auc"] = aucs
    print("AUC (0.5 = 못 가름, 1 = 완전히 가름): " + ", ".join(f"{k} {v}" for k, v in aucs.items()))

    print(f"\n영상별 최종 점수 상위 {TOP}곳")
    top_big = top_all = 0
    out["rank"]["top"] = {}
    for video in sorted({r["video"] for r in cand}):
        rs = sorted((r for r in cand if r["video"] == video), key=lambda r: -r["final"])
        picked, seen = [], []
        for r in rs:  # 같은 덩어리나 3초 안의 이웃은 한 번만
            if all(abs(r["t"] - t) > 3 for t in seen):
                picked.append(r)
                seen.append(r["t"])
            if len(picked) == TOP:
                break
        top_big += sum(big(r) for r in picked)
        top_all += len(picked)
        label = lambda r: "+".join(sorted({e["item"] if e["kind"] in ("crafting", "find") else e["kind"] for e in r["events"]})) or "-"
        out["rank"]["top"][video] = [{"t": r["t"], "final": round(r["final"], 3), "events": label(r), "big": bool(big(r))} for r in picked]
        print(f"  {video[:5]} (후보 {len(rs)}곳, 큰 사건이 붙은 곳 {sum(big(r) for r in rs)}): "
              + ", ".join(f"{r['t']:g}s {r['final']:.2f} [{label(r)}]" for r in picked))
    base = sum(big(r) for r in cand) / len(cand)
    out["rank"]["top_big_share"] = [top_big, top_all, round(base, 3)]
    print(f"상위 {TOP}곳 가운데 큰 사건이 붙은 곳 {top_big}/{top_all} ({top_big / top_all:.2f}). 후보 전체에서는 {sum(big(r) for r in cand)}/{len(cand)} ({base:.2f})")
    dia = [r for r in cand if has(r, "crafting", 3)]
    for r in dia:
        rs = sorted((x for x in cand if x["video"] == r["video"]), key=lambda x: -x["final"])
        rank = rs.index(r) + 1
        out["rank"]["diamond_pickaxe"] = {"video": r["video"], "t": r["t"], "rank": rank, "of": len(rs), "final": round(r["final"], 3),
                                          "progress": r["progress"]["score"]}
        print(f"다이아 곡괭이 제작: {r['video'][:5]} {r['t']:g}s, 그 영상의 후보 {len(rs)}곳 중 {rank}위 (progress {r['progress']['score']:.2f})")
    tokens = sheet_data["meta"]["input_tokens"] + score_data["meta"]["input_tokens"]
    out["cost"] = {"sheet_tokens": sheet_data["meta"]["input_tokens"], "score_tokens": score_data["meta"]["input_tokens"],
                   "usd": round(tokens * 0.042 / 1e6, 4), "per_video_minute_usd": round(tokens * 0.042 / 1e6 / 60, 5)}
    print(f"\n비용: 요청 A {sheet_data['meta']['input_tokens']:,}토큰 + 요청 B {score_data['meta']['input_tokens']:,}토큰 = 약 ${out['cost']['usd']} "
          f"(영상 60분, 분당 약 ${out['cost']['per_video_minute_usd']})")
    (HERE / "results" / "hl_mc_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
