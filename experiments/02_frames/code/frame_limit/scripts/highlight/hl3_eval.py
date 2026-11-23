"""하이라이트 실험 3(마인크래프트, 성능 우선)의 채점. 표준 라이브러리만 쓴다. Jev 요청은 없다.

1. 찾기: 초당 4장·원본 크기로 훑은 답(results/hl3_scan.json). 사건의 앞뒤 1초 안에 0.5 이상인 프레임이 있으면 찾은 것(작업 26과 같은 기준)
2. 순위: 후보 위치의 점수(results/hl3_score.json) + 알림(results/hl2_toast-corner.json)
   v1 = 0.5 × action/3 + 0.5 × progress/3 (작업 26의 식)
   v2 = 0.4 × action/3 + 0.3 × progress/3 + 0.3 × 알림 (실행 전에 고정. 알림 = 그 위치의 −1~+3초 안에 알림이 켜진 모서리 그림이 있으면 1)
   큰 사건 = 싸움, 피해, 제작 2단계 이상, 다이아몬드 주움. 사건은 앞뒤 1.5초 안의 가장 가까운 후보에 붙인다
   영상별 상위 3곳(3초 안의 이웃은 한 번만)에 큰 사건이 붙은 비율, 넓게 붙인 비율(3초 안 또는 2단계 이상을 만든 제작 창 안),
   눈으로 확인한 알림 15건 가운데 상위 3곳이 그 −1~+7초 안에 든 수

실행: python3 experiments/frame_limit/scripts/highlight/hl3_eval.py
결과: 표 출력, results/hl3_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from pathlib import Path

from hl_eval import auc, mean, read

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CUT, TOP = 0.5, 3
KINDS = ("fight", "crafting", "ore")


def main():
    events = read("data/mc_events.json")["events"]
    gui = {it["video"]: it["windows"] for it in read("data/mc_items.json")["items"] if it["kind"] == "gui"}
    scan = read("results/hl3_scan.json")
    first = read("results/hl_mc_eval.json")
    out = {"find": {}, "rank": {}}

    print("===== 1. 찾기: 초당 4장, 원본 크기 (괄호는 작업 26의 1초 간격 60칸 시트) =====")
    print(f"{'사건':10s}{'기록된 사건':>8s}{'찾음':>12s}{'켜진 프레임':>10s}{'사건의 2초 안':>10s}")
    hot = {name: [(r["video"], r["t"]) for r in scan["records"] if r[name] >= CUT] for name in KINDS}
    for name in KINDS:
        evs = [e for e in events if e["kind"] == name]
        by_video = {}
        for v, t in hot[name]:
            by_video.setdefault(v, []).append(t)
        found = sum(any(abs(t - e["t"]) <= 1.0 for t in by_video.get(e["video"], [])) for e in evs)
        near = sum(any(e["video"] == v and abs(t - e["t"]) <= 2.0 for e in evs) for v, t in hot[name])
        row = {"events": len(evs), "found": found, "flagged": len(hot[name]), "flagged_near_event": near, "first_found": first["find"][name]["found"]}
        extra = ""
        if name == "crafting":
            inside = sum(any(s - 0.25 <= t <= e + 0.25 for s, e in gui[v]) for v, t in hot[name])
            open_frames = [(r["video"], r["t"], r[name]) for r in scan["records"] if any(s <= r["t"] <= e for s, e in gui[r["video"]])]
            row.update(flagged_in_open_window=inside, open_frames=len(open_frames), open_frames_flagged=sum(p >= CUT for _, _, p in open_frames))
            extra = f"   창이 열린 구간 안 {inside}/{len(hot[name])}, 창이 열린 프레임 {len(open_frames)}장 중 켜진 것 {row['open_frames_flagged']}"
        out["find"][name] = row
        print(f"{name:10s}{len(evs):>10d}{found:>8d} ({row['first_found']}){len(hot[name]):>10d}{near:>12d}{extra}")

    print("\n===== 2. 순위 =====")
    corner = [(r["video"], r["t"]) for r in read("results/hl2_toast-corner.json")["records"] if r["advancement"] >= CUT]
    toasts = read("data/mc_toasts.json")["toasts"]
    recs = [r for r in read("results/hl3_score.json")["records"] if not r["error"]]
    for r in recs:
        r["toast"] = float(any(v == r["video"] and r["t"] - 1 <= t <= r["t"] + 3 for v, t in corner))
        r["v1"] = 0.5 * r["action"]["score"] / 3 + 0.5 * r["progress"]["score"] / 3
        r["v2"] = 0.4 * r["action"]["score"] / 3 + 0.3 * r["progress"]["score"] / 3 + 0.3 * r["toast"]
        r["events"] = []
    unmatched = 0
    for e in events:
        if e["kind"] in ("kill", "tool_break"):
            continue
        near = [r for r in recs if r["video"] == e["video"] and abs(r["t"] - e["t"]) <= 1.5]
        if near:
            min(near, key=lambda r: abs(r["t"] - e["t"]))["events"].append(e)
        else:
            unmatched += 1

    def has(r, kind, level=None):
        return any(e["kind"] == kind and (level is None or e.get("level") == level) for e in r["events"])

    big = lambda r: has(r, "fight") or has(r, "hurt") or has(r, "find") or has(r, "crafting", 2) or has(r, "crafting", 3)

    def loose(r):
        if any(e["video"] == r["video"] and abs(e["t"] - r["t"]) <= 3 and (e["kind"] in ("fight", "hurt", "find") or e.get("level", 0) >= 2)
               for e in events):
            return True
        return any(s - 0.5 <= r["t"] <= e + 0.5 and any(x["video"] == r["video"] and x["kind"] == "crafting" and x["level"] >= 2
                                                        and s - 0.5 <= x["t"] <= e + 1.0 for x in events) for s, e in gui[r["video"]])

    print(f"후보 {len(recs)}곳(알림에서 온 것 {sum(r['role'] == 'toast' for r in recs)}곳, 알림이 붙은 곳 {int(sum(r['toast'] for r in recs))}곳). "
          f"후보에 붙지 못한 사건 {unmatched}건 (작업 26: 97건)")
    groups = {"싸움 (준 피해)": [r for r in recs if has(r, "fight")], "피해 입음": [r for r in recs if has(r, "hurt")],
              "제작 3단계": [r for r in recs if has(r, "crafting", 3)], "제작 2단계": [r for r in recs if has(r, "crafting", 2) and not has(r, "crafting", 3)],
              "제작 1단계": [r for r in recs if has(r, "crafting", 1) and not big(r)], "다이아몬드 주움": [r for r in recs if has(r, "find")],
              "광석 캠": [r for r in recs if has(r, "ore") and not big(r)], "사건이 안 붙은 후보": [r for r in recs if not r["events"]]}
    print(f"{'묶음':18s}{'곳':>5s}{'action':>8s}{'progress':>9s}{'알림':>6s}{'v1':>7s}{'v2':>7s}")
    out["rank"]["groups"] = {}
    for name, rs in groups.items():
        if not rs:
            continue
        row = {"n": len(rs), "action": mean([r["action"]["score"] for r in rs]), "progress": mean([r["progress"]["score"] for r in rs]),
               "toast": mean([r["toast"] for r in rs]), "v1": mean([r["v1"] for r in rs]), "v2": mean([r["v2"] for r in rs])}
        out["rank"]["groups"][name] = row
        print(f"{name:18s}{row['n']:>5d}{row['action']:>8.2f}{row['progress']:>9.2f}{row['toast']:>6.2f}{row['v1']:>7.2f}{row['v2']:>7.2f}")
    pos, neg = [r for r in recs if big(r)], [r for r in recs if not big(r)]
    aucs = {"v1: 큰 사건 대 나머지": auc([r["v1"] for r in pos], [r["v1"] for r in neg]),
            "v2: 큰 사건 대 나머지": auc([r["v2"] for r in pos], [r["v2"] for r in neg]),
            "action: 싸움·피해 대 나머지": auc([r["action"]["score"] for r in recs if has(r, "fight") or has(r, "hurt")],
                                        [r["action"]["score"] for r in recs if not (has(r, "fight") or has(r, "hurt"))])}
    out["rank"]["auc"] = aucs
    print("AUC: " + ", ".join(f"{k} {v}" for k, v in aucs.items()) + f"   (작업 26: 최종 점수 {first['rank']['auc']['최종 점수: 큰 사건 대 나머지']})")

    out["rank"]["top"] = {}
    for key in ("v1", "v2"):
        strict = wide = total = 0
        picks_all = []
        for video in sorted({r["video"] for r in recs}):
            picked = []
            for r in sorted((r for r in recs if r["video"] == video), key=lambda r: -r[key]):
                if all(abs(r["t"] - p["t"]) > 3 for p in picked):
                    picked.append(r)
                if len(picked) == TOP:
                    break
            strict += sum(big(r) for r in picked)
            wide += sum(loose(r) for r in picked)
            total += len(picked)
            picks_all += picked
        toast_hit = sum(any(p["video"] == x["video"] and x["t"] - 1 <= p["t"] <= x["t"] + 7 for p in picks_all) for x in toasts)
        base_strict, base_wide = sum(big(r) for r in recs), sum(loose(r) for r in recs)
        out["rank"]["top"][key] = {"strict": [strict, total], "wide": [wide, total], "toasts_in_top": [toast_hit, len(toasts)],
                                   "base_strict": [base_strict, len(recs)], "base_wide": [base_wide, len(recs)]}
        print(f"상위 {TOP}곳 ({key}): 큰 사건이 붙은 곳 {strict}/{total} ({strict / total:.2f}), 넓게 붙이면 {wide}/{total} ({wide / total:.2f}), "
              f"알림 {len(toasts)}건 중 상위에 든 것 {toast_hit}건. 후보 전체에서는 {base_strict}/{len(recs)}, 넓게 {base_wide}/{len(recs)}")
        if key == "v2":
            label = lambda r: "+".join(sorted({e["item"] if e["kind"] in ("crafting", "find") else e["kind"] for e in r["events"]})) or "-"
            for video in sorted({r["video"] for r in picks_all}):
                print(f"  {video[:5]}: " + ", ".join(f"{r['t']:g}s {r['v2']:.2f}{'*' if r['toast'] else ''} [{label(r)}]" for r in picks_all if r["video"] == video))
    f = first["rank"]["top_big_share"]
    print(f"  (작업 26: 큰 사건이 붙은 곳 {f[0]}/{f[1]}, 넓게 붙이면 17/35)   * = 알림이 붙은 곳")
    for r in (r for r in recs if has(r, "crafting", 3)):
        rs = [x for x in recs if x["video"] == r["video"]]
        ranks = {key: sorted(rs, key=lambda x: -x[key]).index(r) + 1 for key in ("v1", "v2")}
        out["rank"]["diamond_pickaxe"] = {"video": r["video"], "t": r["t"], "of": len(rs), **ranks}
        print(f"다이아 곡괭이 제작: {r['video'][:5]} {r['t']:g}s, 그 영상의 후보 {len(rs)}곳 중 v1 {ranks['v1']}위, v2 {ranks['v2']}위")
    score_meta = read("results/hl3_score.json")["meta"]
    out["cost"] = {"scan_requests": scan["meta"]["requests"], "scan_tokens": scan["meta"]["input_tokens"], "scan_usd": scan["meta"]["cost_usd"],
                   "score_new_tokens": score_meta["input_tokens"], "score_new_usd": score_meta["cost_usd"]}
    print(f"\n비용: 훑기 {scan['meta']['requests']:,}건 {scan['meta']['input_tokens']:,}토큰 ${scan['meta']['cost_usd']}, "
          f"새 점수 요청 {score_meta['input_tokens']:,}토큰 ${score_meta['cost_usd']}")
    (HERE / "results" / "hl3_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
