"""하이라이트 실험 2(마인크래프트)의 채점. Jev 요청은 없다. (hl_run을 불러 쓰므로 venv의 파이썬으로 돌린다)

(가) 알림: 정답 data/mc_toasts.json(눈으로 확인한 14건). 찾음 = 0.5 이상인 답의 시각이 [t−1, t+7]초 안
(나) 싸움·피해: 정답 data/mc_events.json(준 피해 15건, 입은 피해 5건). 찾음 = 0.5 이상인 답의 시각이 [t−0.5, t+0.75]초 안
지표는 실행 전에 정했다(작업 28). 정답 밖에서 켜진 곳은 목록으로 내어 눈으로 확인한다.

실행: <venv>/bin/python experiments/frame_limit/scripts/highlight/hl2_eval.py
결과: 표 출력, results/hl2_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import hl_run as H

HERE = H.HERE
CUT = 0.5
TOAST_WIN, EVENT_WIN = (-1.0, 7.0), (-0.5, 0.75)


def read(path):
    return json.loads((HERE / path).read_text())


def runs(points, gap):
    """[(영상, 시각)]을 영상별로 gap초 안의 이웃끼리 묶는다. [(영상, 처음, 끝, 개수)]"""
    out = []
    for video, t in sorted(points):
        if out and out[-1][0] == video and t - out[-1][2] <= gap:
            out[-1] = (video, out[-1][1], t, out[-1][3] + 1)
        else:
            out.append((video, t, t, 1))
    return out


def toast_part(out):
    toasts = read("data/mc_toasts.json")["toasts"]
    inside = lambda v, t: any(x["video"] == v and x["t"] + TOAST_WIN[0] <= t <= x["t"] + TOAST_WIN[1] for x in toasts)
    sheet = [(r["video"], r["times"][tid], p) for r in read("results/hl2_toast-sheet.json")["records"] for tid, p in r["nouls"].items()]
    corner = [(r["video"], r["t"], r["advancement"]) for r in read("results/hl2_toast-corner.json")["records"]]
    recipe = [(r["video"], r["t"], r["recipe"]) for r in read("results/hl2_toast-corner.json")["records"]]
    print(f"===== (가) 알림: 눈으로 확인한 {len(toasts)}건 =====")
    print(f"{'보는 법':24s}{'찾음':>8s}{'켜진 곳':>8s}{'그중 알림 둘레':>10s}{'알림 밖 덩어리':>10s}")
    for name, rows, gap in (("60칸 시트 (1초 간격)", sheet, 1.5), ("모서리를 크게 (2초 간격)", corner, 2.5)):
        hot = [(v, t) for v, t, p in rows if p >= CUT]
        found = sum(any(v == x["video"] and x["t"] + TOAST_WIN[0] <= t <= x["t"] + TOAST_WIN[1] for v, t in hot) for x in toasts)
        outside = runs([(v, t) for v, t in hot if not inside(v, t)], gap)
        out["toast"][name] = {"found": found, "toasts": len(toasts), "flagged": len(hot), "flagged_in_window": sum(inside(v, t) for v, t in hot),
                              "outside_runs": [[v, a, b, n] for v, a, b, n in outside]}
        print(f"{name:24s}{found:>6d}/{len(toasts)}{len(hot):>9d}{sum(inside(v, t) for v, t in hot):>12d}{len(outside):>12d}")
        missed = [f"{x['video'][:5]}@{x['t']:g}s({x['title']})" for x in toasts
                  if not any(v == x["video"] and x["t"] + TOAST_WIN[0] <= t <= x["t"] + TOAST_WIN[1] for v, t in hot)]
        if missed:
            print("   못 찾은 것: " + ", ".join(missed))
        print("   알림 밖에서 켜진 덩어리: " + ", ".join(f"{v[:5]} {a:g}~{b:g}s({n})" for v, a, b, n in outside)[:900])
    hot_recipe = sum(p >= CUT for _, _, p in recipe)
    both = sum(a[2] >= CUT and b[2] >= CUT for a, b in zip(corner, recipe))
    out["toast"]["recipe_flagged"] = hot_recipe
    print(f"모서리 {len(corner)}장 중 조합법 알림으로 켜진 것 {hot_recipe}장, 발전 과제 알림으로 켜진 것 {sum(p >= CUT for _, _, p in corner)}장(둘 다 켜진 것 {both}장)")


def event_part(out):
    events = [e for e in read("data/mc_events.json")["events"] if e["kind"] in ("fight", "hurt")]
    frames = read("data/mc_frames.json")
    base = H.tile_scores()
    dense = read("results/hl2_dense-sheet.json")["records"]
    single = read("results/hl2_frames.json")["records"]
    print(f"\n===== (나) 싸움·피해: 준 피해 {sum(e['kind'] == 'fight' for e in events)}건, 입은 피해 {sum(e['kind'] == 'hurt' for e in events)}건 =====")
    print(f"{'사건':6s}{'보는 법':26s}{'찾음':>8s}{'견줄 곳에서 켜진 비율':>16s}")
    for kind in ("fight", "hurt"):
        evs = [e for e in events if e["kind"] == kind]
        win = lambda e, t: e["t"] + EVENT_WIN[0] <= t <= e["t"] + EVENT_WIN[1]
        rows = {}
        b_found = sum(any(p >= CUT and win(e, t) for t, p in zip(frames[e["video"]]["centers"], base[e["video"]][kind])) for e in evs)
        b_all = [p for v in base for p in base[v][kind]]
        rows["60칸 시트, 1초 간격 (작업 26)"] = (b_found, f"{sum(p >= CUT for p in b_all)}/{len(b_all)} (전체 칸)")
        d_found = sum(any(p >= CUT and win(e, t) for r in dense if r["video"] == e["video"] for t, p in zip(r["times"], r["nouls"][kind])) for e in evs)
        ctrl = [p for r in dense if r["role"] == "control" for p in r["nouls"][kind]]
        rows["60칸 시트, 초당 4장"] = (d_found, f"{sum(p >= CUT for p in ctrl)}/{len(ctrl)}")
        s_found = sum(any(r[kind] >= CUT and win(e, r["t"]) for r in single if r["role"] == "event" and r["video"] == e["video"]) for e in evs)
        ctrl = [r[kind] for r in single if r["role"] == "control"]
        rows["원본 한 장, 초당 4장"] = (s_found, f"{sum(p >= CUT for p in ctrl)}/{len(ctrl)}")
        out["events"][kind] = {name: {"found": f, "events": len(evs), "control": c} for name, (f, c) in rows.items()}
        for name, (f, c) in rows.items():
            print(f"{kind:6s}{name:26s}{f:>6d}/{len(evs)}{c:>20s}")
        best = [max((r[kind] for r in single if r["role"] == "event" and r["video"] == e["video"] and win(e, r["t"])), default=0.0) for e in evs]
        out["events"][kind]["single_frame_best"] = [round(b, 2) for b in best]
        print(f"      원본 한 장에서 사건마다 가장 높은 확률: " + " ".join(f"{b:.2f}" for b in best))


def main():
    out = {"toast": {}, "events": {}, "cost": {}}
    toast_part(out)
    event_part(out)
    total = 0
    print("\n비용")
    for stage in ("toast-sheet", "toast-corner", "dense-sheet", "frames"):
        m = read(f"results/hl2_{stage}.json")["meta"]
        out["cost"][stage] = {"requests": m["requests"], "input_tokens": m["input_tokens"], "usd": m["cost_usd"]}
        total += m["input_tokens"]
        print(f"  {stage:14s} {m['requests']:5d}건 {m['input_tokens']:>10,}토큰 ${m['cost_usd']}")
    out["cost"]["total_usd"] = round(total * H.PRICE_PER_MTOK / 1e6, 4)
    print(f"  합 약 ${out['cost']['total_usd']}")
    (HERE / "results" / "hl2_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
