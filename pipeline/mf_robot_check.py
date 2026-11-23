"""로봇 실습 영상 실행을 소유자의 기록과 견준다 (이 Mac). pipeline/README.md 11-2·11-3절의 방법 그대로다.

읽는 것: pipeline/runs/robot_green, robot_red의 scores.json·clips.json·results.json,
        ~/Documents/피지컬실습/robot_videos_20261017/clips/clip_list.csv (시도마다 원본 영상 안의 시작·끝)
정답 구간(사탕이 집게에 들려 있던 초)은 그 폴더의 labels.csv에 적힌 근거를 README 11-2절에 옮긴 것이다(클립 시각 → 영상 시각은 clip_list.csv로 계산).
쓰는 것: pipeline/eval/robot_check.json

실행: uv run --offline --no-project --with numpy python -I pipeline/mf_robot_check.py
"""
import sys

sys.dont_write_bytecode = True
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf_core as C  # noqa: E402

SRC = Path.home() / "Documents" / "피지컬실습" / "robot_videos_20261017" / "clips"
RUN_OF = {"1080p/20261017_132533_1080p.mp4": "robot_green", "1080p/20261017_135605_1080p.mp4": "robot_red"}
# labels.csv의 근거(클립 시각, 초)에서 옮긴 것: 사탕이 집게에 들려 있던 구간
HELD = {"green_02": (34.0, 53.0, "성공"), "green_05": (80.0, 87.0, "맞는 색을 집었지만 접시에 다시 내려놓음"),
        "green_08": (68.5, 85.0, "다른 색(빨강)을 집어 컵 옆에 떨굼"), "red_02": (32.0, 66.0, "성공"),
        "red_05": (29.0, 35.5, "다른 색(초록)을 집어 컵에 넣음")}
EMPTY_TRIP = {"red_04": (40.0, 48.0), "red_06": (57.0, 63.0)}                       # 빈 집게로 컵 위까지 간 구간
SKIP = {"robot_green": (0.0, 71.2)}                                                 # 초가 적혀 있지 않은 연습 시도


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def auc(score, y):
    """정답인 초가 아닌 초보다 위에 놓일 확률(같은 값은 반)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    r = C.pct_rank(score) * len(score) + 0.5
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    trials = {}
    with open(SRC / "clip_list.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = "_".join(row["clip"].split("/")[-1].split("_")[:2])            # green_02_133347 → green_02
            if row["source"] in RUN_OF:
                trials[name] = {"run": RUN_OF[row["source"]], "start": float(row["start"]), "end": float(row["end"]), "task": row["task"]}
    out = {"label": "로봇 실습 영상: 뽑힌 것과 소유자의 기록 (pipeline/README.md 11절)", "runs": {}}
    for run in ("robot_green", "robot_red"):
        d = HERE / "runs" / run
        sc, cj, res = read(d / "scores.json"), read(d / "clips.json"), read(d / "results.json")
        t0, t1 = np.array(sc["t0"]), np.array(sc["t1"])
        center = 0.5 * (t0 + t1)
        events = [{"trial": k, "result": lab, "start": round(trials[k]["start"] + a, 1), "end": round(trials[k]["start"] + b, 1)}
                  for k, (a, b, lab) in HELD.items() if trials[k]["run"] == run]
        y = np.zeros(len(center), bool)
        for e in events:
            y |= (center >= e["start"]) & (center <= e["end"])
        keep = np.ones(len(center), bool)
        if run in SKIP:
            keep &= ~((center >= SKIP[run][0]) & (center <= SKIP[run][1]))
        final, smoothed = np.array(sc["final"]), np.array(sc["smoothed"])
        recs = {r["k"]: r["answers"] for r in res["records"] if "error" not in r}
        raw = {q: np.array([C.answer_value(recs.get(k, {}).get(q)) for k in range(len(center))]) for q in sc["rank"]}
        clips = cj["clips"]
        for e in events:
            hit = [c for c in clips if min(c["end"], e["end"]) - max(c["start"], e["start"]) >= 1.0]
            e["clips"] = [{"rank": c["rank"], "start": c["start"], "end": c["end"]} for c in hit]
            m = (center >= e["start"]) & (center <= e["end"])
            e["score_percentile_mean"] = round(float(C.pct_rank(final)[m].mean()), 3)
            e["raw_means"] = {q: round(float(np.nanmean(v[m])), 3) for q, v in raw.items()}
        cov = C.covered(clips, t0, t1)
        clip_rows = []
        for c in clips:
            inside = (center >= c["start"]) & (center <= c["end"])
            ev = [e["trial"] for e in events if min(c["end"], e["end"]) - max(c["start"], e["start"]) >= 1.0]
            tr = [k for k, t in trials.items() if t["run"] == run and min(c["end"], t["end"]) - max(c["start"], t["start"]) > 0]
            clip_rows.append({"rank": c["rank"], "start": c["start"], "end": c["end"], "length": c["length"], "peak": c["peak"], "why": c.get("why"),
                              "events": ev, "trials": tr, "skipped_part": bool(run in SKIP and c["start"] < SKIP[run][1]),
                              "event_seconds_inside": int((inside & y).sum())})
        empty = [{"trial": k, "start": round(trials[k]["start"] + a, 1), "end": round(trials[k]["start"] + b, 1),
                  "score_percentile_mean": round(float(C.pct_rank(final)[(center >= trials[k]["start"] + a) & (center <= trials[k]["start"] + b)].mean()), 3),
                  "in_clip": bool(cov[(center >= trials[k]["start"] + a) & (center <= trials[k]["start"] + b)].any())}
                 for k, (a, b) in EMPTY_TRIP.items() if trials[k]["run"] == run]
        per_trial = []
        for k, t in sorted(((k, t) for k, t in trials.items() if t["run"] == run), key=lambda kt: kt[1]["start"]):
            m = (center >= t["start"]) & (center <= t["end"])
            per_trial.append({"trial": k, "start": t["start"], "end": t["end"], "held": k in HELD, "result": HELD.get(k, (0, 0, "집게에 사탕이 들린 적 없음(기록)"))[2],
                              "peak_smoothed": round(float(smoothed[m].max()), 4) if m.any() else None,
                              "top_position_s": round(float(center[m][np.argmax(smoothed[m])]), 1) if m.any() else None})
        out["runs"][run] = {
            "positions": int(len(center)), "scored_positions": int(keep.sum()), "event_seconds": int((y & keep).sum()),
            "meta": {k: res["meta"].get(k) for k in ("requests", "answered", "errors", "rate_per_second", "tokens_median", "wall_seconds", "ready_seconds", "gpu")},
            "auc": {"final": round(auc(final[keep], y[keep]), 4), "smoothed": round(auc(smoothed[keep], y[keep]), 4),
                    **{f"q:{q}": round(auc(v[keep], y[keep]), 4) for q, v in raw.items()}},
            "clips_total_seconds": cj["total_length"], "clips_share": cj["share"], "events": events, "clips": clip_rows,
            "events_in_clips": sum(bool(e["clips"]) for e in events), "empty_trips": empty, "trials": per_trial}
    (HERE / "eval").mkdir(exist_ok=True)
    with open(HERE / "eval" / "robot_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for run, r in out["runs"].items():
        print(f"== {run}: 위치 {r['positions']}곳(채점 {r['scored_positions']}), 정답 초 {r['event_seconds']}, {r['meta']}")
        print("  AUC", r["auc"])
        for e in r["events"]:
            print(f"  정답 {e['trial']} {e['start']}~{e['end']}초 ({e['result']}): 담은 클립 {[(c['rank'], c['start'], c['end']) for c in e['clips']]}, 점수 백분위 {e['score_percentile_mean']}, 값 {e['raw_means']}")
        for c in r["clips"]:
            print(f"  클립 #{c['rank']} {c['start']}~{c['end']} ({c['length']}초) 정점 {c['peak']} 정답 {c['events']} 시도 {c['trials']} {'(채점에서 뺀 구간)' if c['skipped_part'] else ''} {c['why']}")
        print("  빈 집게로 컵까지:", r["empty_trips"])
        print("  시도별 정점:", [(t["trial"][:8], t["peak_smoothed"], "들림" if t["held"] else "") for t in r["trials"]])


if __name__ == "__main__":
    main()
