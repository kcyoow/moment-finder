"""뽑은 후보만 다시 확인하는 방식이 가장 잘됐을 때 얼마나 나아지는지(상한)를 받아 둔 답으로 본다 (이 Mac, 돈 없음). pipeline/README.md 17절.

첫 단계: 제품의 기본 꼴(질문 값 평균과 막연한 질문을 영상 안 순위로 반반). 마인크래프트 12편, 큰 사건 56곳.
(1) 첫 단계 점수의 위 15·30·50% 위치 안에 든 큰 사건 수
(2) 후보 클립을 순위대로 N개 만들고, 큰 사건이 든 클립만 남기는 완벽한 확인자를 둔 뒤 예산 15% 안에서 순위대로 다시 고른다

쓰는 것: pipeline/eval/recheck_ceiling.json
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf_recheck_ceiling.py
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf_core as C  # noqa: E402
import mf_views_check as V  # noqa: E402

BUDGET = C.DEFAULTS["budget"]
ARMS = {"Gemini Ma": ("Ma", ["combat_action", "crisis_or_damage", "fast_action_movement", "rare_find_or_milestone"]),
        "Gemini Mb": ("Mb", ["diamond_ore_visible", "mining_diamond_ore", "lava_hazard_close", "hostile_mob_visible"]),
        "손 기준": ("hand", ["action", "progress"]), "막연한 질문만": (None, [])}


def main():
    recs, byv, sets, levels = V.load()
    out = {"label": "뽑은 후보만 완벽하게 다시 확인했을 때의 상한 (pipeline/README.md 17절)", "big_events": int(sum(r["big"] for r in recs)), "arms": {}}
    for arm, (name, keys) in ARMS.items():
        top = {15: 0, 30: 0, 50: 0}
        now = {"events": 0, "clips": 0, "clips_with_event": 0, "seconds": 0.0}
        check = {n: {"events": 0, "kept": 0, "looked": 0, "seconds": 0.0} for n in (5, 10, 20)}
        for v, rs in byv.items():
            vr = C.pct_rank(np.array([r["w"] for r in rs]))
            if name:
                x = np.mean([V.series({v: rs}, sets, levels, name, k)[v] for k in keys], axis=0)
                final = C.pct_rank(0.5 * vr + 0.5 * C.pct_rank(x))
            else:
                final = vr
            big = np.array([bool(r["big"]) for r in rs])
            t = np.array([r["t"] for r in rs])
            t0, t1 = list(t - 0.5), list(t + 0.5)
            dur = float(max(t1))
            for p in top:
                top[p] += int((big & (final >= 1 - p / 100)).sum())
            has = lambda c: bool((big & (t >= c["start"]) & (t < c["end"])).any())
            n_in = lambda cs: int(sum((big & (t >= c["start"]) & (t < c["end"])).sum() for c in cs))
            clips = C.make_clips(final, t0, t1, dur)                              # 지금: 확인 없음
            now["events"] += n_in(clips); now["clips"] += len(clips); now["clips_with_event"] += sum(map(has, clips)); now["seconds"] += sum(c["length"] for c in clips)
            cand = C.make_clips(final, t0, t1, dur, budget=0.9)                    # 후보를 길게 뽑는다(순위는 꼭대기 값 순)
            for n in check:
                looked = cand[:n]
                kept, used = [], 0.0
                for c in looked:                                                   # 완벽한 확인자: 큰 사건이 든 클립만 남긴다. 예산 15% 안에서 순위대로
                    if has(c) and used < BUDGET * dur:
                        kept.append(c); used += c["length"]
                check[n]["events"] += n_in(kept); check[n]["kept"] += len(kept); check[n]["looked"] += len(looked); check[n]["seconds"] += used
        nv = len(byv)
        out["arms"][arm] = {"events_in_top_percent": top,
                            "now": {**now, "seconds": round(now["seconds"], 1), "clips_per_video": round(now["clips"] / nv, 1)},
                            "perfect_check": {n: {**c, "seconds": round(c["seconds"], 1), "looked_per_video": round(c["looked"] / nv, 1)} for n, c in check.items()}}
    with open(HERE / "eval" / "recheck_ceiling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("큰 사건", out["big_events"], "곳, 영상", len(byv), "편 (영상마다 300초)")
    for arm, r in out["arms"].items():
        print(f"== {arm}: 위 15/30/50% 위치 안의 큰 사건 {r['events_in_top_percent']}")
        print(f"   지금(확인 없음): 클립 {r['now']['clips']}개(영상당 {r['now']['clips_per_video']}) 가운데 사건이 든 클립 {r['now']['clips_with_event']}개, 큰 사건 {r['now']['events']}곳, {r['now']['seconds']}초")
        for n, c in r["perfect_check"].items():
            print(f"   후보 {n}개를 완벽히 확인: 본 클립 {c['looked']}개(영상당 {c['looked_per_video']}), 남긴 클립 {c['kept']}개, 큰 사건 {c['events']}곳, {c['seconds']}초")


if __name__ == "__main__":
    main()
