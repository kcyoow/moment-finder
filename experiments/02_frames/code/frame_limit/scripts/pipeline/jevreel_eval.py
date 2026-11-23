"""실제 jevreel의 답을 채점하고, 같은 질문에서의 우리 흐름(저장된 답으로 흉내 낸 것)과 견준다. 표준 라이브러리만 쓴다.

자료: Charades 12개 영상, TimeLens-Bench의 고친 질문 43개(동작)
- 실제 jevreel: results/jevreel_real_sta.json (jevreel_run.py. 기본 설정, 기준 0으로 불러 순위가 가장 높은 한 곳을 봄)
- 우리 흐름: pipeline_sim.py의 two_stage (시트 칸마다 → 후보 k곳 → 앞뒤 r초를 원본 4장으로 확인). 이미 받아 둔 답을 조합한 것이라
  실제로 이어서 돌린 것은 아니다

보는 것 (실행 전에 정함)
- 가리킨 한 점이 정답 구간 안인 수. jevreel은 0.5초 단위로 답하므로 앞뒤 0.5초까지 친 수도 함께
- jevreel의 기본 기준(확률 0.5 이상)으로 보고되는 질문 수와 그중 맞은 수
- 질문당 입력 토큰, 맞힌 수의 차이와 95% 구간(영상 단위 다시 뽑기)

실행: python3 experiments/frame_limit/scripts/pipeline/jevreel_eval.py
결과: 표 출력, results/jevreel_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import pipeline_sim as P
from blend_eval import read
from lev_probe_eval import boot

HERE = P.HERE


def inside(t, it, slack=0.0):
    return any(s - slack <= t <= e + slack for s, e in it["windows"])


def main():
    data = P.load()
    items, tile_p, pick_share, sheet_tok, frame_tokens = data
    videos = [it["video"] for it in items]
    motion = {it["id"] for it in items if not it["y"][P.top(it["one"])] and it["y"][P.top(it["win4"])]}
    real = {r["item"]: r for r in read("results/jevreel_real_sta.json")["records"]}
    meta = read("results/jevreel_real_sta.json")["meta"]

    rows, hits = {}, {}

    def add(name, hit, tokens, loose=None):
        hits[name] = [float(h) for h in hit]
        rows[name] = {"hit": sum(hit), "motion_hit": sum(h for h, it in zip(hit, items) if it["id"] in motion),
                      "hit_within_half_second": None if loose is None else sum(loose), "tokens": round(sum(tokens) / len(tokens))}

    tops = [real[it["id"]]["hits"][0] if real[it["id"]]["hits"] else None for it in items]
    add(f"실제 jevreel {meta['jevreel_version']} (기본 설정)", [bool(h) and inside(h["at"], it) for h, it in zip(tops, items)],
        [real[it["id"]]["input_tokens"] for it in items], [bool(h) and inside(h["at"], it, 0.5) for h, it in zip(tops, items)])
    add("우리가 따라 만든 jevreel 1단계 (시트에서 고르기)", [bool(it["y"][P.top(pick_share[it["id"]])]) for it in items],
        [sheet_tok[it["id"]]["pick"] for it in items])
    add("시트만, 칸마다", [bool(it["y"][P.top(tile_p[it["id"]])]) for it in items], [sheet_tok[it["id"]]["tiles"] for it in items])
    for k, r in ((1, 1), (2, 0), (2, 1), (3, 1)):
        picks, toks, _ = P.two_stage(data, k, r)
        add(f"우리 흐름(흉내): 후보 {k}곳, 앞뒤 {r}초를 4장으로 확인", [bool(it["y"][p]) for it, p in zip(items, picks)], toks)
    add("모든 위치를 4장씩", [bool(it["y"][P.top(it["win4"])]) for it in items],
        [len(it["ts"]) * frame_tokens[(it["video"], "win4")] for it in items])

    base = next(iter(rows))
    print(f"질문 {len(items)}개, 그중 움직임을 봐야 하는 것 {len(motion)}개")
    print(f"{'방식':42s}{'맞힘':>8s}{'±0.5초':>8s}{'움직임':>8s}{'질문당 토큰':>10s}   실제 jevreel과의 차이 (맞힌 비율, 95% 구간)")
    for name, r in rows.items():
        diff = ""
        if name != base:
            m, lo, hi, _, _ = boot(hits[name], hits[base], videos)
            r["diff_vs_jevreel"] = [m, lo, hi]
            diff = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        loose = "" if r["hit_within_half_second"] is None else str(r["hit_within_half_second"])
        print(f"{name:42s}{r['hit']:>5d}/{len(items)}{loose:>8s}{r['motion_hit']:>5d}/{len(motion)}{r['tokens']:>12,d}   {diff}")

    # jevreel 자신의 보고 기준
    verified = sum(bool(h) and h["verified"] for h in tops)
    reported = [(h, it) for h, it in zip(tops, items) if h and h["probability"] >= 0.5]
    strong = [(h, it) for h, it in zip(tops, items) if h and h["probability"] >= 0.7]
    print(f"\n실제 jevreel: 원본 프레임으로 확인까지 간 질문 {verified}/{len(items)}. 기본 기준 0.5 이상으로 보고 {len(reported)}개 중 "
          f"{sum(inside(h['at'], it) for h, it in reported)}개 맞음, 0.7 이상(strong) {len(strong)}개 중 {sum(inside(h['at'], it) for h, it in strong)}개 맞음")
    moved = [(h, it) for h, it in zip(tops, items) if h and h["verified"]]
    # 확인 단계가 시트의 고른 칸을 바꿨을 때: 시트의 칸(anchor)은 0초부터 1초 간격이다
    anchor_t = lambda h: int(h["anchor"][1:]) - 1
    before = sum(inside(anchor_t(h), it) for h, it in moved)
    after = sum(inside(h["at"], it) for h, it in moved)
    print(f"확인 단계의 효과(확인까지 간 {len(moved)}개): 시트가 고른 칸이 정답 안 {before}개 → 확인 뒤의 답이 정답 안 {after}개 "
          f"(맞던 것을 틀리게 {sum(inside(anchor_t(h), it) and not inside(h['at'], it) for h, it in moved)}개, "
          f"틀리던 것을 맞게 {sum(not inside(anchor_t(h), it) and inside(h['at'], it) for h, it in moved)}개)")
    out = {"items": len(items), "motion_items": len(motion), "rows": rows,
           "jevreel": {"version": meta["jevreel_version"], "verified": verified, "reported_at_0.5": len(reported),
                       "reported_correct": sum(inside(h["at"], it) for h, it in reported), "strong": len(strong),
                       "sheet_pick_correct_before_verify": before, "correct_after_verify": after,
                       "requests": meta["requests"], "input_tokens": meta["input_tokens"], "cost_usd": meta["cost_usd"]}}
    (HERE / "results" / "jevreel_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
