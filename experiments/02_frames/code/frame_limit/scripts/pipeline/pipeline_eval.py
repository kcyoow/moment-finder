"""2단계 흐름(시트 칸마다 → 후보 → 프레임 묶음으로 확인)을 실제로 돌린 답을 채점하고, 실제 jevreel과 견준다. 표준 라이브러리만 쓴다.

자료: Charades 12개 영상, TimeLens-Bench의 고친 질문 43개(동작). 이 영상들로 설정을 정했으므로 개발용 수치다
- 실제 jevreel: results/jevreel_real_sta.json
- 2단계 흐름: 시트 단계 results/tl_sheet60_preds.json(칸마다) + 확인 단계 results/pipe_sta_s<긴 변>.json (pipeline_run.py)
  후보 1곳만 쓴 경우는 첫째 후보의 요청만 센다

보는 것 (실행 전에 정함)
- 가리킨 한 점이 정답 구간 안인 수, 질문당 입력 토큰(시트 요청 + 확인 요청), jevreel 대비 토큰 배수
- 맞힌 비율의 차이와 95% 구간(영상 단위 다시 뽑기)
- 프레임을 한 요청에 함께 넣은 답이, 위치마다 따로 보냈던 답(4장 조건)과 얼마나 다른가
- 확인 확률 0.5 이상으로 답한 질문 수와 그중 맞은 수

실행: python3 experiments/frame_limit/scripts/pipeline/pipeline_eval.py
결과: 표 출력, results/pipeline_eval.json
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
    rows, hits = {}, {}

    def add(name, hit, tokens, extra=None):
        hits[name] = [float(h) for h in hit]
        rows[name] = {"hit": sum(hit), "motion_hit": sum(h for h, it in zip(hit, items) if it["id"] in motion),
                      "tokens": round(sum(tokens) / len(tokens)), **(extra or {})}

    tops = [real[it["id"]]["hits"][0] if real[it["id"]]["hits"] else None for it in items]
    add("실제 jevreel 0.2.0", [bool(h) and inside(h["at"], it) for h, it in zip(tops, items)], [real[it["id"]]["input_tokens"] for it in items],
        {"hit_within_half_second": sum(bool(h) and inside(h["at"], it, 0.5) for h, it in zip(tops, items)),
         "answered_at_0.5": sum(bool(h) and h["probability"] >= 0.5 for h in tops),
         "answered_correct": sum(bool(h) and h["probability"] >= 0.5 and inside(h["at"], it) for h, it in zip(tops, items))})
    add("시트만, 칸마다 (1단계)", [bool(it["y"][P.top(tile_p[it["id"]])]) for it in items], [sheet_tok[it["id"]]["tiles"] for it in items])
    gap = {}
    for side in (768, 560):
        recs = {}
        for r in read(f"results/pipe_sta_s{side}.json")["records"]:
            assert not r["error"]
            recs[(r["item"], r["rank"])] = r
        both, sep = [], []
        for k in (1, 2):
            picks, toks, probs = [], [], []
            for it in items:
                seen, tokens = {}, sheet_tok[it["id"]]["tiles"]
                for rank in range(1, k + 1):
                    r = recs[(it["id"], rank)]
                    tokens += r["input_tokens"]
                    for q, p in r["nouls"].items():
                        seen[int(q)] = max(p, seen.get(int(q), 0.0))
                        if k == 2:
                            both.append(p)
                            sep.append(it["win4"][int(q)])
                best = max(sorted(seen), key=seen.__getitem__)
                picks.append(best)
                toks.append(tokens)
                probs.append(seen[best])
            hit = [bool(it["y"][p]) for it, p in zip(items, picks)]
            add(f"2단계: 후보 {k}곳, 프레임 긴 변 {side}px", hit, toks,
                {"answered_at_0.5": sum(p >= 0.5 for p in probs), "answered_correct": sum(h for h, p in zip(hit, probs) if p >= 0.5)})
        gap[side] = sum(abs(a - b) for a, b in zip(both, sep)) / len(both)
    for k, r in ((1, 1), (2, 1)):
        picks, toks, _ = P.two_stage(data, k, r)
        add(f"(흉내) 후보 {k}곳, 위치마다 따로 보낸 4장", [bool(it["y"][p]) for it, p in zip(items, picks)], toks)
    add("(참고) 모든 위치를 4장씩", [bool(it["y"][P.top(it["win4"])]) for it in items],
        [len(it["ts"]) * frame_tokens[(it["video"], "win4")] for it in items])

    base = next(iter(rows))
    print(f"질문 {len(items)}개, 그중 움직임을 봐야 하는 것 {len(motion)}개")
    print(f"{'방식':34s}{'맞힘':>8s}{'움직임':>8s}{'질문당 토큰':>10s}{'jevreel 대비':>10s}{'0.5 이상으로 답함(맞음)':>18s}   jevreel과의 차이 (맞힌 비율, 95% 구간)")
    for name, r in rows.items():
        diff = ""
        if name != base:
            m, lo, hi, _, _ = boot(hits[name], hits[base], videos)
            r["diff_vs_jevreel"] = [m, lo, hi]
            diff = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        r["token_ratio"] = round(r["tokens"] / rows[base]["tokens"], 2)
        ans = f"{r['answered_at_0.5']}({r['answered_correct']})" if "answered_at_0.5" in r else ""
        print(f"{name:34s}{r['hit']:>5d}/{len(items)}{r['motion_hit']:>5d}/{len(motion)}{r['tokens']:>12,d}{r['token_ratio']:>9.2f}배{ans:>16s}   {diff}")
    print(f"\n실제 jevreel은 0.5초 단위로 답한다. 앞뒤 0.5초까지 치면 {rows[base]['hit_within_half_second']}/{len(items)}")
    print("프레임을 한 요청에 함께 넣은 답과 위치마다 따로 보낸 답(4장 조건)의 차이(확률, 평균 절댓값): "
          + ", ".join(f"긴 변 {s}px {g:.3f}" for s, g in gap.items()) + " (같은 요청을 다시 보내면 0.005~0.018)")
    (HERE / "results" / "pipeline_eval.json").write_text(json.dumps({"items": len(items), "motion_items": len(motion), "rows": rows,
                                                                    "packed_vs_separate_gap": gap}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
