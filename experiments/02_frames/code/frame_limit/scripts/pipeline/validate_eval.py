"""새 영상에서 실제 jevreel과 2단계 흐름을 견준다. 설정은 이 영상들의 답을 보기 전에 고정했다. 표준 라이브러리만 쓴다.

자료: 확인용 새 Charades 영상 30개(sta2_fetch.py), TimeLens-Bench의 고친 질문
- 실제 jevreel 0.2.0: results/jevreel_real_<묶음>.json (기본 설정, 캐시 끔)
- 2단계 흐름: 시트 칸마다 results/tl_sheet60_<묶음>_preds.json + 확인 results/pipe_<묶음>_s560.json
  주 설정 = 후보 1곳·확인 프레임 긴 변 560px, 보조 = 후보 2곳·560px

채점 (실행 전에 정함)
- 주 기준: 가리킨 시각이 정답 구간의 앞뒤 0.5초 안. jevreel이 0.5초 단위로 답하기 때문에 jevreel에 유리한 쪽으로 잡았다
- 함께: 엄격한 기준(구간 안), 질문당 입력 토큰과 jevreel 대비 배수, 확률 0.5 이상으로 답한 수와 그중 맞은 수, 정답 구간 길이별
- 성공 기준: 주 설정의 토큰이 jevreel의 1.2배 안이고, 맞힌 비율의 차이(우리 − jevreel)의 95% 구간(영상 단위 다시 뽑기)이 0을 넘는 것
- 정답이 영상 전체이거나 판정 위치에 하나도 걸리지 않는 질문은 뺀다(이전 평가와 같은 규칙)

실행: python3 experiments/frame_limit/scripts/pipeline/validate_eval.py [묶음]     (기본 sta2)
결과: 표 출력, results/validate_<묶음>.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import sys

import siglip_eval as E
from lev_probe_eval import boot
from pipeline_run import tile_probs

HERE = E.HERE
SLACK = 0.5
BINS = [("2초 이하", 0, 2), ("3~5초", 3, 5), ("6초 이상", 6, 10 ** 9)]


def read(path):
    return json.loads((HERE / path).read_text())


def inside(t, it, slack=0.0):
    return any(s - slack <= t <= e + slack for s, e in it["windows"])


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "sta2"
    frames = read(f"data/{tag}_frames.json")
    everything = read(f"data/tl_{tag}_items.json")["items"]
    items = []
    for it in everything:
        y = [inside(t, it) for t in frames[it["video"]]["centers"]]
        if any(y) and not all(y):
            items.append(it)
    videos = [it["video"] for it in items]
    stage1 = tile_probs(items, tag, "data/sheets60.json", f"results/tl_sheet60_{tag}_preds.json")
    verify = {(r["item"], r["rank"]): r for r in read(f"results/pipe_{tag}_s560.json")["records"]}
    assert not any(r["error"] for r in verify.values())
    real = {r["item"]: r for r in read(f"results/jevreel_real_{tag}.json")["records"]}
    assert not any(r.get("error") for r in real.values())

    answers = {}  # 방식 → 항목마다 (가리킨 시각 또는 None, 확률, 토큰)
    answers["실제 jevreel 0.2.0"] = [((h := real[it["id"]]["hits"][0] if real[it["id"]]["hits"] else None) and h["at"],
                                    h["probability"] if h else 0.0, real[it["id"]]["input_tokens"]) for it in items]
    centers = lambda it: frames[it["video"]]["centers"]
    answers["시트만, 칸마다 (1단계)"] = []
    for it in items:
        ps = stage1[it["id"]]["probs"]
        best = max(range(len(ps)), key=ps.__getitem__)
        answers["시트만, 칸마다 (1단계)"].append((centers(it)[best], ps[best], stage1[it["id"]]["tokens"]))
    for k, label in ((1, "2단계: 후보 1곳 (주 설정)"), (2, "2단계: 후보 2곳 (보조)")):
        rows = []
        for it in items:
            seen, tokens = {}, stage1[it["id"]]["tokens"]
            for rank in range(1, k + 1):
                r = verify[(it["id"], rank)]
                tokens += r["input_tokens"]
                for q, p in r["nouls"].items():
                    seen[int(q)] = max(p, seen.get(int(q), 0.0))
            best = max(sorted(seen), key=seen.__getitem__)
            rows.append((centers(it)[best], seen[best], tokens))
        answers[label] = rows

    base = "실제 jevreel 0.2.0"
    out = {"tag": tag, "items": len(items), "dropped": len(everything) - len(items), "videos": len(set(videos)), "rows": {}}
    score = {name: {"loose": [float(t is not None and inside(t, it, SLACK)) for (t, _, _), it in zip(rows, items)],
                    "strict": [float(t is not None and inside(t, it)) for (t, _, _), it in zip(rows, items)]}
             for name, rows in answers.items()}
    print(f"새 영상 {len(set(videos))}개, 질문 {len(items)}개 (전체 {len(everything)}개 중 정답이 영상 전체이거나 위치에 안 걸리는 {out['dropped']}개 제외)")
    print(f"{'방식':24s}{'맞힘(±0.5초)':>11s}{'맞힘(엄격)':>9s}{'질문당 토큰':>10s}{'배수':>7s}{'0.5 이상 답(맞음)':>14s}   jevreel과의 차이: ±0.5초 기준 / 엄격한 기준 (95% 구간)")
    for name, rows in answers.items():
        tokens = sum(r[2] for r in rows) / len(rows)
        sure = [(t, it) for (t, p, _), it in zip(rows, items) if t is not None and p >= 0.5]
        row = {"hit_loose": int(sum(score[name]["loose"])), "hit_strict": int(sum(score[name]["strict"])), "tokens": round(tokens),
               "token_ratio": round(tokens / (sum(r[2] for r in answers[base]) / len(items)), 2),
               "answered_at_0.5": len(sure), "answered_correct_loose": sum(inside(t, it, SLACK) for t, it in sure)}
        diff = ""
        if name != base:
            for key in ("loose", "strict"):
                m, lo, hi, _, _ = boot(score[name][key], score[base][key], videos)
                row[f"diff_{key}"] = [m, lo, hi]
            diff = (f"{row['diff_loose'][0]:+.3f} [{row['diff_loose'][1]:+.3f}, {row['diff_loose'][2]:+.3f}] / "
                    f"{row['diff_strict'][0]:+.3f} [{row['diff_strict'][1]:+.3f}, {row['diff_strict'][2]:+.3f}]")
        out["rows"][name] = row
        print(f"{name:24s}{row['hit_loose']:>8d}/{len(items)}{row['hit_strict']:>6d}/{len(items)}{row['tokens']:>12,d}{row['token_ratio']:>6.2f}배"
              f"{row['answered_at_0.5']:>10d}({row['answered_correct_loose']})   {diff}")
    main_row = out["rows"]["2단계: 후보 1곳 (주 설정)"]
    passed = main_row["token_ratio"] <= 1.2 and main_row["diff_loose"][1] > 0
    out["success"] = passed
    print(f"\n성공 기준(주 설정의 토큰이 1.2배 안, ±0.5초 기준 차이의 구간이 0을 넘음): {'충족' if passed else '충족하지 못함'}")
    print("정답 구간 길이별 맞힌 비율 (±0.5초 기준)")
    out["by_length"] = {}
    for label, lo, hi in BINS:
        keep = [k for k, it in enumerate(items) if lo <= it["windows"][0][1] - it["windows"][0][0] <= hi]
        cell = {name: round(sum(score[name]["loose"][k] for k in keep) / max(len(keep), 1), 3) for name in answers}
        out["by_length"][label] = {"items": len(keep), **cell}
        print(f"  {label:8s}({len(keep):2d}개) " + ", ".join(f"{n} {v:.2f}" for n, v in cell.items()))
    # 서로 다른 질문을 맞혔는가
    a, b = score[base]["loose"], score["2단계: 후보 1곳 (주 설정)"]["loose"]
    out["overlap_main_vs_jevreel"] = {"both": sum(x and y for x, y in zip(a, b)), "only_jevreel": sum(x and not y for x, y in zip(a, b)),
                                      "only_ours": sum(y and not x for x, y in zip(a, b)), "neither": sum(not x and not y for x, y in zip(a, b))}
    o = out["overlap_main_vs_jevreel"]
    print(f"주 설정과 jevreel: 둘 다 맞힘 {o['both']}, jevreel만 {o['only_jevreel']}, 우리만 {o['only_ours']}, 둘 다 놓침 {o['neither']}")
    (HERE / "results" / f"validate_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
