""""나오는 곳을 모두 찾기"를 실제 jevreel과 견준다. 표준 라이브러리만 쓴다. Jev 요청은 없다.

묶음
- mc  : 마인크래프트 12개 영상(각 5분). 정답은 행동 기록의 창이 열린 구간. 질문 두 가지(어떤 창이든 / 인벤토리만)
- qvm : QVHighlights에서 정답 구간이 3개 이상인 질문 20개(각 150초). 정답은 사람이 표시한 구간(2초 단위)

방식
- 실제 jevreel 0.2.0 (results/jevreel_real_<묶음>.json, 기본 설정, 한 번 묻기): 낸 곳 = 확률 0.5 이상인 hit의 시각과 그 hit의
  also 시각. 참고로 기준을 0으로 둔 것(돌려준 것 전부)도 센다
- A 시트만: 60칸 시트의 칸마다 확률 0.5 이상인 위치 전부
- B 시트 → 위치마다 원본 한 장: A의 위치 가운데 원본 프레임 한 장의 답도 0.5 이상인 것 (모든 위치의 한 장 답이 있는 mc에서만)
- C 시트 → 덩어리마다 원본 한 장: A의 이어진 덩어리마다 가운데 한 장만 보고, 0.5 이상이면 덩어리 전체를 냄
  mc의 B·C는 받아 둔 한 장 답(results/mc_preds.json)으로 계산한 것이고, qvm의 C는 runs_verify.py가 실제로 보낸 것이다

채점 (실행 전에 정함)
- 구간을 찾음 = 낸 시각 중 하나가 구간의 앞뒤 0.5초 안. 맞게 냄 = 낸 시각이 어느 구간의 앞뒤 0.5초 안
- 찾은 구간 수(전체, 1초 이상인 구간만), 낸 시각 중 맞는 비율, 같은 1분(0~60초, 60~120초 …)에 시작한 구간이 1·2·3개 이상일 때의
  찾은 비율, 정답이 없는 물음에서 무언가를 낸 수, 물음당 토큰. 차이의 구간은 영상 단위 다시 뽑기
- 우리 쪽은 위치의 묶음(구간)으로 답하므로, 정답 위치를 덮은 비율과 낸 위치 중 정답인 비율(여유 없이)도 적는다. jevreel은 점으로 답해 이 값이 없다

실행: python3 experiments/frame_limit/scripts/pipeline/mc_all_eval.py [mc|qvm]
결과: 표 출력, results/mc_all_eval.json (mc) 또는 results/findall_<묶음>.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import sys
from pathlib import Path

from lev_probe_eval import boot

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
SLACK, CUT = 0.5, 0.5
BASE, EVERYTHING = "실제 jevreel (0.5 이상)", "실제 jevreel (돌려준 것 전부)"
OURS_A, OURS_B, OURS_C = "A 시트만 (칸마다 0.5 이상)", "B 시트 → 위치마다 원본 한 장", "C 시트 → 덩어리마다 원본 한 장"
SETS = {"mc": {"sheet_preds": "results/mc_sheet60_preds.json", "out": "results/mc_all_eval.json",
               "kinds": [("gui", "창이 열려 있다 (어떤 창이든)"), ("inv", "인벤토리 창만")]},
        "qvm": {"sheet_preds": "results/sheet60_qvm_preds.json", "out": "results/findall_qvm.json",
                "kinds": [(None, "QVHighlights, 정답 구간이 3개 이상인 질문")]}}


def read(path):
    return json.loads((HERE / path).read_text())


def runs_of(flags):
    """이어진 True의 덩어리마다 위치 번호의 목록."""
    out, cur = [], []
    for k, f in enumerate(flags):
        if f:
            cur.append(k)
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def one_frame(tag):
    """(항목, 시각) → (원본 한 장의 답, 그 요청의 토큰). 둘째 값은 모든 위치의 답이 있는지."""
    if tag == "mc":  # 위치마다 한 요청에 두 질문을 함께 물은 것이라 토큰은 넉넉하게 잡힌다
        return {(item, r["t"]): (p, r["input_tokens"]) for r in read("results/mc_preds.json")["records"]
                for item, p in r["nouls"].items()}, True
    data = read(f"results/runs_{tag}.json")
    assert data["meta"]["errors"] == 0, "덩어리 확인에 오류가 남아 있다"
    return {(r["item"], r["t"]): (r["noul"], r["input_tokens"]) for r in data["records"]}, False


def reported(items, tag="mc"):
    """방식 → 물음(항목)마다 (낸 시각의 목록, 토큰)."""
    frames = read(f"data/{tag}_frames.json")
    real = {r["item"]: r for r in read(f"results/jevreel_real_{tag}.json")["records"]}
    assert not any(r.get("error") for r in real.values()), "jevreel 답에 오류가 남아 있다"
    sheets = [s for s in read("data/sheets60.json")["sheets"] if s["tag"] == tag]
    tiles = {(r["item"], r["sheet"]): r for r in read(SETS[tag]["sheet_preds"])["records"] if r["kind"] == "tiles" and not r["error"]}
    one, everywhere = one_frame(tag)
    out = {name: {} for name in (BASE, EVERYTHING, OURS_A, *([OURS_B] if everywhere else []), OURS_C)}
    for it in items:
        r = real[it["id"]]
        for name, cut in ((BASE, CUT), (EVERYTHING, 0.0)):
            times = sorted({t for h in r["hits"] if h["probability"] >= cut for t in [h["at"], *(m["at"] for m in h["also"])]})
            out[name][it["id"]] = (times, r["input_tokens"])
        centers = frames[it["video"]]["centers"]
        probs, tokens = {}, 0
        for sh in (s for s in sheets if s["video"] == it["video"]):
            rec = tiles[(it["id"], sh["sheet"])]
            tokens += rec["input_tokens"]
            for t in sh["tiles"]:
                probs[t["pos"]] = rec["nouls"][t["id"]]
        flags = [probs[k] >= CUT for k in range(len(centers))]
        seen = lambda k: one[(it["id"], centers[k])][0] >= CUT
        cost = lambda ks: sum(one[(it["id"], centers[k])][1] for k in ks)
        flagged = [k for k, f in enumerate(flags) if f]
        out[OURS_A][it["id"]] = ([centers[k] for k in flagged], tokens)
        if everywhere:
            out[OURS_B][it["id"]] = ([centers[k] for k in flagged if seen(k)], tokens + cost(flagged))
        groups = runs_of(flags)
        mids = [g[len(g) // 2] for g in groups]
        out[OURS_C][it["id"]] = ([centers[k] for g, m in zip(groups, mids) if seen(m) for k in g], tokens + cost(mids))
    return out


def near(t, win, slack=SLACK):
    return win[0] - slack <= t <= win[1] + slack


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "mc"
    items = read(f"data/{tag}_items.json")["items"]
    frames = read(f"data/{tag}_frames.json")
    answers = reported(items, tag)
    out = {"tag": tag, "slack": SLACK, "cut": CUT, "kinds": {}}
    for kind, title in SETS[tag]["kinds"]:
        its = [it for it in items if kind is None or it["kind"] == kind]
        spans = [(it, w, sum(int(o[0] // 60) == int(w[0] // 60) for o in it["windows"])) for it in its for w in it["windows"]]
        long_ones = [k for k, (_, w, _) in enumerate(spans) if w[1] - w[0] >= 1.0]
        empty = [it for it in its if not it["windows"]]
        truth = {it["id"]: {t for t in frames[it["video"]]["centers"] if any(near(t, w, 0.0) for w in it["windows"])} for it in its}
        print(f"\n===== {title}: 물음 {len(its)}개, 구간 {len(spans)}개(1초 이상 {len(long_ones)}개), 정답이 없는 물음 {len(empty)}개, "
              f"정답 위치 {sum(len(v) for v in truth.values())}개 =====")
        print(f"{'방식':28s}{'찾은 구간':>10s}{'1초 이상':>10s}{'낸 시각(맞음)':>14s}{'맞는 비율':>8s}{'없는데 냄':>8s}{'물음당 토큰':>10s}{'배수':>7s}"
              "   같은 1분에 구간 1 / 2 / 3개 이상   정답 위치를 덮음 · 낸 위치 중 정답   jevreel과의 차이(찾은 구간의 비율, 95% 구간)")
        found = {name: [float(any(near(t, w) for t in rows[it["id"]][0])) for it, w, _ in spans] for name, rows in answers.items()}
        base_tokens = sum(answers[BASE][it["id"]][1] for it in its) / len(its)
        table = {}
        for name, rows in answers.items():
            times = [(t, it) for it in its for t in rows[it["id"]][0]]
            right = sum(any(near(t, w) for w in it["windows"]) for t, it in times)
            tokens = sum(rows[it["id"]][1] for it in its) / len(its)
            buckets = []
            for lo, hi in ((1, 1), (2, 2), (3, 10 ** 9)):
                keep = [k for k, (_, _, n) in enumerate(spans) if lo <= n <= hi]
                buckets.append((sum(found[name][k] for k in keep), len(keep)))
            row = {"found": int(sum(found[name])), "intervals": len(spans), "found_long": int(sum(found[name][k] for k in long_ones)),
                   "intervals_long": len(long_ones), "reported": len(times), "reported_right": right,
                   "precision": round(right / len(times), 3) if times else None,
                   "false_alarm_asks": sum(bool(rows[it["id"]][0]) for it in empty), "empty_asks": len(empty),
                   "tokens": round(tokens), "token_ratio": round(tokens / base_tokens, 2),
                   "by_crowding": {label: list(b) for label, b in zip(("1", "2", "3+"), buckets)}}
            cover = ""
            if not name.startswith("실제 jevreel"):  # 위치의 묶음으로 답하는 방식만
                hit = sum(t in truth[it["id"]] for t, it in times)
                total = sum(len(v) for v in truth.values())
                row.update(covered=round(hit / total, 3), position_precision=round(hit / len(times), 3) if times else None,
                           f1=round(2 * hit / (total + len(times)), 3))
                cover = f"{row['covered']:.3f} · {row['position_precision']:.3f} (F1 {row['f1']:.3f})"
            diff = ""
            if name != BASE:
                row["diff_found"] = list(boot(found[name], found[BASE], [it["video"] for it, _, _ in spans])[:3])
                diff = f"{row['diff_found'][0]:+.3f} [{row['diff_found'][1]:+.3f}, {row['diff_found'][2]:+.3f}]"
            table[name] = row
            crowd = " / ".join(f"{int(a)}/{n}" for a, n in buckets)
            prec = "-" if row["precision"] is None else f"{row['precision']:.3f}"
            print(f"{name:28s}{row['found']:>7d}/{len(spans)}{row['found_long']:>7d}/{len(long_ones)}{len(times):>9d}({right}){prec:>10s}"
                  f"{row['false_alarm_asks']:>6d}/{len(empty)}{row['tokens']:>13,d}{row['token_ratio']:>6.2f}배   {crowd:26s}   {cover:28s}   {diff}")
        missed = [(it["video"], w) for (it, w, _), f in zip(spans, found[BASE]) if not f and w[1] - w[0] >= 3.0]
        print(f"jevreel(0.5 이상)이 못 찾은 3초 이상 구간 {len(missed)}개. 예: " + ", ".join(f"{v[:8]} {w[0]:.1f}~{w[1]:.1f}초" for v, w in missed[:6]))
        per_ask = [(len(it["windows"]), int(sum(any(near(t, w) for t in answers[BASE][it["id"]][0]) for w in it["windows"])),
                    int(sum(any(near(t, w) for t in answers[OURS_C][it["id"]][0]) for w in it["windows"]))) for it in its]
        print("물음별 (정답 구간 수, jevreel이 찾은 수, C가 찾은 수): " + " ".join(f"{a}:{b}:{c}" for a, b, c in per_ask))
        out["kinds"][kind or "all"] = {"rows": table, "jevreel_missed_3s_or_longer": [[v, w] for v, w in missed], "per_ask": per_ask}
    (HERE / SETS[tag]["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
