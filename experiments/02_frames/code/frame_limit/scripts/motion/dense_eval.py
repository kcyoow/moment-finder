"""화면이 갑자기 바뀐 곳의 프레임을 더 넣은 시트의 채점(마인크래프트, "창이 열려 있다"). 표준 라이브러리만 쓴다. Jev 요청은 없다.

견주는 것
- 1초 간격만 : results/mc_sheet60_preds.json (시트 60장, 받아 둔 답)
- 더 넣음     : results/gui_sheet_mcdense_preds.json (1초 간격 + 분당 10장, 시트 72장. dense_sheets.py)
둘 다 칸마다 확률 0.5 이상인 칸의 시각을 "낸 시각"으로 본다.

지표 (실행 전에 정함)
- 찾은 구간: 낸 시각이 구간의 앞뒤 0.5초 안(앞선 76/89와 같은 기준), 그리고 구간 안(엄격). 1초 미만·이상으로 나눠서
- 낸 시각 중 맞는 수(앞뒤 0.5초, 그리고 구간 안)
- 더 넣은 칸만: 창이 열린 순간인 칸 중 0.5 이상인 비율, 닫힌 순간인 칸 중 0.5 이상인 비율
- 물음(영상)당 토큰과 배수. 찾은 비율의 차이는 영상 단위 다시 뽑기

실행: python3 experiments/frame_limit/scripts/motion/dense_eval.py
결과: 표 출력, results/dense_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from pathlib import Path

from lev_probe_eval import boot

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CUT, SLACK = 0.5, 0.5


def read(path):
    return json.loads((HERE / path).read_text())


def load(index_file, preds_file, items):
    """항목 → [(시각, 확률, 1초 간격 프레임인가)], 토큰."""
    sheets = [s for s in read(index_file)["sheets"] if s["tag"] == "mc"]
    recs = {(r["item"], r["sheet"]): r for r in read(preds_file)["records"] if r["kind"] == "tiles" and not r["error"]}
    out = {}
    for it in items:
        rows, tokens = [], 0
        for sh in (s for s in sheets if s["video"] == it["video"]):
            r = recs[(it["id"], sh["sheet"])]
            tokens += r["input_tokens"]
            rows += [(t["t"], r["nouls"][t["id"]], t.get("base", True)) for t in sh["tiles"]]
        out[it["id"]] = (sorted(rows), tokens)
    return out


def main():
    items = [it for it in read("data/mc_items.json")["items"] if it["kind"] == "gui"]
    methods = {"1초 간격만": load("data/sheets60.json", "results/mc_sheet60_preds.json", items),
               "더 넣음 (분당 10장)": load("data/sheets_mcdense.json", "results/gui_sheet_mcdense_preds.json", items)}
    spans = [(it, w) for it in items for w in it["windows"]]
    short = [k for k, (_, w) in enumerate(spans) if w[1] - w[0] < 1.0]
    longer = [k for k in range(len(spans)) if k not in short]
    inside = lambda t, w, slack: w[0] - slack <= t <= w[1] + slack
    out = {"intervals": len(spans), "short": len(short), "rows": {}}
    found = {}
    print(f"창이 열린 구간 {len(spans)}개 (1초 미만 {len(short)}개, 1초 이상 {len(longer)}개), 영상 {len(items)}개")
    print(f"{'방식':16s}{'찾은 구간(±0.5초)':>14s}{'1초 미만':>9s}{'1초 이상':>9s}{'찾은 구간(구간 안)':>14s}{'1초 미만':>9s}{'낸 시각':>8s}{'맞음(±0.5초)':>11s}{'맞음(구간 안)':>10s}{'영상당 토큰':>10s}")
    base_tokens = sum(methods["1초 간격만"][it["id"]][1] for it in items) / len(items)
    for name, data in methods.items():
        times = {it["id"]: [t for t, p, _ in data[it["id"]][0] if p >= CUT] for it in items}
        row = {}
        for key, slack in (("loose", SLACK), ("strict", 0.0)):
            f = [float(any(inside(t, w, slack) for t in times[it["id"]])) for it, w in spans]
            found[(name, key)] = f
            row[f"found_{key}"] = int(sum(f))
            row[f"found_{key}_short"] = int(sum(f[k] for k in short))
            row[f"found_{key}_long"] = int(sum(f[k] for k in longer))
            row[f"right_{key}"] = sum(any(inside(t, w, slack) for w in it["windows"]) for it in items for t in times[it["id"]])
        row["reported"] = sum(len(v) for v in times.values())
        row["tokens"] = round(sum(data[it["id"]][1] for it in items) / len(items))
        row["token_ratio"] = round(row["tokens"] / base_tokens, 2)
        out["rows"][name] = row
        print(f"{name:16s}{row['found_loose']:>11d}/{len(spans)}{row['found_loose_short']:>6d}/{len(short)}{row['found_loose_long']:>6d}/{len(longer)}"
              f"{row['found_strict']:>11d}/{len(spans)}{row['found_strict_short']:>6d}/{len(short)}{row['reported']:>9d}{row['right_loose']:>11d}{row['right_strict']:>12d}"
              f"{row['tokens']:>12,d} ({row['token_ratio']:.2f}배)")
    videos = [it["video"] for it, _ in spans]
    for key, label in (("loose", "±0.5초"), ("strict", "구간 안")):
        m, lo, hi, _, _ = boot(found[("더 넣음 (분당 10장)", key)], found[("1초 간격만", key)], videos)
        out[f"diff_{key}"] = [m, lo, hi]
        print(f"찾은 비율의 차이(더 넣음 − 1초 간격만, {label}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    sv = [videos[k] for k in short]
    m, lo, hi, _, _ = boot([found[("더 넣음 (분당 10장)", "strict")][k] for k in short], [found[("1초 간격만", "strict")][k] for k in short], sv)
    out["diff_strict_short"] = [m, lo, hi]
    print(f"1초 미만 구간만(구간 안): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    extra = [(t, p, it) for it in items for t, p, is_base in methods["더 넣음 (분당 10장)"][it["id"]][0] if not is_base]
    opened = [(t, p) for t, p, it in extra if any(inside(t, w, 0.0) for w in it["windows"])]
    closed = [(t, p) for t, p, it in extra if not any(inside(t, w, 0.0) for w in it["windows"])]
    out["extra_tiles"] = {"all": len(extra), "open": len(opened), "open_flagged": sum(p >= CUT for _, p in opened),
                          "closed": len(closed), "closed_flagged": sum(p >= CUT for _, p in closed)}
    e = out["extra_tiles"]
    print(f"더 넣은 칸 {e['all']}개: 창이 열린 순간 {e['open']}개 중 0.5 이상 {e['open_flagged']}개, 닫힌 순간 {e['closed']}개 중 0.5 이상 {e['closed_flagged']}개")
    missed = [(it["video"][:5], w) for (it, w), f in zip(spans, found[("더 넣음 (분당 10장)", "strict")]) if not f]
    print(f"더 넣고도 구간 안에서 못 찾은 구간 {len(missed)}개: " + ", ".join(f"{v} {w[0]:.2f}~{w[1]:.2f}초" for v, w in missed))
    out["still_missed_strict"] = [[v, w] for v, w in missed]
    (HERE / "results" / "dense_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
