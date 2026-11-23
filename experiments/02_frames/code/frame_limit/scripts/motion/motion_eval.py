"""시트 변형들이, 기준 시트가 놓쳤던 항목을 얼마나 살리는지 견준다. 표준 라이브러리만 쓴다.

자료: Charades 12개 영상, TimeLens-Bench의 고친 질문 43개(동작)
견주는 것
- 기준        : 1초 간격 60칸 시트 한 장 (results/tl_sheet60_preds.json. 놓친 항목 목록을 뽑았던 그 실행)
- 대조군      : 기준을 그대로 한 번 더 보낸 것. 같은 요청도 답이 조금씩 달라지므로, 변형이 "살린" 수는 이것과 견줘야 한다
- 12칸 격자   : 12칸 격자 + 창 겹치기 (T2SGrid식)
- 프레임 차이 : 프레임 차이로 같은 장수를 다시 배정 (MGSampler)
- 전체 이동   : 화면 전체 이동을 맞춘 뒤의 프레임 차이로 다시 배정 (내 제안)
- 점 추적     : 점 추적으로 낸 움직임 에너지로 다시 배정 (Motion-as-Prompt의 표본 선택)
- 점 추적+궤적: 위와 같은 프레임에 궤적을 그림 (Motion-as-Prompt 전체)
묻는 방법 두 가지: 고르기(jevreel) = 있는가가 가장 높은 시트에서 고르기 확률이 가장 높은 칸, 칸마다 = 예 확률이 가장 높은 칸
(창이 겹치는 12칸 격자의 칸마다 점수는 그 위치가 들어간 창들의 평균)

보는 것 (실행 전에 정함)
- 가리킨 한 점의 시각이 정답 구간 안인가. 기준이 놓쳤던 항목 중 살린 수, 맞혔던 항목 중 놓친 수, 전체 맞힌 수
- 놓쳤던 항목 중 움직임을 봐야 하는 것(원본 한 장은 못 잡고 4장만 잡음)만 따로
- 기준과 대조군이 모두 놓친 항목(우연이 아닌 놓침)만 따로
- 1초 격자에서의 AUC(고르지 않은 시트는 칸 점수를 시각으로 이어서 1초 위치의 값을 읽음)와 기준과의 차이, 95% 구간(영상 단위 다시 뽑기)
- 정답 구간에 칸이 하나도 없는 항목 수, 요청당 토큰, 영상당 요청 수
- 결과를 본 뒤 더한 것: 정답 구간 앞뒤 0.5초까지 맞힌 것으로 치는 수. 정답의 시작·끝이 정수 초라 1초 간격 칸은 양 끝에 정확히
  걸리는데(칸 하나가 앞뒤 0.5초를 대표), 고르지 않은 칸은 끝에서 0.1초만 벗어나도 놓친 것이 되기 때문이다

실행: python3 experiments/frame_limit/scripts/motion/motion_eval.py
결과: 표 출력, results/motion_eval.json, results/motion_items.md(항목별 표)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from blend_eval import read, tl_items
from lev_probe_eval import boot
from qv_eval import auc

HERE = E.HERE
TAG = "sta"
VARIANTS = [("기준", "data/sheets60.json", "results/tl_sheet60_preds.json"),
            ("대조군", "data/sheets60.json", "results/tl_sheet60_rerun_preds.json"),
            ("12칸 격자", "data/sheets_grid12.json", "results/tl_sheet_grid12_preds.json"),
            ("프레임 차이", "data/sheets_mdiff.json", "results/tl_sheet_mdiff_preds.json"),
            ("전체 이동", "data/sheets_mshift.json", "results/tl_sheet_mshift_preds.json"),
            ("점 추적", "data/sheets_mtrack.json", "results/tl_sheet_mtrack_preds.json"),
            ("점 추적+궤적", "data/sheets_mtrackmark.json", "results/tl_sheet_mtrackmark_preds.json")]
KINDS = [("pick", "고르기 (jevreel 방식)"), ("tiles", "칸마다 (Noul)")]


def inside(t, it, slack=0.0):
    return any(s - slack <= t <= e + slack for s, e in it["windows"])


def at_centers(times, scores, centers):
    """칸 점수를 시각 순으로 이어서 1초 위치의 값을 읽는다(바깥은 가장 가까운 칸의 값)."""
    out = []
    for c in centers:
        if c <= times[0]:
            out.append(scores[0])
        elif c >= times[-1]:
            out.append(scores[-1])
        else:
            k = next(i for i in range(1, len(times)) if times[i] >= c)
            w = (c - times[k - 1]) / (times[k] - times[k - 1])
            out.append(scores[k - 1] * (1 - w) + scores[k] * w)
    return out


def answers(items, index_file, preds_file):
    """항목마다 두 방법이 가리킨 시각, 1초 위치의 점수, 정답 구간에 든 칸의 수."""
    sheets = [s for s in read(index_file)["sheets"] if s["tag"] == TAG]
    data = read(preds_file)
    assert data["meta"]["errors"] == 0, f"{preds_file}: 오류가 남은 요청이 있다"
    recs = {(r["item"], r["sheet"], r["kind"]): r for r in data["records"] if r["tag"] == TAG}
    tokens = {k: [r["input_tokens"] for r in data["records"] if r["tag"] == TAG and r["kind"] == k] for k, _ in KINDS}
    out = {}
    for it in items:
        mine = sorted((s for s in sheets if s["video"] == it["video"]), key=lambda s: s["sheet"])
        per_time = {}  # 시각 → {"pick": [...], "tiles": [...]} (창이 겹치면 여러 값)
        best_sheet = None
        for sh in mine:
            a, b = recs[(it["id"], sh["sheet"], "pick")], recs[(it["id"], sh["sheet"], "tiles")]
            probs = a["choice"]["probabilities"]
            if best_sheet is None or a["nouls"]["shown"] > best_sheet[0]:
                top = max(sh["tiles"], key=lambda t: probs[t["id"]])
                best_sheet = (a["nouls"]["shown"], top["t"])
            for t in sh["tiles"]:
                cell = per_time.setdefault(t["t"], {"pick": [], "tiles": []})
                cell["pick"].append(a["nouls"]["shown"] * probs[t["id"]])
                cell["tiles"].append(b["nouls"][t["id"]])
        times = sorted(per_time)
        mean = {k: [sum(per_time[t][k]) / len(per_time[t][k]) for t in times] for k in ("pick", "tiles")}
        top_tiles = times[max(range(len(times)), key=mean["tiles"].__getitem__)]
        out[it["id"]] = {"pick_t": best_sheet[1], "tiles_t": top_tiles, "shown": best_sheet[0],
                         "pick_curve": at_centers(times, mean["pick"], it["ts"]),
                         "tiles_curve": at_centers(times, mean["tiles"], it["ts"]),
                         "tiles_in_gt": sum(inside(t, it) for t in times), "sheets": len(mine)}
    return out, {k: sum(v) / len(v) for k, v in tokens.items()}


def main():
    items = [it for it in tl_items(TAG) if any(it["y"]) and not all(it["y"])]
    videos = [it["video"] for it in items]

    def peak_hit(ps, it):
        return bool(it["y"][max(range(len(ps)), key=ps.__getitem__)])

    motion = {it["id"] for it in items if not peak_hit(it["one"], it) and peak_hit(it["win4"], it)}
    have = [(name, i, p) for name, i, p in VARIANTS if (HERE / p).exists() and (HERE / i).exists()]
    ans, tokens = {}, {}
    for name, index_file, preds_file in have:
        ans[name], tokens[name] = answers(items, index_file, preds_file)
    names = [n for n, _, _ in have]
    out = {"items": len(items), "motion_items": len(motion), "variants": names, "kinds": {}}
    table = {}
    for kind, title in KINDS:
        hit = {n: {it["id"]: inside(ans[n][it["id"]][f"{kind}_t"], it) for it in items} for n in names}
        missed = [it["id"] for it in items if not hit["기준"][it["id"]]]
        kept = [it["id"] for it in items if hit["기준"][it["id"]]]
        missed_motion = [i for i in missed if i in motion]
        stable = [i for i in missed if not hit["대조군"][i]]
        loose = {n: {it["id"]: inside(ans[n][it["id"]][f"{kind}_t"], it, 0.5) for it in items} for n in names}
        per = {n: [auc(ans[n][it["id"]][f"{kind}_curve"], it["y"]) for it in items] for n in names}
        print(f"\n===== {title}: 질문 {len(items)}개. 기준이 놓친 것 {len(missed)}개(그중 움직임을 봐야 하는 것 {len(missed_motion)}개, "
              f"대조군도 놓친 것 {len(stable)}개), 맞힌 것 {len(kept)}개 =====")
        print(f"{'방식':10s}{'전체 맞힘':>8s}{'놓쳤던 것 중 살림':>12s}{'움직임 항목':>8s}{'두 번 놓친 것':>9s}{'맞혔던 것 중 놓침':>12s}{'AUC':>7s}  기준과의 AUC 차이 (95% 구간)")
        rows = {}
        for n in names:
            m, lo, hi, _, _ = boot(per[n], per["기준"], videos) if n != "기준" else (0.0, 0.0, 0.0, 0, 0)
            rows[n] = {"hit": sum(hit[n].values()), "recovered": sum(hit[n][i] for i in missed),
                       "recovered_motion": sum(hit[n][i] for i in missed_motion),
                       "recovered_stable": sum(hit[n][i] for i in stable), "broken": sum(not hit[n][i] for i in kept),
                       "auc": round(sum(per[n]) / len(per[n]), 3), "auc_diff": None if n == "기준" else [m, lo, hi],
                       "hit_within_half_second": sum(loose[n].values()),
                       "recovered_within_half_second": sum(loose[n][i] for i in missed),
                       "broken_within_half_second": sum(not loose[n][i] for i in kept),
                       "no_tile_in_gt": sum(ans[n][it["id"]]["tiles_in_gt"] == 0 for it in items),
                       "tokens_per_request": round(tokens[n][kind]),
                       "requests_per_item": round(sum(ans[n][it["id"]]["sheets"] for it in items) / len(items), 2)}
            r = rows[n]
            diff = "" if n == "기준" else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(f"{n:10s}{r['hit']:>6d}/{len(items)}{r['recovered']:>9d}/{len(missed)}{r['recovered_motion']:>7d}/{len(missed_motion)}"
                  f"{r['recovered_stable']:>7d}/{len(stable)}{r['broken']:>9d}/{len(kept)}{r['auc']:>9.3f}  {diff}")
        print("  앞뒤 0.5초까지 치면 (전체 맞힘, 살림, 놓침): " + ", ".join(
            f"{n} {rows[n]['hit_within_half_second']}·{rows[n]['recovered_within_half_second']}·{rows[n]['broken_within_half_second']}" for n in names))
        new = [n for n in names if n not in ("기준", "대조군")]
        any_new = sum(any(hit[n][i] for n in new) for i in missed)
        any_stable = sum(any(hit[n][i] for n in new) for i in stable)
        print(f"  변형 중 하나라도 살린 것: 놓쳤던 {len(missed)}개 중 {any_new}개, 두 번 놓친 {len(stable)}개 중 {any_stable}개")
        out["kinds"][kind] = {"title": title, "missed": len(missed), "missed_motion": len(missed_motion), "stable_missed": len(stable),
                              "kept": len(kept), "rows": rows, "any_variant_recovered": any_new, "any_variant_recovered_stable": any_stable}
        table[kind] = {"hit": hit, "missed": missed, "stable": set(stable)}
    print("\n정답 구간에 칸이 하나도 없는 항목 수: " + ", ".join(f"{n} {out['kinds']['tiles']['rows'][n]['no_tile_in_gt']}" for n in names))
    print("요청 하나의 입력 토큰(고르기 / 칸마다)과 질문 하나에 드는 시트 수: "
          + ", ".join(f"{n} {tokens[n]['pick']:,.0f} / {tokens[n]['tiles']:,.0f}, {out['kinds']['tiles']['rows'][n]['requests_per_item']}장" for n in names))
    (HERE / "results" / "motion_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    lines = ["# 기준 시트가 놓쳤던 항목을 변형들이 맞혔는가", "",
             "motion_eval.py가 만든 표. O = 가리킨 시각이 정답 구간 안, X = 밖. 괄호는 가리킨 시각(초).",
             "움직임 = 원본 한 장은 못 잡고 4장만 잡은 항목. 두 번 = 대조군(기준을 그대로 다시 보냄)도 놓친 항목.", ""]
    by_id = {it["id"]: it for it in items}
    for kind, title in KINDS:
        lines += [f"## {title}", "", "| 질문 | 정답(초) | 움직임 | 두 번 | " + " | ".join(names) + " |", "|---|---|---|---|" + "---|" * len(names)]
        for i in table[kind]["missed"]:
            it = by_id[i]
            s, e = it["windows"][0]
            cells = [f"{'O' if table[kind]['hit'][n][i] else 'X'} ({ans[n][i][f'{kind}_t']:g})" for n in names]
            lines.append(f"| {it['video'][4:]}: {it['sentence']} | {s:g}~{e:g} | {'예' if i in motion else ''} | "
                         f"{'예' if i in table[kind]['stable'] else ''} | " + " | ".join(cells) + " |")
        lines.append("")
    (HERE / "results" / "motion_items.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
