"""12칸 격자의 칸마다 프레임 4장을 2×2로 넣은 시트를 견준다. 표준 라이브러리만 쓴다.

자료: Charades 12개 영상, TimeLens-Bench의 고친 질문 43개(동작)
견주는 것
- 60칸 기준      : 1초 간격 60칸 시트, 칸마다 프레임 한 장
- 12칸 한 장     : 12칸 격자 + 창 겹치기, 칸마다 프레임 한 장 (REPORT_motion.md의 변형 1)
- 12칸 묶음      : 위와 같은 창, 칸마다 그 시각 −0.75, −0.25, +0.25, +0.75초의 프레임 4장을 2×2로
같은 4장을 다르게 보여 준 것 (이미 재 둔 것)
- 60칸 겹침      : 60칸 시트의 칸에 4장을 한 장으로 겹친 그림
- 원본 크기 4장  : 위치마다 4장을 원본 크기로 따로 넣은 요청 (4장 조건)
- 원본 크기 한 장: 위치마다 한 장

보는 것 (실행 전에 정함)
- 주: 칸마다의 항목별 AUC에서 12칸 묶음 − 12칸 한 장, 95% 구간(영상 단위 다시 뽑기). 가리킨 한 점이 정답 안인 수
- 함께: 움직임을 봐야 하는 항목(원본 한 장은 못 잡고 4장만 잡음) 중 맞힌 수, 다른 방식과의 AUC 차이,
        60칸 기준·12칸 한 장과 견준 살림과 놓침, 요청당 토큰

실행: python3 experiments/frame_limit/scripts/sheet/bundle_eval.py
결과: 표 출력, results/bundle_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from blend_eval import tl_items
from lev_probe_eval import boot
from motion_eval import answers, inside
from qv_eval import auc
from relate_eval import tile_scores

HERE = E.HERE
SHEETS = [("60칸 기준", "data/sheets60.json", "results/tl_sheet60_preds.json"),
          ("12칸 한 장", "data/sheets_grid12.json", "results/tl_sheet_grid12_preds.json"),
          ("12칸 묶음", "data/sheets_grid12x4.json", "results/tl_sheet_grid12x4_preds.json")]


def main():
    items = [it for it in tl_items("sta") if any(it["y"]) and not all(it["y"])]
    videos = [it["video"] for it in items]
    top = lambda ps: max(range(len(ps)), key=ps.__getitem__)
    motion = [it["id"] for it in items if not it["y"][top(it["one"])] and it["y"][top(it["win4"])]]
    ans, tokens = {}, {}
    for name, index_file, preds_file in SHEETS:
        ans[name], tokens[name] = answers(items, index_file, preds_file)
    blend, blend_tokens = tile_scores(items, "sta", "results/tl_sheet60_blend_preds.json")

    # 칸마다: 위치별 점수 곡선과 가리킨 한 점
    curves = {name: {it["id"]: ans[name][it["id"]]["tiles_curve"] for it in items} for name, _, _ in SHEETS}
    curves["60칸 겹침"] = blend
    curves["원본 크기 4장"] = {it["id"]: it["win4"] for it in items}
    curves["원본 크기 한 장"] = {it["id"]: it["one"] for it in items}
    hit = {name: {it["id"]: bool(it["y"][top(c[it["id"]])]) for it in items} for name, c in curves.items()}
    per = {name: [auc(c[it["id"]], it["y"]) for it in items] for name, c in curves.items()}
    order = ["60칸 기준", "12칸 한 장", "12칸 묶음", "60칸 겹침", "원본 크기 한 장", "원본 크기 4장"]
    print(f"===== 칸마다 (Noul): 질문 {len(items)}개, 그중 움직임을 봐야 하는 것 {len(motion)}개 =====")
    print(f"{'방식':12s}{'AUC':>7s}{'가리킨 곳이 정답':>11s}{'움직임 항목':>9s}")
    out = {"items": len(items), "motion_items": len(motion), "tiles": {}, "pick": {}}
    for name in order:
        row = {"auc": round(sum(per[name]) / len(per[name]), 3), "hit": sum(hit[name].values()),
               "motion_hit": sum(hit[name][i] for i in motion)}
        out["tiles"][name] = row
        print(f"{name:12s}{row['auc']:>7.3f}{row['hit']:>8d}/{len(items)}{row['motion_hit']:>8d}/{len(motion)}")
    print("AUC 차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
    diffs = {}
    for a, b in (("12칸 묶음", "12칸 한 장"), ("12칸 묶음", "60칸 기준"), ("12칸 묶음", "60칸 겹침"), ("원본 크기 4장", "12칸 묶음"),
                 ("12칸 한 장", "60칸 기준")):
        m, lo, hi, _, _ = boot(per[a], per[b], videos)
        diffs[f"{a} − {b}"] = [m, lo, hi]
        print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out["tiles_diff"] = diffs
    print("12칸 묶음이 다른 방식과 견줘 살린 것 / 놓친 것 (가리킨 한 점)")
    swaps = {}
    for other in ("60칸 기준", "12칸 한 장", "원본 크기 4장"):
        gained = sum(hit["12칸 묶음"][i] and not hit[other][i] for i in hit[other])
        lost = sum(not hit["12칸 묶음"][i] and hit[other][i] for i in hit[other])
        swaps[other] = [gained, lost]
        print(f"  {other}은 놓쳤는데 묶음은 맞힘 {gained}개, {other}은 맞혔는데 묶음은 놓침 {lost}개")
    out["tiles_swaps"] = swaps

    print(f"\n===== 고르기 (jevreel 방식) =====")
    pick_hit = {name: {it["id"]: inside(ans[name][it["id"]]["pick_t"], it) for it in items} for name, _, _ in SHEETS}
    pick_auc = {name: [auc(ans[name][it["id"]]["pick_curve"], it["y"]) for it in items] for name, _, _ in SHEETS}
    for name, _, _ in SHEETS:
        row = {"auc": round(sum(pick_auc[name]) / len(items), 3), "hit": sum(pick_hit[name].values()),
               "motion_hit": sum(pick_hit[name][i] for i in motion)}
        out["pick"][name] = row
        print(f"{name:12s}{row['auc']:>7.3f}{row['hit']:>8d}/{len(items)}{row['motion_hit']:>8d}/{len(motion)}")
    for a, b in (("12칸 묶음", "12칸 한 장"), ("12칸 묶음", "60칸 기준")):
        m, lo, hi, _, _ = boot(pick_auc[a], pick_auc[b], videos)
        gained = sum(pick_hit[a][i] and not pick_hit[b][i] for i in pick_hit[a])
        lost = sum(not pick_hit[a][i] and pick_hit[b][i] for i in pick_hit[a])
        out["pick"][f"{a} − {b}"] = {"auc_diff": [m, lo, hi], "gained": gained, "lost": lost}
        print(f"  {a} − {b}: AUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}], 살림 {gained}개, 놓침 {lost}개")
    print("\n요청 하나의 입력 토큰(고르기 / 칸마다): " + ", ".join(f"{n} {tokens[n]['pick']:,.0f} / {tokens[n]['tiles']:,.0f}" for n, _, _ in SHEETS)
          + f", 60칸 겹침 칸마다 {blend_tokens:,.0f}")
    out["tokens"] = {n: {k: round(v) for k, v in tokens[n].items()} for n, _, _ in SHEETS}
    (HERE / "results" / "bundle_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
