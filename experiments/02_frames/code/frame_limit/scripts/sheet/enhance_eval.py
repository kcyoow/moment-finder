"""시트에 넣기 전의 이미지 처리가 칸마다 방식에 도움이 되는지 본다. 표준 라이브러리만 쓴다.

견주는 것 (격자, 칸 크기, 질문 문구는 모두 같다)
- 기준            : 처리 없음. 시트를 JPEG(품질 90)로 저장
- 밝기·대비 보정 : 프레임마다 가장 어두운 1%·가장 밝은 1%를 잘라 범위를 펴고, 그래도 어두우면 감마로 밝힘 (--enhance levels)
- 윤곽 살리기     : 칸 크기로 줄인 뒤 언샤프 마스크(반경 1, 80%) (--enhance sharpen)
- 손실 없는 저장  : 시트를 PNG로 저장 (--enhance png)

자료: TimeLens-Bench의 고친 질문. Charades 12개 영상(동작, 1초 간격), QVHighlights 본 10개 영상(모습, 2초 간격)
보는 것 (실행 전에 정함): 항목별 AUC의 차이와 95% 구간(영상 단위 다시 뽑기), 가장 높은 칸이 정답인 수,
        어두운 영상(평균 밝기 60 미만: ECW3E, D0AGO)과 나머지를 나눈 AUC

실행: python3 experiments/frame_limit/scripts/sheet/enhance_eval.py
결과: 표 출력, results/enhance_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from blend_eval import tl_items
from lev_probe_eval import boot
from qv_eval import auc
from relate_eval import tile_scores, top_of

HERE = E.HERE
FILES = {"기준": "results/tl_sheet60_preds.json", "밝기·대비 보정": "results/tl_sheet60_enhlevels_preds.json",
         "윤곽 살리기": "results/tl_sheet60_enhsharpen_preds.json", "손실 없는 저장": "results/tl_sheet60_enhpng_preds.json"}
DARK = {"sta_ECW3E", "sta_D0AGO"}  # 프레임 평균 밝기(0~255)가 60 미만인 영상. 요청을 보내기 전에 재서 정함


def main():
    out = {}
    for title, tag in (("Charades (동작, 1초 간격)", "sta"), ("QVHighlights 본 영상 (모습, 2초 간격)", "qv")):
        items = [it for it in tl_items(tag) if any(it["y"]) and not all(it["y"])]
        videos = [it["video"] for it in items]
        scores = {name: tile_scores(items, tag, f)[0] for name, f in FILES.items()}
        print(f"\n===== {title}: 질문 {len(items)}개 =====")
        res, per = {}, {}
        for name, sc in scores.items():
            a = [auc(sc[it["id"]], it["y"]) for it in items]
            per[name] = a
            res[name] = {"auc": round(sum(a) / len(a), 3), "peak_hit": sum(top_of(sc[it["id"]], it)["hit"] for it in items)}
            print(f"  {name:10s} AUC {res[name]['auc']:.3f}   가장 높은 칸이 정답 {res[name]['peak_hit']}/{len(items)}")
        diffs = {}
        print("  차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
        for name in list(FILES)[1:]:
            m, lo, hi, n, g = boot(per[name], per["기준"], videos)
            diffs[f"{name} − 기준"] = [m, lo, hi]
            print(f"    {name} − 기준: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        groups = {}
        if tag == "sta":
            print("  영상 밝기별 AUC 평균 (기준 → 밝기·대비 보정)")
            for label, pick in (("어두운 영상 2개", lambda v: v in DARK), ("나머지 10개", lambda v: v not in DARK)):
                keep = [k for k, it in enumerate(items) if pick(it["video"])]
                b = sum(per["기준"][k] for k in keep) / len(keep)
                e = sum(per["밝기·대비 보정"][k] for k in keep) / len(keep)
                hb = sum(top_of(scores["기준"][items[k]["id"]], items[k])["hit"] for k in keep)
                he = sum(top_of(scores["밝기·대비 보정"][items[k]["id"]], items[k])["hit"] for k in keep)
                groups[label] = {"items": len(keep), "base": round(b, 3), "levels": round(e, 3), "base_hit": hb, "levels_hit": he}
                print(f"    {label}(질문 {len(keep)}개): AUC {b:.3f} → {e:.3f}, 가장 높은 칸이 정답 {hb} → {he}")
        out[tag] = {"title": title, "items": len(items), "rows": res, "diff": diffs, "by_brightness": groups}
    (HERE / "results" / "enhance_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
