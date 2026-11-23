"""시트의 칸 사이 시간 관계를 JSON에 적었을 때 칸마다 방식이 달라지는지 본다. 표준 라이브러리만 쓴다.

견주는 것 (시트 그림은 모두 같다. 프레임 한 장짜리 60칸)
- 기준    : 지금까지의 방식. 상태에 칸 번호 → 시각, "시간 순서로 놓였다"는 설명
- 상태에  : 상태의 칸 목록에 앞 칸·뒤 칸을 넣고 "동작은 앞뒤 칸과 견줘 판단하라"고 미리 알림 (--relate state)
- 질문에  : 칸마다 질문에 앞 칸·뒤 칸을 넣고 "세 칸의 변화로 판단하라"고 함 (--relate question)
- 문구만  : 대조군. 질문에와 문구·구조는 같고 앞 칸·뒤 칸과 그 설명만 뺌 (--relate wording). 질문에의 결과를 본 뒤 더함

자료: TimeLens-Bench의 고친 질문. Charades 12개 영상(1초 간격, 동작), QVHighlights 본 10개 영상(2초 간격, 모습)
보는 것 (실행 전에 정함): 항목별 AUC의 차이와 95% 구간(영상 단위 다시 뽑기), 가장 높은 칸이 정답 안인 수,
        지난 목록에서 "움직임을 봐야 하는 항목"(한 장은 못 잡고 4장만 잡은 것)을 얼마나 맞히는가, 끝난 직후를 가리키는 수

실행: python3 experiments/frame_limit/scripts/sheet/relate_eval.py
결과: 표 출력, results/relate_eval.json
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
FILES = {"기준": "results/tl_sheet60_preds.json", "상태에": "results/tl_sheet60_relstate_preds.json",
         "질문에": "results/tl_sheet60_relquestion_preds.json", "문구만": "results/tl_sheet60_relwording_preds.json"}


def tile_scores(items, tag, preds_file):
    sheets = [s for s in read("data/sheets60.json")["sheets"] if s["tag"] == tag]
    data = read(preds_file)
    recs = {(r["item"], r["sheet"]): r for r in data["records"] if r["tag"] == tag and r["kind"] == "tiles"}
    out, tokens = {}, []
    for it in items:
        ps = [None] * len(it["ts"])
        for sh in (s for s in sheets if s["video"] == it["video"]):
            r = recs[(it["id"], sh["sheet"])]
            assert not r["error"]
            tokens.append(r["input_tokens"])
            for t in sh["tiles"]:
                ps[t["pos"]] = r["nouls"][t["id"]]
        assert None not in ps
        out[it["id"]] = ps
    return out, sum(tokens) / len(tokens)


def top_of(ps, it):
    top = max(range(len(ps)), key=ps.__getitem__)
    gt = [k for k, v in enumerate(it["y"]) if v]
    return {"hit": bool(it["y"][top]), "near": min(abs(top - k) for k in gt) == 1, "after_end": top > max(gt)}


def main():
    out = {}
    for title, tag in (("Charades (동작, 1초 간격)", "sta"), ("QVHighlights 본 영상 (모습, 2초 간격)", "qv")):
        items = [it for it in tl_items(tag) if any(it["y"]) and not all(it["y"])]
        videos = [it["video"] for it in items]
        scores, tokens = {}, {}
        for name, f in FILES.items():
            scores[name], tokens[name] = tile_scores(items, tag, f)
        motion = [it for it in items if not top_of(it["one"], it)["hit"] and top_of(it["win4"], it)["hit"]]
        print(f"\n===== {title}: 질문 {len(items)}개, 그중 움직임을 봐야 하는 것(한 장은 못 잡고 4장만 잡음) {len(motion)}개 =====")
        print(f"{'방식':10s}{'요청당 토큰':>9s}{'AUC':>7s}{'최고 칸이 정답':>10s}{'끝난 직후':>7s}{'움직임 항목 중 맞힘':>12s}")
        res, per = {}, {}
        rows = {"Jev 한 장": {it["id"]: it["one"] for it in items}, "Jev 4장": {it["id"]: it["win4"] for it in items}, **scores}
        for name, sc in rows.items():
            a = [auc(sc[it["id"]], it["y"]) for it in items]
            tops = [top_of(sc[it["id"]], it) for it in items]
            per[name] = a
            res[name] = {"auc": round(sum(a) / len(a), 3), "peak_hit": sum(t["hit"] for t in tops),
                         "after_end_near": sum(t["near"] and t["after_end"] and not t["hit"] for t in tops),
                         "motion_hit": sum(top_of(sc[it["id"]], it)["hit"] for it in motion),
                         "tokens_per_request": round(tokens[name]) if name in tokens else None}
            tok = f"{tokens[name]:,.0f}" if name in tokens else "-"
            print(f"{name:10s}{tok:>9s}{res[name]['auc']:>7.3f}{res[name]['peak_hit']:>7d}/{len(items)}{res[name]['after_end_near']:>8d}"
                  f"{res[name]['motion_hit']:>9d}/{len(motion)}")
        diffs = {}
        print("차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
        for a, b in (("상태에", "기준"), ("질문에", "기준"), ("문구만", "기준"), ("질문에", "문구만"),
                     ("Jev 4장", "질문에"), ("Jev 4장", "기준")):
            m, lo, hi, n, g = boot(per[a], per[b], videos)
            diffs[f"{a} − {b}"] = [m, lo, hi]
            print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out[tag] = {"title": title, "items": len(items), "motion_items": len(motion), "rows": res, "diff": diffs}
    (HERE / "results" / "relate_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
