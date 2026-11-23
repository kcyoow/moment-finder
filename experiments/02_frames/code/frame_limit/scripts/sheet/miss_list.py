"""시트 60칸의 두 방식(고르기 = jevreel 방식, 칸마다 = 칸별 예/아니오)이 놓친 항목을 나열한다. 표준 라이브러리만 쓴다.

새 요청 없이 저장된 결과만 쓴다. 대상은 두 방식을 모두 잰 세 묶음이다.
- 원래 질문·원래 정답 (Charades 32, QVHighlights 본 10·새 10)
- 고친 질문·고친 정답 (TimeLens-Bench: Charades 44, QVHighlights 본 11·새 10)
- 마인크래프트 (영상 12개 × 질문 2개)

두 방식이 "가리킨 곳"
- 고르기: jevreel의 규칙대로, "있는가" 확률이 가장 높은 시트에서 고르기 확률이 가장 높은 칸
  (AUC 표에서는 위치마다 점수가 필요해 있는가 × 칸 확률을 썼지만, 여기서는 jevreel이 실제로 내놓는 한 점을 본다)
- 칸마다: 예 확률이 가장 높은 칸

"놓쳤다"의 기준 (목록을 뽑기 전에 정함)
- 놓침: 가리킨 곳이 정답 구간 밖이다
- 바로 옆: 놓쳤지만 정답 위치의 바로 이웃(한 칸 차이)이다
- 멀리: 그 밖의 놓침

원본 프레임으로 다시 볼 때 살릴 수 있는가 (다시 보는 쪽이 틀리지 않는다고 쳤을 때의 상한)
- 고르기: jevreel의 기본 검증 범위. 있는가 0.1 이상인 시트 중 상위 6장에서, 고른 칸과 둘째 칸(확률 0.15 이상일 때)의 앞뒤 1.5초
- 칸마다: 예 확률 상위 2칸 / 6칸의 앞뒤 1.5초

놓친 항목마다 원본 프레임 조건(Jev 한 장, Jev 4장)이 같은 항목을 맞혔는지 붙여 까닭을 가른다.
- 원본도 못 잡음: 한 장과 4장(있으면) 모두 최고점이 정답 밖. 질문·정답이 애매하거나 어려운 항목
- 한 장은 잡음  : 원본 한 장은 맞혔다. 시트의 칸이 작거나 시트로 묶어서 생긴 문제
- 4장만 잡음    : 한 장은 못 맞히고 4장만 맞혔다. 움직임을 봐야 하는 항목
(4장을 재지 않은 묶음은 한 장만 본다)

실행: python3 experiments/frame_limit/scripts/sheet/miss_list.py
결과: 표 출력, results/miss_list.json, results/miss_list.md(전체 목록)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from blend_eval import read

HERE = E.HERE
VERIFY_TOP, VERIFY_FLOOR, ALSO_MIN, SPREAD = 6, 0.1, 0.15, 1.5  # jevreel config.py, search.py의 기본값
KNOWN_LABEL = {  # 시트를 눈으로 보고 확인한 정답 문제
    "tl_qv_kE-LxJdtTT0_360.0_510.0_1": "고친 정답이 틀려 보임(원래 구간이 맞음)",
    "tl_qv_abRmll8VM9k_660.0_810.0_1": "고친 정답이 틀려 보임(원래 구간이 맞음)",
    "D0AGO_1": "원래 정답이 틀림(실제는 29~36초). 두 방식이 가리킨 곳이 맞음",
    "RX7TC_1": "19~21초에도 먹는 장면이 있음. 정답에서 빠진 것으로 보임",
    "RX7TC_2": "19~21초에도 먹는 장면이 있음. 정답에서 빠진 것으로 보임",
    "q9987": "같은 장면이 19~30초에도 나옴. 두 방식이 가리킨 곳이 그 장면",
    "MJYTA_1": "22~25초에 그릇을 한 번 더 놓음. 정답에서 빠진 것으로 보임",
    "tl_sta_MJYTA_1": "22~25초에 그릇을 한 번 더 놓음. 정답에서 빠진 것으로 보임",
}


def refined(tag):
    items = read(f"data/tl_{tag}_items.json")["items"]
    frames = read(f"data/{tag}_frames.json")
    table = {}
    for r in read("results/tl_preds.json")["records"]:
        if r["tag"] == f"tl_{tag}" and not r["error"]:
            for item_id, p in r["nouls"].items():
                table[(item_id, r["frames"], r["t"])] = p
    for it in items:
        it["ts"] = frames[it["video"]]["centers"]
        it["y"] = [any(s <= t <= e for s, e in it["windows"]) for t in it["ts"]]
        for cond in ("one", "win4"):
            vals = [table.get((it["id"], cond, t)) for t in it["ts"]]
            it[cond] = None if any(v is None for v in vals) else vals
    return items


def minecraft():
    items = read("data/mc_items.json")["items"]
    frames = read("data/mc_frames.json")
    table = {}
    for r in read("results/mc_preds.json")["records"]:
        for item_id, p in r["nouls"].items():
            table[(item_id, r["t"])] = p
    for it in items:
        it["ts"] = frames[it["video"]]["centers"]
        it["y"] = [any(s <= t <= e for s, e in it["windows"]) for t in it["ts"]]
        it["one"], it["win4"] = [table[(it["id"], t)] for t in it["ts"]], None
    return items


def sheets_of(items, tag, preds_file):
    """항목마다 시트별 답: 있는가 확률, 칸마다 (위치 번호, 고르기 확률, 예 확률)."""
    index = [s for s in read("data/sheets60.json")["sheets"] if s["tag"] == tag]
    recs = {(r["item"], r["sheet"], r["kind"]): r for r in read(preds_file)["records"] if r["tag"] == tag}
    out = {}
    for it in items:
        rows = []
        for sh in sorted((s for s in index if s["video"] == it["video"]), key=lambda s: s["sheet"]):
            a, b = recs[(it["id"], sh["sheet"], "pick")], recs[(it["id"], sh["sheet"], "tiles")]
            probs = a["choice"]["probabilities"]
            rows.append({"shown": a["nouls"]["shown"],
                         "tiles": [(t["pos"], probs[t["id"]], b["nouls"][t["id"]]) for t in sh["tiles"]]})
        out[it["id"]] = rows
    return out


def near(t, windows, spread):
    return any(s <= t + spread and e >= t - spread for s, e in windows)


def describe(top, it):
    gt = [k for k, v in enumerate(it["y"]) if v]
    gap = min(abs(top - k) for k in gt)
    return {"top_t": it["ts"][top], "hit": bool(it["y"][top]), "gap_positions": gap,
            "after_end": top > max(gt), "before_start": top < min(gt)}


def verdict(m):
    return "맞힘" if m["hit"] else "바로 옆" if m["gap_positions"] <= 1 else "멀리"


def jevreel_view(rows, it):
    """jevreel의 규칙: 있는가가 높은 시트부터, 시트마다 고르기 확률이 가장 높은 칸."""
    ranked = sorted(rows, key=lambda r: -r["shown"])
    picks = []
    for r in ranked:
        tiles = sorted(r["tiles"], key=lambda x: -x[1])
        picks.append({"shown": r["shown"], "top": tiles[0][0], "also": tiles[1][0] if len(tiles) > 1 and tiles[1][1] >= ALSO_MIN else None})
    out = describe(picks[0]["top"], it)
    out["shown"] = picks[0]["shown"]
    looked = [p for p in picks if p["shown"] >= VERIFY_FLOOR][:VERIFY_TOP]
    spots = [k for p in looked for k in (p["top"], p["also"]) if k is not None]
    out["covered"] = any(near(it["ts"][k], it["windows"], SPREAD) for k in spots)
    out["frames_checked"] = len(spots) * 7  # 고른 칸마다 0.5초 간격 7장
    return out


def tiles_view(rows, it):
    flat = sorted((x for r in rows for x in r["tiles"]), key=lambda x: -x[2])
    out = describe(flat[0][0], it)
    out["prob"] = flat[0][2]
    for k in (2, 6):
        out[f"covered_top{k}"] = any(near(it["ts"][x[0]], it["windows"], SPREAD) for x in flat[:k])
    out["best_gt_rank"] = next(n for n, x in enumerate(flat, 1) if it["y"][x[0]])
    return out


def frame_view(ps, it):
    return describe(max(range(len(ps)), key=ps.__getitem__), it)


def cause(row):
    if row["one"]["hit"]:
        return "한 장은 잡음"
    if row["win4"] is not None and row["win4"]["hit"]:
        return "4장만 잡음"
    return "원본도 못 잡음"


def span(windows):
    return ", ".join(f"{s:g}~{e:g}" for s, e in windows)


def main():
    groups = [("원래 질문·원래 정답", [("Charades", "sta", E.bundle("sta")), ("QV 본 영상", "qv", E.bundle("qv")),
                                  ("QV 새 영상", "qv2", E.bundle("qv2"))], "results/sheet60_preds.json"),
              ("고친 질문·고친 정답", [("Charades", "sta", refined("sta")), ("QV 본 영상", "qv", refined("qv")),
                                  ("QV 새 영상", "qv2", refined("qv2"))], "results/tl_sheet60_preds.json"),
              ("마인크래프트", [("마인크래프트", "mc", minecraft())], "results/mc_sheet60_preds.json")]
    out, lines = {"groups": {}}, []
    for group, sets, preds in groups:
        rows, absent = [], []
        for label, tag, items in sets:
            detail = sheets_of(items, tag, preds)
            for it in items:
                sh = detail[it["id"]]
                if not any(it["y"]):  # 정답이 없는 영상: 잘못 잡았는지만 본다
                    absent.append({"id": it["id"], "set": label, "query": it.get("sentence") or it["query"],
                                   "shown_max": max(r["shown"] for r in sh),
                                   "tiles_max": max(x[2] for r in sh for x in r["tiles"]), "one_max": max(it["one"])})
                    continue
                if all(it["y"]):  # 정답이 영상 전체: 놓칠 수 없다
                    continue
                row = {"id": it["id"], "set": label, "video": it["video"], "query": it.get("sentence") or it["query"],
                       "gt": span(it["windows"]), "gt_positions": sum(it["y"]), "positions": len(it["y"]),
                       "gt_seconds": round(sum(e - s for s, e in it["windows"]), 2),
                       "pick": jevreel_view(sh, it), "tiles": tiles_view(sh, it), "one": frame_view(it["one"], it),
                       "win4": frame_view(it["win4"], it) if it.get("win4") else None,
                       "label_note": KNOWN_LABEL.get(it["id"])}
                row["cause"] = cause(row)
                rows.append(row)
        n = len(rows)
        print(f"\n===== {group}: 항목 {n}개 =====")
        print(f"{'방식':10s}{'맞힘':>7s}{'바로 옆':>6s}{'멀리':>5s}   놓친 것의 까닭 (원본도 못 잡음 / 한 장은 잡음 / 4장만 잡음)")
        summary = {}
        for name, k in (("고르기", "pick"), ("칸마다", "tiles"), ("Jev 한 장", "one"), ("Jev 4장", "win4")):
            have = [r for r in rows if r[k] is not None]
            if not have:
                continue
            v = [verdict(r[k]) for r in have]
            missed = [r for r in have if not r[k]["hit"]]
            causes = {c: sum(r["cause"] == c for r in missed) for c in ("원본도 못 잡음", "한 장은 잡음", "4장만 잡음")}
            summary[name] = {"items": len(have), "hit": v.count("맞힘"), "near": v.count("바로 옆"), "far": v.count("멀리"), "causes": causes}
            extra = "   " + " / ".join(str(c) for c in causes.values()) if k in ("pick", "tiles") else ""
            print(f"{name:10s}{v.count('맞힘'):>4d}/{len(have):<3d}{v.count('바로 옆'):>5d}{v.count('멀리'):>6d}{extra}")
        pm, tm = [r for r in rows if not r["pick"]["hit"]], [r for r in rows if not r["tiles"]["hit"]]
        both = [r for r in rows if not r["pick"]["hit"] and not r["tiles"]["hit"]]
        only_pick = [r for r in pm if r["tiles"]["hit"]]
        only_tiles = [r for r in tm if r["pick"]["hit"]]
        print(f"둘 다 놓침 {len(both)}개, 고르기만 놓침 {len(only_pick)}개, 칸마다만 놓침 {len(only_tiles)}개")
        fix = {"pick_missed": len(pm), "pick_covered": sum(r["pick"]["covered"] for r in pm),
               "tiles_missed": len(tm), "tiles_covered_top2": sum(r["tiles"]["covered_top2"] for r in tm),
               "tiles_covered_top6": sum(r["tiles"]["covered_top6"] for r in tm),
               "pick_low_shown": sum(r["pick"]["shown"] < 0.5 for r in rows),
               "pick_near_after_end": sum(r["pick"]["after_end"] for r in pm if verdict(r["pick"]) == "바로 옆"),
               "pick_near": sum(verdict(r["pick"]) == "바로 옆" for r in pm),
               "tiles_near_after_end": sum(r["tiles"]["after_end"] for r in tm if verdict(r["tiles"]) == "바로 옆"),
               "tiles_near": sum(verdict(r["tiles"]) == "바로 옆" for r in tm)}
        print(f"다시 보면 살릴 수 있는 것(상한): 고르기 {fix['pick_covered']}/{len(pm)}(jevreel 기본 검증 범위), "
              f"칸마다 {fix['tiles_covered_top2']}/{len(tm)}(상위 2칸 둘레), {fix['tiles_covered_top6']}/{len(tm)}(상위 6칸 둘레)")
        print(f"바로 옆으로 놓친 것 중 정답이 끝난 뒤쪽: 고르기 {fix['pick_near_after_end']}/{fix['pick_near']}, 칸마다 {fix['tiles_near_after_end']}/{fix['tiles_near']}. "
              f"고르기에서 시트의 '있는가'가 0.5 미만인 항목 {fix['pick_low_shown']}/{n}")
        cover = {"pick_top1": sum(r["pick"]["hit"] for r in rows),
                 "pick_verify": sum(r["pick"]["hit"] or r["pick"]["covered"] for r in rows),
                 "tiles_top1": sum(r["tiles"]["hit"] for r in rows),
                 "tiles_top2": sum(r["tiles"]["covered_top2"] for r in rows),
                 "tiles_top6": sum(r["tiles"]["covered_top6"] for r in rows),
                 "either_top1": sum(r["pick"]["hit"] or r["tiles"]["hit"] for r in rows),
                 "union_verify": sum(r["pick"]["hit"] or r["pick"]["covered"] or r["tiles"]["covered_top6"] for r in rows)}
        fix["cover_all"] = cover
        print(f"전체 {n}개 중 정답을 보게 되는 항목(상한): 고르기 한 점 {cover['pick_top1']}, 고르기 + jevreel 검증 범위 {cover['pick_verify']}, "
              f"칸마다 한 점 {cover['tiles_top1']}, 칸마다 상위 2칸 둘레 {cover['tiles_top2']}, 상위 6칸 둘레 {cover['tiles_top6']}, "
              f"둘 중 하나라도 한 점이 맞음 {cover['either_top1']}, 둘을 합친 범위 {cover['union_verify']}")
        short = [r for r in rows if r["gt_seconds"] <= 2]
        longer = [r for r in rows if r["gt_seconds"] > 2]
        by_len = {}
        for label, part in (("정답 2초 이하", short), ("정답 2초 넘음", longer)):
            if part:
                by_len[label] = {"items": len(part), "pick_missed": sum(not r["pick"]["hit"] for r in part),
                                 "tiles_missed": sum(not r["tiles"]["hit"] for r in part)}
                print(f"  {label}: {len(part)}개 중 고르기 {by_len[label]['pick_missed']}개, 칸마다 {by_len[label]['tiles_missed']}개 놓침")
        for a in absent:
            print(f"정답이 없는 항목 [{a['query'][:40]}]: 시트의 있는가 최고 {a['shown_max']:.2f}, 칸마다 최고 {a['tiles_max']:.2f}, Jev 한 장 최고 {a['one_max']:.2f}")
        out["groups"][group] = {"items": n, "summary": summary, "both": len(both), "only_pick": len(only_pick),
                                "only_tiles": len(only_tiles), "fix": fix, "by_length": by_len, "rows": rows, "absent": absent}

        lines += [f"## {group}", "", f"항목 {n}개. 둘 다 놓침 {len(both)}개, 고르기만 놓침 {len(only_pick)}개, 칸마다만 놓침 {len(only_tiles)}개.", ""]
        for title, part in (("둘 다 놓침", both), ("고르기(jevreel 방식)만 놓침", only_pick), ("칸마다만 놓침", only_tiles)):
            if not part:
                continue
            lines += [f"### {title} ({len(part)}개)", "",
                      "| 묶음 | 질문 | 정답(초) | 고르기가 가리킨 곳 | 칸마다가 가리킨 곳 | Jev 한 장 / 4장 | 까닭 | 비고 |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in sorted(part, key=lambda r: (r["set"], r["cause"], r["id"])):
                ref = verdict(r["one"]) + " / " + (verdict(r["win4"]) if r["win4"] else "재지 않음")
                p, t = r["pick"], r["tiles"]
                pick = f"{p['top_t']:g}초 ({verdict(p)}), 있는가 {p['shown']:.2f}" + ("" if p["hit"] else ", 검증 범위에 정답 " + ("있음" if p["covered"] else "없음"))
                tiles = f"{t['top_t']:g}초 ({verdict(t)}), 확률 {t['prob']:.2f}" + ("" if t["hit"] else f", 정답은 {t['best_gt_rank']}번째, 상위 6칸 둘레에 " + ("있음" if t["covered_top6"] else "없음"))
                lines.append(f"| {r['set']} | {r['query']} | {r['gt']} | {pick} | {tiles} | {ref} | {r['cause']} | {r['label_note'] or ''} |")
            lines.append("")
        if absent:
            lines += ["### 정답이 없는 항목에서의 최고 점수", ""]
            lines += [f"- {a['query']}: 시트의 있는가 {a['shown_max']:.2f}, 칸마다 {a['tiles_max']:.2f}, Jev 한 장 {a['one_max']:.2f}" for a in absent]
            lines.append("")
    (HERE / "results" / "miss_list.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    (HERE / "results" / "miss_list.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
