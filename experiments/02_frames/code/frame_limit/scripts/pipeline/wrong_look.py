""""정답 밖"으로 채점된 곳을 눈으로 보기 위한 그림을 만든다. Jev 요청은 없다.

QVHighlights(qvm)에서 시트 칸마다 0.5 이상으로 낸 위치 가운데 정답 구간의 앞뒤 0.5초 밖인 것, 그리고 실제 jevreel이 낸 시각 가운데
정답 밖인 것을 모아, 질문마다 그림 한 장으로 만든다. 우리 실수인지 정답 표시가 빠진 것인지를 사람이 보고 가리기 위한 것이다.

정한 것 (그림을 보기 전에 정함)
- 우리 쪽: 정답 밖 위치를 이어진 덩어리로 묶고, 덩어리마다 가운데 프레임 한 장. 질문마다 긴 덩어리부터 최대 8곳
- jevreel 쪽: 정답 밖 시각을 2초 안의 것끼리 한 곳으로 묶고, 가장 가까운 0.5초 위치의 프레임 한 장. 질문마다 최대 2곳
- 견줄 그림: 정답 구간의 가운데 프레임(긴 구간부터 최대 4곳)
- 꼬리표: GT = 정답 구간(초), W = 우리가 낸 정답 밖 덩어리(가운데 시각, 칸 수, 평균 확률, 가장 가까운 정답 구간까지의 초), J = jevreel
- 함께 세는 것: 정답 밖 위치가 가장 가까운 정답 구간에서 얼마나 떨어져 있는가(2초 이하 / 2~5초 / 5초 초과)

실행: <venv>/bin/python experiments/frame_limit/scripts/pipeline/wrong_look.py [qvm]
결과: data/wrong_look/<묶음>/<번호>_<항목>.jpg, results/wrong_look_<묶음>.json(그림에 넣은 곳의 목록)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mc_all_eval import BASE, OURS_A, near, read, reported, runs_of

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
COLS, TILE_W, BAR = 5, 300, 20
MAX_OURS, MAX_JEV, MAX_GT = 8, 2, 4
COLORS = {"GT": (20, 110, 40), "W": (170, 60, 20), "J": (40, 70, 160)}


def distance(t, windows):
    return min(max(s - t, t - e, 0.0) for s, e in windows)


def tile(path, label, kind, tile_h, font):
    im = Image.open(HERE / path).convert("RGB")
    im.thumbnail((TILE_W, tile_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (TILE_W, tile_h + BAR), COLORS[kind])
    out.paste(im, ((TILE_W - im.width) // 2, BAR))
    ImageDraw.Draw(out).text((4, 2), label, fill=(255, 255, 255), font=font)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "qvm"
    items = read(f"data/{tag}_items.json")["items"]
    frames = read(f"data/{tag}_frames.json")
    answers = reported(items, tag)
    out_dir = HERE / "data" / "wrong_look" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    probs = {}
    sheets = [s for s in read("data/sheets60.json")["sheets"] if s["tag"] == tag]
    tiles = {(r["item"], r["sheet"]): r for r in read(f"results/sheet60_{tag}_preds.json")["records"] if r["kind"] == "tiles"}
    for it in items:
        probs[it["id"]] = {t["t"]: tiles[(it["id"], sh["sheet"])]["nouls"][t["id"]]
                           for sh in sheets if sh["video"] == it["video"] for t in sh["tiles"]}
    bins = {"ours": [0, 0, 0], "jevreel": [0, 0, 0]}
    listing = []
    for n, it in enumerate(items, 1):
        v = frames[it["video"]]
        centers, wins = v["centers"], it["windows"]
        frame_at = lambda t: v["frames"][f"{min(centers, key=lambda c: abs(c - t)):.2f}"]
        ours = set(answers[OURS_A][it["id"]][0])
        wrong = [t in ours and not any(near(t, w) for w in wins) for t in centers]
        for name, times in (("ours", [t for t, f in zip(centers, wrong) if f]),
                            ("jevreel", [t for t in answers[BASE][it["id"]][0] if not any(near(t, w) for w in wins)])):
            for t in times:
                d = distance(t, wins)
                bins[name][0 if d <= 2 else 1 if d <= 5 else 2] += 1
        cells = []
        for s, e in sorted(wins, key=lambda w: w[0] - w[1])[:MAX_GT]:
            cells.append(("GT", (s + e) / 2, f"GT {s:g}-{e:g}s"))
        groups = sorted(runs_of(wrong), key=lambda g: (-len(g), g[0]))
        for k, g in enumerate(sorted(groups[:MAX_OURS], key=lambda g: g[0]), 1):
            t = centers[g[len(g) // 2]]
            p = sum(probs[it["id"]][centers[i]] for i in g) / len(g)
            cells.append(("W", t, f"W{k} {t:g}s x{len(g)} p{p:.2f} d{distance(t, wins):.0f}"))
            listing.append({"n": n, "item": it["id"], "kind": "ours", "label": f"W{k}", "t": t, "positions": len(g),
                            "mean_prob": round(p, 3), "seconds_from_answer": round(distance(t, wins), 1)})
        places = []
        for t in [t for t in answers[BASE][it["id"]][0] if not any(near(t, w) for w in wins)]:
            if not places or t - places[-1][-1] > 2:
                places.append([t])
            else:
                places[-1].append(t)
        for k, g in enumerate(places[:MAX_JEV], 1):
            t = g[len(g) // 2]
            cells.append(("J", t, f"J{k} {t:g}s d{distance(t, wins):.0f}"))
            listing.append({"n": n, "item": it["id"], "kind": "jevreel", "label": f"J{k}", "t": t, "positions": len(g),
                            "seconds_from_answer": round(distance(t, wins), 1)})
        with Image.open(HERE / frame_at(centers[0])) as im:
            tile_h = round(TILE_W * im.height / im.width)
        rows = -(-len(cells) // COLS)
        head = 24
        canvas = Image.new("RGB", (COLS * TILE_W, head + rows * (tile_h + BAR)), (25, 25, 25))
        ImageDraw.Draw(canvas).text((4, 3), f"{n}. {it['sentence']}"[:150], fill=(255, 255, 255), font=font)
        for i, (kind, t, label) in enumerate(cells):
            canvas.paste(tile(frame_at(t), label, kind, tile_h, font), ((i % COLS) * TILE_W, head + (i // COLS) * (tile_h + BAR)))
        canvas.save(out_dir / f"{n:02d}_{it['id']}.jpg", format="JPEG", quality=85)
        print(f"{n:2d} {it['id']:7s} 정답 구간 {len(wins)}개, 우리 정답 밖 위치 {sum(wrong)}개(덩어리 {len(groups)}개, 그림에 {min(len(groups), MAX_OURS)}곳), "
              f"jevreel 정답 밖 {len(places)}곳  {it['sentence'][:60]}")
    for name, b in bins.items():
        total = sum(b) or 1
        print(f"{name}: 정답 밖 {sum(b)}개 중 정답 구간에서 2초 이하 {b[0]}({b[0] / total:.0%}), 2~5초 {b[1]}({b[1] / total:.0%}), 5초 초과 {b[2]}({b[2] / total:.0%})")
    (HERE / "results" / f"wrong_look_{tag}.json").write_text(json.dumps({"distance_bins": bins, "places": listing}, ensure_ascii=False, indent=1) + "\n")
    print(f"그림 {len(items)}장: {out_dir.relative_to(HERE)}/, 우리 쪽 {sum(x['kind'] == 'ours' for x in listing)}곳, jevreel 쪽 {sum(x['kind'] == 'jevreel' for x in listing)}곳")


if __name__ == "__main__":
    main()
