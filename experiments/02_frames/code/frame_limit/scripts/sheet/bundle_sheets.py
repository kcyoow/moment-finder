"""12칸 격자의 칸마다 프레임 한 장 대신, 그 1초 안팎의 프레임 4장을 2×2로 나란히 넣은 시트를 만든다. Jev 요청은 없다.

사용자 제안(2026-11-11): "격자 그대로 12칸인데 그 안에 1초를 더 쪼개서 몇 프레임씩 넣은 묶음을 넣는 방식". 따른 논문은 없다.
- 창과 칸의 위치는 12칸 격자 실험(data/sheets_grid12.json)과 똑같다. 칸의 내용만 다르다
- 칸에 넣는 4장은 4장 조건(Jev에 원본 크기로 따로 넣었던 것)과 같은 프레임이다: 칸의 시각 −0.75, −0.25, +0.25, +0.75초
  그래서 같은 프레임을 세 가지로 견줄 수 있다: 한 장으로 겹침(시트 겹침 실험), 원본 크기로 따로(4장 조건), 작게 나란히(이것)
- 칸 안의 순서는 왼쪽 위(가장 이름) → 오른쪽 위 → 왼쪽 아래 → 오른쪽 아래(가장 늦음)
- 칸과 칸 사이에는 검은 띠를 둬서 묶음이 구별되게 한다. 작은 프레임 하나는 60칸 시트의 칸과 거의 같은 크기다

실행: <venv>/bin/python experiments/frame_limit/scripts/sheet/bundle_sheets.py
결과: data/sheets_grid12x4.json, data/sheets_grid12x4/…jpg
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

from PIL import Image, ImageDraw, ImageFont

import sheet_run as R

HERE = R.HERE
GAP = 4  # 칸 둘레에 두는 검은 띠의 두께(화소). 이웃 칸과 합쳐 8화소가 된다


def compose(cells, ids, lay):
    """cells: 칸마다 프레임 4장의 경로(시간 순서). 칸마다 2×2로 붙이고 칸 번호를 왼쪽 위에 적는다."""
    canvas = Image.new("RGB", (lay["cols"] * lay["tile_w"], lay["rows"] * lay["tile_h"]))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, lay["tile_h"] // 16))
    sub_w, sub_h = (lay["tile_w"] - 2 * GAP) // 2, (lay["tile_h"] - 2 * GAP) // 2
    for slot, (paths, tid) in enumerate(zip(cells, ids)):
        x0, y0 = (slot % lay["cols"]) * lay["tile_w"] + GAP, (slot // lay["cols"]) * lay["tile_h"] + GAP
        for k, path in enumerate(paths):
            im = Image.open(HERE / path).convert("RGB")
            im.thumbnail((sub_w, sub_h), Image.Resampling.LANCZOS)
            cx, cy = x0 + (k % 2) * sub_w, y0 + (k // 2) * sub_h
            canvas.paste(im, (cx + (sub_w - im.width) // 2, cy + (sub_h - im.height) // 2))
        left, top, right, bottom = draw.textbbox((x0 + 3, y0 + 2), tid, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 2), tid, fill=(255, 255, 255), font=font)
    return canvas, (sub_w, sub_h)


def main():
    frames = json.loads((HERE / "data" / "sta_frames.json").read_text())
    base = json.loads((HERE / "data" / "sheets_grid12.json").read_text())["sheets"]
    sheets, sizes = [], set()
    for sh in base:
        v = frames[sh["video"]]
        cells = [[v["frames"][f"{round(t['t'] + o, 2):.2f}"] for o in v["window_offsets"]] for t in sh["tiles"]]
        image, sub = compose(cells, [t["id"] for t in sh["tiles"]], sh["layout"])
        sizes.add(sub)
        file = f"data/sheets_grid12x4/{sh['tag']}/{sh['video']}/sheet-{sh['sheet']}.jpg"
        (HERE / file).parent.mkdir(parents=True, exist_ok=True)
        image.save(HERE / file, format="JPEG", quality=90)
        sheets.append({**sh, "file": file, "size": list(image.size), "bundle": len(v["window_offsets"]),
                       "bundle_offsets": v["window_offsets"],
                       "tiles": [{**t, "frames": c} for t, c in zip(sh["tiles"], cells)]})
    note = "12칸 격자와 같은 창. 칸마다 그 시각 둘레의 프레임 4장을 2×2로 넣음. pos는 1초 간격 위치 번호"
    (HERE / "data" / "sheets_grid12x4.json").write_text(json.dumps({"_note": note, "sheets": sheets}, ensure_ascii=False, indent=1) + "\n")
    print(f"시트 {len(sheets)}장, 칸 {sum(len(s['tiles']) for s in sheets)}개, 작은 프레임 크기(너비, 높이) {sorted(sizes)}")


if __name__ == "__main__":
    main()
