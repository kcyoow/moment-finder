"""1초 간격 프레임은 모두 그대로 두고, 움직임이 큰 곳에만 프레임을 더 넣은 60칸 시트를 만든다. Jev 요청은 없다.

사용자 제안(2026-11-11): "움직임 있는 프레임은 늘리는" 방식. 앞서 잰 것은 같은 장수를 옮긴 것이라 조용한 구간이 비었다.
여기서는 아무것도 빼지 않고 시트의 빈칸(60칸 중 22~38칸만 차 있음)에 프레임을 더 넣는다. 따른 논문은 없다.

정한 규칙 (답을 보기 전에 정함)
- 기본: 1초 간격 프레임 전부 (기준 시트와 같은 파일)
- 더 넣는 장수: 시트가 60칸이 되도록 (60 − 1초 프레임 수)
- 어디에: 이웃한 1초 프레임 사이의 1초 구간마다 움직임을 잰다(화면 전체 이동을 맞춘 뒤의 프레임 차이, 8fps. motion_sheets.py가
  이미 계산해 둔 값의 합). 프레임을 한 장씩, "구간의 움직임 ÷ (그 구간에 이미 더 넣은 장수 + 1)"이 가장 큰 구간에 준다.
  한 구간에 최대 3장(0.25초 간격까지). 구간 안에서는 고르게 놓는다(1장이면 가운데, 2장이면 1/3·2/3, 3장이면 1/4·1/2·3/4)
- 시트의 설명은 간격이 고르지 않은 시트의 것을 쓴다("움직임이 큰 곳에 프레임이 더 있다", 칸마다 시각을 소수 첫째 자리까지)

실행: <venv>/bin/python experiments/frame_limit/scripts/motion/madd_sheets.py [묶음]     (기본 sta. data/motion_select.json에 그 영상의 움직임 값이 있어야 한다)
결과: data/sheets_madd.json, data/sheets_madd/…jpg, data/motion_frames/…jpg(더 넣은 프레임)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import sys

from PIL import Image

import sheet_run as R
from motion_sheets import frame_at, save_sheet

HERE = R.HERE
TILES, MAX_EXTRA = 60, 3


def extra_times(centers, times, energy, count):
    """1초 구간마다의 움직임에 따라 더 넣을 프레임의 시각을 정한다."""
    spans = list(zip(centers, centers[1:]))
    motion = [sum(e for t, e in zip(times, energy) if a < t <= b) for a, b in spans]
    given = [0] * len(spans)
    for _ in range(min(count, MAX_EXTRA * len(spans))):
        k = max((i for i in range(len(spans)) if given[i] < MAX_EXTRA), key=lambda i: motion[i] / (given[i] + 1))
        given[k] += 1
    out = []
    for (a, b), n in zip(spans, given):
        out += [round(a + (b - a) * j / (n + 1), 3) for j in range(1, n + 1)]
    return out, given


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "sta"
    frames_index = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
    select = json.loads((HERE / "data" / "motion_select.json").read_text())
    sheets = []
    for video in sorted(frames_index):
        v, row = frames_index[video], select[video]
        centers = v["centers"]
        extras, given = extra_times(centers, row["times"], row["mshift"]["energy"], TILES - len(centers))
        tiles = sorted([(t, v["frames"][f"{t:.2f}"], True) for t in centers] + [(t, frame_at(video, t), False) for t in extras])
        with Image.open(HERE / tiles[0][1]) as im:
            lay = R.choose_layout(im.width / im.height, TILES)
        ids = R.tile_ids(TILES)[:len(tiles)]
        file, size = save_sheet("madd", video, 0, [p for _, p, _ in tiles], ids, lay)
        sheets.append({"tag": tag, "video": video, "sheet": 0, "file": file, "layout": lay, "size": size, "step_seconds": None,
                       "uneven": True, "marks": False, "base_frames": len(centers), "extra_frames": len(extras),
                       "tiles": [{"id": tid, "pos": n, "t": t, "frame": p, "base": base} for n, (tid, (t, p, base)) in enumerate(zip(ids, tiles))]})
        dense = sum(g == MAX_EXTRA for g in given)
        print(f"{video}: 1초 프레임 {len(centers)}장 + 더 넣은 {len(extras)}장 = {len(tiles)}칸. 구간 {len(given)}개 중 더 넣은 구간 "
              f"{sum(g > 0 for g in given)}개(0.25초 간격까지 간 구간 {dense}개)", flush=True)
    path = HERE / "data" / "sheets_madd.json"
    kept = [x for x in json.loads(path.read_text())["sheets"] if x["tag"] != tag] if path.exists() else []
    note = "1초 간격 프레임 전부 + 움직임이 큰 구간에 더 넣은 프레임. pos는 시트 안의 순서, t는 초, base는 1초 간격 프레임인지"
    path.write_text(json.dumps({"_note": note, "sheets": kept + sheets}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
