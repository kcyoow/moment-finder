"""1초 간격 프레임에, 화면이 갑자기 바뀐 곳의 프레임을 더 넣은 60칸 시트를 만든다(마인크래프트). Jev 요청은 없다.

1초보다 짧은 장면을 프레임이 건너뛰는 문제에 대한 시험이다. 더 넣을 곳은 dense_sim.py의 "튀는 곳"으로 고른다:
이웃 프레임의 차이(20fps, 96×54 흑백)가 앞뒤 0.5초의 중앙값보다 몇 배 큰 순서로, 1분마다 EXTRA장, 고른 틱의 0.1초 뒤 프레임.
정답 기록은 쓰지 않는다. 따른 논문은 없다(프레임 차이로 표본을 고르는 생각은 MGSampler에 있으나 규칙은 다르다).

시트는 영상의 프레임을 시간 순서로 60칸씩 묶는다. 간격이 고르지 않으므로 sheet_run.py가 그 사실과 칸마다의 시각을 설명에 적는다.
"창이 열려 있다" 질문만 묻도록 항목 파일(data/gui_mc_items.json)도 만든다.

실행: <venv>/bin/python experiments/frame_limit/scripts/motion/dense_sheets.py
      <venv>/bin/python experiments/frame_limit/scripts/sheet/sheet_run.py --index mcdense --tags mc --prefix gui_ --kinds tiles
결과: data/sheets_mcdense.json, data/sheets_mcdense/…jpg, data/dense_frames/…jpg(더 넣은 프레임), data/gui_mc_items.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

from PIL import Image

import sheet_run as R
from dense_sim import FPS, energy, pick, spike
from extract_frames import fit, grab

HERE = R.HERE
TILES, EXTRA, SHIFT = 60, 10, 2


def main():
    meta = json.loads((HERE / "data" / "mc_items.json").read_text())
    frames_index = json.loads((HERE / "data" / "mc_frames.json").read_text())
    sheets = []
    for video in sorted(frames_index):
        v, path = frames_index[video], HERE / meta["videos"][video]["file"]
        e = energy(path)
        base = {round(t * FPS) for t in v["centers"]}
        extra = sorted(pick(spike(e), base, EXTRA, len(e), SHIFT) - base)
        tiles = [(t, v["frames"][f"{t:.2f}"], True) for t in v["centers"]]
        for k in extra:
            t = min(k / FPS, v["duration"] - 0.1)
            rel = f"data/dense_frames/{video}/{round(t * 1000):06d}.jpg"
            if not (HERE / rel).exists():
                (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
                grab(path, t, HERE / rel)
                fit(HERE / rel)
            tiles.append((round(t, 2), rel, False))
        tiles.sort()
        with Image.open(HERE / tiles[0][1]) as im:
            lay = R.choose_layout(im.width / im.height, TILES)
        for s, start in enumerate(range(0, len(tiles), TILES)):
            chunk = tiles[start:start + TILES]
            ids = R.tile_ids(TILES)[:len(chunk)]
            file = f"data/sheets_mcdense/mc/{video}/sheet-{s}.jpg"
            (HERE / file).parent.mkdir(parents=True, exist_ok=True)
            image = R.compose([p for _, p, _ in chunk], ids, lay)
            image.save(HERE / file, format="JPEG", quality=90)
            sheets.append({"tag": "mc", "video": video, "sheet": s, "file": file, "layout": lay, "size": list(image.size),
                           "step_seconds": None, "uneven": True, "marks": False,
                           "tiles": [{"id": tid, "pos": start + n, "t": t, "frame": p, "base": is_base}
                                     for n, (tid, (t, p, is_base)) in enumerate(zip(ids, chunk))]})
        print(f"{video[:24]:24s} 1초 프레임 {len(v['centers'])}장 + 더 넣은 {len(extra)}장 = {len(tiles)}장, 시트 {-(-len(tiles) // TILES)}장", flush=True)
    note = "1초 간격 프레임 + 화면이 갑자기 바뀐 곳의 프레임(분당 10장). pos는 영상 안의 순서, t는 초, base는 1초 간격 프레임인지"
    (HERE / "data" / "sheets_mcdense.json").write_text(json.dumps({"_note": note, "sheets": sheets}, ensure_ascii=False, indent=1) + "\n")
    gui = {**meta, "_note": "mc_items.json에서 '창이 열려 있다' 질문만 남긴 것 (dense_sheets.py)",
           "items": [it for it in meta["items"] if it["kind"] == "gui"]}
    (HERE / "data" / "gui_mc_items.json").write_text(json.dumps(gui, ensure_ascii=False, indent=1) + "\n")
    print(f"시트 {len(sheets)}장, 프레임 {sum(len(s['tiles']) for s in sheets)}장, 질문 {len(gui['items'])}개")


if __name__ == "__main__":
    main()
