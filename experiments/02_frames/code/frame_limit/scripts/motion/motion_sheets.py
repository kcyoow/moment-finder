"""시트 변형 네 가지의 표본 선택과 시트 그림을 만든다. Jev 요청은 없다. 대상은 Charades 12개 영상.

변형 (모두 기존 1초 간격 60칸 시트와 견준다)
- grid12    : 12칸 격자 + 창 겹치기. 1초 간격 프레임을 12장씩, 7장씩 옮겨 가며(5장 겹침) 시트 여러 장으로 만든다
              T2SGrid(CVPR 2026)의 설정(격자 12장, 옮김 7)을 따랐다. 다른 점: T2SGrid는 격자 여러 장을 한 번에 모델에 넣고
              구간을 글로 답하게 하지만, 여기서는 jevreel처럼 시트마다 따로 묻는다. 프레임은 원본 크기가 아니라 시트 한 변
              1568px에 맞춰 줄인다(Jev의 이미지 크기 제한). 칸 번호를 그려 넣는다(T2SGrid는 번호를 그리지 않는다).
              마지막 창은 12장이 차도록 앞으로 당긴다
- mdiff     : 프레임 차이로 움직임을 재서 같은 장수를 움직임이 큰 곳에 더 배정 (MGSampler). 60칸 시트 한 장
- mtrack    : 점 추적으로 움직임 에너지를 내서 같은 장수를 배정 (Motion-as-Prompt). motion_track.py pass1의 결과가 있어야 한다
- mtrackmark: mtrack과 같은 프레임에, 앞 표본 프레임 이후의 점 궤적을 그려 넣는다 (Motion-as-Prompt의 궤적 표시). pass2 결과가 있어야 한다
              다른 점: 선 굵기와 끝점 크기는 칸이 작아 키웠다(공식 설정은 굵기 1, 반지름 3. 그대로면 칸 크기로 줄였을 때 안 보인다)
- mshift    : 화면 전체 이동을 맞춘 뒤의 프레임 차이로 같은 장수를 배정 (내 제안)
움직임을 따르는 변형은 모두 같은 8fps 타임라인에서, 기준과 같은 장수(영상의 초 수)를 고른다.

실행: <venv>/bin/python experiments/frame_limit/scripts/motion/motion_sheets.py            (만들 수 있는 변형을 모두 만든다)
결과: data/motion_select.json(고른 프레임과 에너지), data/motion_frames/…jpg, data/sheets_<변형>.json, data/sheets_<변형>/…jpg
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import numpy as np
from PIL import Image, ImageDraw

import motion_lib as L
import sheet_run as R
from extract_frames import fit, grab

HERE = L.HERE
TAG = "sta"
WINDOW, STRIDE = 12, 7
SELECT = HERE / "data" / "motion_select.json"


def frame_at(video, t):
    """t초의 프레임을 기준 프레임과 같은 방법(ffmpeg로 한 장, 긴 변 768px, JPEG 90)으로 뽑는다. 이미 있으면 다시 뽑지 않는다."""
    rel = f"data/motion_frames/{video}/{round(t * 1000):06d}.jpg"
    dest = HERE / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        for back in (0.0, 0.05, 0.1, 0.2, 0.35):  # 영상의 맨 끝 프레임은 그 시각으로 찾으면 안 나올 때가 있어 조금 앞에서 다시 찾는다
            try:
                grab(L.video_file(video), max(t - back, 0.0), dest)
                break
            except RuntimeError:
                if back == 0.35:
                    raise
        fit(dest)
    return rel


def select_all(videos, frames_index):
    """영상마다 타임라인을 읽어 에너지를 내고 변형별로 프레임을 고른다."""
    out = json.loads(SELECT.read_text()) if SELECT.exists() else {}
    for video in videos:
        budget = len(frames_index[video]["centers"])
        frames, ts, fps = L.timeline(video)
        row = {"fps": fps, "frames": len(frames), "budget": budget, "times": [round(float(t), 4) for t in ts]}
        e = L.diff_energy(frames)
        row["mdiff"] = {"energy": [round(float(x), 4) for x in e], "indices": L.cumulative_pick(e, budget)}
        e, shifts = L.shift_energy(frames)
        row["mshift"] = {"energy": [round(float(x), 4) for x in e], "indices": L.cumulative_pick(e, budget),
                         "shifts": [list(s) for s in shifts]}
        p1 = HERE / "data" / "motion" / f"pass1_{video}.npz"
        if p1.exists():
            z = np.load(p1)
            assert len(z["timestamps"]) == len(frames), f"{video}: 추적 결과와 타임라인의 길이가 다르다"
            M, _ = L.track_energy(z["tracks"], z["visibility"], tuple(z["hw"]))
            idx, anchors = L.anchor_peak_pick(M, budget)
            row["track"] = {"energy": [round(float(x), 4) for x in M], "indices": [int(k) for k in idx], "anchors": anchors}
        out[video] = row
        moved = sum(1 for s in shifts if s != (0, 0))
        print(f"{video}: {len(frames)}장({fps:g}fps) 중 {budget}장. 화면 전체 이동이 잡힌 프레임 쌍 {moved}/{len(shifts)}"
              + ("" if "track" in row else ". 추적 결과 없음"), flush=True)
    SELECT.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def draw_marks(video, t, paths, scale):
    """프레임에 궤적(선)과 끝점(채운 원)을 그린 그림을 저장하고 경로를 돌려준다. MaP처럼 원본 위에 70%로 얹는다."""
    base = Image.open(HERE / frame_at(video, t)).convert("RGB")
    over = base.copy()
    d = ImageDraw.Draw(over)
    width, radius = max(2, round(max(base.size) / 190)), max(4, round(max(base.size) / 85))  # 칸 크기로 줄여도 보이게
    for xy, seen in paths:
        pts = [(float(x) * scale, float(y) * scale) for x, y in xy]
        color = L.row_color(xy[int(np.argmax(seen))][1], base.height / scale)
        last = None
        for k in range(len(pts) - 1):
            if seen[k] and seen[k + 1]:
                d.line([pts[k], pts[k + 1]], fill=color, width=width)
                last = pts[k + 1]
        if last is not None:
            d.ellipse((last[0] - radius, last[1] - radius, last[0] + radius, last[1] + radius), fill=color)
    rel = f"data/motion_frames/{video}/marked_{round(t * 1000):06d}.jpg"
    Image.blend(base, over, 0.7).save(HERE / rel, "JPEG", quality=90)
    return rel


def write_index(name, sheets, note):
    path = HERE / "data" / f"sheets_{name}.json"
    path.write_text(json.dumps({"_note": note, "sheets": sheets}, ensure_ascii=False, indent=1) + "\n")
    tiles = sum(len(s["tiles"]) for s in sheets)
    sizes = sorted({(s["layout"]["cols"], s["layout"]["rows"], s["layout"]["tile_w"], s["layout"]["tile_h"]) for s in sheets})
    print(f"{name}: 시트 {len(sheets)}장, 칸 {tiles}개, 격자(열, 행, 칸 너비, 칸 높이) {sizes}")


def save_sheet(name, video, s, paths, ids, lay):
    file = f"data/sheets_{name}/{TAG}/{video}/sheet-{s}.jpg"
    (HERE / file).parent.mkdir(parents=True, exist_ok=True)
    image = R.compose(paths, ids, lay)
    image.save(HERE / file, format="JPEG", quality=90)
    return file, list(image.size)


def main():
    frames_index = L.read_json(f"data/{TAG}_frames.json")
    videos = sorted(frames_index)
    select = select_all(videos, frames_index)

    # 12칸 격자 + 창 겹치기 (1초 간격 프레임 그대로)
    sheets = []
    for video in videos:
        v = frames_index[video]
        centers = v["centers"]
        paths = [v["frames"][f"{t:.2f}"] for t in centers]
        with Image.open(HERE / paths[0]) as im:
            lay = R.choose_layout(im.width / im.height, WINDOW)
        starts = list(range(0, max(len(centers) - WINDOW, 0) + 1, STRIDE))
        if starts[-1] + WINDOW < len(centers):
            starts.append(len(centers) - WINDOW)  # 마지막 창은 12장이 차도록 당긴다
        for s, start in enumerate(starts):
            chunk = list(range(start, min(start + WINDOW, len(centers))))
            ids = R.tile_ids(WINDOW)[:len(chunk)]
            file, size = save_sheet("grid12", video, s, [paths[k] for k in chunk], ids, lay)
            sheets.append({"tag": TAG, "video": video, "sheet": s, "file": file, "layout": lay, "size": size,
                           "step_seconds": centers[1] - centers[0],
                           "tiles": [{"id": tid, "pos": k, "t": centers[k], "frame": paths[k]} for tid, k in zip(ids, chunk)]})
    write_index("grid12", sheets, "12칸 격자, 7칸씩 옮긴 창. pos는 1초 간격 위치 번호")

    # 움직임을 따라 고른 프레임으로 60칸 시트 한 장
    for name, key, marks in (("mdiff", "mdiff", False), ("mshift", "mshift", False), ("mtrack", "track", False),
                             ("mtrackmark", "track", True)):
        if not all(key in select[v] for v in videos):
            print(f"{name}: 추적 결과가 없어 건너뜀")
            continue
        if marks and not all((HERE / "data" / "motion" / f"pass2_{v}.npz").exists() for v in videos):
            print(f"{name}: 궤적용 추적 결과(pass2)가 없어 건너뜀")
            continue
        sheets = []
        for video in videos:
            row = select[video]
            idx = row[key]["indices"]
            times = [row["times"][k] for k in idx]
            paths = [frame_at(video, t) for t in times]
            drawn = 0
            if marks:
                z = np.load(HERE / "data" / "motion" / f"pass2_{video}.npz")
                with Image.open(HERE / paths[0]) as im:
                    scale = im.width / int(z["hw"][1])
                found = L.marks_for(z["tracks"], z["visibility"], z["seg_start"], tuple(z["hw"]), float(z["fps"]), idx)
                for n, k in enumerate(idx):
                    if found.get(k):
                        paths[n] = draw_marks(video, times[n], found[k], scale)
                        drawn += len(found[k])
            with Image.open(HERE / paths[0]) as im:
                lay = R.choose_layout(im.width / im.height, 60)
            ids = R.tile_ids(60)[:len(idx)]
            file, size = save_sheet(name, video, 0, paths, ids, lay)
            sheets.append({"tag": TAG, "video": video, "sheet": 0, "file": file, "layout": lay, "size": size,
                           "step_seconds": None, "uneven": True, "marks": marks, "marks_drawn": drawn,
                           "tiles": [{"id": tid, "pos": n, "t": t, "frame": p} for n, (tid, t, p) in enumerate(zip(ids, times, paths))]})
        write_index(name, sheets, "움직임을 따라 고른 프레임의 60칸 시트. pos는 고른 순서, t는 초")


if __name__ == "__main__":
    main()
