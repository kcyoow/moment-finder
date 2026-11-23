"""Charades 12개 영상의 1초 구간마다 그림 두 장을 만든다: 구간의 끝 프레임(plain), 같은 프레임에 그 1초 동안의 이동 경로를 그린 것(path).

생각: 1초 안의 프레임 여러 장을 "선명한 한 장 + 움직인 길"로 줄일 수 있는가(Motion-as-Prompt, arXiv 2608.11655의 경로 그리기를 한 장에 쓰는 것).
논문은 프레임 수를 그대로 두고 경로를 더한다. 한 장만 넣는 것은 논문에 없는 우리 쪽 변형이다.

쓰는 것 (읽기만 한다)
- 점 추적 결과: ../frame_limit/data/motion/pass1_<video>.npz (2026-11-11에 CoTracker3로 만든 것. 1초 구간마다 10×10 격자를 새로 뿌려 그 구간 안에서 추적)
- 영상: ../frame_limit/data/sta_videos (추적 때와 같은 방법으로 프레임을 다시 뽑는다: 긴 변 512, 초당 8장 이하)
- 고르기와 색: ../frame_limit/scripts/motion/motion_lib.py의 marks_for(1초 안에 화면 너비의 3% 넘게 움직인 점 가운데 많이 움직인 순으로 8개), row_color
- 그리기: ../frame_limit/scripts/motion/motion_sheets.py의 draw_marks와 같다(선과 끝점의 원, 원본 위에 70%로 얹음)

실행: <numpy·Pillow·imageio-ffmpeg가 있는 파이썬> traj_make.py   → img/<video>/<끝 프레임 번호>_plain.jpg, _path.jpg, index.json
Charades 영상은 비상업 연구용 자료라 실험에만 쓰고 다시 배포하지 않는다. CoTracker3 가중치는 CC BY-NC 4.0이다.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent / "frame_limit"
sys.path.insert(0, str(FL / "scripts" / "motion"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import motion_lib as L  # noqa: E402


def draw(frame, paths):
    base = Image.fromarray(frame)
    over = base.copy()
    d = ImageDraw.Draw(over)
    width, radius = max(2, round(max(base.size) / 190)), max(4, round(max(base.size) / 85))
    for xy, seen in paths:
        pts = [(float(x), float(y)) for x, y in xy]
        color = L.row_color(xy[int(np.argmax(seen))][1], base.height)
        last = None
        for k in range(len(pts) - 1):
            if seen[k] and seen[k + 1]:
                d.line([pts[k], pts[k + 1]], fill=color, width=width)
                last = pts[k + 1]
        if last is not None:
            d.ellipse((last[0] - radius, last[1] - radius, last[0] + radius, last[1] + radius), fill=color)
    return Image.blend(base, over, 0.7)


def main():
    videos = sorted(p.stem.removeprefix("pass1_") for p in (FL / "data" / "motion").glob("pass1_*.npz"))
    index = []
    for video in videos:
        z = np.load(FL / "data" / "motion" / f"pass1_{video}.npz")
        frames, ts, fps = L.timeline(video)
        assert len(frames) == z["tracks"].shape[0], (video, len(frames), z["tracks"].shape)
        bounds = sorted({int(x) for x in z["seg_start"]} | {len(frames) - 1})
        marks = L.marks_for(z["tracks"], z["visibility"], z["seg_start"], z["hw"], float(z["fps"]), bounds)
        out = HERE / "img" / video
        out.mkdir(parents=True, exist_ok=True)
        for lo, hi in zip(bounds, bounds[1:]):
            paths = marks.get(hi, [])
            Image.fromarray(frames[hi]).save(out / f"{hi:03d}_plain.jpg", "JPEG", quality=92)
            draw(frames[hi], paths).save(out / f"{hi:03d}_path.jpg", "JPEG", quality=92)
            index.append({"video": video, "frame": hi, "t0": round(float(ts[lo]), 3), "t": round(float(ts[hi]), 3), "paths": len(paths),
                          "plain": f"img/{video}/{hi:03d}_plain.jpg", "path": f"img/{video}/{hi:03d}_path.jpg",
                          "size": [int(frames.shape[2]), int(frames.shape[1])]})
        n = [x for x in index if x["video"] == video]
        print(video, f"구간 {len(n)}개, 경로가 그려진 것 {sum(x['paths'] > 0 for x in n)}개, 프레임 {frames.shape[2]}x{frames.shape[1]}, fps {fps:.2f}")
    (HERE / "index.json").write_text(json.dumps({"note": "t0~t = 구간의 처음과 끝(초). 그림은 끝 프레임. paths = 그려진 경로 수(0~8)", "items": index},
                                                ensure_ascii=False, indent=1) + "\n")
    print("합계", len(index), "구간")


if __name__ == "__main__":
    main()
