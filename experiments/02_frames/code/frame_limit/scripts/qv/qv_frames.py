"""QVHighlights 영상에서 판정용 프레임을 뽑는다.

판정 위치는 2초 간격(1, 3, ..., 149초)으로, 주석의 2초 단위 조각 가운데에 맞춘다. 위치마다
- 한 장: 그 시각의 프레임 1장
- 창: 그 시각을 가운데에 둔 4장 (-0.75, -0.25, +0.25, +0.75초)
을 쓴다. 영상이 길어서 프레임을 하나씩 찾지 않고, 영상마다 한 번에 초당 4장을 뽑은 뒤 필요한 것만 남긴다.

실행: <venv>/bin/python experiments/frame_limit/scripts/qv/qv_frames.py [--tag qv2] [--centers-only]
      --stride 1 --first 0.5   1초 간격(0.5, 1.5, …초)으로 뽑는다 (시트 60칸용)
결과: data/<tag>_frames/<영상 id>/<밀리초>.jpg, data/<tag>_frames.json  (loc_frames.json과 같은 형식)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from extract_frames import FFMPEG, fit

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
FPS = 4
STRIDE, FIRST = 2.0, 1.0
WINDOW_OFFSETS = [-0.75, -0.25, 0.25, 0.75]
TAIL_MARGIN = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qv")
    ap.add_argument("--centers-only", action="store_true", help="한 장 조건만 쓸 때. 창에 쓰는 프레임은 뽑지 않는다")
    ap.add_argument("--stride", type=float, default=STRIDE, help="판정 위치의 간격(초)")
    ap.add_argument("--first", type=float, default=FIRST, help="첫 판정 위치(초)")
    args = ap.parse_args()
    out_root = HERE / "data" / f"{args.tag}_frames"
    meta = json.loads((HERE / "data" / f"{args.tag}_items.json").read_text())
    index = {}
    for vid, v in meta["videos"].items():
        centers, c = [], args.first
        while c + WINDOW_OFFSETS[-1] <= v["duration"] - TAIL_MARGIN:
            centers.append(c)
            c += args.stride
        times = sorted(set(centers) | (set() if args.centers_only else
                                       {round(c + o, 2) for c in centers for o in WINDOW_OFFSETS}))
        out = out_root / vid
        out.mkdir(parents=True, exist_ok=True)
        frames = {f"{t:.2f}": out / f"{round(t * 1000):06d}.jpg" for t in times}
        if not all(p.exists() for p in frames.values()):
            with tempfile.TemporaryDirectory() as tmp:
                cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(HERE / v["file"]),
                       "-vf", f"fps={FPS}", "-q:v", "2", str(Path(tmp) / "%05d.jpg")]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg 실패 {vid}: {proc.stderr.strip()[:200]}")
                for t, dest in zip(times, frames.values()):
                    src = Path(tmp) / f"{round(t * FPS) + 1:05d}.jpg"  # n번째 출력 프레임의 시각은 (n-1)/FPS초
                    shutil.copyfile(src, dest)
                    fit(dest)
        index[vid] = {"duration": v["duration"], "centers": centers, "window_offsets": WINDOW_OFFSETS,
                      "frames": {t: str(p.relative_to(HERE)) for t, p in frames.items()}}
        print(f"{vid[:40]:40s} {v['duration']:6.2f}s  위치 {len(centers)}개, 프레임 {len(frames)}장")
    (HERE / "data" / f"{args.tag}_frames.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print(f"\n위치 {sum(len(x['centers']) for x in index.values())}개, 프레임 {sum(len(x['frames']) for x in index.values())}장")


if __name__ == "__main__":
    main()
