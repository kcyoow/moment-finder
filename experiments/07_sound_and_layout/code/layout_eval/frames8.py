"""마인크래프트 12편·로봇 2편의 원본 영상에서 0.125초 간격 프레임(320×180)을 뽑는다(C 모양용). 돈은 들지 않는다.
프레임 i(0부터)는 t = i/8초로 본다(ffmpeg fps 필터). 실행: uv run --no-project --with imageio-ffmpeg python -I experiments/layout_eval/frames8.py
"""
import json, subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parents[1] / "pipeline/runs"
NAMES = [f"gem_mc_{i:02d}" for i in range(1, 13)] + ["gem_robot_green", "gem_robot_red"]
FF = imageio_ffmpeg.get_ffmpeg_exe()
for nm in NAMES:
    v = json.load(open(RUNS / nm / "run.json"))["video"]
    out = ROOT / "data/f8" / nm; out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.jpg")):
        print(nm, "있음", flush=True); continue
    subprocess.run([FF, "-v", "error", "-hwaccel", "videotoolbox", "-i", v, "-an", "-vf", "fps=8,scale=320:180", "-q:v", "3", str(out / "%06d.jpg")], check=True)
    print(nm, len(list(out.glob("*.jpg"))), flush=True)
