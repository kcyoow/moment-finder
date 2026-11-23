"""통과한 클립과 빠진 후보의 프레임을 한 장에 모아(영상에서 8장씩) 눈으로 확인한다: results/<이름>/cand_<순위>_<통과|빠짐>.jpg

  uv run --no-project --with imageio-ffmpeg --with pillow python experiments/final_qwen_h2o/sheets.py mc_diamond [--all]
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

HERE = Path(__file__).resolve().parent
UPLOADS = HERE.parent.parent / "pipeline" / "uploads"
VIDEO = {"robot_ok": "web_1116_200731_2.mp4", "robot_none": "web_1116_200730_1.mp4", "mc_diamond": "web_1116_201838_3.mp4",
         "mc_creeper": "web_1116_201839_4.mp4", "mc_fight": "web_1116_041133_3.mp4", "mc_mixed": "web_1116_043326_3.mp4"}
FF = imageio_ffmpeg.get_ffmpeg_exe()


def grab(video, t, h=200):
    p = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale=-2:{h}",
                        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"], capture_output=True)
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def main():
    name = sys.argv[1]
    every = "--all" in sys.argv
    st = json.loads((HERE / "results" / name / "status.json").read_text(encoding="utf-8"))
    for c in st["result"]["details"]["candidates"]:
        if not (c.get("match") or every):
            continue
        ts = c["times_a"] + c["times_b"]
        imgs = [grab(UPLOADS / VIDEO[name], t) for t in ts]
        w, h = imgs[0].size
        sheet = Image.new("RGB", (4 * w + 6, 2 * h + 2), "black")
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % 4) * (w + 2), (i // 4) * (h + 2)))
        out = HERE / "results" / name / f"cand_{c['rank']}_{'pass' if c.get('match') else 'drop'}.jpg"
        sheet.save(out, quality=85)
        print(out.name, f"{c['start']}~{c['end']}", "p", c.get("p"), "noul", c.get("p_noul"), "choice", c.get("p_choice"), "seen_as", c.get("seen_as"), "from", c["from"])


if __name__ == "__main__":
    main()
