# @title E7 prep: download CC BY test videos from Wikimedia Commons, extract 1 fps frames
"""Needs /content/jev/videos.json (list of {id, title, url, license, ...}).

Frame command = PLAN.md section 4 [A]: first frame of each second + real pts_time + long side 1024.
Writes /content/jev/videos/<id>/%06d.jpg, frames.csv (file, seconds) and /content/jev/video_frames.tgz.
"""
import csv
import re
import subprocess
import tarfile
import urllib.request

UA = "moment-finder-eval/0.1 (personal research script)"
VIDS = WORK / "videos"
VIDS.mkdir(exist_ok=True)
videos = json.loads((WORK / "videos.json").read_text())
VF = ("select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,1)',showinfo,"
      "scale=1024:1024:force_original_aspect_ratio=decrease:force_divisible_by=2")

for v in videos:
    src = VIDS / f"{v['id']}.webm"
    if not src.exists():
        req = urllib.request.Request(v["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            src.write_bytes(r.read())
    out = VIDS / v["id"]
    out.mkdir(exist_ok=True)
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-vf", VF, "-fps_mode", "passthrough",
                           "-q:v", "2", str(out / "%06d.jpg")], capture_output=True, text=True)
    times = [float(t) for t in re.findall(r"pts_time:\s*([0-9.]+)", proc.stderr)]
    files = sorted(out.glob("*.jpg"))
    assert proc.returncode == 0 and len(times) == len(files), (proc.returncode, len(times), len(files), proc.stderr[-500:])
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "seconds"])
        w.writerows((p.name, f"{t:.3f}") for p, t in zip(files, times))
    size = PILImage.open(files[0]).size
    print(f"{v['id']}: {src.stat().st_size / 1e6:.1f} MB -> {len(files)} frames {size}, t = {times[0]:.2f} .. {times[-1]:.2f}s")

with tarfile.open(WORK / "video_frames.tgz", "w:gz") as tar:
    for v in videos:
        tar.add(VIDS / v["id"], arcname=v["id"])
print("tar", (WORK / "video_frames.tgz").stat().st_size / 1e6, "MB")
