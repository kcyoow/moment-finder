"""목록(data/videos.json)의 각 영상에서 판정용 프레임을 뽑는다.

영상마다 세 묶음을 만든다. 시각은 영상 길이에 대한 비율로 정한다.
- mid: 가운데 1장
- f4: 고르게 4장
- f8: 고르게 8장
질문에 start, end가 있으면(TVBench의 "언제 일어나나", "순서" 과제) 그 구간에서 같은 세 묶음을
질문별로 따로 뽑아 "bounded"에 넣는다. 벤치마크가 그 구간만 보여 주고 묻기 때문이다.

Jev는 각 변 280~1568px 이미지만 받으므로 긴 변을 768px로 맞추고, 짧은 변이 280px보다 작으면
짧은 변을 280px로 맞춘다. 모든 조건이 같은 프레임을 쓰도록 크기는 한 가지로만 만든다.

실행: <venv>/bin/python experiments/frame_limit/scripts/base/extract_frames.py
      (imageio-ffmpeg, pillow 필요. imageio-ffmpeg가 없으면 PATH의 ffmpeg를 쓴다)
결과: data/frames/<id>/<묶음>_<번호>.jpg, data/frames.json
"""
import json
import subprocess
from pathlib import Path

from PIL import Image

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
FRAMES = HERE / "data" / "frames"
SETS = {"mid": 1, "f4": 4, "f8": 8}
LONG_SIDE, MIN_SIDE, MAX_SIDE = 768, 280, 1568


def grab(video, seconds, dest):
    """seconds 시각의 프레임 한 장을 원본 크기로 뽑는다."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{seconds:.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg 실패 {video.name} @ {seconds:.3f}s: {proc.stderr.strip()[:200]}")


def fit(path):
    """긴 변 768px로 맞추고 Jev 크기 제한 안에 들어오는지 확인한다. (가로, 세로)를 돌려준다."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = LONG_SIDE / max(im.size)
        if min(im.size) * scale < MIN_SIDE:
            scale = MIN_SIDE / min(im.size)
        size = (round(im.width * scale), round(im.height * scale))
        im = im.resize(size, Image.LANCZOS)
        im.save(path, "JPEG", quality=90)
    assert MIN_SIDE <= min(size) and max(size) <= MAX_SIDE, f"크기 제한 밖: {path} {size}"
    return size


def frame_sets(video, out, start, end, prefix=""):
    """start~end 구간에서 세 묶음(mid, f4, f8)을 뽑는다."""
    sets = {}
    for name, n in SETS.items():
        rows = []
        for i in range(n):
            t = round(start + (end - start) * (i + 0.5) / n, 3)
            dest = out / f"{prefix}{name}_{i}.jpg"
            grab(HERE / video["file"], t, dest)
            w, h = fit(dest)
            rows.append({"seconds": t, "file": str(dest.relative_to(HERE)), "w": w, "h": h})
        sets[name] = rows
    return sets


def main():
    videos = json.loads((HERE / "data" / "videos.json").read_text())["videos"]
    index = {}
    for v in videos:
        if not v.get("duration"):
            print("길이 정보 없음, 건너뜀:", v["id"])
            continue
        out = FRAMES / v["id"]
        out.mkdir(parents=True, exist_ok=True)
        index[v["id"]] = frame_sets(v, out, 0.0, v["duration"])
        # TVBench의 일부 과제는 영상의 start~end 구간만 보여 주고 묻는다. 그런 질문은 그 구간에서 따로 뽑는다
        bounded = {}
        for k, q in enumerate(v.get("qa") or []):
            if q.get("start") is not None and q.get("end") is not None:
                bounded[f"mc_{k}"] = frame_sets(v, out, q["start"], min(q["end"], v["duration"]), prefix=f"mc{k}_")
        if bounded:
            index[v["id"]]["bounded"] = bounded
        size = index[v["id"]]["mid"][0]
        print(f"{v['id'][:48]:48s} {v['duration']:6.2f}s  {size['w']}x{size['h']}"
              + (f"  구간 지정 질문 {len(bounded)}개" if bounded else ""))
    (HERE / "data" / "frames.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    total = sum(len(s[name]) for s in index.values() for name in SETS)
    total += sum(len(b[name]) for s in index.values() for b in s.get("bounded", {}).values() for name in SETS)
    print(f"\n{len(index)}개 영상, 프레임 {total}장")


if __name__ == "__main__":
    main()
