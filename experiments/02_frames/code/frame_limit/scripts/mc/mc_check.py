"""받은 마인크래프트 영상이 행동 기록과 맞는지 확인한다.

행동 기록은 틱(1/20초)마다 한 줄이고 영상은 초당 20장이므로, 영상의 프레임 수와 행동 기록의 줄 수가 같아야
"k번째 줄 = k번째 프레임"으로 정답 시각을 쓸 수 있다.

확인하는 것: 파일 크기(받을 때 적어 둔 값과 같은가), 화면 크기와 초당 프레임 수, 프레임 수와 틱 수의 차이
실행: <venv>/bin/python experiments/frame_limit/scripts/mc/mc_check.py   (imageio_ffmpeg가 필요하다)
결과: 표 출력, data/mc/videos.json에 frames, fps, size, frame_minus_ticks를 더한다
"""
import json
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
OUT = HERE / "data" / "mc"


def main():
    path = OUT / "videos.json"
    data = json.loads(path.read_text())
    bad = 0
    for v in data["videos"]:
        file = OUT / f"{v['name']}.mp4"
        if not file.exists():
            print(f"  없음 {v['name']}.mp4")
            bad += 1
            continue
        reader = imageio_ffmpeg.read_frames(str(file))
        meta = next(reader)
        reader.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(file))
        v.update(frames=frames, fps=meta["fps"], size=list(meta["size"]), frame_minus_ticks=frames - v["ticks"])
        same_size = file.stat().st_size == v["bytes"]
        ok = same_size and abs(frames - v["ticks"]) <= 2 and abs(meta["fps"] - 20) < 0.01
        bad += not ok
        print(f"  {'맞음' if ok else '확인 필요'} {v['name']}: {meta['size'][0]}x{meta['size'][1]}, 초당 {meta['fps']:g}장, "
              f"프레임 {frames}장, 틱 {v['ticks']}개(차이 {frames - v['ticks']:+d}), 크기 {'같음' if same_size else '다름'}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    total = sum((OUT / f"{v['name']}.mp4").stat().st_size for v in data["videos"] if (OUT / f"{v['name']}.mp4").exists())
    print(f"영상 {len(data['videos'])}개 중 확인 필요 {bad}개, 영상 크기 합 {total / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
