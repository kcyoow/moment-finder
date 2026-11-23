"""위치 찾기 테스트용 프레임을 뽑는다.

판정 위치는 1초 간격(1.0, 2.0, ...초)이다. 위치마다 두 가지 입력을 만든다.
- 한 장: 그 시각의 프레임 1장
- 창: 그 시각을 가운데에 둔 4장 (-0.75, -0.25, +0.25, +0.75초. 0.5초 간격)

실행: <venv>/bin/python experiments/frame_limit/scripts/loc/loc_frames.py [--tag sta]
결과: data/<tag>_frames/<영상 id>/<밀리초>.jpg, data/<tag>_frames.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import json
from pathlib import Path

from extract_frames import fit, grab

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
WINDOW_OFFSETS = [-0.75, -0.25, 0.25, 0.75]
TAIL_MARGIN = 0.1  # 마지막 프레임이 영상 끝에 너무 붙지 않게 둔다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="loc")
    args = ap.parse_args()
    out_root = HERE / "data" / f"{args.tag}_frames"
    meta = json.loads((HERE / "data" / f"{args.tag}_items.json").read_text())
    items = meta["items"]
    # 영상 정보는 항목 파일에 있으면 그것을, 없으면 처음 받은 23개 목록을 쓴다
    videos = meta.get("videos") or {v["id"]: v for v in
                                    json.loads((HERE / "data" / "videos.json").read_text())["videos"]}
    index = {}
    for vid in sorted({it["video"] for it in items}):
        v = videos[vid]
        centers = [float(k) for k in range(1, int(v["duration"]) + 1)
                   if k + WINDOW_OFFSETS[-1] <= v["duration"] - TAIL_MARGIN]
        times = sorted({c for c in centers} | {round(c + o, 2) for c in centers for o in WINDOW_OFFSETS})
        out = out_root / vid
        out.mkdir(parents=True, exist_ok=True)
        frames = {}
        for t in times:
            dest = out / f"{round(t * 1000):06d}.jpg"
            if not dest.exists():
                grab(HERE / v["file"], t, dest)
                fit(dest)
            frames[f"{t:.2f}"] = str(dest.relative_to(HERE))
        index[vid] = {"duration": v["duration"], "centers": centers, "window_offsets": WINDOW_OFFSETS,
                      "frames": frames}
        print(f"{vid[:48]:48s} {v['duration']:6.2f}s  위치 {len(centers)}개, 프레임 {len(frames)}장")
    (HERE / "data" / f"{args.tag}_frames.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    print(f"\n위치 {sum(len(x['centers']) for x in index.values())}개, 프레임 {sum(len(x['frames']) for x in index.values())}장")


if __name__ == "__main__":
    main()
