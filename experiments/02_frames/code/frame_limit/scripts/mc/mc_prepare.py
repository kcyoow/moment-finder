"""마인크래프트 기록에서 판정용 프레임을 뽑고 항목 파일을 만든다.

위치는 1초 간격이고 각 초의 가운데다(0.5, 1.5, …초 = 틱 10, 30, …). 행동 기록의 i번째 줄이 영상의 i번째 프레임이다
(공식 로더 data_loader.py가 줄마다 프레임을 하나씩 읽는다).

정답은 그 틱의 값 그대로다.
- gui: isGuiOpen (화면 창이 열려 있다. 인벤토리, 작업대, 화로, 상자 등)
- inv: isGuiInventory (그 창이 플레이어의 인벤토리다)
항목의 windows는 값이 참인 틱의 연속 구간을 [첫 틱 시각, 마지막 틱 시각](초, 양끝 포함)으로 적은 것이다.
행동 기록의 그 밖의 필드(마우스, 키보드, 위치, 통계)는 쓰지 않는다.

실행: <venv>/bin/python experiments/frame_limit/scripts/mc/mc_prepare.py   (ffmpeg는 imageio_ffmpeg의 것을 쓴다)
결과: data/mc_frames/<영상>/<밀리초>.jpg, data/mc_frames.json, data/mc_items.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from extract_frames import FFMPEG, fit

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
SRC = HERE / "data" / "mc"
TICKS = 20  # 초당 틱 수 = 초당 프레임 수
FIRST_TICK = 10
QUERIES = {
    "gui": ("isGuiOpen", "an inventory, crafting or other menu window is open on the screen"),
    "inv": ("isGuiInventory", "the player's inventory screen is open, the one that shows the player's character next to the armor slots"),
}


def runs(flags):
    out, start = [], None
    for k, f in enumerate(flags):
        if f and start is None:
            start = k
        elif not f and start is not None:
            out.append([start, k - 1])
            start = None
    if start is not None:
        out.append([start, len(flags) - 1])
    return out


def main():
    chosen = json.loads((SRC / "videos.json").read_text())["videos"]
    index, items, videos = {}, [], {}
    for n, v in enumerate(chosen, 1):
        vid = f"mc_{n:02d}_{v['alias']}"
        flags = {key: [] for key in QUERIES}
        with open(SRC / f"{v['name']}.jsonl") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    for key, (field, _) in QUERIES.items():
                        flags[key].append(bool(d[field]))
        ticks = list(range(FIRST_TICK, len(flags["gui"]), TICKS))
        centers = [t / TICKS for t in ticks]
        out = HERE / "data" / "mc_frames" / vid
        out.mkdir(parents=True, exist_ok=True)
        frames = {f"{c:.2f}": out / f"{round(c * 1000):06d}.jpg" for c in centers}
        if not all(p.exists() for p in frames.values()):
            with tempfile.TemporaryDirectory() as tmp:
                cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(SRC / f"{v['name']}.mp4"),
                       "-vf", f"select='eq(mod(n,{TICKS}),{FIRST_TICK})'", "-fps_mode", "vfr", "-q:v", "2",
                       str(Path(tmp) / "%05d.jpg")]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg 실패 {vid}: {proc.stderr.strip()[:300]}")
                got = sorted(Path(tmp).glob("*.jpg"))
                assert len(got) >= len(centers), f"{vid}: 프레임 {len(got)}장, 위치 {len(centers)}개"
                for src, dest in zip(got, frames.values()):  # k번째 출력 = 틱 10 + 20(k-1)
                    shutil.copyfile(src, dest)
                    fit(dest)
        index[vid] = {"duration": len(flags["gui"]) / TICKS, "centers": centers, "window_offsets": [],
                      "frames": {t: str(p.relative_to(HERE)) for t, p in frames.items()}}
        videos[vid] = {"name": v["name"], "file": f"data/mc/{v['name']}.mp4", "duration": len(flags["gui"]) / TICKS}
        for key, (field, sentence) in QUERIES.items():
            windows = [[a / TICKS, b / TICKS] for a, b in runs(flags[key])]
            positive = sum(flags[key][t] for t in ticks)
            items.append({"id": f"{vid[:5]}_{key}", "video": vid, "kind": key, "field": field, "sentence": sentence,
                          "query": sentence, "windows": windows, "positive_positions": positive})
        print(f"{vid:38s} 위치 {len(centers)}개, 창이 열린 위치 {sum(flags['gui'][t] for t in ticks)}개, "
              f"인벤토리 위치 {sum(flags['inv'][t] for t in ticks)}개")
    (HERE / "data" / "mc_frames.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    (HERE / "data" / "mc_items.json").write_text(json.dumps({
        "_note": "mc_prepare.py가 만든 항목. windows는 [첫 틱 시각, 마지막 틱 시각](초, 양끝 포함).",
        "source": "https://github.com/openai/Video-Pre-Training (contractor data 10.0, MIT)",
        "videos": videos, "items": items}, ensure_ascii=False, indent=1) + "\n")
    none = [it["id"] for it in items if it["positive_positions"] == 0]
    print(f"영상 {len(videos)}개, 항목 {len(items)}개, 위치 {sum(len(x['centers']) for x in index.values())}개. "
          f"정답 위치가 없는 항목 {len(none)}개: {none}")


if __name__ == "__main__":
    main()
