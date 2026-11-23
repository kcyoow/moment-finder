"""Charades 영상(TVBench의 action_localization 묶음에 들어 있는 것) 중 새 영상을 규칙대로 골라 받는다.

정답 구간은 Charades-STA 주석(영상 id, 시작, 끝, 문장)에서 가져온다. Charades 영상과 주석은 비상업 연구용이며
다시 배포하지 않는다. 표준 라이브러리만 쓴다.

고르는 규칙 (내용을 보고 고르지 않는다)
- TVBench action_localization.zip에 있고 Charades-STA 주석이 있는 영상 중, 앞선 실험에서 쓴 5개를 뺀다
- 고정된 시드로 섞어 앞에서부터 N개
- 그 영상의 Charades-STA 문장을 모두 항목으로 쓴다. 한 영상에서 구간이 완전히 같은 문장은 첫 문장만 쓴다

실행: python3 experiments/frame_limit/scripts/loc/sta_fetch.py [N]     (기본 12)
결과: data/sta_videos/<id>.mp4, data/sta_items.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import hashlib
import json
import random
import sys
from pathlib import Path

from fetch_videos import TVBENCH, http_get, mp4_duration, open_remote_zip

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
STA_FILES = [
    "https://huggingface.co/datasets/jwnt4/charades-sta-test/resolve/main/charades_sta_test.txt",
    "https://raw.githubusercontent.com/Alvin-Zeng/DRN/master/data/dataset/Charades/Charades_sta_train.txt",
]
USED = {"LR9KL", "QOYH2", "98W87", "NKCXF", "43FG9"}
SEED = 20261110


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sta = {}
    for url in STA_FILES:
        for line in http_get(url).decode().splitlines():
            if "##" not in line:
                continue
            head, sentence = line.split("##", 1)
            vid, start, end = head.split()
            sta.setdefault(vid, []).append((float(start), float(end), sentence.strip()))
    z = open_remote_zip(TVBENCH + "video/action_localization.zip")
    names = {Path(name).stem: name for name in z.namelist() if name.endswith(".mp4")}
    pool = sorted(v for v in names if v in sta and v not in USED)
    random.Random(SEED).shuffle(pool)
    out_dir = HERE / "data" / "sta_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos, items = {}, []
    for vid in pool[:n]:
        dest = out_dir / f"{vid}.mp4"
        if not dest.exists():
            dest.write_bytes(z.read(names[vid]))  # zipfile이 CRC를 확인한다
        data = dest.read_bytes()
        duration = mp4_duration(dest)
        key = "sta_" + vid
        videos[key] = {"file": str(dest.relative_to(HERE)), "duration": duration, "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
        seen = set()
        for start, end, sentence in sta[vid]:
            end = min(end, duration)
            if (start, end) in seen or end <= start:
                continue
            seen.add((start, end))
            items.append({"id": f"{vid}_{len(seen)}", "video": key, "sentence": sentence.rstrip(". "),
                          "query": sentence, "windows": [[start, end]]})
            print(f"{vid} {duration:6.2f}s  {start:5.1f}~{end:5.1f}  {sentence}")
    note = ("TVBench action_localization 묶음에 있는 Charades 영상 중 규칙으로 고른 것. windows는 Charades-STA 주석의 "
            "동작 구간(초)이다. Charades 영상과 주석은 비상업 연구용이며 다시 배포하지 않는다.")
    meta = {"_note": note, "source": {"videos": TVBENCH + "video/action_localization.zip", "annotations": STA_FILES},
            "selection": {"seed": SEED, "pool": len(pool), "picked": len(videos), "excluded": sorted(USED)},
            "videos": videos, "items": items}
    (HERE / "data" / "sta_items.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"\n후보 {len(pool)}개 중 영상 {len(videos)}개, 항목 {len(items)}개, "
          f"합계 {sum(v['size'] for v in videos.values()) / 1e6:.0f}MB")


if __name__ == "__main__":
    main()
