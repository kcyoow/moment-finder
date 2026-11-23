"""확인용 새 Charades 영상을 받는다. 지금까지 쓴 12개 영상(설정을 정한 영상)과 겹치지 않는다. 표준 라이브러리만 쓴다.

고르는 규칙 (내용을 보고 고르지 않는다)
- sta_fetch.py와 같은 후보(TVBench action_localization 묶음에 있고 Charades-STA 주석이 있는 영상, 앞선 실험의 5개 제외)를
  같은 시드로 섞는다. 앞의 12개는 이미 쓴 것이므로 그 뒤부터 순서대로 본다
- TimeLens-Bench(사람이 고친 질문과 구간)에 있는 영상만, 앞에서부터 N개
- 질문과 정답 구간은 TimeLens-Bench의 것을 그대로 쓴다

Charades 영상과 주석은 비상업 연구용, TimeLens-Bench는 연구용이며 다시 배포하지 않는다.

실행: python3 experiments/frame_limit/scripts/loc/sta2_fetch.py [N]     (기본 30)
결과: data/sta2_videos/<id>.mp4, data/sta2_items.json, data/tl_sta2_items.json (같은 항목. sheet_run.py의 --prefix tl_ 에 맞춘 이름)
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
from sta_fetch import SEED, STA_FILES, USED

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
ALREADY = 12  # sta_fetch.py가 앞에서부터 쓴 수


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sta = set()
    for url in STA_FILES:
        for line in http_get(url).decode().splitlines():
            if "##" in line:
                sta.add(line.split("##", 1)[0].split()[0])
    refined = json.loads((HERE / "data" / "timelens" / "charades-timelens.json").read_text())
    z = open_remote_zip(TVBENCH + "video/action_localization.zip")
    names = {Path(name).stem: name for name in z.namelist() if name.endswith(".mp4")}
    pool = sorted(v for v in names if v in sta and v not in USED)
    random.Random(SEED).shuffle(pool)
    old = {p.stem for p in (HERE / "data" / "sta_videos").glob("*.mp4")}
    assert set(pool[:ALREADY]) == old, "앞의 12개가 이미 쓴 영상과 다르다. 고르는 규칙이 바뀌었는지 확인한다"
    rest = pool[ALREADY:]
    picked = [v for v in rest if v in refined][:n]
    out_dir = HERE / "data" / "sta2_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos, items = {}, []
    for vid in picked:
        dest = out_dir / f"{vid}.mp4"
        if not dest.exists():
            dest.write_bytes(z.read(names[vid]))  # zipfile이 CRC를 확인한다
        data = dest.read_bytes()
        duration = mp4_duration(dest)
        entry = refined[vid]
        assert len(entry["queries"]) == len(entry["spans"]), vid
        assert abs(entry["duration"] - duration) < 1.0, f"{vid}: 길이가 다르다 ({entry['duration']} / {duration})"
        key = "sta2_" + vid
        videos[key] = {"file": str(dest.relative_to(HERE)), "duration": duration, "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
        for k, (query, span) in enumerate(zip(entry["queries"], entry["spans"]), 1):
            items.append({"id": f"tl_{key}_{k}", "video": key, "sentence": query.strip().rstrip(".").strip(), "query": query,
                          "windows": [[float(span[0]), float(span[1])]]})
    note = ("확인용 새 Charades 영상. sta2_fetch.py가 규칙으로 골랐고, 질문과 구간은 TimeLens-Bench의 것이다(초, 시작·끝 포함). "
            "Charades 영상은 비상업 연구용, TimeLens-Bench는 연구용이며 다시 배포하지 않는다.")
    meta = {"_note": note,
            "source": {"videos": TVBENCH + "video/action_localization.zip",
                       "annotations": "https://huggingface.co/datasets/TencentARC/TimeLens-Bench (charades-timelens.json)"},
            "selection": {"seed": SEED, "pool": len(pool), "skipped_already_used": ALREADY,
                          "not_in_timelens": sum(v not in refined for v in rest[:rest.index(picked[-1]) + 1]), "picked": len(videos)},
            "videos": videos, "items": items}
    for name in ("sta2_items.json", "tl_sta2_items.json"):
        (HERE / "data" / name).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    lengths = sorted(it["windows"][0][1] - it["windows"][0][0] for it in items)
    print(f"후보 {len(pool)}개 중 새 영상 {len(videos)}개, 질문 {len(items)}개, 합계 {sum(v['size'] for v in videos.values()) / 1e6:.0f}MB, "
          f"영상 길이 {min(v['duration'] for v in videos.values()):.0f}~{max(v['duration'] for v in videos.values()):.0f}초, "
          f"구간 길이 중앙값 {lengths[len(lengths) // 2]:g}초, TimeLens에 없어 건너뛴 영상 {meta['selection']['not_in_timelens']}개")


if __name__ == "__main__":
    main()
