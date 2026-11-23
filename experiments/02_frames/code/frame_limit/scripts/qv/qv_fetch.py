"""QVHighlights 검증 세트에서 영상 몇 개를 규칙대로 골라 받는다. 표준 라이브러리만 쓴다.

QVHighlights는 사람이 쓴 요청문과 150초 영상(브이로그·뉴스), 요청에 맞는 구간(2초 단위, 여러 개 가능)으로
이루어진 벤치마크다. 주석은 CC BY-NC-SA 4.0이고 영상은 유튜브에서 온 것이다. 로컬 실험에만 쓰고 다시 배포하지 않는다.

고르는 규칙 (내용을 보고 고르지 않는다)
- 길이 150초, 정답 구간의 합이 4~40초인 항목만 남긴다 (영상 대부분이 정답이면 엉뚱한 곳을 잴 수 없다)
- 고정된 시드로 섞어 앞에서부터 N개. 같은 유튜브 영상에서 나온 조각은 하나만 쓴다

실행: python3 experiments/frame_limit/scripts/qv/qv_fetch.py [N]     (기본 10)
      --offset 10 --tag qv2   앞에서 고른 10개를 건너뛰고 그다음 N개를 data/qv2_items.json에 적는다
      --min-windows 3 --tag qvm   정답 구간이 3개 이상인 항목만 고른다
결과: data/qv_videos/<vid>.mp4, data/<tag>_items.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import hashlib
import json
import random
import urllib.parse
from pathlib import Path

from fetch_videos import http_get, mp4_duration

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
ANNOTATIONS = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
VIDEO_REPO = "https://huggingface.co/datasets/ayushsdev/qvhighlights-videos"
SEED = 20261110
MIN_RELEVANT, MAX_RELEVANT = 4, 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="qv")
    ap.add_argument("--min-windows", type=int, default=1, help="정답 구간이 이 수 이상인 항목만 (여러 번 나오는 장면을 볼 때)")
    args = ap.parse_args()
    n, skip = args.n, args.offset
    rows = [json.loads(line) for line in http_get(ANNOTATIONS).decode().splitlines() if line.strip()]
    pool = [r for r in rows if r["duration"] == 150
            and MIN_RELEVANT <= sum(e - s for s, e in r["relevant_windows"]) <= MAX_RELEVANT
            and len(r["relevant_windows"]) >= args.min_windows]
    pool.sort(key=lambda r: r["qid"])
    random.Random(SEED).shuffle(pool)
    out_dir = HERE / "data" / "qv_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos, items, seen = {}, [], set()
    for r in pool:
        if len(items) == n:
            break
        youtube_id = r["vid"].rsplit("_", 2)[0]
        if youtube_id in seen:
            continue
        seen.add(youtube_id)
        if skip:  # 앞선 실행에서 이미 고른 항목
            skip -= 1
            continue
        dest = out_dir / f"{r['vid']}.mp4"
        if not dest.exists():
            # 미러는 파일 이름 첫 글자를 소문자로 바꾼 폴더에 파일을 나눠 둔다
            shard = urllib.parse.quote(r["vid"][0].lower())
            url = f"{VIDEO_REPO}/resolve/main/{shard}/{urllib.parse.quote(r['vid'])}.mp4"
            dest.write_bytes(http_get(url, timeout=300))
        data = dest.read_bytes()
        vid = "qv_" + r["vid"]
        videos[vid] = {"file": str(dest.relative_to(HERE)), "duration": mp4_duration(dest), "size": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "page": f"https://www.youtube.com/watch?v={youtube_id}"}
        items.append({"id": f"q{r['qid']}", "video": vid, "sentence": r["query"].strip().rstrip("."),
                      "query": r["query"], "windows": r["relevant_windows"],
                      "relevant_clip_ids": r["relevant_clip_ids"], "saliency_scores": r["saliency_scores"]})
        print(f"{r['vid']:32s} {videos[vid]['duration']}s {len(data) / 1e6:5.1f}MB  {r['relevant_windows']}  {r['query']}")
    note = ("QVHighlights 검증 세트에서 규칙으로 고른 항목. windows는 요청에 맞는 구간(초)이다. "
            "주석은 CC BY-NC-SA 4.0, 영상은 유튜브에서 온 것이며 Hugging Face의 제3자 미러에서 받았다. "
            "로컬 실험용이며 다시 배포하지 않는다.")
    meta = {"_note": note, "source": {"annotations": ANNOTATIONS, "videos": VIDEO_REPO},
            "selection": {"seed": SEED, "duration": 150, "relevant_seconds": [MIN_RELEVANT, MAX_RELEVANT],
                          "min_windows": args.min_windows,
                          "pool": len(pool), "offset": args.offset, "picked": len(items)},
            "videos": videos, "items": items}
    (HERE / "data" / f"{args.tag}_items.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"\n후보 {len(pool)}개 중 {len(items)}개, 합계 {sum(v['size'] for v in videos.values()) / 1e6:.0f}MB")


if __name__ == "__main__":
    main()
