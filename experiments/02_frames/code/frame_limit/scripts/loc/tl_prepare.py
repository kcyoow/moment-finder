"""TimeLens-Bench(사람이 다시 고친 정답)에서 우리가 프레임을 가진 영상의 주석을 뽑아 항목 파일을 만든다. 표준 라이브러리만 쓴다.

TimeLens-Bench(CVPR 2026, HF TencentARC/TimeLens-Bench, 연구용)는 Charades-STA와 QVHighlights의 주석을
사람이 다시 검토해 질문을 고쳐 쓰고 구간을 다시 표시한 자료다. 주석은 영상마다 질문 목록과 구간 하나씩이다
(초 단위, 영상 조각의 처음이 0).

입력
- data/timelens/charades-timelens.json, data/timelens/qvhighlights-timelens.json  (그 저장소의 파일 그대로)
- data/<묶음>_items.json, data/<묶음>_frames.json  (묶음: sta, qv, qv2. 이미 있는 것)

출력
- data/tl_<묶음>_items.json: 항목마다 id, video, sentence(Jev 질문에 넣는 문장. 끝의 마침표를 뗌), query(원문),
  windows([[시작, 끝]]), same_as(원래 항목 중 문장이 글자까지 같은 것의 id, 없으면 null), orig_windows(그 항목의 원래 구간)

쓰지 않는 필드: duration(프레임 목록의 길이와 같은지만 확인한다).

실행: python3 experiments/frame_limit/scripts/loc/tl_prepare.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
SOURCES = {"sta": "charades-timelens.json", "qv": "qvhighlights-timelens.json", "qv2": "qvhighlights-timelens.json"}


def norm(text):
    return text.strip().rstrip(".").strip()


def main():
    for tag, file in SOURCES.items():
        refined = json.loads((HERE / "data" / "timelens" / file).read_text())
        base = json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]
        frames = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
        by_video = {}
        for it in base:
            by_video.setdefault(it["video"], []).append(it)
        items, empty = [], []
        for video, originals in by_video.items():
            key = video.split("_", 1)[1]  # 앞의 묶음 표시(sta_, qv_)를 뗀 것이 원래 영상 이름
            entry = refined.get(key)
            if entry is None:
                raise SystemExit(f"{tag}: {key}가 TimeLens-Bench에 없다")
            assert len(entry["queries"]) == len(entry["spans"]), key
            assert abs(entry["duration"] - frames[video]["duration"]) < 1.0, f"{key}: 길이가 다르다"
            centers = frames[video]["centers"]
            for k, (query, span) in enumerate(zip(entry["queries"], entry["spans"]), 1):
                sentence = norm(query)
                same = [o for o in originals if norm(o["sentence"]) == sentence]
                item = {"id": f"tl_{video}_{k}", "video": video, "sentence": sentence, "query": query,
                        "windows": [[float(span[0]), float(span[1])]],
                        "same_as": same[0]["id"] if same else None,
                        "orig_windows": same[0]["windows"] if same else None}
                if not any(span[0] <= t <= span[1] for t in centers):
                    empty.append(item["id"])
                items.append(item)
        out = {"_note": ("TimeLens-Bench의 주석에서 뽑은 항목. tl_prepare.py가 만든다. windows는 초(시작, 끝 포함). "
                         "same_as는 원래 묶음에서 문장이 글자까지 같은 항목."),
               "source": f"https://huggingface.co/datasets/TencentARC/TimeLens-Bench ({file}), 연구용",
               "items": items}
        path = HERE / "data" / f"tl_{tag}_items.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        lengths = sorted(it["windows"][0][1] - it["windows"][0][0] for it in items)
        print(f"{tag}: 영상 {len(by_video)}개, 질문 {len(items)}개(원래 {len(base)}개), 문장이 같은 것 "
              f"{sum(bool(it['same_as']) for it in items)}개, 구간 길이 중앙값 {lengths[len(lengths) // 2]:g}초, "
              f"2초 이하 {sum(x <= 2 for x in lengths)}개 → {path.relative_to(HERE)}")
        if empty:
            print(f"  주의: 정답 구간 안에 판정 위치가 하나도 없는 항목 {empty}")


if __name__ == "__main__":
    main()
