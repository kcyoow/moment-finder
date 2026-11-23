"""SigLIP 2 비교에 넣을 프레임 목록과 요청문 목록을 만든다. 표준 라이브러리만 쓴다.

Jev로 이미 판정한 네 묶음(loc, qv, qv2, sta)의 프레임과 요청문을 그대로 쓴다.
- frames: 프레임 파일 경로(이 폴더 기준). 한 장 조건의 프레임과 4장 조건의 프레임을 모두 넣는다
- queries: 요청문(Jev 질문에 넣은 문장 그대로). 같은 문장은 한 번만 넣는다

실행: python3 experiments/frame_limit/scripts/embed/siglip_prepare.py
결과: data/siglip_input.json  (siglip_colab.py가 읽는다)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
TAGS = ["loc", "qv", "qv2", "sta"]


def main():
    frames, queries = [], []
    for tag in TAGS:
        index = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
        items = json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]
        used = {it["video"] for it in items}
        paths = sorted({p for video in used for p in index[video]["frames"].values()})
        missing = [p for p in paths if not (HERE / p).exists()]
        if missing:
            raise SystemExit(f"{tag}: 프레임 파일 {len(missing)}개가 없다 (예: {missing[0]})")
        frames.extend(paths)
        for it in items:
            q = it["sentence"]  # loc_run.py가 Jev 질문에 넣은 문장(끝의 마침표 없음)
            if q not in queries:
                queries.append(q)
        print(f"{tag}: 영상 {len(used)}개, 프레임 {len(paths)}장, 요청문 {len(items)}개")
    out = HERE / "data" / "siglip_input.json"
    out.write_text(json.dumps({"frames": frames, "queries": queries}, ensure_ascii=False, indent=1) + "\n")
    print(f"프레임 {len(frames)}장, 서로 다른 요청문 {len(queries)}개 → {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
