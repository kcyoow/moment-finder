"""'찾는 장면이 없는 영상' 시험에 쓸 질문 목록을 만든다. 표준 라이브러리만 쓴다.

새 QVHighlights 영상(qv2) 10개 각각에 대해, 다른 영상의 요청문 19개를 붙인다.
- 같은 묶음(qv2)의 다른 영상 요청문 9개
- 앞서 본 묶음(qv)의 요청문 10개
다른 영상의 요청문은 대체로 그 영상에 없는 장면이다(우연히 있을 수는 있다).

실행: python3 experiments/frame_limit/scripts/qv/absent_prepare.py
결과: data/qv2_foreign.json  (loc_run.py --parts-file 로 쓴다)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit


def main():
    own = json.loads((HERE / "data" / "qv2_items.json").read_text())["items"]
    seen = json.loads((HERE / "data" / "qv_items.json").read_text())["items"]
    pool = [(it["id"], it["sentence"]) for it in own + seen]
    parts, source = {}, {}
    for it in own:
        others = [(qid, s) for qid, s in pool if qid != it["id"]]
        parts[it["id"]] = [s for _, s in others]
        source[it["id"]] = [qid for qid, _ in others]
    note = ("영상마다 다른 영상의 요청문을 묻기 위한 목록. parts[영상의 요청 id]의 n번째 문장이 질문 c<n>이 되고, "
            "source가 그 문장이 어느 요청에서 왔는지 알려 준다.")
    (HERE / "data" / "qv2_foreign.json").write_text(
        json.dumps({"_note": note, "parts": parts, "source": source}, ensure_ascii=False, indent=1) + "\n")
    print(f"영상 {len(parts)}개 × 다른 요청문 {len(pool) - 1}개 = 없는 쌍 {len(parts) * (len(pool) - 1)}개")


if __name__ == "__main__":
    main()
