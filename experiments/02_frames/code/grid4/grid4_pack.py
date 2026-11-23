"""원본 크기 프레임 4장을 2×2 한 그림으로 붙인 묶음을 만든다. 모델 요청은 없다. Pillow가 필요하다.

질문(사용자, 2026-11-12): 4장을 한 요청에 따로 넣는 것과 한 그림으로 붙여 넣는 것이 같은가.
앞선 2×2 실험(frame_limit/reports/REPORT_bundle.md)은 12칸 시트의 칸 안에 작게 넣은 것이라 "한 그림이라서"와 "작아서"가 섞였다.
여기서는 크기를 그대로 둔다: 768×432 프레임 4장 → 1544×872 그림 한 장(프레임 사이에 검은 띠 8화소). 줄이지 않는다.

- 위치와 프레임은 open_model 묶음의 charades 항목과 똑같다(영상 30개, 위치 959곳, 그 시각 −0.75, −0.25, +0.25, +0.75초의 4장)
- 칸의 순서는 왼쪽 위(가장 이름) → 오른쪽 위 → 왼쪽 아래 → 오른쪽 아래(가장 늦음). 칸에 글자는 넣지 않는다
- 항목의 id, 질문, 정답, 받아 둔 Jev 답(4장을 따로 넣었을 때의 답)은 open_model 묶음의 것을 그대로 옮긴다. 그래서 같은 채점 코드로 견줄 수 있다

읽는 곳: ../frame_limit/data (프레임), ../open_model/pack/manifest.json (항목). 둘 다 고치지 않는다
실행: <Pillow가 있는 파이썬> experiments/grid4/grid4_pack.py
결과: pack/manifest.json, pack/img/grid4/<영상>__<시각>.jpg
"""
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"
OPEN = HERE.parent / "open_model"
OUT = HERE / "pack"
GAP, QUALITY = 8, 95  # 프레임 사이 검은 띠(화소), JPEG 품질(다시 압축하며 잃는 것을 줄이려고 높게 둔다)


def main():
    frames = json.loads((SRC / "data" / "sta2_frames.json").read_text())
    items = [it for it in json.loads((OPEN / "pack" / "manifest.json").read_text())["items"] if it["task"] == "charades"]
    (OUT / "img" / "grid4").mkdir(parents=True, exist_ok=True)
    out, sizes = [], {}
    for it in items:
        paths = [SRC / frames[it["video"]]["frames"][f"{t:.2f}"] for t in it["seconds"]]
        ims = [Image.open(p).convert("RGB") for p in paths]
        w, h = ims[0].size
        if any(im.size != (w, h) for im in ims):
            raise SystemExit(f"프레임 크기가 서로 다름: {it['id']}")
        canvas = Image.new("RGB", (2 * w + GAP, 2 * h + GAP))  # 바탕이 검정이라 띠가 저절로 생긴다
        for k, im in enumerate(ims):
            canvas.paste(im, ((k % 2) * (w + GAP), (k // 2) * (h + GAP)))
        rel = f"img/grid4/{it['video']}__{it['t']}.jpg"
        canvas.save(OUT / rel, "JPEG", quality=QUALITY)
        sizes[canvas.size] = sizes.get(canvas.size, 0) + 1
        out.append({**{k: it[k] for k in ("id", "task", "video", "t", "seconds", "questions", "gt", "jev")},
                    "kind": "grid4", "images": [rel], "frame_size": [w, h]})
    note = ("grid4_pack.py가 만든 묶음. 원본 크기 프레임 4장을 2×2 한 그림으로 붙였다(왼쪽 위 → 오른쪽 위 → 왼쪽 아래 → 오른쪽 아래). "
            "jev는 같은 4장을 한 요청에 따로 넣었을 때 받아 둔 Jev(jev-1.14.0) 확률이다. Charades 프레임은 비상업 연구용 자료라 실험에만 쓴다")
    (OUT / "manifest.json").write_text(json.dumps({"_note": note, "gap": GAP, "quality": QUALITY, "items": out}, ensure_ascii=False) + "\n")
    total = sum(p.stat().st_size for p in (OUT / "img" / "grid4").glob("*.jpg"))
    print(f"항목 {len(out)}개, 그림 크기별 {dict((f'{w}x{h}', n) for (w, h), n in sizes.items())}, 합 {total / 1e6:.0f}MB")
    print("질문 수:", sum(len(it["questions"]) for it in out), "| 영상:", len({it["video"] for it in out}))


if __name__ == "__main__":
    main()
