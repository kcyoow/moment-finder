"""모델 없이 내는 무료 기준선: 화면 가운데에서 마인크래프트 창의 바탕색에 가까운 화소가 차지하는 비율.

마인크래프트의 창(인벤토리, 작업대, 화로, 상자)은 바탕이 밝은 회색(빨강·초록·파랑 198)이다.
규칙은 보기 전에 정했고 고치지 않았다: 프레임의 가운데 절반(가로·세로 각 1/4~3/4)에서
세 채널이 모두 198±14 안에 드는 화소의 비율을 그 위치의 점수로 쓴다.
이 규칙은 창이 열렸는지만 볼 뿐 어떤 창인지는 가리지 못한다(인벤토리 질문에도 같은 점수를 쓴다).

실행: <venv>/bin/python experiments/frame_limit/scripts/mc/mc_rule.py   (Pillow, numpy 필요)
결과: results/mc_rule.json  {"scores": {영상: [위치별 점수]}}
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
PANEL, TOLERANCE = 198, 14


def score(path):
    im = np.asarray(Image.open(HERE / path).convert("RGB"), dtype=np.int16)
    h, w, _ = im.shape
    mid = im[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float((np.abs(mid - PANEL) <= TOLERANCE).all(axis=2).mean())


def main():
    frames = json.loads((HERE / "data" / "mc_frames.json").read_text())
    scores = {video: [round(score(v["frames"][f"{t:.2f}"]), 5) for t in v["centers"]] for video, v in frames.items()}
    (HERE / "results" / "mc_rule.json").write_text(json.dumps({
        "_note": "가운데 절반에서 창 바탕색(198±14)에 가까운 화소의 비율. mc_rule.py가 만든다.",
        "panel": PANEL, "tolerance": TOLERANCE, "scores": scores}, ensure_ascii=False) + "\n")
    flat = [s for v in scores.values() for s in v]
    print(f"영상 {len(scores)}개, 위치 {len(flat)}개, 점수 범위 {min(flat):.3f}~{max(flat):.3f}, 0.05 넘는 위치 {sum(s > 0.05 for s in flat)}개")


if __name__ == "__main__":
    main()
