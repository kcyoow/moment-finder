"""Colab(GPU)에서 SigLIP 2로 프레임과 요청문의 유사도를 계산한다.

입력 (작업 폴더 기준, 기본 /content/mf)
- siglip_input.json: {"frames": [프레임 경로...], "queries": [요청문...]}  (siglip_prepare.py가 만든다)
- data/<묶음>_frames/...: 프레임 파일

출력
- siglip_scores_<모델 줄임말>.json: 모든 프레임 × 모든 요청문의 코사인 유사도, 확률로 바꾸는 계수, 처리 속도

요청문은 처리기가 소문자로 바꾸고 64토큰으로 맞춘다(공식 문서의 권장 설정).
임베딩으로 계산한 값이 모델이 직접 내는 값과 같은지 앞의 몇 장으로 확인한 뒤 저장한다.

실행: python scripts/embed/siglip_colab.py [작업 폴더]
"""
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODELS = {"base": "google/siglip2-base-patch16-224", "so400m": "google/siglip2-so400m-patch14-384"}
BATCH = 32


def features(out):
    """transformers 버전에 따라 텐서 또는 출력 객체가 온다."""
    return out if torch.is_tensor(out) else out.pooler_output


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/mf")
    spec = json.loads((root / "siglip_input.json").read_text())
    frames, queries = spec["frames"], spec["queries"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "없음"
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, 장치 {device} ({gpu})")
    print(f"프레임 {len(frames)}장, 요청문 {len(queries)}개")

    for short, model_id in MODELS.items():
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            text_in = processor(text=queries, padding="max_length", max_length=64, truncation=True,
                                return_tensors="pt").to(device)
            text = torch.nn.functional.normalize(features(model.get_text_features(**text_in)), dim=-1)

            # 점검: 앞의 4장에서 임베딩으로 낸 값과 모델이 직접 낸 값을 비교한다
            probe = [Image.open(root / p).convert("RGB") for p in frames[:4]]
            both = processor(text=queries, images=probe, padding="max_length", max_length=64, truncation=True,
                             return_tensors="pt").to(device)
            direct = model(**both).logits_per_image
            img = torch.nn.functional.normalize(
                features(model.get_image_features(**processor(images=probe, return_tensors="pt").to(device))), dim=-1)
            scale, bias = model.logit_scale.exp().item(), model.logit_bias.item()
            gap = (img @ text.T * scale + bias - direct).abs().max().item()
            print(f"[{short}] 매개변수 {params / 1e6:.0f}M, 계수 scale {scale:.3f} bias {bias:.3f}, 점검 차이 {gap:.5f}")
            if gap > 0.05:
                raise SystemExit(f"{short}: 임베딩 계산이 모델 출력과 다르다 (차이 {gap})")

            rows, started = [], time.time()
            for i in range(0, len(frames), BATCH):
                images = [Image.open(root / p).convert("RGB") for p in frames[i:i + BATCH]]
                emb = features(model.get_image_features(**processor(images=images, return_tensors="pt").to(device)))
                rows.append((torch.nn.functional.normalize(emb, dim=-1) @ text.T).float().cpu())
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.time() - started
        cos = torch.cat(rows)
        out = {"model": model_id, "params_m": round(params / 1e6, 1), "device": gpu,
               "transformers": transformers.__version__, "logit_scale": scale, "logit_bias": bias,
               "seconds_images": round(seconds, 1), "frames_per_second": round(len(frames) / seconds, 1),
               "frames": frames, "queries": queries,
               "cos": [[round(v, 5) for v in row] for row in cos.tolist()]}
        path = root / f"siglip_scores_{short}.json"
        path.write_text(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[{short}] {len(frames)}장 {seconds:.1f}초 (초당 {len(frames) / seconds:.1f}장, 파일 읽기 포함) → {path.name} "
              f"{path.stat().st_size / 1e6:.1f}MB, 유사도 범위 {cos.min():.3f}~{cos.max():.3f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
