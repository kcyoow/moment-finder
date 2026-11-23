"""Charades 영상의 판정 위치마다 LeVJEPA 임베딩을 뽑고 처리 속도를 잰다.

입력은 모델 카드대로 만든다: 16프레임 × 224px, 7.5fps(2초), ImageNet 정규화, (B, C, T, H, W).
판정 위치 t를 가운데에 둔 2초(t-1.0 ~ t+1.0)를 한 클립으로 쓴다.
모델: galilai-group/LeVJEPA-1.5-Large (CC BY-NC 4.0, 비상업). 모델 코드가 가중치와 함께 오므로
trust_remote_code가 필요하다.

위치마다 저장하는 것
- cls        : 클립 전체를 요약한 임베딩 (1024차원)
- patch_mean : 모든 패치 토큰의 평균 (1024차원)
- motion_lev : 프레임별 평균 토큰이 이웃 프레임과 얼마나 다른가 (1 - 코사인 유사도의 평균)
- motion_pix : 이웃 프레임의 픽셀 차이 평균 (싼 기준선)
- pix16      : 가운데 프레임을 16×16 흑백으로 줄인 것 (싼 기준선용 임베딩)

실행: HF_HOME=<폴더> <venv>/bin/python experiments/frame_limit/scripts/embed/lev_embed.py [--limit N] [--cpu-sample]
      (torch, transformers, numpy, imageio-ffmpeg 필요)
결과: results/lev_emb.npz, results/lev_timing.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from extract_frames import FFMPEG

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "galilai-group/LeVJEPA-1.5-Large"
FPS, FRAMES, SIZE, HALF = 7.5, 16, 224, 1.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH = 2


def decode(path):
    """영상을 7.5fps, 224×224(짧은 변을 맞춘 뒤 가운데를 자름)로 읽는다. (프레임 수, 224, 224, 3) uint8"""
    vf = f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)


def clip_indices(t, n):
    start = round((t - HALF) * FPS)
    return np.clip(start + np.arange(FRAMES), 0, n - 1)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def encode(model, clips, device):
    """clips: (B, T, H, W, 3) uint8 → cls, patch_mean, motion_lev"""
    x = (clips.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous().to(device)  # (B, C, T, H, W)
    with torch.no_grad():
        out = model(pixel_values=x)
        cls = out["pooler_output"]
        tokens = out.last_hidden_state[:, 1:].reshape(x.shape[0], FRAMES, -1, cls.shape[-1])
        frame_mean = tokens.mean(2)
        cos = torch.nn.functional.cosine_similarity(frame_mean[:, 1:], frame_mean[:, :-1], dim=-1)
        result = cls.float().cpu().numpy(), tokens.mean((1, 2)).float().cpu().numpy(), (1 - cos).mean(1).cpu().numpy()
    sync(device)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="처음 N개 영상만 (시험용)")
    ap.add_argument("--cpu-sample", action="store_true", help="CPU 속도도 클립 4개로 잰다")
    args = ap.parse_args()
    videos = {}  # 영상 id → (파일, 판정 위치)
    first = {v["id"]: v for v in json.loads((HERE / "data" / "videos.json").read_text())["videos"]}
    for tag in ("sta", "loc"):
        meta = json.loads((HERE / "data" / f"{tag}_items.json").read_text())
        frames = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
        info = meta.get("videos") or first
        for vid, f in frames.items():
            videos[vid] = (HERE / info[vid]["file"], f["centers"])
    if args.limit:
        videos = dict(list(videos.items())[:args.limit])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval().to(device)
    load_seconds = time.perf_counter() - t0
    encode(model, np.zeros((1, FRAMES, SIZE, SIZE, 3), np.uint8), device)  # 첫 호출의 준비 시간은 재지 않는다

    rows = {k: [] for k in ("vid", "t", "cls", "patch_mean", "motion_lev", "motion_pix", "pix16")}
    decode_seconds = model_seconds = 0.0
    for vid, (path, centers) in videos.items():
        t0 = time.perf_counter()
        frames = decode(path)
        decode_seconds += time.perf_counter() - t0
        clips = np.stack([frames[clip_indices(t, len(frames))] for t in centers])
        for i in range(0, len(clips), BATCH):
            batch = clips[i:i + BATCH]
            t0 = time.perf_counter()
            cls, patch_mean, motion = encode(model, batch, device)
            model_seconds += time.perf_counter() - t0
            rows["cls"].extend(cls)
            rows["patch_mean"].extend(patch_mean)
            rows["motion_lev"].extend(motion)
        gray = clips.astype(np.float32).mean(-1)  # (N, T, H, W)
        rows["motion_pix"].extend(np.abs(np.diff(gray, axis=1)).mean((1, 2, 3)))
        mid = gray[:, FRAMES // 2].reshape(len(clips), 16, SIZE // 16, 16, SIZE // 16).mean((2, 4))
        rows["pix16"].extend(mid.reshape(len(clips), -1))
        rows["vid"].extend([vid] * len(centers))
        rows["t"].extend(centers)
        print(f"{vid[:40]:40s} 클립 {len(centers):3d}개  누적 모델 시간 {model_seconds:6.1f}초", flush=True)

    n = len(rows["t"])
    timing = {"model": MODEL, "device": device, "torch": torch.__version__, "clips": n, "batch": BATCH,
              "load_seconds": round(load_seconds, 1), "decode_seconds": round(decode_seconds, 1),
              "model_seconds": round(model_seconds, 1), "clips_per_second": round(n / model_seconds, 2),
              "seconds_per_clip": round(model_seconds / n, 3)}
    if args.cpu_sample:
        cpu_model = model.to("cpu")
        sample = np.stack([frames[clip_indices(t, len(frames))] for t in centers[:4]])
        encode(cpu_model, sample[:1], "cpu")
        t0 = time.perf_counter()
        for i in range(0, 4, BATCH):
            encode(cpu_model, sample[i:i + BATCH], "cpu")
        timing["cpu_seconds_per_clip"] = round((time.perf_counter() - t0) / 4, 3)
    print(json.dumps(timing, ensure_ascii=False))
    if not args.limit:
        np.savez_compressed(HERE / "results" / "lev_emb.npz", vid=np.array(rows["vid"]),
                            t=np.array(rows["t"], np.float32), cls=np.array(rows["cls"], np.float32),
                            patch_mean=np.array(rows["patch_mean"], np.float32),
                            motion_lev=np.array(rows["motion_lev"], np.float32),
                            motion_pix=np.array(rows["motion_pix"], np.float32),
                            pix16=np.array(rows["pix16"], np.float32))
        (HERE / "results" / "lev_timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
