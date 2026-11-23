"""CoTracker3(Meta, 가중치는 CC BY-NC 4.0)로 격자점을 추적해 저장한다. Motion-as-Prompt의 추적 단계다.

공식 코드(map_kit/data/cotracker_runner.py)의 "segment_track" 방식을 따른다.
영상을 구간으로 나누고(이웃 구간은 프레임 한 장을 함께 가짐), 구간마다 첫 프레임에 10×10 격자를 새로 뿌려 그 구간 안에서만 추적한다.
- pass1: 구간 = 1초. 움직임 에너지를 내는 데 쓴다
- pass2: 구간 = 고른 표본 프레임 사이. 궤적을 그리는 데 쓴다 (data/motion_select.json의 "track" 선택이 있어야 한다)

실행: <venv>/bin/python experiments/frame_limit/scripts/motion/motion_track.py pass1 --src <co-tracker 저장소> --ckpt <scaled_offline.pth>
      --videos sta_RFH6M   일부 영상만
      --limit 3            구간 몇 개만 돌려 속도를 잰다 (저장하지 않음)
      --device cpu|mps
결과: data/motion/<pass>_<video>.npz (tracks (T,N,2) 512px 프레임 좌표, visibility (T,N), seg_start (N,), timestamps (T,), hw, fps)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import sys
import time

import numpy as np

import motion_lib as L

OUT = L.HERE / "data" / "motion"


def load(src, ckpt, device):
    import torch
    sys.path.insert(0, src)
    from cotracker.predictor import CoTrackerPredictor
    window = int(torch.load(ckpt, map_location="cpu", weights_only=True)["time_emb"].shape[1])  # 공식 코드처럼 가중치에서 읽는다
    model = CoTrackerPredictor(checkpoint=ckpt, offline=True, v2=False, window_len=window).to(device).eval()
    return model


def track(model, frames, spans, device):
    """구간마다 격자를 새로 뿌려 추적한다. 돌려주는 값: tracks (T,N,2), visibility (T,N), seg_start (N,)."""
    import torch
    T = len(frames)
    tracks, vis, starts = [], [], []
    for s, e in spans:
        sub = torch.from_numpy(frames[s:e + 1].copy()).permute(0, 3, 1, 2)[None].float().to(device)
        with torch.no_grad():
            tr, vi = model(sub, grid_size=L.GRID, grid_query_frame=0, backward_tracking=False)
        n = tr.shape[2]
        block_t, block_v = np.zeros((T, n, 2), dtype=np.float32), np.zeros((T, n), dtype=bool)
        block_t[s:e + 1], block_v[s:e + 1] = tr[0].float().cpu().numpy(), vi[0].bool().cpu().numpy()
        tracks.append(block_t)
        vis.append(block_v)
        starts += [s] * n
    return np.concatenate(tracks, axis=1), np.concatenate(vis, axis=1), np.array(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["pass1", "pass2"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    videos = args.videos.split(",") if args.videos else sorted(L.read_json("data/sta_frames.json"))
    picks = L.read_json("data/motion_select.json") if args.which == "pass2" else None
    model = load(args.src, args.ckpt, args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for video in videos:
        frames, ts, fps = L.timeline(video)
        T = len(frames)
        if args.which == "pass1":
            step = max(1, int(round(L.RESEED_SECONDS * fps)))
            spans = [(s, min(s + step, T - 1)) for s in range(0, T - 1, step)]
        else:
            b = picks[video]["track"]["indices"]
            spans = [(lo, hi) for lo, hi in zip(b, b[1:]) if hi > lo]
        if args.limit:
            spans = spans[:args.limit]
        t0 = time.perf_counter()
        tracks, vis, starts = track(model, frames, spans, args.device)
        took = time.perf_counter() - t0
        total += took
        n_frames = sum(e - s + 1 for s, e in spans)
        seen = sum(vis[s:e + 1][:, starts == s].mean() for s, e in spans) / len(spans)  # 구간 안에서 점이 보인 비율
        print(f"{video}: 프레임 {T}장({fps:g}fps), 구간 {len(spans)}개, 넣은 프레임 {n_frames}장, {took:.1f}초 "
              f"(프레임당 {took / n_frames * 1000:.0f}ms), 보이는 점 비율 {seen:.2f}", flush=True)
        if not args.limit:
            np.savez_compressed(OUT / f"{args.which}_{video}.npz", tracks=tracks, visibility=vis, seg_start=starts,
                                timestamps=ts, hw=np.array(frames.shape[1:3]), fps=fps)
    print(f"모두 {total:.1f}초")


if __name__ == "__main__":
    main()
