"""움직임을 따라 프레임을 고르는 세 방식(프레임 차이, 점 추적, 전체 이동 보정)의 공용 함수.

원문과의 관계 (읽은 범위: 방법 절과 공식 코드)
- 프레임 차이: MGSampler(ICCV 2021, arXiv 2104.09952) 3.2~3.3절.
  이웃 프레임의 화소 차이 합을 움직임으로 보고, 지수 μ=0.5로 누른 뒤 누적 분포를 같은 간격으로 잘라 구간마다 한 장을 뽑는다.
  다른 점: 원문은 학습 때 구간 안에서 무작위로 뽑는데 여기서는 구간의 가운데 값을 쓴다(다시 돌려도 같게).
           원문은 모든 프레임에서 고르는데 여기서는 추적 타임라인(8fps)에서 고른다. 같은 프레임이 두 번 뽑히면 옆 프레임으로 옮긴다.
- 점 추적: Motion-as-Prompt(arXiv 2608.11655) 3.2~3.4절과 공식 코드(github.com/SunVictor23/MaP)의
  motion_energy.py, frame_selector.py, track_marker.py를 그대로 옮겼다(numpy). 설정은 공식 실행 스크립트의 기본값이다
  (추적 8fps, 10×10 격자, 1초 구간마다 새 격자, 카메라 보정, 균등 1/4 + 에너지 높은 순 + 최소 간격).
  다른 점은 motion_track.py와 motion_sheets.py의 설명에 적었다.
- 전체 이동 보정: 논문에 없는 내 제안. 이웃 프레임 사이의 화면 전체 이동(가로·세로)을 위상 상관으로 구해 맞춘 뒤 차이를 낸다.
  회전·확대는 보정하지 않는다. 배정 규칙은 프레임 차이 방식과 같다.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import colorsys
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from extract_frames import FFMPEG

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
TRACK_FPS = 8.0      # MaP 공식 기본값 (tracking_fps)
LONG_SIDE = 512      # 추적·차이 계산에 쓰는 프레임의 긴 변
GRID = 10            # MaP 논문 설정 (10×10 격자)
RESEED_SECONDS = 1.0
MU = 0.5             # MGSampler의 지수 (논문에서 가장 좋았던 값)
STATIC_FRAC = 0.03   # MaP: 1초 안의 최대 이동이 화면 너비의 3% 미만이면 멈춘 점
POINTS_PER_GAP = 8   # 표본 프레임 사이마다 그리는 궤적의 최대 수 (아래 marks_for 설명)


def video_file(video):
    return HERE / "data" / "sta_videos" / f"{video.removeprefix('sta_')}.mp4"


def native_fps(video):
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video_file(video))], capture_output=True, text=True).stderr
    line = next(x for x in err.splitlines() if "Video:" in x)
    return float(re.search(r"([\d.]+) fps", line).group(1))


def timeline(video):
    """영상을 추적 속도(8fps, 원본이 더 느리면 원본 속도)로 다시 뽑는다. 돌려주는 값: 프레임 (T,H,W,3), 시각 (T,), fps.

    MaP 공식 코드는 원본 프레임 번호를 고르게 골라 같은 수를 뽑는다. 여기서는 ffmpeg의 fps 필터를 써서 거의 같은 프레임이 나온다.
    공식 코드의 프레임 수 상한(200)은 두지 않는다. 논문의 영상은 5초 안팎이라 상한에 걸리지 않는데,
    우리 영상(24~40초)에 걸면 추적 속도가 5~7fps로 내려가기 때문이다.
    """
    path = video_file(video)
    fps = min(TRACK_FPS, native_fps(video))
    probe = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "ppm", "-"], capture_output=True, check=True).stdout
    w0, h0 = (int(x) for x in probe.split(maxsplit=3)[1:3])
    scale = LONG_SIDE / max(w0, h0)
    w, h = round(w0 * scale / 2) * 2, round(h0 * scale / 2) * 2
    raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    return frames, np.arange(len(frames)) / fps, fps


# ---------- 프레임 차이 (MGSampler) ----------

def diff_energy(frames):
    """프레임 t의 값 = t−1과 t의 화소 차이의 평균. 첫 프레임은 다음 값으로 채운다."""
    f = frames.astype(np.float32)
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    return np.concatenate([d[:1], d])


def cumulative_pick(energy, budget, mu=MU):
    """MGSampler: 움직임을 μ제곱해 고르게 한 뒤 누적 분포를 budget개의 같은 구간으로 나누고 구간마다 한 장."""
    weight = np.power(np.maximum(energy, 0.0), mu)
    weight = weight / weight.sum() if weight.sum() > 0 else np.full(len(energy), 1 / len(energy))
    cdf = np.cumsum(weight)
    chosen = []
    for q in (np.arange(budget) + 0.5) / budget:  # 구간의 가운데 값
        k = min(int(np.searchsorted(cdf, q)), len(energy) - 1)
        if k in chosen:  # 같은 프레임이 또 뽑히면 가장 가까운 빈 프레임으로
            free = [j for j in range(len(energy)) if j not in chosen]
            k = min(free, key=lambda j: (abs(j - k), j))
        chosen.append(k)
    return sorted(chosen)


# ---------- 전체 이동 보정 (내 제안) ----------

def shift_between(a, b):
    """흑백 프레임 a를 얼마나 옮기면 b와 겹치는지(세로, 가로 화소). 위상 상관."""
    h, w = a.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fa, fb = np.fft.rfft2((a - a.mean()) * window), np.fft.rfft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    corr = np.fft.irfft2(cross / np.maximum(np.abs(cross), 1e-9), s=(h, w))
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
    if abs(dy) > h // 4 or abs(dx) > w // 4:  # 화면의 4분의 1 넘게 옮기라는 답은 믿지 않는다
        return 0, 0
    return int(dy), int(dx)


def shift_energy(frames):
    """이웃 프레임을 전체 이동만큼 맞춘 뒤, 겹치는 부분에서 낸 화소 차이의 평균. 돌려주는 값: 에너지, 이동량."""
    gray = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    f = frames.astype(np.float32)
    out, shifts = [], []
    for t in range(1, len(f)):
        dy, dx = shift_between(gray[t - 1], gray[t])
        h, w = gray[t].shape
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))       # 지금 프레임에서 겹치는 부분
        yp, xp = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))   # 앞 프레임에서 겹치는 부분
        out.append(float(np.abs(f[t][ys, xs] - f[t - 1][yp, xp]).mean()))
        shifts.append((dy, dx))
    return np.array(out[:1] + out), [shifts[0]] + shifts


# ---------- 점 추적 (Motion-as-Prompt 공식 코드를 옮김) ----------

def _robust_norm(x, eps=1e-8):
    if x.size == 0:
        return x
    hi = np.percentile(x, 95)
    return np.zeros_like(x) if hi < eps else np.clip(x / hi, 0.0, 1.0)


def _camera_flow(p0, p1, mask):
    """보이는 점들의 이동을 가장 잘 설명하는 전체 변환(이동·회전·크기)을 최소제곱으로 구해, 모든 점의 예측 이동을 돌려준다."""
    n = p0.shape[0]
    if mask.sum() < 3:
        return np.zeros((n, 2))
    x, y = p0[mask, 0], p0[mask, 1]
    A = np.zeros((2 * len(x), 4))
    A[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], axis=1)
    A[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], axis=1)
    try:
        (a, b, tx, ty), *_ = np.linalg.lstsq(A, p1[mask].reshape(-1), rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros((n, 2))
    pred = np.stack([a * p0[:, 0] - b * p0[:, 1] + tx, b * p0[:, 0] + a * p0[:, 1] + ty], axis=1)
    return pred - p0


def track_energy(tracks, visibility, hw, smooth=3):
    """MaP의 움직임 에너지 M(t): 카메라 움직임을 뺀 점 속도의 속력 + 가속 + 방향 전환. 돌려주는 값: M (T,), 점 속도 (T,N,2)."""
    T, N, _ = tracks.shape
    diag = float(np.hypot(*hw)) + 1e-8
    tracks, vis = tracks.astype(np.float64), visibility.astype(bool)
    vel = np.zeros((T, N, 2))
    for t in range(T - 1):
        pair = vis[t] & vis[t + 1]
        disp = tracks[t + 1] - tracks[t] - _camera_flow(tracks[t], tracks[t + 1], pair)
        disp[~pair] = 0.0
        vel[t] = disp
    vel[T - 1] = vel[T - 2] if T >= 2 else 0.0
    vis_f = vis.astype(np.float64)
    denom = np.maximum(vis_f.sum(axis=1), 1.0)
    speed = (np.linalg.norm(vel, axis=2) / diag * vis_f).sum(axis=1) / denom
    acc_vec = np.zeros_like(vel)
    acc_vec[1:] = vel[1:] - vel[:-1]
    accel = (np.linalg.norm(acc_vec, axis=2) / diag * vis_f).sum(axis=1) / denom
    prev, nxt = vel[:-1], vel[1:]
    nrm = np.linalg.norm(prev, axis=2) * np.linalg.norm(nxt, axis=2)
    cos = np.divide((prev * nxt).sum(axis=2), nrm, out=np.ones_like(nrm), where=nrm > 1e-8)
    curv_pt = np.zeros((T, N))
    curv_pt[1:] = np.arccos(np.clip(cos, -1.0, 1.0)) * np.minimum(np.linalg.norm(prev, axis=2), np.linalg.norm(nxt, axis=2)) / diag
    curvature = (curv_pt * vis_f).sum(axis=1) / denom
    M = _robust_norm(_robust_norm(speed) + _robust_norm(accel) + _robust_norm(curvature))
    if smooth > 1:
        pad = smooth // 2
        M = np.convolve(np.pad(M, (pad, pad), mode="edge"), np.ones(smooth) / smooth, mode="valid")[: len(M)]
        M = _robust_norm(M)
    return M, vel


def anchor_peak_pick(M, budget):
    """MaP의 표본 선택: 예산의 1/4은 균등(처음과 끝 포함), 나머지는 에너지 높은 순으로 최소 간격을 두고, 모자라면 높은 순으로 채움."""
    T = len(M)
    if budget >= T:
        return list(range(T)), list(range(T))
    radius = max(1, T // (2 * budget))
    anchors = np.unique(np.linspace(0, T - 1, num=min(max(2, int(np.ceil(budget / 4))), budget)).round().astype(int))
    selected = set(anchors.tolist())
    blocked = np.zeros(T, dtype=bool)
    for k in selected:
        blocked[max(0, k - radius): k + radius + 1] = True
    order = np.argsort(-M)
    for k in order:
        if len(selected) >= budget:
            break
        if not blocked[k]:
            selected.add(int(k))
            blocked[max(0, k - radius): k + radius + 1] = True
    for k in order:  # 최소 간격 때문에 모자라면 간격 없이 채운다
        if len(selected) >= budget:
            break
        selected.add(int(k))
    return sorted(selected)[:budget], anchors.tolist()


def marks_for(tracks, visibility, seg_start, hw, fps, bounds):
    """표본 프레임 사이마다 그릴 궤적을 고른다. 돌려주는 값: {뒤 프레임 번호: [(경로 (L,2), 보임 (L,)), ...]}.

    MaP의 "points" 방식(select_motion_points): 보이는 프레임 가운데 1초 창 안의 이동이 화면 너비의 3% 이상인 점만 남기고,
    많이 움직인 순(프레임 사이 이동 거리의 합)으로 자른다.
    다른 점: 공식 코드는 영상 전체에서 최대 40개를 남긴다(논문의 영상은 5초 안팎, 표본 5장 안팎이라 프레임당 8개꼴).
             우리 영상은 표본이 22~38장이라 같은 밀도가 되게 표본 프레임 사이마다 최대 POINTS_PER_GAP개를 남긴다.
    """
    vis = visibility.astype(bool)
    win, thr = max(1, int(round(fps))), STATIC_FRAC * float(hw[1])
    out = {}
    for lo, hi in zip(bounds, bounds[1:]):
        kept = []
        for n in np.where(seg_start == lo)[0]:  # 이 구간의 앞 프레임에 뿌린 점
            seen = np.where(vis[lo:hi + 1, n])[0]
            if seen.size < 2:
                continue
            xy = tracks[lo:hi + 1, n]
            moved = 0.0
            for i, a in enumerate(seen):
                b = seen[np.searchsorted(seen, a + win, side="right") - 1]  # 창 안에서 가장 뒤의 보이는 프레임
                moved = max(moved, float(np.linalg.norm(xy[b] - xy[a])))
            if moved >= thr:
                step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                total = float(step[vis[lo + 1:hi + 1, n]].sum())
                kept.append((total, int(n)))
        kept.sort(reverse=True)
        out[int(hi)] = [(tracks[lo:hi + 1, n].copy(), vis[lo:hi + 1, n].copy()) for _, n in kept[:POINTS_PER_GAP]]
    return out


def row_color(y, H):
    """MaP의 "row" 색: 점이 처음 있던 높이에 따라 색상(HSV)을 정한다. 한 점의 선은 한 가지 색."""
    hue = int(float(np.clip(y / max(H - 1, 1), 0.0, 1.0)) * 179) * 2 / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 220 / 255, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def read_json(path):
    return json.loads((HERE / path).read_text())
