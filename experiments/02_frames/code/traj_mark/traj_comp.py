"""카메라 움직임을 뺀 경로 그림(comp)을 만든다. traj_make.py(plain, path)와 같은 401개 구간, 같은 끝 프레임.

**우리 쪽 변경이고 논문에 없다.** Motion-as-Prompt(arXiv 2608.11655)는 3.2절에서 카메라를 빼지만 그것은 프레임을 고르는 움직임 점수에만 쓰고,
3.4절의 점 고르기·경로 그리기는 원래 위치의 이동으로 한다. 여기서는 점 고르기와 그리기에 카메라 보정을 넣었다.

방법 (README의 "카메라 움직임을 뺀 변형"과 같다)
- 구간마다 각 프레임 → 끝 프레임의 화면 전체 변환을 그 구간의 점(100개)으로 맞춘다. 주 방법 sim = 이동·회전·크기(논문 3.2절의 모형)를 RANSAC으로
  (2점씩 300번, 어긋남 < 화면 너비 1%면 일치, 가장 많이 일치한 것을 일치한 점들로 최소제곱 다시). 함께 보이는 점이 3개 미만이면 보정 없음
- 카메라를 뺀 경로 q = 각 프레임의 위치를 그 변환으로 끝 프레임 시점에 옮긴 것(끝점은 실제 위치)
- 고르기·그리기는 motion_lib.marks_for, traj_make.draw와 같은 규칙(3% 문턱, 이동 거리 합 순 8개, 행 색, 끝점 원, 70%)을 q에 적용한다.
  하나 더: 끝 프레임에서 보이는 점만 그린다(논문 3.4절의 "visible point". 첫 그림에서 화면 밖으로 나간 점이 8개 자리를 차지하는 것을 보고 넣었다)
- 참고용(판정에 안 씀): ls = 논문 그대로의 최소제곱(모든 점), aff = RANSAC 아핀, hom = RANSAC 호모그래피. 그림은 sim만 만든다

쓰는 것 (읽기만 한다): ../frame_limit/data/motion/pass1_<video>.npz, ../frame_limit/data/sta_videos, motion_lib.py, traj_make.py(draw)
실행: <numpy·Pillow·imageio-ffmpeg가 있는 파이썬> traj_comp.py   → img_comp/<video>/<끝 프레임 번호>_comp.jpg, index_comp.json
Charades 영상은 비상업 연구용 자료라 실험에만 쓰고 다시 배포하지 않는다. CoTracker3 가중치는 CC BY-NC 4.0이다.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent / "frame_limit"
sys.path[:0] = [str(FL / "scripts" / "motion"), str(HERE)]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import motion_lib as L  # noqa: E402
from traj_make import draw  # noqa: E402

ITERS = 300          # RANSAC 뽑기 횟수
INLIER_FRAC = 0.01   # 일치로 치는 어긋남: 화면 너비의 1% (고르기 문턱 3%의 3분의 1)
MODELS = ("sim", "ls", "aff", "hom")   # sim이 주 방법
MIN_PTS = {"sim": 2, "aff": 3, "hom": 4}
EYE = np.eye(3)


# ---------- 변환 맞추기 (모두 3×3 행렬로 돌려준다) ----------

def apply(H, p):
    """H (3,3) 또는 (k,3,3)을 점 p (n,2)에 적용. 돌려주는 값 (n,2) 또는 (k,n,2). 분모가 0에 가까우면 nan."""
    ph = np.concatenate([p, np.ones((len(p), 1))], axis=1)
    out = ph @ np.swapaxes(H, -1, -2)
    z = out[..., 2:3]
    return np.where(np.abs(z) > 1e-9, out[..., :2] / np.where(np.abs(z) > 1e-9, z, 1.0), np.nan)


def fit_sim(src, dst):
    """이동·회전·크기 최소제곱(논문 3.2절, motion_lib._camera_flow와 같은 식)."""
    x, y = src[:, 0], src[:, 1]
    A = np.zeros((2 * len(x), 4))
    A[0::2] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], axis=1)
    A[1::2] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], axis=1)
    (a, b, tx, ty), *_ = np.linalg.lstsq(A, dst.reshape(-1), rcond=None)
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]])


def fit_aff(src, dst):
    X = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return np.vstack([M.T, [0, 0, 1.0]])


def _norm_T(p):
    c = p.mean(axis=0)
    s = np.sqrt(2) / max(float(np.linalg.norm(p - c, axis=1).mean()), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _dlt_rows(s, d):
    x, y, u, v = s[..., 0], s[..., 1], d[..., 0], d[..., 1]
    o, z = np.ones_like(x), np.zeros_like(x)
    r1 = np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], axis=-1)
    r2 = np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], axis=-1)
    return np.concatenate([r1, r2], axis=-2)


def fit_hom(src, dst):
    Ts, Td = _norm_T(src), _norm_T(dst)
    s, d = apply(Ts, src), apply(Td, dst)
    h = np.linalg.svd(_dlt_rows(s, d))[2][-1]
    H = np.linalg.inv(Td) @ h.reshape(3, 3) @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else EYE


FIT = {"sim": fit_sim, "aff": fit_aff, "hom": fit_hom}


def minimal(model, s, d, scale):
    """뽑은 점 묶음 s, d (k,m,2)마다 변환 하나. 돌려주는 값 (k,3,3)과 쓸 수 있는지 (k,)."""
    k = len(s)
    if model == "sim":
        zs, zd = s[..., 0] + 1j * s[..., 1], d[..., 0] + 1j * d[..., 1]
        den = zs[:, 1] - zs[:, 0]
        ok = np.abs(den) > 1e-6
        a = (zd[:, 1] - zd[:, 0]) / np.where(ok, den, 1)
        t = zd[:, 0] - a * zs[:, 0]
        H = np.zeros((k, 3, 3))
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = a.real, -a.imag, t.real
        H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = a.imag, a.real, t.imag
        H[:, 2, 2] = 1
        return H, ok
    if model == "aff":
        X = np.concatenate([s, np.ones((k, 3, 1))], axis=2)
        det = np.linalg.det(X)
        ok = np.abs(det) > 1e-6 * scale * scale
        M = np.linalg.solve(np.where(ok[:, None, None], X, np.eye(3)), d)
        H = np.zeros((k, 3, 3))
        H[:, :2, :] = np.swapaxes(M, 1, 2)
        H[:, 2, 2] = 1
        return H, ok
    # hom: 4점 DLT. 좌표를 화면 크기로 나눠 수치를 안정시킨다
    sn, dn = s / scale, d / scale
    h = np.linalg.svd(_dlt_rows(sn, dn))[2][:, -1]
    Hn = h.reshape(k, 3, 3)
    ok = np.abs(Hn[:, 2, 2]) > 1e-9
    Hn = Hn / np.where(ok, Hn[:, 2, 2], 1)[:, None, None]
    S, Si = np.diag([1 / scale, 1 / scale, 1.0]), np.diag([scale, scale, 1.0])
    return Si @ Hn @ S, ok


def ransac(model, src, dst, thr, rng, scale):
    """돌려주는 값: 변환 (3,3), 일치한 점의 비율."""
    n, m = len(src), MIN_PTS[model]
    if n <= m:
        return FIT[model](src, dst), 1.0
    pick = rng.random((ITERS, n)).argsort(axis=1)[:, :m]
    H, ok = minimal(model, src[pick], dst[pick], scale)
    err = np.linalg.norm(apply(H, src) - dst[None], axis=2)
    err = np.where(np.isnan(err), np.inf, err)
    inl = (err < thr) & ok[:, None]
    count = inl.sum(axis=1)
    cost = np.minimum(err, thr).sum(axis=1)   # 같은 수면 일치한 점들의 어긋남이 작은 쪽
    best = int(np.lexsort((cost, -count))[0])
    mask = inl[best]
    if mask.sum() <= m:
        return H[best], float(mask.mean())
    Hr = FIT[model](src[mask], dst[mask])
    return Hr, float(mask.mean())


# ---------- 구간 하나 ----------

def stabilize(P, V, model, thr, rng, scale):
    """P (L,n,2), V (L,n). 돌려주는 값: 카메라를 뺀 경로 Q (L,n,2), 처음 프레임→끝 프레임 변환, 프레임별 일치 비율."""
    Q, inl, H0 = P.copy(), [], EYE
    for j in range(len(P) - 1):
        common = V[j] & V[-1]
        if common.sum() < 3:
            H, frac = EYE, float("nan")
        elif model == "ls":
            H, frac = fit_sim(P[j, common], P[-1, common]), float("nan")
        else:
            H, frac = ransac(model, P[j, common], P[-1, common], thr, rng, scale)
        Q[j] = apply(H, P[j])
        Q[j] = np.where(np.isnan(Q[j]), P[j], Q[j])
        inl.append(frac)
        if j == 0:
            H0 = H
    return Q, H0, inl


def moved_total(xy, seen_mask, win):
    """motion_lib.marks_for와 같은 두 값: 1초 창 안의 최대 이동, 이동 거리 합(보이는 프레임으로 들어오는 걸음만)."""
    seen = np.where(seen_mask)[0]
    if seen.size < 2:
        return None, None
    moved = 0.0
    for a in seen:
        b = seen[np.searchsorted(seen, a + win, side="right") - 1]
        moved = max(moved, float(np.linalg.norm(xy[b] - xy[a])))
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return moved, float(step[seen_mask[1:]].sum())


def choose(XY, V, win, thr, need_end=False):
    """돌려주는 값: 고른 점 번호(최대 8, 이동 거리 합 순), 남은 양(큰 8개의 최대 이동 평균, 화소), 문턱을 넘은 점 수,
    끝 프레임에서 보이는 점만으로 잰 남은 양(나중에 덧붙인 참고값).

    need_end=True면 끝 프레임에서 보이는 점만 그린다(논문 3.4절 "retained visible point"). 카메라를 뺀 경로에서만 쓴다:
    화면 밖으로 나간 점은 끝 프레임 시점으로 옮기면 그림 밖에 놓여 8개 자리만 차지하고 보이지 않기 때문이다(그림을 보고 나서 고친 것).
    남은 양과 문턱을 넘은 점 수는 README에 미리 적은 대로 모든 점으로 잰다."""
    kept, moves, moves_end, over = [], [], [], 0
    for n in range(XY.shape[1]):
        moved, total = moved_total(XY[:, n], V[:, n], win)
        if moved is None:
            continue
        moves.append(moved)
        if V[-1, n]:
            moves_end.append(moved)
        if moved >= thr:
            over += 1
            if V[-1, n] or not need_end:
                kept.append((total, n))
    kept.sort(reverse=True)
    top = sorted(moves, reverse=True)[:L.POINTS_PER_GAP]
    top_end = sorted(moves_end, reverse=True)[:L.POINTS_PER_GAP]
    return ([n for _, n in kept[:L.POINTS_PER_GAP]], (float(np.mean(top)) if top else 0.0), over,
            (float(np.mean(top_end)) if top_end else 0.0))


def main():
    videos = sorted(p.stem.removeprefix("pass1_") for p in (FL / "data" / "motion").glob("pass1_*.npz"))
    old = {(x["video"], x["frame"]): x for x in json.loads((HERE / "index.json").read_text())["items"]}
    index = []
    for video in videos:
        z = np.load(FL / "data" / "motion" / f"pass1_{video}.npz")
        tracks, vis, seg_start = z["tracks"].astype(np.float64), z["visibility"].astype(bool), z["seg_start"]
        W = float(z["hw"][1])
        win, thr, inl_thr = max(1, int(round(float(z["fps"])))), L.STATIC_FRAC * W, INLIER_FRAC * W
        scale = float(max(z["hw"]))
        frames, ts, fps = L.timeline(video)
        assert len(frames) == tracks.shape[0], (video, len(frames), tracks.shape)
        bounds = sorted({int(x) for x in seg_start} | {len(frames) - 1})
        rng = np.random.default_rng(0)
        out = HERE / "img_comp" / video
        out.mkdir(parents=True, exist_ok=True)
        for lo, hi in zip(bounds, bounds[1:]):
            idx = np.where(seg_start == lo)[0]
            P, V = tracks[lo:hi + 1, idx], vis[lo:hi + 1, idx]
            raw_ids, U, n_raw, U_end = choose(P, V, win, thr)
            raw_end_ids = choose(P, V, win, thr, need_end=True)[0]   # 보정 없이 끝 프레임 규칙만(카메라가 멈춘 구간에서 보정 전후를 같은 규칙으로 견주려고)
            rec = {"video": video, "frame": hi, "t0": round(float(ts[lo]), 3), "t": round(float(ts[hi]), 3),
                   "U": round(U / W, 5), "U_end": round(U_end / W, 5), "over_raw": n_raw, "paths_raw": len(raw_ids), "ids_raw": [int(idx[n]) for n in raw_ids],
                   "ids_raw_end": [int(idx[n]) for n in raw_end_ids]}
            for model in MODELS:
                Q, H0, inl = stabilize(P, V, model, inl_thr, rng, scale)
                ids, R, n_over, R_end = choose(Q, V, win, thr, need_end=True)
                key = "" if model == "sim" else "_" + model
                rec[f"R{key}"], rec[f"over{key}"], rec[f"paths{key}"] = round(R / W, 5), n_over, len(ids)
                if model == "sim":
                    rec["R_end"] = round(R_end / W, 5)
                    p0 = P[0, V[0]]
                    cam = np.linalg.norm(apply(H0, p0) - p0, axis=1) if len(p0) else np.zeros(1)
                    rec["cam"] = round(float(np.median(cam)) / W, 5)          # 처음→끝 카메라 이동(화면 너비 비율, 중앙값)
                    rec["inlier"] = round(float(np.nanmedian(inl)), 3) if np.isfinite(inl).any() else None
                    rec["ids_comp"] = [int(idx[n]) for n in ids]
                    paths = [(Q[:, n].copy(), V[:, n].copy()) for n in ids]
                    draw(frames[hi], paths).save(out / f"{hi:03d}_comp.jpg", "JPEG", quality=92)
            rec["comp"] = f"img_comp/{video}/{hi:03d}_comp.jpg"
            rec["plain"], rec["path"] = f"img/{video}/{hi:03d}_plain.jpg", f"img/{video}/{hi:03d}_path.jpg"
            rec["size"] = [int(frames.shape[2]), int(frames.shape[1])]
            o = old.get((video, hi))
            assert o is not None and o["paths"] == rec["paths_raw"], ("img/와 구간이 다르다", video, hi, o)
            index.append(rec)
        n = [x for x in index if x["video"] == video]
        print(video, f"구간 {len(n)}개, 경로 있는 구간 원래 {sum(x['paths_raw'] > 0 for x in n)} → 보정 {sum(x['paths'] > 0 for x in n)}, "
              f"카메라 이동 중앙값 {np.median([x['cam'] for x in n]):.3f}W, 일치 비율 중앙값 {np.median([x['inlier'] for x in n if x['inlier'] is not None]):.2f}")
    note = ("t0~t = 구간(초), 그림은 끝 프레임. U = 카메라를 빼지 않은 '큰 8개 점의 1초 최대 이동' 평균 ÷ 화면 너비, R = 같은 것을 카메라를 뺀 경로로(주 방법 sim). "
            "R_ls/R_aff/R_hom = 참고 방법. over = 3% 문턱을 넘은 점 수, paths = 그린 경로 수(0~8), cam = 처음→끝 카메라 이동(÷너비), inlier = RANSAC 일치 비율 중앙값. "
            "ids = 고른 점의 번호(npz의 점 축). paths(보정)는 끝 프레임에서 보이는 점만 그린 수. "
            "U_end/R_end = 끝 프레임에서 보이는 점만으로 잰 U/R(나중에 덧붙인 참고값)")
    (HERE / "index_comp.json").write_text(json.dumps({"note": note, "items": index}, ensure_ascii=False, indent=1) + "\n")
    print("합계", len(index), "구간, 경로 있는 구간 원래", sum(x["paths_raw"] > 0 for x in index), "→ 보정", sum(x["paths"] > 0 for x in index),
          ", 8개가 다 찬 구간 원래", sum(x["paths_raw"] == 8 for x in index), "→ 보정", sum(x["paths"] == 8 for x in index))


if __name__ == "__main__":
    main()
