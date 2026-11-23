"""1초 안쪽의 짧은 장면을 잡으려면 프레임을 어디에 더 넣어야 하나: 정답 기록만으로 세어 본다. Jev 요청은 없다.

질문: "중요한 곳만 촘촘히"가 1초보다 짧은 장면에 통하는가. 앞선 시험(작업 17·20)은 Charades(몇 초짜리 동작)에서 움직임 크기로
프레임을 더 넣었고 이득이 없었다. 여기서는 마인크래프트의 1초 미만 창 구간(기록의 isGuiOpen)을 대상으로,
프레임을 더 넣었을 때 그 구간 안에 프레임이 하나라도 들어가는지만 센다(들어간 프레임을 Jev가 알아보는지는 따로 물어야 한다).

넣는 방법
- 고르게  : 같은 장수를 일정한 간격으로
- 변화 큼 : 이웃 프레임의 차이(20fps, 96×54 흑백)가 큰 순서로. 카메라를 돌릴 때도 커진다
- 튀는 곳 : 차이가 앞뒤 0.5초의 중앙값보다 몇 배 큰 순서로(갑자기 바뀌는 곳). 고른 곳의 0.1초 뒤 프레임을 넣는다
1분(시트 한 장)마다 N장을 더 넣는다. 기본은 1초 간격 60장(0.5, 1.5, …초).

받은 파일은 믿지 않는 자료이므로 -I로 돌린다.
실행: <venv>/bin/python -I experiments/frame_limit/scripts/motion/dense_sim.py
결과: 표 출력, results/dense_sim.json
"""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
W, H, FPS, MINUTE = 96, 54, 20, 1200
BUDGETS = [10, 20, 30, 60]


def energy(path):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vf", f"fps={FPS},scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H * W).astype(np.int16)
    diff = np.abs(frames[1:] - frames[:-1]).mean(axis=1)
    return np.concatenate([[0.0], diff])  # diff[i] = i-1번째와 i번째 프레임의 차이


def spike(e, half=10):
    out = np.zeros_like(e)
    for i in range(len(e)):
        near = np.concatenate([e[max(0, i - half):i], e[i + 1:i + 1 + half]])
        out[i] = e[i] / (np.median(near) + 0.5)
    return out


def pick(score, base, budget, n, shift):
    """1분마다 점수가 큰 순서로 budget장. 이미 고른 곳의 0.25초 안은 건너뛴다. 넣는 프레임은 고른 틱의 shift틱 뒤."""
    chosen = set()
    for start in range(0, n, MINUTE):
        order = np.argsort(-score[start:start + MINUTE]) + start
        got = []
        for i in order:
            t = min(int(i) + shift, n - 1)
            if len(got) == budget:
                break
            if all(abs(t - x) > 5 for x in got) and all(abs(t - b) > 2 for b in base if abs(t - b) < 10):
                got.append(t)
        chosen.update(got)
    return chosen


def uniform(budget, n):
    per = 60 + budget
    return {min(n - 1, round((k + 0.5) * MINUTE / per) + start) for start in range(0, n, MINUTE) for k in range(per)}


def main():
    videos = json.loads((HERE / "data" / "mc_items.json").read_text())["videos"]
    events = [e for e in json.loads((HERE / "data" / "mc_events.json").read_text())["events"] if e["kind"] == "hurt"]
    short, longer, hurts, plans = [], [], [], {}
    for vid, v in videos.items():
        path = HERE / v["file"]
        gui = [json.loads(line)["isGuiOpen"] for line in open(path.with_suffix(".jsonl"))]
        e = energy(path)
        n = min(len(gui), len(e))
        e, sp = e[:n], spike(e[:n])
        base = set(range(10, n, 20))
        runs, i = [], 0
        while i < n:
            if gui[i]:
                j = i
                while j + 1 < n and gui[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        plans[vid] = {"base": base}
        for b in BUDGETS:
            plans[vid][("고르게", b)] = uniform(b, n)
            plans[vid][("변화 큼", b)] = base | pick(e, base, b, n, 2)
            plans[vid][("튀는 곳", b)] = base | pick(sp, base, b, n, 2)
        for a, b in runs:
            (short if b - a + 1 < 20 else longer).append((vid, a, b))
        for x in events:
            if x["video"] == vid:  # 피해 입은 순간의 앞뒤 0.25초에서 가장 큰 "튀는 정도"와 그 1분에서의 순위
                t = round(x["t"] * FPS)
                start = (t // MINUTE) * MINUTE
                peak = float(sp[max(0, t - 5):t + 6].max())
                hurts.append({"video": vid, "t": x["t"], "times": round(peak, 1), "rank": int((sp[start:start + MINUTE] > peak).sum()) + 1})
    inside = lambda samples, a, b: any(a <= t <= b for t in samples)
    out = {"short_intervals": len(short), "longer_intervals": len(longer), "rows": {}}
    print(f"창이 열린 구간: 1초 미만 {len(short)}개, 1초 이상 {len(longer)}개. 영상 {len(videos)}개 × 5분")
    row = sum(inside(plans[v]["base"], a, b) for v, a, b in short)
    print(f"{'넣는 방법':10s}{'분당 더 넣는 장수':>10s}{'1초 미만 구간 중 프레임이 든 것':>20s}")
    print(f"{'1초 간격만':10s}{0:>12d}{row:>14d}/{len(short)}")
    out["rows"]["base"] = row
    for b in BUDGETS:
        for name in ("고르게", "변화 큼", "튀는 곳"):
            got = sum(inside(plans[v][(name, b)], a, c) for v, a, c in short)
            out["rows"][f"{name} +{b}"] = got
            print(f"{name:10s}{b:>12d}{got:>14d}/{len(short)}")
    print("피해 입은 순간(앞뒤 0.25초)에서 가장 큰 '튀는 정도'와 그 1분(1,200틱)에서의 순위:")
    for h in hurts:
        print(f"  {h['video'][:5]} {h['t']:.2f}s: {h['times']}배, {h['rank']}번째")
    out["hurt"] = hurts
    (HERE / "results" / "dense_sim.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
