"""하이라이트 실험 3(마인크래프트): 성능 우선. 영상 전체를 초당 4장, 원본 크기로 훑고 알림을 더해 순위를 다시 낸다.

--stage scan   0.25초마다(0, 0.25, …초) 원본 640×360 프레임 한 장에 Noul 세 개(fight, crafting, ore. 문구는 hl_run.py와 같음)
--stage score  후보 위치에 점수 요청(hl_run.py의 요청 B와 같은 질문). 후보 =
               · 어느 1초의 프레임 4장 중 하나라도 0.5 이상이면 그 초(위치 x.5)를 켠 것으로 보고 hl_run.py와 같은 규칙으로 고른 곳
               · 알림(hl2_run.py의 모서리 그림)이 켜진 덩어리마다 첫 그림의 시각에 가장 가까운 위치
               작업 26에서 이미 물은 위치(results/hl_mc_score.json)는 다시 쓰고 새 위치만 보낸다
피해 입음은 기준 문구가 화면과 달랐던 것으로 드러나 묻지 않는다. 따른 논문은 없다.

실행: <venv>/bin/python experiments/frame_limit/scripts/highlight/hl3_run.py --stage scan [--dry-run]     (환경변수 TYPESAFE_API_KEY 필요)
      <venv>/bin/python experiments/frame_limit/scripts/highlight/hl3_run.py --stage score [--dry-run]
결과: results/hl3_scan.json, results/hl3_score.json, data/hl_dense4/…jpg(훑는 프레임)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time

import hl_run as H
from extract_frames import FFMPEG, grab

HERE = H.HERE
KINDS = ("fight", "crafting", "ore")
STEP, CUT = 0.25, 0.5


def dense_frames(video, v):
    """영상을 한 번 읽어 0.25초마다의 프레임을 원본 크기로 뽑아 둔다. [(시각, 경로)]"""
    out_dir = HERE / "data" / "hl_dense4" / video
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "00001.jpg").exists():
        for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
            cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(HERE / v["file"]), "-vf", "select='not(mod(n\\,5))'",
                   *sync, "-q:v", "3", str(out_dir / "%05d.jpg")]
            if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                break
        else:
            raise RuntimeError(f"ffmpeg 실패: {video}")
    files = sorted(out_dir.glob("*.jpg"))
    return [(round((int(p.stem) - 1) * STEP, 2), str(p.relative_to(HERE))) for p in files]


def stage_scan(args):
    from typesafe_sdk import Image, Noul, TypeSafeClient
    videos = H.read("data/mc_items.json")["videos"]
    jobs = [{"video": video, "t": t, "file": p} for video in sorted(videos) for t, p in dense_frames(video, videos[video])]
    out_path = HERE / "results" / "hl3_scan.json"
    done = {(r["video"], r["t"]): r for r in json.loads(out_path.read_text())["records"] if not r["error"]} if out_path.exists() else {}
    todo = [j for j in jobs if (j["video"], j["t"]) not in done]
    print(f"훑기: 프레임 {len(jobs)}장 중 보낼 것 {len(todo)}건. 어림: 약 {len(todo) * 1000:,}토큰, 약 ${len(todo) * 1000 * H.PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return

    def one(j):
        state = {"game": H.GAME, "frame": Image((HERE / j["file"]).read_bytes())}
        qs = {name: Noul(instructions=f"Does `frame` show this: {H.EVENTS[name]}") for name in KINDS}
        resp, error = H.ask(client, state, qs)
        rec = {"video": j["video"], "t": j["t"], "error": error}
        if resp is not None:
            rec.update({name: resp.nouls[name].noul for name in KINDS}, input_tokens=resp.usage.input_tokens)
        return rec

    def write():
        recs = [done[(j["video"], j["t"])] for j in jobs if (j["video"], j["t"]) in done]
        tokens = sum(r.get("input_tokens", 0) for r in recs)
        meta = {"model": H.MODEL, "requests": len(recs), "errors": len(jobs) - len(recs), "input_tokens": tokens,
                "cost_usd": round(tokens * H.PRICE_PER_MTOK / 1e6, 4), "step_seconds": STEP, "events": {k: H.EVENTS[k] for k in KINDS},
                "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False) + "\n")
        return meta

    with TypeSafeClient(model=H.MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        for n, rec in enumerate(pool.map(one, todo), 1):
            if not rec["error"]:
                done[(rec["video"], rec["t"])] = rec
            if n % 3000 == 0:  # 길게 도는 실행이라 중간에 저장해 둔다
                write()
                print(f"  {n}/{len(todo)}", flush=True)
    meta = write()
    print(f"요청 {meta['requests']}건, 남은 오류 {meta['errors']}건, 입력 토큰 {meta['input_tokens']:,}, 약 ${meta['cost_usd']}, "
          f"요청당 평균 {meta['input_tokens'] / max(meta['requests'], 1):,.0f}토큰")


def second_flags():
    """영상 → 사건 → 초(0~299)별 그 초의 프레임 4장 중 가장 높은 확률."""
    out = {}
    for r in H.read("results/hl3_scan.json")["records"]:
        sec = int(r["t"])
        for name in KINDS:
            row = out.setdefault(r["video"], {}).setdefault(name, {})
            row[sec] = max(row.get(sec, 0.0), r[name])
    return {v: {name: [row.get(k, 0.0) for k in range(max(row) + 1)] for name, row in kinds.items()} for v, kinds in out.items()}


def toast_runs():
    """알림이 켜진 모서리 그림의 덩어리. [(영상, 첫 시각, 끝 시각)]"""
    hot = sorted((r["video"], r["t"]) for r in H.read("results/hl2_toast-corner.json")["records"] if r["advancement"] >= CUT)
    runs = []
    for video, t in hot:
        if runs and runs[-1][0] == video and t - runs[-1][2] <= 2.5:
            runs[-1] = (video, runs[-1][1], t)
        else:
            runs.append((video, t, t))
    return runs


def plan(every=3):
    flags, frames = second_flags(), H.read("data/mc_frames.json")
    jobs, seen = [], set()
    for video in sorted(flags):
        centers = frames[video]["centers"]
        scores = {name: (row + [0.0] * len(centers))[:len(centers)] for name, row in flags[video].items()}
        cand, _ = H.candidates(scores, every)
        for pos, a, b in cand:
            jobs.append({"video": video, "t": centers[pos], "pos": pos, "role": "candidate", "run": [a, b],
                         "tile": {name: round(scores[name][pos], 3) for name in scores}})
            seen.add((video, pos))
    for video, first, _ in toast_runs():
        centers = frames[video]["centers"]
        pos = min(range(len(centers)), key=lambda k: abs(centers[k] - first))
        if (video, pos) not in seen:
            seen.add((video, pos))
            jobs.append({"video": video, "t": centers[pos], "pos": pos, "role": "toast", "run": None, "tile": {}})
    return jobs


def stage_score(args):
    from typesafe_sdk import Choice, Image, Noul, Score, TypeSafeClient
    jobs = plan()
    old = {(r["video"], r["t"]): r for r in H.read("results/hl_mc_score.json")["records"] if not r["error"]}
    todo = [j for j in jobs if (j["video"], j["t"]) not in old]
    print(f"점수: 후보 {len(jobs)}곳(알림에서 온 것 {sum(j['role'] == 'toast' for j in jobs)}곳) 중 이미 물은 것 {len(jobs) - len(todo)}곳, 보낼 것 {len(todo)}건. "
          f"어림: 약 {len(todo) * 3320:,}토큰, 약 ${len(todo) * 3320 * H.PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    videos = H.read("data/mc_items.json")["videos"]
    for j in todo:
        last = videos[j["video"]]["duration"] - 0.1
        j["times"] = [round(min(max(j["t"] + off, 0.0), last), 2) for off in H.OFFSETS]
        j["frames"] = []
        for t in j["times"]:
            dest = HERE / "data" / "hl_frames" / j["video"] / f"{round(t * 1000):07d}.jpg"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                grab(HERE / videos[j["video"]]["file"], t, dest)
            j["frames"].append(str(dest.relative_to(HERE)))
    print("프레임 준비 끝", flush=True)

    def one(j):
        state = {"game": H.GAME, "viewer_request": H.REQUEST,
                 "about": "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first).",
                 "frames": [{"seconds": t, "frame": Image((HERE / p).read_bytes())} for t, p in zip(j["times"], j["frames"])]}
        qs = {"action": Score(instructions="How much is the player fighting or in danger in `frames`?", criteria=H.ACTION),
              "progress": Score(instructions="How much progress toward better tools do `frames` show?", criteria=H.PROGRESS),
              "kind": Choice(instructions="What is the player doing in `frames`?", criteria=H.KINDS),
              "worth": Noul(instructions="Would this moment fit `viewer_request`?")}
        resp, error = H.ask(client, state, qs)
        rec = {**j, "error": error, "reused": False}
        if resp is not None:
            for name in ("action", "progress"):
                a = resp.scores[name]
                rec[name] = {"score": a.score, "confidence": a.confidence, "probabilities": {str(k): v for k, v in a.probabilities.items()}}
            rec["kind"] = dict(resp.choices["kind"].probabilities)
            rec["worth"] = resp.nouls["worth"].noul
            rec["input_tokens"] = resp.usage.input_tokens
        return rec

    with TypeSafeClient(model=H.MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        fresh = {(r["video"], r["t"]): r for r in pool.map(one, todo)}
    recs = []
    for j in jobs:
        key = (j["video"], j["t"])
        if key in fresh:
            recs.append(fresh[key])
        else:
            o = old[key]
            recs.append({**j, "error": None, "reused": True, "action": o["action"], "progress": o["progress"], "kind": o["kind"],
                         "worth": o["worth"], "input_tokens": 0})
    H.save("results/hl3_score.json", recs, {"note": "input_tokens는 이번에 새로 보낸 요청만 센다(reused는 0)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["scan", "score"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    (stage_scan if args.stage == "scan" else stage_score)(args)


if __name__ == "__main__":
    main()
