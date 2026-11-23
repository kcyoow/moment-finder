"""하이라이트 실험 2(마인크래프트): 작은 사건과 짧은 사건을 정확도 우선으로 본다. Jev 요청 네 종류.

(가) 게임의 "발전 과제 달성" 알림 — 화면 오른쪽 위의 작은 상자, 약 5.5초 떠 있다
  --stage toast-sheet   기존 60칸 시트(1초 간격)에 칸마다 알림이 있는지 묻는다
  --stage toast-corner  화면 오른쪽 위 240×100px을 2초마다 잘라 672×280으로 키워 한 장씩 묻는다(알림 / 조합법 알림)
(나) 싸움·피해 — 1초보다 짧다
  --stage dense-sheet   사건이 든 15초 구간을 초당 4장으로 60칸 시트 한 장에. 사건 없는 구간 8개도 같이
  --stage frames        사건 앞뒤 1초를 0.25초 간격으로, 원본 크기 프레임을 한 장씩. 사건에서 먼 프레임 300장도 같이

정답은 요청을 고르는 데만 쓴다(사건 둘레와 견줄 곳을 정함). 채점은 hl2_eval.py. 따른 논문은 없다.
모서리를 잘라 크게 보는 것은 TypeSafe 문서의 "필요한 곳만 가까이 보기"와 같은 생각이다.

실행: <venv>/bin/python experiments/frame_limit/scripts/highlight/hl2_run.py --stage <단계> [--dry-run]     (환경변수 TYPESAFE_API_KEY 필요)
결과: results/hl2_<단계>.json, data/hl_corner/…jpg, data/hl_frames/…jpg, data/sheets_mc4fps/…jpg
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess

import sheet_run as R
from extract_frames import FFMPEG, grab
from hl_run import EVENTS, GAME, PRICE_PER_MTOK, ask, mc_sheets, read, save

HERE = R.HERE
MODEL = "jev-1.14.0"
SEED = 20261111
TOAST = "an 'Advancement Made!' notification box is shown in the top-right corner of the screen"
BLOCK, CONTROL_BLOCKS, CONTROL_FRAMES = 15, 8, 300
KINDS = ("fight", "hurt")


def events_of(kinds):
    return [e for e in read("data/mc_events.json")["events"] if e["kind"] in kinds]


def frame(video, t, videos):
    """t초의 프레임을 원본 크기 그대로 뽑아 둔다(hl_run.py의 요청 B와 같은 폴더)."""
    t = round(min(max(t, 0.0), videos[video]["duration"] - 0.1), 3)
    dest = HERE / "data" / "hl_frames" / video / f"{round(t * 1000):07d}.jpg"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        grab(HERE / videos[video]["file"], t, dest)
    return str(dest.relative_to(HERE))


def run_all(name, jobs, one, workers, extra):
    from typesafe_sdk import TypeSafeClient
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(workers) as pool:
        recs = list(pool.map(lambda j: one(client, j), jobs))
    save(f"results/hl2_{name}.json", recs, extra)


def toast_sheet(args):
    from typesafe_sdk import Image, Noul
    sheets = mc_sheets()
    print(f"알림·시트: 시트 {len(sheets)}장, 요청당 질문 60개. 어림: 약 {len(sheets) * 9300:,}토큰, 약 ${len(sheets) * 9300 * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return

    def one(client, sheet):
        state = {"game": GAME, **R.state_of(sheet, Image((HERE / sheet["file"]).read_bytes()))}
        qs = {t["id"]: Noul(instructions=f"Does tile {t['id']} of `sheet` show this: {TOAST}") for t in sheet["tiles"]}
        resp, error = ask(client, state, qs)
        rec = {"video": sheet["video"], "sheet": sheet["sheet"], "error": error}
        if resp is not None:
            rec.update(nouls={t["id"]: resp.nouls[t["id"]].noul for t in sheet["tiles"]}, times={t["id"]: t["t"] for t in sheet["tiles"]},
                       input_tokens=resp.usage.input_tokens)
        return rec
    run_all("toast-sheet", sheets, one, args.workers, {"criterion": TOAST})


def corner_crops(video, v):
    """영상을 한 번 읽어 0.5, 2.5, 4.5, …초 프레임의 오른쪽 위를 잘라 둔다. [(시각, 경로)]"""
    out_dir = HERE / "data" / "hl_corner" / video
    out_dir.mkdir(parents=True, exist_ok=True)
    count = int((v["duration"] - 0.5 - 0.1) // 2) + 1
    paths = [(0.5 + 2 * k, out_dir / f"{k + 1:04d}.jpg") for k in range(count)]
    if not all(p.exists() for _, p in paths):
        flt = "select='gte(n\\,10)*not(mod(n-10\\,40))',crop=240:100:400:0,scale=672:280:flags=lanczos"
        for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
            cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(HERE / v["file"]), "-vf", flt, *sync,
                   "-q:v", "2", str(out_dir / "%04d.jpg")]
            if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                break
        else:
            raise RuntimeError(f"ffmpeg 실패: {video}")
    return [(t, str(p.relative_to(HERE))) for t, p in paths if p.exists()]


def toast_corner(args):
    from typesafe_sdk import Image, Noul
    videos = read("data/mc_items.json")["videos"]
    jobs = [{"video": video, "t": t, "file": p} for video in sorted(videos) for t, p in corner_crops(video, videos[video])]
    print(f"알림·모서리: 영상 {len(videos)}개, 2초마다 한 장 = {len(jobs)}건. 어림: 약 {len(jobs) * 700:,}토큰, 약 ${len(jobs) * 700 * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return

    def one(client, j):
        state = {"about": "`corner` is the top-right corner of a Minecraft game screen, enlarged. Notification boxes appear in this corner.",
                 "corner": Image((HERE / j["file"]).read_bytes())}
        qs = {"advancement": Noul(instructions="Does `corner` show a notification box whose title reads 'Advancement Made!'?"),
              "recipe": Noul(instructions="Does `corner` show a notification box whose title reads 'New Recipes Unlocked!'?")}
        resp, error = ask(client, state, qs)
        rec = {**j, "error": error}
        if resp is not None:
            rec.update(advancement=resp.nouls["advancement"].noul, recipe=resp.nouls["recipe"].noul, input_tokens=resp.usage.input_tokens)
        return rec
    run_all("toast-corner", jobs, one, args.workers, {"crop": [400, 0, 240, 100], "every_seconds": 2})


def far_from(events, video, t, seconds):
    return not any(e["video"] == video and abs(e["t"] - t) <= seconds for e in events)


def dense_sheet(args):
    from typesafe_sdk import Image, Noul
    videos = read("data/mc_items.json")["videos"]
    busy = events_of(("fight", "hurt", "kill"))
    blocks = sorted({(e["video"], int(e["t"] // BLOCK)) for e in events_of(KINDS)})
    rng = random.Random(SEED)
    free = [(v, b) for v in sorted(videos) for b in range(int(videos[v]["duration"] // BLOCK))
            if all(far_from(busy, v, b * BLOCK + x, 2) for x in (0, BLOCK / 2, BLOCK))
            and not any(e["video"] == v and b * BLOCK <= e["t"] < (b + 1) * BLOCK for e in busy)]
    controls = sorted(rng.sample(free, CONTROL_BLOCKS))
    jobs = [{"video": v, "block": b, "role": role} for role, group in (("event", blocks), ("control", controls)) for v, b in group]
    print(f"촘촘한 시트: 사건이 든 15초 구간 {len(blocks)}개 + 견줄 구간 {len(controls)}개 = {len(jobs)}건(시트마다 60칸, 질문 120개). "
          f"어림: 약 {len(jobs) * 12000:,}토큰, 약 ${len(jobs) * 12000 * PRICE_PER_MTOK / 1e6:.3f}. 프레임 {len(jobs) * 60}장을 뽑는다")
    if args.dry_run:
        return
    ids = R.tile_ids(60)
    for j in jobs:
        times = [round(j["block"] * BLOCK + 0.125 + 0.25 * k, 3) for k in range(60)]
        paths = [frame(j["video"], t, videos) for t in times]
        from PIL import Image as PILImage
        with PILImage.open(HERE / paths[0]) as im:
            lay = R.choose_layout(im.width / im.height, 60)
        file = HERE / "data" / "sheets_mc4fps" / j["video"] / f"block-{j['block']:02d}.jpg"
        file.parent.mkdir(parents=True, exist_ok=True)
        R.compose(paths, ids, lay).save(file, format="JPEG", quality=90)
        j.update(times=times, file=str(file.relative_to(HERE)))
    print("시트 준비 끝", flush=True)

    def one(client, j):
        state = {"game": GAME,
                 "layout": ("`sheet` is a contact sheet of 60 frames from one video, 0.25 seconds apart, in time order from left to right "
                            "and top to bottom. Each tile has its ID (T01, T02, ...) in its top-left corner. `tiles` maps each ID to "
                            "its time in the video (minutes:seconds)."),
                 "tiles": {tid: f"{int(t) // 60}:{t % 60:05.2f}" for tid, t in zip(ids, j["times"])},
                 "sheet": Image((HERE / j["file"]).read_bytes())}
        qs = {f"{name}.{tid}": Noul(instructions=f"Does tile {tid} of `sheet` show this: {EVENTS[name]}") for name in KINDS for tid in ids}
        resp, error = ask(client, state, qs)
        rec = {**j, "error": error}
        if resp is not None:
            rec.update(nouls={name: [resp.nouls[f"{name}.{tid}"].noul for tid in ids] for name in KINDS}, input_tokens=resp.usage.input_tokens)
        return rec
    run_all("dense-sheet", jobs, one, args.workers, {"block_seconds": BLOCK, "events": {k: EVENTS[k] for k in KINDS}})


def frames_stage(args):
    from typesafe_sdk import Image, Noul
    videos = read("data/mc_items.json")["videos"]
    busy = events_of(("fight", "hurt", "kill"))
    wanted = sorted({(e["video"], round(e["t"] + 0.25 * k, 2)) for e in events_of(KINDS) for k in range(-4, 5)})
    rng = random.Random(SEED)
    pool = [(v, round(0.25 * k, 2)) for v in sorted(videos) for k in range(4, int(videos[v]["duration"] * 4) - 4)
            if far_from(busy, v, 0.25 * k, 5)]
    controls = sorted(rng.sample(pool, CONTROL_FRAMES))
    jobs = [{"video": v, "t": t, "role": role} for role, group in (("event", wanted), ("control", controls)) for v, t in group]
    print(f"원본 한 장: 사건 둘레 {len(wanted)}장 + 견줄 프레임 {len(controls)}장 = {len(jobs)}건. "
          f"어림: 약 {len(jobs) * 950:,}토큰, 약 ${len(jobs) * 950 * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    for j in jobs:
        j["file"] = frame(j["video"], j["t"], videos)
    print("프레임 준비 끝", flush=True)

    def one(client, j):
        state = {"game": GAME, "frame": Image((HERE / j["file"]).read_bytes())}
        qs = {name: Noul(instructions=f"Does `frame` show this: {EVENTS[name]}") for name in KINDS}
        resp, error = ask(client, state, qs)
        rec = {**j, "error": error}
        if resp is not None:
            rec.update({name: resp.nouls[name].noul for name in KINDS}, input_tokens=resp.usage.input_tokens)
        return rec
    run_all("frames", jobs, one, args.workers, {"events": {k: EVENTS[k] for k in KINDS}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["toast-sheet", "toast-corner", "dense-sheet", "frames"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    {"toast-sheet": toast_sheet, "toast-corner": toast_corner, "dense-sheet": dense_sheet, "frames": frames_stage}[args.stage](args)


if __name__ == "__main__":
    main()
