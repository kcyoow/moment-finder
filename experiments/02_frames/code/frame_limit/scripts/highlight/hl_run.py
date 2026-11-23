"""하이라이트 뽑기 첫 실험(마인크래프트): Jev 요청 두 종류를 보낸다.

막연한 요청("이 판의 하이라이트")을 구체 기준으로 바꾸는 단계는 원래 LLM이 맡을 부분이다. 여기서는 사건 목록·단계 설명을 손으로 썼다.
따른 논문은 없다. Score의 단계 작성법은 TypeSafe 문서(단계는 상황으로, Score 하나에 한 가지 차원, 코드에서 가중합)를 따랐다.

요청 A (--stage sheet): 60칸 시트 한 장에 사건 종류마다 칸별 Noul. 질문 id는 "<사건>.<칸>"
요청 B (--stage score): 후보 위치마다 프레임 4장(±0.25, ±0.75초, 원본 크기 640×360)에 Score action·progress + Choice kind + Noul worth
  후보 = 요청 A에서 어느 사건이든 0.5 이상인 칸을 이어 붙인 덩어리. 덩어리 안에서 --every초마다 한 곳(짧으면 가운데 한 곳).
         광석만 켜진 덩어리는 길어도 가운데 한 곳
  견줄 곳 = 영상마다 4곳, 기록된 사건과 켜진 칸이 앞뒤 3초 안에 없고 창이 닫힌 위치(고정 시드)

실행: <venv>/bin/python experiments/frame_limit/scripts/highlight/hl_run.py --stage sheet [--dry-run]     (환경변수 TYPESAFE_API_KEY 필요)
      <venv>/bin/python experiments/frame_limit/scripts/highlight/hl_run.py --stage score [--every 3] [--dry-run]
결과: results/hl_mc_sheet.json, results/hl_mc_score.json, data/hl_frames/…jpg(요청 B의 프레임)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import time
from pathlib import Path

import sheet_run as R
from extract_frames import grab

HERE = R.HERE
MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
PRICE_PER_MTOK = 0.042
CUT, SEED, CONTROLS = 0.5, 20261111, 4
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
GAME = "Minecraft survival gameplay, first-person view"
REQUEST = "the most exciting and important moments of this Minecraft session"
EVENTS = {
    "fight": "the player is hitting an animal or monster, or an animal or monster is flashing red from being hit",
    "hurt": ("the player's own view is tinted red or jolted because the player was just hurt "
             "(not an animal or monster being hit)"),
    "crafting": "a crafting table, furnace or inventory window is open on the screen",
    "ore": "an ore block with colored specks (coal, iron, gold, redstone or diamond) is right in front of the player",
}
ACTION = ["Calm: walking, looking around, digging or using a menu. Nothing is attacking or being attacked",
          "An animal or monster is close to the player, but nobody is hitting anybody",
          "The player is hitting an animal or monster, or is being hit",
          "The player is about to die or has died: almost no hearts left, or the death screen is shown"]
PROGRESS = ["No progress: moving around, or digging plain dirt or stone",
            "Gathering common materials such as wood, cobblestone or coal",
            "Making or using a workstation or tool: a crafting table, furnace, pickaxe or axe",
            "A rare find or top item: diamonds, or a diamond tool"]
KINDS = {"fight": "The player is fighting or hunting.", "hurt": "The player is taking damage.",
         "crafting": "A crafting, furnace or inventory window is open.", "mining": "The player is digging or mining blocks.",
         "moving": "The player is walking, running or looking around.", "other": "None of the above."}


def read(path):
    return json.loads((HERE / path).read_text())


def ask(client, state, questions):
    """서버 쪽 일시 오류는 간격을 늘려 가며 다시 시도한다. (응답, 오류 문자열)"""
    from typesafe_sdk import TypeSafeAPIError
    error = None
    for wait in (0, 3, 8, 20):
        time.sleep(wait)
        try:
            return client.system_one(state=state, questions=questions, extra_headers=IMAGES), None
        except TypeSafeAPIError as err:
            error = f"{type(err).__name__}: {err}"[:300]
            if "BadRequest" in type(err).__name__ or "Authentication" in type(err).__name__:
                break
    return None, error


def mc_sheets():
    return sorted((s for s in read("data/sheets60.json")["sheets"] if s["tag"] == "mc"), key=lambda s: (s["video"], s["sheet"]))


def stage_sheet(args):
    from typesafe_sdk import Image, Noul, TypeSafeClient
    sheets = mc_sheets()
    per = len(EVENTS) * 60
    print(f"요청 A: 시트 {len(sheets)}장, 요청당 질문 {per}개. 어림: 요청당 약 16,400토큰 → 약 {len(sheets) * 16400:,}토큰, "
          f"약 ${len(sheets) * 16400 * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return

    def one(sheet):
        state = {"game": GAME, **R.state_of(sheet, Image((HERE / sheet["file"]).read_bytes()))}
        qs = {f"{name}.{t['id']}": Noul(instructions=f"Does tile {t['id']} of `sheet` show this: {text}")
              for name, text in EVENTS.items() for t in sheet["tiles"]}
        resp, error = ask(client, state, qs)
        rec = {"video": sheet["video"], "sheet": sheet["sheet"], "error": error}
        if resp is not None:
            rec["nouls"] = {name: {t["id"]: resp.nouls[f"{name}.{t['id']}"].noul for t in sheet["tiles"]} for name in EVENTS}
            rec["input_tokens"] = resp.usage.input_tokens
        return rec

    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        recs = list(pool.map(one, sheets))
    save("results/hl_mc_sheet.json", recs, {"events": EVENTS})


def save(path, recs, extra):
    errors = [r for r in recs if r["error"]]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "requests": len(recs), "errors": len(errors), "input_tokens": tokens,
            "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4), "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (HERE / path).write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, 요청당 평균 {tokens / max(len(recs), 1):,.0f}토큰")
    for e in errors[:5]:
        print("  오류:", e["video"], e["error"])


def tile_scores():
    """영상 → 사건 → 위치(0~299)별 확률."""
    sheets = {(s["video"], s["sheet"]): s for s in mc_sheets()}
    out = {}
    for r in read("results/hl_mc_sheet.json")["records"]:
        assert not r["error"], "요청 A에 오류가 남아 있다"
        for name, by_tile in r["nouls"].items():
            row = out.setdefault(r["video"], {}).setdefault(name, {})
            for t in sheets[(r["video"], r["sheet"])]["tiles"]:
                row[t["pos"]] = by_tile[t["id"]]
    return {v: {name: [row[k] for k in sorted(row)] for name, row in kinds.items()} for v, kinds in out.items()}


def candidates(scores, every):
    """영상의 후보 위치: 어느 사건이든 CUT 이상인 칸의 덩어리에서 every칸마다 한 곳. [(위치, 덩어리 처음, 덩어리 끝)]

    광석(ore)만 켜진 덩어리는 가운데 한 곳만 본다(캐는 장면이 길게 이어져 후보가 어림의 두 배를 넘었기 때문에, 요청 B의 답을 보기 전에 정함).
    """
    n = len(next(iter(scores.values())))
    flags = [any(scores[name][k] >= CUT for name in scores) for k in range(n)]
    other = [any(scores[name][k] >= CUT for name in scores if name != "ore") for k in range(n)]
    out, k = [], 0
    while k < n:
        if not flags[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and flags[j + 1]:
            j += 1
        length = j - k + 1
        parts = math.ceil(length / every) if any(other[k:j + 1]) else 1
        for p in range(parts):
            out.append((k + min(length - 1, int((p + 0.5) * length / parts)), k, j))
        k = j + 1
    return out, flags


def plan(every):
    scores, frames = tile_scores(), read("data/mc_frames.json")
    events = read("data/mc_events.json")["events"]
    gui = {it["video"]: it["windows"] for it in read("data/mc_items.json")["items"] if it["kind"] == "gui"}
    rng = random.Random(SEED)
    jobs = []
    for video in sorted(scores):
        centers = frames[video]["centers"]
        cand, flags = candidates(scores[video], every)
        for pos, a, b in cand:
            jobs.append({"video": video, "t": centers[pos], "pos": pos, "role": "candidate", "run": [a, b],
                         "tile": {name: round(scores[video][name][pos], 3) for name in scores[video]}})
        busy = [e["t"] for e in events if e["video"] == video]
        quiet = [k for k, t in enumerate(centers)
                 if 2 <= k < len(centers) - 2 and not any(flags[max(0, k - 3):k + 4]) and not any(abs(t - x) <= 3 for x in busy)
                 and not any(s - 1 <= t <= e + 1 for s, e in gui[video])]
        for pos in sorted(rng.sample(quiet, min(CONTROLS, len(quiet)))):
            jobs.append({"video": video, "t": centers[pos], "pos": pos, "role": "control", "run": None,
                         "tile": {name: round(scores[video][name][pos], 3) for name in scores[video]}})
    return jobs


def stage_score(args):
    from typesafe_sdk import Choice, Image, Noul, Score, TypeSafeClient
    every = args.every
    jobs = plan(every)
    cand =sum(j["role"] == "candidate" for j in jobs)
    guess = len(jobs) * 3500
    print(f"요청 B: 후보 {cand}곳(덩어리 {len({(j['video'], tuple(j['run'])) for j in jobs if j['run']})}개, {every}초마다) + 견줄 곳 {len(jobs) - cand}곳 "
          f"= {len(jobs)}건. 어림: 약 {guess:,}토큰, 약 ${guess * PRICE_PER_MTOK / 1e6:.3f}. 프레임 {len(jobs) * 4}장을 뽑는다")
    if args.dry_run:
        return
    videos = read("data/mc_items.json")["videos"]
    for j in jobs:  # 프레임은 한 번만 뽑아 둔다
        last = videos[j["video"]]["duration"] - 0.1  # 영상 끝을 넘는 시각은 마지막 프레임 쪽으로 당긴다
        j["times"] = [round(min(max(j["t"] + off, 0.0), last), 2) for off in OFFSETS]
        j["frames"] = []
        for t in j["times"]:
            dest = HERE / "data" / "hl_frames" / j["video"] / f"{round(t * 1000):07d}.jpg"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                grab(HERE / videos[j["video"]]["file"], t, dest)  # 원본 크기(640×360) 그대로. 키워도 정보가 늘지 않는다
            j["frames"].append(str(dest.relative_to(HERE)))
    print("프레임 준비 끝", flush=True)

    def one(j):
        state = {"game": GAME, "viewer_request": REQUEST,
                 "about": "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first).",
                 "frames": [{"seconds": t, "frame": Image((HERE / p).read_bytes())} for t, p in zip(j["times"], j["frames"])]}
        qs = {"action": Score(instructions="How much is the player fighting or in danger in `frames`?", criteria=ACTION),
              "progress": Score(instructions="How much progress toward better tools do `frames` show?", criteria=PROGRESS),
              "kind": Choice(instructions="What is the player doing in `frames`?", criteria=KINDS),
              "worth": Noul(instructions="Would this moment fit `viewer_request`?")}
        resp, error = ask(client, state, qs)
        rec = {**j, "error": error}
        if resp is not None:
            for name in ("action", "progress"):
                a = resp.scores[name]
                rec[name] = {"score": a.score, "confidence": a.confidence, "probabilities": {str(k): v for k, v in a.probabilities.items()}}
            rec["kind"] = dict(resp.choices["kind"].probabilities)
            rec["worth"] = resp.nouls["worth"].noul
            rec["input_tokens"] = resp.usage.input_tokens
        return rec

    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        recs = list(pool.map(one, jobs))
    save("results/hl_mc_score.json", recs, {"every": every, "request": REQUEST, "action": ACTION, "progress": PROGRESS, "kinds": KINDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sheet", "score"])
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    (stage_sheet if args.stage == "sheet" else stage_score)(args)


if __name__ == "__main__":
    main()
