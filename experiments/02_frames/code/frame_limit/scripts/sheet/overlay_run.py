"""겹친 그림을 시트의 작은 칸이 아니라 원본 크기 한 장으로 Jev에 묻는다 (칸 크기 탓인지 가리는 확인).

판정 위치마다 그 시각 둘레의 4프레임을 겹쳐 한 장으로 만들고, 한 장 조건과 같은 식으로 묻는다.
- avg : 같은 비중으로 평균 낸 그림
- tint: 프레임마다 시간 순서대로 색(빨강, 노랑, 청록, 파랑)을 입혀 겹친 그림. 색의 뜻을 질문에 적는다
같은 영상의 질문은 한 요청에 함께 넣는다(tl_run.py와 같다). 질문은 TimeLens-Bench의 고친 질문이다.

실행: <venv>/bin/python experiments/frame_limit/scripts/sheet/overlay_run.py   (환경변수 TYPESAFE_API_KEY 필요)
      --dry-run   보내지 않고 요청 수와 어림 비용을 출력한다
      --smoke     종류마다 첫 요청 하나씩만 보내고 응답을 출력한다 (저장하지 않음)
      --tags tl_sta
결과: results/tl_overlay_preds.json. 결과 파일이 있으면 성공한 요청은 건너뛰고 나머지만 보낸다
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import io
import json
import os
import time
from pathlib import Path

import sheet_run as R

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
PRICE_PER_MTOK = 0.042
KINDS = ["avg", "tint"]
TOKENS_GUESS = 1300
NAMES = [name for name, _ in R.TINTS]
ASK = {
    "avg": ("`frame` is a multiple exposure made from one video: 4 frames taken within about 2 seconds are overlaid, so "
            "things that move appear as ghosted or blurred copies and things that stay still look sharp. "
            "Does it show the following: {s}?"),
    "tint": ("`frame` is a multiple exposure made from one video: 4 frames taken within about 2 seconds were tinted by "
             f"time order (the earliest {NAMES[0]}, the second {NAMES[1]}, the third {NAMES[2]}, the latest {NAMES[3]}) "
             "and overlaid. Things that stay still keep their normal colors. Things that move leave colored copies or "
             f"fringes, running from {NAMES[0]} (where it was first) to {NAMES[3]} (where it was last). "
             "Does it show the following: {s}?"),
}


def build_jobs(tags):
    jobs, frames = [], {}
    for tag in tags:
        frames[tag] = json.loads((HERE / "data" / f"{tag[3:]}_frames.json").read_text())
        by_video = {}
        for it in json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]:
            by_video.setdefault(it["video"], []).append(it)
        for video, items in by_video.items():
            for center in frames[tag][video]["centers"]:
                for kind in KINDS:
                    jobs.append((tag, video, center, kind, items))
    return jobs, frames


def key_of(job):
    return (job[0], job[1], job[2], job[3])


def picture(v, center, kind):
    paths = [v["frames"][f"{round(center + o, 2):.2f}"] for o in v["window_offsets"]]
    buf = io.BytesIO()
    R.overlay(paths, tint=kind == "tint").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run(client, job, frames):
    from typesafe_sdk import Image, Noul, TypeSafeAPIError
    tag, video, center, kind, items = job
    rec = {"tag": tag, "video": video, "t": center, "kind": kind, "error": None, "attempts": 0}
    state = {"frame": Image(picture(frames[tag][video], center, kind))}
    names = {f"q{k}": it["id"] for k, it in enumerate(items, 1)}
    qs = {f"q{k}": Noul(instructions=ASK[kind].format(s=it["sentence"])) for k, it in enumerate(items, 1)}
    resp = None
    for wait in (0, 3, 8, 20):
        time.sleep(wait)
        rec["attempts"] += 1
        try:
            resp = client.system_one(state=state, questions=qs, extra_headers=R.IMAGES)
            rec["error"] = None
            break
        except TypeSafeAPIError as err:
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
            if "BadRequest" in type(err).__name__ or "Authentication" in type(err).__name__:
                break
    if resp is None:
        return rec
    rec["nouls"] = {names[q]: a.noul for q, a in resp.nouls.items()}
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="tl_sta")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    jobs, frames = build_jobs(args.tags.split(","))
    out_path = HERE / "results" / "tl_overlay_preds.json"
    done = {}
    if out_path.exists():
        done = {(r["tag"], r["video"], r["t"], r["kind"]): r
                for r in json.loads(out_path.read_text())["records"] if not r["error"]}
    todo = [j for j in jobs if key_of(j) not in done]
    print(f"요청 {len(jobs)}건 중 보낼 것 {len(todo)}건(종류 {KINDS}, 요청당 질문 수 {sorted({len(j[4]) for j in jobs})}). "
          f"어림: 입력 약 {len(todo) * TOKENS_GUESS:,}토큰, 약 ${len(todo) * TOKENS_GUESS * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient
    if args.smoke:
        with TypeSafeClient(model=MODEL) as client:
            for kind in KINDS:
                r = run(client, next(j for j in todo if j[3] == kind), frames)
                print({k: r.get(k) for k in ("video", "t", "kind", "nouls", "input_tokens", "error")})
        return
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(lambda j: run(client, j, frames), todo):
            done[(r["tag"], r["video"], r["t"], r["kind"])] = r
    recs = [done[k] for k in map(key_of, jobs) if k in done]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "jobs": len(jobs), "saved": len(recs), "errors": sum(bool(r["error"]) for r in recs),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "wall_seconds_this_run": round(time.perf_counter() - t0, 1), "sent_this_run": len(todo),
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"저장 {meta['saved']}/{meta['jobs']}건, 오류 {meta['errors']}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, "
          f"{meta['wall_seconds_this_run']}초, 요청당 평균 {tokens / max(1, len(recs)):,.0f}토큰")


if __name__ == "__main__":
    main()
