"""TimeLens-Bench의 고친 질문으로, 영상의 판정 위치마다 Jev에 "이 프레임에 이것이 보이나"를 묻는다.

loc_run.py와 같은 프레임, 같은 질문 틀(PLAIN)을 쓴다. 다른 점은 하나다: 같은 영상의 질문을 한 요청에 함께 넣는다
(이미지가 토큰의 대부분이라 질문을 묶으면 비용이 영상당 질문 수만큼 줄어든다. 저장된 QVHighlights 결과에서
질문을 묶었을 때와 따로 보냈을 때의 답 차이는 평균 0.007로, 같은 요청을 다시 보냈을 때의 차이 0.005와 비슷했다).

조건
- one : 그 시각의 1장            (세 묶음 모두)
- win4: 그 시각을 가운데에 둔 4장 (Charades, QVHighlights 본 영상. 지난 실행에서 4장을 잰 묶음과 같다)

실행: <venv>/bin/python experiments/frame_limit/scripts/loc/tl_run.py   (환경변수 TYPESAFE_API_KEY 필요)
      --dry-run   보내지 않고 요청 수와 어림 비용을 출력한다
      --smoke     묶음마다 첫 요청 하나씩만 보내고 응답을 출력한다 (저장하지 않음)
      --tags tl_sta,tl_qv,tl_qv2   tl_로 시작하지 않는 묶음(예: mc)은 data/<묶음>_items.json을 한 장 조건으로 묻는다
      --out tl    결과 파일 이름(results/<이름>_preds.json)
      --workers 6
결과: results/tl_preds.json. 결과 파일이 있으면 성공한 요청은 건너뛰고 나머지만 보낸다
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
PRICE_PER_MTOK = 0.042
CONDS = {"tl_sta": ["one", "win4"], "tl_qv": ["one", "win4"], "tl_qv2": ["one"], "tl_sta2": ["one", "win4"], "tl_sta2h": ["win4"]}
TOKENS_GUESS = {"one": 1300, "win4": 4400}  # 지난 실행의 요청당 평균. 보내기 전 어림에만 쓴다


def build_jobs(tags):
    jobs, frames = [], {}
    for tag in tags:
        base = tag[3:] if tag.startswith("tl_") else tag
        frames[tag] = json.loads((HERE / "data" / f"{base}_frames.json").read_text())
        by_video = {}
        for it in json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]:
            by_video.setdefault(it["video"], []).append(it)
        for video, items in by_video.items():
            for center in frames[tag][video]["centers"]:
                for cond in CONDS.get(tag, ["one"]):
                    jobs.append((tag, video, center, cond, items))
    return jobs, frames


def key_of(job):
    return (job[0], job[1], job[2], job[3])


def run(client, job, frames):
    from typesafe_sdk import Noul, TypeSafeAPIError

    from loc_run import PLAIN, state_for
    from run_jev import IMAGES
    tag, video, center, cond, items = job
    rec = {"tag": tag, "video": video, "t": center, "frames": cond, "error": None, "attempts": 0}
    state = state_for(frames[tag][video], center, cond)
    names = {f"q{k}": it["id"] for k, it in enumerate(items, 1)}
    qs = {f"q{k}": Noul(instructions=PLAIN[cond].format(s=it["sentence"])) for k, it in enumerate(items, 1)}
    resp = None
    for wait in (0, 3, 8, 20):  # 서버 쪽 일시 오류는 간격을 늘려 가며 다시 시도한다
        time.sleep(wait)
        rec["attempts"] += 1
        t0 = time.perf_counter()
        try:
            resp = client.system_one(state=state, questions=qs, extra_headers=IMAGES)
            rec["error"] = None
            break
        except TypeSafeAPIError as err:
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
            if "BadRequest" in type(err).__name__ or "Authentication" in type(err).__name__:
                break
    if resp is None:
        return rec
    rec["latency"] = round(time.perf_counter() - t0, 3)
    rec["nouls"] = {names[q]: a.noul for q, a in resp.nouls.items()}
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id)
    return rec


def save(out_path, jobs, done, wall, sent):
    recs = [done[k] for k in map(key_of, jobs) if k in done]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "jobs": len(jobs), "saved": len(recs), "errors": sum(bool(r["error"]) for r in recs),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "wall_seconds_this_run": round(wall, 1), "sent_this_run": sent, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="tl_sta,tl_qv,tl_qv2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="tl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    jobs, frames = build_jobs(args.tags.split(","))
    out_path = HERE / "results" / f"{args.out}_preds.json"
    done = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())["records"]
        wanted = set(map(key_of, jobs))
        lost = [r for r in existing if (r["tag"], r["video"], r["t"], r["frames"]) not in wanted]
        if lost:  # 이번에 고르지 않은 묶음의 기록이 지워지지 않게 멈춘다
            raise SystemExit(f"{out_path.name}에 이번 실행에 없는 기록 {len(lost)}건이 있다. --out으로 다른 이름을 준다")
        done = {(r["tag"], r["video"], r["t"], r["frames"]): r for r in existing if not r["error"]}
    todo = [j for j in jobs if key_of(j) not in done]
    guess = sum(TOKENS_GUESS[j[3]] for j in todo)
    for tag in args.tags.split(","):
        for cond in CONDS.get(tag, ["one"]):
            mine = [j for j in jobs if j[0] == tag and j[3] == cond]
            left = [j for j in mine if key_of(j) not in done]
            per = sorted({len(j[4]) for j in mine})
            print(f"{tag} {cond}: 요청 {len(mine)}건(보낼 것 {len(left)}건), 요청당 질문 수 {per}, "
                  f"질문 {len({it['id'] for j in mine for it in j[4]})}개")
    print(f"전체 {len(jobs)}건 중 보낼 것 {len(todo)}건. 어림: 입력 약 {guess:,}토큰, 약 ${guess * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient
    if args.smoke:
        seen = set()
        with TypeSafeClient(model=MODEL) as client:
            for job in todo:
                if (job[0], job[3]) in seen:
                    continue
                seen.add((job[0], job[3]))
                r = run(client, job, frames)
                print({k: r.get(k) for k in ("tag", "video", "t", "frames", "nouls", "input_tokens", "error")})
        return
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, client, j, frames) for j in todo]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            done[(r["tag"], r["video"], r["t"], r["frames"])] = r
            if n % 200 == 0:  # 중간에 끊겨도 이어 할 수 있게 저장한다
                save(out_path, jobs, done, time.perf_counter() - t0, n)
                print(f"  {n}/{len(todo)}", flush=True)
    meta = save(out_path, jobs, done, time.perf_counter() - t0, len(todo))
    print(f"저장 {meta['saved']}/{meta['jobs']}건, 오류 {meta['errors']}건, 입력 토큰 {meta['input_tokens']:,}, "
          f"약 ${meta['cost_usd']}, {meta['wall_seconds_this_run']}초")
    for cond in ("one", "win4"):
        t = [r["input_tokens"] for r in done.values() if r["frames"] == cond and not r["error"]]
        if t:
            print(f"  {cond}: 요청당 평균 {sum(t) / len(t):,.0f}토큰")
    for r in [r for r in done.values() if r["error"]][:5]:
        print("  오류:", r["tag"], r["video"], r["t"], r["frames"], r["error"])


if __name__ == "__main__":
    main()
