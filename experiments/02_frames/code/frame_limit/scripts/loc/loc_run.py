"""위치 찾기 테스트: 영상의 1초 간격 위치마다 Jev에 "지금 이 동작을 하고 있나"를 묻는다.

조건 2×2
- 프레임: one(그 시각의 1장) / win4(그 시각을 가운데에 둔 4장, 0.5초 간격)
- 질문: plain(틀에 문장만 넣은 짧은 질문) / refined(구조화된 instructions + criteria)

요청 하나에 질문 하나만 넣는다. 결과 파일이 있으면 성공한 요청은 건너뛰고 나머지만 보낸다.

실행: <venv>/bin/python experiments/frame_limit/scripts/loc/loc_run.py   (환경변수 TYPESAFE_API_KEY 필요)
      --smoke          첫 항목의 한 위치만 보내고 응답을 출력한다 (저장하지 않음)
      --tag qv         data/qv_items.json, data/qv_frames.json을 읽고 results/qv_preds.json에 쓴다 (기본 loc)
      --name qv_parts  결과 파일 이름을 따로 정한다 (기본은 tag). 결과 파일은 이번에 고른 조건의 기록만 담는다
      --frames one     프레임 조건을 고른다 (기본 one,win4)
      --prompts plain  질문 조건을 고른다 (기본 plain,refined). parts는 data/<tag>_parts.json의 조건별 질문과
                       요청 전체 질문을 한 요청에 함께 넣는다
      --workers 8      동시에 보내는 요청 수 (기본 6)
      --parts-file F   parts 조건의 질문 목록 파일을 따로 정한다 (기본 data/<tag>_parts.json)
      --dry-run        보내지 않고 요청 수만 출력한다. 비용이 드는 실행 전에 먼저 확인한다
결과: results/<name>_preds.json
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

from typesafe_sdk import Image, Noul, NoulCriteria, TypeSafeAPIError, TypeSafeClient

from run_jev import IMAGES, MODEL, PRICE_PER_MTOK

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
FRAME_CONDS = ["one", "win4"]

PLAIN = {
    "one": "`frame` is one frame taken from a video. Does this frame show the following: {s}?",
    "win4": ("`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). "
             "Do these frames show the following: {s}?"),
}
INSPECT = {
    "one": "`frame`, one frame taken from a video",
    "win4": "`frames`, 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first)",
}


def question(item, cond, prompt):
    if prompt == "plain":
        return Noul(instructions=PLAIN[cond].format(s=item["sentence"]))
    r = item["refined"]
    focus = r["focus_one"] if cond == "one" else r["focus_window"]
    return Noul(instructions={"question": r["question"], "inspect": INSPECT[cond], "focus": focus},
                criteria=NoulCriteria(true=r["true"], false=r["false"]))


def questions(item, cond, prompt):
    """요청 하나에 넣을 질문들. parts는 요청 전체를 묻는 질문과 조건별 질문을 한 요청에 함께 넣는다."""
    if prompt != "parts":
        return {"q": question(item, cond, prompt)}
    qs = {"whole": Noul(instructions=PLAIN[cond].format(s=item["sentence"]))}
    for i, part in enumerate(item.get("parts") or [], 1):
        qs[f"c{i}"] = Noul(instructions=PLAIN[cond].format(s=part))
    return qs


def state_for(frames, center, cond):
    v = frames
    if cond == "one":
        return {"frame": Image((HERE / v["frames"][f"{center:.2f}"]).read_bytes())}
    times = [round(center + o, 2) for o in v["window_offsets"]]
    return {"frames": [{"seconds": t, "frame": Image((HERE / v["frames"][f"{t:.2f}"]).read_bytes())} for t in times]}


def run(client, job, frames):
    item, center, cond, prompt = job
    rec = {"item": item["id"], "video": item["video"], "t": center, "frames": cond, "prompt": prompt,
           "error": None, "attempts": 0}
    state, qs = state_for(frames[item["video"]], center, cond), questions(item, cond, prompt)
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
    nouls = {qid: a.noul for qid, a in resp.nouls.items()}
    rec["latency"] = round(time.perf_counter() - t0, 3)
    if "q" in nouls:
        rec["noul"] = nouls["q"]
    else:
        rec["nouls"] = nouls
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id)
    return rec


def key(r):
    return (r["item"], r["t"], r["frames"], r["prompt"])


def save(out_path, jobs, done, wall, sent):
    recs = [done[k] for k in ((j[0]["id"], j[1], j[2], j[3]) for j in jobs) if k in done]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "jobs": len(jobs), "saved": len(recs), "errors": sum(bool(r["error"]) for r in recs),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "wall_seconds_this_run": round(wall, 1), "sent_this_run": sent,
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="loc")
    ap.add_argument("--name", default=None)
    ap.add_argument("--frames", default=",".join(FRAME_CONDS))
    ap.add_argument("--prompts", default="plain,refined")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parts-file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_path = HERE / "results" / f"{args.name or args.tag}_preds.json"
    items = json.loads((HERE / "data" / f"{args.tag}_items.json").read_text())["items"]
    frames = json.loads((HERE / "data" / f"{args.tag}_frames.json").read_text())
    if "parts" in args.prompts.split(","):
        parts_path = HERE / (args.parts_file or f"data/{args.tag}_parts.json")
        parts = json.loads(parts_path.read_text())["parts"]
        for it in items:
            it["parts"] = parts[it["id"]]
    jobs = [(it, c, cond, prompt) for it in items for c in frames[it["video"]]["centers"]
            for cond in args.frames.split(",") for prompt in args.prompts.split(",")]
    if args.dry_run:  # 보내지 않고 요청 수만 센다
        done = ({key(r) for r in json.loads(out_path.read_text())["records"] if not r["error"]}
                if out_path.exists() else set())
        todo = sum((j[0]["id"], j[1], j[2], j[3]) not in done for j in jobs)
        questions_per_request = sorted({1 + len(j[0].get("parts") or []) if j[3] == "parts" else 1 for j in jobs})
        print(f"전체 {len(jobs)}건 중 보낼 요청 {todo}건, 요청당 질문 수 {questions_per_request}")
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    if args.smoke:
        with TypeSafeClient(model=MODEL) as client:
            for job in jobs[:4]:
                r = run(client, job, frames)
                print({k: r.get(k) for k in ("item", "t", "frames", "prompt", "noul", "nouls", "input_tokens", "error")})
        return
    done = {}
    if out_path.exists():
        done = {key(r): r for r in json.loads(out_path.read_text())["records"] if not r["error"]}
    todo = [j for j in jobs if (j[0]["id"], j[1], j[2], j[3]) not in done]
    print(f"전체 {len(jobs)}건 중 {len(todo)}건을 보냅니다", flush=True)
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, client, j, frames) for j in todo]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            r = fut.result()
            done[key(r)] = r
            if n % 100 == 0:  # 중간에 끊겨도 이어 할 수 있게 저장한다
                save(out_path, jobs, done, time.perf_counter() - t0, n)
                print(f"  {n}/{len(todo)}", flush=True)
    meta = save(out_path, jobs, done, time.perf_counter() - t0, len(todo))
    print(f"저장 {meta['saved']}/{meta['jobs']}건, 오류 {meta['errors']}건, 입력 토큰 {meta['input_tokens']:,}, "
          f"약 ${meta['cost_usd']}, {meta['wall_seconds_this_run']}초")
    for cond in FRAME_CONDS:
        t = [r["input_tokens"] for r in done.values() if r["frames"] == cond and not r["error"]]
        if t:
            print(f"  {cond}: 요청당 평균 {sum(t) / len(t):,.0f} 토큰")
    for r in [r for r in done.values() if r["error"]][:5]:
        print("  오류:", r["item"], r["t"], r["frames"], r["prompt"], r["error"])


if __name__ == "__main__":
    main()
