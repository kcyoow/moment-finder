"""벤치마크 질문이 붙은 영상에 Jev를 네 가지 방식으로 돌린다.

조건
- one    : 가운데 프레임 1장
- sep4   : 4장을 한 장씩 따로 묻는다 (요청 4건, 채점할 때 평균). 지금 파이프라인과 같은 방식
- joint4 : 4장을 시간 순서대로 한 요청에 넣는다
- joint8 : 8장을 시간 순서대로 한 요청에 넣는다

질문은 벤치마크 문장을 그대로 쓰고, 프레임이 영상에서 온 것이라는 설명만 앞에 붙인다.
TempCompass 예·아니오 질문은 Noul, TVBench 객관식은 Choice로 묻는다. 한 영상의 질문은 한 요청에 모두 넣는다.
단, start~end 구간이 지정된 질문(TVBench "언제 일어나나", "순서")은 그 구간의 프레임으로 질문마다 따로 묻는다.

실행: <venv>/bin/python experiments/frame_limit/scripts/base/run_jev.py   (환경변수 TYPESAFE_API_KEY 필요)
      --retry-errors 를 붙이면 기존 결과에서 성공한 요청은 그대로 두고 나머지(실패했거나 새로 생긴 요청)만 보낸다
결과: results/jev_preds.json
"""
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

from typesafe_sdk import Choice, Image, Noul, TypeSafeAPIError, TypeSafeClient

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
PRICE_PER_MTOK = 0.042

ONE = "`frame` is one frame taken from a video. Answer this question about the video: {q}"
JOINT = ("`frames` holds {n} frames sampled from one video, in time order (earliest first; `seconds` is the "
         "time in the video). Answer this question about the video: {q}")


def build_items(video):
    """영상 하나의 질문 목록. 각 항목: id, 종류, 질문, 정답, (객관식이면) 보기."""
    items = []
    for dim, qs in (video.get("yes_no") or {}).items():
        for k, q in enumerate(qs):
            items.append({"id": f"{dim}_{k}", "type": "noul", "dim": dim, "question": q["question"],
                          "truth": q["answer"]})
    for k, q in enumerate(video.get("qa") or []):
        options = {f"opt{j + 1}": c for j, c in enumerate(q["candidates"])}
        truth = next(key for key, c in options.items() if c == q["answer"])
        items.append({"id": f"mc_{k}", "type": "choice", "dim": video.get("task"), "question": q["question"],
                      "options": options, "truth": truth})
    return items


def questions(items, template, n=None):
    out = {}
    for it in items:
        text = template.format(q=it["question"], n=n)
        out[it["id"]] = (Noul(instructions=text) if it["type"] == "noul"
                         else Choice(instructions=text, criteria=it["options"]))
    return out


def run(client, job):
    video_id, condition, frame_index, rows, items, group = job
    images = [Image((HERE / r["file"]).read_bytes()) for r in rows]
    if condition in ("one", "sep4"):
        state, qs = {"frame": images[0]}, questions(items, ONE)
    else:
        state = {"frames": [{"seconds": r["seconds"], "frame": im} for r, im in zip(rows, images)]}
        qs = questions(items, JOINT, n=len(rows))
    rec = {"video": video_id, "group": group, "condition": condition, "frame_index": frame_index,
           "seconds": [r["seconds"] for r in rows], "error": None, "attempts": 0}
    resp = None
    for wait in (0, 3, 8, 20):  # 서버 쪽 일시 오류(504 등)는 간격을 늘려 가며 다시 시도한다
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
                break  # 요청 자체가 잘못된 경우는 다시 보내도 같다
    if resp is None:
        return rec
    rec["latency"] = round(time.perf_counter() - t0, 3)
    answers = {}
    for qid, a in resp.nouls.items():
        answers[qid] = {"noul": a.noul}
    for qid, a in resp.choices.items():
        answers[qid] = {"choice": a.choice, "probabilities": dict(a.probabilities), "confidence": a.confidence}
    rec.update(answers=answers, input_tokens=resp.usage.input_tokens, request_id=resp.request_id,
               model=getattr(resp, "model", None))
    return rec


def main():
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    videos = json.loads((HERE / "data" / "videos.json").read_text())["videos"]
    frames = json.loads((HERE / "data" / "frames.json").read_text())
    items_by_video, jobs = {}, []
    for v in videos:
        items = build_items(v)
        if not items:
            continue  # 질문이 없는 영상(Commons)은 이번 실험에서 뺀다
        items_by_video[v["id"]] = items
        f = frames[v["id"]]
        bounded = f.get("bounded", {})
        # 묶음 "": 영상 전체에서 뽑은 프레임으로 묻는 질문들. 묶음 "mc_k": 구간이 지정된 질문 하나
        groups = [("", f, [it for it in items if it["id"] not in bounded])]
        groups += [(qid, bounded[qid], [it for it in items if it["id"] == qid]) for qid in bounded]
        for group, fs, its in groups:
            if not its:
                continue
            jobs.append((v["id"], "one", 0, fs["mid"], its, group))
            jobs += [(v["id"], "sep4", i, [row], its, group) for i, row in enumerate(fs["f4"])]
            jobs.append((v["id"], "joint4", 0, fs["f4"], its, group))
            jobs.append((v["id"], "joint8", 0, fs["f8"], its, group))

    def rec_key(r):
        return (r["video"], r.get("group", ""), r["condition"], r["frame_index"])

    def job_key(j):
        return (j[0], j[5], j[1], j[2])

    out_path = HERE / "results" / "jev_preds.json"
    done = {}
    if "--retry-errors" in sys.argv and out_path.exists():
        # 이미 성공한 요청은 그대로 두고 나머지만 보낸다
        done = {rec_key(r): r for r in json.loads(out_path.read_text())["records"] if not r["error"]}
    todo = [j for j in jobs if job_key(j) not in done]
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(4) as pool:
        fresh = {rec_key(r): r for r in pool.map(lambda j: run(client, j), todo)}
    wall = time.perf_counter() - t0
    recs = [done.get(job_key(j)) or fresh[job_key(j)] for j in jobs]
    errors = [r for r in recs if r["error"]]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    out = {"meta": {"model": MODEL, "requests": len(recs), "errors": len(errors), "input_tokens": tokens,
                    "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4), "wall_seconds": round(wall, 1),
                    "sent_this_run": len(todo), "run_at": time.strftime("%Y-%m-%d %H:%M:%S")},
           "items": items_by_video, "records": recs}
    (HERE / "results").mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${out['meta']['cost_usd']}, {wall:.1f}초")
    by_cond = {}
    for r in recs:
        if not r["error"]:
            by_cond.setdefault(r["condition"], []).append(r["input_tokens"])
    for c, t in by_cond.items():
        print(f"  {c}: 요청당 평균 {sum(t) / len(t):,.0f} 토큰")
    for e in errors[:5]:
        print("  오류:", e["video"], e["condition"], e["error"])


if __name__ == "__main__":
    main()
