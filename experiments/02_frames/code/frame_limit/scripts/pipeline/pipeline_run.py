"""2단계 흐름의 확인 단계: 시트가 낸 칸별 확률에서 후보를 뽑아, 후보 둘레를 원본 프레임 묶음으로 Jev에 다시 묻는다.

흐름 (설정은 실행 전에 정해 둔 것)
1. 시트 단계는 sheet_run.py가 이미 받은 답(칸마다 Noul)을 읽는다
2. 확률이 높은 순으로 후보 k곳을 고른다. 이미 고른 칸의 바로 옆 칸은 건너뛴다
3. 후보마다 요청 하나: 후보 위치와 앞뒤 1초, 세 위치의 프레임을 한 요청에 넣는다.
   위치마다 프레임 4장(그 시각 −0.75, −0.25, +0.25, +0.75초)이고 이웃 위치가 2장씩 겹쳐 모두 8장이 된다.
   질문은 위치마다 하나: "이 4장에 이것이 보이는가"
   (지금까지의 4장 조건은 위치마다 요청을 따로 보냈다. 프레임을 함께 넣으면 같은 판정을 3분의 2 토큰으로 받는다.
    함께 넣어도 답이 같은지는 pipeline_eval.py에서 따로 보낸 답과 견준다)
4. 확인 확률이 가장 높은 위치가 답이다 (pipeline_eval.py)

실행: <venv>/bin/python experiments/frame_limit/scripts/pipeline/pipeline_run.py --tag sta   (환경변수 TYPESAFE_API_KEY 필요)
      --k 2          후보 수
      --side 768     확인 프레임의 긴 변(화소). 저장된 프레임이 768이고, 더 작게 주면 줄여서 보낸다
      --sheet-preds results/tl_sheet60_preds.json --sheet-index data/sheets60.json   시트 단계의 답과 시트 목록
      --dry-run / --smoke / --workers 6
결과: results/pipe_<묶음>_s<side>.json (후보마다 기록 하나). 이미 받은 요청은 건너뛴다
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import time
from pathlib import Path

from PIL import Image as PILImage

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
PRICE_PER_MTOK = 0.042
MIN_SIDE = 280


def candidates(ps, k):
    """확률이 높은 순으로 k곳. 이미 고른 칸의 바로 옆 칸은 건너뛴다."""
    chosen = []
    for p in sorted(range(len(ps)), key=lambda q: -ps[q]):
        if all(abs(p - c) > 1 for c in chosen):
            chosen.append(p)
        if len(chosen) == k:
            break
    return chosen


def tile_probs(items, tag, index_file, preds_file):
    """항목마다 위치별 칸 확률(여러 시트에 든 위치는 평균)과 시트 요청의 토큰 합."""
    sheets = [s for s in json.loads((HERE / index_file).read_text())["sheets"] if s["tag"] == tag]
    recs = {(r["item"], r["sheet"]): r for r in json.loads((HERE / preds_file).read_text())["records"]
            if r["tag"] == tag and r["kind"] == "tiles" and not r["error"]}
    out = {}
    for it in items:
        cells, tokens = {}, 0
        for sh in (s for s in sheets if s["video"] == it["video"]):
            r = recs[(it["id"], sh["sheet"])]
            tokens += r["input_tokens"]
            for t in sh["tiles"]:
                cells.setdefault(t["pos"], []).append(r["nouls"][t["id"]])
        out[it["id"]] = {"probs": [sum(cells[k]) / len(cells[k]) for k in sorted(cells)], "tokens": tokens}
    return out


def frame_bytes(path, side):
    """프레임을 긴 변 side로 줄여 JPEG로 낸다. 이미 그 크기 이하이면 파일 그대로."""
    with PILImage.open(HERE / path) as im:
        if max(im.size) <= side:
            return (HERE / path).read_bytes()
        scale = side / max(im.size)
        size = (round(im.width * scale), round(im.height * scale))
        assert min(size) >= MIN_SIDE, f"{path}: {size}는 Jev의 최소 크기보다 작다"
        buf = io.BytesIO()
        im.convert("RGB").resize(size, PILImage.Resampling.LANCZOS).save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def build_job(item, video_frames, rank, pos):
    """후보 하나의 요청: 세 위치(pos−1, pos, pos+1)의 프레임과 위치마다의 질문."""
    centers, offsets = video_frames["centers"], video_frames["window_offsets"]
    positions = [q for q in (pos - 1, pos, pos + 1) if 0 <= q < len(centers)]
    times = sorted({round(centers[q] + o, 2) for q in positions for o in offsets})
    windows = {q: [times.index(round(centers[q] + o, 2)) for o in offsets] for q in positions}
    return {"item": item, "rank": rank, "pos": pos, "positions": positions, "times": times, "windows": windows}


def run(client, job, frames, side):
    from typesafe_sdk import Image, Noul, TypeSafeAPIError
    item, v = job["item"], frames[job["item"]["video"]]
    rec = {"item": item["id"], "video": item["video"], "rank": job["rank"], "pos": job["pos"], "positions": job["positions"],
           "frames": len(job["times"]), "error": None, "attempts": 0}
    state = {"about": (f"`frames` holds {len(job['times'])} consecutive frames from one stretch of a video, 0.5 seconds apart, "
                       "in time order (earliest first)."),
             "frames": [{"seconds": t, "frame": Image(frame_bytes(v["frames"][f"{t:.2f}"], side))} for t in job["times"]]}
    qs = {f"p{q}": Noul(instructions=(f"Do the {len(idx)} consecutive frames `frames[{idx[0]}]` to `frames[{idx[-1]}]` "
                                      f"show the following: {item['sentence']}?"))
          for q, idx in job["windows"].items()}
    resp = None
    for wait in (0, 3, 8, 20):
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
    rec["nouls"] = {int(q[1:]): a.noul for q, a in resp.nouls.items()}
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sta")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--sheet-preds", default="results/tl_sheet60_preds.json")
    ap.add_argument("--sheet-index", default="data/sheets60.json")
    ap.add_argument("--items", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    items = json.loads((HERE / (args.items or f"data/tl_{args.tag}_items.json")).read_text())["items"]
    frames = json.loads((HERE / "data" / f"{args.tag}_frames.json").read_text())
    stage1 = tile_probs(items, args.tag, args.sheet_index, args.sheet_preds)
    jobs = []
    for it in items:
        for rank, pos in enumerate(candidates(stage1[it["id"]]["probs"], args.k), 1):
            jobs.append(build_job(it, frames[it["video"]], rank, pos))
    out_path = HERE / "results" / f"pipe_{args.tag}_s{args.side}.json"
    done = {}
    if out_path.exists():
        done = {(r["item"], r["rank"]): r for r in json.loads(out_path.read_text())["records"] if not r["error"]}
        moved = [j for j in jobs if (j["item"]["id"], j["rank"]) in done and done[(j["item"]["id"], j["rank"])]["pos"] != j["pos"]]
        assert not moved, "저장된 후보 위치가 지금의 시트 답과 다르다. 결과 파일을 따로 두고 다시 돌린다"
    todo = [j for j in jobs if (j["item"]["id"], j["rank"]) not in done]
    per_frame = 1050 * (args.side / 768) ** 2  # 긴 변 768 프레임 한 장이 약 1,050토큰이었다
    guess = sum(len(j["times"]) * per_frame + 300 for j in todo)
    print(f"질문 {len(items)}개, 후보 {args.k}곳씩 요청 {len(jobs)}건 중 보낼 것 {len(todo)}건 (요청당 프레임 "
          f"{min(len(j['times']) for j in jobs)}~{max(len(j['times']) for j in jobs)}장, 긴 변 {args.side}px). "
          f"어림: 입력 약 {guess:,.0f}토큰, 약 ${guess * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient
    if args.smoke:
        with TypeSafeClient(model=MODEL) as client:
            print(json.dumps(run(client, todo[0], frames, args.side), ensure_ascii=False)[:800])
        return
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        fresh = {(r["item"], r["rank"]): r for r in pool.map(lambda j: run(client, j, frames, args.side), todo)}
    recs = [done.get(k) or fresh[k] for k in ((j["item"]["id"], j["rank"]) for j in jobs)]
    errors = [r for r in recs if r["error"]]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "tag": args.tag, "k": args.k, "side": args.side, "sheet_preds": args.sheet_preds,
            "requests": len(recs), "errors": len(errors), "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "sent_this_run": len(todo), "wall_seconds_this_run": round(time.perf_counter() - t0, 1),
            "stage1_tokens": {k: v["tokens"] for k, v in stage1.items()}, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, 요청당 평균 {tokens / len(recs):,.0f}토큰")
    for e in errors[:5]:
        print("  오류:", e["item"], e["rank"], e["error"])


if __name__ == "__main__":
    main()
