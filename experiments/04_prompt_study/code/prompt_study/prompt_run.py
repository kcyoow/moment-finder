"""H2O-Lightning-4B(/v1/systemone)에 같은 프레임 4장을 두고 질문 문장, criteria, state 모양만 바꿔 묻는다. 표준 라이브러리만 쓴다.

무엇을 보내나
- 항목 하나 = 영상의 한 위치(그 시각 −0.75, −0.25, +0.25, +0.75초의 프레임 4장). 그 영상의 질문을 모두 함께 묻는다
- state 모양(layout)마다 요청을 따로 보낸다. 같은 state 모양을 쓰는 변형들은 한 요청에 질문으로 함께 넣는다
  (H2O는 질문마다 따로 읽으므로 서로 영향을 주지 않는다. 서버 쪽 그림 처리와 올리는 양을 한 번으로 줄이려는 것이다)
- 변형의 문장은 variants.py에 있고, 결과를 보기 전에 정해 둔다

자료 (읽기만 한다)
- dev : Charades 30개 영상, 위치 959곳, 질문 100개. open_model 묶음의 charades 항목과 같다(프레임은 ../frame_limit/data/sta2_frames)
- hold: 다른 Charades 12개 영상, 위치 377곳, 질문 44개. 주석은 dev와 같은 TimeLens-Bench(../frame_limit/data/tl_sta_items.json).
        변형을 고르는 데 쓰지 않고, 고른 것을 확인하는 데만 쓴다

실행:
  python3 prompt_run.py --dry-run                                   # 보내지 않고 요청·질문 수만 센다
  python3 prompt_run.py --url https://… --auth gcloud --limit 12     # 형식 확인용 몇 건 (결과는 …_try12.json)
  python3 prompt_run.py --url https://… --auth gcloud                # 전부. 다시 돌리면 이어서 한다
결과: results/h2o__<layout>.json = {meta, records[{id, probs{변형: {질문 id: 확률}}, seconds, shim_ms, tokens, retries, error}]}
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"
OPEN = HERE.parent / "open_model"
sys.path.insert(0, str(HERE.parent / "grid4"))
from grid4_run import Fatal, Token, pct, post, uri  # noqa: E402  (보내기와 토큰은 grid4 실험의 것을 그대로 쓴다)
from variants import LAYOUTS, VARIANTS  # noqa: E402


def inside(t, windows):
    return any(s <= t <= e for s, e in windows)


def wide_frames(v, t):
    """1초 간격 4장(t−1.25 … t+1.75). 영상의 처음과 끝에서는 프레임이 있는 쪽으로 창을 0.5초씩 옮긴다."""
    for shift in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0):
        secs = [round(t + o + shift, 2) for o in (-1.25, -0.25, 0.75, 1.75)]
        if all(f"{x:.2f}" in v["frames"] for x in secs):
            return secs
    raise ValueError(f"1초 간격 4장을 만들 수 없음: t={t}")


def frame_sets(v, t, win):
    wide = wide_frames(v, t)
    return {"win": (win, [v["frames"][f"{x:.2f}"] for x in win]), "wide": (wide, [v["frames"][f"{x:.2f}"] for x in wide])}


def load_items():
    items = []
    frames2 = json.loads((SRC / "data" / "sta2_frames.json").read_text())
    for it in json.loads((OPEN / "pack" / "manifest.json").read_text())["items"]:
        if it["task"] == "charades":
            items.append({"id": it["id"], "set": "dev", "video": it["video"], "t": float(it["t"]),
                          "frames": frame_sets(frames2[it["video"]], float(it["t"]), it["seconds"]), "questions": it["questions"], "gt": it["gt"]})
    frames1 = json.loads((SRC / "data" / "sta_frames.json").read_text())
    by_video = {}
    for q in json.loads((SRC / "data" / "tl_sta_items.json").read_text())["items"]:
        by_video.setdefault(q["video"], []).append(q)
    for video in sorted(by_video):
        v = frames1[video]
        for t in v["centers"]:
            win = [round(t + o, 2) for o in v["window_offsets"]]
            items.append({"id": f"hold/{video}/{t}", "set": "hold", "video": video, "t": float(t), "frames": frame_sets(v, float(t), win),
                          "questions": {q["id"]: q["sentence"] for q in by_video[video]},
                          "gt": {q["id"]: int(inside(t, q["windows"])) for q in by_video[video]}})
    return items


def question(variant, sentence):
    v = VARIANTS[variant]
    q = {"type": "noul", "instructions": v["q"].format(s=sentence)}
    if v.get("criteria"):
        q["criteria"] = {k: text.format(s=sentence) for k, text in v["criteria"].items()}
    return q


def body(item, layout, variants, dry=False):
    """(요청 본문, {질문 키: (변형, 질문 id)}). dry이면 그림을 읽지 않고 자리만 채운다."""
    secs, paths = item["frames"][LAYOUTS[layout]["frames"]]
    uris = ["[그림]"] * len(paths) if dry else [uri(SRC / p) for p in paths]
    qs, names = {}, {}
    for i, variant in enumerate(variants):
        for j, (qid, sentence) in enumerate(item["questions"].items()):
            qs[f"v{i}q{j}"] = question(variant, sentence)
            names[f"v{i}q{j}"] = (variant, qid)
    return {"state": LAYOUTS[layout]["state"](secs, uris), "questions": qs}, names


def ask(url, token, stop, item, layout, variants):
    payload, names = body(item, layout, variants)
    out, seconds, retries = post(url, token, payload, stop)
    probs = {}
    for key, (variant, qid) in names.items():
        probs.setdefault(variant, {})[qid] = float(out["answers"][key]["noul"])
    usage = out.get("usage") or {}
    return {"id": item["id"], "probs": probs, "seconds": round(seconds, 4), "shim_ms": usage.get("latency_ms"),
            "tokens": usage.get("input_tokens"), "retries": retries, "error": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="서비스 주소(끝에 /v1/systemone을 붙여 보낸다)")
    ap.add_argument("--auth", default="none", choices=["none", "gcloud", "metadata"])
    ap.add_argument("--layouts", default="", help="쉼표로 나눈 state 모양. 비우면 variants.py에 나오는 것 전부")
    ap.add_argument("--only", default="", help="쉼표로 나눈 변형 이름. 비우면 전부")
    ap.add_argument("--sets", default="dev,hold")
    ap.add_argument("--name", default="h2o")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="고르게 이만큼만(0이면 전부). 결과는 …_tryN.json에 따로 둔다")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    only = [v for v in args.only.split(",") if v] or list(VARIANTS)
    by_layout = {}
    for name in only:
        by_layout.setdefault(VARIANTS[name]["layout"], []).append(name)
    layouts = [x for x in args.layouts.split(",") if x] or list(by_layout)
    items = [it for it in load_items() if it["set"] in args.sets.split(",")]
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]

    if args.dry_run:
        total_reads = 0
        for layout in layouts:
            n_q = sum(len(it["questions"]) for it in items) * len(by_layout[layout])
            total_reads += n_q
            payload, _ = body(items[0], layout, by_layout[layout], dry=True)
            print(f"{layout}: 요청 {len(items)}건, 변형 {len(by_layout[layout])}개 {by_layout[layout]}, 질문(모델이 읽는 횟수) {n_q}개")
            print("  state:", json.dumps(payload["state"], ensure_ascii=False)[:200])
            for key in list(payload["questions"])[::len(items[0]["questions"])]:
                print("  ", key, json.dumps(payload["questions"][key], ensure_ascii=False)[:420])
        sets = {s: sum(it["set"] == s for it in items) for s in ("dev", "hold")}
        print(f"합계: 요청 {len(items) * len(layouts)}건 (dev {sets['dev']}, hold {sets['hold']} × state 모양 {len(layouts)}), 읽는 횟수 {total_reads}")
        return

    if not args.url:
        raise SystemExit("--url 이 필요하다")
    url, token = args.url.rstrip("/") + "/v1/systemone", Token(args.auth, args.url.rstrip("/"))
    try:
        health = urllib.request.urlopen(urllib.request.Request(args.url.rstrip("/") + "/health", headers=token.header()), timeout=60).read()[:400].decode(errors="replace")
    except Exception as err:
        raise SystemExit(f"서버가 준비되지 않음: {err}")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    for layout in layouts:
        variants = by_layout[layout]
        out_path = out_dir / (f"{args.name}__{layout}_try{args.limit}.json" if args.limit else f"{args.name}__{layout}.json")
        done, runs = {}, []
        if out_path.exists():
            before = json.loads(out_path.read_text())
            # 변형이 빠진 기록은 다시 묻는다
            done = {r["id"]: r for r in before["records"] if not r["error"] and all(v in r["probs"] for v in variants)}
            runs = before["meta"].get("runs") or []
        todo = [it for it in items if it["id"] not in done]
        meta = {"model": "h2o-lightning-4b", "repo": "h2oai/h2o-lightning-4b", "url": args.url, "server": health, "note": args.note,
                "date": time.strftime("%Y-%m-%d %H:%M"), "layout": layout, "frames": LAYOUTS[layout]["frames"], "variants": {v: VARIANTS[v] for v in variants},
                "options": {"workers": args.workers, "limit": args.limit}}
        print(f"{layout}: 보낼 요청 {len(todo)}건 (전체 {len(items)}건), 변형 {len(variants)}개, 동시 {args.workers}개", flush=True)

        def save():
            recs = list(done.values())
            meta.update(tokens_total=sum(r.get("tokens") or 0 for r in recs), retries_total=sum(r.get("retries") or 0 for r in recs))
            out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False) + "\n")

        stop, errors, shown, start = threading.Event(), [], 0, time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(ask, url, token, stop, it, layout, variants): it for it in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                item = futures[fut]
                try:
                    rec = fut.result()
                    done[item["id"]] = rec
                    if args.show and shown < 1:
                        shown += 1
                        print("답:", json.dumps(rec, ensure_ascii=False)[:600], flush=True)
                except Fatal as err:
                    if str(err) != "멈춤":
                        print("멈춤:", err, flush=True)
                except Exception as err:
                    errors.append(f"{item['id']}: {type(err).__name__}: {err}"[:400])
                    print("오류", errors[-1], flush=True)
                    if len(errors) >= 10 and len(errors) > len(done):
                        stop.set()
                if n % 100 == 0:
                    save()
                    print(f"  {n}/{len(todo)} {time.perf_counter() - start:.0f}초", flush=True)
        runs.append({"items_run": len(todo), "wall_seconds": round(time.perf_counter() - start, 1), "errors": len(errors)})
        meta.update(runs=runs, wall_seconds=round(sum(r["wall_seconds"] for r in runs), 1), items_run=sum(r["items_run"] for r in runs), errors=errors[:20])
        save()
        recs = list(done.values())
        print(f"{layout} 끝: 답 {len(recs)}건, 오류 {len(errors)}건, 왕복 중앙값 {pct([r['seconds'] for r in recs], 0.5)}초, "
              f"서버 안 {pct([r['shim_ms'] for r in recs], 0.5)}ms, 전체 {runs[-1]['wall_seconds']}초 → {out_path}", flush=True)
        if stop.is_set():
            raise SystemExit("멈춤: 오류가 많다")


if __name__ == "__main__":
    main()
