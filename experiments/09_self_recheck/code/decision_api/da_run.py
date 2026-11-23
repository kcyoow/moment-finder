"""받아 둔 후보(pipeline/verify_runs, verify_rr)를 OpenRouter의 판정 모델에 묻는다. README.md 3절. 표준 라이브러리만 쓴다.

후보 하나 = 요청 하나. 공유 입력: 안내 한 줄 + 프레임 8장. 질문 넷: noul_plain, noul_def, choice_fwd, choice_rev.
키는 환경 변수 OPENROUTER_API_KEY, 없으면 --env-file에서 읽는다. 어디에도 찍지 않는다.

  python3 da_run.py --key-info                                   # 키의 한도와 쓴 돈 (돈 없음)
  python3 da_run.py --dry-run                                    # 보내지 않고 요청 수와 예상 토큰
  python3 da_run.py --model microsoft --limit 6 --show           # 6건만 → results/<모델>_try6.json
  python3 da_run.py --model microsoft --budget 0.5               # 전부 → results/<모델>.json (이어서 한다)
"""
import argparse
import base64
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
URL = "https://openrouter.ai/api/alpha/decisions"
MODELS = {"microsoft": "microsoft/microsoft-decision-1.1", "gemini": "google/gemini-decision-1"}
PRICE = {"microsoft": 0.042, "gemini": 0.10}          # 달러 / 입력 100만 토큰 (2026-11-23에 읽음)
DIRS = ("verify_runs", "verify_rr")
LEAD = ("These 8 frames come from one candidate segment (about {n} seconds long) of a longer video. "
        "Frames 1 to 4 are consecutive frames, 0.5 seconds apart, from one second of the segment. "
        "Frames 5 to 8 are spread evenly across the whole segment, in time order.")
PLAIN = "Does this segment show what the viewer asked for: {vr}?"
CHOICE = "Which option best describes what happens in this segment?"
NONE = "None of the other options is visible in these frames."
RETRY, STOP, TRIES = (408, 429, 500, 502, 503, 504), (401, 402, 403), 6


class Fatal(Exception):
    """키, 권한, 잔액 문제. 다시 보내도 같으므로 전체를 멈춘다."""


def load_key(env_file):
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY가 없다(환경 변수 또는 --env-file)")
    return key


def key_info(key):
    for path in ("/api/v1/key", "/api/v1/auth/key"):
        try:
            with urllib.request.urlopen(urllib.request.Request("https://openrouter.ai" + path, headers={"Authorization": f"Bearer {key}"}), timeout=30) as r:
                d = json.loads(r.read()).get("data", {})
            return {k: d.get(k) for k in ("limit", "usage", "limit_remaining", "is_free_tier")}   # label에는 키의 일부가 들어 있어 뺀다
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
    return {"error": last}


def load_specs():
    specs = {}
    for p in sorted((HERE / "specs").glob("*.json")):
        if "_response_" in p.name:
            continue
        s = json.loads(p.read_text())
        specs[s["request"]] = s
    return specs


def load_candidates():
    out = []
    for d in DIRS:
        for item_path in sorted((ROOT / "pipeline" / d).glob("*/item.json")):
            item = json.loads(item_path.read_text())
            for c in item["candidates"]:
                out.append({"uid": f"{d}/{item['name']}#{c['id']}", "dir": item_path.parent, "request": item["request"], "length": c["length"], "frames": c["frames"]})
    return out


def questions_of(spec):
    opts = [(spec["target"]["key"], spec["target"]["is"])] + [(c["key"], c["is"]) for c in spec["confusers"]] + [("none_of_these", NONE)]
    return {"noul_plain": {"type": "noul", "instructions": PLAIN.format(vr=spec["viewer_request"])},
            "noul_def": {"type": "noul", "instructions": spec["noul"]["instructions"], "criteria": spec["noul"]["criteria"]},
            "choice_fwd": {"type": "choice", "instructions": CHOICE, "criteria": dict(opts)},
            "choice_rev": {"type": "choice", "instructions": CHOICE, "criteria": dict(reversed(opts))}}


def body_of(model, cand, spec, detail=""):
    state = [LEAD.format(n=round(cand["length"]))]
    for f in cand["frames"]:
        raw = (cand["dir"] / "frames" / f).read_bytes()
        part = {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}
        state.append({"type": "image_url", "image_url": part | ({"detail": detail} if detail else {})})   # OpenRouter는 Gemini에 low / high만 넘긴다
    return {"model": MODELS[model], "state": state, "questions": questions_of(spec)}


def post(headers, payload, stop):
    data = json.dumps(payload).encode()
    for attempt in range(TRIES):
        if stop.is_set():
            raise Fatal("멈춤")
        req = urllib.request.Request(URL, data=data, headers=headers | {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            if "answers" not in out:
                raise ValueError(f"답이 없음: {json.dumps(out, ensure_ascii=False)[:300]}")
            return out, time.perf_counter() - t0, attempt
        except urllib.error.HTTPError as err:
            msg = err.read()[:600].decode(errors="replace")
            try:
                e = json.loads(msg)["error"]
                msg = f"{e.get('status') or e.get('code')}: {e.get('message')}"
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
            if err.code in STOP:
                stop.set()
                raise Fatal(f"HTTP {err.code} {msg[:300]}") from None
            if err.code in RETRY and attempt < TRIES - 1:
                time.sleep(float(err.headers.get("Retry-After") or 0) or min(60.0, 2 ** attempt + random.random()))
                continue
            raise ValueError(f"HTTP {err.code} {msg[:300]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt < TRIES - 1:
                time.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            raise ValueError(f"연결 실패: {err}") from None


def read_answers(out, target):
    """질문마다 값 하나. noul은 예일 확률, choice는 target이 받은 확률. 답을 못 받은 것은 None."""
    vals, picks = {}, {}
    for q, a in out["answers"].items():
        if a.get("type") == "noul" and isinstance(a.get("noul"), (int, float)):
            vals[q] = float(a["noul"])
        elif a.get("type") == "choice" and isinstance(a.get("probabilities"), dict):
            vals[q] = float(a["probabilities"].get(target, 0.0))
            picks[q] = {"choice": a.get("choice"), "probabilities": {k: round(float(v), 4) for k, v in a["probabilities"].items()}}
        else:
            vals[q] = None
            picks[q] = {"type": a.get("type")}
    return vals, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="항목을 건너뛰며 고르게 이만큼만. 결과는 <모델>_tryN.json에 따로 둔다")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.5, help="이 모델에 쓴 돈(usage.cost의 합)이 이 값을 넘으면 멈춘다")
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--key-info", action="store_true")
    ap.add_argument("--show", action="store_true", help="처음 두 요청의 답을 그대로 보여 준다")
    ap.add_argument("--detail", default="", choices=["", "low", "high"], help="그림 해상도. 비우면 API의 기본값. 결과는 <모델>_<detail>.json에 따로 둔다")
    args = ap.parse_args()
    if args.key_info:
        print(json.dumps(key_info(load_key(args.env_file)), ensure_ascii=False))
        return
    specs, cands = load_specs(), load_candidates()
    missing = sorted({c["request"] for c in cands} - set(specs))
    if missing:
        raise SystemExit(f"질문이 없는 요청: {missing} (da_write.py를 먼저 돌린다)")
    if args.dry_run:
        chars = sum(len(json.dumps(questions_of(specs[c["request"]]))) + len(LEAD) for c in cands)
        n_img = sum(len(c["frames"]) for c in cands)
        print(f"후보 {len(cands)}개 = 모델마다 요청 {len(cands)}건, 그림 {n_img}장, 질문 {len(cands) * 4}개, 글 약 {chars // 4:,}토큰")
        for m, per_img in (("gemini", (1120, 1120)), ("microsoft", (280, 1120))):
            lo, hi = (n_img * per_img[0] + chars // 4, n_img * per_img[1] + chars // 4)
            worst = hi + 3 * n_img * per_img[1] if m == "microsoft" else hi
            print(f"  {m:9s} 예상 토큰 {lo:,}~{hi:,} ≈ ${lo / 1e6 * PRICE[m]:.2f}~{hi / 1e6 * PRICE[m]:.2f}" +
                  (f" (공유 그림이 질문마다 과금되면 ${worst / 1e6 * PRICE[m]:.2f})" if m == "microsoft" else ""))
        return
    if not args.model:
        raise SystemExit("--model이 필요하다")
    if args.limit:
        cands = cands[::max(1, len(cands) // args.limit)][:args.limit]
    stem = f"{args.model}_{args.detail}" if args.detail else args.model
    out_path = HERE / "results" / (f"{stem}_try{args.limit}.json" if args.limit else f"{stem}.json")
    done, runs = {}, []
    if out_path.exists():
        before = json.loads(out_path.read_text())
        done = {r["uid"]: r for r in before["records"] if not r.get("error")}
        runs = before["meta"].get("runs") or []
    todo = [c for c in cands if c["uid"] not in done]
    print(f"{MODELS[args.model]}: 보낼 요청 {len(todo)}건 (전체 {len(cands)}건 중 이미 받은 것 {len(cands) - len(todo)}건), 멈추는 선 ${args.budget}")
    if not todo:
        return
    headers = {"Authorization": f"Bearer {load_key(args.env_file)}"}
    meta = {"model": MODELS[args.model], "detail": args.detail or "기본", "url": URL, "date": time.strftime("%Y-%m-%d"), "lead": LEAD, "plain": PLAIN, "choice": CHOICE, "none": NONE, "served_by": None}
    stop, lock = threading.Event(), threading.Lock()

    def cost_now():
        return sum(r.get("cost") or 0 for r in done.values())

    def work(cand):
        spec = specs[cand["request"]]
        out, seconds, retries = post(headers, body_of(args.model, cand, spec, args.detail), stop)
        vals, picks = read_answers(out, spec["target"]["key"])
        usage = out.get("usage") or {}
        if out.get("provider"):
            meta["served_by"] = out["provider"]
        return out, {"uid": cand["uid"], "p": vals, "picks": picks, "seconds": round(seconds, 3), "retries": retries, "error": None,
                     "tokens": usage.get("input_tokens"), "cost": usage.get("cost")}

    def save():
        recs = list(done.values())
        meta.update(tokens_total=sum(r.get("tokens") or 0 for r in recs), cost_usd=round(cost_now(), 5))
        out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False) + "\n")

    errors, shown, start, fatal = [], 0, time.perf_counter(), None
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(work, c): c for c in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            cand = futures[fut]
            try:
                raw, rec = fut.result()
                with lock:
                    done[cand["uid"]] = rec
                    if cost_now() > args.budget and not stop.is_set():
                        stop.set()
                        fatal = f"멈추는 선 ${args.budget}를 넘었다"
                        print("멈춤:", fatal, flush=True)
                if args.show and shown < 2:
                    shown += 1
                    print("답 그대로:", json.dumps(raw, ensure_ascii=False)[:1200], flush=True)
            except Fatal as err:
                if fatal is None and str(err) != "멈춤":
                    fatal = str(err)
                    print("멈춤:", fatal, flush=True)
            except Exception as err:
                errors.append(f"{cand['uid']}: {type(err).__name__}: {err}"[:400])
                print("오류", errors[-1], flush=True)
                if len(errors) >= 10 and len(errors) > len(done):
                    stop.set()
            if n % 50 == 0:
                save()
                print(f"{n}/{len(todo)} {time.perf_counter() - start:.0f}초, 토큰 {meta['tokens_total']:,}, ${meta['cost_usd']}", flush=True)
    runs.append({"items_run": len(todo), "wall_seconds": round(time.perf_counter() - start, 1), "errors": len(errors)})
    meta.update(runs=runs, errors=errors[:20], fatal=fatal)
    save()
    recs = list(done.values())
    times = sorted(r["seconds"] for r in recs)
    toks = sorted(r["tokens"] or 0 for r in recs)
    print(f"끝: 답 {len(recs)}개, 오류 {len(errors)}개" + (f", 멈춤({fatal})" if fatal else "") +
          f", 요청당 중앙값 {times[len(times) // 2] if times else None}초 · {toks[len(toks) // 2] if toks else None}토큰, "
          f"토큰 {meta['tokens_total']:,}, ${meta['cost_usd']} (usage.cost의 합), 보낸 곳 {meta['served_by']} → {out_path.name}")


if __name__ == "__main__":
    main()
