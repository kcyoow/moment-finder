"""H2O-Lightning-4B(/v1/systemone)에 같은 프레임 4장을 "따로 4장" 또는 "2×2 한 그림"으로 물어 답을 받는다. 속도 시험도 한다. 표준 라이브러리만 쓴다.

요청의 모양
- sep  : Colab 실험(open_model/open_eval_colab.py의 H2O)과 똑같다. state = {"frames": [{"seconds": t, "frame": 그림}, … 4개]},
         질문 = JEV_Q["win4"] ("`frames` holds 4 consecutive frames …")
- grid : state = {"seconds": [t1…t4], "grid": 그림 한 장}, 질문 = GRID_Q (칸의 순서를 글로 적는다)
- one  : state = {"frame": 그림 한 장}, 질문 = JEV_Q["one"]  (속도 시험용)
문장은 이 한 가지만 쓰고, 결과를 보고 고치지 않는다. 항목 하나가 요청 하나이고 그 영상의 질문을 모두 함께 넣는다.

보내는 곳: --url 의 /v1/systemone. Cloud Run의 비공개 서비스이면 --auth gcloud (gcloud auth print-identity-token으로 받은 토큰을 붙인다).
시간: seconds = 이 Mac에서 잰 왕복 시간, shim_ms = 답의 usage.latency_ms(서버 안에서 잰 시간).

실행:
  python3 grid4_run.py --url https://… --auth gcloud --mode sep  --tag sep         # 따로 4장, 959건 → results/<name>__sep.json
  python3 grid4_run.py --url https://… --auth gcloud --mode grid --tag grid        # 2×2 한 그림
  python3 grid4_run.py --url https://… --auth gcloud --speed                       # 동시 요청 수별 속도 → results/<name>__speed.json
결과: {meta, records[{id, probs{질문 id: 확률}, seconds, shim_ms, tokens, retries, error}]}. 100개마다 저장하고, 다시 돌리면 이어서 한다
"""
import argparse
import base64
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"
OPEN = HERE.parent / "open_model"
sys.path.insert(0, str(OPEN))
from open_eval_colab import JEV_Q  # noqa: E402  (Colab 실험과 같은 문장)

GRID_Q = ("`grid` is one picture that holds 4 consecutive frames from a video, 0.5 seconds apart, laid out 2x2 in time order: "
          "top-left (earliest), top-right, bottom-left, bottom-right (latest). Do these frames show the following: {s}?")
RETRY, STOP, TRIES = (408, 429, 500, 502, 503, 504), (401, 403), 6


class Fatal(Exception):
    pass


class Token:
    """gcloud의 ID 토큰. 한 시간짜리라 40분마다 새로 받는다."""

    def __init__(self, how, audience=""):
        self.how, self.audience, self.lock, self.value, self.at = how, audience, threading.Lock(), None, 0.0

    def fetch(self):
        if self.how == "gcloud":   # 이 Mac에서: 로그인한 사용자의 토큰
            return subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True, check=True).stdout.strip()
        # metadata: Cloud Run·Compute Engine 안에서, 그 자원의 서비스 계정 토큰(받는 쪽 주소를 audience로)
        req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=" + self.audience,
                                     headers={"Metadata-Flavor": "Google"})
        return urllib.request.urlopen(req, timeout=10).read().decode().strip()

    def header(self):
        if self.how == "none":
            return {}
        with self.lock:
            if self.value is None or time.time() - self.at > 2400:
                self.value, self.at = self.fetch(), time.time()
            return {"Authorization": "Bearer " + self.value}


def uri(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def frame_paths(item, frames):
    return [SRC / frames[item["video"]]["frames"][f"{t:.2f}"] for t in item["seconds"]]


def body(item, mode, frames, shift=0.0, which=0, one_question=False, path=None, nonce=None):
    """shift: 속도 시험에서 같은 항목을 다시 보낼 때 state의 시각을 옮겨, 서버의 앞부분 캐시에 걸리지 않게 한다.
    path, nonce: 속도 시험의 한 장 요청. 프레임을 직접 지정하고, state 맨 앞에 요청마다 다른 숫자를 둔다(같은 까닭)."""
    qids = list(item["questions"])[:1] if one_question else list(item["questions"])
    names = {f"q{k}": qid for k, qid in enumerate(qids, 1)}
    if mode == "sep":
        state = {"frames": [{"seconds": round(t + shift, 2), "frame": uri(p)} for t, p in zip(item["seconds"], frame_paths(item, frames))]}
        text = JEV_Q["win4"]
    elif mode == "grid":
        state = {"seconds": [round(t + shift, 2) for t in item["seconds"]], "grid": uri(HERE / "pack" / item["grid"])}
        text = GRID_Q
    else:  # one
        state = {"frame": uri(path or frame_paths(item, frames)[which])}
        if nonce is not None:
            state = {"shot": nonce, **state}
        text = JEV_Q["one"]
    qs = {n: {"type": "noul", "instructions": text.format(s=item["questions"][qid])} for n, qid in names.items()}
    return {"state": state, "questions": qs}, names


def post(url, token, payload, stop):
    data = json.dumps(payload).encode()
    for attempt in range(TRIES):
        if stop.is_set():
            raise Fatal("멈춤")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **token.header()})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
            if "answers" not in out:
                raise ValueError(f"답이 없음: {json.dumps(out, ensure_ascii=False)[:300]}")
            return out, time.perf_counter() - t0, attempt
        except urllib.error.HTTPError as err:
            msg = err.read()[:400].decode(errors="replace")
            if err.code in STOP:
                stop.set()
                raise Fatal(f"HTTP {err.code} {msg[:300]}") from None
            if err.code in RETRY and attempt < TRIES - 1:
                time.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            raise ValueError(f"HTTP {err.code} {msg[:300]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt < TRIES - 1:
                time.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            raise ValueError(f"연결 실패: {err}") from None


def ask(url, token, stop, item, mode, frames, **kw):
    payload, names = body(item, mode, frames, **kw)
    out, seconds, retries = post(url, token, payload, stop)
    usage = out.get("usage") or {}
    return {"id": item["id"], "probs": {qid: float(out["answers"][n]["noul"]) for n, qid in names.items()}, "seconds": round(seconds, 4),
            "shim_ms": usage.get("latency_ms"), "tokens": usage.get("input_tokens"), "retries": retries, "error": None}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def speed(url, token, items, frames, out_path, meta, modes=("one", "sep", "grid"), skip_one=0, shift_base=0.0):
    """동시 요청 수를 바꿔 가며 잰다. 실행마다 서버가 본 적 없는 입력을 쓴다(한 장은 서로 다른 프레임, 묶음은 시각을 옮긴 것).
    skip_one: 앞선 실행이 이미 보낸 한 장 프레임 수(그만큼 건너뛴다). shift_base: 앞선 실행과 겹치지 않게 시각에 더하는 값."""
    plan = [("one", 1, 40), ("one", 8, 240), ("one", 16, 240), ("one", 32, 320), ("one", 64, 400), ("one", 128, 400),
            ("sep", 1, 40), ("sep", 8, 200), ("sep", 16, 200), ("sep", 32, 240), ("sep", 64, 240),
            ("grid", 1, 40), ("grid", 8, 200), ("grid", 32, 240)]
    plan = [p for p in plan if p[0] in modes]
    runs, cursor, stop = [], {"one": skip_one, "sep": 0, "grid": 0}, threading.Event()
    # 한 장 요청은 서로 다른 프레임을 차례로 쓴다(이웃한 위치는 프레임 2장을 함께 가지므로 위치로 세면 같은 그림이 다시 나온다)
    uniq = {}
    for it in items:
        for path in frame_paths(it, frames):
            uniq.setdefault(str(path), it)
    uniq = sorted(uniq.items())
    for n_run, (mode, workers, n) in enumerate(plan):
        jobs = []
        for _ in range(n):
            c = cursor[mode]
            cursor[mode] += 1
            if mode == "one":
                path, it = uniq[c % len(uniq)]
                jobs.append((it, {"path": path, "nonce": c, "one_question": True}))
            else:               # 항목을 돌려 쓰되, 돌 때마다 시각을 1,000초씩 옮긴다
                jobs.append((items[c % len(items)], {"shift": shift_base + 1000.0 * (1 + c // len(items)) + 7 * n_run}))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(ask, url, token, stop, it, mode, frames, **kw) for it, kw in jobs]
            recs, errors = [], []
            for f in futs:
                try:
                    recs.append(f.result())
                except Exception as err:
                    errors.append(f"{type(err).__name__}: {err}"[:200])
        wall = time.perf_counter() - t0
        run = {"mode": mode, "workers": workers, "n": n, "errors": len(errors), "wall_seconds": round(wall, 2),
               "requests_per_s": round(len(recs) / wall, 2), "frames_per_s": round(len(recs) * (1 if mode == "one" else 4) / wall, 2),
               "client_s_p50": pct([r["seconds"] for r in recs], 0.5), "client_s_p95": pct([r["seconds"] for r in recs], 0.95),
               "shim_ms_p50": pct([r["shim_ms"] for r in recs], 0.5), "shim_ms_p95": pct([r["shim_ms"] for r in recs], 0.95),
               "tokens_p50": pct([r["tokens"] for r in recs], 0.5), "retries": sum(r["retries"] for r in recs)}
        if errors:
            run["first_error"] = errors[0]
        runs.append(run)
        print(run, flush=True)
        out_path.write_text(json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=1) + "\n")
        if stop.is_set():
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="서비스 주소(끝에 /v1/systemone을 붙여 보낸다)")
    ap.add_argument("--auth", default="none", choices=["none", "gcloud", "metadata"])
    ap.add_argument("--mode", default="sep", choices=["sep", "grid"])
    ap.add_argument("--name", default="h2o-rtx", help="결과 파일 이름의 앞부분")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="고르게 이만큼만(0이면 전부). 결과는 …_tryN.json에 따로 둔다")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--modes", default="one,sep,grid", help="--speed에서 잴 방식")
    ap.add_argument("--skip-one", type=int, default=0, help="--speed: 앞선 실행이 이미 보낸 한 장 프레임 수")
    ap.add_argument("--shift-base", type=float, default=0.0, help="--speed: 앞선 실행과 겹치지 않게 시각에 더하는 값")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--note", default="", help="서버 설정 등 meta에 적어 둘 말")
    args = ap.parse_args()

    frames = json.loads((SRC / "data" / "sta2_frames.json").read_text())
    grid = {it["id"]: it["images"][0] for it in json.loads((HERE / "pack" / "manifest.json").read_text())["items"]}
    items = [dict(it, grid=grid[it["id"]]) for it in json.loads((OPEN / "pack" / "manifest.json").read_text())["items"] if it["task"] == "charades"]
    url, token = args.url.rstrip("/") + "/v1/systemone", Token(args.auth, args.url.rstrip("/"))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    try:
        health = urllib.request.urlopen(urllib.request.Request(args.url.rstrip("/") + "/health", headers=token.header()), timeout=60).read()[:400].decode(errors="replace")
    except Exception as err:
        raise SystemExit(f"서버가 준비되지 않음: {err}")
    meta = {"model": "h2o-lightning-4b", "repo": "h2oai/h2o-lightning-4b", "gpu": "Cloud Run", "url": args.url, "server": health, "note": args.note,
            "date": time.strftime("%Y-%m-%d %H:%M"), "templates": {"sep": JEV_Q["win4"], "grid": GRID_Q, "one": JEV_Q["one"]}}
    if args.speed:
        return speed(url, token, items, frames, out_dir / f"{args.name}__speed{'_' + args.tag if args.tag else ''}.json", meta,
                     modes=tuple(args.modes.split(",")), skip_one=args.skip_one, shift_base=args.shift_base)

    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    stem = f"{args.name}__{args.tag or args.mode}"
    out_path = out_dir / (f"{stem}_try{args.limit}.json" if args.limit else f"{stem}.json")
    done, runs = {}, []
    if out_path.exists():
        before = json.loads(out_path.read_text())
        done, runs = {r["id"]: r for r in before["records"] if not r["error"]}, before["meta"].get("runs") or []
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.mode}: 보낼 요청 {len(todo)}건 (전체 {len(items)}건), 질문 {sum(len(it['questions']) for it in todo)}개, 동시 {args.workers}개", flush=True)
    meta.update(mode=args.mode, options={"workers": args.workers, "limit": args.limit})

    def save():
        recs = list(done.values())
        meta.update(tokens_total=sum(r.get("tokens") or 0 for r in recs), retries_total=sum(r.get("retries") or 0 for r in recs))
        out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False) + "\n")

    stop, errors, shown, start = threading.Event(), [], 0, time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(ask, url, token, stop, it, args.mode, frames): it for it in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                rec = fut.result()
                done[item["id"]] = rec
                if args.show and shown < 2:
                    shown += 1
                    print("답:", json.dumps(rec, ensure_ascii=False)[:400], flush=True)
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
                print(f"{n}/{len(todo)} {time.perf_counter() - start:.0f}초", flush=True)
    runs.append({"items_run": len(todo), "wall_seconds": round(time.perf_counter() - start, 1), "errors": len(errors)})
    meta.update(runs=runs, wall_seconds=round(sum(r["wall_seconds"] for r in runs), 1), items_run=sum(r["items_run"] for r in runs), errors=errors[:20])
    save()
    recs = list(done.values())
    print(f"끝: 답 {len(recs)}개, 오류 {len(errors)}개, 왕복 중앙값 {pct([r['seconds'] for r in recs], 0.5)}초, 서버 안 {pct([r['shim_ms'] for r in recs], 0.5)}ms, "
          f"요청당 토큰 {pct([r['tokens'] for r in recs], 0.5)}, 전체 {runs[-1]['wall_seconds']}초 → {out_path}")


if __name__ == "__main__":
    main()
