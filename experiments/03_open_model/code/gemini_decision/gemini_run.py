"""Gemini Decision 1(gemini-decision-1)을 평가 묶음(../open_model/open_pack.py가 만든 것)에 돌린다. 표준 라이브러리만 쓴다.

요청의 모양은 오픈 모델 때와 같다(항목 하나 = 요청 하나, 그 항목의 질문을 모두 함께).
이 모델은 질문에 글만 받으므로 그림은 공유 입력에 넣고, 문장은 d1에 쓴 것(open_eval_colab.py의 PIC_Q)을 그대로 쓴다.
- 한 장(one)   : 입력 = 그림
- 4장 묶음     : 입력 = 프레임마다 "Frame k of 4, at t s:" 한 줄 + 그림 (Jev 상태에 프레임마다 시각을 적었던 것과 같은 정보)
- 싸움(fight)  : 입력 = {"game": …} + 그림
- 모서리(corner): 입력 = {"about": "The picture is …"} + 그림
- 시트(sheet)  : 입력 = {"layout": …, "tiles": …} + 그림, 칸마다 질문 하나 (d1 때와 같은 문장)
문장과 입력 모양은 이 한 가지만 쓰고, 결과를 보고 고치지 않는다.

보내는 곳(--provider)
- openrouter : POST https://openrouter.ai/api/alpha/decisions, 모델 google/gemini-decision-1 (OpenRouter가 Google AI Studio로 넘긴다).
               입력은 state 배열(글은 문자열, 그림은 image_url 부품). 해상도는 detail의 low / high만 넘어간다. 키는 OPENROUTER_API_KEY
- google     : POST https://generativelanguage.googleapis.com/v1beta/decisions, 모델 gemini-decision-1.
               입력은 input 블록 배열. 해상도는 low / medium / high / ultra_high. 키는 GEMINI_API_KEY
키는 환경 변수, 없으면 --env-file의 같은 이름에서 읽는다. 어디에도 찍지 않는다.

시간은 요청 하나를 보내고 답을 받기까지의 왕복 시간이다(다시 보낸 경우 마지막 시도만). 요청은 --workers개를 동시에 보낸다.
토큰과 비용은 답의 usage에서 읽어 항목마다 적는다(openrouter는 cost도 준다). 가격표는 입력 100만 토큰에 $0.10, 출력 0이다(2026-11-12).

실행:
  python3 gemini_run.py --pack <푼 묶음 폴더> --dry-run                                   # 보내지 않고 요청 수와 예상 토큰만
  python3 gemini_run.py --pack <푼 묶음 폴더> --env-file <.env> --limit 4 --show          # 형식 확인(과제마다 4개) → results/<이름>_try4.json
  python3 gemini_run.py --pack <푼 묶음 폴더> --env-file <.env>                           # 전부 → results/gemini-decision-1.json
결과: {meta, records[{id, probs{질문 id: 확률}, seconds, tokens, cost, retries, error}]}. 100개마다 저장하고, 다시 돌리면 이어서 한다
"""
import argparse
import base64
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "open_model"))
from open_eval_colab import CORNER, GAME, PIC_Q, form  # noqa: E402  (같은 문장을 한 곳에서만 관리한다)

NAME = "gemini-decision-1"
PROVIDERS = {
    "openrouter": {"url": "https://openrouter.ai/api/alpha/decisions", "model": "google/gemini-decision-1", "key": "OPENROUTER_API_KEY",
                   "resolutions": ("", "low", "high")},
    "google": {"url": "https://generativelanguage.googleapis.com/v1beta/decisions", "model": "gemini-decision-1", "key": "GEMINI_API_KEY",
               "resolutions": ("", "low", "medium", "high", "ultra_high")},
}
PRICE = 0.10  # 달러 / 입력 100만 토큰
IMAGE_TOKENS = {"": 1120, "low": 280, "medium": 560, "high": 1120, "ultra_high": 2240}  # Google 문서의 표. --dry-run의 예상에만 쓴다
RETRY, STOP, TRIES = (408, 429, 500, 502, 503, 504), (401, 402, 403), 7


class Fatal(Exception):
    """키, 권한, 잔액 문제. 다시 보내도 같으므로 전체를 멈춘다."""


def text(s):
    return {"type": "text", "text": s}


def build(item, pack):
    """(입력 블록, 질문, {보낸 이름: 질문 id}). 블록은 보내는 곳과 무관한 모양이고 body()가 맞게 바꾼다."""
    imgs = [{"type": "image", "data": base64.b64encode((pack / p).read_bytes()).decode(), "mime_type": "image/jpeg"} for p in item["images"]]
    if item["kind"] == "sheet":
        state = {"layout": item["layout"].replace("`sheet` is", "The picture is"), "tiles": item["tiles"]}
        names = {tid: tid for tid in item["questions"]}
        qs = {tid: {"type": "noul", "instructions": f"Does tile {tid} of the picture show this: {s}"} for tid, s in item["questions"].items()}
        return [text(json.dumps(state) + "\n"), imgs[0]], qs, names
    if item["kind"] == "win4":
        blocks = [b for k, (t, im) in enumerate(zip(item["seconds"], imgs), 1) for b in (text(f"Frame {k} of 4, at {t:.2f} s:\n"), im)]
    elif item["kind"] == "corner":
        blocks = [text(json.dumps({"about": "The picture is " + CORNER}) + "\n"), imgs[0]]
    elif item["task"] == "fight":
        blocks = [text(json.dumps({"game": GAME}) + "\n"), imgs[0]]
    else:
        blocks = [imgs[0]]
    names = {f"q{k}": qid for k, qid in enumerate(item["questions"], 1)}
    qs = {n: {"type": "noul", "instructions": PIC_Q[form(item)].format(s=item["questions"][qid])} for n, qid in names.items()}
    return blocks, qs, names


def body(provider, blocks, qs, resolution):
    model = PROVIDERS[provider]["model"]
    if provider == "google":
        return {"model": model, "input": [b | ({"resolution": resolution} if resolution and b["type"] == "image" else {}) for b in blocks], "questions": qs}
    state = []  # openrouter: 글은 문자열 그대로, 그림은 Chat Completions와 같은 image_url 부품
    for b in blocks:
        if b["type"] == "text":
            state.append(b["text"])
        else:
            part = {"url": f"data:{b['mime_type']};base64,{b['data']}"}
            state.append({"type": "image_url", "image_url": part | ({"detail": resolution} if resolution else {})})
    return {"model": model, "state": state, "questions": qs}


def estimate(item, resolution):
    """보내기 전의 예상 토큰: 그림은 문서의 표, 글은 4글자에 1토큰으로 어림한다."""
    chars = sum(len(s) + 60 for s in item["questions"].values()) + len(item.get("layout", "")) + len(json.dumps(item.get("tiles", "")))
    return len(item["images"]) * IMAGE_TOKENS[resolution] + chars // 4


class Pace:
    """분당 토큰 한도(--tpm)에 맞춰 요청을 띄엄띄엄 내보낸다. 0이면 아무것도 하지 않는다."""

    def __init__(self, tpm):
        self.tpm, self.lock, self.next = tpm, threading.Lock(), time.monotonic()

    def wait(self, tokens):
        if not self.tpm:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + tokens / self.tpm * 60
        time.sleep(max(0.0, start - now))


def post(url, headers, payload, stop):
    data = json.dumps(payload).encode()
    for attempt in range(TRIES):
        if stop.is_set():
            raise Fatal("멈춤")
        req = urllib.request.Request(url, data=data, headers=headers | {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            if "answers" not in out:  # 200인데 오류를 담아 주는 경우
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


def load_key(var, env_file):
    key = os.environ.get(var, "")
    if not key and env_file:
        for line in Path(env_file).read_text().splitlines():
            if line.startswith(var + "="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit(f"{var}가 없다(환경 변수 또는 --env-file)")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="푼 묶음 폴더(manifest의 img/ 경로가 여기서 풀린다)")
    ap.add_argument("--provider", default="openrouter", choices=sorted(PROVIDERS))
    ap.add_argument("--manifest", default=str(HERE.parent / "open_model" / "pack" / "manifest.json"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tasks", default="", help="쉼표로 나눈 과제 이름(없으면 묶음의 전부)")
    ap.add_argument("--limit", type=int, default=0, help="과제마다 고르게 이만큼만(0이면 전부). 결과는 <이름>_tryN.json에 따로 둔다")
    ap.add_argument("--subset", type=int, default=0, help="과제마다 영상을 이름순으로 놓고 K번째마다 하나씩, 그 영상의 항목을 모두 쓴다")
    ap.add_argument("--resolution", default="", choices=sorted(IMAGE_TOKENS), help="그림 해상도. 비우면 API의 기본값")
    ap.add_argument("--workers", type=int, default=8, help="동시에 보내는 요청 수")
    ap.add_argument("--tpm", type=int, default=0, help="분당 토큰 한도에 맞춰 보낸다(0이면 맞추지 않고 429가 오면 기다린다)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 gemini-decision-1__<tag>.json")
    ap.add_argument("--env-file", default="")
    ap.add_argument("--dry-run", action="store_true", help="보내지 않고 요청 수와 예상 토큰만 찍는다")
    ap.add_argument("--show", action="store_true", help="처음 두 요청의 답을 그대로 보여 준다")
    args = ap.parse_args()
    prov = PROVIDERS[args.provider]
    if args.resolution not in prov["resolutions"]:
        raise SystemExit(f"{args.provider}로는 해상도 '{args.resolution}'를 넘길 수 없다(되는 값: {prov['resolutions']})")

    pack, out_dir = Path(args.pack), Path(args.out)
    items, every = [], json.loads(Path(args.manifest).read_text())["items"]
    for task in (args.tasks.split(",") if args.tasks else list(dict.fromkeys(it["task"] for it in every))):
        part = [it for it in every if it["task"] == task]
        if args.subset:
            keep = set(sorted({it["video"] for it in part})[::args.subset])
            part = [it for it in part if it["video"] in keep]
        items += part[::max(1, len(part) // args.limit)][:args.limit] if args.limit else part
    stem = f"{NAME}__{args.tag}" if args.tag else NAME
    out_path = out_dir / (f"{stem}_try{args.limit}.json" if args.limit else f"{stem}.json")
    done, runs, served = {}, [], None
    if out_path.exists():
        before = json.loads(out_path.read_text())
        done = {r["id"]: r for r in before["records"] if not r["error"]}
        runs = before["meta"].get("runs") or []  # 이어서 돌린 기록. 전체 시간과 보낸 수는 이 목록의 합으로 적는다
        served = before["meta"].get("served_by")
    todo = [it for it in items if it["id"] not in done]

    est = {}
    for it in todo:
        e = est.setdefault(it["task"], [0, 0, 0, 0])
        e[0] += 1
        e[1] += len(it["images"])
        e[2] += len(it["questions"])
        e[3] += estimate(it, args.resolution)
    total = sum(e[3] for e in est.values())
    print(f"보낼 요청 {len(todo)}건 (전체 {len(items)}건 중 이미 받은 것 {len(items) - len(todo)}건), {args.provider}, 해상도 '{args.resolution or '기본'}'")
    for task, (n, im, q, tok) in est.items():
        print(f"  {task:11s} 요청 {n:5d}  그림 {im:5d}  질문 {q:6d}  예상 토큰 {tok:9,d}")
    print(f"  예상 토큰 합 {total:,} ≈ ${total / 1e6 * PRICE:.2f} (그림 토큰은 문서의 표, 글은 어림)")
    missing = [p for it in todo for p in it["images"] if not (pack / p).exists()]
    if missing:
        raise SystemExit(f"묶음에 없는 그림 {len(missing)}장 (예: {missing[0]})")
    if args.dry_run or not todo:
        return

    key = load_key(prov["key"], args.env_file)
    headers = {"Authorization": f"Bearer {key}"} if args.provider == "openrouter" else {"x-goog-api-key": key}
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": NAME, "repo": "google/" + NAME, "provider": args.provider, "url": prov["url"], "gpu": "API", "templates": PIC_Q,
            "date": time.strftime("%Y-%m-%d"), "estimated_tokens": total, "served_by": served,
            "options": {k: getattr(args, k) for k in ("resolution", "workers", "tpm", "tasks", "subset", "limit")} | {"manifest": Path(args.manifest).name}}
    stop, pace, lock = threading.Event(), Pace(args.tpm), threading.Lock()

    def work(item):
        blocks, qs, names = build(item, pack)
        pace.wait(estimate(item, args.resolution))
        out, seconds, retries = post(prov["url"], headers, body(args.provider, blocks, qs, args.resolution), stop)
        probs = {qid: float(out["answers"][n]["noul"]) for n, qid in names.items()}
        usage = out.get("usage") or {}
        rec = {"id": item["id"], "probs": probs, "seconds": round(seconds, 4), "retries": retries, "error": None,
               "tokens": usage.get("input_tokens", usage.get("total_input_tokens")), "cost": usage.get("cost")}
        if usage.get("input_tokens_by_modality"):
            rec["by_modality"] = {m["modality"]: m["tokens"] for m in usage["input_tokens_by_modality"]}
        if out.get("provider"):
            meta["served_by"] = out["provider"]
        return out, rec

    def save():
        recs = list(done.values())
        meta.update(tokens_total=sum(r.get("tokens") or 0 for r in recs), retries_total=sum(r.get("retries") or 0 for r in recs))
        billed = [r["cost"] for r in recs if r.get("cost") is not None]
        meta["cost_usd"] = round(sum(billed), 4) if billed else round(meta["tokens_total"] / 1e6 * PRICE, 4)
        meta["cost_source"] = "usage.cost" if billed else "토큰 × 가격표"
        out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False) + "\n")

    errors, shown, start, fatal = [], 0, time.perf_counter(), None
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(work, it): it for it in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                raw, rec = fut.result()
                with lock:
                    done[item["id"]] = rec
                if args.show and shown < 2:
                    shown += 1
                    print("답 그대로:", json.dumps(raw, ensure_ascii=False)[:700], flush=True)
            except Fatal as err:
                if fatal is None and str(err) != "멈춤":
                    fatal = str(err)
                    print("멈춤:", fatal, flush=True)
            except Exception as err:  # 무엇이 안 맞는지 보려고 넓게 잡는다. 계속 틀리면 아래에서 멈춘다
                errors.append(f"{item['id']}: {type(err).__name__}: {err}"[:400])
                print("오류", errors[-1], flush=True)
                if len(errors) >= 10 and len(errors) > len(done):
                    stop.set()
            if n % 100 == 0:
                save()
                print(f"{n}/{len(todo)} {time.perf_counter() - start:.0f}초, 토큰 {meta['tokens_total']:,}, ${meta['cost_usd']}, 다시 보냄 {meta['retries_total']}", flush=True)
    runs.append({"items_run": len(todo), "wall_seconds": round(time.perf_counter() - start, 1), "errors": len(errors)})
    meta.update(runs=runs, wall_seconds=round(sum(r["wall_seconds"] for r in runs), 1), items_run=sum(r["items_run"] for r in runs),
                errors=errors[:20], fatal=fatal)  # errors는 마지막 실행에서도 답을 못 받은 것
    save()
    times = sorted(r["seconds"] for r in done.values())
    print(f"끝: 답 {len(done)}개, 오류 {len(errors)}개" + (f", 멈춤({fatal})" if fatal else "") +
          f", 요청당 중앙값 {times[len(times) // 2] if times else None}초, 전체 {meta['wall_seconds']}초, "
          f"토큰 {meta['tokens_total']:,} (${meta['cost_usd']}, {meta['cost_source']}) → {out_path}")


if __name__ == "__main__":
    main()
