"""Colab(GPU)에서 오픈 결정 모델을 평가 묶음(open_pack.py가 만든 것)에 돌린다. Jev 요청은 없다.

모델 (실행법은 각 모델 카드와 README에서 확인함, 2026-11-11)
- d1-2b, d1-3b, d1-omni : LiquidAI d1. transformers로 불러 model.system_one(state, questions, media=[그림])
- h2o-4b                : H2O-Lightning-4B. 모델 저장소의 serve.sh로 vLLM과 shim을 먼저 띄워 두고 /v1/systemone에 JSON으로 묻는다.
                          --batch N이면 N개를 동시에 보낸다
- decider-2b, decider-4b: Mapika decider. 저장소의 scripts/serve.sh로 서버를 띄우고 /v1/systemone에 JSON으로 묻는다(--decider-dir에 저장소).
                          그림은 assets에 base64로 넣고 상태에서 이름으로 가리킨다(TypeSafe SDK 0.7.5의 그림 요청은 이 서버가 읽지 못했다)

요청의 모양은 Jev에 보냈던 것과 같게 한다(항목 하나 = 요청 하나, 그 항목의 질문을 모두 함께).
- decider: 상태 안에 그림을 넣고 질문이 `frame`처럼 상태의 이름을 가리킨다. Jev에 보낸 문장 그대로다(JEV_Q)
- d1     : 그림을 상태와 따로 준다. 같은 문장에서 가리키는 말만 "the picture"로 바꾼다(PIC_Q)
문장은 모델마다 이 한 가지만 쓰고, 결과를 보고 고치지 않는다.

시간은 요청 하나를 끝까지 처리하는 데 걸린 시간이다(그림 파일을 읽고 푸는 시간 포함, 한 번에 요청 하나).

가속 실험(d1 계열만): --dtype fp16, --quant 8bit|4bit(bitsandbytes), --compile(model.compile()), --batch N(system_one_batch),
--scale 0.5(그림을 줄여 넣는다), --video(4장 묶음을 영상으로 넣는다). --subset K는 과제마다 K번째 영상만 통째로 쓰고, --tag는 결과 파일 이름에 붙는다.

실행 (Colab):
  python open_eval_colab.py --model d1-2b --pack /content/open/pack --limit 4 --show   # 0단계: 형식 확인(과제마다 4개)
  python open_eval_colab.py --model d1-2b --pack /content/open/pack                    # 1단계: 전부
결과: <out>/<model>.json = {meta, records[{id, probs{질문 id: 확률}, seconds, error}]}. 100개마다 저장하고, 다시 돌리면 이어서 한다
"""
import argparse
import base64
import inspect
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = {"h2o-4b": ("h2o", "h2oai/h2o-lightning-4b"), "d1-2b": ("d1", "LiquidAI/d1-2B"), "d1-3b": ("d1", "LiquidAI/d1-3B"), "d1-omni": ("d1", "LiquidAI/d1-omni-600M"),
          "decider-2b": ("decider", "Mapika/decider-2b"), "decider-4b": ("decider", "Mapika/decider-4b")}
GAME = "Minecraft survival gameplay, first-person view"
CORNER = "the top-right corner of a Minecraft game screen, enlarged. Notification boxes appear in this corner."
WIN4 = "4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first)"
JEV_Q = {"one": "`frame` is one frame taken from a video. Does this frame show the following: {s}?",
         "win4": "`frames` holds " + WIN4 + ". Do these frames show the following: {s}?",
         "fight": "Does `frame` show this: {s}",
         "corner": "Does `corner` show {s}?"}
VID_Q = "The video clip is 4 consecutive frames from a video, 0.5 seconds apart. Does it show the following: {s}?"
PIC_Q = {"one": "The picture is one frame taken from a video. Does this frame show the following: {s}?",
         "win4": "The pictures are " + WIN4 + ". Do these frames show the following: {s}?",
         "fight": "Does the picture show this: {s}",
         "corner": "Does the picture show {s}?"}


def form(item):
    return "fight" if item["task"] == "fight" else item["kind"]


def prob(answer):
    """Noul 답에서 '예'의 확률을 꺼낸다. 모양이 다르면 그대로 보여 주고 멈춘다."""
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict):
        for key in ("noul", "probability", "yes"):
            if isinstance(answer.get(key), (int, float)):
                return float(answer[key])
    raise ValueError(f"확률을 읽지 못함: {str(answer)[:200]}")


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, name = out.strip().split("\n")[0].split(", ")
    return int(used), name


class D1:
    def __init__(self, repo, args):
        import torch
        import transformers
        from transformers import AutoModel
        self.torch, self.args = torch, args
        if args.quant:
            from transformers import BitsAndBytesConfig
            q = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "8bit" else
                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
            self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, quantization_config=q, device_map={"": 0}).eval()
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
            self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=dtype).to("cuda").eval()
        if args.compile:
            self.model.compile()
        self.extra = {"set_size": args.set_size} if args.set_size else {}
        params = inspect.signature(self.model.system_one).parameters
        self.media = "media" if "media" in params else "images"  # d1-3B의 카드는 images=로 적는다
        self.meta = {"transformers": transformers.__version__, "torch": torch.__version__, "media_argument": self.media, "templates": PIC_Q}

    def request(self, item, pack):
        from PIL import Image
        images = [Image.open(pack / p).convert("RGB") for p in item["images"]]
        if item["kind"] == "sheet":  # 60칸 시트: Jev에 보낸 설명에서 `sheet`만 "the picture"로 바꾸고, 칸마다 예/아니오를 묻는다
            state = {"layout": item["layout"].replace("`sheet` is", "The picture is"), "tiles": item["tiles"]}
            qs = {tid: {"type": "noul", "instructions": f"Does tile {tid} of the picture show this: {s}"} for tid, s in item["questions"].items()}
            return state, qs, images, {tid: tid for tid in item["questions"]}
        if self.args.scale != 1:
            images = [im.resize((round(im.width * self.args.scale), round(im.height * self.args.scale)), Image.BICUBIC) for im in images]
        video = self.args.video and item["kind"] == "win4"
        media = [{"type": "video", "video": images, "fps": 2}] if video else images
        state = {"game": GAME} if item["task"] == "fight" else {"about": "The picture is " + CORNER} if item["kind"] == "corner" else None
        names = {f"q{k}": qid for k, qid in enumerate(item["questions"], 1)}
        text = VID_Q if video else PIC_Q[form(item)]
        return state, {n: {"type": "noul", "instructions": text.format(s=item["questions"][qid])} for n, qid in names.items()}, media, names

    def ask(self, item, pack):
        state, qs, media, names = self.request(item, pack)
        with self.torch.inference_mode():
            out = self.model.system_one(state, qs, **{self.media: media}, **self.extra)
        self.torch.cuda.synchronize()
        return out, {qid: prob(out["answers"][n]) for n, qid in names.items()}

    def ask_batch(self, items, pack):
        reqs = [self.request(it, pack) for it in items]
        with self.torch.inference_mode():
            outs = self.model.system_one_batch([r[:3] for r in reqs], **self.extra)
        self.torch.cuda.synchronize()
        return [(out, {qid: prob(out["answers"][n]) for n, qid in r[3].items()}) for out, r in zip(outs, reqs)]

    def close(self):
        self.meta["torch_max_memory_mb"] = round(self.torch.cuda.max_memory_allocated() / 2 ** 20)


class Decider:
    def __init__(self, repo, args, port=8000):
        if not (Path(args.decider_dir) / "scripts" / "serve.sh").exists():
            raise SystemExit("--decider-dir에 Mapika/decider 저장소가 없다(scripts/serve.sh)")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            raise SystemExit("이미 떠 있는 서버가 있다(다른 모델일 수 있다). 끄고 다시 한다")
        except OSError:
            pass
        self.log = open(Path(args.out) / f"decider_server_{args.model}.log", "w")
        self.proc = subprocess.Popen(["bash", "scripts/serve.sh", repo, str(port)], cwd=args.decider_dir, stdout=self.log, stderr=subprocess.STDOUT)
        self.url = url = f"http://127.0.0.1:{port}"
        for _ in range(900):  # 가중치를 받고 그래프를 잡는 동안 기다린다(최대 30분)
            if self.proc.poll() is not None:
                raise SystemExit("decider 서버가 꺼졌다. decider_server.log를 본다")
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except urllib.error.HTTPError:
                break  # 404라도 답이 오면 떠 있는 것
            except Exception:
                time.sleep(2)
        else:
            raise SystemExit("decider 서버가 30분 안에 뜨지 않았다")
        health = json.loads(urllib.request.urlopen(url + "/health", timeout=10).read())
        self.meta = {"server": {k: health.get(k) for k in ("model", "temperature_by_type", "images")}, "templates": JEV_Q}

    def ask(self, item, pack):
        # TypeSafe SDK(0.7.5)가 보내는 그림 요청은 이 서버가 읽지 못해(500), 서버 소스(decider/images.py)의 형식으로 직접 보낸다:
        # assets에 {이름: {type, format, data_base64}}를 넣고, 상태에서는 그 이름의 문자열로 그림을 가리킨다
        refs = [f"video-frame-image-{k}" for k in range(1, len(item["images"]) + 1)]
        assets = {r: {"type": "image", "format": "jpg", "data_base64": base64.b64encode((pack / p).read_bytes()).decode()}
                  for r, p in zip(refs, item["images"])}
        if item["kind"] == "win4":
            state = {"frames": [{"seconds": t, "frame": r} for t, r in zip(item["seconds"], refs)]}
        elif item["kind"] == "corner":
            state = {"about": "`corner` is " + CORNER, "corner": refs[0]}
        elif item["task"] == "fight":
            state = {"game": GAME, "frame": refs[0]}
        else:
            state = {"frame": refs[0]}
        names = {f"q{k}": qid for k, qid in enumerate(item["questions"], 1)}
        qs = {n: {"type": "noul", "instructions": JEV_Q[form(item)].format(s=item["questions"][qid])} for n, qid in names.items()}
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps({"state": state, "questions": qs, "assets": assets}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as err:
            raise ValueError(f"HTTP {err.code}: {err.read()[:200].decode(errors='replace')}") from None
        return out, {qid: prob(out["answers"][n]) for n, qid in names.items()}

    def close(self):
        self.proc.terminate()
        self.log.close()


class H2O:
    """H2O-Lightning-4B: 모델 저장소의 serve.sh가 띄운 vLLM + shim의 /v1/systemone에 묻는다(서버는 따로 띄워 둔다, --url).

    그림은 Jev 때처럼 상태 안에 넣는다(모델 카드: data URI를 상태의 어디에나 둘 수 있다). 문장은 Jev에 보낸 것 그대로(JEV_Q).
    """

    def __init__(self, repo, args):
        self.url, self.attention, self.scale = args.url, args.attention, args.scale
        health = urllib.request.urlopen(self.url + "/health", timeout=10).read()[:300].decode(errors="replace")
        self.meta = {"server": health, "templates": JEV_Q, "url": self.url}

    def jpeg(self, path):
        """그림 파일의 바이트. --scale이 1이 아니면 그만큼 줄여 JPEG로 다시 담는다(시각 토큰을 줄이려는 것)."""
        if self.scale == 1:
            return path.read_bytes()
        import io
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im = im.resize((round(im.width * self.scale), round(im.height * self.scale)), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return buf.getvalue()

    def body(self, item, pack):
        uris = ["data:image/jpeg;base64," + base64.b64encode(self.jpeg(pack / p)).decode() for p in item["images"]]
        if item["kind"] == "win4":
            state = {"frames": [{"seconds": t, "frame": u} for t, u in zip(item["seconds"], uris)]}
        elif item["kind"] == "corner":
            state = {"about": "`corner` is " + CORNER, "corner": uris[0]}
        elif item["kind"] == "sheet":
            state = {"layout": item["layout"], "tiles": item["tiles"], "sheet": uris[0]}
        elif item["task"] == "fight":
            state = {"game": GAME, "frame": uris[0]}
        else:
            state = {"frame": uris[0]}
        if item["kind"] == "sheet":
            names = {tid: tid for tid in item["questions"]}
            qs = {tid: {"type": "noul", "instructions": f"Does tile {tid} of `sheet` show this: {s}"} for tid, s in item["questions"].items()}
        else:
            names = {f"q{k}": qid for k, qid in enumerate(item["questions"], 1)}
            qs = {n: {"type": "noul", "instructions": JEV_Q[form(item)].format(s=item["questions"][qid])} for n, qid in names.items()}
        body = {"state": state, "questions": qs}
        if self.attention:
            body["attention_mode"] = self.attention  # "causal" = 한 방향 읽기(기본은 서버 설정대로 양방향)
        return body, names

    def ask(self, item, pack):
        body, names = self.body(item, pack)
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=300).read())
        except urllib.error.HTTPError as err:
            raise ValueError(f"HTTP {err.code}: {err.read()[:300].decode(errors='replace')}") from None
        return out, {qid: prob(out["answers"][n]) for n, qid in names.items()}

    def ask_batch(self, items, pack):  # --batch N: N개를 동시에 보낸다(서버가 함께 처리한다)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(items)) as pool:
            return list(pool.map(lambda it: self.ask(it, pack), items))

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tasks", default="", help="쉼표로 나눈 과제 이름(없으면 묶음의 전부)")
    ap.add_argument("--manifest", default="manifest.json", help="묶음 안의 항목 파일(둘째 묶음은 manifest2.json)")
    ap.add_argument("--set-size", type=int, default=0, help="d1: 한 번에 함께 읽는 질문 수(0이면 모델의 기본값 16)")
    ap.add_argument("--limit", type=int, default=0, help="과제마다 고르게 이만큼만(0이면 전부). 결과는 <model>_tryN.json에 따로 둔다")
    ap.add_argument("--show", action="store_true", help="처음 두 요청의 답을 그대로 보여 준다")
    ap.add_argument("--decider-dir", default="/content/decider")
    ap.add_argument("--url", default="http://127.0.0.1:8741", help="h2o: shim의 주소")
    ap.add_argument("--attention", default="", choices=["", "causal"], help="h2o: causal이면 요청마다 한 방향 읽기를 고른다")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--quant", default="", choices=["", "8bit", "4bit"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="과제마다 영상을 이름순으로 놓고 K번째마다 하나씩, 그 영상의 항목을 모두 쓴다")
    ap.add_argument("--tag", default="", help="결과 파일 이름 <model>__<tag>.json")
    args = ap.parse_args()

    pack, out_dir = Path(args.pack), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, every = [], json.loads((pack / args.manifest).read_text())["items"]
    for task in (args.tasks.split(",") if args.tasks else list(dict.fromkeys(it["task"] for it in every))):
        part = [it for it in every if it["task"] == task]
        if args.subset:
            keep = set(sorted({it["video"] for it in part})[::args.subset])
            part = [it for it in part if it["video"] in keep]
        items += part[::max(1, len(part) // args.limit)][:args.limit] if args.limit else part
    stem = f"{args.model}__{args.tag}" if args.tag else args.model
    out_path = out_dir / (f"{stem}_try{args.limit}.json" if args.limit else f"{stem}.json")
    done = {}
    if out_path.exists():
        done = {r["id"]: r for r in json.loads(out_path.read_text())["records"] if not r["error"]}

    kind, repo = MODELS[args.model]
    t0 = time.perf_counter()
    backend = {"d1": D1, "decider": Decider, "h2o": H2O}[kind](repo, args)
    meta = {"model": args.model, "repo": repo, "load_seconds": round(time.perf_counter() - t0, 1), **backend.meta,
            "options": {k: getattr(args, k) for k in ("dtype", "quant", "compile", "batch", "scale", "video", "subset", "tasks", "manifest", "set_size", "attention")}}
    print(f"{args.model} 준비 {meta['load_seconds']}초. 항목 {len(items)}개 중 남은 것 {sum(it['id'] not in done for it in items)}개", flush=True)

    def save():
        used, name = gpu_used_mb()
        meta.update(gpu=name, gpu_used_mb=max(used, meta.get("gpu_used_mb", 0)))
        out_path.write_text(json.dumps({"meta": meta, "records": list(done.values())}, ensure_ascii=False) + "\n")

    errors, shown, start = [], 0, time.perf_counter()
    todo = [it for it in items if it["id"] not in done]
    for n in range(0, len(todo), args.batch):
        chunk = todo[n:n + args.batch]
        t0 = time.perf_counter()
        try:
            answers = [backend.ask(chunk[0], pack)] if args.batch == 1 else backend.ask_batch(chunk, pack)
            each = round((time.perf_counter() - t0) / len(chunk), 4)  # 묶어 넣으면 묶음 시간을 항목 수로 나눈 값
            for item, (raw, probs) in zip(chunk, answers):
                done[item["id"]] = {"id": item["id"], "probs": probs, "seconds": each, "error": None}
            if args.show and shown < 2:
                shown += 1
                print("답 그대로:", str(answers[0][0])[:700], flush=True)
        except Exception as err:  # 무엇이 안 맞는지 보려고 넓게 잡는다. 계속 틀리면 아래에서 멈춘다
            errors.append(f"{chunk[0]['id']}: {type(err).__name__}: {err}"[:300])
            print("오류", errors[-1], flush=True)
            if len(errors) >= 10 and len(errors) > len(done):
                break
        if (n // args.batch) % max(1, 100 // args.batch) == max(1, 100 // args.batch) - 1:
            save()
            print(f"{n + len(chunk)}/{len(todo)} {time.perf_counter() - start:.0f}초", flush=True)
    meta.update(wall_seconds=round(time.perf_counter() - start, 1), items_run=len(todo))
    backend.close()
    meta["errors"] = errors[:20]
    save()
    times = sorted(r["seconds"] for r in done.values())
    mid = times[len(times) // 2] if times else None
    print(f"끝: 답 {len(done)}개, 오류 {len(errors)}개, 요청당 중앙값 {mid}초, GPU {meta.get('gpu')} {meta.get('gpu_used_mb')}MB → {out_path}")


if __name__ == "__main__":
    main()
