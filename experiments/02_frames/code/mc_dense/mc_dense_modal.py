"""마인크래프트 영상에서 짧게 열렸다 닫히는 창을 찾으려면 프레임을 얼마나 촘촘히, 어떤 모양으로 넣어야 하는가. Modal에서 돈다.

자료: OpenAI VPT의 플레이 기록 10.0판(MIT). 영상 640×360, 초당 20장, 틱마다 한 줄인 행동 기록.
  - old : 앞선 실험의 5분 조각 12개(인벤토리 구간 46개. 상자를 연 일은 없다)
  - new : 상자를 연 5분 조각을 규칙으로 새로 고른 것(pick). 고르는 데는 행동 기록의 첫 줄과 끝 줄만 읽는다
판정 모델: H2O-Lightning-4B. Cloud Run 때와 같은 설정(한 방향 읽기, vLLM 패치, FP8).

단계 (모두 Modal 안에서 돈다. 이 Mac에서는 올리거나 받지 않는다)
  pick  : (CPU) 색인을 씨앗 20261112로 섞은 순서대로 보며, 5분짜리이고 그 조각 안에서 상자를 한 번 이상 연 것을 참여자마다 하나씩 고른다
  prep  : (CPU, 조각마다) 영상과 행동 기록을 받아 초당 8장 프레임(틱 0, 2, 5, 7, 10, 12, 15, 17)과 창 구간 정답을 만든다
  judge : (GPU) H2O를 같은 컨테이너 안에 띄우고 모든 모양을 물은 뒤 끝난다. 밖으로 열린 주소가 없고, 끝나면 과금도 끝난다

실행 (프로필은 명령마다 고른다):
  MODAL_PROFILE=<modal-profile> modal run experiments/mc_dense/mc_dense_modal.py::pick  --count 8
  MODAL_PROFILE=<modal-profile> modal run experiments/mc_dense/mc_dense_modal.py::prep
  MODAL_PROFILE=<modal-profile> modal run experiments/mc_dense/mc_dense_modal.py::pack                # 프레임을 영상별 tar로(GPU 작업이 이것을 메모리에 올린다)
  MODAL_PROFILE=<modal-profile> modal run experiments/mc_dense/mc_dense_modal.py::sample              # 묶음 그림 몇 장을 볼륨에 저장(눈으로 확인용)
  MODAL_PROFILE=<modal-profile> modal run experiments/mc_dense/mc_dense_modal.py::fetch_model         # H2O 가중치를 볼륨에(CPU)
  MODAL_PROFILE=<modal-profile> modal run --detach experiments/mc_dense/mc_dense_modal.py::judge [--limit 40]       # 모든 위치(오래 걸린다)
  MODAL_PROFILE=<modal-profile> modal run --detach experiments/mc_dense/mc_dense_modal.py::judge_small            # 작은 시험(약 1,500곳)
  MODAL_PROFILE=<modal-profile> modal volume get mf-mc-dense results experiments/mc_dense/     # 답과 정답 받기
볼륨: mf-mc-dense(/data: 영상, 프레임, 정답, 답), mf-models(/models: H2O 가중치. 한 번 받으면 다시 받지 않는다)
"""
import modal

app = modal.App("moment-finder-mc-dense")
data = modal.Volume.from_name("mf-mc-dense", create_if_missing=True)
models = modal.Volume.from_name("mf-models", create_if_missing=True)

BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
INDEX_URL = BASE + "snapshots/all_10xx_Jun_29.json"
SEED = 20261112
TICKS = 20                                # 초당 틱(= 프레임)
DENSE = (0, 2, 5, 7, 10, 12, 15, 17)      # 1초 안에서 뽑는 틱. 20fps라 정확한 초당 8장은 없어서 0.10초와 0.15초 간격이 번갈아 온다
OLD = [  # 앞선 실험의 12개 (experiments/frame_limit/data/mc/videos.json의 순서 = mc_01 … mc_12)
    "shabby-viridian-beaver-a7083c83103d-20220416-150841",
    "trippy-red-llama-a0e3295e243d-20220416-000222",
    "scaly-fuchsia-wasp-9633765921fd-20220419-133407",
    "squeaky-magnolia-ocelot-75bba839d2d3-20220420-185345",
    "shabby-pink-molly-f153ac423f61-20220424-110731",
    "woozy-ruby-ostrich-744fc8da10e2-20220415-122826",
    "lovely-persimmon-angora-7f8e825a27df-20220416-015033",
    "cheeky-cornflower-setter-edf57f76f9a2-20220416-095312",
    "snippy-chartreuse-mastiff-f153ac423f61-20220418-215115",
    "thirsty-lavender-koala-9ef285ede667-20220422-160013",
    "tasty-brass-devil-69d7c7e8c97d-20220422-223616",
    "gimpy-jade-panda-22be9074ab94-20220413-203808",
]
H2O_REVISION = "659fc412de8486dbc2477b0d1cb4121e97e72471"   # 2026-11-12 Colab·Cloud Run 실험과 같은 판

# ---- 묻는 문장 (결과를 보기 전에 정함)
SENT = {
    "inv": "the player's inventory screen is open, the one that shows the player's character next to the armor slots",   # 앞선 실험의 문장 그대로
    "chest": "a chest's storage screen is open, the one titled Chest or Large Chest that shows the chest's rows of item slots above the player's inventory rows",
}
HEAD = {
    "sep": "`frames` holds 4 consecutive frames from a video, {gap} seconds apart, in time order (earliest first). ",
    "grid8": "`grid` is one picture that holds 8 consecutive frames from a video, about 0.125 seconds apart, laid out in 2 rows of 4 in time order: "
             "top row left to right (earliest first), then bottom row left to right (latest last). ",
}
ASK = {"show": "Do these frames show the following: {s}?",                      # 지금까지 쓴 물음
       "any": "Does at least one of these frames show the following: {s}?"}     # 한 장에만 보여도 되는지를 분명히 한 물음
ONE_Q = "`frame` is one frame taken from a video. Does this frame show the following: {s}?"

# ---- 모양: 위치 k(= k초부터 1초)마다 어떤 틱의 프레임을 어떻게 넣나. 틱은 20k에 더하는 값
LAYOUTS = {   # 적힌 순서대로 돈다(중요한 것부터)
    "sep4_half":    {"ticks": (-5, 5, 15, 25), "kind": "sep", "gap": "0.5"},        # 지금 방법: 0.5초 간격 4장을 따로. 이웃 위치와 2장이 겹친다
    "grid8":        {"ticks": DENSE, "kind": "grid8", "scale": 1.0},               # 8장을 원본 크기 그대로 한 그림(4×2)으로
    "grid8_small":  {"ticks": DENSE, "kind": "grid8", "scale": 0.7071},            # 8장을 한 그림으로, 화소를 절반으로 줄여 4장 값에 맞춤
    "sep4_quarter": {"ticks": (0, 5, 10, 15), "kind": "sep", "gap": "0.25"},       # 겹침 없이 0.25초 간격 4장을 따로. 비용은 지금 방법과 같다
    "one":          {"ticks": (10,), "kind": "one"},                               # 가운데 한 장(1초에 한 장). 가장 싼 기준
}
GAP = 8  # 붙일 때 프레임 사이의 검은 띠(화소)

cpu_image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").pip_install("pillow")
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")   # Modal의 vLLM 예제와 같은 바탕
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
)


hf_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")


@app.function(image=hf_image, volumes={"/models": models}, timeout=3600, cpu=4)
def fetch_model():
    """H2O 가중치를 볼륨에 받아 둔다(GPU가 떠 있는 동안 받지 않으려는 것). 이미 있으면 아무것도 하지 않는다."""
    import os
    import time
    from huggingface_hub import snapshot_download
    if os.path.exists("/models/h2o/serve.sh") and os.path.exists("/models/h2o/_complete"):
        print("이미 있음")
        return "있음"
    t0 = time.time()
    for attempt in range(5):
        try:
            snapshot_download("h2oai/h2o-lightning-4b", local_dir="/models/h2o", revision=H2O_REVISION, max_workers=16)
            break
        except Exception as err:
            print(f"모델 받기 실패 {attempt + 1}/5: {str(err)[:200]}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError("모델을 받지 못함")
    open("/models/h2o/_complete", "w").write(H2O_REVISION)
    models.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("/models/h2o") for f in fs)
    print(f"모델 받음 {time.time() - t0:.0f}초, {size / 1e9:.1f}GB")
    return "받음"


def _range(url, first=None, last=None, tail=None, timeout=60):
    """파일의 일부만 읽는다. tail: 끝에서 그만큼."""
    import urllib.request
    h = {"Range": f"bytes=-{tail}" if tail else f"bytes={first}-{last}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def _alias(name):
    return "-".join(name.split("-")[:3])


@app.function(image=cpu_image, volumes={"/data": data}, timeout=3600, cpu=4)
def pick(count: int = 10, scan_limit: int = 3000, per_alias: int = 2):
    import concurrent.futures as cf
    import json
    import random
    import urllib.request
    K = "minecraft.custom:minecraft.open_chest"
    index = json.loads(urllib.request.urlopen(INDEX_URL, timeout=120).read())
    names = sorted(p.split("/")[-1][:-4] for p in index["relpaths"] if p.endswith(".mp4"))
    random.Random(SEED).shuffle(names)
    used = {}   # 참여자 → 고른 조각 수. 처음에는 참여자마다 하나로 했으나 3,072개를 봐도 3개뿐이라(상자를 여는 참여자가 적다) 참여자마다 per_alias개로 풀었다.
                # 모델의 답을 보기 전에 바꿨다. 앞선 12개와 같은 참여자도 허용한다(다른 녹화다)

    def probe(name):
        url = BASE + "data/10.0/" + name + ".jsonl"
        for attempt in range(3):
            try:
                first = json.loads(_range(url, 0, 65535).split(b"\n")[0])
                last = json.loads([x for x in _range(url, tail=65536).split(b"\n") if x.strip()][-1])
                s0, s1 = first.get("stats") or {}, last.get("stats") or {}
                return {"name": name, "ticks": last["tick"] - first["tick"] + 1, "chest": s1.get(K, 0) - s0.get(K, 0)}
            except Exception as err:
                e = f"{type(err).__name__}: {err}"[:100]
        return {"name": name, "err": e}

    chosen, seen, hits, errs, all_hits, err_kinds = [], 0, 0, 0, [], {}
    with cf.ThreadPoolExecutor(64) as ex:
        for start in range(0, min(scan_limit, len(names)), 128):     # 섞은 순서를 지키려고 128개씩 끊어 본다
            for r in ex.map(probe, names[start:start + 128]):
                seen += 1
                if "err" in r:
                    errs += 1
                    err_kinds[r["err"][:40]] = err_kinds.get(r["err"][:40], 0) + 1
                    continue
                if r["ticks"] < 5900 or r["chest"] < 1 or r["name"] in OLD:
                    continue
                hits += 1
                all_hits.append(r)
                if used.get(_alias(r["name"]), 0) >= per_alias or len(chosen) >= count:
                    continue
                used[_alias(r["name"])] = used.get(_alias(r["name"]), 0) + 1
                chosen.append(r)
            print(f"  본 기록 {seen}, 상자를 연 5분 조각 {hits}, 고른 것 {len(chosen)}", flush=True)
            if len(chosen) >= count:
                break
    out = {"seed": SEED, "rule": f"섞은 순서대로, 5분짜리(5900틱 이상)이고 그 조각 안에서 open_chest가 1 이상 늘어난 것을 참여자마다 {per_alias}개까지",
           "scanned": seen, "errors": errs, "error_kinds": err_kinds, "with_chest": hits, "chosen": chosen, "all_hits": all_hits}
    with open("/data/chosen.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    data.commit()
    print(json.dumps(out, ensure_ascii=False)[:1500])
    return out


def _intervals(recs):
    """행동 기록 → 창이 열린 구간과 그 종류. 구간은 [첫 틱 시각, 마지막 틱 시각](초, 양끝 포함)."""
    KEYS = {"minecraft.custom:minecraft.open_chest": "chest", "minecraft.custom:minecraft.trigger_trapped_chest": "container",
            "minecraft.custom:minecraft.open_barrel": "container", "minecraft.custom:minecraft.open_enderchest": "container",
            "minecraft.custom:minecraft.open_shulker_box": "container",
            "minecraft.custom:minecraft.interact_with_crafting_table": "crafting", "minecraft.custom:minecraft.interact_with_furnace": "furnace"}
    ev = []
    for i in range(1, len(recs)):
        a, b = recs[i - 1].get("stats") or {}, recs[i].get("stats") or {}
        for k, name in KEYS.items():
            if b.get(k, 0) > a.get(k, 0):
                ev.append((i, name))
    out, start = [], None
    for i, d in enumerate(recs + [{"isGuiOpen": False}]):
        if d["isGuiOpen"] and start is None:
            start = i
        elif not d["isGuiOpen"] and start is not None:
            s, e = start, i - 1
            start = None
            if any(recs[j]["isGuiInventory"] for j in range(s, e + 1)):
                kind = "inv"
            else:
                near = [name for j, name in ev if s - 10 <= j <= s + 10]   # 창이 열린 틱 앞뒤 0.5초 안에 늘어난 통계
                kind = near[0] if near else "other"
            out.append({"kind": kind, "start": s / TICKS, "end": e / TICKS})
    return out


@app.function(image=cpu_image, volumes={"/data": data}, timeout=3600, cpu=2, memory=4096)
def prep_one(job):
    """job = (vid, name). 프레임과 정답을 볼륨에 쓴다. 프레임 n의 시각은 n/20초(앞선 실험 mc_prepare.py와 같은 맞춤)."""
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    import urllib.request
    vid, name = job
    raw, out = f"/data/raw/{name}", f"/data/frames/{vid}"
    os.makedirs("/data/raw", exist_ok=True)
    for ext in (".jsonl", ".mp4"):
        if not os.path.exists(raw + ext):
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(BASE + "data/10.0/" + name + ext, timeout=120) as r, open(raw + ext + ".part", "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    os.replace(raw + ext + ".part", raw + ext)
                    break
                except Exception as err:
                    print(vid, ext, "받기 실패", attempt + 1, str(err)[:120], flush=True)
            else:
                raise RuntimeError(f"{name}{ext}를 받지 못함")
    recs = [json.loads(line) for line in open(raw + ".jsonl") if line.strip()]
    marker = out + "/_done.json"
    if not os.path.exists(marker):
        os.makedirs(out, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            sel = "+".join(f"eq(mod(n,{TICKS}),{r})" for r in DENSE)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", raw + ".mp4", "-vf", f"select='{sel}'", "-fps_mode", "vfr", "-q:v", "2", tmp + "/%05d.jpg"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg 실패 {vid}: {proc.stderr.strip()[:300]}")
            got = sorted(os.listdir(tmp))
            ns = [n for n in range(len(recs) + 40) if n % TICKS in DENSE][:len(got)]   # i번째로 나온 그림 = 조건에 맞는 i번째 프레임 번호
            for fname, n in zip(got, ns):
                shutil.copyfile(f"{tmp}/{fname}", f"{out}/{n:05d}.jpg")
        with open(marker, "w") as f:
            json.dump({"frames": len(got), "last": ns[-1]}, f)
    info = {"vid": vid, "name": name, "ticks": len(recs), "seconds": len(recs) // TICKS, "intervals": _intervals(recs),
            "frames": json.load(open(marker))}
    os.remove(raw + ".mp4") if os.path.exists(raw + ".mp4") else None   # 프레임을 뽑았으면 영상은 지운다(볼륨을 아낀다)
    data.commit()
    return info


@app.function(image=cpu_image, volumes={"/data": data}, timeout=7200)
def prep():
    import json
    chosen = json.load(open("/data/chosen.json"))["chosen"]
    jobs = [(f"mc_{i:02d}", n) for i, n in enumerate(OLD, 1)] + [(f"ch_{i:02d}", c["name"]) for i, c in enumerate(chosen, 1)]
    infos = list(prep_one.map(jobs))
    data.reload()
    with open("/data/truth.json", "w") as f:
        json.dump({"note": "구간은 [첫 틱 시각, 마지막 틱 시각](초). kind: inv 인벤토리, chest 상자, container 그 밖의 보관함, crafting 작업대, furnace 화로, other 모름",
                   "videos": infos}, f, ensure_ascii=False, indent=1)
    data.commit()
    for v in infos:
        kinds = {}
        for it in v["intervals"]:
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        print(v["vid"], v["name"][:30], v["seconds"], "초, 프레임", v["frames"]["frames"], kinds)
    return infos


@app.function(image=cpu_image, volumes={"/data": data}, timeout=3600, cpu=4)
def pack_one(vid):
    """한 영상의 프레임을 tar 하나로 묶어 볼륨에 둔다(/data/packs/<vid>.tar)."""
    import concurrent.futures as cf
    import io
    import os
    import tarfile
    src, dest = f"/data/frames/{vid}", f"/data/packs/{vid}.tar"
    os.makedirs("/data/packs", exist_ok=True)
    names = sorted(os.listdir(src))
    with cf.ThreadPoolExecutor(64) as ex:
        blobs = list(ex.map(lambda n: open(f"{src}/{n}", "rb").read(), names))
    with tarfile.open(dest + ".part", "w") as tar:
        for n, b in zip(names, blobs):
            info = tarfile.TarInfo(n)
            info.size = len(b)
            tar.addfile(info, io.BytesIO(b))
    os.replace(dest + ".part", dest)
    data.commit()
    return vid, len(names), sum(len(b) for b in blobs)


@app.function(image=cpu_image, volumes={"/data": data}, timeout=3600)
def pack():
    import json
    vids = [v["vid"] for v in json.load(open("/data/truth.json"))["videos"]]
    out = list(pack_one.map(vids))
    print(out)
    return out


FRAMES = {}   # vid → {파일 이름: 바이트}. GPU 작업이 시작할 때 묶음 파일에서 올려 둔다


def _load_packs(vids):
    import concurrent.futures as cf
    import tarfile

    def load(vid):
        with tarfile.open(f"/data/packs/{vid}.tar") as tar:
            FRAMES[vid] = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        return vid
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(load, vids))


def _frame(vid, name):
    if vid in FRAMES:
        return FRAMES[vid][name]
    return open(f"/data/frames/{vid}/{name}", "rb").read()


_LAST = {}


def _picture(vid, k, layout):
    """(그림들의 JPEG 바이트, 프레임 시각들). 영상의 처음과 끝에서는 있는 프레임으로 당겨 쓴다."""
    import io
    import json
    from PIL import Image
    spec = LAYOUTS[layout]
    if vid not in _LAST:
        _LAST[vid] = json.loads(_frame(vid, "_done.json"))["last"]
    last = _LAST[vid]
    ticks = [min(max(TICKS * k + t, 0), last) for t in spec["ticks"]]
    ticks = [n if n % TICKS in DENSE else n - (n % TICKS) + max(d for d in DENSE if d <= n % TICKS) for n in ticks]   # 끝에서 당겨 썼으면 뽑아 둔 틱으로
    secs = [round(n / TICKS, 2) for n in ticks]
    if spec["kind"] != "grid8":
        return [_frame(vid, f"{n:05d}.jpg") for n in ticks], secs
    frames = [Image.open(io.BytesIO(_frame(vid, f"{n:05d}.jpg"))).convert("RGB") for n in ticks]
    w, h = frames[0].size
    if spec["scale"] != 1.0:
        w, h = round(w * spec["scale"]), round(h * spec["scale"])
        frames = [f.resize((w, h), Image.LANCZOS) for f in frames]
    sheet = Image.new("RGB", (4 * w + 3 * GAP, 2 * h + GAP), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % 4) * (w + GAP), (i // 4) * (h + GAP)))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=95)
    return [buf.getvalue()], secs


def _body(vid, k, layout):
    import base64
    spec = LAYOUTS[layout]
    pics, secs = _picture(vid, k, layout)
    uris = ["data:image/jpeg;base64," + base64.b64encode(p).decode() for p in pics]
    if spec["kind"] == "one":
        return {"state": {"frame": uris[0]},
                "questions": {c: {"type": "noul", "instructions": ONE_Q.format(s=s)} for c, s in SENT.items()}}
    if spec["kind"] == "sep":
        state, head = {"frames": [{"seconds": t, "frame": u} for t, u in zip(secs, uris)]}, HEAD["sep"].format(gap=spec["gap"])
    else:
        state, head = {"seconds": secs, "grid": uris[0]}, HEAD["grid8"]
    return {"state": state, "questions": {f"{c}_{a}": {"type": "noul", "instructions": head + ask.format(s=s)} for c, s in SENT.items() for a, ask in ASK.items()}}


@app.function(image=cpu_image, volumes={"/data": data}, timeout=600)
def sample():
    """눈으로 확인할 그림 몇 장: 상자·인벤토리 구간이 시작하는 위치의 grid8과 grid8_small."""
    import json
    import os
    truth = json.load(open("/data/truth.json"))["videos"]
    os.makedirs("/data/sample", exist_ok=True)
    made = []
    for want in ("chest", "inv"):
        for v in truth:
            its = [it for it in v["intervals"] if it["kind"] == want]
            if its:
                k = int(its[0]["start"])
                for layout in ("grid8", "grid8_small"):
                    pics, secs = _picture(v["vid"], k, layout)
                    path = f"/data/sample/{want}_{v['vid']}_{k}_{layout}.jpg"
                    open(path, "wb").write(pics[0])
                    made.append({"file": path, "kind": want, "interval": its[0], "seconds": secs, "bytes": len(pics[0])})
                break
    data.commit()
    print(json.dumps(made, ensure_ascii=False, indent=1))
    return made


def _subset(truth, n_rest=300):
    """작은 시험에 쓰는 위치(결과를 보기 전에 정함): 인벤토리·상자·보관함 구간의 앞뒤 1초 안에 드는 위치 전부,
    다른 창(작업대·화로 등)과 겹치는 위치 전부, 나머지에서 씨앗 SEED로 뽑은 n_rest곳."""
    import random

    def ov(k, a, b, lo=0.0, hi=1.0):
        return a < k + hi and b + 0.05 > k + lo
    keep, rest = [], []
    for v in truth:
        tgt = [(i["start"], i["end"]) for i in v["intervals"] if i["kind"] in ("inv", "chest", "container")]
        oth = [(i["start"], i["end"]) for i in v["intervals"] if i["kind"] not in ("inv", "chest", "container")]
        for k in range(v["seconds"]):
            if any(ov(k, a, b, -1.0, 2.0) for a, b in tgt) or any(ov(k, a, b) for a, b in oth):
                keep.append((v["vid"], k))
            else:
                rest.append((v["vid"], k))
    return sorted(keep + random.Random(SEED).sample(rest, min(n_rest, len(rest))))


WORKERS = {}   # 모양별로 동시 요청 수를 낮추고 싶을 때. (grid8이 느렸던 까닭은 서버가 아니라 볼륨의 작은 파일 읽기였다. 프레임을 메모리에 올린 뒤에는 필요 없다)


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models}, timeout=80 * 60, cpu=8, memory=32768)
def judge(limit: int = 0, workers: int = 24, layouts: str = ""):
    """모든 위치(6,599곳)에 모든 모양을 묻는다. 2026-11-12에는 sep4_half만 끝내고 멈췄다(grid8이 느려 비용이 커짐)."""
    return _judge(limit, workers, layouts, small=False)


@app.function(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models}, timeout=30 * 60, cpu=8, memory=32768)
def judge_small(workers: int = 16, layouts: str = "one,sep4_quarter,grid8_small,grid8"):
    """작은 시험: _subset의 위치(약 1,500곳)에만 묻는다. 답은 results/<모양>_small.json. 30분에서 끊긴다."""
    return _judge(0, workers, layouts, small=True)


def _judge(limit, workers, layouts, small):
    import concurrent.futures as cf
    import json
    import os
    import subprocess
    import time
    import urllib.error
    import urllib.request
    t_start = time.time()
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)

    # 1) 모델 (볼륨에 없을 때만 받는다)
    if not (os.path.exists("/models/h2o/serve.sh") and os.path.exists("/models/h2o/_complete")):
        from huggingface_hub import snapshot_download
        for attempt in range(5):
            try:
                snapshot_download("h2oai/h2o-lightning-4b", local_dir="/models/h2o", revision=H2O_REVISION, max_workers=16)
                break
            except Exception as err:
                print(f"모델 받기 실패 {attempt + 1}/5: {str(err)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError("모델을 받지 못함")
        open("/models/h2o/_complete", "w").write(H2O_REVISION)
        models.commit()
    print(f"모델 준비 {time.time() - t_start:.0f}초", flush=True)

    # 2) 서버: 모델 저장소의 serve.sh (vLLM + shim). Cloud Run 때와 같은 설정
    env = {**os.environ, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": "--max-num-batched-tokens 8192 --quantization fp8",
           "MAX_PIXELS": "2000000", "SHIM_MAX_PIXELS": "2000000", "SHIM_QUESTION_THREADS": "512", "PORT": "8741"}
    log = open("/tmp/serve.log", "w")
    server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=log, stderr=subprocess.STDOUT, env=env)
    url = "http://127.0.0.1:8741/v1/systemone"
    truth = json.load(open("/data/truth.json"))["videos"]
    t_load = time.time()
    _load_packs([v["vid"] for v in truth])     # 서버가 뜨는 몇 분 동안 끝난다
    print(f"프레임 묶음 올림 {time.time() - t_load:.0f}초, 영상 {len(FRAMES)}개, {sum(len(b) for f in FRAMES.values() for b in f.values()) / 1e9:.2f}GB", flush=True)

    def post(payload, timeout=300):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    # 준비 확인: serve.sh가 제 점검(예열 질문)을 마쳐야 /health가 200을 준다. 그 전에는 질문에 답하더라도 503("warming")이다
    health, last = None, ""
    while health is None:
        if server.poll() is not None or time.time() - t_start > 20 * 60:
            print(open("/tmp/serve.log").read()[-4000:])
            raise RuntimeError(f"서버가 뜨지 못함: {last}")
        try:
            health = urllib.request.urlopen("http://127.0.0.1:8741/health", timeout=30).read().decode()
        except urllib.error.HTTPError as err:
            last = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            time.sleep(5)
        except Exception as err:
            last = f"{type(err).__name__}: {err}"[:200]
            time.sleep(5)
    t_ready = time.time()
    print(f"서버 준비 {t_ready - t_start:.0f}초 | {health[:600]}", flush=True)

    # 3) 묻기
    items = _subset(truth) if small else [(v["vid"], k) for v in truth for k in range(v["seconds"])]
    if limit:
        items = items[::max(1, len(items) // limit)][:limit]
    names = [x for x in layouts.split(",") if x] or list(LAYOUTS)
    os.makedirs("/data/results", exist_ok=True)

    def ask(vid, k, layout):
        for attempt in range(4):
            try:
                t0 = time.time()
                out = post(_body(vid, k, layout))
                u = out.get("usage") or {}
                return {"vid": vid, "k": k, "probs": {q: float(a["noul"]) for q, a in out["answers"].items()},
                        "tokens": u.get("input_tokens"), "ms": u.get("latency_ms"), "seconds": round(time.time() - t0, 3)}
            except urllib.error.HTTPError as err:
                msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
            except Exception as err:
                msg = f"{type(err).__name__}: {err}"[:200]
            time.sleep(2 ** attempt)
        return {"vid": vid, "k": k, "error": msg}

    # 형식 확인: 모양마다 4건을 먼저 보내, 답이 오지 않으면 여기서 멈춘다
    for layout in names:
        first = [ask(vid, k, layout) for vid, k in items[:: max(1, len(items) // 4)][:4]]
        print("형식 확인", layout, json.dumps(first[0], ensure_ascii=False)[:400], flush=True)
        if any("error" in r for r in first):
            print(open("/tmp/serve.log").read()[-3000:])
            raise RuntimeError(f"형식 확인 실패: {layout}: {[r.get('error') for r in first]}")

    summary = {}
    for layout in names:
        suffix = "_small" if small else (f"_try{limit}" if limit else "")
        t0, recs, n_workers = time.time(), [], min(workers, WORKERS.get(layout, workers))
        with cf.ThreadPoolExecutor(n_workers) as ex:
            for n, r in enumerate(ex.map(lambda it: ask(it[0], it[1], layout), items), 1):
                recs.append(r)
                if n % 500 == 0:
                    print(f"  {layout} {n}/{len(items)} {time.time() - t0:.0f}초", flush=True)
        wall, errs = time.time() - t0, [r for r in recs if "error" in r]
        good = [r for r in recs if "error" not in r]
        toks = sorted(r["tokens"] or 0 for r in good)
        meta = {"layout": layout, "spec": {k: v for k, v in LAYOUTS[layout].items()}, "sentences": SENT, "requests": len(recs), "errors": len(errs),
                "first_error": errs[0]["error"] if errs else None, "wall_seconds": round(wall, 1), "workers": n_workers, "subset": "small" if small else "all",
                "tokens_median": toks[len(toks) // 2] if toks else None, "tokens_total": sum(toks), "server": health[:600],
                "settings": {k: env[k] for k in ("TWO_WAY", "VLLM_PATCHES", "VLLM_ARGS", "MAX_PIXELS")}, "gpu": "Modal RTX-PRO-6000", "h2o_revision": H2O_REVISION}
        with open(f"/data/results/{layout}{suffix}.json", "w") as f:
            json.dump({"meta": meta, "records": recs}, f)
        data.commit()
        summary[layout] = {k: meta[k] for k in ("requests", "errors", "wall_seconds", "tokens_median", "first_error")}
        print("끝", layout, summary[layout], flush=True)
    server.terminate()
    summary["_times"] = {"model_and_server_seconds": round(t_ready - t_start), "total_seconds": round(time.time() - t_start)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary
