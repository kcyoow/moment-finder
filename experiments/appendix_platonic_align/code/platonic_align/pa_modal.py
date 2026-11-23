"""platonic_align: Modal에서 도는 부분. 계정은 <modal-profile>(MODAL_PROFILE=<modal-profile>), 볼륨은 mf-platonic-align.

논문(arXiv 2610.39302)의 공개 코드(github.com/galilai-group/prh-proof, 커밋 a2b9949)를 컨테이너 안에 받아 그대로 쓴다.
이 Mac에서는 그 코드를 실행하지 않는다.

함수
  fetch_data : (CPU) Flickr30k(Hugging Face nlphuji/flickr30k)의 주석 CSV와 그림 zip을 볼륨에 받는다
  smoke      : (GPU) 글 JEPA 학습을 50걸음, 250걸음 돌려 걸음당 시간을 재고, DINOv2와 LeVJEPA의 속도를 잰다 → /vol/out/smoke.json
  train_text : (GPU) 좋은 뷰 글 JEPA. 저장소의 `train.py --modality text --caption-views`(빠른 시작) 그대로 → /vol/runs/text_good
  embed_all  : (GPU) Flickr 그림·캡션 임베딩, 회전, 정렬 표. 우리 프레임(TVSum, 마인크래프트)과 문장의 임베딩 → /vol/out/
  msrvtt_frames, embed_video : (2026-11-16에 더함, README 12절) MSR-VTT 클립으로 회전을 다시 맞추기 위한 것. 파일 끝에 있다

실행(승인 뒤, 이 Mac에서. 차례대로):
  MODAL_PROFILE=<modal-profile> modal volume create mf-platonic-align
  MODAL_PROFILE=<modal-profile> modal run experiments/platonic_align/pa_modal.py::fetch_data
  MODAL_PROFILE=<modal-profile> modal run experiments/platonic_align/pa_modal.py::smoke
  MODAL_PROFILE=<modal-profile> modal run --detach experiments/platonic_align/pa_modal.py::train_text --epochs 180 --budget 7900   # 에폭은 smoke 결과로 정한다
  MODAL_PROFILE=<modal-profile> modal volume put mf-platonic-align experiments/platonic_align/data/packs /ours/packs
  MODAL_PROFILE=<modal-profile> modal volume put mf-platonic-align experiments/platonic_align/data/positions.json /ours/positions.json
  MODAL_PROFILE=<modal-profile> modal volume put mf-platonic-align experiments/platonic_align/data/queries.json /ours/queries.json
  MODAL_PROFILE=<modal-profile> modal run --detach experiments/platonic_align/pa_modal.py::embed_all
  (결과를 받은 뒤 /ours/packs 를 지우고, 끝나면 볼륨을 지운다)

Modal에 올라가는 것: 224×224 프레임 묶음, 위치의 시각, 문장(TVSum 제목과 기준의 설명). 사람 점수는 올리지 않는다.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

PRH_SHA = "a2b9949fa907d4fbe562742d4b1ffa7205ab7427"      # galilai-group/prh-proof, 2026-10-30 "Initial release"
SPT_SHA = "ab836bf699a2be712dfcf980a5eb70d36391e876"      # galilai-group/stable-pretraining, 논문이 쓴 판(docs/reproduce.md)
LEV = "galilai-group/LeVJEPA-1.5-Large"                   # 가중치 CC BY-NC 4.0
GPU = "L40S"
FPS, CLIP = 7.5, 16
DINO_OFFSETS = (-0.75, -0.25, 0.25, 0.75)
ENV = {"PRH_FLICKR": "/tmp/flickr30k", "HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

app = modal.App("moment-finder-platonic-align")
vol = modal.Volume.from_name("mf-platonic-align", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip")
    .run_commands(
        f"git clone https://github.com/galilai-group/prh-proof /prh && cd /prh && git checkout {PRH_SHA}",
        f"pip install 'stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@{SPT_SHA}'",
        "pip install 'transformers>=4.56' 'timm>=1.0' numpy pandas pyarrow pillow scipy 'POT>=0.9' 'datasets>=2.20' "
        "pycocoevalcap matplotlib scikit-learn huggingface_hub",
        "cd /prh && pip install --no-deps -e .",
    )
)


# ---------------------------------------------------------------------  순수 계산(이 Mac에서도 시험한다)
def frame_indices(centers, n_frames):
    """위치의 가운데 시각들 → (DINOv2가 볼 4장의 번호, LeVJEPA 클립 16장의 번호). 번호는 0부터, 범위를 벗어나면 끝 장으로 채운다."""
    dino, clip = [], []
    for c in centers:
        dino.append([min(max(int(round((c + d) * FPS)), 0), n_frames - 1) for d in DINO_OFFSETS])
        mid = int(round(c * FPS))
        clip.append([min(max(mid - CLIP // 2 + j, 0), n_frames - 1) for j in range(CLIP)])
    return dino, clip


def pick_epochs(seconds_per_step, steps_per_epoch=113, budget=6600, want=200):
    """미리 정한 규칙: 200에폭이 110분에 안 들어가면 10의 배수로 줄인다. 100 미만이면 돌리지 않고 묻는다(None)."""
    fit = int(budget / (seconds_per_step * steps_per_epoch))
    if fit >= want:
        return want
    fit = fit // 10 * 10
    return fit if fit >= 100 else None


# ---------------------------------------------------------------------  컨테이너 안의 도우미
def _env():
    return {**os.environ, **ENV}


def _flickr(images):
    src, dst = Path("/vol/data/flickr30k"), Path("/tmp/flickr30k")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "flickr_annotations_30k.csv", dst / "flickr_annotations_30k.csv")
    if images and not (dst / "flickr30k-images").exists():
        t = time.time()
        subprocess.run(["unzip", "-q", str(src / "flickr30k-images.zip"), "-d", str(dst)], check=True)
        print(f"그림 zip 풀기 {time.time() - t:.0f}초, {sum(1 for _ in (dst / 'flickr30k-images').iterdir())}장", flush=True)


def _versions():
    import timm
    import torch
    import transformers
    return {"torch": str(torch.__version__), "transformers": str(transformers.__version__), "timm": str(timm.__version__),   # 글자열로(이 Mac에 torch가 없다)
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "prh": PRH_SHA, "stable_pretraining": SPT_SHA}


def _norm(x):
    import torch
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255 - m) / s


def _lev():
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(LEV, trust_remote_code=True).eval().to("cuda", torch.float32)


def _lev_cls(net, clips):
    """clips: (B, 3, 16, 224, 224) 정규화된 것 → [cls] (B, 1024)."""
    import torch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(pixel_values=clips.contiguous())
    return out["pooler_output"].float().cpu()


def _train(extra, cache, budget, log):
    """저장소의 train.py를 그대로 돌린다. 출력은 파일로 보내고 1분마다 마지막 줄만 찍는다. budget초를 넘기면 멈춘다."""
    cmd = [sys.executable, "train.py", "--modality", "text", "--caption-views", "--workers", "8", "--cache-dir", str(cache), *extra]
    t = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd="/prh", env=_env(), stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(15)
            if int(time.time() - t) % 60 < 15:
                tail = subprocess.run(["tail", "-c", "300", log], capture_output=True, text=True).stdout.replace("\r", "\n").strip().splitlines()
                print(f"[{time.time() - t:.0f}초] {tail[-1][:200] if tail else ''}", flush=True)
            if time.time() - t > budget:
                p.kill()
                p.wait()
                return "timeout", time.time() - t
    return ("finished" if p.returncode == 0 else f"failed({p.returncode})"), time.time() - t


# ---------------------------------------------------------------------  fetch_data
@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192)
def fetch_data():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    dst = Path("/vol/data/flickr30k")
    for f in ("flickr_annotations_30k.csv", "flickr30k-images.zip"):
        if not (dst / f).exists():
            hf_hub_download("nlphuji/flickr30k", f, repo_type="dataset", local_dir=str(dst))
    df = pd.read_csv(dst / "flickr_annotations_30k.csv")
    out = {"files": {f.name: f.stat().st_size for f in dst.iterdir() if f.is_file()}, "columns": list(df.columns), "splits": df.split.value_counts().to_dict()}
    Path("/vol/out").mkdir(parents=True, exist_ok=True)
    Path("/vol/out/fetch.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------  smoke
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=900, cpu=8, memory=32768)
def smoke():
    import torch
    _flickr(images=False)
    out = {"versions": _versions(), "train": {}}
    for steps in (50, 250):
        status, secs = _train(["--steps", str(steps)], f"/tmp/smoke{steps}", 400, f"/tmp/smoke{steps}.log")
        out["train"][steps] = {"status": status, "seconds": round(secs, 1)}
        print(steps, status, round(secs, 1), flush=True)
        if status != "finished":
            print(subprocess.run(["tail", "-n", "60", f"/tmp/smoke{steps}.log"], capture_output=True, text=True).stdout[-6000:])
            break
    if all(v["status"] == "finished" for v in out["train"].values()) and len(out["train"]) == 2:
        sps = (out["train"][250]["seconds"] - out["train"][50]["seconds"]) / 200
        out["seconds_per_step"] = round(sps, 4)
        out["epochs_by_rule"] = pick_epochs(sps)
        out["projected_seconds_200_epochs"] = round(sps * 113 * 200)
        ck = sorted(Path("/tmp/smoke250").rglob("*.ckpt"))
        out["run_files"] = sorted(str(p.relative_to("/tmp/smoke250")) for p in Path("/tmp/smoke250").rglob("*") if p.is_file())[:40]
        out["ckpt_bytes"] = ck[-1].stat().st_size if ck else None
    # 인코더 속도(무작위 입력)
    os.environ.update(ENV)
    sys.path.insert(0, "/prh")
    os.chdir("/prh")
    from prh import features
    x = torch.rand(64, 3, 224, 224, device="cuda")
    net = features._image_net("dinov2", False)[0].eval().to("cuda")
    with torch.no_grad():
        net(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            y = net(x)
        torch.cuda.synchronize()
    out["dinov2"] = {"images_per_second": round(5 * 64 / (time.time() - t)), "dim": int(y.shape[1])}
    del net
    lev = _lev()
    clips = torch.rand(32, 3, CLIP, 224, 224, device="cuda")
    y = _lev_cls(lev, clips)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        y = _lev_cls(lev, clips)
    torch.cuda.synchronize()
    out["levjepa"] = {"clips_per_second": round(3 * 32 / (time.time() - t), 1), "dim": int(y.shape[1]),
                      "gpu_gib_peak": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    Path("/vol/out").mkdir(parents=True, exist_ok=True)
    Path("/vol/out/smoke.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------  train_text
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=8100, cpu=8, memory=32768)
def train_text(epochs: int = 200, budget: int = 7500):
    _flickr(images=False)
    cache = Path("/tmp/runs")
    status, secs = _train(["--epochs", str(epochs)], cache, budget, "/tmp/train.log")
    cks = sorted(cache.rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
    out = {"status": status, "seconds": round(secs), "epochs": epochs, "versions": _versions(), "ckpt": None}
    dest = Path("/vol/runs/text_good")
    if cks:
        run = cks[-1].parent.parent
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "checkpoints").mkdir(parents=True)
        shutil.copy(cks[-1], dest / "checkpoints" / "last.ckpt")
        embs = [p for p in run.rglob("validation_epoch=*.pt")]
        (dest / "embeddings").mkdir()
        for p in embs:
            shutil.copy(p, dest / "embeddings" / p.name)
        out.update({"ckpt": "runs/text_good/checkpoints/last.ckpt", "ckpt_bytes": cks[-1].stat().st_size, "run_dir": str(run),
                    "validation_dumps": len(embs)})
    shutil.copy("/tmp/train.log", "/vol/out/train.log")
    Path("/vol/out/train.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    print(subprocess.run(["tail", "-n", "25", "/tmp/train.log"], capture_output=True, text=True).stdout[-4000:])
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------  embed_all
@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=3600, cpu=8, memory=49152)
def embed_all(lev_flickr: str = "all"):
    """lev_flickr: 'all'이면 Flickr 31k장을 모두 LeVJEPA에 넣는다. '10k'면 검증·시험 2,014장과 학습 쪽 앞 9,000장만 넣는다(느릴 때의 미리 정한 대안)."""
    import io
    import tarfile

    import numpy as np
    import torch
    from PIL import Image
    t0 = time.time()
    _flickr(images=True)
    os.environ.update(ENV)
    sys.path.insert(0, "/prh")
    os.chdir("/prh")
    from prh import data, features, metrics
    from prh.align import procrustes
    ckpt = "/vol/runs/text_good/checkpoints/last.ckpt"
    out_dir = Path("/vol/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"versions": _versions(), "lev_flickr": lev_flickr, "seconds": {}}

    def mark(name):
        log["seconds"][name] = round(time.time() - t0)
        print(f"== {name} {log['seconds'][name]}초", flush=True)

    # ---- Flickr: 그림과 첫 캡션 (논문의 embed와 같다)
    pairs = data.flickr_pairs(("train", "val", "test"))
    vt = set(data.flickr_pairs(("val", "test")).pair_id.tolist())
    paths, caps = pairs.path.tolist(), [c[0] for c in pairs.captions]
    is_test = torch.tensor(pairs.is_test.values)
    in_vt = torch.tensor([p in vt for p in pairs.pair_id.tolist()])
    xt = features.embed_texts(ckpt, caps).double()
    mark("flickr_text")
    xd = features.embed_images("dinov2", paths).double()
    mark("flickr_dinov2")

    class Img(torch.utils.data.Dataset):
        def __init__(self, ps):
            self.ps = ps

        def __len__(self):
            return len(self.ps)

        def __getitem__(self, i):      # 짧은 변 256 → 가운데 224 (논문의 image_eval_view와 같은 모양)
            im = Image.open(self.ps[i]).convert("RGB")
            w, h = im.size
            s = 256 / min(w, h)
            im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
            w, h = im.size
            l, u = (w - 224) // 2, (h - 224) // 2
            return torch.from_numpy(np.asarray(im.crop((l, u, l + 224, u + 224))).copy()).permute(2, 0, 1)

    lev = _lev()
    n_all = len(paths)
    if lev_flickr == "all":
        sub = torch.ones(n_all, dtype=torch.bool)
    else:
        train_rows = torch.nonzero(~in_vt).flatten()[:9000]
        sub = in_vt.clone()
        sub[train_rows] = True
    rows = torch.nonzero(sub).flatten().tolist()
    feats = []
    for x in torch.utils.data.DataLoader(Img([paths[i] for i in rows]), batch_size=32, num_workers=8):
        feats.append(_lev_cls(lev, _norm(x.cuda()).unsqueeze(2).expand(-1, -1, CLIP, -1, -1)))
    xl = torch.cat(feats).double()
    mark("flickr_levjepa")

    def table(a, b, test, keep):
        return {"fit_all": metrics.alignment(a, b, test),
                "fit_1k_val": metrics.alignment(a[keep], b[keep], test[keep])}
    tables = {"n_pairs": n_all, "n_test": int(is_test.sum()), "n_val_test": int(in_vt.sum()), "n_levjepa": len(rows),
              "dinov2": table(xd, xt, is_test, in_vt),
              "dinov2_on_levjepa_rows": table(xd[sub], xt[sub], is_test[sub], in_vt[sub]),
              "levjepa": table(xl, xt[sub], is_test[sub], in_vt[sub])}
    print(json.dumps(tables, indent=1), flush=True)
    rot = {}
    for name, a, b, test in (("dinov2", xd, xt, is_test), ("levjepa", xl, xt[sub], is_test[sub])):
        mu_a, mu_b = a[~test].mean(0), b[~test].mean(0)
        rot[name] = {"R": procrustes(a[~test] - mu_a, b[~test] - mu_b).float().numpy(), "mu_a": mu_a.float().numpy(), "mu_b": mu_b.float().numpy()}
    mark("rotations")

    # ---- 우리 문장
    q = json.loads(Path("/vol/ours/queries.json").read_text())
    xq = features.embed_texts(ckpt, q["texts"]).float().numpy()
    mark("query_texts")

    # ---- 우리 프레임
    pos = json.loads(Path("/vol/ours/positions.json").read_text())
    dino = features._image_net("dinov2", False)[0].eval().to("cuda")

    class Frames(torch.utils.data.Dataset):
        def __init__(self, blobs):
            self.blobs = blobs

        def __len__(self):
            return len(self.blobs)

        def __getitem__(self, i):
            return torch.from_numpy(np.asarray(Image.open(io.BytesIO(self.blobs[i])).convert("RGB")).copy()).permute(2, 0, 1)

    ours, counts = {}, {}
    for ds in ("tvsum", "minecraft"):
        for vid, centers in pos[ds].items():
            with tarfile.open(f"/vol/ours/packs/{vid}.tar") as tar:
                names = sorted(m.name for m in tar.getmembers())
                blobs = [tar.extractfile(n).read() for n in names]
            X = torch.cat(list(torch.utils.data.DataLoader(Frames(blobs), batch_size=256, num_workers=8)))     # (N, 3, 224, 224) uint8
            di, ci = frame_indices(centers, len(X))
            need = sorted({i for row in di for i in row})                                                       # DINOv2는 쓰는 프레임만
            at = {f: j for j, f in enumerate(need)}
            need_t = torch.tensor(need)
            with torch.no_grad():
                fd = torch.cat([dino(_norm(X[need_t[i:i + 256]].cuda())).float().cpu() for i in range(0, len(need), 256)])
            e_d = fd[torch.tensor([[at[i] for i in row] for row in di])].mean(1)                                # (P, 768) 4장 평균
            e_l, e_s = [], []
            ci_t = torch.tensor(ci)
            for i in range(0, len(ci), 32):
                idx = ci_t[i:i + 32]                                                       # (B, 16)
                clips = _norm(X[idx.flatten()].cuda()).view(len(idx), CLIP, 3, 224, 224).permute(0, 2, 1, 3, 4)
                e_l.append(_lev_cls(lev, clips))
                mid = _norm(X[idx[:, CLIP // 2]].cuda())                                   # 가운데 한 장을 16번 되풀이한 정지 클립
                e_s.append(_lev_cls(lev, mid.unsqueeze(2).expand(-1, -1, CLIP, -1, -1)))
            ours[f"{ds}/{vid}/dino"] = e_d.numpy()
            ours[f"{ds}/{vid}/lev"] = torch.cat(e_l).numpy()
            ours[f"{ds}/{vid}/lev_static"] = torch.cat(e_s).numpy()
            counts[vid] = {"frames": len(X), "positions": len(centers)}
            print(f"  {vid}: {len(X)}장 {len(centers)}곳 ({time.time() - t0:.0f}초)", flush=True)
    mark("ours")
    np.savez_compressed(out_dir / "ours.npz", q=xq, **{f"rot/{k}/{p}": v for k, r in rot.items() for p, v in r.items()}, **ours)
    log["tables"], log["counts"] = tables, counts
    (out_dir / "embed.json").write_text(json.dumps(log, indent=1))
    vol.commit()
    mark("saved")
    return log["seconds"]


# =====================================================================  12절: 회전을 진짜 영상으로 맞춘다 (2026-11-16에 더함)
# 실행(승인 뒤, 이 Mac에서. 차례대로):
#   MODAL_PROFILE=<modal-profile> modal volume put mf-platonic-align experiments/platonic_align/results/text_good_last.ckpt /runs/text_good/checkpoints/last.ckpt
#   MODAL_PROFILE=<modal-profile> modal run experiments/platonic_align/pa_modal.py::msrvtt_frames --limit 200     # 먼저 돌려 보기
#   MODAL_PROFILE=<modal-profile> modal run --detach experiments/platonic_align/pa_modal.py::msrvtt_frames
#   MODAL_PROFILE=<modal-profile> modal run experiments/platonic_align/pa_modal.py::embed_video --limit 320       # 먼저 돌려 보기
#   MODAL_PROFILE=<modal-profile> modal run --detach experiments/platonic_align/pa_modal.py::embed_video
# Modal에 올라가는 것: 글 JEPA 체크포인트뿐이다. 우리 영상의 프레임, 문장, 점수는 올리지 않는다.
MSRVTT_REPO, MSRVTT_REV = "friedrichor/MSR-VTT", "c1af215a96934854f42683c19c51391aaee6f962"
VF = f"fps={FPS},scale=w='if(gt(iw,ih),-2,256)':h='if(gt(iw,ih),256,-2)',crop=224:224"      # pa_prepare._extract와 같은 필터
SHARD = 500
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("imageio-ffmpeg==0.8.1", "huggingface_hub")   # 이 Mac에서 프레임을 뽑은 판(ffmpeg 9.0)


def msrvtt_ids(ann, limit=0):
    """주석 폴더 → (맞추는 쪽 9,000개, 시험 쪽 1,000개, {시험 클립: JSFUSION 문장}). limit > 0이면 4분의 3과 4분의 1로 앞에서 자른다."""
    import csv

    def rows(name):
        with open(Path(ann) / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    train = [r["video_id"] for r in rows("MSRVTT_train.9k.csv")]
    js = rows("MSRVTT_JSFUSION_test.csv")
    test, sent = [r["video_id"] for r in js], {r["video_id"]: r["sentence"] for r in js}
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise RuntimeError("MSR-VTT 나눔에 겹치는 클립이 있다")
    if limit:
        k = limit * 3 // 4
        train, test = train[:k], test[:limit - k]
    return train, test, sent


def clip_frames(job):
    """클립 하나: 초당 7.5장을 모두 뽑아 세고, 가운데 창(LeVJEPA 16장, DINOv2 4장)에 드는 장만 돌려준다."""
    import tempfile

    import imageio_ffmpeg
    vid, src = job
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf", VF, "-q:v", "3", str(Path(td) / "%06d.jpg")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        files = sorted(Path(td).glob("*.jpg"))
        if p.returncode != 0 or not files:
            return vid, None, (p.stderr or "프레임 없음")[:200]
        n = len(files)
        di, ci = frame_indices([n / FPS / 2], n)
        need = sorted(set(di[0]) | set(ci[0]))
        return vid, {"n": n, "dino": di[0], "clip": ci[0], "blobs": {i: files[i].read_bytes() for i in need}}, None


@app.function(image=cpu_image, volumes={"/vol": vol}, timeout=1500, cpu=16, memory=16384)
def msrvtt_frames(limit: int = 0):
    """MSR-VTT를 받아 클립마다 가운데 창의 프레임만 500개씩 tar로 묶는다. limit > 0이면 그만큼만(먼저 돌려 보기, 폴더가 다르다)."""
    import io
    import tarfile
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    import imageio_ffmpeg
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    raw = Path("/vol/msrvtt/raw")
    files = ["MSRVTT_Videos.zip"] + [f"raw_data/{n}" for n in ("MSRVTT_data.json", "MSRVTT_train.9k.csv", "MSRVTT_train.7k.csv", "MSRVTT_JSFUSION_test.csv")]
    for f in files:
        if not (raw / f).exists():
            hf_hub_download(MSRVTT_REPO, f, repo_type="dataset", revision=MSRVTT_REV, local_dir=str(raw))
    vol.commit()
    t_dl = time.time() - t0
    train, test, _ = msrvtt_ids(raw / "raw_data", limit)
    ids = train + test
    want = set(ids)
    tmp = Path("/tmp/msrvtt_videos")
    tmp.mkdir(parents=True, exist_ok=True)
    src = {}
    local_zip = Path("/tmp/MSRVTT_Videos.zip")          # 볼륨에서 한 번에 읽어 온 뒤 컨테이너 디스크에서 푼다
    shutil.copy(raw / "MSRVTT_Videos.zip", local_zip)
    with zipfile.ZipFile(local_zip) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".mp4")]
        for n in names:
            stem = Path(n).stem
            if stem in want:
                dst = tmp / f"{stem}.mp4"
                dst.write_bytes(z.read(n))
                src[stem] = dst
    t_unzip = time.time() - t0 - t_dl
    print(f"받기 {t_dl:.0f}초, 풀기 {t_unzip:.0f}초. zip 안의 mp4 {len(names)}개, 쓸 클립 {len(src)}/{len(ids)}", flush=True)
    out = Path("/vol/msrvtt/frames_smoke" if limit else "/vol/msrvtt/frames")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    clips, failed, total = {}, {v: "zip에 없음" for v in ids if v not in src}, 0
    t1 = time.time()
    with ThreadPoolExecutor(16) as pool:
        for k in range(0, len(ids), SHARD):
            part = [v for v in ids[k:k + SHARD] if v in src]
            name = f"shard_{k // SHARD:03d}.tar"
            with tarfile.open(out / name, "w") as tar:
                for vid, got, err in pool.map(clip_frames, [(v, src[v]) for v in part]):
                    if got is None:
                        failed[vid] = err
                        continue
                    for i, b in got["blobs"].items():
                        info = tarfile.TarInfo(f"{vid}/{i:06d}.jpg")
                        info.size = len(b)
                        tar.addfile(info, io.BytesIO(b))
                        total += len(b)
                    clips[vid] = {"n": got["n"], "dino": got["dino"], "clip": got["clip"], "shard": name}
            print(f"  {name}: 누적 {len(clips)}개, {time.time() - t1:.0f}초", flush=True)
    secs = time.time() - t1
    ns = sorted(c["n"] for c in clips.values())
    summ = {"repo": MSRVTT_REPO, "revision": MSRVTT_REV, "limit": limit, "clips": len(clips), "failed": dict(list(failed.items())[:30]), "n_failed": len(failed),
            "frame_bytes": total, "seconds": {"download": round(t_dl), "unzip": round(t_unzip), "frames": round(secs), "all": round(time.time() - t0)},
            "clips_per_second": round(len(clips) / max(secs, 1e-9), 1), "ffmpeg": str(imageio_ffmpeg.get_ffmpeg_version()),
            "frames_per_clip": {"min": ns[0], "median": ns[len(ns) // 2], "max": ns[-1]} if ns else None,
            "shorter_than_window": sum(n < CLIP for n in ns), "filter": VF}
    (out / "manifest.json").write_text(json.dumps({"summary": summ, "clips": clips}))
    Path("/vol/out").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/out/video_frames{'_smoke' if limit else ''}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    vol.commit()
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=2100, cpu=8, memory=32768)
def embed_video(limit: int = 0):
    """MSR-VTT 클립의 임베딩(L, Ls, D)과 글 임베딩 → /vol/out/video.npz. 저장소의 함수로 낸 1k-A 표 → /vol/out/video.json."""
    import io
    import tarfile

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    t0 = time.time()
    os.environ.update(ENV)
    sys.path.insert(0, "/prh")
    os.chdir("/prh")
    from prh import features, metrics
    from prh.align import procrustes
    tag = "_smoke" if limit else ""
    ckpt = "/vol/runs/text_good/checkpoints/last.ckpt"
    fdir = Path(f"/vol/msrvtt/frames{tag}")
    ann = Path("/vol/msrvtt/raw/raw_data")
    out_dir = Path("/vol/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {"versions": _versions(), "limit": limit, "seconds": {}}

    def mark(name):
        log["seconds"][name] = round(time.time() - t0)
        print(f"== {name} {log['seconds'][name]}초", flush=True)

    man = json.loads((fdir / "manifest.json").read_text())["clips"]
    train, test, sent = msrvtt_ids(ann, limit)
    ids = [v for v in train + test if v in man]
    is_test = np.array([v in set(test) for v in ids])
    log["missing"] = [v for v in train + test if v not in man][:30]

    # ---- 글: 클립마다 캡션 20개(평균과 첫 캡션), 시험 클립의 JSFUSION 문장
    caps = {}
    for s in sorted(json.loads((ann / "MSRVTT_data.json").read_text())["sentences"], key=lambda s: s["sen_id"]):
        caps.setdefault(s["video_id"], []).append(s["caption"])
    counts = [len(caps[v]) for v in ids]
    log["captions_per_clip"] = {"min": min(counts), "max": max(counts), "total": sum(counts)}
    flat = [c for v in ids for c in caps[v]]
    js_ids = [v for v in ids if v in set(test)]
    xt = features.embed_texts(ckpt, flat + [sent[v] for v in js_ids]).double()
    mark("texts")
    off = np.cumsum([0] + counts)
    text_mean = torch.stack([xt[off[i]:off[i + 1]].mean(0) for i in range(len(ids))])
    text_first = torch.stack([xt[off[i]] for i in range(len(ids))])
    js_text = xt[len(flat):]

    # ---- 영상: 묶음마다 읽어 L(가운데 16장), Ls(가운데 한 장을 16번), D(4장 평균)
    lev = _lev()
    dino = features._image_net("dinov2", False)[0].eval().to("cuda")
    mark("models")

    class Clips(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            blobs, ci, di = self.items[i]
            dec = {k: torch.from_numpy(np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).copy()).permute(2, 0, 1) for k, b in blobs.items()}
            return torch.stack([dec[k] for k in ci]), torch.stack([dec[k] for k in di])

    emb = {}
    by_shard = {}
    for v in ids:
        by_shard.setdefault(man[v]["shard"], []).append(v)
    for shard in sorted(by_shard):
        blobs = {}
        with tarfile.open(fdir / shard) as tar:
            for m in tar.getmembers():
                vid, fname = m.name.split("/")
                blobs.setdefault(vid, {})[int(fname.split(".")[0])] = tar.extractfile(m).read()
        vids = by_shard[shard]
        items = [(blobs[v], man[v]["clip"], man[v]["dino"]) for v in vids]
        e_l, e_s, e_d = [], [], []
        for c, d in torch.utils.data.DataLoader(Clips(items), batch_size=32, num_workers=8):
            b = len(c)
            x = _norm(c.cuda().flatten(0, 1)).view(b, CLIP, 3, 224, 224)
            e_l.append(_lev_cls(lev, x.permute(0, 2, 1, 3, 4)))
            e_s.append(_lev_cls(lev, x[:, CLIP // 2].unsqueeze(2).expand(-1, -1, CLIP, -1, -1)))
            with torch.no_grad():
                e_d.append(dino(_norm(d.cuda().flatten(0, 1))).float().cpu().view(b, len(DINO_OFFSETS), -1).mean(1))
        for v, a, s, d in zip(vids, torch.cat(e_l), torch.cat(e_s), torch.cat(e_d)):
            emb[v] = (a, s, d)
        print(f"  {shard}: 누적 {len(emb)}개 ({time.time() - t0:.0f}초)", flush=True)
    mark("clips")
    xl, xs, xd = (torch.stack([emb[v][k] for v in ids]) for k in range(3))
    np.savez_compressed(out_dir / f"video{tag}.npz", ids=np.array(ids), is_test=is_test, js_ids=np.array(js_ids),
                        n_frames=np.array([man[v]["n"] for v in ids]), lev=xl.numpy(), lev_static=xs.numpy(), dino=xd.numpy(),
                        text_mean=text_mean.float().numpy(), text_first=text_first.float().numpy(), js_text=js_text.float().numpy())
    vol.commit()
    mark("saved_npz")

    # ---- 저장소의 함수로 낸 1k-A 표(영상 회전). 논문 evaluate.py video-align과 같은 식
    te = torch.tensor(is_test)
    tables = {}
    try:
        for name, a in (("L", xl.double()), ("Ls", xs.double()), ("D", xd.double())):
            a_tr, a_te, b_tr = a[~te], a[te], text_mean[~te]
            mu_a, mu_b = a_tr.mean(0), b_tr.mean(0)
            v, t = (a_te - mu_a) @ procrustes(a_tr - mu_a, b_tr - mu_b), js_text - mu_b
            s = F.normalize(t, dim=1) @ F.normalize(v, dim=1).T
            rank = {"t2v": (s >= s.diag()[:, None]).sum(1), "v2t": (s.T >= s.diag()[:, None]).sum(1)}
            row = {"mknn": float(metrics.mutual_knn(a_te - mu_a, js_text - mu_b, 10)), "n_fit": int((~te).sum()), "n_test": int(te.sum())}
            row.update({f"{d} R@{k}": float((r <= k).double().mean()) for d, r in rank.items() for k in (1, 5, 10)})
            tables[name] = row
    except Exception as e:      # 임베딩은 이미 저장했다. 표는 이 Mac에서도 낸다
        tables["error"] = f"{type(e).__name__}: {e}"[:500]
    log["tables"], log["n"] = tables, {"clips": len(ids), "test": int(is_test.sum())}
    mark("tables")
    (out_dir / f"video{tag}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    vol.commit()
    print(json.dumps(log, ensure_ascii=False, indent=1))
    return log["seconds"]
