"""Colab(GPU)에서 LeVJEPA 특징이 장면 판정에 쓸 만한지 본다. Jev 요청은 없다.

단계 (작업 폴더 기본 /content/lp)
  prepare  : Charades-STA 주석과 Charades 영상(480p 묶음)을 받고, 쓸 영상을 고르고, LeVJEPA 속도를 잰다
  features : 고른 영상의 판정 위치마다 LeVJEPA 특징과 SigLIP 2 이미지 특징을 뽑는다
  train    : 고정 특징 위에 작은 머리를 학습해 "이 클립이 이 문장에 해당하는가"를 판정하게 하고 평가한다

자료
- 학습: Charades-STA 공식 train 쪽 영상 중 무작위 일부
- 평가: 공식 test 쪽 영상 중 무작위 일부(우리 12개 제외) + 우리 12개 영상(lev_probe_input.json)
- 클립: 판정 위치 t를 가운데에 둔 2초, 7.5fps 16프레임, 224px(짧은 변을 맞춘 뒤 가운데를 자름). lev_embed.py와 같다
- 판정 위치: 1초 간격 (loc_frames.py와 같은 규칙)

머리: 영상 특징과 글 특징(SigLIP 2 작은 모델의 글 임베딩)을 각각 512차원으로 옮기고, 둘과 둘의 곱을 이어 붙여
      2층 신경망으로 예/아니오 점수를 낸다. 위치마다 따로 판정한다.
학습 쌍: 같은 영상의 문장과 각 위치(구간 안이면 1, 밖이면 0) + 다른 영상의 문장을 붙인 쌍(0)
변형과 설정은 미리 정한 것을 모두 돌리고 씨앗 3개의 평균을 낸다. 평가 결과로 고르지 않는다.

실행: python scripts/embed/lev_probe_colab.py prepare|features|train [--root 폴더] [--n-train N] [--n-test N]
"""
import argparse
import json
import math
import random
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

LEV = "galilai-group/LeVJEPA-1.5-Large"
SIG = "google/siglip2-base-patch16-224"
FPS, FRAMES, SIZE, HALF = 7.5, 16, 224, 1.0
OFFSETS = [-0.75, -0.25, 0.25, 0.75]
TAIL_MARGIN = 0.1
MAX_TRAIN, MAX_TEST = 1200, 400
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VARIANTS = {  # 이름: (영상 특징 묶음, 글을 쓰는가)
    "LeVJEPA cls": (["lev_cls"], True),
    "LeVJEPA 패치 평균": (["lev_patch"], True),
    "LeVJEPA cls+패치": (["lev_cls", "lev_patch"], True),
    "SigLIP 이미지 (가운데 한 장)": (["sig_center"], True),
    "SigLIP 이미지 (4장 평균)": (["sig_mean4"], True),
    "LeVJEPA cls+패치 + SigLIP 4장": (["lev_cls", "lev_patch", "sig_mean4"], True),
    "글 없이 LeVJEPA cls+패치": (["lev_cls", "lev_patch"], False),
    "글 없이 SigLIP 4장": (["sig_mean4"], False),
}
SEEDS = [0, 1, 2]
EPOCHS, BATCH, HIDDEN = 30, 2048, 512


def log(*a):
    print(*a, flush=True)


def read_sta(url):
    rows = []
    for line in urllib.request.urlopen(url, timeout=60).read().decode().splitlines():
        if "##" not in line:
            continue
        head, sentence = line.split("##", 1)
        vid, s, e = head.split()
        rows.append((vid, float(s), float(e), sentence.strip().rstrip(".")))
    return rows


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def centers_for(dur):
    return [float(k) for k in range(1, int(dur) + 1) if k + OFFSETS[-1] <= dur - TAIL_MARGIN]


def decode(path):
    vf = f"fps={FPS},scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}"
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)


def clip_indices(t, n):
    return np.clip(round((t - HALF) * FPS) + np.arange(FRAMES), 0, n - 1)


def load_lev(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(LEV, trust_remote_code=True).eval().to(device)


def lev_encode(model, clips, device, half=True):
    x = torch.from_numpy(clips).to(device).float().div_(255.0)
    x = (x - torch.tensor(IMNET_MEAN, device=device)) / torch.tensor(IMNET_STD, device=device)
    x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half and device == "cuda"):
        out = model(pixel_values=x)
        cls = out["pooler_output"].float()
        patch = out.last_hidden_state[:, 1:].float().mean(1)
    return cls, patch


def prepare(root, spec):
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(spec["seed"])
    train_rows, test_rows = read_sta(spec["urls"]["train"]), read_sta(spec["urls"]["test"])
    ours = set(spec["eval_videos"])
    train_vids = sorted({r[0] for r in train_rows})
    test_vids = sorted({r[0] for r in test_rows} - ours)
    assert not ours & set(train_vids), "우리 평가 영상이 학습 쪽에 있다"
    rng.shuffle(train_vids)
    rng.shuffle(test_vids)
    chosen = {"train": train_vids[:MAX_TRAIN], "test": test_vids[:MAX_TEST], "ours": sorted(ours)}
    log(f"주석: train 문장 {len(train_rows)}개·영상 {len(train_vids)}개, test 문장 {len(test_rows)}개·영상 {len(test_vids) + len(ours)}개")

    zpath = root / "charades.zip"
    if not zpath.exists():
        t0 = time.time()
        subprocess.run(["wget", "-q", "-O", str(zpath), spec["urls"]["videos"]], check=True)
        log(f"영상 묶음 {zpath.stat().st_size / 1e9:.1f}GB, {time.time() - t0:.0f}초")
    (root / "videos").mkdir(exist_ok=True)
    need = {v for vs in chosen.values() for v in vs}
    with zipfile.ZipFile(zpath) as zf:
        names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".mp4")}
        missing = sorted(need - set(names))
        for v in sorted(need & set(names)):
            dest = root / "videos" / f"{v}.mp4"
            if not dest.exists():
                dest.write_bytes(zf.read(names[v]))
    log(f"꺼낸 영상 {len(need) - len(missing)}개, 묶음에 없는 영상 {len(missing)}개 {missing[:5]}")

    plan = {"chosen": {k: [v for v in vs if v not in missing] for k, vs in chosen.items()}, "videos": {}}
    for split, vs in plan["chosen"].items():
        for v in vs:
            dur = duration(root / "videos" / f"{v}.mp4")
            cs = spec["eval_videos"][v] if split == "ours" else centers_for(dur)
            plan["videos"][v] = {"split": split, "duration": round(dur, 2), "centers": cs}
    sentences = {"train": train_rows, "test": [r for r in test_rows if r[0] not in ours]}
    plan["items"] = {split: [{"video": v, "window": [s, e], "sentence": text} for v, s, e, text in rows
                             if v in set(plan["chosen"][split])] for split, rows in sentences.items()}
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False) + "\n")
    for split in ("train", "test", "ours"):
        vs = plan["chosen"][split]
        log(f"{split}: 영상 {len(vs)}개, 위치 {sum(len(plan['videos'][v]['centers']) for v in vs)}개"
            + (f", 문장 {len(plan['items'][split])}개" if split in plan["items"] else ""))

    # 속도 측정: 영상 2개의 클립으로 반정밀과 단정밀을 잰다
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "없음"
    model = load_lev(device)
    frames = decode(root / "videos" / f"{plan['chosen']['train'][0]}.mp4")
    cs = plan["videos"][plan["chosen"]["train"][0]]["centers"][:16]
    clips = np.stack([frames[clip_indices(t, len(frames))] for t in cs])
    lev_encode(model, clips[:2], device)
    result = {"gpu": gpu}
    for half in (True, False):
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        for _ in range(3):
            cls, patch = lev_encode(model, clips, device, half=half)
        torch.cuda.synchronize() if device == "cuda" else None
        result["half" if half else "full"] = round(3 * len(clips) / (time.time() - t0), 1)
        if half:
            keep = cls
        else:
            result["cos_half_vs_full"] = round(torch.nn.functional.cosine_similarity(keep, cls).min().item(), 5)
    t0 = time.time()
    for v in plan["chosen"]["train"][:8]:
        decode(root / "videos" / f"{v}.mp4")
    result["decode_seconds_per_video"] = round((time.time() - t0) / 8, 2)
    (root / "timing.json").write_text(json.dumps(result) + "\n")
    log("속도:", json.dumps(result, ensure_ascii=False))


def features(root, n_train, n_test):
    from transformers import AutoModel
    plan = json.loads((root / "plan.json").read_text())
    device = "cuda"
    lev = load_lev(device)
    sig = AutoModel.from_pretrained(SIG).eval().to(device)
    todo = plan["chosen"]["ours"] + plan["chosen"]["test"][:n_test] + plan["chosen"]["train"][:n_train]
    done_path = root / "features.npz"
    rows = {k: [] for k in ("vid", "t", "lev_cls", "lev_patch", "sig_center", "sig_mean4")}
    started = time.time()
    lev_seconds = sig_seconds = 0.0
    with ThreadPoolExecutor(8) as pool:
        decoded = pool.map(lambda v: decode(root / "videos" / f"{v}.mp4"), todo)
        for n, (v, frames) in enumerate(zip(todo, decoded), 1):
            cs = plan["videos"][v]["centers"]
            clips = np.stack([frames[clip_indices(t, len(frames))] for t in cs])
            t0 = time.time()
            out_cls, out_patch = [], []
            for i in range(0, len(clips), 16):
                c, p = lev_encode(lev, clips[i:i + 16], device)
                out_cls.append(c)
                out_patch.append(p)
            torch.cuda.synchronize()
            lev_seconds += time.time() - t0
            # SigLIP: 필요한 프레임만 한 번씩 넣는다
            t0 = time.time()
            idx = lambda sec: int(np.clip(round(sec * FPS), 0, len(frames) - 1))
            need = sorted({idx(t) for t in cs} | {idx(t + o) for t in cs for o in OFFSETS})
            x = torch.from_numpy(frames[need]).to(device).float().div_(255.0).sub_(0.5).div_(0.5).permute(0, 3, 1, 2)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                emb = torch.cat([f if torch.is_tensor(f) else f.pooler_output for f in
                                 (sig.get_image_features(pixel_values=x[i:i + 256]) for i in range(0, len(x), 256))]).float()
            torch.cuda.synchronize()
            sig_seconds += time.time() - t0
            pos = {k: j for j, k in enumerate(need)}
            center = torch.stack([emb[pos[idx(t)]] for t in cs])
            mean4 = torch.stack([torch.stack([emb[pos[idx(t + o)]] for o in OFFSETS]).mean(0) for t in cs])
            rows["vid"].extend([v] * len(cs))
            rows["t"].extend(cs)
            rows["lev_cls"].append(torch.cat(out_cls).cpu().numpy().astype(np.float16))
            rows["lev_patch"].append(torch.cat(out_patch).cpu().numpy().astype(np.float16))
            rows["sig_center"].append(center.cpu().numpy().astype(np.float16))
            rows["sig_mean4"].append(mean4.cpu().numpy().astype(np.float16))
            if n % 50 == 0 or n == len(todo):
                log(f"영상 {n}/{len(todo)}  클립 {len(rows['t'])}개  경과 {time.time() - started:.0f}초 "
                    f"(LeVJEPA {lev_seconds:.0f}초, SigLIP {sig_seconds:.0f}초)")
    np.savez(done_path, vid=np.array(rows["vid"]), t=np.array(rows["t"], np.float32),
             **{k: np.concatenate(rows[k]) for k in ("lev_cls", "lev_patch", "sig_center", "sig_mean4")})
    info = {"videos": len(todo), "clips": len(rows["t"]), "n_train": n_train, "n_test": n_test,
            "seconds_total": round(time.time() - started, 1), "lev_seconds": round(lev_seconds, 1),
            "sig_seconds": round(sig_seconds, 1), "lev_clips_per_second": round(len(rows["t"]) / lev_seconds, 1),
            "gpu": torch.cuda.get_device_name(0)}
    (root / "features_info.json").write_text(json.dumps(info) + "\n")
    log("끝:", json.dumps(info, ensure_ascii=False))


class Head(nn.Module):
    def __init__(self, dv, dt, use_text):
        super().__init__()
        self.use_text = use_text
        self.v = nn.Sequential(nn.Linear(dv, HIDDEN), nn.GELU(), nn.Dropout(0.2))
        self.t = nn.Sequential(nn.Linear(dt, HIDDEN), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Sequential(nn.Linear(3 * HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(0.2), nn.Linear(HIDDEN, 1))

    def forward(self, v, t):
        a = self.v(v)
        b = self.t(t if self.use_text else torch.zeros_like(t))
        return self.out(torch.cat([a, b, a * b], -1)).squeeze(-1)


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def uni5(ps):
    padded = [ps[0]] * 2 + list(ps) + [ps[-1]] * 2
    return [sum(padded[i:i + 5]) / 5 for i in range(len(ps))]


def train(root, spec):
    from transformers import AutoModel, AutoProcessor
    device = "cuda"
    plan = json.loads((root / "plan.json").read_text())
    info = json.loads((root / "features_info.json").read_text())
    z = np.load(root / "features.npz")
    vids, ts = z["vid"], z["t"]
    feats = {k: torch.from_numpy(z[k].astype(np.float32)) for k in ("lev_cls", "lev_patch", "sig_center", "sig_mean4")}
    rows_of = {}
    for i, v in enumerate(vids):
        rows_of.setdefault(str(v), []).append(i)
    have = set(rows_of)

    items = {"train": [it for it in plan["items"]["train"] if it["video"] in have],
             "test": [it for it in plan["items"]["test"] if it["video"] in have],
             "ours": [{"id": it["id"], "video": it["video"], "windows": it["windows"], "sentence": it["sentence"]}
                      for it in spec["eval_items"]]}
    for split in ("train", "test"):
        for it in items[split]:
            it["windows"] = [it.pop("window")]
    sentences = sorted({it["sentence"] for its in items.values() for it in its})
    sig = AutoModel.from_pretrained(SIG).eval().to(device)
    proc = AutoProcessor.from_pretrained(SIG)
    with torch.no_grad():
        chunks = []
        for i in range(0, len(sentences), 512):
            tok = proc(text=sentences[i:i + 512], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            f = sig.get_text_features(**tok)
            chunks.append(torch.nn.functional.normalize(f if torch.is_tensor(f) else f.pooler_output, dim=-1).float().cpu())
    text = dict(zip(sentences, torch.cat(chunks)))
    del sig

    def labels(it):
        return [any(s <= float(ts[i]) <= e for s, e in it["windows"]) for i in rows_of[it["video"]]]

    for its in items.values():
        for it in its:
            it["rows"], it["y"] = rows_of[it["video"]], labels(it)
    train_videos = sorted({it["video"] for it in items["train"]})
    log(f"학습 문장 {len(items['train'])}개(영상 {len(train_videos)}개), 평가 문장 {len(items['test'])}개, 우리 문장 {len(items['ours'])}개")

    # 학습 쌍: (클립 행, 문장 번호, 정답)
    sid = {s: i for i, s in enumerate(sentences)}
    T = torch.stack([text[s] for s in sentences]).to(device)
    pair_rows = torch.tensor([r for it in items["train"] for r in it["rows"]])
    pair_sent = torch.tensor([sid[it["sentence"]] for it in items["train"] for _ in it["rows"]])
    pair_y = torch.tensor([float(y) for it in items["train"] for y in it["y"]])
    sent_video = {sid[it["sentence"]]: it["video"] for it in items["train"]}
    train_sent = torch.tensor(sorted(sent_video))
    train_rows = torch.tensor([r for v in train_videos for r in rows_of[v]])
    log(f"학습 쌍 {len(pair_y)}개 (정답 비율 {pair_y.mean():.2f}) + 같은 수의 다른 영상 문장 쌍")

    def zero_shot(key):
        f = torch.nn.functional.normalize(feats[key], dim=-1)
        return lambda it: (f[it["rows"]] @ text[it["sentence"]]).tolist()

    def evaluate(score_fn, its):
        raw, sm, hit, per_item = [], [], 0, []
        for it in its:
            ps = score_fn(it)
            a = auc(ps, it["y"])
            if a is None:
                per_item.append(None)
                continue
            raw.append(a)
            sm.append(auc(uni5(ps), it["y"]))
            hit += it["y"][max(range(len(ps)), key=lambda k: ps[k])]
            per_item.append(round(a, 4))
        return {"items": len(raw), "auc_raw": round(sum(raw) / len(raw), 4), "auc_smooth": round(sum(sm) / len(sm), 4),
                "peak_hit": hit, "per_item": per_item}

    result = {"info": info, "timing": json.loads((root / "timing.json").read_text()),
              "counts": {k: len(v) for k, v in items.items()}, "train_videos": len(train_videos),
              "test_item_video": [it["video"] for it in items["test"]], "variants": {}, "ours_scores": {}}
    for name, key in (("학습 없는 SigLIP (가운데 한 장)", "sig_center"), ("학습 없는 SigLIP (4장 평균)", "sig_mean4")):
        fn = zero_shot(key)
        result["variants"][name] = {"test": evaluate(fn, items["test"]), "ours": evaluate(fn, items["ours"])}
        result["ours_scores"][name] = {it["id"]: [round(p, 5) for p in fn(it)] for it in items["ours"]}
        log(f"{name}: 평가 AUC {result['variants'][name]['test']['auc_raw']:.3f}, 우리 {result['variants'][name]['ours']['auc_raw']:.3f}")

    for name, (keys, use_text) in VARIANTS.items():
        V = torch.cat([feats[k] for k in keys], -1)
        mean, std = V[train_rows].mean(0), V[train_rows].std(0).clamp_min(1e-6)
        V = ((V - mean) / std).to(device)
        logits_sum = {split: [torch.zeros(len(it["rows"])) for it in items[split]] for split in ("test", "ours")}
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            g = torch.Generator().manual_seed(seed)
            model = Head(V.shape[1], T.shape[1], use_text).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
            steps = EPOCHS * math.ceil(2 * len(pair_y) / BATCH)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
            for _ in range(EPOCHS):
                # 다른 영상의 문장을 붙인 쌍을 새로 뽑는다
                neg_rows = train_rows[torch.randint(len(train_rows), (len(pair_y),), generator=g)]
                neg_sent = train_sent[torch.randint(len(train_sent), (len(pair_y),), generator=g)]
                same = torch.tensor([sent_video[int(s)] == str(vids[int(r)]) for r, s in zip(neg_rows, neg_sent)])
                rows_all = torch.cat([pair_rows, neg_rows[~same]])
                sent_all = torch.cat([pair_sent, neg_sent[~same]])
                y_all = torch.cat([pair_y, torch.zeros(int((~same).sum()))])
                order = torch.randperm(len(y_all), generator=g)
                model.train()
                for i in range(0, len(order), BATCH):
                    b = order[i:i + BATCH]
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(V[rows_all[b].to(device)], T[sent_all[b].to(device)]), y_all[b].to(device))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()

            def score(it):
                with torch.no_grad():
                    q = text[it["sentence"]].to(device).expand(len(it["rows"]), -1)
                    return model(V[torch.tensor(it["rows"]).to(device)], q).cpu()

            seed_eval = {}
            for split in ("test", "ours"):
                outs = [score(it) for it in items[split]]
                for acc, o in zip(logits_sum[split], outs):
                    acc += o
                cache = {id(it): o.tolist() for it, o in zip(items[split], outs)}
                seed_eval[split] = evaluate(lambda it: cache[id(it)], items[split])["auc_raw"]
            per_seed.append(seed_eval)
        avg = {split: {id(it): (acc / len(SEEDS)).tolist() for it, acc in zip(items[split], logits_sum[split])}
               for split in ("test", "ours")}
        result["variants"][name] = {split: evaluate(lambda it: avg[split][id(it)], items[split]) for split in ("test", "ours")}
        result["variants"][name]["per_seed"] = per_seed
        result["ours_scores"][name] = {it["id"]: [round(1 / (1 + math.exp(-x)), 5) for x in avg["ours"][id(it)]]
                                       for it in items["ours"]}
        r = result["variants"][name]
        log(f"{name}: 평가 AUC {r['test']['auc_raw']:.3f} (씨앗별 {[round(s['test'], 3) for s in per_seed]}), "
            f"우리 {r['ours']['auc_raw']:.3f}, 5칸 평균 뒤 {r['test']['auc_smooth']:.3f} / {r['ours']['auc_smooth']:.3f}")
    (root / "lev_probe_result.json").write_text(json.dumps(result, ensure_ascii=False) + "\n")
    log("저장: lev_probe_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "features", "train"])
    ap.add_argument("--root", default="/content/lp")
    ap.add_argument("--n-train", type=int, default=MAX_TRAIN)
    ap.add_argument("--n-test", type=int, default=MAX_TEST)
    args = ap.parse_args()
    root = Path(args.root)
    spec = json.loads((root / "lev_probe_input.json").read_text())
    if args.stage == "prepare":
        prepare(root, spec)
    elif args.stage == "features":
        features(root, args.n_train, args.n_test)
    else:
        train(root, spec)


if __name__ == "__main__":
    main()
