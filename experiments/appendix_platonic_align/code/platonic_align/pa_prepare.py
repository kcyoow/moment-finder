"""platonic_align 준비 (이 Mac, CPU, 돈 없음). GPU, Modal, LLM 호출이 없다.

단계
  positions : 영상마다 위치의 가운데 시각 → data/positions.json (TVSum 6,316곳, 마인크래프트 3,600곳. 다른 실험의 것을 그대로)
  queries   : 글 인코더에 넣을 문장 → data/queries.json (TVSum 제목, highlight_auto (나) 기준의 설명, 마인크래프트 손 기준의 설명)
  frames    : 영상마다 초당 7.5장을 뽑아(짧은 변 256 → 가운데 224×224) tar 하나로 → data/packs/<영상>.tar, data/frames_summary.json
  dryrun    : 올릴 크기, 클립 수, 시간과 비용 어림 → data/dryrun.json

실행: uv run --offline --no-project --with numpy --with pillow --with imageio-ffmpeg python -I experiments/platonic_align/pa_prepare.py <단계>

다른 실험 폴더(highlight_auto, highlight_h2o, frame_limit)는 경로로 읽기만 한다. 거기에 파일을 쓰지 않는다.
TVSum의 평가자 점수는 읽지 않는다.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HA = ROOT / "experiments" / "highlight_auto"
HH = ROOT / "experiments" / "highlight_h2o"
FL = ROOT / "experiments" / "frame_limit"
FPS = 7.5                 # LeVJEPA 모델 카드: 16프레임, 약 7.5fps(약 2초)
CLIP = 16
DINO_OFFSETS = (-0.75, -0.25, 0.25, 0.75)   # 판정 모델(H2O)이 본 4장과 같은 간격
# 비용 어림에 쓰는 값(2026-11-12에 modal.com/pricing에서 본 값. 실행 전에 다시 확인한다)
GPU_PER_S, CPU_PER_CORE_S, MEM_PER_GIB_S = 0.000542, 0.0000131, 0.00000222   # L40S
CORES, MEM_GIB = 8, 32


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def tv_videos():
    items = read(HA / "data" / "tvsum_items.json")["videos"]
    vdir = HA / "data" / "tvsum_src" / "x" / "unzip" / "ydata-tvsum50-video" / "video"
    out = {}
    for vid, it in items.items():
        f = vdir / f"{vid}.mp4"
        if not f.exists():
            raise SystemExit(f"TVSum 영상이 없다: {f}")
        out[vid] = {"file": f, "centers": [round(p["center"], 4) for p in it["positions"]]}
    return out


def mc_videos():
    meta = read(FL / "data" / "mc_items.json")["videos"]
    full = read(HH / "data" / "items.json")["full"]
    out = {}
    for it in full:
        out.setdefault(it["video"], {"file": FL / meta[it["video"]]["file"], "centers": []})["centers"].append(it["t"])
    for v, e in out.items():
        if not e["file"].exists():
            raise SystemExit(f"마인크래프트 영상이 없다: {e['file']}")
        if e["centers"] != sorted(e["centers"]):
            raise SystemExit(f"{v}: 위치가 시간 순서가 아니다")
    return out


def stage_positions(args):
    tv, mc = tv_videos(), mc_videos()
    out = {"what": "위치의 가운데 시각(초). 프레임 번호 = round(시각 × 7.5), 0부터. LeVJEPA 클립 = 가운데 번호 − 8 … + 7 (16장), DINOv2 = 가운데 ± 0.25, ± 0.75초의 4장",
           "fps": FPS, "clip_frames": CLIP, "dino_offsets": list(DINO_OFFSETS),
           "tvsum": {v: e["centers"] for v, e in tv.items()}, "minecraft": {v: e["centers"] for v, e in mc.items()}}
    write(HERE / "data" / "positions.json", out)
    print("TVSum", len(tv), "개", sum(len(e["centers"]) for e in tv.values()), "곳 / 마인크래프트", len(mc), "개", sum(len(e["centers"]) for e in mc.values()), "곳")


# ---------------------------------------------------------------------  queries
def parts_of(q):
    """질문 하나를 글 인코더에 넣을 문장들로. noul: 예·아니오 설명 둘. score: 단계 설명 모두(낮은 단계부터)."""
    c = q["criteria"]
    if q["type"] == "noul":
        return [("true", c["true"]), ("false", c["false"])]
    return [(f"level{i}", t) for i, t in enumerate(c)]


def stage_queries(args):
    sys.path.insert(0, str(HH))
    import hh_prepare as HP   # noqa: E402  (읽기만: 손 기준의 문장)
    crit = read(HA / "writer" / "criteria.json")["videos"]
    wi = read(HA / "writer" / "writer_input.json")
    rows = []
    for x in wi["tvsum"]:
        vid = x["id"]
        rows.append({"dataset": "tvsum", "video": vid, "arm": "title", "question": "title", "type": "title", "part": "text", "text": x["title"]})
        for key, q in crit[vid]["questions"].items():
            for part, text in parts_of(q):
                rows.append({"dataset": "tvsum", "video": vid, "arm": "writer", "question": key, "type": q["type"], "part": part, "text": text})
    for key, q in crit["minecraft"]["questions"].items():
        for part, text in parts_of(q):
            rows.append({"dataset": "minecraft", "video": None, "arm": "writer", "question": key, "type": q["type"], "part": part, "text": text})
    hand = {"action": {"type": "score", "criteria": list(HP.ACTION)}, "progress": {"type": "score", "criteria": list(HP.PROGRESS)}}
    for key, q in hand.items():
        for part, text in parts_of(q):
            rows.append({"dataset": "minecraft", "video": None, "arm": "hand", "question": key, "type": q["type"], "part": part, "text": text})
    texts = sorted({r["text"] for r in rows})
    at = {t: i for i, t in enumerate(texts)}
    for r in rows:
        r["text_index"] = at[r.pop("text")]
    out = {"what": "글 인코더에 넣을 문장. rows의 text_index가 texts의 번호다. 판정 모델(H2O)에 이미 물었던 기준((나), 손 기준)의 설명 문장과 TVSum 제목이다. 사람 점수는 없다",
           "score_rule": "noul: cos(예 설명) − cos(아니오 설명). score: Σ (단계 번호 − 가운데) × cos(단계 설명). title: cos(제목)",
           "texts": texts, "rows": rows}
    write(HERE / "data" / "queries.json", out)
    from collections import Counter
    print("문장", len(texts), "개(겹침 제거), 줄", len(rows), Counter((r["dataset"], r["arm"]) for r in rows))


# ---------------------------------------------------------------------  frames
def _extract(job):
    import imageio_ffmpeg
    name, src, dest = job
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        vf = f"fps={FPS},scale=w='if(gt(iw,ih),-2,256)':h='if(gt(iw,ih),256,-2)',crop=224:224"
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf", vf, "-q:v", "3", str(Path(td) / "%06d.jpg")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        files = sorted(Path(td).glob("*.jpg"))
        if p.returncode != 0 or not files:
            raise RuntimeError(f"ffmpeg 실패 {name}: {p.stderr[:300]}")
        tmp = Path(str(dest) + ".tmp")
        with tarfile.open(tmp, "w") as tar:
            for f in files:
                tar.add(f, arcname=f.name)
        tmp.rename(dest)
        return name, len(files), dest.stat().st_size, hashlib.sha256(dest.read_bytes()).hexdigest()


def stage_frames(args):
    import concurrent.futures as cf
    pos = read(HERE / "data" / "positions.json")
    vids = {**{v: ("tvsum", e) for v, e in tv_videos().items()}, **{v: ("minecraft", e) for v, e in mc_videos().items()}}
    packs = HERE / "data" / "packs"
    packs.mkdir(parents=True, exist_ok=True)
    jobs = [(v, e["file"], packs / f"{v}.tar") for v, (_, e) in vids.items() if not (packs / f"{v}.tar").exists()]
    print("뽑을 영상", len(jobs), "/", len(vids), flush=True)
    done = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, n, size, digest in ex.map(_extract, jobs):
            done[name] = (n, size, digest)
            print(f"  {name}: {n}장 {size / 1e6:.1f}MB", flush=True)
    summ = {"what": "영상마다 초당 7.5장(짧은 변 256 → 가운데 224×224, JPEG -q:v 3). 파일 이름은 000001.jpg부터, 번호 n의 시각은 (n − 1) ÷ 7.5초",
            "fps": FPS, "packs": {}}
    bad = []
    for v, (ds, e) in vids.items():
        f = packs / f"{v}.tar"
        with tarfile.open(f) as tar:
            n = len(tar.getnames())
        last = max(pos[ds][v])
        need = round((last + max(DINO_OFFSETS)) * FPS)
        if need > n + CLIP:      # 끝의 몇 장이 모자라면 마지막 장으로 채운다. 그 이상 모자라면 멈춘다
            bad.append((v, n, need))
        summ["packs"][v] = {"dataset": ds, "frames": n, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                            "positions": len(pos[ds][v]), "last_center": last}
    summ["total_frames"] = sum(p["frames"] for p in summ["packs"].values())
    summ["total_bytes"] = sum(p["bytes"] for p in summ["packs"].values())
    write(HERE / "data" / "frames_summary.json", summ)
    print("모두", summ["total_frames"], "장", round(summ["total_bytes"] / 1e6, 1), "MB. 프레임이 모자란 영상:", bad if bad else "없음")
    if bad:
        raise SystemExit(1)


# ---------------------------------------------------------------------  dryrun
def stage_dryrun(args):
    pos = read(HERE / "data" / "positions.json")
    q = read(HERE / "data" / "queries.json")
    fs = read(HERE / "data" / "frames_summary.json")
    n_pos = {ds: sum(len(v) for v in pos[ds].values()) for ds in ("tvsum", "minecraft")}
    per_s = GPU_PER_S + CORES * CPU_PER_CORE_S + MEM_GIB * MEM_PER_GIB_S
    plan = {"smoke": (300, 600), "train_text": (3600, 7200), "embed_all": (1500, 2400)}     # (빠른 경우, 느린 경우) 초
    caps = {"smoke": 900, "train_text": 8100, "embed_all": 3000}                          # 함수 제한 시간
    out = {"what": "platonic_align 미리 세기. GPU는 Modal L40S 하나, CPU 8코어, 메모리 32GiB로 셌다",
           "positions": n_pos, "clips_levjepa": sum(n_pos.values()), "frames_uploaded": fs["total_frames"], "upload_bytes": fs["total_bytes"],
           "query_texts": len(q["texts"]),
           "price_per_second": round(per_s, 6), "price_per_hour": round(per_s * 3600, 2),
           "functions": {k: {"fast_seconds": a, "slow_seconds": b, "timeout_seconds": caps[k], "fast_usd": round(a * per_s, 2),
                             "slow_usd": round(b * per_s, 2), "cap_usd": round(caps[k] * per_s, 2)} for k, (a, b) in plan.items()}}
    out["total"] = {"fast_usd": round(sum(f["fast_usd"] for f in out["functions"].values()), 2),
                    "slow_usd": round(sum(f["slow_usd"] for f in out["functions"].values()), 2),
                    "cap_usd": round(sum(f["cap_usd"] for f in out["functions"].values()), 2)}
    out["note"] = ("학습 시간은 논문 저장소 README의 '약 1 GPU시간'(GPU 종류는 적혀 있지 않다)에서 잡은 어림이다. smoke 함수가 실제 속도를 잰 뒤 다시 센다. "
                   "자료 받기(fetch_data)는 CPU만 쓰고 몇 센트다")
    write(HERE / "data" / "dryrun.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["positions", "queries", "frames", "dryrun"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"positions": stage_positions, "queries": stage_queries, "frames": stage_frames, "dryrun": stage_dryrun}[args.stage](args)


if __name__ == "__main__":
    main()
