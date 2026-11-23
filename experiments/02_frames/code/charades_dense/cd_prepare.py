"""charades_dense 준비 (이 Mac, 가벼운 일만). Pillow와 imageio-ffmpeg가 필요하다:
  uv run --offline --no-project --with pillow --with imageio-ffmpeg python -I experiments/charades_dense/cd_prepare.py <단계>

단계
  items   : 위치·질문·정답을 data/items.json으로(prompt_study의 load_items 그대로 + 질문마다 정답 구간)
  frames  : 영상마다 0.125초 격자의 프레임(0.25 … K+0.75초)을 뽑아 data/packs/<영상>.tar로 묶는다.
            뽑는 법은 앞선 실험(frame_limit loc_frames.py: `ffmpeg -ss t`로 t 이후 첫 프레임 → 긴 변 768px, LANCZOS, JPEG 90)과 같다.
            여기서는 영상을 한 번 풀어 프레임 시각을 읽고(showinfo), 시각 t마다 "t 이상인 첫 프레임"을 골라 한 번에 뽑는다.
            앞선 실험의 프레임 파일과 겹치는 시각(0.25초의 배수)은 바이트가 같은지 확인한다
  q3      : Q3(경로 그림): traj_mark의 401개 구간을 data/items.json의 "q3"에 넣고 data/packs/q3_<영상>.tar를 만든다(numpy도 필요)
  samples : 모양마다 그림 몇 장을 samples/에(눈으로 확인용)
  dryrun  : 서버 없이 모든 요청 본문을 만들고, 모양마다 요청 수·질문 읽기 수·토큰 어림·그림 크기와 시간·비용 어림을 찍는다
"""
import concurrent.futures as cf
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FL = EXP / "frame_limit"
sys.path.insert(0, str(HERE))
import cd_layouts as L  # noqa: E402

DATA, PACKS, SAMPLES = HERE / "data", HERE / "data" / "packs", HERE / "samples"
SETS = {"dev": ("sta2", "sta2_videos"), "hold": ("sta", "sta_videos")}


# ---------------------------------------------------------------- items
def make_items():
    sys.path.insert(0, str(EXP / "prompt_study"))
    from prompt_run import load_items
    windows = {}
    for name in ("tl_sta2_items.json", "tl_sta_items.json"):
        for q in json.loads((FL / "data" / name).read_text())["items"]:
            windows[q["id"]] = q["windows"]
    out, videos = [], {}
    for it in load_items():
        for qid, y in it["gt"].items():   # 정답이 구간과 맞는지 확인
            assert y == int(any(s <= it["t"] <= e for s, e in windows[qid])), (it["id"], qid)
        out.append({"id": it["id"], "set": it["set"], "video": it["video"], "t": it["t"], "questions": it["questions"], "gt": it["gt"]})
        videos.setdefault(it["video"], {"set": it["set"], "positions": 0})["positions"] += 1
    for name, (tag, vdir) in SETS.items():
        frames = json.loads((FL / "data" / f"{tag}_frames.json").read_text())
        for v, info in videos.items():
            if info["set"] == name:
                info.update(file=str((FL / "data" / vdir / (v.split("_", 1)[1] + ".mp4")).relative_to(EXP.parent)),
                            duration=frames[v]["duration"], last_center=frames[v]["centers"][-1])
                assert info["positions"] == len(frames[v]["centers"])
    meta = {"_note": "위치·질문·정답은 prompt_study/prompt_run.py load_items 그대로(dev = sta2 30개 영상, hold = sta 12개 영상). "
                     "windows는 TimeLens-Bench의 정답 구간(초). Charades·TimeLens는 연구용이며 다시 배포하지 않는다.",
            "videos": videos, "windows": {q: windows[q] for it in out for q in it["questions"]}, "items": out}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "items.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    for s in ("dev", "hold"):
        its = [i for i in out if i["set"] == s]
        print(s, "영상", len({i["video"] for i in its}), "위치", len(its), "질문", len({q for i in its for q in i["questions"]}),
              "위치×질문", sum(len(i["questions"]) for i in its), "정답 안", sum(sum(i["gt"].values()) for i in its))


def load_items():
    return json.loads((DATA / "items.json").read_text())


# ---------------------------------------------------------------- frames
def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fit(raw):
    """frame_limit extract_frames.fit과 같다: 긴 변 768px(짧은 변이 280px 밑이면 짧은 변 280px), LANCZOS, JPEG 90."""
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = 768 / max(im.size)
    if min(im.size) * scale < 280:
        scale = 280 / min(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), im.size


def extract(job):
    video, info = job
    ff, src = ffmpeg(), EXP.parent / info["file"]
    p = subprocess.run([ff, "-hide_banner", "-nostats", "-i", str(src), "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    found = re.findall(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-\d.e]+)", p.stderr)
    num, den = map(int, re.search(r"config in time_base:\s*(\d+)/(\d+)", p.stderr).groups())
    pts = [float(t) for _, _, t in found]
    ipts = [int(x) for _, x, _ in found]
    assert [int(n) for n, _, _ in found] == list(range(len(pts))), video
    ticks = [m * L.TICK for m in range(2, round((info["last_center"] + 0.75) / L.TICK) + 1)]
    pick = {}
    for t in ticks:
        # ffmpeg -ss처럼 목표 시각(마이크로초)을 영상의 시간 단위로 바꿔 반올림(0.5는 올림)해 견준다.
        # RBQ9Y 10.25초는 10.249991초의 프레임이, 4WVTC 20.25초(337000.5 → 337001)는 그다음 프레임이 뽑혀야 앞선 프레임과 같다
        target = math.floor(Fraction(round(t * 1_000_000)) * den / (num * 1_000_000) + Fraction(1, 2))
        j = next((i for i, x in enumerate(ipts) if x >= target), None)
        assert j is not None, f"{video}: {t}초 뒤에 프레임이 없다"
        pick[t] = j
    need = sorted(set(pick.values()))

    def tree(xs):   # ffmpeg 식 해석기는 긴 덧셈 줄을 못 받아(깊이 제한) 균형 잡힌 괄호로 묶는다
        return f"eq(n,{xs[0]})" if len(xs) == 1 else "(" + tree(xs[:len(xs) // 2]) + "+" + tree(xs[len(xs) // 2:]) + ")"
    with tempfile.TemporaryDirectory(dir=os.environ.get("CD_TMP")) as tmp:
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-vf", f"select='{tree(need)}'",
                        "-fps_mode", "passthrough", "-q:v", "2", f"{tmp}/%05d.jpg"], check=True)
        got = sorted(os.listdir(tmp))
        assert len(got) == len(need), (video, len(got), len(need))
        raw = {n: open(f"{tmp}/{f}", "rb").read() for n, f in zip(need, got)}
    fitted = {n: fit(b) for n, b in raw.items()}
    old_dir = json.loads((FL / "data" / f"{SETS[info['set']][0]}_frames.json").read_text())[video]["frames"]
    same = checked = 0
    index = {"video": video, "source_frames": len(pts), "fps": round(len(pts) / pts[-1], 3) if pts[-1] else None, "ticks": {}}
    PACKS.mkdir(parents=True, exist_ok=True)
    dest = PACKS / f"{video}.tar"
    with tarfile.open(str(dest) + ".part", "w") as tar:
        for t in ticks:
            b, size = fitted[pick[t]]
            if f"{t:.2f}" in old_dir and abs(t * 4 - round(t * 4)) < 1e-9:
                checked += 1
                same += (FL / old_dir[f"{t:.2f}"]).read_bytes() == b
            ti = tarfile.TarInfo(L.fname(t))
            ti.size = len(b)
            tar.addfile(ti, io.BytesIO(b))
            index["ticks"][L.fname(t)] = {"t": t, "n": pick[t], "pts": round(pts[pick[t]], 4), "size": list(size)}
        blob = json.dumps(index).encode()
        ti = tarfile.TarInfo("_index.json")
        ti.size = len(blob)
        tar.addfile(ti, io.BytesIO(blob))
    os.replace(str(dest) + ".part", dest)
    # 한 위치의 8장(그 1초) 가운데 서로 다른 원본 프레임 수
    distinct = []
    for k in range(1, int(info["last_center"]) + 1):
        distinct.append(len({pick[round(k + o, 3)] for o in L.GRID_OFFSETS}))
    return {"video": video, "frames": len(ticks), "bytes": dest.stat().st_size, "checked_vs_old": checked, "identical": same,
            "fps": index["fps"], "size": list(fitted[need[0]][1]), "distinct_per_second_min": min(distinct),
            "distinct_per_second_mean": round(sum(distinct) / len(distinct), 2)}


def make_frames(workers):
    meta = load_items()
    jobs = sorted(meta["videos"].items())
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(extract, jobs))
    summary = {"_note": "영상마다 뽑은 프레임 수, tar 크기, 앞선 실험 프레임과 바이트가 같은 수, 원본 fps, 그 1초 8장 가운데 서로 다른 프레임 수",
               "videos": out, "total_frames": sum(o["frames"] for o in out), "total_bytes": sum(o["bytes"] for o in out),
               "checked_vs_old": sum(o["checked_vs_old"] for o in out), "identical": sum(o["identical"] for o in out)}
    (DATA / "frames_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    for o in out:
        print(f"{o['video']:12s} fps {o['fps']:6.2f} 프레임 {o['frames']:4d} {o['bytes'] / 1e6:5.1f}MB 크기 {o['size']} "
              f"앞선 프레임과 같음 {o['identical']}/{o['checked_vs_old']}  1초 8장 중 다른 프레임 최소 {o['distinct_per_second_min']} 평균 {o['distinct_per_second_mean']}")
    print(f"합계 프레임 {summary['total_frames']}장, {summary['total_bytes'] / 1e6:.0f}MB, 앞선 프레임과 같음 {summary['identical']}/{summary['checked_vs_old']}, {time.time() - t0:.0f}초")


# ---------------------------------------------------------------- 공통: 묶음 파일에서 프레임 읽기
_PACK = {}


def get(video, name):
    if video not in _PACK:
        _PACK[video] = {}
        for f in (PACKS / f"{video}.tar", PACKS / f"q3_{video}.tar"):
            if f.exists():
                with tarfile.open(f) as tar:
                    _PACK[video].update({m.name: tar.extractfile(m).read() for m in tar.getmembers()})
    return _PACK[video][name]


# ---------------------------------------------------------------- Q3 (경로 그림)
TM = EXP / "traj_mark"


def make_q3():
    """traj_mark의 401개 구간(hold 12개 영상)을 Q3 항목으로 만들고, 영상마다 q3_<영상>.tar에 그림을 묶는다.
    (a)(b)(c)의 그림은 traj_mark의 파일을 그대로 넣는다(다시 만들지 않는다).
    (d)의 4장은 traj_mark와 같은 프레임 줄기(frame_limit motion_lib.timeline: 8fps 이하, 긴 변 512, ffmpeg fps 필터)에서
    구간 [t0, t1]을 고르게 덮는 4장(구간 길이의 1/8, 3/8, 5/8, 7/8 지점에 가장 가까운 프레임)을 traj_make와 같은 JPEG 92로 저장한다.
    줄기가 같은지는 끝 프레임을 같은 방법으로 저장해 traj_mark의 plain 그림과 바이트가 같은지로 확인한다."""
    import numpy as np  # noqa: F401  (motion_lib가 쓴다)
    from PIL import Image
    sys.path.insert(0, str(FL / "scripts" / "motion"))
    import motion_lib
    meta = load_items()
    plain = json.loads((TM / "index.json").read_text())["items"]
    comp = json.loads((TM / "index_comp.json").read_text())["items"]
    assert len(plain) == len(comp) == 401 and all((a["video"], a["frame"]) == (b["video"], b["frame"]) for a, b in zip(plain, comp))
    qs = {}
    for it in meta["items"]:
        if it["set"] == "hold":
            qs.setdefault(it["video"], it["questions"])
    segs, report = [], {}
    for video in sorted({p["video"] for p in plain}):
        frames, ts, fps = motion_lib.timeline(video)
        mine = [(a, b) for a, b in zip(plain, comp) if a["video"] == video]
        same = 0
        blobs = {}
        for a, b in mine:
            hi = a["frame"]
            lo = int(round(a["t0"] * fps))
            assert abs(ts[lo] - a["t0"]) < 0.002 and abs(ts[hi] - a["t"]) < 0.002, (video, hi)
            d = hi - lo
            dn = [min(hi, max(lo, lo + math.floor(d * (2 * j + 1) / 8 + 0.5))) for j in range(4)]
            for n in dn + [hi]:
                if n not in blobs:
                    buf = io.BytesIO()
                    Image.fromarray(frames[n]).save(buf, "JPEG", quality=92)
                    blobs[n] = buf.getvalue()
            same += blobs[hi] == (TM / a["plain"]).read_bytes()
            m = (a["t0"] + a["t"]) / 2
            w = {q: meta["windows"][q] for q in qs[video]}
            segs.append({"id": f"q3/{video}/{hi}", "set": "hold", "video": video, "frame": hi, "t0": a["t0"], "t1": a["t"], "t": round(m, 4),
                         "paths_path": a["paths"], "paths_comp": b["paths"], "d_frames": dn, "d_seconds": [round(float(ts[n]), 3) for n in dn],
                         "questions": qs[video], "gt": {q: int(any(s <= m <= e for s, e in w[q])) for q in w},
                         "gt_end": {q: int(any(s <= a["t"] <= e for s, e in w[q])) for q in w}})
        dest = PACKS / f"q3_{video}.tar"
        with tarfile.open(str(dest) + ".part", "w") as tar:
            def add(name, data):
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
            for a, b in mine:
                add(f"q3/{a['frame']:03d}_plain.jpg", (TM / a["plain"]).read_bytes())
                add(f"q3/{a['frame']:03d}_path.jpg", (TM / a["path"]).read_bytes())
                add(f"q3/{a['frame']:03d}_comp.jpg", (TM / b["comp"]).read_bytes())
            used = sorted({n for s in segs if s["video"] == video for n in s["d_frames"]})
            for n in used:
                add(f"q3/tl{n:03d}.jpg", blobs[n])
        os.replace(str(dest) + ".part", dest)
        report[video] = {"segments": len(mine), "fps": round(float(fps), 3), "end_frame_identical": same, "pack_mb": round(dest.stat().st_size / 1e6, 1)}
        print(video, report[video], flush=True)
    meta["q3"] = segs
    meta["q3_note"] = ("traj_mark의 401개 구간. 정답(gt)은 구간 가운데 시각 m = (t0+t1)/2가 정답 구간 안인가(prompt_eval과 같은 '시각 하나' 규칙). "
                       "gt_end는 끝 프레임 시각 t1로 매긴 참고값. d_frames는 (d)의 4장(traj_mark 프레임 줄기의 번호)")
    (DATA / "items.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    pairs = sum(len(s["questions"]) for s in segs)
    print(f"구간 {len(segs)}, 구간×질문 {pairs}, 정답 안 {sum(sum(s['gt'].values()) for s in segs)}, "
          f"경로 있음 (b) {sum(s['paths_path'] > 0 for s in segs)} (c) {sum(s['paths_comp'] > 0 for s in segs)} 둘 다 {sum(s['paths_path'] > 0 and s['paths_comp'] > 0 for s in segs)}, "
          f"끝 프레임이 traj_mark plain과 같음 {sum(r['end_frame_identical'] for r in report.values())}/{len(segs)}")
    (DATA / "q3_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


def all_units(meta, layouts=None):
    out = []
    for layout in layouts or L.LAYOUTS:
        src = meta["q3"] if L.LAYOUTS[layout]["kind"] in L.Q3_KINDS else meta["items"]
        out += L.units(src, layout)
    return out


def build(unit):
    """본문을 만들고 크기·토큰 어림만 돌려준다(본문 자체는 버린다)."""
    from PIL import Image
    pics = L.pictures(unit, get)
    sizes = [Image.open(io.BytesIO(p)).size for p in pics]
    payload, keys = L.body(unit, pics)
    inp, proc, img = L.estimate(payload, sizes)
    return {"id": unit["id"], "layout": unit["layout"], "set": unit["set"], "video": unit["video"], "reads": len(payload["questions"]),
            "sizes": sizes, "pixels": max(w * h for w, h in sizes), "pixels_sum": sum(w * h for w, h in sizes),
            "resized": any(L.smart_resize(h, w) != L.smart_resize(h, w, max_pixels=10 ** 9) for w, h in sizes),
            "bytes": sum(len(p) for p in pics), "body_bytes": len(json.dumps(payload)), "input_tokens": inp, "processed_tokens": proc,
            "image_tokens": img, "seconds_of_video": len(unit["items"]), "positions": len(unit["items"])}


# ---------------------------------------------------------------- samples
def make_samples():
    """모양마다 같은 장면의 그림: dev 영상 둘(가로 하나, 세로 하나)의 정답 구간 가운데 위치."""
    meta = load_items()
    items = meta["items"]
    SAMPLES.mkdir(exist_ok=True)
    made = []
    for video in ("sta2_3QL7J", "sta2_I52A6"):
        its = [i for i in items if i["video"] == video]
        qid = next(q for i in its for q, y in i["gt"].items() if y)
        pos = [i for i in its if i["gt"][qid]]
        mid = pos[len(pos) // 2]
        for layout in [x for x in L.LAYOUTS if L.LAYOUTS[x]["kind"] not in L.Q3_KINDS]:
            us = [u for u in L.units(its, layout) if any(i["t"] == mid["t"] for i in u["items"])]
            made.append(_sample(us[0], f"{video}_t{mid['t']:g}_{layout}", its[0]["questions"][qid], mid["t"]))
    # Q3: hold 영상 둘의 정답 구간 안 구간(경로가 그려진 것). traj_mark README의 예시 JVOM3 19~20초(옷 벗기 시작)와 RX7TC
    for video, frame in (("sta_JVOM3", 160), ("sta_RX7TC", 16)):
        seg = next(s for s in meta["q3"] if s["video"] == video and s["frame"] == frame)
        qid = next((q for q, y in seg["gt"].items() if y), next(iter(seg["gt"])))
        for layout in [x for x in L.LAYOUTS if L.LAYOUTS[x]["kind"] in L.Q3_KINDS]:
            made.append(_sample(L.units([seg], layout)[0], f"{video}_f{frame}_{layout}", seg["questions"][qid], seg["t"]))
    (SAMPLES / "samples.json").write_text(json.dumps(made, ensure_ascii=False, indent=1) + "\n")
    print("\n".join(sorted(p.name for p in SAMPLES.glob("*.jpg"))))


def _sample(u, stem, question, t):
    pics = L.pictures(u, get)
    for n, p in enumerate(pics):
        (SAMPLES / f"{stem}{'' if len(pics) == 1 else f'_{n + 1}'}.jpg").write_bytes(p)
    from PIL import Image
    payload, _ = L.body(u, pics)
    shown = json.loads(json.dumps(payload))
    if "frames" in shown["state"]:
        shown["state"]["frames"] = [{**x, "frame": "<그림>"} for x in shown["state"]["frames"]]
    for k in ("grid", "sheet", "frame"):
        if k in shown["state"]:
            shown["state"][k] = "<그림>"
    return {"file": stem, "video": u["video"], "t": t, "question": question, "layout": u["layout"], "files": len(pics),
            "sizes": [list(Image.open(io.BytesIO(p)).size) for p in pics], "state": shown["state"],
            "first_questions": dict(list(shown["questions"].items())[:3]), "questions_in_request": len(shown["questions"])}


# ---------------------------------------------------------------- dry run
# 시간 어림에 쓰는 처리 속도(2026-11-12 mc_dense, Modal RTX PRO 6000, 동시 16, 질문 4개):
#   4장 따로 640×360 15.9건/초, 8장 줄인 그림 16.8건/초, 8장 원본 그림(1,881,152화소) 6.4건/초.
# H2O shim은 질문마다 그림을 함께 vLLM에 따로 보내고(vLLM의 그림 처리 캐시는 꺼져 있다), 그림 처리는 화소에 비례한다고 보고
# "질문 읽기 × 요청의 화소"를 초당 처리량으로 나눈다. 4장 따로: 15.9 × 4 × 921,600 = 58.6M, 줄인 그림: 16.8 × 4 × 945,312 = 63.5M,
# 원본 그림: 6.4 × 4 × 1,881,152 = 48.2M 화소·읽기/초. 큰 그림(150만 화소 넘음)은 원본 그림 값을 다시 1.4로 나눈 34M을 쓴다(더 큰 그림의 몫, 보수적으로).
RATE_SMALL, RATE_BIG, BIG = 55e6, 34e6, 1_500_000
REQ_CAP = 16.0       # 그림이 작아도 초당 요청은 이보다 많지 않다고 본다(mc_dense 최고 16.8건/초, h2o_startup 16.5건/초)
STARTUP = 200        # 함수 시작부터 서버 준비까지(초). h2o_startup: 그래프 512 + 컴파일 캐시로 153초 + 컨테이너·볼륨 몫
PER_SEC = 0.000842 + 8 * 0.0000131 + 32 * 0.00000222   # RTX PRO 6000 + CPU 8코어 + 메모리 32GiB, $/초 (modal.com/pricing, 2026-11-14 확인)


def calibrate():
    """받아 둔 기준 답(prompt_study h2o__ts.json: 같은 프레임, 변형 10개를 한 요청에)의 usage.input_tokens와 어림을 견준다."""
    sys.path.insert(0, str(EXP / "prompt_study"))
    from variants import VARIANTS
    recs = {r["id"]: r for r in json.loads((EXP / "prompt_study" / "results" / "h2o__ts.json").read_text())["records"]}
    meta = load_items()
    ratios = []
    for it in meta["items"][::7]:
        u = {"id": "x", "layout": "sep4", "set": it["set"], "video": it["video"], "items": [it]}
        payload, _ = L.body(u, [b"x"] * 4)
        qs = {}
        for i, (vname, v) in enumerate(VARIANTS.items()):
            if v["layout"] != "ts":
                continue
            for j, (qid, s) in enumerate(it["questions"].items()):
                q = {"type": "noul", "instructions": v["q"].format(s=s)}
                if v.get("criteria"):
                    q["criteria"] = v["criteria"]
                qs[f"v{i}q{j}"] = q
        for q in qs.values():   # 어림 함수는 noul의 기본 선택지 글만 센다. 변형의 criteria 글은 물음 길이에 넣어 센다
            q["instructions"] += " " + " ".join((q.pop("criteria", None) or {}).values())
        payload["questions"] = qs
        inp, _, _ = L.estimate(payload, [frame_size(it["video"])] * 4)
        ratios.append(recs[it["id"]]["tokens"] / inp)
    ratios.sort()
    return ratios[len(ratios) // 2], ratios[0], ratios[-1], len(ratios)


def frame_size(video):
    with tarfile.open(PACKS / f"{video}.tar") as tar:
        return tuple(next(iter(json.loads(tar.extractfile("_index.json").read())["ticks"].values()))["size"])


def dryrun(workers):
    meta = load_items()
    items = meta["items"]
    med, lo, hi, n = calibrate()
    print(f"토큰 어림 확인: 받아 둔 기준 답의 usage ÷ 어림 = 중앙값 {med:.3f} (최소 {lo:.3f}, 최대 {hi:.3f}, 위치 {n}곳)")
    units = all_units(meta)
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(build, units, chunksize=8))
    print(f"본문 {len(res)}개를 {time.time() - t0:.0f}초에 만듦 (프로세스 {workers}개)\n")
    rows, total_secs = [], 0.0
    hdr = f"{'모양':14s}{'요청':>6s}{'읽기':>7s}{'요청당 토큰':>10s}{'처리 토큰 합':>13s}{'영상 1초당 토큰':>14s}{'그림 최대 화소':>14s}{'줄어듦':>6s}{'본문 MB':>8s}{'어림 초':>8s}"
    print(hdr)
    for layout in L.LAYOUTS:
        rs = [r for r in res if r["layout"] == layout]
        reads = sum(r["reads"] for r in rs)
        inp = sum(r["input_tokens"] for r in rs) * med
        proc = sum(r["processed_tokens"] for r in rs) * med
        secs_video = sum(r["seconds_of_video"] for r in rs)
        px_reads = sum(r["reads"] * r["pixels_sum"] for r in rs)
        big = max(r["pixels"] for r in rs) > BIG
        est = max(px_reads / (RATE_BIG if big else RATE_SMALL), len(rs) / REQ_CAP)
        total_secs += est
        row = {"layout": layout, "requests": len(rs), "reads": reads, "input_tokens_per_request": round(inp / len(rs)),
               "input_tokens_total": round(inp), "processed_tokens_total": round(proc), "tokens_per_video_second": round(inp / secs_video, 1),
               "max_pixels": max(r["pixels"] for r in rs), "over_cap": sum(r["pixels"] > L.CAP for r in rs), "resized": sum(r["resized"] for r in rs),
               "body_mb_total": round(sum(r["body_bytes"] for r in rs) / 1e6, 1), "max_body_mb": round(max(r["body_bytes"] for r in rs) / 1e6, 2),
               "pixel_reads": px_reads, "est_seconds": round(est), "by_set": {s: {"requests": sum(r["set"] == s for r in rs),
                                                                                   "reads": sum(r["reads"] for r in rs if r["set"] == s)} for s in ("dev", "hold")},
               "image_sizes": sorted({tuple(sz) for r in rs for sz in r["sizes"]})[:6]}
        rows.append(row)
        print(f"{layout:14s}{len(rs):6d}{reads:7d}{row['input_tokens_per_request']:10,d}{row['processed_tokens_total']:13,d}{row['tokens_per_video_second']:14,.0f}"
              f"{row['max_pixels']:14,d}{row['resized']:6d}{row['body_mb_total']:8.0f}{est:8.0f}")
    out = {"calibration": {"median": med, "min": lo, "max": hi, "n": n}, "layouts": rows, "rates": {"small": RATE_SMALL, "big": RATE_BIG, "big_over": BIG},
           "startup_seconds": STARTUP, "per_second_usd": PER_SEC, "estimated_serving_seconds": round(total_secs)}
    (DATA / "dryrun.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"\n서버 일 어림 {total_secs:.0f}초 + 준비 {STARTUP}초 = {total_secs + STARTUP:.0f}초 × ${PER_SEC:.8f}/초 = ${(total_secs + STARTUP) * PER_SEC:.2f}")
    print("그림 크기 예:", {r["layout"]: r["image_sizes"][:3] for r in rows})


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    workers = int(os.environ.get("CD_WORKERS", "4"))
    if step == "items":
        make_items()
    elif step == "frames":
        make_frames(workers)
    elif step == "q3":
        make_q3()
    elif step == "samples":
        make_samples()
    elif step == "dryrun":
        dryrun(workers)
    else:
        raise SystemExit(__doc__)
