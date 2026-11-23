"""moment-finder 웹사이트의 서버 (이 Mac에서 돈다. 127.0.0.1에만 연다).

영상을 올리면 영상마다 작업 하나가 따로 돈다(동시에 여러 개):
  영상 확인 → 중간 LLM(Gemini 4 Hydrogen)이 요청만 보고 질문을 쓴다 → 프레임 뽑기 → Modal에 올리기 → 판정 모델(H2O)이 1초마다 답한다
  → 합쳐서 후보를 고른다 → Gemini가 후보의 프레임을 보고 다시 확인한다(README 22절) → 클립을 자른다
계산은 pipeline/의 코드 그대로다(mf.py, mf_core.py, mf_writer.py, mf_recheck.py, mf_serve_modal.py). 여기는 순서를 잇고 진행 상황을 내놓는다.

실행:
  MODAL_PROFILE=<modal-profile> uv run --no-project --with fastapi --with uvicorn --with python-multipart --with numpy --with pillow --with imageio-ffmpeg --with modal \\
      python pipeline/web/server.py            # http://127.0.0.1:8770
  MF_FAKE=1 을 주면 Gemini와 판정 모델을 부르지 않고 받아 둔 답으로 흉내만 낸다(화면 확인용, 돈이 들지 않는다).
  MF_VERIFY=0 을 주면 다시 확인 단계를 건너뛴다(후보의 프레임을 Gemini에 보내지 않는다).
  MF_VERIFY=h2o 를 주면 판정 모델(H2O)이 스스로 다시 확인한다(후보의 프레임을 Gemini에 보내지 않는다. 확인 질문은 Gemini가 요청만 보고 쓴다).
판정 모델 앱은 미리 올려 둔다: MODAL_PROFILE=<modal-profile> modal deploy pipeline/mf_serve_modal.py
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MODAL_PROFILE", "<modal-profile>")
HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE))
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402

import mf  # noqa: E402
import mf_core as C  # noqa: E402
import mf_recheck as RC  # noqa: E402
import mf_writer as W  # noqa: E402

RUNS, UPLOADS = mf.RUNS, PIPE / "uploads"
FAKE = bool(os.environ.get("MF_FAKE"))
_V = os.environ.get("MF_VERIFY", "1").strip().lower()  # 다시 확인 단계: "1" Gemini가 후보의 프레임을 본다(기본. README 22절), "h2o" 판정 모델이 스스로 다시 본다(프레임이 Gemini로 가지 않는다), "0" 끈다
VERIFY_BY = "off" if _V == "0" else ("h2o" if _V == "h2o" else "gemini")
VERIFY = VERIFY_BY != "off"
SERVE_APP = os.environ.get("MF_SERVE_APP") or (RC.SERVE_APP_H2O if VERIFY_BY == "h2o" else "moment-finder-serve")   # 판정 모델 앱. h2o 방식은 고르기 질문을 받는 앱이 있어야 한다(mf_serve2_modal.py)
PRICE_GPU = mf.PRICE
STEPS = [("probe", "영상 확인", 0.02), ("write", "질문 쓰기 · Gemini", 0.14), ("frames", "프레임 뽑기", 0.08), ("upload", "판정 모델에 올리기", 0.04),
         ("judge", "판정 모델이 보는 중 · H2O", 0.58), ("verify", "다시 확인 · H2O" if VERIFY_BY == "h2o" else "다시 확인 · Gemini", 0.08), ("clips", "클립 만들기", 0.06)]
JOBS, LOCK = {}, threading.Lock()
VIDEOS, BUILD_LOCKS = {}, {}          # 작업 이름 → 올린 영상의 자리 / 그림을 만드는 동안의 잠금
SCALEDOWN = 300                       # mf_serve_modal.py의 SCALEDOWN과 같다
STRIP_H, STRIP_COLS, STRIP_MAX = 90, 20, 300
SLOW = float(os.environ.get("MF_FAKE_SLOW", "1"))      # 흉내 모드의 빠르기(클수록 느리다)
POOL = ThreadPoolExecutor(max_workers=4)
VQ_POOL = ThreadPoolExecutor(max_workers=4)            # 확인 질문 쓰기(MF_VERIFY=h2o). 판정 모델이 영상을 보는 동안 함께 돈다
SERVER = {"state": "unknown", "since": None, "ready_seconds": None, "last_used": None, "error": None}
app = FastAPI(title="moment-finder")


def now():
    return time.time()


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ---------------------------------------------------------------------  작업
def new_job(name, title, request):
    job = {"id": name, "title": title, "request": request, "status": "queued", "created": now(), "started": None, "finished": None, "error": None,
           "step": -1, "steps": [{"key": k, "label": lab, "state": "todo", "detail": ""} for k, lab, _ in STEPS], "progress": 0.0,
           "judge": {"done": 0, "total": 0, "errors": 0}, "costs": {}, "live": True}
    with LOCK:
        JOBS[name] = job
    return job


def set_step(job, key, detail="", frac=0.0):
    idx = next(i for i, (k, _, _) in enumerate(STEPS) if k == key)
    with LOCK:
        if job["step"] != idx:                       # 단계가 바뀌는 때를 적어 둔다(단계마다 걸린 시간)
            t = now()
            for s in job["steps"][:idx]:
                if s.get("started") and not s.get("ended"):
                    s["ended"] = t
            job["steps"][idx]["started"] = t
        job["step"] = idx
        for i, s in enumerate(job["steps"]):
            s["state"] = "done" if i < idx else ("doing" if i == idx else "todo")
        job["steps"][idx]["detail"] = detail
        job["progress"] = round(sum(w for _, _, w in STEPS[:idx]) + STEPS[idx][2] * min(max(frac, 0.0), 1.0), 4)


def judge_remote(job, name):
    """Modal의 판정 모델 앱에 올리고, 진행 상황을 받아 적고, 답을 돌려준다."""
    import modal
    run = RUNS / name
    set_step(job, "upload", "프레임 묶음을 올리는 중")
    vol = modal.Volume.from_name("mf-pipeline", create_if_missing=True)
    with vol.batch_upload(force=True) as b:
        b.put_file(str(run / "pack.tar"), f"/{name}/pack.tar")
        b.put_file(str(run / "requests.json"), f"/{name}/requests.json")
    Judge = modal.Cls.from_name(SERVE_APP, "Judge")
    set_step(job, "judge", "판정 모델 서버를 깨우는 중 (처음에는 3~5분)")
    if SERVER["state"] not in ("warm",):
        SERVER.update(state="warming", since=now())
    for ev in Judge().run.remote_gen(name):
        SERVER.update(state="warm", last_used=now(), ready_seconds=ev.get("ready_seconds", SERVER["ready_seconds"]))
        if ev.get("stage") == "done":
            return ev
        with LOCK:
            job["judge"] = {"done": ev["done"], "total": ev["total"], "errors": ev["errors"]}
        set_step(job, "judge", f"{ev['done']:,} / {ev['total']:,}초", ev["done"] / max(1, ev["total"]))
    raise RuntimeError("판정 모델이 답을 끝까지 돌려주지 않았다")


def judge_fake(job, name):
    """화면 확인용: 받아 둔 로봇 영상의 답을 천천히 흘려보낸다."""
    src = mf.read(RUNS / "gem_robot_red" / "results.json")      # Gemini의 질문(ra_ 벌)으로 실제 받은 답. 흉내 모드의 기준(gem_robot_success)과 같은 질문이다
    pos = mf.read(RUNS / name / "positions.json")
    set_step(job, "upload", "(흉내)")
    time.sleep(0.8 * SLOW)
    set_step(job, "judge", "판정 모델 서버를 깨우는 중 (흉내)")
    with LOCK:
        job["judge"] = {"done": 0, "total": len(pos), "errors": 0}
    time.sleep(3.0 * SLOW)
    n = len(pos)
    for i in list(range(0, n, max(1, n // 80))) + [n]:
        with LOCK:
            job["judge"] = {"done": i, "total": n, "errors": 0}
        set_step(job, "judge", f"{i:,} / {n:,}초", i / n)
        time.sleep(0.13 * SLOW)
    pick = lambda a: {(k[3:] if k.startswith("ra_") else k): v for k, v in a.items() if k.startswith("ra_") or k == "worth"}
    recs = [{"k": q["k"], "answers": pick(src["records"][(q["k"] + 80) % len(src["records"])]["answers"])} for q in pos]
    return {"records": recs, "meta": {"name": name, "requests": n, "answered": n, "errors": 0, "wall_seconds": 5.0, "rate_per_second": 10.0, "fake": True}}


def run_job(job, video):
    name = job["id"]
    run = RUNS / name
    try:
        with LOCK:
            job.update(status="running", started=now())
        set_step(job, "probe", "길이와 화면 크기를 읽는 중")
        info = mf.probe(video)
        with LOCK:
            job["info"] = {k: info[k] for k in ("duration", "width", "height", "fps")}
        facts = [f"길이 {mmss(info['duration'])}", f"화면 {info['width']}×{info['height']}"]
        set_step(job, "write", "요청을 판정 모델에 물을 질문으로 바꾸는 중 (영상은 보여 주지 않는다)")
        if FAKE:
            time.sleep(3.0 * SLOW)
            crit = mf.read(PIPE / "writer_runs" / "gem_robot_success" / "criteria.json")
            crit = {**crit, "request": job["request"], "viewer_request": "the moments where the robot puts a candy into the cup"}
        else:
            crit = W.write(name, job["request"], facts, search=True, with_viewer_request=True, out=run / "writer", quiet=True)
        mf.write(run / "writer.json", crit)
        with LOCK:
            job["costs"]["gemini_usd"] = crit.get("usage", {}).get("usd_estimate")
            job["writer"] = writer_view(crit)
        vq = VQ_POOL.submit(RC.write_question, name, job["request"], facts, run / "verify_question", "HIGH", FAKE) if VERIFY_BY == "h2o" else None
        set_step(job, "frames", "1초마다 4장씩 뽑는 중")
        mf.stage_prepare(argparse.Namespace(video=str(video), name=name, criteria=str(run / "writer.json"), title=None, request=crit["viewer_request"], game=None,
                                            force=True, max_questions=6))
        with LOCK:
            job["positions"] = len(mf.read(run / "positions.json"))
            job["strip"] = strip_meta(name)
        t0 = now()
        out = judge_fake(job, name) if FAKE else judge_remote(job, name)
        mf.write(run / "results.json", {"meta": out["meta"], "records": out["records"]})
        with LOCK:
            job["costs"]["h2o_seconds"] = round(now() - t0, 1)
            job["costs"]["h2o_usd"] = round((now() - t0) * PRICE_GPU, 3)
        set_step(job, "verify", "점수를 합쳐 후보를 고르는 중")
        mf.stage_clips(argparse.Namespace(name=name, budget=None, max_clips=None, grow_q=None, combine=None, no_vague=False), quiet=True)
        if VERIFY:
            try:                                  # 확인이 안 돼도 작업은 끝까지 간다(확인 전의 클립을 보여 준다)
                if FAKE:
                    time.sleep(2.0 * SLOW)
                on = lambda text, frac=0.0: set_step(job, "verify", text, frac)
                if VERIFY_BY == "h2o":                # 판정 모델이 스스로 다시 본다(experiments/decision_api/README.md 10·12절). 확인 질문은 위에서 쓰기 시작했다
                    v = RC.recheck_h2o(name, job["request"], question=vq.result(), fake=FAKE, on_stage=on, app=SERVE_APP)
                else:
                    v = RC.recheck(name, job["request"], fake=FAKE, on_stage=on)
                with LOCK:
                    job["costs"]["verify_usd"] = (v.get("usage") or {}).get("usd_estimate")
            except BaseException as e:
                traceback.print_exc()
                RC.mark_skipped(name, f"{type(e).__name__}: {e}"[:300])
        else:
            RC.mark_skipped(name, "꺼 둠 (MF_VERIFY=0)")
        set_step(job, "clips", "클립을 자르는 중")
        mf.stage_render(argparse.Namespace(name=name, top=0))
        build_strip(name)
        cj = mf.read(run / "clips.json")
        with LOCK:
            job.update(status="done", finished=now(), progress=1.0, duration=info["duration"], share=cj["share"], total_length=cj["total_length"], **clip_summary(cj))
            for s in job["steps"]:
                s["state"] = "done"
                if s.get("started") and not s.get("ended"):
                    s["ended"] = job["finished"]
        mf.write(run / "job.json", job)
    except BaseException as e:      # mf.py는 오류를 SystemExit로 낸다
        traceback.print_exc()
        with LOCK:
            job.update(status="error", finished=now(), error=f"{type(e).__name__}: {e}"[:600])
            for s in job["steps"]:
                if s["state"] == "doing":
                    s["state"] = "error"


# ---------------------------------------------------------------------  결과를 화면용으로
def clip_summary(cj):
    """작업 목록에 쓰는 요약. 다시 확인을 했으면 통과한 클립만 센다."""
    hit = [c for c in cj["clips"] if c.get("match", True)]
    v = cj.get("verify")
    return {"clips": len(hit), "spans": [[c["start"], c["end"]] for c in hit],
            "verify": {k: v.get(k) for k in ("status", "candidates", "matched", "reason")} if v else None}


def short_q(text):
    m = re.search(r"earliest first\)\.\s*(.*)$", text, re.S)
    t = (m.group(1) if m else text).strip()
    return re.sub(r"^Do these frames show the following:\s*", "", t)


def writer_view(crit):
    """중간 LLM이 쓴 것 가운데 화면에 보일 것."""
    return {"model": crit.get("writer"), "understanding": crit.get("understanding"), "kind": crit.get("kind"), "limits": crit.get("limits"),
            "looked_up": crit.get("looked_up"), "search_queries": crit.get("search_queries") or [],
            "sources": [s.get("domain") or s.get("title") for s in (crit.get("search_sources") or [])][:8], "viewer_request": crit.get("viewer_request"),
            "questions": [{"key": k, "type": q["type"], "text": short_q(q["instructions"]), "criteria": q.get("criteria")} for k, q in (crit.get("questions") or {}).items()]}


SIZES = {}


def run_info(name):
    """run.json. 받아 둔 답으로 만든 실행에는 화면 크기가 없어서 영상에서 읽어 채운다."""
    info = mf.read(RUNS / name / "run.json")
    if "width" not in info:
        if name not in SIZES:
            p = mf.probe(info["video"])
            SIZES[name] = {k: p[k] for k in ("width", "height", "fps")}
        info.update(SIZES[name])
    return info


def strip_meta(name):
    """필름 띠(영상 전체에서 고르게 뽑은 작은 그림들을 한 장에 붙인 것)의 모양."""
    d = RUNS / name
    if not (d / "run.json").exists():
        return None
    info = run_info(name)
    n = int(min(STRIP_MAX, max(1, info["duration"] // 1)))
    w = max(2, int(round(STRIP_H * info["width"] / info["height"] / 2)) * 2)
    return {"url": f"/media/{name}/strip", "n": n, "cols": STRIP_COLS, "w": w, "h": STRIP_H, "duration": info["duration"]}


def build_strip(name):
    """필름 띠를 만든다. 프레임 묶음(pack.tar)이 있으면 거기서 꺼내고, 없으면 ffmpeg로 영상을 한 번 훑는다."""
    import io
    import subprocess
    import tarfile
    from PIL import Image
    d = RUNS / name
    out = d / "strip_v1.jpg"
    with LOCK:
        lock = BUILD_LOCKS.setdefault(name, threading.Lock())
    with lock:
        if out.exists():
            return out
        m, info = strip_meta(name), run_info(name)
        n, cols, w, h, dur = m["n"], m["cols"], m["w"], m["h"], m["duration"]
        rows = -(-n // cols)
        if (d / "pack.tar").exists():
            pos = mf.read(d / "positions.json")
            sheet = Image.new("RGB", (cols * w, rows * h), (16, 17, 19))
            with tarfile.open(d / "pack.tar") as tar:
                members = {x.name: x for x in tar.getmembers()}
                for i in range(n):
                    t = (i + 0.5) * dur / n
                    q = pos[min(len(pos) - 1, int(t))]
                    j = min(range(len(q["seconds"])), key=lambda j: abs(q["seconds"][j] - t))
                    im = Image.open(io.BytesIO(tar.extractfile(members[q["frames"][j]]).read())).convert("RGB").resize((w, h), Image.BILINEAR)
                    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
            tmp = d / "strip_tmp.jpg"
            sheet.save(tmp, quality=80)
            tmp.replace(out)
        else:
            p = subprocess.run([mf.ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", info["video"], "-an", "-vf", f"fps={n}/{dur:.3f},scale={w}:{h},tile={cols}x{rows}",
                                "-frames:v", "1", "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True)
            if not p.stdout:
                raise HTTPException(500, f"필름 띠를 만들지 못했다: {p.stderr[-200:].decode(errors='replace')}")
            out.write_bytes(p.stdout)
        return out


def past_runs():
    out = []
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if d.name in JOBS or not (d / "clips.json").exists() or not (d / "clips").exists() or d.name.startswith("gem_"):
            continue
        info = mf.read(d / "run.json")
        saved = mf.read(d / "job.json") if (d / "job.json").exists() else {}
        cj = mf.read(d / "clips.json")
        out.append({"id": d.name, "title": saved.get("title") or Path(info["video"]).name, "request": saved.get("request") or info.get("viewer_request") or "", "status": "done",
                    "progress": 1.0, "created": saved.get("created") or (d / "clips.json").stat().st_mtime, "live": False, "duration": info["duration"],
                    "share": cj["share"], "total_length": cj["total_length"], **clip_summary(cj),
                    "strip": strip_meta(d.name),
                    "elapsed": round(saved["finished"] - saved["started"], 1) if saved.get("finished") and saved.get("started") else None,
                    "costs": saved.get("costs", {}), "asked_in_english": not saved.get("request")})
    return out


def detail(name):
    d = RUNS / name
    if not (d / "clips.json").exists():
        raise HTTPException(404, "아직 결과가 없다")
    info, cj, sc, qj = run_info(name), mf.read(d / "clips.json"), mf.read(d / "scores.json"), mf.read(d / "questions.json")
    res = mf.read(d / "results.json")
    writer = mf.read(d / "writer.json") if (d / "writer.json").exists() else {}
    recs = {r["k"]: r.get("answers") for r in res["records"] if "error" not in r}
    n = len(sc["t0"])
    centers = [round(0.5 * (a + b), 2) for a, b in zip(sc["t0"], sc["t1"])]
    questions = []
    for key, q in qj["questions"].items():
        vals = np.array([C.answer_value((recs.get(k) or {}).get(key)) for k in range(n)], float)
        top = (len(q["criteria"]) - 1) if q.get("type") == "score" else 1
        unit = np.nan_to_num(vals / top)              # 답이 없는 곳은 0으로 그린다
        questions.append({"key": key, "vague": key == qj.get("vague_key"), "type": q["type"], "text": short_q(q["instructions"]), "criteria": q.get("criteria"),
                          "mean": round(float(unit.mean()), 3), "peak": round(float(unit.max()), 3), "peak_t": centers[int(unit.argmax())],
                          "yes_seconds": int((unit >= 0.5).sum()), "series": [round(float(x), 3) for x in unit]})
    clips = []
    for c in cj["clips"]:
        f = next((p.name for p in (d / "clips").glob(f"{c['rank']:02d}_*.mp4")), None)
        clips.append({**c, "url": f"/media/{name}/clips/{f}" if f else None, "poster": f"/media/{name}/poster/{c['rank']}"})
    meta = res.get("meta", {})
    job = JOBS.get(name) or (mf.read(d / "job.json") if (d / "job.json").exists() else {})
    wv = writer_view(writer) if writer else {}
    wv["model"] = wv.get("model") or info.get("criteria_writer")
    wv["usage"] = writer.get("usage")
    return {"id": name, "title": job.get("title") or Path(info["video"]).name, "request": job.get("request") or writer.get("request") or info.get("viewer_request"),
            "asked_in_english": not (job.get("request") or writer.get("request")),
            "viewer_request": info.get("viewer_request"), "duration": info["duration"], "size": [info.get("width"), info.get("height")], "fps": info.get("fps"),
            "source": f"/media/{name}/source", "strip": strip_meta(name), "t": centers, "final": sc["final"], "smoothed": sc["smoothed"], "clips": clips,
            "share": cj["share"], "total_length": cj["total_length"], "positions": cj["positions"], "params": cj.get("params"), "combine": cj.get("combine"),
            "verify": cj.get("verify"),
            "questions": questions, "writer": wv,
            "judge": {k: meta.get(k) for k in ("requests", "answered", "errors", "rate_per_second", "wall_seconds", "ready_seconds", "fake")},
            "steps": job.get("steps"), "elapsed": round(job["finished"] - job["started"], 1) if job.get("finished") and job.get("started") else None,
            "costs": job.get("costs", {})}


# ---------------------------------------------------------------------  길
@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})


def server_view():
    """판정 모델 서버가 지금 켜져 있을지. 마지막으로 쓴 뒤 SCALEDOWN초가 지나면 Modal이 끈다."""
    busy = any(j["status"] == "running" and 3 <= j["step"] <= 4 for j in JOBS.values())      # 올리기와 판정 단계만 GPU를 쓴다
    if SERVER["state"] == "warm" and not busy and SERVER["last_used"] and now() - SERVER["last_used"] > SCALEDOWN + 20:
        SERVER.update(state="cold", since=now())
    v = dict(SERVER, now=now(), scaledown=SCALEDOWN, price_per_second=PRICE_GPU)
    if v["state"] == "warm" and not busy and v["last_used"]:
        v["off_in"] = max(0, round(SCALEDOWN - (now() - v["last_used"])))
    return v


@app.get("/api/jobs")
def jobs():
    with LOCK:
        live = [dict(j, elapsed=round((j["finished"] or now()) - (j["started"] or j["created"]), 1)) for j in JOBS.values()]
    return {"jobs": sorted(live + past_runs(), key=lambda j: -j["created"]), "server": server_view(), "fake": FAKE, "now": now()}


@app.get("/api/jobs/{name}")
def job_detail(name: str):
    if not re.fullmatch(r"[\w\-]+", name):
        raise HTTPException(400)
    return detail(name)


@app.post("/api/jobs")
async def create(file: UploadFile = File(...), request: str = Form("")):
    request = request.strip()
    if not request:
        raise HTTPException(400, "어떤 장면을 찾을지 한 줄 적어 주세요")
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"):
        raise HTTPException(400, "영상 파일만 올릴 수 있습니다")
    with LOCK:
        name = f"web_{time.strftime('%m%d_%H%M%S')}_{len(JOBS) + 1}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{name}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job = new_job(name, file.filename or dest.name, request)
    VIDEOS[name] = dest
    POOL.submit(run_job, job, dest)
    return job


@app.post("/api/server/warm")
def warm():
    if FAKE:
        SERVER.update(state="warm", since=now(), ready_seconds=0, last_used=now())
        return SERVER
    if SERVER["state"] == "warming":
        return SERVER
    SERVER.update(state="warming", since=now(), error=None)

    def go():
        try:
            import modal
            r = modal.Cls.from_name(SERVE_APP, "Judge")().ping.remote()
            SERVER.update(state="warm", ready_seconds=r.get("ready_seconds"), last_used=now())
        except Exception as e:
            SERVER.update(state="error", error=f"{type(e).__name__}: {e}"[:300])
    threading.Thread(target=go, daemon=True).start()
    return SERVER


def ranged(path, request: Request, media_type):
    size = path.stat().st_size
    m = re.match(r"bytes=(\d*)-(\d*)$", request.headers.get("range") or "")
    if not m:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    a, b = m.group(1), m.group(2)
    start = int(a) if a else max(0, size - int(b or 0))
    end = min(int(b), size - 1) if (a and b) else min(size - 1, start + (8 << 20) - 1)
    if start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    return Response(data, status_code=206, media_type=media_type,
                    headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(len(data))})


@app.get("/media/{name}/clips/{file}")
def media_clip(name: str, file: str, request: Request):
    p = RUNS / name / "clips" / file
    if not re.fullmatch(r"[\w\-]+", name) or not re.fullmatch(r"[\w\-]+\.mp4", file) or not p.exists():
        raise HTTPException(404)
    return ranged(p, request, "video/mp4")


@app.get("/media/{name}/source")
def media_source(name: str, request: Request):
    if not re.fullmatch(r"[\w\-]+", name) or not (RUNS / name / "run.json").exists():
        raise HTTPException(404)
    p = Path(mf.read(RUNS / name / "run.json")["video"])
    if not p.exists():
        raise HTTPException(404)
    return ranged(p, request, "video/mp4" if p.suffix.lower() in (".mp4", ".m4v", ".mov") else "video/webm")


@app.get("/media/{name}/strip")
def media_strip(name: str):
    if not re.fullmatch(r"[\w\-]+", name) or not (RUNS / name / "run.json").exists():
        raise HTTPException(404)
    return FileResponse(build_strip(name), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/media/{name}/thumb")
def media_thumb(name: str):
    """작업 목록에 쓰는 작은 그림 하나(영상의 10% 자리)."""
    if not re.fullmatch(r"[\w\-]+", name):
        raise HTTPException(404)
    d = RUNS / name
    out = d / "thumb.jpg"
    if not out.exists():
        video = VIDEOS.get(name) or (Path(mf.read(d / "run.json")["video"]) if (d / "run.json").exists() else None)
        if video is None or not Path(video).exists():
            raise HTTPException(404)
        data = mf.grab(video, min(30.0, 0.1 * mf.probe(video)["duration"]), width=240)
        if not data:
            raise HTTPException(404)
        d.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/media/{name}/poster/{rank}")
def media_poster(name: str, rank: int):
    d = RUNS / name
    if not re.fullmatch(r"[\w\-]+", name) or not (d / "clips.json").exists():
        raise HTTPException(404)
    out = d / "posters" / f"{rank:02d}.jpg"
    if not out.exists():
        c = next((c for c in mf.read(d / "clips.json")["clips"] if c["rank"] == rank), None)
        if c is None:
            raise HTTPException(404)
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(mf.grab(Path(mf.read(d / "run.json")["video"]), c["peak_t"], width=480))
    return FileResponse(out, media_type="image/jpeg")


@app.exception_handler(Exception)
async def oops(request: Request, exc: Exception):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"[:400]}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MF_PORT", "8770")), log_level="warning")
