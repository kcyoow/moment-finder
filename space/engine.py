"""moment-finder Space의 흐름. 모델 없이 돌아가는 부분을 모두 여기에 둔다(app.py가 모델을 올리고 GPU 함수 둘을 GPU에 넣어 준다).

흐름(2026-11-23, experiments/final_qwen_h2o/README.md):
  요청 → 중간 LLM(Qwen)이 계획을 쓴다(1초마다 물을 질문 + 다시 확인용 정의) → 사용자에게 "이렇게 찾을까요?" 되묻는다
       → 맞다고 하면 판정 모델(H2O)이 1초마다 프레임 4장을 보고 답한다 → 후보 6개까지
       → 판정 모델이 후보마다 프레임 8장을 다시 본다(예/아니오 + 고르기, 둘 다 0.5 이상이어야 통과) → 클립

GPU 호출은 요청 하나에 한 번만 한다. ZeroGPU는 요청 머리글의 토큰으로 GPU를 잡는데 그 토큰이 요청이 들어온 뒤 한두 분이면 만료되기 때문이다
(2026-11-18: 요청 하나에서 질문 쓰기 146초 뒤의 판정 호출이 "Expired ZeroGPU proxy token"으로 거절됨). 그래서 일을 "단계"로 잘라
job에 기억해 두고, 화면의 타이머(또는 API의 /step)가 단계를 하나씩 부른다. 끊기면 같은 job으로 이어 한다.
"""
import base64
import hashlib
import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import numpy as np
import pandas as pd

import mf_core as C
import plan as P
import writer as WR

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"
ABOUT = "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first)."
Q_VAGUE = "Would this moment fit `viewer_request`?"
REQ_GENERIC = "the most important moments of this video"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
MAX_SIDE = 640
MAX_SECONDS = 600                                         # 영상 길이 한도
SHOW_CLIPS = 5
N_CAND, KEEP_P = 6, 0.5                                   # pipeline/mf_recheck.py와 같다
LEAD_A = "These 4 frames are consecutive frames, 0.5 seconds apart, from one second of one candidate segment of a longer video."
LEAD_B = "These 4 frames are spread evenly across one candidate segment of a longer video, in time order."
WRITER_MAX_NEW = 1400

# 한 번 읽는 데 드는 초(2026-11-18 Space에서 0.153~0.161). 호출마다 잰 값으로 바꾼다
RATE = {"per_read": 0.16, "first": 15.0, "measured": False}
# tok_s는 프롬프트 읽기까지 넣은 초당 토큰, overhead는 부른 쪽에서 본 시간과 함수 안 시간의 차이(GPU 잡기·모델 옮기기)
WRATE = {"tok_s": 12.0, "first": 15.0, "overhead": 25.0, "measured": False}
GPU = {"judge": None, "write": None, "writer_name": "Qwen/Qwen3.8-27B-FP8"}     # app.py가 채운다
JOBS, PLAN_CACHE, JUDGE_CACHE = {}, {}, {}
WORK = ("write", "judge", "recheck")


def judge_duration(batch, *args, **kwargs):
    n = len(batch["items"]) * len(batch["questions"])
    return int(min(115, RATE["first"] + n * RATE["per_read"] * 1.3 + 5))


def writer_duration(messages, max_new, *args, **kwargs):
    return int(min(115, max(60, WRATE["overhead"] + WRATE["first"] + max_new / WRATE["tok_s"])))


# ---------------------------------------------------------------- 영상 (pipeline/mf.py와 같은 격자)
FF = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    err = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    v = re.search(r"Video: [^\n]*?, (\d{2,5})x(\d{2,5})[^\n]*?, ([\d.]+) fps", err)
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not v or not d:
        raise gr.Error("영상을 읽지 못했습니다.")
    return {"width": int(v.group(1)), "height": int(v.group(2)), "fps": float(v.group(3)),
            "duration": round(int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)), 3)}


def positions_for(duration, step=1.0):
    out, k = [], 0
    while (k + 1) * step <= duration + 1e-9:
        c = (k + 0.5) * step
        out.append({"k": k, "t0": round(k * step, 3), "t1": round((k + 1) * step, 3), "center": round(c, 3),
                    "times": [round(min(max(c + o, 0.0), max(duration - 0.1, 0.0)), 3) for o in OFFSETS]})
        k += 1
    return out


def grid_frames(video, info):
    """0.25초부터 0.5초마다 그 시각에 가장 가까운 프레임 한 장(긴 변 640)."""
    half = 0.5 / info["fps"]
    sel = f"gt(floor((t+{half:.6f}-0.25)*2)\\,floor((prev_t+{half:.6f}-0.25)*2))"
    vf = f"select='{sel}',scale=w='if(gte(iw,ih),min({MAX_SIDE},iw),-2)':h='if(gte(iw,ih),-2,min({MAX_SIDE},ih))'"
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf, "-fps_mode", "vfr", "-q:v", "2",
                            str(Path(td) / "%06d.jpg")], capture_output=True, text=True)
        got = sorted(Path(td).glob("*.jpg"))
        if p.returncode != 0 or not got:
            raise gr.Error(f"프레임을 뽑지 못했습니다: {p.stderr[-300:]}")
        return [base64.b64encode(f.read_bytes()).decode() for f in got]


def cut(video, a, b, out):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-i", str(video), "-t", f"{b - a:.2f}",
                    "-vf", "scale=-2:'min(540,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(out)], check=True)


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def find_moments(score, t0, t1, duration, n_max=5, cut_q=0.85, grow_q=0.75, drop=0.08, gap=2, pad=1.0, min_len=3.0):
    """찾기용 클립(2026-11-17, pipeline/README.md 25-10절). 요약용 make_clips(영상의 15%, 앞뒤 평균)와 달리
    1초 점수(0~1, 앞뒤 평균 없음)가 가장 높은 곳부터 봉우리로 잡고, 봉우리 점수에서 drop 안쪽이면서 영상 안 위 25%에 드는 이웃 초까지 넓힌다.
    gap초 이하로 떨어진 구간은 잇고, 앞뒤에 pad초를 붙이고, min_len초보다 짧으면 늘린다(영상 끝·처음에서는 안쪽으로).
    봉우리 점수가 영상 안 위 15%(cut_q)에 들지 못하거나 n_max개가 차면 멈춘다. 길이 합의 한도는 없다."""
    score = np.nan_to_num(np.asarray(score, float), nan=0.0)
    n = len(score)
    cut_v, floor_q = float(np.quantile(score, cut_q)), float(np.quantile(score, grow_q))
    taken = np.zeros(n, bool)
    segs = []
    for pk in np.argsort(-score, kind="stable"):
        if score[pk] < cut_v or len(segs) >= n_max:
            break
        if taken[pk]:
            continue
        floor = max(floor_q, score[pk] - drop)
        a = b = int(pk)
        while a > 0 and not taken[a - 1] and score[a - 1] >= floor:
            a -= 1
        while b < n - 1 and not taken[b + 1] and score[b + 1] >= floor:
            b += 1
        for sg in segs:
            if a - sg["b"] - 1 <= gap and sg["a"] - b - 1 <= gap:
                sg["a"], sg["b"] = min(sg["a"], a), max(sg["b"], b)
                break
        else:
            segs.append({"a": a, "b": b, "pk": int(pk)})
        taken[a:b + 1] = True
    clips = []
    for sg in segs:
        s0, s1 = max(0.0, float(t0[sg["a"]]) - pad), min(float(duration), float(t1[sg["b"]]) + pad)
        if s1 - s0 < min_len:
            if s1 >= duration:
                s0 = max(0.0, float(duration) - min_len)
            elif s0 <= 0:
                s1 = min(float(duration), min_len)
            else:
                need = (min_len - (s1 - s0)) / 2
                s0, s1 = max(0.0, s0 - need), min(float(duration), s1 + need)
        clips.append({"start": round(s0, 2), "end": round(s1, 2), "length": round(s1 - s0, 2), "peak": round(float(score[sg["pk"]]), 4),
                      "peak_t": round(0.5 * float(t0[sg["pk"]] + t1[sg["pk"]]), 2)})
    clips.sort(key=lambda c: -c["peak"])
    for r, c in enumerate(clips, 1):
        c["rank"] = r
    return clips


def frame_times(clip, duration):
    """다시 확인용 8장의 시각(pipeline/mf_verify.py와 같다): a = 가장 확신한 1초의 연속 4장, b = 구간 전체에서 고르게 4장."""
    a = [min(max(clip["peak_t"] + d, 0.0), duration - 0.05) for d in OFFSETS]
    b = [clip["start"] + (i + 0.5) * (clip["end"] - clip["start"]) / 4 for i in range(4)]
    return [round(x, 2) for x in a], [round(min(x, duration - 0.05), 2) for x in b]


# ---------------------------------------------------------------- job: 요청 하나의 일과 그 단계
def _facts(info, game):
    facts = [f"길이 {mmss(info['duration'])}", f"화면 {info['width']}×{info['height']}"]
    if (game or "").strip():
        facts.append(f"영상 설명(사용자가 적음): {game.strip()}")
    return facts


def _video_key(video):
    h = hashlib.sha1()
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _trim_jobs(keep=6):
    for k in list(JOBS)[:-keep]:
        JOBS.pop(k, None)


def job_ask(job_id, video, text, game="", plan_json=""):
    """요청(또는 고쳐 말하기)을 받아 job을 만들거나 고친다. 계획을 써야 하면 stage는 write, 이미 있으면 confirm."""
    text, plan_json = (text or "").strip(), (plan_json or "").strip()
    if not video:
        raise gr.Error("영상을 올려 주세요.")
    if not text and not plan_json:
        raise gr.Error("어떤 장면을 찾을지 적어 주세요.")
    vkey = _video_key(video)
    job = JOBS.get(job_id or "")
    if job and job["vkey"] == vkey and job.get("plan") and not plan_json:
        job["feedback"].append(text)                       # 같은 영상에 계획이 이미 있으면 고쳐 말한 것으로 본다
        job["video"] = video
    else:
        info = probe(video)
        if info["duration"] > MAX_SECONDS:
            raise gr.Error(f"영상이 {info['duration'] / 60:.1f}분입니다. {MAX_SECONDS // 60}분까지 받습니다(하루 GPU 할당량 때문).")
        job = {"id": uuid.uuid4().hex[:12], "video": video, "vkey": vkey, "info": info, "request": text, "game": (game or "").strip(),
               "feedback": [], "plan": None, "log": [f"영상 {mmss(info['duration'])} ({info['width']}×{info['height']}, {info['fps']:.0f}fps)"]}
        JOBS[job["id"]] = job
        _trim_jobs()
    for k in ("cands", "result", "error", "frames", "items", "recs"):
        job.pop(k, None)
    job["paused"] = False
    facts = _facts(job["info"], job["game"])
    if plan_json:
        try:
            job["plan"] = P.parse_plan_json(plan_json)
        except ValueError as e:
            raise gr.Error(str(e))
        job["request"] = job["plan"].get("request") or text
        job["stage"] = "confirm"
        job["log"].append("붙여 넣은 계획을 그대로 씀(중간 LLM을 부르지 않음)")
        return job
    key = (job["request"], tuple(facts), tuple(job["feedback"]))
    if key in PLAN_CACHE:
        job["plan"], job["stage"] = PLAN_CACHE[key], "confirm"
        job["log"].append("같은 요청에 앞서 쓴 계획을 그대로 씀")
        return job
    prev = job["plan"]["raw_reply"] if (job["feedback"] and job.get("plan")) else None
    job["w"] = {"messages": P.build_messages(job["request"], facts, job["feedback"], prev), "text": "", "used": 0, "n": 0, "attempt": 0,
                "calls": [], "key": key, "facts": facts}
    job["stage"] = "write"
    job["log"].append("중간 LLM이 계획을 쓰는 중" + (" (고쳐 말한 것을 반영)" if prev else ""))
    return job


def _step_write(job):
    w = job["w"]
    t = time.time()
    res = GPU["write"](w["messages"], WRITER_MAX_NEW - w["used"], w["text"])
    wall = time.time() - t
    w["used"] += res["new_tokens"]
    w["text"] = res["text"]
    w["n"] += 1
    w["calls"].append({"wall_seconds": round(wall, 1), **{k: res.get(k) for k in ("prompt_tokens", "new_tokens", "stop", "gen_seconds", "seconds",
                                                                                 "first_token_seconds", "decode_tok_s", "room", "peak_gb")}})
    if res.get("decode_tok_s") and res["new_tokens"] > 20:      # 다음 호출의 GPU 시간을 잰 값으로 맞춘다
        WRATE.update(tok_s=max(4.0, res["decode_tok_s"] * 0.85), first=max(3.0, (res.get("first_token_seconds") or 0) + 2),
                     overhead=min(40.0, max(8.0, wall - res["seconds"] + 5)), measured=True)
    if res["stop"] == "time" and w["used"] < WRITER_MAX_NEW and w["n"] < 3:
        job["log"].append(f"계획 쓰는 중: {w['used']}토큰까지 씀(GPU {res['seconds']:.0f}초). 이어서 씁니다")
        return
    data, errs = P.check_plan(w["text"])
    print("[plan-check]", json.dumps({"attempt": w["attempt"] + 1, "errors": errs, "stop": res["stop"], "head": w["text"][:300], "tail": w["text"][-200:]},
                                     ensure_ascii=False), flush=True)
    if errs and w["attempt"] == 0 and res["stop"] == "eos":       # 형식만 한 번 고치게 한다(잘린 답은 고쳐 봐야 소용없다)
        w["messages"] = w["messages"] + [{"role": "assistant", "content": w["text"]}, {"role": "user", "content": WR.FIX_MESSAGE + "\n- ".join(errs)}]
        w.update(text="", used=0, n=0, attempt=1)
        job["log"].append(f"계획의 형식이 맞지 않아 한 번 더 씁니다: {errs[0][:120]}")
        return
    if errs:
        job["stage"], job["error"] = "error", f"중간 LLM의 답에 형식 오류가 남았습니다(멈춘 이유 {res['stop']}, {w['used']}토큰): {errs[:2]}"
        job["log"].append(job["error"])
        return
    plan = P.to_plan(data, job["request"], w["facts"], GPU["writer_name"], job["feedback"], w["text"])
    plan["writer_calls"], plan["format_retries"] = w["calls"], w["attempt"]
    PLAN_CACHE[w["key"]] = plan
    job["plan"], job["stage"] = plan, "confirm"
    c = w["calls"]
    job["log"].append(f"계획을 씀: 질문 {len(plan['questions'])}개({', '.join(plan['questions'])}), 헷갈리는 장면 {len(plan['check']['confusers'])}개. "
                      f"GPU {sum(x['seconds'] for x in c):.0f}초(호출 {len(c)}번) · 글 {c[0]['prompt_tokens']}토큰 → 답 {sum(x['new_tokens'] for x in c)}토큰 · "
                      f"첫 토큰까지 {c[0]['first_token_seconds']}초 · 초당 {c[-1]['decode_tok_s']}토큰")


def job_confirm(job_id, answers_json=""):
    """계획이 맞다고 했다: 프레임을 뽑고 판정할 준비를 한다(GPU 없음). answers_json은 시험용(앞서 받은 판정 답을 그대로 넣는다)."""
    job = JOBS.get(job_id or "")
    if not job or not job.get("plan"):
        raise gr.Error("먼저 요청을 보내 계획을 받아 주세요(Space가 다시 시작됐다면 처음부터 다시 보내야 합니다).")
    plan, info = job["plan"], job["info"]
    dur = info["duration"]
    questions = {"worth": {"type": "noul", "instructions": Q_VAGUE}, **plan["questions"]}
    base_state = {**({"game": job["game"]} if job["game"] else {}), "viewer_request": plan["viewer_request"] or job["request"] or REQ_GENERIC, "about": ABOUT}
    frames = grid_frames(job["video"], info)
    pos = positions_for(dur)
    ng = len(frames)
    gi = lambda t: min(max(int(round((t - 0.25) / 0.5)), 0), ng - 1)
    items = [{"k": p["k"], "frames": [frames[gi(t)] for t in p["times"]], "seconds": [round(0.25 + 0.5 * gi(t), 2) for t in p["times"]]} for p in pos]
    jkey = (job["vkey"], json.dumps(questions, sort_keys=True), json.dumps(base_state, sort_keys=True))
    recs = JUDGE_CACHE.setdefault(jkey, {})
    if (answers_json or "").strip():
        recs.update({int(k): v for k, v in json.loads(answers_json).items()})
    todo = sum(1 for it in items if it["k"] not in recs)
    job.update(questions=questions, base_state=base_state, frames=frames, pos=pos, items=items, recs=recs, stage="judge", judge_calls=[], paused=False)
    for k in ("cands", "result", "error"):
        job.pop(k, None)
    job["log"].append(f"프레임 {ng}장, 위치 {len(items)}곳(1초마다 4장), 질문 {len(questions)}개(막연한 질문 포함). "
                      f"물을 위치 {todo}곳, 어림 GPU {todo * len(questions) * RATE['per_read'] + RATE['first']:.0f}초")
    return job


def _step_judge(job):
    todo = [it for it in job["items"] if it["k"] not in job["recs"]]
    if todo:
        nq = len(job["questions"])
        room = (110 - RATE["first"] - 5) / (RATE["per_read"] * 1.3)        # 한 호출을 115초 안에
        n = max(1, min(len(todo), int(room // nq)))
        t = time.time()
        res = GPU["judge"]({"base_state": job["base_state"], "questions": job["questions"], "items": todo[:n]})
        for r in res["records"]:
            job["recs"][r["k"]] = r["answers"]
        per = res["seconds"] / max(res["reads"], 1)
        RATE.update(per_read=max(per, 0.02), first=max(8.0, min(40.0, time.time() - t - res["seconds"] + 5)), measured=True)
        job["judge_calls"].append({"positions": n, "reads": res["reads"], "gpu_seconds": res["seconds"], "wall_seconds": round(time.time() - t, 1)})
        job["log"].append(f"판정 {len(job['items']) - len(todo) + n}/{len(job['items'])} · 이번 호출 {res['seconds']:.0f}초(대기 포함 {time.time() - t:.0f}초), "
                          f"한 번 읽기 {per:.3f}초, 입력 토큰 중앙값 {res['tokens_median']}")
        if len(todo) > n:
            return
    _make_candidates(job)
    job["stage"] = "recheck" if job["cands"] else "done"
    if not job["cands"]:
        _finish(job)


def _make_candidates(job):
    """받은 답 → 후보 클립. 목록 여럿(찾기 봉우리, 합친 점수, 질문 하나하나, 막연한 질문)에서 돌아가며 N_CAND개(pipeline/mf_recheck.py candidates + 찾기 봉우리)."""
    pos, questions, dur = job["pos"], job["questions"], job["info"]["duration"]
    answers = [job["recs"].get(p["k"]) for p in pos]
    keys = [k for k in questions if k != "worth"]
    levels = {k: len(q["criteria"]) for k, q in questions.items() if q["type"] == "score"}
    comb = C.combine(answers, keys, "worth", levels=levels)
    t0 = np.array([p["t0"] for p in pos]); t1 = np.array([p["t1"] for p in pos])
    parts = []
    w = np.asarray(comb["raw"]["worth"], float)
    parts.append(np.where(np.isnan(w), np.nanmean(w) if not np.all(np.isnan(w)) else 0.0, w))
    if comb.get("criteria") is not None:
        parts.append(np.asarray(comb["criteria"], float))
    match = np.mean(parts, axis=0)                              # 요청에 맞는 정도(0~1): 막연한 질문의 "예" 확률과 기준 질문 값의 평균
    lists = [("찾기 봉우리", find_moments(match, t0, t1, dur, n_max=N_CAND)), ("합친 점수", C.make_clips(comb["final"], t0, t1, dur, budget=0.9))]
    lists += [(k, C.make_clips(comb["rank"][k], t0, t1, dur, budget=0.9)) for k in keys]
    if keys and "vague" in comb:
        lists.append(("막연한 질문", C.make_clips(comb["vague"], t0, t1, dur, budget=0.9)))
    cands = []
    for i, c in enumerate(C.round_robin([l for _, l in lists], N_CAND), 1):
        ta, tb = frame_times(c, dur)
        cands.append({"id": i, "start": c["start"], "end": c["end"], "length": c["length"], "peak_t": c["peak_t"], "from": lists[c["source"]][0],
                      "score": round(float(match[min(int(c["peak_t"]), len(match) - 1)]), 4), "times_a": ta, "times_b": tb})
    job["cands"], job["match"] = cands, [round(float(x), 4) for x in match]
    job["raw"] = {k: [None if np.isnan(x) else round(float(x), 4) for x in v] for k, v in comb["raw"].items()}
    job["log"].append(f"후보 {len(cands)}개: " + ", ".join(f"{mmss(c['start'])}~{mmss(c['end'])}" for c in cands))


def _step_recheck(job):
    plan, frames = job["plan"], job["frames"]
    ng = len(frames)
    gi = lambda t: min(max(int(round((t - 0.25) / 0.5)), 0), ng - 1)
    items = []
    for c in job["cands"]:
        for tag, lead, ts in (("a", LEAD_A, c["times_a"]), ("b", LEAD_B, c["times_b"])):
            items.append({"k": f"{c['id']}{tag}", "frames": [frames[gi(t)] for t in ts], "seconds": [round(0.25 + 0.5 * gi(t), 2) for t in ts],
                          "state": {"about": lead}})
    qs = P.recheck_questions(plan["check"])
    t = time.time()
    res = GPU["judge"]({"base_state": {}, "questions": qs, "items": items})
    ans = {r["k"]: r["answers"] for r in res["records"]}
    tkey = plan["check"]["target"]["key"]
    names = list(qs["choice_fwd"]["criteria"])
    for c in job["cands"]:
        a, b = ans[f"{c['id']}a"], ans[f"{c['id']}b"]
        na, nb = float(a["verify"]["noul"]), float(b["verify"]["noul"])
        opt = {n: float(np.mean([x[k]["probabilities"][n] for x in (a, b) for k in ("choice_fwd", "choice_rev")])) for n in names}
        ca = float(np.mean([a[k]["probabilities"][tkey] for k in ("choice_fwd", "choice_rev")]))
        cb = float(np.mean([b[k]["probabilities"][tkey] for k in ("choice_fwd", "choice_rev")]))
        yn, ch = (na + nb) / 2, (ca + cb) / 2
        c.update(p=round(min(yn, ch), 4), p_noul=round(yn, 4), p_choice=round(ch, 4), noul_consecutive=round(na, 4), noul_spread=round(nb, 4),
                 choice_consecutive=round(ca, 4), choice_spread=round(cb, 4), options={n: round(v, 4) for n, v in opt.items()},
                 seen_as=max(opt, key=opt.get), match=bool(min(yn, ch) >= KEEP_P))
    job["recheck_call"] = {"reads": res["reads"], "gpu_seconds": res["seconds"], "wall_seconds": round(time.time() - t, 1)}
    job["log"].append(f"다시 확인: 후보 {len(job['cands'])}개 × 프레임 4장씩 두 번 × 질문 3개 = {res['reads']}번 읽음(GPU {res['seconds']:.0f}초). "
                      f"통과 {sum(c['match'] for c in job['cands'])}개")
    _finish(job)


def _label(plan, key):
    if key == "none_of_these":
        return "어느 것도 아님"
    for o in [plan["check"]["target"]] + plan["check"]["confusers"]:
        if o["key"] == key:
            return P._ko(o)
    return key


def _finish(job):
    plan, cands = job["plan"], job.get("cands") or []
    cands.sort(key=lambda c: (not c.get("match"), -c.get("p", 0.0), c["start"]))
    outdir = Path(tempfile.mkdtemp())
    clips, rows = [], []
    for r, c in enumerate(cands, 1):
        c["rank"] = r
        if c.get("match") and len(clips) < SHOW_CLIPS:
            f = outdir / f"clip_{len(clips) + 1}.mp4"
            cut(job["video"], c["start"], c["end"], f)
            clips.append({"file": str(f), "label": f"{mmss(c['start'])}~{mmss(c['end'])} · 다시 확인 {c['p']:.2f}", "start": c["start"], "end": c["end"], "p": c["p"]})
        rows.append([r, mmss(c["start"]), mmss(c["end"]), round(c["length"], 1), "통과" if c.get("match") else "빠짐", c.get("p"), c.get("p_noul"),
                     c.get("p_choice"), _label(plan, c.get("seen_as", "")), c["from"]])
    hit = [c for c in cands if c.get("match")]
    if hit:
        summary = (f"찾았습니다. 후보 {len(cands)}개를 다시 확인해 **{len(hit)}개**가 통과했습니다: " + ", ".join(f"{mmss(c['start'])}~{mmss(c['end'])}" for c in hit))
    elif cands:
        summary = (f"찾는 장면이 보이지 않습니다. 후보 {len(cands)}개를 다시 확인했지만 통과한 것이 없습니다"
                   f"(가장 높은 값 {max(c.get('p', 0.0) for c in cands):.2f}, 기준 {KEEP_P}). 아래 표에 후보마다 무엇으로 봤는지가 있습니다.")
    else:
        summary = "후보를 하나도 뽑지 못했습니다."
    pos = job["pos"]
    job["result"] = {"request": job["request"], "game": job["game"], "proposal": P.proposal_text(plan), "summary": summary, "clips": clips,
                     "columns": ["순위", "시작", "끝", "길이(초)", "다시 확인", "값", "예/아니오", "고르기", "무엇으로 봤나", "후보가 나온 목록"], "rows": rows,
                     "curve": {"초": [p["center"] for p in pos], "점수": job.get("match") or []}, "log": list(job["log"]),
                     "details": {"plan": {k: v for k, v in plan.items() if k != "raw_reply"}, "questions": job["questions"], "base_state": job["base_state"],
                                 "judge_calls": job.get("judge_calls"), "recheck_call": job.get("recheck_call"), "candidates": cands,
                                 "keep_p": KEEP_P, "video": {**job["info"]}}}
    job["stage"] = "done"
    job["log"].append(summary.replace("**", ""))
    for k in ("frames", "items"):
        job.pop(k, None)


def job_step(job_id):
    """단계 하나(GPU 호출 한 번). 멈추면 받은 것은 남기고 paused로 둔다(같은 단계부터 이어 한다)."""
    job = JOBS.get(job_id or "")
    if not job or job["stage"] not in WORK or job.get("paused"):
        return job
    try:
        {"write": _step_write, "judge": _step_judge, "recheck": _step_recheck}[job["stage"]](job)
    except Exception as e:  # noqa: BLE001 — 할당량·토큰 만료 등
        job["paused"], job["error"] = True, f"{type(e).__name__}: {str(e)[:300]}"
        job["log"].append(f"멈췄습니다: {job['error']}")
        print("[step-error]", job["stage"], job["error"], flush=True)
    return job


def more(job):
    return bool(job) and job["stage"] in WORK and not job.get("paused")


def status_of(job, full=False):
    if not job:
        return {"stage": "none"}
    out = {"job": job["id"], "stage": job["stage"], "paused": bool(job.get("paused")), "error": job.get("error"), "more": more(job), "log": job["log"][-40:]}
    if job.get("plan"):
        out["plan"] = job["plan"]
    if job["stage"] == "done":
        out["result"] = job["result"]
    if full and job["stage"] in ("recheck", "done") and job.get("recs") is not None:     # 시험용: 받은 판정 답을 부른 쪽에도 남긴다
        out["answers"] = {str(k): v for k, v in job["recs"].items()}
    return out


# ---------------------------------------------------------------- 화면
PROGRESS = {"write": "요청을 읽고 어떻게 찾을지 정리하는 중입니다… (중간 LLM, GPU 1~2분)",
            "judge": "판정 모델이 1초마다 프레임을 보는 중입니다…", "recheck": "판정 모델이 후보를 다시 확인하는 중입니다…"}
HEAD = ("# 🎬 Moment Finder\n영상을 올리고 찾을 장면을 편하게 적으면, **중간 LLM(Qwen3.8-27B)** 이 어떻게 찾을지 정리해 되묻습니다. "
        "맞다고 하면 **판정 모델(H2O-Lightning-4B)** 이 1초마다 프레임 4장을 보고 후보를 고르고, 후보마다 프레임 8장을 다시 보고 확인한 것만 클립으로 보여 줍니다. "
        "찾는 장면이 없으면 없다고 말합니다.\n\n"
        "한 번 돌리는 데 GPU가 2~6분 듭니다(60초 영상 약 3분). ZeroGPU의 하루 한도는 로그인하지 않으면 2분, 무료 계정 5분이라 끝까지 못 돌 수 있습니다. "
        "그럴 때는 아래 **미리 돌려 둔 예시**를 눌러 보세요(GPU를 쓰지 않습니다).")


def _table(res):
    return pd.DataFrame(res["rows"], columns=res["columns"]) if res else None


def _results(res):
    """(gallery, table, curve, details)"""
    if not res:
        return None, None, None, None
    return ([(c["file"], c["label"]) for c in res["clips"]] or None, _table(res), pd.DataFrame(res["curve"]) if res["curve"]["점수"] else None, res["details"])


def _render(job, chat):
    """job의 지금 상태 → (chat, status, gallery, table, curve, details, confirm 버튼, 이어 하기 버튼, 타이머)."""
    chat = list(chat or [])
    st = job["stage"]
    log = "\n".join(job["log"][-14:])
    say = lambda text: chat[:-1] + [{"role": "assistant", "content": text}] if chat and chat[-1]["role"] == "assistant" else chat + [{"role": "assistant", "content": text}]
    if job.get("paused"):
        return (say(f"멈췄습니다: {job.get('error')}\n\n받은 것은 기억해 두었습니다. **이어서 하기**를 누르면 같은 곳부터 다시 합니다."), log,
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False), gr.update(visible=True), gr.Timer(active=False))
    if st in WORK:
        tail = f"\n\n`{job['log'][-1]}`" if job["log"] else ""
        return say(PROGRESS[st] + tail), log, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False), gr.update(visible=False), gr.Timer(active=True)
    if st == "confirm":
        return (say(P.proposal_text(job["plan"])), log, None, None, None, {"plan": {k: v for k, v in job["plan"].items() if k != "raw_reply"}},
                gr.update(visible=True), gr.update(visible=False), gr.Timer(active=False))
    if st == "done":
        return (say(job["result"]["summary"]), log, *_results(job["result"]), gr.update(visible=False), gr.update(visible=False), gr.Timer(active=False))
    return (say(f"끝까지 하지 못했습니다: {job.get('error')}\n\n요청을 다시 보내 주세요."), log, gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            gr.update(visible=False), gr.update(visible=False), gr.Timer(active=False))


def ui_send(video, text, game, chat, job_id):
    job = job_ask(job_id, video, text, game)
    chat = list(chat or []) + [{"role": "user", "content": text.strip()}]
    return (*_render(job, chat), job["id"], "")


def ui_confirm(chat, job_id):
    job = job_confirm(job_id)
    chat = list(chat or []) + [{"role": "user", "content": "이대로 찾아 주세요"}]
    return _render(job, chat)


def ui_step(chat, job_id):
    job = JOBS.get(job_id or "")
    if not more(job):
        return (gr.skip(),) * 8 + (gr.Timer(active=False),)
    return _render(job_step(job_id), chat)


def ui_resume(chat, job_id):
    job = JOBS.get(job_id or "")
    if not job:
        raise gr.Error("이어 할 일이 없습니다(Space가 다시 시작됐다면 요청을 다시 보내 주세요).")
    job["paused"] = False
    job.pop("error", None)
    return _render(job, chat)


def list_examples():
    out = []
    for d in sorted(EXAMPLES.glob("*/example.json")) if EXAMPLES.exists() else []:
        try:
            out.append((d.parent.name, json.loads(d.read_text(encoding="utf-8"))))
        except ValueError:
            continue
    return out


def list_inputs():
    f = EXAMPLES / "inputs" / "inputs.json"
    if not f.exists():
        return []
    return [x for x in json.loads(f.read_text(encoding="utf-8")) if (EXAMPLES / "inputs" / x["file"]).exists()]


def show_example(slug):
    """미리 돌려 둔 결과를 그대로 보여 준다(GPU 없음). 파일은 examples/<slug>/에 있다."""
    ex = json.loads((EXAMPLES / slug / "example.json").read_text(encoding="utf-8"))
    res = {**ex, "clips": [{**c, "file": str(EXAMPLES / slug / c["file"])} for c in ex["clips"]]}
    chat = [{"role": "user", "content": ex["request"]}, {"role": "assistant", "content": ex["proposal"]},
            {"role": "user", "content": "이대로 찾아 주세요"}, {"role": "assistant", "content": ex["summary"] + "\n\n(미리 돌려 둔 결과입니다)"}]
    return (chat, "\n".join(ex.get("log", [])[-14:]), *_results(res), gr.update(visible=False), gr.update(visible=False), gr.Timer(active=False))


# API (시험과 자동화용): 화면과 같은 job을 단계마다 부른다
def api_ask(video, text, game, job_id, plan_json):
    job = job_ask(job_id, video, text, game, plan_json)
    return job["id"], status_of(job)


def api_confirm(job_id, answers_json):
    return status_of(job_confirm(job_id, answers_json))


def api_step(job_id):
    job = JOBS.get(job_id or "")
    if job and job.get("paused"):                 # API에서는 부르는 쪽이 다시 부르면 이어 한다
        job["paused"] = False
        job.pop("error", None)
    job = job_step(job_id)
    st = status_of(job, full=True)
    files = [c["file"] for c in job["result"]["clips"]] if job and job["stage"] == "done" else None
    return st, files


def build_ui():
    examples = list_examples()
    with gr.Blocks(title="Moment Finder") as demo:
        gr.Markdown(HEAD)
        job_id = gr.State("")
        timer = gr.Timer(1.5, active=False)
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="영상 (10분까지)", sources=["upload"], height=280)
                chatbot = gr.Chatbot(label="대화", height=380)
                with gr.Row():
                    msg = gr.Textbox(label="어떤 장면을 찾을까요?", placeholder="예: 로봇이 사탕 컵에 넣는 거 성공한 장면 찾아줘", scale=4)
                    send = gr.Button("보내기", variant="primary", scale=1)
                with gr.Row():
                    confirm = gr.Button("이대로 찾기", variant="primary", visible=False)
                    resume = gr.Button("이어서 하기", visible=False)
                    reset = gr.Button("새 요청", size="sm")
                with gr.Accordion("영상 설명 한 줄 (선택)", open=False):
                    game = gr.Textbox(label="영상 설명", placeholder="예: Minecraft survival gameplay, first-person view", show_label=False)
                inputs = list_inputs()
                if inputs:
                    gr.Examples(examples=[[str(EXAMPLES / "inputs" / x["file"]), x["request"]] for x in inputs], inputs=[video, msg],
                                label="이 영상으로 직접 돌려 보기 (영상과 요청이 채워집니다. 그다음 보내기)", cache_examples=False)
                if examples:
                    gr.Markdown("**미리 돌려 둔 예시** (GPU를 쓰지 않습니다)")
                    ex_buttons = [(slug, gr.Button(ex.get("title") or slug, size="sm")) for slug, ex in examples]
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="클립 (다시 확인을 통과한 것만)", columns=1, height="auto")
                table = gr.Dataframe(label="후보와 다시 확인 결과", interactive=False, wrap=True)
                curve = gr.LinePlot(x="초", y="점수", label="1초마다의 점수 (요청에 맞는 정도 0~1)", height=200)
                status = gr.Textbox(label="진행", lines=8, interactive=False)
                with gr.Accordion("자세한 값 (계획, 질문, 후보별 답)", open=False):
                    details = gr.JSON(label="자세한 값")
        outs = [chatbot, status, gallery, table, curve, details, confirm, resume, timer]
        send.click(ui_send, inputs=[video, msg, game, chatbot, job_id], outputs=outs + [job_id, msg], api_name=False)
        msg.submit(ui_send, inputs=[video, msg, game, chatbot, job_id], outputs=outs + [job_id, msg], api_name=False)
        confirm.click(ui_confirm, inputs=[chatbot, job_id], outputs=outs, api_name=False)
        resume.click(ui_resume, inputs=[chatbot, job_id], outputs=outs, api_name=False)
        timer.tick(ui_step, inputs=[chatbot, job_id], outputs=outs, api_name=False, trigger_mode="once", show_progress="hidden")
        reset.click(lambda: ([], "", None, None, None, None, gr.update(visible=False), gr.update(visible=False), gr.Timer(active=False), "", ""),
                    inputs=None, outputs=outs + [job_id, msg], api_name=False)
        if examples:
            for slug, btn in ex_buttons:
                btn.click(lambda s=slug: show_example(s), inputs=None, outputs=outs, api_name=False)
        # API: 화면과 같은 job을 단계마다 부른다(/ask → /step 반복 → /confirm → /step 반복)
        with gr.Accordion("API (시험용)", open=False, visible=False):
            a_job, a_plan, a_answers = gr.Textbox(label="job"), gr.Textbox(label="plan_json"), gr.Textbox(label="answers_json")
            a_out, a_files = gr.JSON(label="status"), gr.Files(label="clips")
        gr.Button(visible=False).click(api_ask, inputs=[video, msg, game, a_job, a_plan], outputs=[a_job, a_out], api_name="ask")
        gr.Button(visible=False).click(api_confirm, inputs=[a_job, a_answers], outputs=[a_out], api_name="confirm")
        gr.Button(visible=False).click(api_step, inputs=[a_job], outputs=[a_out, a_files], api_name="step")
        # ZeroGPU는 시작할 때 이벤트에 묶인 @spaces.GPU 함수를 찾는다. GPU 함수는 단계 안에서 부르므로 보이지 않는 곳에 한 번 더 묶어 둔다
        if GPU["judge"] is not None:
            h_in, h_out = gr.JSON(visible=False), gr.JSON(visible=False)
            gr.Button(visible=False).click(GPU["judge"], inputs=h_in, outputs=h_out, api_name=False)
            h_in2, h_n, h_out2 = gr.JSON(visible=False), gr.Number(visible=False), gr.JSON(visible=False)
            gr.Button(visible=False).click(GPU["write"], inputs=[h_in2, h_n], outputs=h_out2, api_name=False)
    return demo
