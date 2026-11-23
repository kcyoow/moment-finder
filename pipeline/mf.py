"""moment-finder 실전 흐름의 명령 (이 Mac). 흐름과 기본값은 pipeline/README.md.

  prepare <영상> --name N --criteria c.json [--title T | --request R] [--game G]
                         → runs/N/{run.json, positions.json, questions.json, pack.tar, requests.json}  (프레임을 뽑고 요청 글을 만든다. GPU 없음)
  clips  N               : runs/N/results.json(판정 모델의 답) → scores.json, clips.json
  render N [--top K]     : clips.json → clips/*.mp4, sheet.jpg, index.html
  stored tvsum|minecraft [--videos a,b | --pick K]
                         : 실험에서 받아 둔 답으로 runs/<이름>/을 만들고 clips까지 돌린다(GPU 없음)
  eval-clips             : README 4절의 확인 → eval/clip_rule.json

실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I pipeline/mf.py <명령> …
(stored와 eval-clips는 experiments/의 채점 코드와 받아 둔 답을 읽는다. TVSum의 사람 점수는 메모리에서만 쓰고, 여기에는 지표만 쓴다.)
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import mf_core as C  # noqa: E402

RUNS = HERE / "runs"
EXP = ROOT / "experiments"
HA, HH, FL = EXP / "highlight_auto", EXP / "highlight_h2o", EXP / "frame_limit"
# 요청의 글(실험과 글자까지 같다: experiments/highlight_auto/ha_prepare.py, experiments/highlight_probe/hp_writers.py)
ABOUT = "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first)."
Q_VAGUE = "Would this moment fit `viewer_request`?"
REQ_TITLE = 'the most important moments of the video titled "{title}"'
REQ_GENERIC = "the most important moments of this video"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
MAX_SIDE, JPEG_Q = 640, "2"
PRICE = 0.000842 + 8 * 0.0000131 + 32 * 0.00000222          # 초당 달러: RTX PRO 6000 + 8코어 + 32GiB (modal.com/pricing, 2026-11-16 확인)
PX_READS, REQ_CAP = 55e6, 16.0                               # 처리량 어림(experiments/highlight_auto/README.md 11절)
READY_SECONDS = (153, 290)                                   # 서버 준비 시간의 잰 범위(PLAN.md 0-2절)
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
SEED, RESAMPLES = 20261112, 2000


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    v = re.search(r"Video: [^\n]*?, (\d{2,5})x(\d{2,5})[^\n]*?, ([\d.]+) fps", err)
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not v or not d:
        raise SystemExit(f"영상을 읽지 못했다: {path}\n{err[-400:]}")
    return {"width": int(v.group(1)), "height": int(v.group(2)), "fps": float(v.group(3)),
            "duration": round(int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)), 3), "audio": "Audio:" in err}


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# =====================================================================  기준 파일
def check_questions(qs, most=6):
    """기준 파일의 형식 검사(experiments/highlight_auto/ha_prepare.py check_criteria의 규칙. 질문 수만 1~6으로 넓혔다).
    most: 질문 수의 상한. 여러 벌의 기준을 한 요청에 함께 묻는 실험에서만 올린다(--max-questions)."""
    errs = []
    if not isinstance(qs, dict) or not 1 <= len(qs) <= most:
        return [f"questions는 질문 1~{most}개의 객체여야 한다 (지금 {len(qs) if isinstance(qs, dict) else type(qs).__name__})"]
    for key, q in qs.items():
        if not KEY_RE.match(key) or key == "worth":
            errs.append(f"{key}: 키는 영어 소문자·숫자·밑줄 40자 이하, 'worth'는 쓸 수 없다")
        if not isinstance(q, dict) or q.get("type") not in ("noul", "score"):
            errs.append(f"{key}: type은 noul 또는 score")
            continue
        ins, c = q.get("instructions", ""), q.get("criteria")
        texts = [ins]
        if not isinstance(ins, str) or not ins.strip() or "`frames`" not in ins:
            errs.append(f"{key}: instructions가 비었거나 `frames`를 가리키지 않는다")
        if q["type"] == "noul":
            if not isinstance(c, dict) or set(c) != {"true", "false"}:
                errs.append(f"{key}: noul의 criteria는 true와 false 설명 둘")
            else:
                texts += list(c.values())
        elif not isinstance(c, list) or not 3 <= len(c) <= 5:
            errs.append(f"{key}: score의 criteria는 단계 설명 3~5개의 목록(낮은 단계부터)")
        else:
            texts += c
        for t in texts:
            if not isinstance(t, str) or not t.strip() or len(t) > 300 or any(ord(ch) > 126 or ord(ch) < 32 for ch in t):
                errs.append(f"{key}: 비었거나 300자를 넘거나 영어(ASCII) 밖의 글자가 있다: {str(t)[:60]!r}")
                break
    return errs


# =====================================================================  prepare: 새 영상
def positions_for(duration, step=1.0):
    """1초 칸마다 위치 하나. 그림 4장은 칸 가운데의 앞뒤 0.25초, 0.75초(experiments/highlight_h2o의 '영상 전체 1초마다'와 같다)."""
    out = []
    k = 0
    while (k + 1) * step <= duration + 1e-9:
        c = (k + 0.5) * step
        out.append({"k": k, "t0": round(k * step, 3), "t1": round((k + 1) * step, 3), "center": round(c, 3),
                    "times": [round(min(max(c + o, 0.0), max(duration - 0.1, 0.0)), 3) for o in OFFSETS]})
        k += 1
    return out


def stage_prepare(args):
    video = Path(args.video).expanduser().resolve()
    run = RUNS / args.name
    if (run / "requests.json").exists() and not args.force:
        raise SystemExit(f"이미 있다: {run} (다시 만들려면 --force)")
    crit = read(args.criteria)
    qs = crit.get("questions", crit)
    errs = check_questions(qs, args.max_questions)
    if errs:
        raise SystemExit("기준 파일 형식 오류:\n" + "\n".join(errs))
    info = probe(video)
    pos = positions_for(info["duration"])
    # 프레임: 0.25초부터 0.5초마다, 그 시각에 가장 가까운 프레임 한 장을 한 번에 뽑는다(실험과 같은 고르기: 시각 × fps를 반올림한 프레임).
    # 위치의 네 시각은 모두 이 격자 위에 있다(양 끝의 당긴 시각만 가장 가까운 장으로). ffmpeg의 fps 필터는 시각이 0.2초쯤 밀려서 쓰지 않는다
    half = 0.5 / info["fps"]
    sel = f"gt(floor((t+{half:.6f}-0.25)*2)\\,floor((prev_t+{half:.6f}-0.25)*2))"
    vf = (f"select='{sel}',scale=w='if(gte(iw,ih),min({MAX_SIDE},iw),-2)':h='if(gte(iw,ih),-2,min({MAX_SIDE},ih))'")
    files = {}
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf, "-fps_mode", "vfr", "-q:v", JPEG_Q,
                            str(Path(td) / "%06d.jpg")], capture_output=True, text=True)
        got = sorted(Path(td).glob("*.jpg"))
        if p.returncode != 0 or not got:
            raise SystemExit(f"프레임을 뽑지 못했다: {p.stderr[-400:]}")
        for f in got:
            files[f.name] = f.read_bytes()
    n_grid = len(files)
    expect = int((info["duration"] - 0.25) * 2) + 1
    if abs(n_grid - expect) > 2:                                # 프레임 간격이 고르지 않은 영상이면 격자가 밀릴 수 있다
        raise SystemExit(f"뽑힌 프레임 수({n_grid})가 영상 길이로 센 수({expect})와 다르다. 프레임 간격이 고르지 않은 영상일 수 있다")

    def grid(t):
        return min(max(int(round((t - 0.25) / 0.5)), 0), n_grid - 1)
    for q in pos:
        idx = [grid(t) for t in q["times"]]
        q["frames"] = [f"{i + 1:06d}.jpg" for i in idx]
        q["seconds"] = [round(0.25 + 0.5 * i, 2) for i in idx]
    from PIL import Image
    w, h = Image.open(io.BytesIO(files["000001.jpg"])).size
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in sorted(files):
            ti = tarfile.TarInfo(name)
            ti.size = len(files[name])
            tar.addfile(ti, io.BytesIO(files[name]))
    run.mkdir(parents=True, exist_ok=True)
    (run / "pack.tar").write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    request = args.request or (REQ_TITLE.format(title=args.title) if args.title else REQ_GENERIC)
    questions = {"worth": {"type": "noul", "instructions": Q_VAGUE}, **qs}
    rate = min(REQ_CAP, PX_READS / (len(questions) * 4 * w * h))
    est = len(pos) / rate
    state = {**({"game": args.game} if args.game else {}), "viewer_request": request, "about": ABOUT}
    write(run / "run.json", {"name": args.name, "video": str(video), "source": "live", **info, "frame_size": [w, h], "grid_frames": n_grid,
                             "viewer_request": request, "game": args.game, "criteria_file": str(Path(args.criteria).resolve()),
                             "criteria_writer": crit.get("writer") or crit.get("writer_model")})
    write(run / "positions.json", pos)
    write(run / "questions.json", {"vague_key": "worth", "criteria_keys": list(qs), "questions": questions})
    write(run / "requests.json", {"state": state, "questions": questions, "pack_sha256": sha,
                                  "positions": [{"k": q["k"], "frames": q["frames"], "seconds": q["seconds"]} for q in pos],
                                  "plan": {"requests": len(pos), "est_seconds": round(est, 1), "rate_per_second": round(rate, 2)}})
    cost = [(r + est) * PRICE for r in READY_SECONDS]
    print(json.dumps({"run": str(run), "video": video.name, **info, "frame_size": [w, h], "grid_frames": n_grid, "pack_mb": round(len(buf.getvalue()) / 1e6, 1),
                      "positions": len(pos), "questions": len(questions), "est_ask_seconds": round(est), "ready_seconds": list(READY_SECONDS),
                      "est_usd": [round(c, 2) for c in cost]}, ensure_ascii=False, indent=1))


def bodies(run):
    """요청 본문을 하나씩 만든다(Modal 안의 것과 같은 식. 세어 보는 데 쓴다)."""
    import base64
    req = read(run / "requests.json")
    with tarfile.open(run / "pack.tar") as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    for q in req["positions"]:
        frames = [{"seconds": s, "frame": "data:image/jpeg;base64," + base64.b64encode(files[n]).decode()} for n, s in zip(q["frames"], q["seconds"])]
        yield q["k"], {"state": {**req["state"], "frames": frames}, "questions": req["questions"]}


# =====================================================================  clips
def levels_of(questions):
    return {k: len(q["criteria"]) for k, q in questions.items() if q.get("type") == "score"}


def load_scores(run, rule="value_mean", use_vague=True):
    pos = read(run / "positions.json")
    qj = read(run / "questions.json")
    recs = {r["k"]: r.get("answers") for r in read(run / "results.json")["records"] if "error" not in r}
    answers = [recs.get(q["k"]) for q in pos]
    comb = C.combine(answers, qj["criteria_keys"], qj.get("vague_key") if use_vague else None, rule=rule, levels=levels_of(qj["questions"]))
    t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
    return pos, qj, comb, t0, t1, sum(a is None for a in answers)


def stage_clips(args, quiet=False):
    run = RUNS / args.name
    info = read(run / "run.json")
    rule, use_vague = getattr(args, "combine", None) or "value_mean", not getattr(args, "no_vague", False)
    pos, qj, comb, t0, t1, missing = load_scores(run, rule, use_vague)
    params = {k: getattr(args, k) for k in ("budget", "max_clips", "grow_q") if getattr(args, k, None)}
    clips = C.make_clips(comb["final"], t0, t1, info["duration"], **params)
    for c in clips:
        c["why"] = C.why(c, {k: comb["rank"][k] for k in qj["criteria_keys"]}, t0, t1)
    write(run / "scores.json", {"t0": t0, "t1": t1, "final": [round(float(x), 5) for x in comb["final"]],
                                "smoothed": [round(float(x), 5) for x in C.smooth(comb["final"])],
                                "rank": {k: [round(float(x), 5) for x in v] for k, v in comb["rank"].items()}, "missing_positions": missing})
    total = sum(c["length"] for c in clips)
    write(run / "clips.json", {"params": {**C.DEFAULTS, **params}, "combine": rule, "vague": use_vague, "duration": info["duration"], "positions": len(pos), "missing_positions": missing,
                               "total_length": round(total, 2), "share": round(total / info["duration"], 4), "clips": clips})
    if not quiet:
        print(f"{args.name}: 영상 {mmss(info['duration'])}, 위치 {len(pos)}곳(답 없음 {missing}), 클립 {len(clips)}개, 합 {total:.0f}초({total / info['duration']:.1%})")
        for c in clips:
            print(f"  #{c['rank']:<2d} {mmss(c['start'])} ~ {mmss(c['end'])} ({c['length']:.0f}초) 정점 {c['peak']:.3f}  {', '.join(f'{k} {v:.2f}' for k, v in c['why'])}")
    return clips


# =====================================================================  render
def grab(video, t, width=320):
    p = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"], capture_output=True)
    return p.stdout


def stage_render(args):
    from PIL import Image, ImageDraw
    run = RUNS / args.name
    info, cj, sc = read(run / "run.json"), read(run / "clips.json"), read(run / "scores.json")
    clips = cj["clips"][: args.top] if args.top else cj["clips"]
    video = Path(info["video"])
    (run / "clips").mkdir(exist_ok=True)
    for old in (run / "clips").glob("*.mp4"):
        old.unlink()
    rows = []
    for c in clips:
        out = run / "clips" / f"{c['rank']:02d}_{int(c['start']):05d}s.mp4"
        cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{c['start']:.3f}", "-i", str(video), "-t", f"{c['length']:.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"클립을 자르지 못했다: {p.stderr[-300:]}")
        shots = [grab(video, t) for t in (c["start"] + 0.3, c["peak_t"], max(c["end"] - 0.3, c["start"]))]
        rows.append((c, out, [Image.open(io.BytesIO(b)).convert("RGB") for b in shots if b]))
    # 한눈에 보는 그림: 클립마다 시작, 정점, 끝의 세 장
    if rows:
        tw, th = rows[0][2][0].size
        bar = 22
        sheet = Image.new("RGB", (tw * 3, (th + bar) * len(rows)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (c, _, ims) in enumerate(rows):
            y = i * (th + bar)
            d.text((6, y + 5), f"#{c['rank']}  {mmss(c['start'])} - {mmss(c['end'])}  ({c['length']:.0f}s)  peak {c['peak']:.3f} @ {mmss(c['peak_t'])}   "
                   + "  ".join(f"{k}={v:.2f}" for k, v in c.get("why", [])), fill="black")
            for j, im in enumerate(ims):
                sheet.paste(im.resize((tw, th)), (j * tw, y + bar))
        sheet.save(run / "sheet.jpg", quality=85)
    # 보기 페이지
    D = info["duration"]
    W, H = 900, 90
    pts = " ".join(f"{(0.5 * (a + b)) / D * W:.1f},{H - s * (H - 6) - 3:.1f}" for a, b, s in zip(sc["t0"], sc["t1"], sc["smoothed"]))
    rects = "".join(f'<rect x="{c["start"] / D * W:.1f}" y="0" width="{max(c["length"] / D * W, 1):.1f}" height="{H}" fill="#f2c94c" opacity="0.55"><title>#{c["rank"]}</title></rect>'
                    for c in clips)
    items = "".join(
        f'<section><h3>#{c["rank"]} · {mmss(c["start"])} ~ {mmss(c["end"])} ({c["length"]:.0f}초) · 정점 {c["peak"]:.3f}</h3>'
        f'<p>{" · ".join(f"{k} {v:.2f}" for k, v in c.get("why", [])) or "&nbsp;"}</p>'
        f'<video controls preload="metadata" src="clips/{out.name}"></video></section>' for c, out, _ in rows)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{info['name']} 하이라이트</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;margin:24px auto;max-width:940px;padding:0 16px;background:#fff;color:#111}}
video{{max-width:100%;max-height:78vh;background:#000}} section{{margin:28px 0}} h3{{margin:0 0 4px}} p{{margin:0 0 8px;color:#555}}
svg{{width:100%;height:auto;border:1px solid #ddd}} small{{color:#666}}</style></head><body>
<h1>{info['name']}</h1>
<p>요청: {info.get('viewer_request') or ''}<br><small>영상 {mmss(D)} · 위치 {cj['positions']}곳 · 클립 {len(clips)}개 · 합 {sum(c['length'] for c in clips):.0f}초</small></p>
<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">{rects}<polyline points="{pts}" fill="none" stroke="#2f6fde" stroke-width="1.2"/></svg>
<small>파란 선: 위치별 점수(앞뒤 평균한 것). 노란 띠: 뽑힌 클립.</small>
{items}</body></html>"""
    (run / "index.html").write_text(html, encoding="utf-8")
    print(f"{args.name}: 클립 {len(rows)}개 → {run / 'clips'}, 그림 {run / 'sheet.jpg'}, 페이지 {run / 'index.html'}")


# =====================================================================  stored: 실험에서 받아 둔 답으로
def tvsum_stored():
    """highlight_auto의 받아 둔 답 → {영상: (정보, 위치, 질문, 답)}."""
    items = read(HA / "data" / "tvsum_items.json")["videos"]
    req = read(HA / "data" / "requests.json")["tvsum"]["videos"]
    recs = {}
    for r in read(HA / "results" / "tvsum.json")["records"]:
        if "error" not in r:
            recs.setdefault(r["video"], {})[r["k"]] = r["answers"]
    vdir = HA / "data" / "tvsum_src" / "x" / "unzip" / "ydata-tvsum50-video" / "video"
    out = {}
    for v, it in items.items():
        fps = it["fps"]
        pos = [{"k": p["k"], "t0": round(p["start"] / fps, 3), "t1": round(p["end"] / fps, 3), "center": p["center"], "times": p["seconds"]} for p in it["positions"]]
        qs = req[v]["questions"]
        info = {"name": f"tvsum_{v}", "video": str(vdir / f"{v}.mp4"), "source": "stored: experiments/highlight_auto/results/tvsum.json",
                "duration": round(it["frames_video"] / fps, 3), "fps": fps, "width": it["width"], "height": it["height"],
                "viewer_request": req[v]["viewer_request"], "category": it["category"]}
        out[v] = (info, pos, {"vague_key": "worth", "criteria_keys": [k for k in qs if k != "worth"], "questions": qs}, recs.get(v, {}))
    return out


def minecraft_stored():
    """highlight_h2o의 받아 둔 답(손으로 쓴 기준 action·progress와 막연한 질문 worth) → 같은 모양."""
    sys.path.insert(0, str(HH))
    import hh_prepare as HP
    meta = read(FL / "data" / "mc_items.json")["videos"]
    qs_all = HP.questions()
    qs = {k: qs_all[k] for k in ("worth", "action", "progress")}
    by = {}
    for r in read(HH / "results" / "full.json")["records"]:
        if "error" not in r:
            w = r["worth"]                                          # 받아 둔 파일에는 "예" 확률이 숫자로 들어 있다
            by.setdefault(r["video"], {})[r["t"]] = {"worth": w if isinstance(w, dict) else {"noul": float(w)},
                                                     "action": {"score": r["action"]["score"]}, "progress": {"score": r["progress"]["score"]}}
    out = {}
    for it in HP.load_items()["full"]:
        out.setdefault(it["video"], []).append(it)
    res = {}
    for v, its in out.items():
        pos = [{"k": i, "t0": it["t"] - 0.5, "t1": it["t"] + 0.5, "center": it["t"], "times": it["times"]} for i, it in enumerate(its)]
        info = {"name": v, "video": str(FL / meta[v]["file"]), "source": "stored: experiments/highlight_h2o/results/full.json (손으로 쓴 기준)",
                "duration": float(len(its)), "fps": None, "viewer_request": HP.REQUEST, "game": HP.GAME}
        res[v] = (info, pos, {"vague_key": "worth", "criteria_keys": ["action", "progress"], "questions": qs}, {i: by[v].get(it["t"]) for i, it in enumerate(its) if by[v].get(it["t"])})
    return res


def stage_stored(args):
    data = tvsum_stored() if args.dataset == "tvsum" else minecraft_stored()
    vids = sorted(data)
    if args.videos:
        pick = args.videos.split(",")
    else:
        rng = np.random.default_rng(20261116)                       # 고르는 것은 씨앗으로(결과를 보고 고르지 않는다)
        pick = [vids[i] for i in sorted(rng.choice(len(vids), size=min(args.pick, len(vids)), replace=False))]
    for v in pick:
        info, pos, qj, recs = data[v]
        run = RUNS / info["name"]
        write(run / "run.json", info)
        write(run / "positions.json", pos)
        write(run / "questions.json", qj)
        write(run / "results.json", {"meta": {"source": info["source"]}, "records": [{"k": k, "answers": a} for k, a in sorted(recs.items())]})
        stage_clips(argparse.Namespace(name=info["name"], budget=None, max_clips=None, grow_q=None, combine=args.combine, no_vague=args.no_vague))
    print("만든 실행:", ", ".join(data[v][0]["name"] for v in pick))


# =====================================================================  eval-clips: README 4절
VARIANTS = {"default": {}, "no_smooth": {"smooth": False}, "no_pad": {"pre": 0.0, "post": 0.0, "min_len": 0.0}, "grow_top15": {"grow_q": 0.85},
            "cap30": {"hard_max": 30.0}}       # cap30 = 2026-11-16 두 번째 고침 전의 규칙(한 대목도 30초에서 자름)


def uniform_mask(n_sel, n_clips, n):
    """같은 수의 위치를, 같은 수의 클립으로, 영상에 고르게 놓는다."""
    m = np.zeros(n, bool)
    n_clips = max(1, min(n_clips, n_sel))
    sizes = [n_sel // n_clips + (1 if i < n_sel % n_clips else 0) for i in range(n_clips)]
    for i, size in enumerate(sizes):
        c = (i + 0.5) / n_clips * n
        a = int(round(c - size / 2))
        a = min(max(a, 0), n - size)
        m[a:a + size] = True
    return m


def boot(per_video, vids):
    rng = np.random.default_rng(SEED)
    x = np.array([per_video[v] for v in vids], float)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(RESAMPLES)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def stage_eval(args):
    for p in (HA, HH):
        sys.path.insert(0, str(p))
    import ha_eval as E
    out = {"label": "클립 규칙의 확인 (pipeline/README.md 4절). 받아 둔 답만 쓴다", "variants": VARIANTS, "defaults": C.DEFAULTS}
    # ---- TVSum
    tv = E.TV()
    data = tvsum_stored()
    per = {name: {k: {} for k in ("clip", "raw", "uniform", "share", "n_clips", "mean_len")} for name in VARIANTS}
    tau_check = {}
    for v in tv.videos:
        info, pos, qj, recs = data[v]
        comb = C.combine([recs.get(q["k"]) for q in pos], qj["criteria_keys"], "worth", levels=levels_of(qj["questions"]))
        t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
        gt = np.asarray(tv.gt[v], float)
        tau_check[v] = E.video_metrics(tv, v, comb["final"])["tau"]
        for name, kw in VARIANTS.items():
            clips = C.make_clips(comb["final"], t0, t1, info["duration"], **kw)
            m = C.covered(clips, t0, t1)
            k = int(m.sum())
            per[name]["clip"][v] = float(gt[m].mean() - gt.mean()) if k else 0.0
            per[name]["raw"][v] = float(E.topk_expect(comb["final"], gt, k) - gt.mean()) if k else 0.0
            per[name]["uniform"][v] = float(gt[uniform_mask(k, len(clips), len(gt))].mean() - gt.mean()) if k else 0.0
            per[name]["share"][v] = k / len(gt)
            per[name]["n_clips"][v] = len(clips)
            per[name]["mean_len"][v] = float(np.mean([c["length"] for c in clips])) if clips else 0.0
    vids = tv.videos
    tvs = {"videos": len(vids), "final_tau_mean": round(float(np.mean(list(tau_check.values()))), 4), "combine": "value_mean + vague (기본값)", "variants": {}}
    for name in VARIANTS:
        P = per[name]
        d_raw = {v: P["clip"][v] - P["raw"][v] for v in vids}
        d_uni = {v: P["clip"][v] - P["uniform"][v] for v in vids}
        tvs["variants"][name] = {
            "positions_share": round(float(np.mean(list(P["share"].values()))), 4), "clips_per_video": round(float(np.mean(list(P["n_clips"].values()))), 2),
            "mean_clip_seconds": round(float(np.mean(list(P["mean_len"].values()))), 2),
            "lift_clips": {"mean": round(float(np.mean(list(P["clip"].values()))), 4), "ci": boot(P["clip"], vids)},
            "lift_raw_same_count": {"mean": round(float(np.mean(list(P["raw"].values()))), 4), "ci": boot(P["raw"], vids)},
            "lift_uniform_same_count": {"mean": round(float(np.mean(list(P["uniform"].values()))), 4), "ci": boot(P["uniform"], vids)},
            "clips_minus_raw": {"mean": round(float(np.mean(list(d_raw.values()))), 4), "ci": boot(d_raw, vids)},
            "clips_minus_uniform": {"mean": round(float(np.mean(list(d_uni.values()))), 4), "ci": boot(d_uni, vids)}}
    out["tvsum"] = tvs
    # ---- 마인크래프트 (대리 정답: 큰 사건)
    import hh_eval as HE
    import hh_prepare as HP
    items = HP.load_items()["full"]
    want = {it["id"]: it for it in items}
    order = {i: k for k, i in enumerate(want)}
    recs = [{**{k: want[r["id"]][k] for k in ("id", "video", "t", "role")}, **{k: r[k] for k in ("action", "progress", "kind", "worth")}}
            for r in sorted((r for r in HP.read(HH / "results" / "full.json")["records"] if "error" not in r), key=lambda r: order[r["id"]])]
    HE.finish(recs, "full", items)
    big = {}
    for r in recs:
        big.setdefault(r["video"], {})[r["t"]] = bool(r["big"])
    mc = minecraft_stored()
    mvs = sorted(mc)
    res = {name: {"hit": {}, "raw_hit": {}, "sel": {}, "n": {}, "big": {}, "n_clips": {}} for name in VARIANTS}
    for v in mvs:
        info, pos, qj, ans = mc[v]
        comb = C.combine([ans.get(q["k"]) for q in pos], qj["criteria_keys"], "worth", levels=levels_of(qj["questions"]))
        t0, t1 = [q["t0"] for q in pos], [q["t1"] for q in pos]
        y = np.array([big[v].get(q["center"], False) for q in pos])
        for name, kw in VARIANTS.items():
            clips = C.make_clips(comb["final"], t0, t1, info["duration"], **kw)
            m = C.covered(clips, t0, t1)
            k = int(m.sum())
            R = res[name]
            R["hit"][v], R["sel"][v], R["n"][v], R["big"][v], R["n_clips"][v] = int((m & y).sum()), k, len(y), int(y.sum()), len(clips)
            R["raw_hit"][v] = float(E.topk_expect(comb["final"], y.astype(float), k) * k) if k else 0.0
    mcs = {"videos": len(mvs), "big_events": int(sum(res["default"]["big"].values())), "variants": {}}
    for name in VARIANTS:
        R = res[name]

        def pooled(pick):
            hit = sum(R["hit"][v] for v in pick)
            raw = sum(R["raw_hit"][v] for v in pick)
            b = sum(R["big"][v] for v in pick)
            share = sum(R["sel"][v] for v in pick) / sum(R["n"][v] for v in pick)
            return (hit / b if b else np.nan), (raw / b if b else np.nan), share
        rng = np.random.default_rng(SEED)
        draws = [pooled([mvs[i] for i in rng.integers(0, len(mvs), len(mvs))]) for _ in range(RESAMPLES)]
        rec, raw, share = pooled(mvs)
        ci = lambda j: [round(float(x), 3) for x in np.nanpercentile([d[j] for d in draws], [2.5, 97.5])]   # noqa: E731
        ci_d = [round(float(x), 3) for x in np.nanpercentile([d[0] - d[1] for d in draws], [2.5, 97.5])]
        mcs["variants"][name] = {"positions_share": round(share, 4), "clips_per_video": round(float(np.mean(list(R["n_clips"].values()))), 2),
                                 "big_events_in_clips": {"share": round(rec, 4), "ci": ci(0), "count": int(sum(R["hit"].values()))},
                                 "big_events_raw_same_count": {"share": round(raw, 4), "ci": ci(1)},
                                 "clips_minus_raw": {"mean": round(rec - raw, 4), "ci": ci_d}, "chance": round(share, 4)}
    out["minecraft"] = mcs
    write(HERE / "eval" / "clip_rule.json", out)
    print(json.dumps({"tvsum": tvs, "minecraft": mcs}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("video")
    s.add_argument("--name", required=True)
    s.add_argument("--criteria", required=True)
    s.add_argument("--title")
    s.add_argument("--request")
    s.add_argument("--game")
    s.add_argument("--force", action="store_true")
    s.add_argument("--max-questions", dest="max_questions", type=int, default=6)
    for name in ("clips", "render"):
        s = sub.add_parser(name)
        s.add_argument("name")
        s.add_argument("--budget", type=float)
        s.add_argument("--max-clips", dest="max_clips", type=int)
        s.add_argument("--grow-q", dest="grow_q", type=float)
        s.add_argument("--combine", choices=C.COMBINE_RULES)
        s.add_argument("--no-vague", dest="no_vague", action="store_true")
        s.add_argument("--top", type=int, default=0)
    s = sub.add_parser("stored")
    s.add_argument("dataset", choices=["tvsum", "minecraft"])
    s.add_argument("--videos")
    s.add_argument("--pick", type=int, default=3)
    s.add_argument("--combine", choices=C.COMBINE_RULES)
    s.add_argument("--no-vague", dest="no_vague", action="store_true")
    sub.add_parser("eval-clips")
    args = ap.parse_args()
    {"prepare": stage_prepare, "clips": stage_clips, "render": stage_render, "stored": stage_stored, "eval-clips": stage_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
