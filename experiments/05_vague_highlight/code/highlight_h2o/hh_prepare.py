"""highlight_h2o 준비: 첫 하이라이트 실험의 요청 B(2026-11-11, TypeSafe Jev)를 판정 모델(H2O)에 다시 묻기 위한 위치·프레임·요청 본문.

이 Mac에서 돈다. GPU·Modal·유료 API는 쓰지 않는다. hh_modal.py가 이 파일을 모듈로 불러 같은 본문을 만든다
(그래서 모듈 맨 위에서는 표준 라이브러리만 불러온다. Pillow·imageio-ffmpeg는 함수 안에서만).

단계
  items    : 두 묶음의 위치 목록과 사건 다시 세기, 프레임 확인 → data/items.json, data/counts.json
             같은 434곳 = 원래 요청 B의 434곳(받아 둔 답 파일의 시각·프레임 그대로)
             영상 전체 1초마다 = 12개 영상의 모든 1초 위치(0.5, 1.5, …초), 같은 4장 묶음 규칙
  frames   : 영상마다 tar 하나(data/packs/<영상>.tar). 원래 실험이 뽑아 둔 파일(frame_limit/data/hl_frames)은 읽어 넣고,
             없는 시각은 같은 방법(ffmpeg -ss t, 원본 크기, -q:v 2)으로 뽑는다 → data/frames_summary.json
  dryrun   : 서버 없이 모든 요청 본문을 만들어 센다(요청, 그림, 질문, 토큰 어림, 그림 크기). 보기 본문 2개 → data/dryrun.json, samples/
실행: uv run --offline --no-project --with pillow --with imageio-ffmpeg python -I experiments/highlight_h2o/hh_prepare.py items
      uv run --offline --no-project --with pillow --with imageio-ffmpeg python -I experiments/highlight_h2o/hh_prepare.py frames
      uv run --offline --no-project --with pillow python -I experiments/highlight_h2o/hh_prepare.py dryrun

문장과 선택지는 frame_limit/scripts/highlight/hl_run.py에서 글자 그대로 옮겼다. items 단계가 원래 파일과 받아 둔 답 파일의 meta로 같은지 확인한다.
"""
import base64
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_P = HERE.parents
ROOT = _P[1] if len(_P) > 2 else HERE          # moment-finder (Modal 컨테이너 안에서는 쓰지 않는다)
FL = ROOT / "experiments" / "frame_limit"
MCD = ROOT / "experiments" / "mc_dense"
CDD = ROOT / "experiments" / "charades_dense"

# ---- 원래 요청 B (hl_run.py 그대로. 바꾸지 않는다)
GAME = "Minecraft survival gameplay, first-person view"
REQUEST = "the most exciting and important moments of this Minecraft session"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
ABOUT = "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first)."
ACTION = ["Calm: walking, looking around, digging or using a menu. Nothing is attacking or being attacked",
          "An animal or monster is close to the player, but nobody is hitting anybody",
          "The player is hitting an animal or monster, or is being hit",
          "The player is about to die or has died: almost no hearts left, or the death screen is shown"]
PROGRESS = ["No progress: moving around, or digging plain dirt or stone",
            "Gathering common materials such as wood, cobblestone or coal",
            "Making or using a workstation or tool: a crafting table, furnace, pickaxe or axe",
            "A rare find or top item: diamonds, or a diamond tool"]
KINDS = {"fight": "The player is fighting or hunting.", "hurt": "The player is taking damage.",
         "crafting": "A crafting, furnace or inventory window is open.", "mining": "The player is digging or mining blocks.",
         "moving": "The player is walking, running or looking around.", "other": "None of the above."}
Q_ACTION = "How much is the player fighting or in danger in `frames`?"
Q_PROGRESS = "How much progress toward better tools do `frames` show?"
Q_KIND = "What is the player doing in `frames`?"
Q_WORTH = "Would this moment fit `viewer_request`?"

# ---- 정상 확인: mc_dense의 sep4_half 요청 그대로(mc_dense_modal.py, h2o_startup_modal.py의 _body). 받아 둔 H2O 답과 견준다
TICKS = 20
SANITY_VIDEOS = ("mc_08", "ch_04")         # h2o_startup이 다시 물어 흔들림을 잰 것과 같은 200곳(영상 둘 × 위치 0~99)
SANITY_K = 100
SANITY_TICKS = (-5, 5, 15, 25)
SENT = {
    "inv": "the player's inventory screen is open, the one that shows the player's character next to the armor slots",
    "chest": "a chest's storage screen is open, the one titled Chest or Large Chest that shows the chest's rows of item slots above the player's inventory rows",
}
HEAD_SEP = "`frames` holds 4 consecutive frames from a video, {gap} seconds apart, in time order (earliest first). "
ASK = {"show": "Do these frames show the following: {s}?", "any": "Does at least one of these frames show the following: {s}?"}

# ---- 채점에 쓰는 사건 붙이기 (hl_eval.py 그대로: 사건을 앞뒤 1.5초 안의 가장 가까운 위치에 붙인다. kill·tool_break는 쓰지 않는다)
NEAR_SCORE = 1.5
SKIP_KINDS = ("kill", "tool_break")


def fname(t):
    return f"{round(t * 1000):07d}.jpg"     # frame_limit/data/hl_frames와 같은 이름(밀리초)


def times_of(t, duration):
    """hl_run.py stage_score와 같은 규칙: 위치 ± 0.25, 0.75초. 영상 끝을 넘으면 (길이 − 0.1초)로, 0초 앞이면 0으로 당긴다."""
    last = duration - 0.1
    return [round(min(max(t + off, 0.0), last), 2) for off in OFFSETS]


def questions():
    """요청 B의 질문 넷. 질문의 JSON 모양은 TypeSafe SDK(typesafe_sdk/_core/question_types.py)가 보내는 것과 같다."""
    return {"action": {"type": "score", "instructions": Q_ACTION, "criteria": list(ACTION)},
            "progress": {"type": "score", "instructions": Q_PROGRESS, "criteria": list(PROGRESS)},
            "kind": {"type": "choice", "instructions": Q_KIND, "criteria": dict(KINDS)},
            "worth": {"type": "noul", "instructions": Q_WORTH}}


def uri(b):
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def body(item, get):
    """요청 B의 본문(판정 모델(H2O)의 /v1/systemone 형식). get(영상, 파일 이름) → JPEG 바이트.
    상태의 칸 이름·순서·글은 원래와 같고, 그림만 SDK의 Image 대신 data URI 글자열이다."""
    frames = [{"seconds": t, "frame": uri(get(item["video"], fname(t)))} for t in item["times"]]
    state = {"game": GAME, "viewer_request": REQUEST, "about": ABOUT, "frames": frames}
    return {"state": state, "questions": questions()}


def sanity_body(get, vid, k, last):
    """mc_dense sep4_half의 본문 그대로. get(영상, 파일 이름) → 바이트(mf-mc-dense의 packs/<vid>.tar 안 이름 05d.jpg)."""
    ticks = [min(max(TICKS * k + t, 0), last) for t in SANITY_TICKS]
    state = {"frames": [{"seconds": round(n / TICKS, 2), "frame": uri(get(vid, f"{n:05d}.jpg"))} for n in ticks]}
    head = HEAD_SEP.format(gap="0.5")
    qs = {f"{c}_{a}": {"type": "noul", "instructions": head + ask.format(s=s)} for c, s in SENT.items() for a, ask in ASK.items()}
    return {"state": state, "questions": qs}


def parse(out):
    """서버 답 → 원래 답 파일과 같은 모양. score는 서버가 준 기댓값을 쓰고, 확률로 다시 센 기댓값과 0.01 넘게 다르면 오류로 본다."""
    ans, rec = out["answers"], {}
    for name in ("action", "progress"):
        a = ans[name]
        probs = {str(k): float(v) for k, v in a["probabilities"].items()}
        if sorted(probs) != ["0", "1", "2", "3"] or abs(sum(probs.values()) - 1) > 0.01:
            raise ValueError(f"{name}: 단계 확률이 이상하다 {probs}")
        expect = sum(int(k) * v for k, v in probs.items())
        if abs(expect - float(a["score"])) > 0.01:
            raise ValueError(f"{name}: score {a['score']}가 확률의 기댓값 {expect:.3f}과 다르다")
        rec[name] = {"score": float(a["score"]), "confidence": a.get("confidence"), "probabilities": probs}
    kind = {k: float(v) for k, v in ans["kind"]["probabilities"].items()}
    if set(kind) != set(KINDS) or abs(sum(kind.values()) - 1) > 0.01:
        raise ValueError(f"kind: 선택지 확률이 이상하다 {kind}")
    rec["kind"] = kind
    p = float(ans["worth"]["noul"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"worth: 확률 {p}")
    rec["worth"] = p
    return rec


def parse_sanity(out):
    return {q: float(a["noul"]) for q, a in out["answers"].items()}


def attach(items, events, eligible):
    """사건 → 위치 붙이기. (위치 id → 사건 목록, 붙지 못한 사건 수). 동률이면 목록에서 앞선 위치(원래 코드의 min과 같다)."""
    out = {it["id"]: [] for it in items}
    by_video = {}
    for it in items:
        if eligible(it):
            by_video.setdefault(it["video"], []).append(it)
    unmatched = 0
    for e in events:
        if e["kind"] in SKIP_KINDS:
            continue
        near = [it for it in by_video.get(e["video"], []) if abs(it["t"] - e["t"]) <= NEAR_SCORE]
        if near:
            out[min(near, key=lambda it: abs(it["t"] - e["t"]))["id"]].append(e)
        else:
            unmatched += 1
    return out, unmatched


def has(evs, kind, level=None):
    return any(e["kind"] == kind and (level is None or e.get("level") == level) for e in evs)


def label(evs):
    """원래 채점의 세 가지 정답. fight = 싸움·피해(action의 정답), prog = 제작 2단계 이상·다이아몬드(progress의 정답), big = 둘 중 하나(큰 사건)."""
    fight = has(evs, "fight") or has(evs, "hurt")
    prog = has(evs, "crafting", 2) or has(evs, "crafting", 3) or has(evs, "find")
    return {"fight": fight, "prog": prog, "big": fight or prog}


# =====================================================================  이 Mac에서만 쓰는 부분
def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def _literals(path):
    """파이썬 파일의 맨 위 대입(이름 = 값)과 모든 글자열 상수. 파일을 실행하지 않고 읽는다."""
    import ast
    tree = ast.parse(Path(path).read_text())
    names, strings = {}, set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                names[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                if isinstance(node.value, ast.Dict):   # 값 일부가 이름인 표(mc_dense의 LAYOUTS): 읽히는 칸만
                    part = {}
                    for k, v in zip(node.value.keys, node.value.values):
                        try:
                            part[ast.literal_eval(k)] = ast.literal_eval(v)
                        except Exception:
                            pass
                    names[node.targets[0].id] = part
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return names, strings


def check_wording():
    """옮긴 문장이 원래 파일·받아 둔 답의 meta와 글자까지 같은가. 다르면 멈춘다."""
    names, strings = _literals(FL / "scripts" / "highlight" / "hl_run.py")
    meta = read(FL / "results" / "hl_mc_score.json")["meta"]
    checks = {"GAME": names["GAME"] == GAME, "REQUEST": names["REQUEST"] == REQUEST == meta["request"],
              "OFFSETS": tuple(names["OFFSETS"]) == OFFSETS, "ACTION": names["ACTION"] == ACTION == meta["action"],
              "PROGRESS": names["PROGRESS"] == PROGRESS == meta["progress"], "KINDS": names["KINDS"] == KINDS == meta["kinds"],
              "KINDS 순서": list(names["KINDS"]) == list(KINDS) == list(meta["kinds"]),
              **{n: s in strings for n, s in (("ABOUT", ABOUT), ("Q_ACTION", Q_ACTION), ("Q_PROGRESS", Q_PROGRESS), ("Q_KIND", Q_KIND), ("Q_WORTH", Q_WORTH))}}
    mnames, mstrings = _literals(MCD / "mc_dense_modal.py")
    checks.update({"정상 확인 SENT": mnames["SENT"] == SENT, "정상 확인 HEAD": mnames["HEAD"]["sep"] == HEAD_SEP,
                   "정상 확인 ASK": mnames["ASK"] == ASK, "정상 확인 틱": tuple(mnames["LAYOUTS"]["sep4_half"]["ticks"]) == SANITY_TICKS})
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"옮긴 문장이 원래와 다르다: {bad}")
    return checks


def derive_events(videos):
    """hl_events.py와 같은 규칙으로 행동 기록(jsonl)에서 사건을 다시 뽑는다(파일에 쓰지 않음). 받아 둔 mc_events.json과 견주려는 것."""
    tools, stations = ("_pickaxe", "_axe", "_sword", "_shovel", "_hoe"), {"crafting_table", "furnace", "iron_ingot"}
    events = []
    for vid, v in videos.items():
        prev = None
        with open((FL / v["file"]).with_suffix(".jsonl")) as f:
            for i, line in enumerate(f):
                stats = json.loads(line).get("stats") or {}
                if prev is not None:
                    for key, value in stats.items():
                        if value <= prev.get(key, 0):
                            continue
                        group, _, item = key.partition(":")
                        group, item = group.replace("minecraft.", ""), item.replace("minecraft.", "")
                        ev = {"video": vid, "t": round(i / TICKS, 2), "item": item, "amount": value - prev.get(key, 0)}
                        if group == "craft_item":
                            lvl = 3 if item.startswith("diamond_") else (2 if item in stations or item.endswith(tools) else 1)
                            events.append({**ev, "kind": "crafting", "level": lvl})
                        elif group == "kill_entity":
                            events.append({**ev, "kind": "kill"})
                        elif group == "break_item":
                            events.append({**ev, "kind": "tool_break"})
                        elif group == "mine_block" and item.endswith("_ore"):
                            events.append({**ev, "kind": "ore"})
                        elif group == "pickup" and item == "diamond":
                            events.append({**ev, "kind": "find"})
                        elif key == "minecraft.custom:minecraft.damage_dealt":
                            events.append({**ev, "kind": "fight"})
                        elif key == "minecraft.custom:minecraft.damage_taken":
                            events.append({**ev, "kind": "hurt"})
                prev = stats
    return events


def load_items():
    return read(HERE / "data" / "items.json")


def stage_items(args):
    import collections
    import subprocess
    from PIL import Image
    import imageio_ffmpeg
    checks = check_wording()
    videos = read(FL / "data" / "mc_items.json")["videos"]
    frames_idx = read(FL / "data" / "mc_frames.json")
    stored = read(FL / "results" / "hl_mc_score.json")
    events = read(FL / "data" / "mc_events.json")["events"]
    again = derive_events(videos)
    key = lambda e: (e["video"], e["t"], e["kind"], e["item"], e.get("level"), e["amount"])
    events_same = sorted(map(key, again)) == sorted(map(key, events))

    # 같은 434곳: 받아 둔 답 파일의 순서·시각·프레임 그대로
    same, problems = [], []
    for r in stored["records"]:
        it = {"id": f"{r['video'][:5]}@{r['t']:g}", "arm": "same434", "video": r["video"], "t": r["t"], "pos": r["pos"], "role": r["role"],
              "times": r["times"], "src": r["frames"]}
        if r["times"] != times_of(r["t"], videos[r["video"]]["duration"]):
            problems.append(f"{it['id']}: 시각이 규칙과 다르다 {r['times']}")
        if [Path(p).name for p in r["frames"]] != [fname(t) for t in r["times"]]:
            problems.append(f"{it['id']}: 프레임 파일 이름이 시각과 맞지 않는다")
        if frames_idx[r["video"]]["centers"][r["pos"]] != r["t"]:
            problems.append(f"{it['id']}: pos와 시각이 맞지 않는다")
        same.append(it)
    ids = [it["id"] for it in same]
    if len(set(ids)) != len(ids):
        problems.append("같은 434곳에 겹치는 위치가 있다")

    # 434곳의 프레임: 있는가, 640×360인가
    sizes, missing = collections.Counter(), []
    for it in same:
        for p in it["src"]:
            f = FL / p
            if not f.exists():
                missing.append(p)
                continue
            with Image.open(f) as im:
                sizes[f"{im.width}x{im.height}"] += 1
    unique_434 = {(it["video"], t) for it in same for t in it["times"]}

    # 영상 전체 1초마다
    in434 = set(ids)
    full = []
    for video in sorted(frames_idx):
        for t in frames_idx[video]["centers"]:
            iid = f"{video[:5]}@{t:g}"
            full.append({"id": iid, "arm": "full", "video": video, "t": t, "role": "position", "in434": iid in in434,
                         "times": times_of(t, videos[video]["duration"])})
    have = {(v.name, f.name) for v in (FL / "data" / "hl_frames").iterdir() if v.is_dir() for f in v.iterdir()}
    need_full = {(it["video"], t) for it in full for t in it["times"]}
    present_full = sum((v, fname(t)) in have for v, t in need_full)

    # 정상 확인 200곳 (프레임은 Modal 볼륨 mf-mc-dense에만 있다)
    sanity = [{"id": f"{v}@{k}", "arm": "sanity", "video": v, "k": k} for v in SANITY_VIDEOS for k in range(SANITY_K)]

    # 세기: 영상마다 위치, 사건 종류마다
    per_video = {v: {"positions_full": sum(it["video"] == v for it in full),
                     "same434_candidates": sum(it["video"] == v and it["role"] == "candidate" for it in same),
                     "same434_controls": sum(it["video"] == v and it["role"] == "control" for it in same)} for v in sorted(frames_idx)}
    kinds = collections.Counter(e["kind"] for e in events)
    levels = collections.Counter(e["level"] for e in events if e["kind"] == "crafting")
    by_video_kind = {v: dict(collections.Counter(e["kind"] for e in events if e["video"] == v)) for v in sorted(videos)}
    arms = {}
    for name, its, ok in (("same434", same, lambda it: it["role"] != "control"), ("full", full, lambda it: True)):
        att, unmatched = attach(its, events, ok)
        lab = [label(att[it["id"]]) for it in its]
        arms[name] = {"positions": len(its), "attached_events": sum(len(v) for v in att.values()), "unmatched_events": unmatched,
                      "positives": {k: sum(l[k] for l in lab) for k in ("fight", "prog", "big")},
                      "big_by_video": {v: sum(l["big"] for it, l in zip(its, lab) if it["video"] == v) for v in sorted(frames_idx)}}

    # 소리: 영상마다 흐름(stream) 목록
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    audio = {}
    for vid, v in sorted(videos.items()):
        err = subprocess.run([ff, "-hide_banner", "-i", str(FL / v["file"])], capture_output=True, text=True).stderr
        audio[vid] = [ln.strip() for ln in err.splitlines() if "Stream #" in ln]
    has_audio = {vid: any("Audio:" in s for s in ss) for vid, ss in audio.items()}

    counts = {"wording_checks": checks, "events_rederived_from_jsonl_equal_to_mc_events_json": events_same,
              "events_total": len(events), "events_by_kind": dict(kinds), "crafting_by_level": dict(sorted(levels.items())),
              "events_by_video_kind": by_video_kind, "per_video": per_video,
              "same434": {"positions": len(same), "roles": dict(collections.Counter(it["role"] for it in same)),
                          "frames_listed": sum(len(it["src"]) for it in same), "unique_frames": len(unique_434),
                          "missing_frames": missing, "frame_sizes": dict(sizes), "problems": problems},
              "full": {"positions": len(full), "overlap_with_434": sum(it["in434"] for it in full), "unique_frames": len(need_full),
                       "already_in_hl_frames": present_full, "to_extract": len(need_full) - present_full},
              "sanity": {"positions": len(sanity), "videos": list(SANITY_VIDEOS)},
              "labels": arms, "audio_streams": audio, "has_audio": has_audio}
    write(HERE / "data" / "items.json", {"_note": "hh_prepare.py items가 만든 위치 목록. times = 프레임 시각(초), src = 원래 실험의 프레임 경로(frame_limit 기준)",
                                         "same434": same, "full": full, "sanity": sanity})
    write(HERE / "data" / "counts.json", counts)
    print(json.dumps({k: counts[k] for k in ("events_rederived_from_jsonl_equal_to_mc_events_json", "events_total", "events_by_kind",
                                              "crafting_by_level", "same434", "full", "labels", "has_audio")}, ensure_ascii=False, indent=1))
    if problems or missing:
        raise SystemExit(f"문제 {len(problems)}건, 없는 프레임 {len(missing)}장")


def stage_frames(args):
    import concurrent.futures as cf
    import hashlib
    import io
    import random
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import time
    from PIL import Image
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    videos = read(FL / "data" / "mc_items.json")["videos"]
    items = load_items()
    need = sorted({(it["video"], t) for arm in ("same434", "full") for it in items[arm] for t in it["times"]})

    def grab(video, t, dest):   # frame_limit/scripts/base/extract_frames.py의 grab과 같은 명령
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(FL / videos[video]["file"]),
               "-frames:v", "1", "-q:v", "2", str(dest)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not dest.exists():
            raise RuntimeError(f"ffmpeg 실패 {video} @ {t}: {p.stderr[:200]}")
        return dest

    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="hh_frames_", dir=HERE / "data"))
    try:
        # 1) 이 ffmpeg로 뽑으면 원래 파일과 바이트가 같은가(원래 파일 60장을 다시 뽑아 본다. 씨앗 20261112)
        have = [(v, t) for v, t in need if (FL / "data" / "hl_frames" / v / fname(t)).exists()]
        probe = random.Random(20261112).sample(have, min(60, len(have)))
        same_bytes = 0
        for v, t in probe:
            d = grab(v, t, tmp / f"probe_{v[:5]}_{fname(t)}")
            same_bytes += d.read_bytes() == (FL / "data" / "hl_frames" / v / fname(t)).read_bytes()
        # 2) 없는 시각을 뽑는다
        todo = [(v, t) for v, t in need if not (FL / "data" / "hl_frames" / v / fname(t)).exists()]
        t1 = time.time()
        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda vt: grab(vt[0], vt[1], tmp / f"{vt[0]}__{fname(vt[1])}"), todo))
        extract_seconds = time.time() - t1
        # 3) 영상마다 tar 하나. 그림은 모두 640×360인지 본다
        out = HERE / "data" / "packs"
        out.mkdir(parents=True, exist_ok=True)
        summary, sizes = {}, {}
        for video in sorted({v for v, _ in need}):
            names, n_old, n_new, total = [], 0, 0, 0
            with tarfile.open(out / f"{video}.tar.part", "w") as tar:
                for v, t in need:
                    if v != video:
                        continue
                    old = FL / "data" / "hl_frames" / v / fname(t)
                    src = old if old.exists() else tmp / f"{v}__{fname(t)}"
                    b = src.read_bytes()
                    with Image.open(io.BytesIO(b)) as im:
                        sizes[f"{im.width}x{im.height}"] = sizes.get(f"{im.width}x{im.height}", 0) + 1
                    info = tarfile.TarInfo(fname(t))
                    info.size = len(b)
                    tar.addfile(info, io.BytesIO(b))
                    n_old += old.exists()
                    n_new += not old.exists()
                    total += len(b)
                    names.append(fname(t))
            (out / f"{video}.tar.part").replace(out / f"{video}.tar")
            summary[video] = {"frames": len(names), "from_hl_frames": n_old, "extracted": n_new, "image_bytes": total,
                              "tar_bytes": (out / f"{video}.tar").stat().st_size,
                              "sha256": hashlib.sha256((out / f"{video}.tar").read_bytes()).hexdigest()[:16]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = {"ffmpeg": ff.rsplit("/", 1)[-1], "frames_needed": len(need), "from_hl_frames": len(need) - len(todo), "extracted": len(todo),
           "extract_seconds": round(extract_seconds, 1), "extract_workers": 8, "probe_regrab_same_bytes": [same_bytes, len(probe)],
           "image_sizes": sizes, "packs": summary, "packs_total_bytes": sum(s["tar_bytes"] for s in summary.values()),
           "stage_seconds": round(time.time() - t0, 1)}
    write(HERE / "data" / "frames_summary.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "packs"}, ensure_ascii=False, indent=1))


def load_packs(folder=None):
    import tarfile
    folder = Path(folder or HERE / "data" / "packs")
    frames = {}
    for p in sorted(folder.glob("*.tar")):
        with tarfile.open(p) as tar:
            frames[p.stem] = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    return frames


# ---- 시간 모형 (README의 비용 어림과 같은 수)
RATE_MEASURED = 15.9          # 초당 요청: 640×360 4장 따로 + 질문 4개, 동시 16 (mc_dense REPORT "속도", 묶음을 메모리에서 읽을 때)
PRICE = 0.000842 + 8 * 0.00001314 + 32 * 0.00000222   # 초당 달러 (h2o_startup README, modal billing rates 2026-11-14)


def estimate(payload, sizes, L):
    """cd_layouts.estimate(charades_dense)와 같은 어림에 score 질문(선택지가 목록)만 더했다.
    (H2O usage.input_tokens 어림, processed_input_tokens 어림 = 질문마다 다시 읽는 몫까지, 그림 토큰).
    score 선택지를 shim이 어떤 글로 펴는지는 확인하지 않았다. 여기서는 "단계 번호: 설명" 한 줄씩으로 셌다."""
    img = 0
    for w, h in sizes:
        hb, wb = L.smart_resize(h, w)
        img += hb * wb // L.PIXELS_PER_TOKEN + 2
    n = [0]

    def strip(x):
        if isinstance(x, str) and x.startswith("data:image"):
            n[0] += 1
            return f"[image {n[0]}]"
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    record = json.dumps(strip(payload["state"]), ensure_ascii=False, separators=(",", ":"))
    head = img + (len(L.SYSTEM) + len("record: ") + len(record) + 60) / L.CHARS_PER_TOKEN
    tails = []
    for q in payload["questions"].values():
        if q["type"] == "noul":
            opts = ["true: the statement holds", "false: it does not"]
        elif q["type"] == "score":
            opts = [f"{i}: {v}" for i, v in enumerate(q["criteria"])]
        else:
            opts = [f"{k}: {v if v is not None else k}" for k, v in q["criteria"].items()]
        text = "question: " + q["instructions"] + "\noptions:\n" + "\n".join(f"{lab}) {o}" for lab, o in zip(L._labels(len(opts)), opts))
        tails.append(len(text) / L.CHARS_PER_TOKEN)
    return round(head + sum(tails)), round(len(tails) * head + sum(tails)), img


def stage_dryrun(args):
    import io
    import statistics
    from PIL import Image
    sys.path.insert(0, str(CDD))
    sys.dont_write_bytecode = True   # charades_dense 폴더에 __pycache__를 만들지 않는다
    import cd_layouts as L   # 그림 크기 맞춤·글 토큰 어림 상수만 빌려 쓴다(charades_dense, 고치지 않음)
    items = load_items()
    frames = load_packs()
    get = lambda v, n: frames[v][n]
    # 정상 확인 본문은 프레임이 볼륨에만 있어 같은 크기(640×360)의 회색 그림으로 만든다(토큰·크기 세기용)
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (128, 128, 128)).save(buf, "JPEG", quality=90)
    gray = buf.getvalue()
    stored = {(r["vid"], r["k"]): r["tokens"] for r in read(MCD / "results" / "sep4_half.json")["records"] if "error" not in r}

    def sizes_of(payload):
        out = []

        def walk(x):
            if isinstance(x, str) and x.startswith("data:image"):
                with Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))) as im:
                    out.append(im.size)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(payload["state"])
        return out

    report, samples = {}, {}
    ratio_rows = []
    for arm in ("sanity", "same434", "full"):
        rows = []
        for it in items[arm]:
            if arm == "sanity":
                payload = sanity_body(lambda v, n: gray, it["video"], it["k"], 6000)
            else:
                payload = body(it, get)
            sz = sizes_of(payload)
            est, proc, img = estimate(payload, sz, L)
            rows.append({"id": it["id"], "images": len(sz), "questions": len(payload["questions"]), "sizes": sz, "est": est, "proc": proc,
                         "img_tokens": img, "bytes": len(json.dumps(payload))})
            if arm == "sanity" and (it["video"], it["k"]) in stored:
                ratio_rows.append(stored[(it["video"], it["k"])] / est)
            if arm != "sanity" and arm not in samples:
                samples[arm] = (it, payload)
        report[arm] = rows
    calib = statistics.median(ratio_rows)
    out = {"calibration": {"what": "정상 확인 200건의 실제 usage.input_tokens(mc_dense sep4_half.json) ÷ cd_layouts.estimate 어림",
                           "median_ratio": round(calib, 3), "min": round(min(ratio_rows), 3), "max": round(max(ratio_rows), 3)},
           "pixel_caps": {"server_MAX_PIXELS": 2_000_000, "h2o_default": 1_638_400}, "arms": {}}
    base_proc = statistics.median(r["proc"] for r in report["sanity"])
    for arm, rows in report.items():
        px = [w * h for r in rows for w, h in r["sizes"]]
        proc_med = statistics.median(r["proc"] for r in rows)
        rate_px = RATE_MEASURED                                   # 화소·읽기 모형(charades_dense README): 그림 4장·질문 4개가 같으니 같은 속도
        rate_cons = RATE_MEASURED * min(1.0, base_proc / proc_med)  # 보수적: 처리 토큰(질문마다 다시 읽는 몫 포함)에 반비례
        out["arms"][arm] = {"requests": len(rows), "images": sum(r["images"] for r in rows), "questions": sum(r["questions"] for r in rows),
                            "images_per_request": sorted({r["images"] for r in rows}), "questions_per_request": sorted({r["questions"] for r in rows}),
                            "image_sizes": sorted({f"{w}x{h}" for r in rows for w, h in r["sizes"]}), "max_image_pixels": max(px),
                            "under_cap": max(px) <= 1_638_400, "image_tokens_per_request": statistics.median(r["img_tokens"] for r in rows),
                            "input_tokens_est_raw_median": statistics.median(r["est"] for r in rows),
                            "input_tokens_est_calibrated_median": round(statistics.median(r["est"] for r in rows) * calib),
                            "processed_tokens_est_median": proc_med, "body_bytes_median": statistics.median(r["bytes"] for r in rows),
                            "rate_px_model": round(rate_px, 2), "rate_conservative": round(rate_cons, 2),
                            "est_seconds_px_model": round(len(rows) / rate_px, 1), "est_seconds_conservative": round(len(rows) / rate_cons, 1)}
    out["price_per_second"] = round(PRICE, 8)
    write(HERE / "data" / "dryrun.json", out)
    # 보기 본문 2개: 그림 자리를 짧은 표시로 바꿔 저장
    sd = HERE / "samples"
    sd.mkdir(exist_ok=True)
    for arm, (it, payload) in samples.items():
        p = json.loads(json.dumps(payload))
        for fr, t in zip(p["state"]["frames"], it["times"]):
            fr["frame"] = f"<image: JPEG 640x360, {len(frames[it['video']][fname(t)])} bytes, packs/{it['video']}.tar:{fname(t)}>"
        write(sd / f"{arm}_{it['id'].replace('@', '_t')}.json", {"item": {k: v for k, v in it.items() if k != "src"}, "payload": p})
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["items", "frames", "dryrun"])
    args = ap.parse_args()
    {"items": stage_items, "frames": stage_frames, "dryrun": stage_dryrun}[args.stage](args)


if __name__ == "__main__":
    main()
