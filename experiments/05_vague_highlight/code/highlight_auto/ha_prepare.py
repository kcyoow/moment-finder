"""highlight_auto 준비(이 Mac). 판정 모델 요청은 없다.

단계
  extract : tgz와 안의 zip을 검사(절대 경로, "..", 링크, 장치 파일)한 뒤 data/tvsum_src/x/ 아래에만 푼다. 푼 파일은 실행하지 않는다
  check   : info.tsv, anno.tsv, 영상(프레임 수, fps, 해상도, 소리 흐름)을 다시 센다 → data/tvsum_src/derived/tvsum_check.json (점수 값은 쓰지 않음. info.tsv의 길이·범주가 들어 있어 tvsum_src 안에 둔다)
  writer  : info.tsv로 writer/writer_input.json을 다시 만든다(status "final")
  items   : TVSum 위치와 프레임 번호 → data/tvsum_items.json (점수 없음)
  frames  : TVSum 프레임 뽑기(긴 변 640 이하, JPEG -q:v 2) → data/packs/<id>.tar, data/frames_summary.json
  audio   : 샷마다 소리 크기(RMS dBFS) → data/tvsum_loudness.json
  requests: Modal에 보낼 요청 글(상태의 글과 질문, 프레임 파일 이름) → data/requests.json (writer/criteria.json이 있어야 함)
  dryrun  : 모든 본문을 만들어 세기와 시간 어림 → data/dryrun.json

규칙 (README 2·3절, Part 2a의 조건)
  - TVSum의 주석·메타데이터 파일(anno.tsv, info.tsv, .mat)은 data/tvsum_src/ 안에만 있다. 경로로 읽기만 한다. 복사하지 않는다
  - 점수 값은 이 파일이 만드는 어떤 파일에도 쓰지 않는다(채점은 ha_eval.py가 메모리에서 한다)
  - 묶음(data/packs/)에는 프레임 JPEG와 _index.json(프레임 번호·시각)만 들어간다

실행: uv run --offline --no-project --with numpy --with pillow --with imageio-ffmpeg python -I experiments/highlight_auto/ha_prepare.py <단계>
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_P = HERE.parents
ROOT = _P[1] if len(_P) > 2 else HERE                       # moment-finder (Modal 컨테이너 안에서는 쓰지 않는다)
SRC = HERE / "data" / "tvsum_src"
X = SRC / "x"                                                # 푼 곳
UNZ = X / "unzip"
PKG = X / "ydata-tvsum50-v1_1"
DERIVED = SRC / "derived"                                    # info.tsv의 내용(제목·길이)이 든 파생 파일도 tvsum_src 안에만 둔다
HH = ROOT / "experiments" / "highlight_h2o"

# ---- 요청 글 (README 4절. 결과를 보기 전에 정함)
ABOUT = "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first)."   # highlight_h2o와 같다
TV_REQUEST = 'the most important moments of the video titled "{title}"'
Q_WORTH = "Would this moment fit `viewer_request`?"                                                               # highlight_h2o와 같다
MC_GAME = "Minecraft survival gameplay, first-person view"
MC_REQUEST = "the most exciting and important moments of this Minecraft session"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)
MAX_SIDE = 640
JPEG_Q = "2"
CAT = {"VT": "Changing Vehicle Tire", "VU": "Getting Vehicle Unstuck", "GA": "Grooming an Animal", "MS": "Making Sandwich",
       "PK": "Parkour", "PR": "Parade", "FM": "Flash Mob Gathering", "BK": "Bee Keeping", "BT": "Attempting Bike Tricks", "DS": "Dog Show"}

# ---- 시간 모형 (README 11절)
PRICE = 0.000842 + 8 * 0.00001314 + 32 * 0.00000222        # 초당 달러
PX_READS = 55e6                                              # 초당 화소·읽기
REQ_CAP = 16.0                                               # 초당 요청 상한


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def uri(b):
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


# =====================================================================  extract
def _safe_name(n):
    return not (n.startswith("/") or "\\" in n or ".." in Path(n).parts or re.match(r"^[A-Za-z]:", n))


def stage_extract(args):
    tgz = SRC / "tvsum50_ver_1_1.tgz"
    log = {"tgz": str(tgz.relative_to(HERE)), "tgz_bytes": tgz.stat().st_size,
           "tgz_sha256": hashlib.sha256(tgz.read_bytes()).hexdigest(), "zips": {}}
    with tarfile.open(tgz, "r:gz") as t:
        members = t.getmembers()
        bad = [m.name for m in members if not _safe_name(m.name) or not (m.isfile() or m.isdir())]
        if bad:
            raise SystemExit(f"tgz에 위험한 항목: {bad}")
        X.mkdir(parents=True, exist_ok=True)
        t.extractall(X, filter="data")
        log["tgz_members"] = [[m.name, m.size] for m in members]
    UNZ.mkdir(exist_ok=True)
    for z in sorted(PKG.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            infos = zf.infolist()
            bad = [i.filename for i in infos if not _safe_name(i.filename) or (i.external_attr >> 16) & 0o170000 == 0o120000]
            if bad:
                raise SystemExit(f"{z.name}에 위험한 항목: {bad}")
            dest = UNZ / z.stem
            dest.mkdir(exist_ok=True)
            for i in infos:
                target = (dest / i.filename).resolve()
                if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
                    raise SystemExit(f"{z.name}: 폴더 밖으로 나가는 항목 {i.filename}")
            zf.extractall(dest)
            log["zips"][z.name] = {"bytes": z.stat().st_size, "members": len(infos),
                                   "files": sorted(i.filename for i in infos if not i.is_dir())[:12] + (["…"] if len(infos) > 12 else [])}
    write(HERE / "data" / "extract_log.json", log)
    print(json.dumps(log, ensure_ascii=False, indent=1)[:4000])


# =====================================================================  자료 읽기 (점수는 메모리에서만)
INFO = UNZ / "ydata-tvsum50-data" / "data" / "ydata-tvsum50-info.tsv"
ANNO = UNZ / "ydata-tvsum50-data" / "data" / "ydata-tvsum50-anno.tsv"
VIDEOS = UNZ / "ydata-tvsum50-video" / "video"


def load_info():
    """info.tsv → 영상 목록(파일 순서). 머리줄: category video_id title url length."""
    lines = INFO.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    if head != ["category", "video_id", "title", "url", "length"]:
        raise SystemExit(f"info.tsv 머리줄이 다르다: {head}")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    for r in rows:
        # 쉼표가 든 제목 5개는 표 형식의 따옴표로 싸여 있다("…, …"). 따옴표는 제목이 아니므로 벗긴다(README 2절)
        t = r["title"]
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"' and "," in t:
            r["title"] = t[1:-1].replace('""', '"')
    return rows


def load_anno():
    """anno.tsv → {영상 id: [20개의 프레임별 점수 목록]} (파일 순서). 메모리에서만 쓴다."""
    out, cats = {}, {}
    for l in ANNO.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        vid, cat, s = l.split("\t")
        out.setdefault(vid, []).append([int(x) for x in s.split(",")])
        cats.setdefault(vid, set()).add(cat)
    return out, cats


def secs(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def probe(path):
    """ffmpeg -i의 머리말에서 영상·소리 흐름을 읽고, 흐름 복사로 영상 패킷 수를 센다(디코딩 없음)."""
    exe = ffmpeg()
    err = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    v = re.search(r"Stream #\d+:\d+[^:]*: Video: (\w+)[^\n]*?, (\d{2,5})x(\d{2,5})[^\n]*?, ([\d.]+) fps", err)
    vline = next((l for l in err.splitlines() if "Video:" in l), "")
    audio = [l.strip() for l in err.splitlines() if "Audio:" in l]
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    tbr = re.search(r"([\d.]+) tbr", vline)
    cnt = subprocess.run([exe, "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    fr = re.findall(r"frame=\s*(\d+)", cnt)
    sar = re.search(r"SAR (\d+):(\d+) DAR (\d+):(\d+)", vline)
    return {"codec": v.group(1) if v else None, "width": int(v.group(2)) if v else None, "height": int(v.group(3)) if v else None,
            "fps": float(v.group(4)) if v else None, "tbr": float(tbr.group(1)) if tbr else None,
            "sar": f"{sar.group(1)}:{sar.group(2)}" if sar else None,
            "duration": round(int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)), 3) if dur else None,
            "packets": int(fr[-1]) if fr else None, "audio_streams": len(audio), "audio": audio[:1], "video_line": vline.strip()[:200]}


def shot_grid(rows):
    """평가자 20명의 프레임별 점수에서 샷 길이 L과 점수가 바뀌는 곳을 본다."""
    from collections import Counter
    n = len(rows[0])
    changes = sorted({i for r in rows for i in range(1, n) if r[i] != r[i - 1]})
    runs = Counter()
    for r in rows:
        start = 0
        for i in range(1, n + 1):
            if i == n or r[i] != r[i - 1]:
                runs[i - start] += 1
                start = i
    gaps = Counter(b - a for a, b in zip([0] + changes, changes))
    L = gaps.most_common(1)[0][0] if gaps else n
    off = [c for c in changes if min(c % L, L - c % L) > 1]
    return {"n": n, "L": L, "change_points": len(changes), "off_grid": len(off), "off_grid_first": off[:5],
            "gap_counts_top": gaps.most_common(4)}


def stage_check(args):
    info = load_info()
    anno, cats = load_anno()
    out = {"info_rows": len(info), "anno_rows": sum(len(v) for v in anno.values()), "anno_videos": len(anno), "videos": {}}
    problems = []
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(8) as ex:
        probes = dict(zip([r["video_id"] for r in info], ex.map(lambda r: probe(VIDEOS / f"{r['video_id']}.mp4"), info)))
    for r in info:
        vid = r["video_id"]
        rows = anno.get(vid, [])
        lens = sorted({len(x) for x in rows})
        p = probes[vid]
        g = shot_grid(rows) if rows else {}
        vals = sorted({v for x in rows for v in x})
        rec = {"category": r["category"], "length": r["length"], "length_seconds": secs(r["length"]),
               "annotators": len(rows), "row_lengths": lens, "anno_category": sorted(cats.get(vid, [])),
               "score_values": vals, **p, **{f"grid_{k}": v for k, v in g.items()}}
        if rows:
            rec["frames_minus_packets"] = lens[0] - p["packets"] if p["packets"] else None
            rec["L_over_fps"] = round(g["L"] / p["fps"], 3) if p["fps"] else None
            rec["shots"] = math.ceil(lens[0] / g["L"])
        if len(rows) != 20 or len(lens) != 1 or g.get("off_grid") or vals and (vals[0] < 1 or vals[-1] > 5):
            problems.append(vid)
        out["videos"][vid] = rec
    v = out["videos"].values()
    out["summary"] = {
        "problems": problems,
        "categories": {c: sum(1 for x in v if x["category"] == c) for c in CAT},
        "total_frames_anno": sum(x["row_lengths"][0] for x in v),
        "total_packets": sum(x["packets"] for x in v),
        "total_length_seconds_info": sum(x["length_seconds"] for x in v),
        "total_duration_container": round(sum(x["duration"] for x in v), 1),
        "total_shots": sum(x["shots"] for x in v),
        "fps": sorted({x["fps"] for x in v}), "L": sorted({x["grid_L"] for x in v}),
        "resolutions": sorted({f"{x['width']}x{x['height']}" for x in v}),
        "resolution_counts": {s: sum(1 for x in v if f"{x['width']}x{x['height']}" == s) for s in sorted({f"{x['width']}x{x['height']}" for x in v})},
        "audio_streams": {k: sum(1 for x in v if x["audio_streams"] == k) for k in sorted({x["audio_streams"] for x in v})},
        "frames_minus_packets": sorted({x["frames_minus_packets"] for x in v}),
        "codecs": sorted({x["codec"] for x in v}),
    }
    write(DERIVED / "tvsum_check.json", out)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


# =====================================================================  items: 위치 (README 3절, 2026-11-14 Part 2a에서 샷의 경계 규칙을 실제 자료에 맞춤)
def exact_fps(rec):
    """ffmpeg가 보여 준 fps(소수 둘째 자리)를 정확한 값으로. 29.97 → 30000/1001, 23.98 → 24000/1001, 정수는 그대로.
    그 밖(fWutDQy1nnY의 29.92)은 tbn/1000(29917 → 29.917)을 쓴다."""
    f = rec["fps"]
    for exact in (30000 / 1001, 24000 / 1001, 30.0, 25.0, 24.0):
        if abs(f - exact) < 0.006:
            return exact
    m = re.search(r"(\d+) tbn", rec.get("video_line", ""))
    if m and abs(int(m.group(1)) / 1000 - f) < 0.01:
        return int(m.group(1)) / 1000
    return rec["packets"] / rec["duration"]


def shot_bounds(rows, fps):
    """샷의 시작 프레임 목록(0번 포함). 점수가 바뀌는 곳은 모두 샷의 경계다. 20명 모두 같은 점수를 준 이웃 샷 사이의 경계는
    점수로 보이지 않으므로 시간 규칙 floor(2k·fps) + 1로 채운다(규칙에서 2프레임 안에 실제 바뀜이 있으면 그것을 쓴다).
    이 규칙은 점수가 바뀌는 곳 대부분과 정확히 맞는다(data/tvsum_items.json의 grid_rule)."""
    n = len(rows[0])
    changes = sorted({i for r in rows for i in range(1, n) if r[i] != r[i - 1]})
    rule = [math.floor(2 * k * fps) + 1 for k in range(1, int(n / (2 * fps)) + 2)]
    rule = [b for b in rule if b < n]
    cs = set(changes)
    bounds = set(changes)
    for b in rule:
        if not any((b + d) in cs for d in (-2, -1, 0, 1, 2)):
            bounds.add(b)
    return [0] + sorted(bounds), changes, rule


def times_for(start, end, fps, n):
    """샷 [start, end)의 가운데 시각 c와 그림 4장의 프레임 번호(c ± 0.25, ± 0.75초, 0..n−1로 당김)."""
    c = (start + end) / 2 / fps
    idx = [min(max(round((c + off) * fps), 0), n - 1) for off in OFFSETS]
    return c, idx


def stage_items(args):
    chk = read(DERIVED / "tvsum_check.json")["videos"]
    info = load_info()
    anno, _ = load_anno()
    out, rule_stats = {"note": "점수 없음. 위치(샷)마다 프레임 범위, 가운데 시각, 그림 4장의 프레임 번호와 시각", "videos": {}}, {}
    for r in info:
        vid = r["video_id"]
        rows = anno[vid]
        rec = chk[vid]
        fps = exact_fps(rec)
        n = len(rows[0])
        nv = min(n, rec["packets"])                       # 그림은 영상에 실제로 있는 프레임에서만
        bounds, changes, rule = shot_bounds(rows, fps)
        ends = bounds[1:] + [n]
        lens = [e - s for s, e in zip(bounds, ends)]
        pos = []
        for k, (s, e) in enumerate(zip(bounds, ends)):
            c, idx = times_for(s, e, fps, nv)
            pos.append({"k": k, "start": s, "end": e, "center": round(c, 4), "frames": idx, "seconds": [round(i / fps, 4) for i in idx]})
        rs = set(rule)
        rule_stats[vid] = {"changes": len(changes), "changes_on_rule": sum(1 for c in changes if c in rs),
                           "changes_within_1": sum(1 for c in changes if any((c + d) in rs for d in (-1, 0, 1))),
                           "filled_from_rule": len(bounds) - 1 - len(changes)}
        out["videos"][vid] = {"category": r["category"], "fps": fps, "fps_shown": rec["fps"], "frames_anno": n, "frames_video": rec["packets"],
                              "width": rec["width"], "height": rec["height"], "positions": pos,
                              "shot_len_min": min(lens[:-1]) if len(lens) > 1 else lens[0], "shot_len_max": max(lens[:-1]) if len(lens) > 1 else lens[0],
                              "last_shot_len": lens[-1]}
    v = out["videos"].values()
    out["summary"] = {"videos": len(out["videos"]), "positions": sum(len(x["positions"]) for x in v), "frames_to_extract": sum(len({i for p in x["positions"] for i in p["frames"]}) for x in v),
                      "grid_rule": {"changes": sum(s["changes"] for s in rule_stats.values()),
                                    "changes_exactly_on_floor_2k_fps_plus_1": sum(s["changes_on_rule"] for s in rule_stats.values()),
                                    "changes_within_1_frame": sum(s["changes_within_1"] for s in rule_stats.values()),
                                    "boundaries_filled_from_rule": sum(s["filled_from_rule"] for s in rule_stats.values()),
                                    "videos_not_all_exact": {k: s for k, s in rule_stats.items() if s["changes_on_rule"] != s["changes"]}},
                      "irregular_shots": {k: [x["shot_len_min"], x["shot_len_max"]] for k, x in out["videos"].items()
                                          if x["shot_len_max"] - x["shot_len_min"] > 2},
                      "short_last_shots": {k: x["last_shot_len"] for k, x in out["videos"].items() if x["last_shot_len"] < 1.5 * x["fps"]}}
    write(HERE / "data" / "tvsum_items.json", out)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))


# =====================================================================  writer: 글쓴이 입력을 info.tsv로
def stage_writer(args):
    info = load_info()
    path = HERE / "writer" / "writer_input.json"
    old = read(path)
    prov = {x["id"]: x for x in old["tvsum"]}
    tv = []
    changed = []
    for r in info:
        rec = {"id": r["video_id"], "title": r["title"], "category": CAT[r["category"]], "duration": r["length"], "duration_seconds": secs(r["length"])}
        p = prov.get(r["video_id"])
        if p is None:
            raise SystemExit(f"Part 1 목록에 없는 영상: {r['video_id']}")
        diff = {k: [p[k], rec[k]] for k in ("title", "category", "duration") if p[k] != rec[k]}
        if diff:
            changed.append({"id": r["video_id"], **diff})
        tv.append(rec)
    if len(tv) != 50 or {x["id"] for x in tv} != set(prov):
        raise SystemExit("영상 목록이 Part 1과 다르다")
    out = {"status": "final",
           "status_note": "TVSum titles, categories and durations are from ydata-tvsum50-info.tsv (exact). Category names are from the dataset README.",
           "tvsum": tv, "minecraft": old["minecraft"]}
    write(path, out)
    write(DERIVED / "writer_input_changes.json", {"changed_entries": len(changed), "changes": changed})
    print(f"바뀐 항목 {len(changed)}개 (data/tvsum_src/derived/writer_input_changes.json)")


# =====================================================================  frames: 프레임 뽑기와 묶음
PACKS = HERE / "data" / "packs"


def fname(i):
    return f"{i:06d}.jpg"           # 프레임 번호(0부터, 영상을 푼 순서)


def _tar_bytes(files):
    """같은 내용이면 같은 바이트가 나오게(이름 순서, mtime 0, uid 0) tar를 만든다."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(files):
            data = files[name]
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _extract_video(job):
    """영상을 한 번 풀면서 고른 프레임 번호만 JPEG로 낸다. 긴 변이 MAX_SIDE를 넘을 때만 줄인다(키우지 않음)."""
    vid, idx, width, height = job
    import tempfile
    exe = ffmpeg()
    want = sorted(set(idx))
    files = {}
    # ffmpeg의 식은 eq() 60개쯤에서 막힌다(120개는 실패). 그래서 50개씩 나눠 매번 처음부터 푼다(n이 영상 처음부터의 번호로 남게, 찾아가기 없음)
    for c0 in range(0, len(want), 50):
        chunk = want[c0:c0 + 50]
        vf = "select='" + "+".join(f"eq(n\\,{i})" for i in chunk) + "'"
        if max(width, height) > MAX_SIDE:
            vf += f",scale='if(gte(iw,ih),{MAX_SIDE},-2)':'if(gte(iw,ih),-2,{MAX_SIDE})'"
        with tempfile.TemporaryDirectory(dir=HERE / "data") as tmp:
            cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(VIDEOS / f"{vid}.mp4"), "-map", "0:v:0",
                   "-vf", vf, "-fps_mode", "passthrough", "-q:v", JPEG_Q, "-f", "image2", f"{tmp}/%06d.jpg"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            got = sorted(Path(tmp).glob("*.jpg"))
            if p.returncode != 0 or len(got) != len(chunk):
                return vid, None, f"ffmpeg 종료 {p.returncode}, 낸 장수 {len(got)} / 원한 장수 {len(chunk)} (묶음 {c0}): {p.stderr[-300:]}"
            files.update({fname(i): g.read_bytes() for i, g in zip(chunk, got)})
    return vid, files, None


def stage_frames(args):
    import concurrent.futures as cf
    import time
    from PIL import Image
    items = read(HERE / "data" / "tvsum_items.json")["videos"]
    PACKS.mkdir(parents=True, exist_ok=True)
    jobs = [(vid, [i for p in v["positions"] for i in p["frames"]], v["width"], v["height"]) for vid, v in items.items()]
    t0 = time.time()
    summary, errors = {}, {}
    with cf.ProcessPoolExecutor(8) as ex:
        for vid, files, err in ex.map(_extract_video, jobs):
            if err:
                errors[vid] = err
                continue
            sizes = {Image.open(io.BytesIO(b)).size for b in files.values()}
            v = items[vid]
            files["_index.json"] = json.dumps({"video": vid, "fps": v["fps"], "frames": sorted(int(n[:6]) for n in files if n.endswith(".jpg"))}).encode()
            raw = _tar_bytes(files)
            (PACKS / f"{vid}.tar").write_bytes(raw)
            jpg = [len(b) for n, b in files.items() if n.endswith(".jpg")]
            summary[vid] = {"frames": len(jpg), "sizes": sorted(f"{w}x{h}" for w, h in sizes), "source": f"{v['width']}x{v['height']}",
                            "jpeg_bytes": sum(jpg), "tar_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            print(f"{vid}: {len(jpg)}장 {sum(jpg) / 1e6:.1f}MB {time.time() - t0:.0f}초", flush=True)
    s = summary.values()
    out = {"seconds": round(time.time() - t0, 1), "errors": errors, "packs": summary,
           "total": {"packs": len(summary), "frames": sum(x["frames"] for x in s), "jpeg_bytes": sum(x["jpeg_bytes"] for x in s),
                     "tar_bytes": sum(x["tar_bytes"] for x in s), "mean_jpeg_kb": round(sum(x["jpeg_bytes"] for x in s) / max(1, sum(x["frames"] for x in s)) / 1e3, 1),
                     "sizes": sorted({z for x in s for z in x["sizes"]}), "command": "ffmpeg -i <영상> -map 0:v:0 -vf select=eq(n,…) [+ 긴 변 640 넘을 때만 scale] -fps_mode passthrough -q:v 2"}}
    write(HERE / "data" / "frames_summary.json", out)
    print(json.dumps(out["total"], ensure_ascii=False), "오류", errors)


def load_packs(folder=None):
    folder = Path(folder or PACKS)
    frames = {}
    for p in sorted(folder.glob("*.tar")):
        with tarfile.open(p) as tar:
            frames[p.stem] = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    return frames


# =====================================================================  audio: 샷마다 소리 크기
def _loudness(job):
    import numpy as np
    vid, positions, fps = job
    exe = ffmpeg()
    sr = 16000
    p = subprocess.run([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(VIDEOS / f"{vid}.mp4"), "-map", "0:a:0",
                        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0:
        return vid, None, p.stderr.decode(errors="replace")[-300:]
    a = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    out = []
    for pos in positions:
        s, e = int(pos["start"] / fps * sr), int(pos["end"] / fps * sr)
        seg = a[s:e]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        out.append(round(20 * math.log10(max(rms, 1e-6)), 3))
    return vid, {"db": out, "samples": int(len(a)), "audio_seconds": round(len(a) / sr, 2)}, None


def stage_audio(args):
    import concurrent.futures as cf
    items = read(HERE / "data" / "tvsum_items.json")["videos"]
    res, errors = {}, {}
    with cf.ProcessPoolExecutor(8) as ex:
        for vid, r, err in ex.map(_loudness, [(vid, v["positions"], v["fps"]) for vid, v in items.items()]):
            if err:
                errors[vid] = err
            else:
                res[vid] = r
    write(HERE / "data" / "tvsum_loudness.json", {"what": "샷 구간의 소리 RMS(dBFS, 모노 16kHz). 소리가 없는 구간은 −120dB", "errors": errors, "videos": res})
    print("소리 크기:", len(res), "영상, 오류", errors)


# =====================================================================  requests: Modal에 보낼 요청 글 (점수 없음)
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def check_criteria(crit, tv_ids):
    """README 4절의 형식 검사. 틀린 곳의 목록(빈 목록이면 통과). 내용은 보지 않는다."""
    errs = []
    if not isinstance(crit, dict) or not isinstance(crit.get("videos"), dict):
        return ["맨 위에 videos 객체가 없다"]
    for k in ("writer_model", "written_at"):
        if not crit.get(k):
            errs.append(f"맨 위의 {k}가 없다")
    want = set(tv_ids) | {"minecraft"}
    got = set(crit["videos"])
    if want - got:
        errs.append(f"빠진 영상: {sorted(want - got)}")
    if got - want:
        errs.append(f"모르는 영상: {sorted(got - want)}")
    for vid, e in crit["videos"].items():
        qs = e.get("questions") if isinstance(e, dict) else None
        if not isinstance(qs, dict):
            errs.append(f"{vid}: questions가 없다")
            continue
        if not 3 <= len(qs) <= 6:
            errs.append(f"{vid}: 질문 수 {len(qs)} (3~6이어야 함)")
        for key, q in qs.items():
            where = f"{vid}.{key}"
            if not KEY_RE.match(key) or key == "worth":
                errs.append(f"{where}: 키는 영어 소문자·숫자·밑줄 40자 이하, 'worth'는 쓸 수 없다")
            if not isinstance(q, dict) or q.get("type") not in ("noul", "score"):
                errs.append(f"{where}: type은 noul 또는 score")
                continue
            ins = q.get("instructions", "")
            texts = [ins]
            if not isinstance(ins, str) or not ins.strip() or len(ins) > 300:
                errs.append(f"{where}: instructions가 비었거나 300자를 넘는다")
            if "`frames`" not in ins:
                errs.append(f"{where}: instructions가 `frames`를 가리키지 않는다")
            for bad in ("`viewer_request`", "`game`", "`about`"):
                if bad in ins:
                    errs.append(f"{where}: instructions가 {bad}를 가리킨다")
            c = q.get("criteria")
            if q["type"] == "noul":
                if not isinstance(c, dict) or set(c) != {"true", "false"} or not all(isinstance(x, str) and x.strip() for x in c.values()):
                    errs.append(f"{where}: noul의 criteria는 true와 false 설명 둘")
                else:
                    texts += list(c.values())
            else:
                if not isinstance(c, list) or not 3 <= len(c) <= 5 or not all(isinstance(x, str) and x.strip() for x in c):
                    errs.append(f"{where}: score의 criteria는 단계 설명 3~5개의 목록")
                else:
                    texts += c
            for t in texts:
                if isinstance(t, str) and (len(t) > 300 or any(ord(ch) > 126 or (ord(ch) < 32) for ch in t)):
                    errs.append(f"{where}: 300자를 넘거나 영어(ASCII) 밖의 글자가 있다: {t[:60]!r}")
                    break
    return errs


def rate_tv(q_req, px4):
    return min(REQ_CAP, PX_READS / (q_req * px4))


def build_requests(crit, write_to=None):
    """글쓴이 기준 + 제목 → 요청 글. 점수·URL·길이 같은 메타데이터는 넣지 않는다(제목은 막연한 질문의 글이라 들어간다)."""
    items = read(HERE / "data" / "tvsum_items.json")["videos"]
    fsum = read(HERE / "data" / "frames_summary.json")
    titles = {r["video_id"]: r["title"] for r in load_info()}
    errs = check_criteria(crit, list(items))
    if errs:
        raise SystemExit("criteria.json 형식 오류:\n" + "\n".join(errs))
    tv = {}
    plan_tv = 0.0
    for vid, it in items.items():
        qs = {"worth": {"type": "noul", "instructions": Q_WORTH}}
        qs.update(crit["videos"][vid]["questions"])
        w, h = (int(x) for x in fsum["packs"][vid]["sizes"][0].split("x"))
        rate = rate_tv(len(qs), 4 * w * h)
        plan_tv += len(it["positions"]) / rate
        tv[vid] = {"viewer_request": TV_REQUEST.format(title=titles[vid]), "questions": qs,
                   "positions": [{"k": p["k"], "frames": [fname(i) for i in p["frames"]], "seconds": p["seconds"]} for p in it["positions"]]}
    mcq = crit["videos"]["minecraft"]["questions"]
    plan_mc = 3600 / min(REQ_CAP, PX_READS / (len(mcq) * 4 * 640 * 360))
    out = {"note": "Modal 실행에 필요한 요청 글. TVSum 점수, URL, 길이는 없다. 제목은 막연한 질문의 viewer_request 글로만 들어 있다",
           "about": ABOUT, "tvsum": {"videos": tv}, "minecraft": {"game": MC_GAME, "viewer_request": MC_REQUEST, "questions": mcq},
           "packs": {vid: fsum["packs"][vid]["sha256"] for vid in items},
           "plan": {"sanity": {"requests": 200, "est_seconds": round(200 / 16, 1)},
                    "tvsum": {"requests": sum(len(x["positions"]) for x in tv.values()), "est_seconds": round(plan_tv, 1)},
                    "minecraft": {"requests": 3600, "est_seconds": round(plan_mc, 1)},
                    "model": "초당 요청 = min(16, 5,500만 ÷ (요청당 질문 수 × 4장의 화소 합)) (README 11절)"}}
    if write_to:
        write(write_to, out)
    return out


def stage_requests(args):
    crit = read(HERE / "writer" / "criteria.json")
    out = build_requests(crit, write_to=HERE / "data" / "requests.json")
    print(json.dumps(out["plan"], ensure_ascii=False, indent=1))


# =====================================================================  dryrun: 모든 본문을 만들어 세기
def tv_body(vreq, pos, get, vid):
    """TVSum 요청 본문(판정 모델(H2O)의 /v1/systemone 형식). get(영상, 파일 이름) → JPEG 바이트."""
    frames = [{"seconds": s, "frame": uri(get(vid, n))} for n, s in zip(pos["frames"], pos["seconds"])]
    return {"state": {"viewer_request": vreq["viewer_request"], "about": ABOUT, "frames": frames}, "questions": vreq["questions"]}


def stage_dryrun(args):
    import statistics
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HH))
    sys.path.insert(0, str(ROOT / "experiments" / "charades_dense"))
    import hh_prepare as HP   # noqa: E402  (읽기만)
    import cd_layouts as L    # noqa: E402  (그림 크기 맞춤·글 토큰 어림 상수만)
    from PIL import Image
    req = read(args.requests or HERE / "data" / "requests.json")
    packs = load_packs()
    get = lambda v, n: packs[v][n]   # noqa: E731
    calib = read(HH / "data" / "dryrun.json")["calibration"]["median_ratio"]
    rows = {"sanity": [], "tvsum": [], "minecraft": []}
    # 정상 확인 200건: 프레임은 Modal 볼륨(mf-mc-dense)에만 있어, highlight_h2o의 미리 세기처럼 같은 크기(640×360)의 회색 그림으로 만든다
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (128, 128, 128)).save(buf, "JPEG", quality=90)
    gray = buf.getvalue()
    for vid in HP.SANITY_VIDEOS:
        for k in range(HP.SANITY_K):
            body = HP.sanity_body(lambda v, n: gray, vid, k, 6000)
            est, proc, img = HP.estimate(body, [(640, 360)] * 4, L)
            rows["sanity"].append({"q": len(body["questions"]), "img": len(body["state"]["frames"]), "px": 4 * 640 * 360, "est": est * calib, "proc": proc,
                                   "bytes": len(json.dumps(body))})
    sample = None
    for vid, vreq in req["tvsum"]["videos"].items():
        for pos in vreq["positions"]:
            body = tv_body(vreq, pos, get, vid)
            sizes = [Image.open(io.BytesIO(get(vid, n))).size for n in pos["frames"]]
            est, proc, img = HP.estimate(body, sizes, L)
            rows["tvsum"].append({"q": len(body["questions"]), "img": len(body["state"]["frames"]), "px": sum(w * h for w, h in sizes), "est": est * calib,
                                  "proc": proc, "bytes": len(json.dumps(body)), "max_px": max(w * h for w, h in sizes)})
            if sample is None:
                sample = (vid, pos, body)
    mc_packs = HP.load_packs()
    mc_items = HP.load_items()["full"]
    mc_sample = None
    for it in mc_items:
        body = HP.body(it, lambda v, n: mc_packs[v][n])
        body["questions"] = req["minecraft"]["questions"]
        est, proc, img = HP.estimate(body, [(640, 360)] * 4, L)
        rows["minecraft"].append({"q": len(body["questions"]), "img": len(body["state"]["frames"]), "px": 4 * 640 * 360, "est": est * calib, "proc": proc,
                                  "bytes": len(json.dumps(body))})
        if mc_sample is None:
            mc_sample = (it, body)
    out = {"calibration_from_highlight_h2o": calib, "arms": {}}
    for arm, rs in rows.items():
        secs_ = sum(1 / min(REQ_CAP, PX_READS / (r["q"] * r["px"])) for r in rs)
        out["arms"][arm] = {"requests": len(rs), "images": sum(r["img"] for r in rs), "questions": sum(r["q"] for r in rs),
                            "questions_per_request": sorted({r["q"] for r in rs}), "max_image_pixels": max(r.get("max_px", r["px"] // 4) for r in rs),
                            "pixels_per_request": sorted({r["px"] for r in rs}), "input_tokens_est_median": round(statistics.median(r["est"] for r in rs)),
                            "input_tokens_est_total": round(sum(r["est"] for r in rs)),
                            "processed_tokens_est_median": round(statistics.median(r["proc"] for r in rs)), "body_bytes_median": statistics.median(r["bytes"] for r in rs),
                            "body_bytes_total": sum(r["bytes"] for r in rs), "est_seconds": round(secs_, 1)}
    a = out["arms"]
    ask = a["tvsum"]["est_seconds"] + a["minecraft"]["est_seconds"] + a["sanity"]["est_seconds"]
    out["time"] = {"ask_seconds": round(ask, 1), "fast_seconds": round(169 + 20 + ask), "slow_seconds": round(286 + 20 + 1.3 * ask),
                   "fast_usd": round((169 + 20 + ask) * PRICE, 3), "slow_usd": round((286 + 20 + 1.3 * ask) * PRICE, 3),
                   "timeout_seconds": 2400, "cap_usd": round(2400 * PRICE, 3), "price_per_second": round(PRICE, 8)}
    write(Path(args.out or HERE / "data") / "dryrun.json", out)
    sd = Path(args.out or HERE / "samples")
    for name, body, label in (("tvsum", sample[2], f"{sample[0]}_k{sample[1]['k']}"), ("minecraft", mc_sample[1], mc_sample[0]["id"].replace("@", "_t"))):
        b = json.loads(json.dumps(body))
        for fr in b["state"]["frames"]:
            fr["frame"] = f"<image: JPEG, {len(base64.b64decode(fr['frame'].split(',', 1)[1]))} bytes>"
        write(sd / f"{name}_{label}.json", b)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--requests", default=None, help="dryrun: 요청 글 파일(기본 data/requests.json)")
    ap.add_argument("--out", default=None, help="dryrun: 결과와 보기 본문을 쓸 폴더(기본 data/, samples/)")
    args = ap.parse_args()
    fn = globals().get(f"stage_{args.stage}")
    if fn is None:
        raise SystemExit(f"단계 없음: {args.stage}")
    fn(args)


if __name__ == "__main__":
    main()
