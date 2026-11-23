"""highlight_probe 준비 (이 Mac, CPU, 돈 없음). GPU, Modal, LLM 호출이 없다.

단계
  check     : probe/probe_questions.json의 형식 검사
  positions : 영상마다 훑어볼 위치를 고른다 → data/probe_positions.json
  dryrun    : 모든 훑어보기 본문을 실제 프레임으로 만들어 세고 시간과 비용을 어림한다 → data/dryrun.json, samples/
              본 실행(글쓴이 팔 여럿)의 경우별 어림도 함께 낸다
  packs     : 훑어볼 TVSum 742곳에 필요한 프레임만 영상마다 tar 하나로 → data/probe_packs/<id>.tar, data/probe_frames_summary.json
  requests  : Modal에 올릴 요청 글(보내는 질문, about, 위치마다 프레임 이름과 시각, 묶음 sha256, 예상 초) → data/probe_requests.json
              제목, 범주, 길이, 점수는 넣지 않는다
  profile   : 훑어보기 답 → 영상마다의 요약 글. --synth면 가짜 답으로 모양만 본다(파일을 쓰지 않는다)
              --answers에 답 파일을 여럿 줄 수 있다(results/tvsum.json results/minecraft.json). --out으로 쓸 곳을 바꾼다(기본 data/profiles.json)

실행: uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_probe/hp_prepare.py <단계>

다른 실험 폴더(highlight_auto, highlight_h2o, charades_dense)는 경로로 읽기만 한다. 거기에 파일을 쓰지 않는다.
TVSum의 평가자 점수는 읽지 않는다. 제목·범주도 훑어보기 본문에 넣지 않는다.
"""
import sys

sys.dont_write_bytecode = True          # 다른 실험 폴더에 __pycache__를 만들지 않는다
import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import tarfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HA = ROOT / "experiments" / "highlight_auto"
HH = ROOT / "experiments" / "highlight_h2o"
for p in (HA, HH, ROOT / "experiments" / "charades_dense"):
    sys.path.insert(0, str(p))
import ha_prepare as P   # noqa: E402  (읽기만: 경로, 그림 이름, 값·속도 상수)
import hh_prepare as HP  # noqa: E402  (읽기만: 마인크래프트 위치, 그림 이름, 토큰 어림)

QUESTIONS = HERE / "probe" / "probe_questions.json"
SHARE, K_MIN, K_MAX = 0.10, 12, 30      # 영상마다 위치의 10%, 적어도 12곳, 많아도 30곳
READY_FAST, READY_SLOW, CHECK = 169, 286, 20   # highlight_auto 11절과 같은 수(초)
SANITY_SECONDS = 200 / 16
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
TOP_OPTIONS, MIN_SHARE = 4, 0.05


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


# =====================================================================  check
def check_questions(spec):
    """형식만 본다. 틀린 곳의 목록(빈 목록이면 통과)."""
    errs = []

    def text_ok(t):
        return isinstance(t, str) and t.strip() and len(t) <= 300 and all(32 <= ord(c) <= 126 for c in t)
    qs = spec.get("questions")
    if not isinstance(qs, dict) or not qs:
        return ["questions가 없다"]
    if "`frames`" not in spec.get("state", {}).get("about", ""):
        errs.append("state.about이 `frames`를 설명하지 않는다")
    for key, q in qs.items():
        if not KEY_RE.match(key):
            errs.append(f"{key}: 키는 영어 소문자·숫자·밑줄 40자 이하")
        if q.get("type") not in ("choice", "noul", "score"):
            errs.append(f"{key}: type은 choice, noul, score 중 하나")
            continue
        if not text_ok(q.get("instructions")) or "`frames`" not in q["instructions"]:
            errs.append(f"{key}: instructions는 영어(ASCII) 300자 이하이고 `frames`를 가리켜야 한다")
        if not text_ok(q.get("heading")):
            errs.append(f"{key}: heading이 없다")
        c = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= 16:
                errs.append(f"{key}: choice의 선택지는 2~16개 (모델 문서: 16개쯤까지 잘 가른다)")
                continue
            if set(q.get("labels", {})) != set(c):
                errs.append(f"{key}: labels의 키가 선택지와 다르다")
            for k, v in list(c.items()) + list(q.get("labels", {}).items()):
                if not KEY_RE.match(k) or not text_ok(v):
                    errs.append(f"{key}.{k}: 선택지 이름이나 설명이 형식 밖이다")
        elif q["type"] == "noul":
            if not isinstance(c, dict) or set(c) != {"true", "false"} or not all(text_ok(v) for v in c.values()):
                errs.append(f"{key}: noul의 criteria는 true와 false 설명 둘")
        else:
            if not isinstance(c, list) or not 3 <= len(c) <= 5 or not all(text_ok(v) for v in c):
                errs.append(f"{key}: score의 criteria는 단계 설명 3~5개")
    return errs


def sent_questions(spec):
    """판정 모델에 보내는 모양(type, instructions, criteria만)."""
    return {k: {x: q[x] for x in ("type", "instructions", "criteria")} for k, q in spec["questions"].items()}


def stage_check(args):
    spec = read(QUESTIONS)
    errs = check_questions(spec)
    if errs:
        raise SystemExit("probe_questions.json 형식 오류:\n" + "\n".join(errs))
    qs = spec["questions"]
    kinds = {t: sum(q["type"] == t for q in qs.values()) for t in ("choice", "noul", "score")}
    print(f"형식 검사 통과: 질문 {len(qs)}개 {kinds}, 선택지 수 {[len(q['criteria']) for q in qs.values() if q['type'] == 'choice']}")


# =====================================================================  positions
def pick(n):
    """n곳 중 고르게 k곳. k = 위치의 10%(반올림), 12~30 사이. i번째는 ⌊(i + 0.5) × n ÷ k⌋."""
    k = min(n, max(K_MIN, min(K_MAX, int(SHARE * n + 0.5))))
    return sorted({int((i + 0.5) * n / k) for i in range(k)})


def stage_positions(args):
    tv = read(HA / "data" / "tvsum_items.json")["videos"]
    out_tv = {}
    for vid, it in tv.items():
        idx = pick(len(it["positions"]))
        out_tv[vid] = {"n": len(it["positions"]), "k": [it["positions"][i]["k"] for i in idx]}
    mc_by = {}
    for it in HP.load_items()["full"]:
        mc_by.setdefault(it["video"], []).append(it)
    out_mc = {}
    for video, its in mc_by.items():
        idx = pick(len(its))
        out_mc[video] = {"n": len(its), "ids": [its[i]["id"] for i in idx], "t": [its[i]["t"] for i in idx]}
    tot_tv, tot_mc = sum(len(v["k"]) for v in out_tv.values()), sum(len(v["ids"]) for v in out_mc.values())
    ks = sorted(len(v["k"]) for v in out_tv.values())
    out = {"rule": f"영상마다 위치의 {SHARE:.0%}(반올림), 적어도 {K_MIN}곳, 많아도 {K_MAX}곳. 처음부터 끝까지 고르게: i번째는 floor((i + 0.5) × n ÷ k)번째 위치",
           "totals": {"tvsum_videos": len(out_tv), "tvsum_probe": tot_tv, "tvsum_all": sum(v["n"] for v in out_tv.values()),
                      "tvsum_k_min_median_max": [ks[0], statistics.median(ks), ks[-1]],
                      "minecraft_videos": len(out_mc), "minecraft_probe": tot_mc, "minecraft_all": sum(v["n"] for v in out_mc.values())},
           "tvsum": out_tv, "minecraft": out_mc}
    write(HERE / "data" / "probe_positions.json", out)
    print(json.dumps(out["totals"], ensure_ascii=False))


# =====================================================================  본문
def tar_get(path, names):
    with tarfile.open(path) as tar:
        return {n: tar.extractfile(tar.getmember(n)).read() for n in names}


def probe_body(about, questions, seconds, jpegs):
    """훑어보기 요청 본문(판정 모델(H2O)의 /v1/systemone 형식). 상태에는 그림 설명 한 줄과 그림 4장뿐이다(제목, 범주, 게임 이름 없음)."""
    return {"state": {"about": about, "frames": [{"seconds": s, "frame": P.uri(b)} for s, b in zip(seconds, jpegs)]}, "questions": questions}


def rate(q, px4):
    """초당 요청 = min(16, 5,500만 ÷ (요청당 질문 수 × 4장의 화소 합)) (highlight_auto README 11절의 식)."""
    return min(P.REQ_CAP, P.PX_READS / (q * px4))


def window(ask):
    fast, slow = READY_FAST + CHECK + ask, READY_SLOW + CHECK + 1.3 * ask
    return {"ask_seconds": round(ask, 1), "fast_seconds": round(fast), "slow_seconds": round(slow),
            "fast_usd": round(fast * P.PRICE, 2), "slow_usd": round(slow * P.PRICE, 2)}


# =====================================================================  dryrun
def stage_dryrun(args):
    import cd_layouts as L   # noqa: E402  (그림 크기 맞춤과 글 토큰 어림 상수만)
    from PIL import Image
    spec = read(QUESTIONS)
    errs = check_questions(spec)
    if errs:
        raise SystemExit("probe_questions.json 형식 오류:\n" + "\n".join(errs))
    qs, about = sent_questions(spec), spec["state"]["about"]
    pos = read(HERE / "data" / "probe_positions.json")
    calib = read(HH / "data" / "dryrun.json")["calibration"]["median_ratio"]
    tv_items = read(HA / "data" / "tvsum_items.json")["videos"]
    rows = {"tvsum": [], "minecraft": []}
    samples = {}
    for vid, pv in pos["tvsum"].items():
        by_k = {p["k"]: p for p in tv_items[vid]["positions"]}
        need = sorted({P.fname(i) for k in pv["k"] for i in by_k[k]["frames"]})
        got = tar_get(P.PACKS / f"{vid}.tar", need)
        for k in pv["k"]:
            p = by_k[k]
            jp = [got[P.fname(i)] for i in p["frames"]]
            body = probe_body(about, qs, p["seconds"], jp)
            sizes = [Image.open(io.BytesIO(b)).size for b in jp]
            est, proc, _ = HP.estimate(body, sizes, L)
            px = sum(w * h for w, h in sizes)
            rows["tvsum"].append({"px": px, "est": est * calib, "proc": proc, "bytes": len(json.dumps(body)), "sec": 1 / rate(len(qs), px)})
            samples.setdefault("tvsum", (f"{vid}_k{k}", body))
    mc_items = {it["id"]: it for it in HP.load_items()["full"]}
    for video, pv in pos["minecraft"].items():
        its = [mc_items[i] for i in pv["ids"]]
        got = tar_get(HH / "data" / "packs" / f"{video}.tar", sorted({HP.fname(t) for it in its for t in it["times"]}))
        for it in its:
            jp = [got[HP.fname(t)] for t in it["times"]]
            body = probe_body(about, qs, it["times"], jp)
            sizes = [Image.open(io.BytesIO(b)).size for b in jp]
            est, proc, _ = HP.estimate(body, sizes, L)
            px = sum(w * h for w, h in sizes)
            rows["minecraft"].append({"px": px, "est": est * calib, "proc": proc, "bytes": len(json.dumps(body)), "sec": 1 / rate(len(qs), px)})
            samples.setdefault("minecraft", (it["id"].replace("@", "_t"), body))
    out = {"what": "훑어보기 실행(A 단계)의 미리 세기. 모든 본문을 실제 프레임으로 만들었다. 식과 값은 highlight_auto README 11절과 같다",
           "questions_per_request": len(qs), "calibration_from_highlight_h2o": calib, "price_per_second": round(P.PRICE, 8), "arms": {}}
    for arm, rs in rows.items():
        out["arms"][arm] = {"requests": len(rs), "images": 4 * len(rs), "question_reads": len(qs) * len(rs), "pixels_per_request": sorted({r["px"] for r in rs}),
                            "input_tokens_est_median": round(statistics.median(r["est"] for r in rs)), "input_tokens_est_total": round(sum(r["est"] for r in rs)),
                            "processed_tokens_est_median": round(statistics.median(r["proc"] for r in rs)), "body_bytes_total": sum(r["bytes"] for r in rs),
                            "est_seconds": round(sum(r["sec"] for r in rs), 1)}
    ask = sum(a["est_seconds"] for a in out["arms"].values()) + SANITY_SECONDS
    out["probe_run"] = {**window(ask), "includes": "정상 확인 200건(12.5초), 준비 169~286초, 형식 확인 20초", "proposed_timeout_seconds": 900,
                        "cap_usd": round(900 * P.PRICE, 2)}

    # 본 실행(C 단계)의 경우별 어림: 글쓴이 팔마다 질문을 더 묻는다. 그림은 팔이 몇이든 같은 4장이다
    fsum = read(HA / "data" / "frames_summary.json")["packs"]
    before = {v: len(e["questions"]) for v, e in read(HA / "writer" / "criteria.json")["videos"].items()}

    def main_run(tv_arms, tv_q, mc_arms, mc_q):
        tv = 0.0
        for vid, it in tv_items.items():
            w, h = (int(x) for x in fsum[vid]["sizes"][0].split("x"))
            q = tv_arms * (before[vid] if tv_q == "before" else tv_q)
            tv += len(it["positions"]) / rate(q, 4 * w * h) if q else 0.0
        mc = 3600 / rate(mc_arms * mc_q, 4 * 640 * 360) if mc_arms else 0.0
        w_ = window(tv + mc + SANITY_SECONDS)
        return {"tvsum_seconds": round(tv), "minecraft_seconds": round(mc), **w_,
                "split_needed": w_["slow_seconds"] > 2300, "note": "느린 경우가 2,300초를 넘으면 TVSum과 마인크래프트를 두 함수로 나눈다(준비 시간이 한 번 더 든다)"}
    out["main_run_scenarios"] = {
        "tvsum_only_3arms": {"what": "TVSum만: 팔 셋(T, TP, P), 팔마다 질문 수는 지난번과 같다고 봄(3~4개)", **main_run(3, "before", 0, 0)},
        "base": {"what": "기본안: TVSum 팔 셋(지난번 질문 수) + 마인크래프트 팔 둘(GP, P) × 질문 6개", **main_run(3, "before", 2, 6)},
        "with_mc_control": {"what": "대조 포함: TVSum 팔 셋 + 마인크래프트 팔 셋(G, GP, P) × 질문 6개", **main_run(3, "before", 3, 6)},
        "upper": {"what": "상한: 모든 팔이 질문 6개를 다 씀", **main_run(3, 6, 3, 6)}}
    write(HERE / "data" / "dryrun.json", out)
    for name, (label, body) in samples.items():
        b = json.loads(json.dumps(body))
        for fr in b["state"]["frames"]:
            fr["frame"] = "<image: JPEG>"
        write(HERE / "samples" / f"probe_{name}_{label}.json", b)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# =====================================================================  packs: 훑어볼 TVSum 프레임만 묶기
PROBE_PACKS = HERE / "data" / "probe_packs"
REQUESTS = HERE / "data" / "probe_requests.json"
FRAMES_SUMMARY = HERE / "data" / "probe_frames_summary.json"


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def stage_packs(args):
    """훑어볼 742곳의 그림 4장만 영상마다 tar 하나로 묶는다. 원본은 highlight_auto의 묶음(경로로 읽기만, sha256을 그쪽 요약과 견줌).
    이름은 ha_prepare.fname, tar는 ha_prepare._tar_bytes(이름 순서, mtime 0이라 같은 내용이면 같은 바이트). 묶음 안은 JPEG뿐이다."""
    from PIL import Image
    pos = read(HERE / "data" / "probe_positions.json")["tvsum"]
    tv = read(HA / "data" / "tvsum_items.json")["videos"]
    src_sum = read(HA / "data" / "frames_summary.json")["packs"]
    PROBE_PACKS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for vid, pv in pos.items():
        raw_src = (P.PACKS / f"{vid}.tar").read_bytes()
        if sha256(raw_src) != src_sum[vid]["sha256"]:
            raise SystemExit(f"{vid}: 원본 묶음의 sha256이 highlight_auto/data/frames_summary.json과 다르다")
        by_k = {p["k"]: p for p in tv[vid]["positions"]}
        need = sorted({P.fname(i) for k in pv["k"] for i in by_k[k]["frames"]})
        with tarfile.open(fileobj=io.BytesIO(raw_src)) as tar:
            files = {n: tar.extractfile(tar.getmember(n)).read() for n in need}
        raw = P._tar_bytes(files)
        (PROBE_PACKS / f"{vid}.tar").write_bytes(raw)
        summary[vid] = {"positions": len(pv["k"]), "frames": len(files), "sizes": sorted({"%dx%d" % Image.open(io.BytesIO(b)).size for b in files.values()}),
                        "jpeg_bytes": sum(len(b) for b in files.values()), "tar_bytes": len(raw), "sha256": sha256(raw)}
    extra = sorted(p.name for p in PROBE_PACKS.glob("*") if p.suffix != ".tar" or p.stem not in summary)
    if extra:
        raise SystemExit(f"data/probe_packs/에 다른 파일이 있다(올리기 전에 지운다): {extra}")
    s = summary.values()
    out = {"what": "훑어볼 TVSum 위치(data/probe_positions.json)의 그림 4장만 묶은 것. Modal 볼륨 mf-highlight-probe의 packs/에 올린다",
           "source": "../highlight_auto/data/packs/<id>.tar (sha256을 ../highlight_auto/data/frames_summary.json과 견줌). 프레임은 새로 뽑지 않았다",
           "packs": summary,
           "total": {"packs": len(summary), "positions": sum(x["positions"] for x in s), "frames": sum(x["frames"] for x in s),
                     "jpeg_bytes": sum(x["jpeg_bytes"] for x in s), "tar_bytes": sum(x["tar_bytes"] for x in s),
                     "sizes": sorted({z for x in s for z in x["sizes"]})}}
    write(FRAMES_SUMMARY, out)
    print(json.dumps(out["total"], ensure_ascii=False))


# =====================================================================  requests: Modal에 올릴 요청 글
FORBIDDEN_KEYS = {"title", "category", "viewer_request", "game", "duration", "duration_seconds", "frames_anno", "fps", "scores", "score_mean"}


def forbidden_keys(obj):
    """요청 글 안에 있으면 안 되는 칸 이름(제목, 범주, 길이, 점수 등)의 목록. 값이 아니라 칸 이름만 본다."""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                found.add(k)
            found |= forbidden_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= forbidden_keys(v)
    return found


def stage_requests(args):
    """요청 글: 보내는 질문(type, instructions, criteria), about, 위치마다 프레임 이름과 시각, TVSum 묶음의 sha256, 묶음마다 예상 초.
    예상 초는 dryrun과 같은 식(실제 그림 크기)이다."""
    from PIL import Image
    spec = read(QUESTIONS)
    errs = check_questions(spec)
    if errs:
        raise SystemExit("probe_questions.json 형식 오류:\n" + "\n".join(errs))
    qs, about = sent_questions(spec), spec["state"]["about"]
    pos = read(HERE / "data" / "probe_positions.json")
    fsum = read(FRAMES_SUMMARY)["packs"]
    tv_items = read(HA / "data" / "tvsum_items.json")["videos"]

    def px4(files, names):
        return sum(w * h for w, h in (Image.open(io.BytesIO(files[n])).size for n in names))

    tv, sec_tv = {}, 0.0
    for vid, pv in pos["tvsum"].items():
        by_k = {p["k"]: p for p in tv_items[vid]["positions"]}
        ps = [{"k": k, "frames": [P.fname(i) for i in by_k[k]["frames"]], "seconds": by_k[k]["seconds"]} for k in pv["k"]]
        raw = (PROBE_PACKS / f"{vid}.tar").read_bytes()
        if sha256(raw) != fsum[vid]["sha256"]:
            raise SystemExit(f"{vid}: data/probe_packs의 묶음이 probe_frames_summary.json과 다르다(packs 단계를 다시 돌린다)")
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        for p in ps:
            sec_tv += 1 / rate(len(qs), px4(files, p["frames"]))
        tv[vid] = {"positions": ps}
    mc_items = {it["id"]: it for it in HP.load_items()["full"]}
    mc, sec_mc = {}, 0.0
    for video, pv in pos["minecraft"].items():
        its = [mc_items[i] for i in pv["ids"]]
        ps = [{"id": it["id"], "t": it["t"], "frames": [HP.fname(t) for t in it["times"]], "seconds": it["times"]} for it in its]
        files = tar_get(HH / "data" / "packs" / f"{video}.tar", sorted({n for p in ps for n in p["frames"]}))
        for p in ps:
            sec_mc += 1 / rate(len(qs), px4(files, p["frames"]))
        mc[video] = {"positions": ps}
    out = {"note": "highlight_probe A 단계(훑어보기)의 요청 글. Modal 볼륨 mf-highlight-probe에 올린다. 제목, 범주, 길이, 점수는 없다. "
                   "본문의 상태는 about과 frames뿐이다(hp_prepare.probe_body). 마인크래프트 그림은 볼륨 mf-highlight-h2o의 묶음에서 읽는다",
           "questions_file_sha256": sha256(QUESTIONS.read_bytes()), "about": about, "questions": qs,
           "tvsum": {"videos": tv}, "minecraft": {"videos": mc},
           "packs": {vid: fsum[vid]["sha256"] for vid in tv},
           "plan": {"sanity": {"requests": 200, "est_seconds": round(SANITY_SECONDS, 1)},
                    "tvsum": {"requests": sum(len(x["positions"]) for x in tv.values()), "est_seconds": round(sec_tv, 1)},
                    "minecraft": {"requests": sum(len(x["positions"]) for x in mc.values()), "est_seconds": round(sec_mc, 1)},
                    "model": "초당 요청 = min(16, 5,500만 ÷ (요청당 질문 수 × 4장의 화소 합)) (highlight_auto README 11절, data/dryrun.json과 같은 식)"}}
    bad = forbidden_keys(out)
    if bad:
        raise SystemExit(f"요청 글에 넣으면 안 되는 칸이 있다: {sorted(bad)}")
    write(REQUESTS, out)
    dry = read(HERE / "data" / "dryrun.json")["arms"]
    print(json.dumps(out["plan"], ensure_ascii=False))
    print(f"dryrun.json의 어림 초: tvsum {dry['tvsum']['est_seconds']}, minecraft {dry['minecraft']['est_seconds']}. 크기 {REQUESTS.stat().st_size:,}바이트")


# =====================================================================  profile: 답 → 요약 글
def profile_text(spec, answers, duration_seconds):
    """한 영상의 훑어보기 답(시간 순서, 위치마다 {질문: 답}) → 글쓴이에게 줄 영어 요약. 코드만 쓴다(LLM 없음).
    답의 모양은 서버가 주는 그대로다: choice {"probabilities": {...}}, noul {"noul": p}, score {"score": x, "probabilities": {...}}."""
    n = len(answers)
    mm, ss = divmod(int(round(duration_seconds)), 60)
    lines = [f"Probe summary. A judging model looked at {n} moments spread evenly over this {mm}:{ss:02d} video. Each moment is 4 frames covering 1.5 seconds, "
             "without sound. The numbers are its average probabilities; treat them as rough hints, not facts."]

    def mean_probs(key, rows):
        opts = list(spec["questions"][key]["criteria"])
        return {o: sum(r[key]["probabilities"][o] for r in rows) / len(rows) for o in opts}

    def top(key, rows):
        m = mean_probs(key, rows)
        order = sorted(m, key=lambda o: -m[o])
        keep = [o for o in order if m[o] >= MIN_SHARE][:TOP_OPTIONS] or order[:1]
        lab = spec["questions"][key]["labels"]
        return "; ".join(f"{lab[o]} {m[o]:.2f}" for o in keep), order[0]

    for key, q in spec["questions"].items():
        if q["type"] == "choice":
            lines.append(f"- {q['heading']}: {top(key, answers)[0]}")
        elif q["type"] == "noul":
            ps = [a[key]["noul"] for a in answers]
            lines.append(f"- {q['heading']}: average {sum(ps) / n:.2f}; likely at {sum(p >= 0.5 for p in ps)} of {n} moments")
        else:
            levels = len(q["criteria"]) - 1
            sc = [a[key]["score"] for a in answers]
            high = sum(int(max(a[key]["probabilities"], key=lambda k: a[key]["probabilities"][k])) >= levels - 1 for a in answers)
            lines.append(f"- {q['heading']}: average level {sum(sc) / n:.1f} of {levels}; the two highest levels are the most likely answer at {high} of {n} moments")
    if "activity" in spec["questions"] and n >= 6:
        lab = spec["questions"]["activity"]["labels"]
        thirds = [answers[: n // 3], answers[n // 3: n - n // 3], answers[n - n // 3:]]
        tops = [lab[top("activity", t)[1]] for t in thirds]
        lines.append("- Over time (beginning / middle / end), main activity: " + (" / ".join(tops) if len(set(tops)) > 1 else f"{tops[0]} throughout"))
    return "\n".join(lines)


def synth_answers(spec, n, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = {}
        for key, q in spec["questions"].items():
            if q["type"] == "choice":
                p = rng.dirichlet([0.4] * len(q["criteria"]))
                a[key] = {"probabilities": dict(zip(q["criteria"], map(float, p)))}
            elif q["type"] == "noul":
                a[key] = {"noul": float(rng.beta(0.6, 1.8))}
            else:
                p = rng.dirichlet([1.0] * len(q["criteria"]))
                a[key] = {"score": float(sum(i * x for i, x in enumerate(p))), "probabilities": {str(i): float(x) for i, x in enumerate(p)}}
        out.append(a)
    return out


def stage_profile(args):
    spec = read(QUESTIONS)
    if args.synth:
        print("(가짜 답으로 만든 보기. 내용에는 뜻이 없다)\n")
        print(profile_text(spec, synth_answers(spec, 12), 354))
        return
    if not args.answers:
        raise SystemExit("--answers <훑어보기 답 파일 하나 이상> 또는 --synth")
    recs = [r for path in args.answers for r in read(path)["records"] if "error" not in r]
    by = {}
    for r in recs:
        by.setdefault(r["video"], []).append(r)
    tv = read(HA / "data" / "tvsum_items.json")["videos"]
    out = {}
    for video, rs in by.items():
        rs.sort(key=lambda r: r["k"] if "k" in r else r["t"])
        dur = tv[video]["frames_anno"] / tv[video]["fps"] if video in tv else 300.0
        out[video] = {"moments": len(rs), "text": profile_text(spec, [r["answers"] for r in rs], dur)}
    out_path = Path(args.out) if args.out else HERE / "data" / "profiles.json"
    write(out_path, out)
    print(f"요약 {len(out)}개를 {out_path}에 썼다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "positions", "dryrun", "packs", "requests", "profile"])
    ap.add_argument("--synth", action="store_true", help="profile: 가짜 답으로 모양만 본다")
    ap.add_argument("--answers", nargs="+", default=None, help="profile: 훑어보기 답 파일(여럿 가능)")
    ap.add_argument("--out", default=None, help="profile: 쓸 곳(기본 data/profiles.json)")
    args = ap.parse_args()
    {"check": stage_check, "positions": stage_positions, "dryrun": stage_dryrun, "packs": stage_packs, "requests": stage_requests,
     "profile": stage_profile}[args.stage](args)


if __name__ == "__main__":
    main()
