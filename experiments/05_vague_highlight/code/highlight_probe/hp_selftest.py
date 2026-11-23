"""highlight_probe A 단계의 GPU 없는 자체 시험 (이 Mac, 돈 없음). Modal에 올리거나 돌리지 않는다. 판정 모델도 LLM도 부르지 않는다.

(가) 본문: hp_modal이 만드는 본문 = hp_prepare.probe_body(about, sent_questions(spec), seconds, jpegs) (JSON 글자까지).
     hp_modal 쪽은 요청 글(data/probe_requests.json)과 훑어보기 묶음(data/probe_packs, 마인크래프트는 highlight_h2o의 묶음)에서,
     hp_prepare 쪽은 위치 파일과 원래 묶음(highlight_auto/data/packs, highlight_h2o/data/items.json)에서 만든다. 1,102곳 모두 견준다.
     덧붙여 훑어보기 묶음의 그림이 원래 묶음의 바이트와 같은지, sha256이 요약·요청 글과 같은지, 요청 글에 제목·범주·길이·점수 칸이 없는지,
     samples/의 보기 본문 2개와 같은지 본다
(나) 답 검사: 정상 답은 통과, 어긋난 답(확률 합, 선택지 이름, 기댓값, 단계 수, 범위, NaN, 빠진 질문)은 오류
(다) 가짜 /v1/systemone 서버(이 파일을 --fake-server로 띄운 다른 프로세스)로 hp_modal._probe를 끝까지 두 번 돌린다. 임시 폴더에만 쓴다
     1회: 정상 확인 200 + TVSum 3편 + 마인크래프트 2편. 동시 요청, /health 기다리기, 다시 묻기(한 번 500), 어긋난 답 3건이 오류로 남는지, 결과 파일
     2회: 시간 어림으로 건너뛰기(정상 확인), 처음 48건의 속도로 멈추기(마인크래프트 4편, 예상 초를 아주 작게 줌)
(라) 1회 결과로 hp_prepare.py profile이 요약을 만드는지(임시 폴더에 씀), 정상 확인 답의 모양이 highlight_auto의 sanity_check와 맞는지(800쌍)

실행: uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_probe/hp_selftest.py [--keep]
"""
import sys

sys.dont_write_bytecode = True          # 다른 실험 폴더에 __pycache__를 만들지 않는다
import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HA = ROOT / "experiments" / "highlight_auto"
HH = ROOT / "experiments" / "highlight_h2o"
sys.path.insert(0, str(HERE))
FAKE_DELAY = 0.02          # 가짜 서버가 답마다 쉬는 초(동시 요청이 쌓이게)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'통과' if ok else '실패'}] {name}" + (f" | {detail}" if detail else ""), flush=True)
    return ok


def stub_modal():
    """이 Mac의 uv 환경에는 modal이 없다. hp_modal을 불러 쓰려고 겉모양(App, Volume, Image)만 흉내 낸다. 원격 호출은 없다."""
    try:
        import modal  # noqa: F401
        return "real"
    except ImportError:
        pass
    m = types.ModuleType("modal")

    class Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    class App:
        def __init__(self, name):
            self.name = name

        def function(self, **kw):
            def deco(fn):
                fn.modal_options = kw
                return fn
            return deco

    class Volume:
        def __init__(self, name):
            self.name = name

        @classmethod
        def from_name(cls, name):
            return cls(name)

        def read_only(self):
            return self

        def commit(self):
            raise RuntimeError("자체 시험에서는 볼륨에 쓰지 않는다")

    class Image:
        @staticmethod
        def from_registry(*a, **k):
            return Chain()
    m.App, m.Volume, m.Image = App, Volume, Image
    sys.modules["modal"] = m
    return "stub"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_tar(path):
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


# =====================================================================  가짜 서버 (다른 프로세스)
def fake_answer(q, rng):
    if q["type"] == "noul":
        return {"type": "noul", "noul": rng.random()}
    opts = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    w = [rng.random() ** 3 + 1e-6 for _ in opts]
    p = {o: x / sum(w) for o, x in zip(opts, w)}
    best = max(p, key=p.get)
    if q["type"] == "choice":
        return {"type": "choice", "choice": best, "confidence": p[best], "probabilities": p}
    return {"type": "score", "score": sum(int(o) * x for o, x in p.items()), "confidence": p[best], "probabilities": p}


def break_answer(ans, questions, mode):
    """어긋난 답 만들기: sum = 고르기 확률에 0.95를 곱함, name = 고르기 선택지 이름 하나를 바꿈, expect = score를 0.2 올림."""
    kind = "score" if mode == "expect" else "choice"
    key = next(k for k, q in questions.items() if q["type"] == kind)
    a = ans[key]
    if mode == "sum":
        a["probabilities"] = {o: p * 0.95 for o, p in a["probabilities"].items()}
    elif mode == "name":
        first = next(iter(a["probabilities"]))
        a["probabilities"]["wrong_name"] = a["probabilities"].pop(first)
    else:
        a["score"] += 0.2


def fake_server(port, stats_path, faults_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    faults = json.load(open(faults_path))
    lock = threading.Lock()
    st = {"requests": 0, "inflight": 0, "max_inflight": 0, "health_calls": 0, "health_503": 0, "flaky_500": 0, "bad_sent": {},
          "state_keys": [], "body_problems": []}
    flaky_done = set()

    def save():
        tmp = stats_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, stats_path)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path != "/health":
                return self.send(404, {})
            with lock:
                st["health_calls"] += 1
                first = st["health_calls"] == 1
                st["health_503"] += first
                save()
            if first:                      # 처음 한 번은 아직 준비 중이라고 답한다(기다리는 고리를 시험)
                return self.send(503, {"status": "starting"})
            self.send(200, {"status": "ok", "model": "fake server (hp_selftest.py)"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.send(404, {})
            body = self.rfile.read(int(self.headers["Content-Length"]))
            digest = sha(body)
            req = json.loads(body)
            frames = req["state"].get("frames", [])
            probs = []
            if len(frames) != 4:
                probs.append(f"그림 {len(frames)}장")
            for fr in frames:
                u = fr["frame"]
                if not u.startswith("data:image/jpeg;base64,") or base64.b64decode(u.split(",", 1)[1])[:2] != b"\xff\xd8":
                    probs.append("JPEG data URI가 아님")
            with lock:
                st["requests"] += 1
                st["inflight"] += 1
                st["max_inflight"] = max(st["max_inflight"], st["inflight"])
                keys = ",".join(sorted(req["state"]))
                if keys not in st["state_keys"]:
                    st["state_keys"].append(keys)
                if probs and len(st["body_problems"]) < 20:
                    st["body_problems"].append(probs)
                flaky = digest in faults["flaky"] and digest not in flaky_done
                if flaky:
                    flaky_done.add(digest)
                    st["flaky_500"] += 1
                save()                         # 답을 보내기 전에 센 것을 저장한다(끝에 서버가 바로 꺼져도 잃지 않게)
            try:
                time.sleep(FAKE_DELAY)
                if flaky:
                    code, reply = 500, {"error": "fake: 한 번만 실패"}
                else:
                    rng = random.Random(digest)
                    ans = {k: fake_answer(q, rng) for k, q in req["questions"].items()}
                    mode = faults["bad"].get(digest)
                    if mode:
                        break_answer(ans, req["questions"], mode)
                        with lock:
                            st["bad_sent"][mode] = st["bad_sent"].get(mode, 0) + 1
                    code, reply = 200, {"answers": ans, "usage": {"input_tokens": 300 * len(req["questions"]), "processed_input_tokens": 1200 * len(req["questions"]),
                                                                   "images": len(frames), "latency_ms": int(FAKE_DELAY * 1000)}}
            finally:
                with lock:                     # 답을 보내기 전에 줄인다(보낸 뒤에 줄이면 같은 일꾼의 다음 요청과 겹쳐 한 건 더 세어진다)
                    st["inflight"] -= 1
                    save()
            self.send(code, reply)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64
    save()
    Server(("127.0.0.1", port), H).serve_forever()


# =====================================================================  묻는 고리 (다른 프로세스: fork 전에 urllib을 쓰지 않게)
def run_loop(cfg_path):
    cfg = json.load(open(cfg_path))
    stub_modal()
    import hp_modal as M
    M.RETRY_SLEEP = 0.01
    procs = []

    def start(port):
        log = open(cfg["server_log"], "w")
        proc = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--fake-server", str(port), cfg["stats"], cfg["faults"]],
                                stdout=log, stderr=subprocess.STDOUT)
        procs.append(proc)
        return proc, cfg["server_log"], {"settings": {"fake_server": True}, "compile_cache": None, "compile_cache_restored": False,
                                         "gpu": "none (fake server)", "h2o_revision": None}
    try:
        M._probe(cfg["arms"], cfg["budget"], cfg["fn"], roots=cfg["roots"], start=start, commit=lambda: None, port=cfg["port"])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


# =====================================================================  (가) 본문
def part_body(M, HPP):
    import hh_prepare as HP
    P = HPP.P
    spec = HPP.read(HPP.QUESTIONS)
    qs, about = HPP.sent_questions(spec), spec["state"]["about"]
    req = HPP.read(HPP.REQUESTS)
    pos = HPP.read(HERE / "data" / "probe_positions.json")
    fsum = HPP.read(HPP.FRAMES_SUMMARY)
    check("요청 글의 질문이 sent_questions(spec)와 글자까지 같다", json.dumps(req["questions"]) == json.dumps(qs))
    check("요청 글의 about이 질문 파일의 state.about과 같다", req["about"] == about)
    check("요청 글에 제목·범주·게임·길이·점수 칸이 없다", not HPP.forbidden_keys(req), str(sorted(HPP.forbidden_keys(req))))
    check("요청 글의 위치 = probe_positions.json (TVSum k, 마인크래프트 id·t)",
          {v: [p["k"] for p in x["positions"]] for v, x in req["tvsum"]["videos"].items()} == {v: x["k"] for v, x in pos["tvsum"].items()}
          and {v: [p["id"] for p in x["positions"]] for v, x in req["minecraft"]["videos"].items()} == {v: x["ids"] for v, x in pos["minecraft"].items()}
          and {v: [p["t"] for p in x["positions"]] for v, x in req["minecraft"]["videos"].items()} == {v: x["t"] for v, x in pos["minecraft"].items()})
    hashes_ok = True
    M.REQ.clear()
    M.REQ.update(req)
    M.FRAMES.clear()
    for vid in req["packs"]:
        raw = (HPP.PROBE_PACKS / f"{vid}.tar").read_bytes()
        hashes_ok &= sha(raw) == req["packs"][vid] == fsum["packs"][vid]["sha256"]
        M.FRAMES[vid] = read_tar(HPP.PROBE_PACKS / f"{vid}.tar")
    check("훑어보기 묶음 50개의 sha256 = probe_frames_summary.json = probe_requests.json", hashes_ok and len(req["packs"]) == 50)
    mc_sum = HPP.read(HH / "data" / "frames_summary.json")["packs"]
    mc_ok = True
    for video in req["minecraft"]["videos"]:
        raw = (HH / "data" / "packs" / f"{video}.tar").read_bytes()
        mc_ok &= sha(raw)[:16] == mc_sum[video]["sha256"]
        M.FRAMES[video] = read_tar(HH / "data" / "packs" / f"{video}.tar")
    check("마인크래프트 묶음 12개의 sha256 앞 16자 = highlight_h2o/data/frames_summary.json (볼륨에서 같은 값으로 확인함)", mc_ok)

    tv_items = HPP.read(HA / "data" / "tvsum_items.json")["videos"]
    same_frames, same_body, n_tv, state_ok, extra = True, True, 0, True, 0
    units = M._units("tvsum")
    by_video = {}
    for u in units:
        by_video.setdefault(u["video"], []).append(u)
    for vid, pv in pos["tvsum"].items():
        by_k = {p["k"]: p for p in tv_items[vid]["positions"]}
        need = sorted({P.fname(i) for k in pv["k"] for i in by_k[k]["frames"]})
        orig = HPP.tar_get(P.PACKS / f"{vid}.tar", need)
        same_frames &= sorted(M.FRAMES[vid]) == need and all(M.FRAMES[vid][n] == orig[n] for n in need)
        extra += len(set(M.FRAMES[vid]) - set(need))
        for u in by_video[vid]:
            mine, n_img = M._build(("tvsum", u))
            p = by_k[u["k"]]
            ref = json.dumps(HPP.probe_body(about, qs, p["seconds"], [orig[P.fname(i)] for i in p["frames"]])).encode()
            same_body &= mine == ref and n_img == 4
            st = json.loads(mine)["state"]
            state_ok &= sorted(st) == ["about", "frames"]
            n_tv += 1
    check("훑어보기 묶음의 그림 = 원래 묶음의 같은 이름 그림(바이트), 남는 그림 없음", same_frames and extra == 0, f"{sum(len(f) for v, f in M.FRAMES.items() if v in req['packs'])}장")
    check(f"TVSum 본문 {n_tv}곳: hp_modal._build = hp_prepare.probe_body (JSON 바이트)", same_body and n_tv == 742)

    mc_items = {it["id"]: it for it in HP.load_items()["full"]}
    same_mc, n_mc = True, 0
    for u in M._units("minecraft"):
        it = mc_items[u["id"]]
        orig = M.FRAMES[it["video"]]
        mine, n_img = M._build(("minecraft", u))
        ref = json.dumps(HPP.probe_body(about, qs, it["times"], [orig[HP.fname(t)] for t in it["times"]])).encode()
        same_mc &= mine == ref and n_img == 4 and u["video"] == it["video"] and u["t"] == it["t"]
        state_ok &= sorted(json.loads(mine)["state"]) == ["about", "frames"]
        n_mc += 1
    check(f"마인크래프트 본문 {n_mc}곳: hp_modal._build = hp_prepare.probe_body (JSON 바이트)", same_mc and n_mc == 360)
    check("모든 본문의 상태 칸은 about과 frames뿐이다", state_ok)

    def masked(b):
        x = json.loads(b)
        for fr in x["state"]["frames"]:
            fr["frame"] = "<image: JPEG>"
        return x
    s_tv = next(u for u in units if u["video"] == "AwmHb44_ouw" and u["k"] == 4)
    s_mc = next(u for u in M._units("minecraft") if u["id"] == "mc_01@5.5")
    check("samples/의 보기 본문 2개와 같다(그림 자리만 가림)",
          masked(M._build(("tvsum", s_tv))[0]) == HPP.read(HERE / "samples" / "probe_tvsum_AwmHb44_ouw_k4.json")
          and masked(M._build(("minecraft", s_mc))[0]) == HPP.read(HERE / "samples" / "probe_minecraft_mc_01_t5.5.json"))
    return req


# =====================================================================  (나) 답 검사
def part_parse(M, req):
    qs = req["questions"]
    rng = random.Random(1)
    good = {"answers": {k: fake_answer(q, rng) for k, q in qs.items()}}
    rec = M.parse(json.loads(json.dumps(good)), qs)
    shape_ok = all(set(rec[k]) == ({"noul"} if q["type"] == "noul" else {"probabilities"} if q["type"] == "choice" else {"score", "probabilities"})
                   for k, q in qs.items())
    check("정상 답은 통과하고 모양이 profile_text가 읽는 것과 같다", shape_ok)
    choice = next(k for k, q in qs.items() if q["type"] == "choice")
    score = next(k for k, q in qs.items() if q["type"] == "score")
    noul = next(k for k, q in qs.items() if q["type"] == "noul")

    def variant(fn):
        x = json.loads(json.dumps(good))
        fn(x["answers"])
        return x

    def scale(a, f):
        a["probabilities"] = {o: p * f for o, p in a["probabilities"].items()}

    def rename(a):
        first = next(iter(a["probabilities"]))
        a["probabilities"]["other"] = a["probabilities"].pop(first)

    def drop_level(a):
        last = list(a["probabilities"])[-1]
        p = a["probabilities"].pop(last)
        first = next(iter(a["probabilities"]))
        a["probabilities"][first] += p
        a["score"] = sum(int(o) * v for o, v in a["probabilities"].items())

    def nan_first(a):
        a["probabilities"][next(iter(a["probabilities"]))] = float("nan")
    bad = {
        "choice 확률 합 0.95": lambda x: scale(x[choice], 0.95),
        "choice 선택지 이름이 다름": lambda x: rename(x[choice]),
        "choice 선택지 하나 빠짐": lambda x: x[choice]["probabilities"].pop(next(iter(x[choice]["probabilities"]))),
        "choice 확률에 NaN": lambda x: nan_first(x[choice]),
        "score 기댓값이 0.05 어긋남": lambda x: x[score].__setitem__("score", x[score]["score"] + 0.05),
        "score 확률 합 1.05": lambda x: scale(x[score], 1.05),
        "score 단계 수가 모자람": lambda x: drop_level(x[score]),
        "score 값이 NaN": lambda x: x[score].__setitem__("score", float("nan")),
        "noul 1.2": lambda x: x[noul].__setitem__("noul", 1.2),
        "noul NaN": lambda x: x[noul].__setitem__("noul", float("nan")),
        "질문 하나 빠짐": lambda x: x.pop(score),
    }
    for name, fn in bad.items():
        try:
            M.parse(variant(fn), qs)
            check(f"어긋난 답은 오류: {name}", False, "통과해 버림")
        except Exception as err:
            check(f"어긋난 답은 오류: {name}", True, f"{type(err).__name__}: {str(err)[:70]}")
    edge = {"choice 확률 합 1.009(0.01 안)": lambda x: scale(x[choice], 1.009),
            "score 기댓값 0.009 어긋남(0.01 안)": lambda x: x[score].__setitem__("score", x[score]["score"] + 0.009)}
    for name, fn in edge.items():
        try:
            M.parse(variant(fn), qs)
            check(f"허용 범위 안은 통과: {name}", True)
        except Exception as err:
            check(f"허용 범위 안은 통과: {name}", False, str(err)[:80])


# =====================================================================  (다) 가짜 서버로 끝까지
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def sanity_packs(dest):
    """정상 확인 묶음(mf-mc-dense의 packs/mc_08.tar, ch_04.tar)의 가짜: 같은 이름 규칙(05d.jpg)과 _done.json. 그림은 회색 한 장."""
    import hh_prepare as HP
    from PIL import Image
    import ha_prepare as P
    buf = io.BytesIO()
    Image.new("RGB", (640, 360), (128, 128, 128)).save(buf, "JPEG", quality=90)
    last = HP.TICKS * (HP.SANITY_K - 1) + max(HP.SANITY_TICKS)
    names = {f"{min(max(HP.TICKS * k + t, 0), last):05d}.jpg" for k in range(HP.SANITY_K) for t in HP.SANITY_TICKS}
    dest.mkdir(parents=True, exist_ok=True)
    for v in HP.SANITY_VIDEOS:
        files = {n: buf.getvalue() for n in names}
        files["_done.json"] = json.dumps({"last": last}).encode()
        (dest / f"{v}.tar").write_bytes(P._tar_bytes(files))


def fixture(tmp, name, req, tv_videos, mc_videos, plan_over):
    data = tmp / name / "data"
    (data / "packs").mkdir(parents=True)
    sub = dict(req)
    sub["tvsum"] = {"videos": {v: req["tvsum"]["videos"][v] for v in tv_videos}}
    sub["minecraft"] = {"videos": {v: req["minecraft"]["videos"][v] for v in mc_videos}}
    sub["packs"] = {v: req["packs"][v] for v in tv_videos}
    plan = json.loads(json.dumps(req["plan"]))
    for arm in ("tvsum", "minecraft"):
        n = sum(len(x["positions"]) for x in sub[arm]["videos"].values())
        plan[arm] = {"requests": n, "est_seconds": round(req["plan"][arm]["est_seconds"] * n / req["plan"][arm]["requests"], 2)}
    for arm, est in plan_over.items():
        plan[arm]["est_seconds"] = est
    sub["plan"] = plan
    (data / "probe_requests.json").write_text(json.dumps(sub))
    for v in tv_videos:
        os.symlink(Path(HERE / "data" / "probe_packs" / f"{v}.tar"), data / "packs" / f"{v}.tar")
    return data, sub


def units_of(M, sub, arm):
    keep = dict(M.REQ)
    M.REQ.clear()
    M.REQ.update(sub)
    try:
        return M._units(arm)
    finally:
        M.REQ.clear()
        M.REQ.update(keep)


def loop(tmp, name, data, faults, budget=840):
    cfg = {"arms": "sanity,tvsum,minecraft", "budget": budget, "fn": f"selftest_{name}", "port": free_port(),
           "roots": {"data": str(data), "hh": str(HH / "data"), "mcd": str(tmp / "mcd")},
           "stats": str(tmp / name / "server_stats.json"), "faults": str(tmp / name / "faults.json"), "server_log": str(tmp / name / "server.log")}
    Path(cfg["faults"]).write_text(json.dumps(faults))
    (tmp / name / "cfg.json").write_text(json.dumps(cfg))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--loop", str(tmp / name / "cfg.json")], capture_output=True, text=True, timeout=240)
    (tmp / name / "loop.log").write_text(p.stdout + p.stderr)
    for line in p.stdout.splitlines():
        if line.startswith(("서버 준비", "  ", "끝", "건너뜀", "묶음", "본문")):
            print("    " + line[:200])
    if p.returncode != 0:
        print(p.stderr[-3000:])
    return p.returncode, json.load(open(cfg["stats"])), time.time() - t0


def part_loop(M, HPP, req, tmp):
    sanity_packs(tmp / "mcd" / "packs")
    tv3 = sorted(req["tvsum"]["videos"], key=lambda v: (len(req["tvsum"]["videos"][v]["positions"]), v))[:3]
    mc_all = list(req["minecraft"]["videos"])

    # ---- 1회: 정상 흐름 + 다시 묻기 + 어긋난 답
    data1, sub1 = fixture(tmp, "run1", req, tv3, mc_all[:2], {})
    tvu, mcu = units_of(M, sub1, "tvsum"), units_of(M, sub1, "minecraft")
    dig = lambda arm, u: sha(M._build((arm, u))[0])   # noqa: E731  (가짜 서버가 받을 본문의 해시)
    faults = {"bad": {dig("tvsum", tvu[5]): "sum", dig("tvsum", tvu[9]): "name", dig("minecraft", mcu[7]): "expect"},
              "flaky": [dig("tvsum", tvu[3]), dig("minecraft", mcu[12])]}
    print(f"  1회: TVSum {tv3} ({len(tvu)}곳), 마인크래프트 {mc_all[:2]} ({len(mcu)}곳), 정상 확인 200곳")
    code, st, secs = loop(tmp, "run1", data1, faults)
    check("1회: 고리가 오류 없이 끝남", code == 0, f"{secs:.1f}초")
    res = data1 / "results"
    files = sorted(p.name for p in res.glob("*.json")) if res.exists() else []
    check("1회: 결과 파일 넷(sanity, tvsum, minecraft, 실행 요약)이 임시 폴더에만 생김",
          files == ["_summary_selftest_run1.json", "minecraft.json", "sanity.json", "tvsum.json"] and not (HERE / "results").exists(), str(files))
    if code != 0 or len(files) != 4:
        return None
    r = {n: json.load(open(res / f"{n}.json")) for n in ("sanity", "tvsum", "minecraft")}
    m = {n: r[n]["meta"] for n in r}
    check("1회: 정상 확인 200건 모두 답, 오류 0", m["sanity"]["answered"] == 200 and m["sanity"]["errors"] == 0 and m["sanity"]["partial"] is None)
    check("1회: TVSum 오류 2건(확률 합, 선택지 이름), 나머지 답, 한 번 500 난 곳은 다시 물어 답함",
          m["tvsum"]["errors"] == 2 and m["tvsum"]["answered"] == len(tvu) - 2 and "선택지 확률" in (m["tvsum"]["first_error"] or "")
          and "error" not in r["tvsum"]["records"][3] and all("error" in r["tvsum"]["records"][i] for i in (5, 9)),
          f"답 {m['tvsum']['answered']}/{len(tvu)}, 첫 오류: {(m['tvsum']['first_error'] or '')[:60]}")
    check("1회: 마인크래프트 오류 1건(기댓값), 나머지 답",
          m["minecraft"]["errors"] == 1 and m["minecraft"]["answered"] == len(mcu) - 1 and "기댓값" in (m["minecraft"]["first_error"] or "")
          and "error" in r["minecraft"]["records"][7] and "error" not in r["minecraft"]["records"][12],
          f"답 {m['minecraft']['answered']}/{len(mcu)}")
    want = (200 + 2) + (len(tvu) + 2) + (len(mcu) + 2) + 3 * 3 + 2
    check("1회: 서버가 받은 요청 수 = 형식 확인 2건씩 + 모든 위치 + 어긋난 답 3곳의 다시 묻기 3번씩 + 500 난 2곳의 한 번",
          st["requests"] == want and st["bad_sent"] == {"sum": 4, "name": 4, "expect": 4} and st["flaky_500"] == 2, f"{st['requests']} = {want}")
    check("1회: 동시 요청이 16까지 쌓임(16을 넘지 않음)", st["max_inflight"] == 16, f"최대 {st['max_inflight']}")
    check("1회: /health가 처음에 503이어도 기다렸다가 이어감", st["health_503"] == 1 and st["health_calls"] >= 2)
    check("1회: 서버가 받은 본문의 상태 칸은 about+frames(훑어보기)와 frames(정상 확인)뿐, 그림은 모두 JPEG 4장",
          sorted(st["state_keys"]) == ["about,frames", "frames"] and not st["body_problems"], str(st["state_keys"]))
    tv_rec = r["tvsum"]["records"][0]
    mc_rec = r["minecraft"]["records"][0]
    check("1회: 기록 모양(TVSum: video, k, answers / 마인크래프트: video, t, answers / 정상 확인: id, probs)",
          {"video", "k", "answers"} <= set(tv_rec) and "t" not in tv_rec and {"video", "t", "answers"} <= set(mc_rec) and "k" not in mc_rec
          and set(tv_rec["answers"]) == set(req["questions"]) and {"id", "probs"} <= set(r["sanity"]["records"][0]))
    summ = json.load(open(res / "_summary_selftest_run1.json"))
    check("1회: 실행 요약에 묶음마다 줄과 시간이 있음", {"_checks", "sanity", "tvsum", "minecraft", "_times"} <= set(summ))

    # ---- 2회: 시간 어림으로 건너뛰기, 처음 48건의 속도로 멈추기
    data2, sub2 = fixture(tmp, "run2", req, tv3[:1], mc_all[:4], {"sanity": 1e6, "minecraft": 0.001})
    n_mc2 = sum(len(x["positions"]) for x in sub2["minecraft"]["videos"].values())
    print(f"  2회: 정상 확인 예상 1e6초(건너뛰어야 함), 마인크래프트 {n_mc2}곳 예상 0.001초(48건 뒤 멈춰야 함)")
    code2, st2, secs2 = loop(tmp, "run2", data2, {"bad": {}, "flaky": []})
    res2 = data2 / "results"
    summ2 = json.load(open(res2 / "_summary_selftest_run2.json")) if (res2 / "_summary_selftest_run2.json").exists() else {}
    check("2회: 정상 확인은 시간 어림으로 건너뜀(결과 파일 없음)", code2 == 0 and "skipped" in summ2.get("sanity", {}) and not (res2 / "sanity.json").exists(),
          summ2.get("sanity", {}).get("skipped", ""))
    mc2 = json.load(open(res2 / "minecraft.json"))["meta"] if (res2 / "minecraft.json").exists() else {}
    check("2회: 마인크래프트는 처음 48건의 속도로 멈추고 답을 partial로 저장(남은 요청은 보내지 않음)",
          (mc2.get("partial") or "").startswith("48건 뒤 멈춤") and 48 <= mc2.get("answered", 0) < n_mc2,
          f"{mc2.get('partial')} | 답 {mc2.get('answered')}/{n_mc2}")
    tv2 = json.load(open(res2 / "tvsum.json"))["meta"] if (res2 / "tvsum.json").exists() else {}
    check("2회: TVSum은 그대로 끝남", tv2.get("partial") is None and tv2.get("answered") == tv2.get("requests"))
    return data1, tv3, mc_all[:2], r


# =====================================================================  (라) 요약과 정상 확인 모양
def part_profile(HPP, tmp, data1, tv3, mc2, r):
    out = tmp / "profiles.json"
    p = subprocess.run([sys.executable, "-I", str(HERE / "hp_prepare.py"), "profile", "--answers", str(data1 / "results" / "tvsum.json"),
                        str(data1 / "results" / "minecraft.json"), "--out", str(out)], capture_output=True, text=True, timeout=120)
    check("profile 단계가 두 답 파일을 읽고 임시 폴더에 씀", p.returncode == 0 and out.exists(), p.stdout.strip()[:120] + p.stderr[-300:])
    if not out.exists():
        return
    prof = json.load(open(out))
    spec = HPP.read(HPP.QUESTIONS)
    heads = [q["heading"] for q in spec["questions"].values()]
    want = {v: sum("error" not in x for x in r["tvsum"]["records"] if x["video"] == v) for v in tv3}
    want.update({v: sum("error" not in x for x in r["minecraft"]["records"] if x["video"] == v) for v in mc2})
    ok = set(prof) == set(want) and all(prof[v]["moments"] == n and prof[v]["text"].startswith(f"Probe summary. A judging model looked at {n} moments")
                                         and all(f"- {h}:" in prof[v]["text"] for h in heads) and "- Over time" in prof[v]["text"] for v, n in want.items())
    check("요약 5개(TVSum 3, 마인크래프트 2): 위치 수 = 오류 아닌 답 수, 9항목 + 시간 흐름 줄", ok, str(want))
    print("    보기(가짜 답이라 내용에는 뜻이 없다): " + prof[tv3[0]]["text"].splitlines()[1][:150])
    sys.path.insert(0, str(HA))
    import ha_eval as E   # 읽기만(정상 확인의 비교 규칙)
    s = E.sanity_check(data1 / "results")
    check("정상 확인 답 200곳이 highlight_auto의 sanity_check와 800쌍으로 맞물림(값은 가짜라 통과 여부는 보지 않음)", s.get("pairs") == 800, json.dumps(s, ensure_ascii=False)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="임시 폴더를 지우지 않는다")
    ap.add_argument("--fake-server", nargs=3, metavar=("PORT", "STATS", "FAULTS"), help=argparse.SUPPRESS)
    ap.add_argument("--loop", metavar="CFG", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.fake_server:
        return fake_server(int(args.fake_server[0]), args.fake_server[1], args.fake_server[2])
    if args.loop:
        return run_loop(args.loop)
    t0 = time.time()
    kind = stub_modal()
    print(f"modal: {kind} (이 uv 환경에 modal이 없으면 겉모양만 흉내 낸다. 원격 호출은 없다)")
    import hp_modal as M
    import hp_prepare as HPP
    print("(가) 본문")
    req = part_body(M, HPP)
    print("(나) 답 검사")
    part_parse(M, req)
    tmp = Path(tempfile.mkdtemp(prefix="hp_selftest_"))
    print(f"(다) 가짜 서버로 끝까지 (임시 폴더 {tmp})")
    try:
        got = part_loop(M, HPP, req, tmp)
        if got:
            print("(라) 요약과 정상 확인 모양")
            part_profile(HPP, tmp, *got)
    finally:
        if args.keep:
            print(f"임시 폴더를 남김: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} 통과, {time.time() - t0:.0f}초" + (f". 실패: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
