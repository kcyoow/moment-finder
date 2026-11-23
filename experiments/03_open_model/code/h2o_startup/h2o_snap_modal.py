"""H2O-Lightning-4B 서버를 Modal의 GPU 메모리 스냅샷으로 되살려 준비 시간을 줄일 수 있는지 본다.

방법 (Modal의 vLLM 스냅샷 예제 https://modal.com/docs/examples/vllm_snapshot 를 H2O의 serve.sh에 맞춘 것)
- start (스냅샷에 들어가는 부분): serve.sh로 vLLM + shim을 띄우고 /health가 200이 될 때까지 기다린 뒤, vLLM을 잠재운다(POST /sleep?level=1:
  가중치를 CPU 메모리로 내리고 KV 캐시를 비움). 이 상태가 통째로 저장된다
- wake (되살린 뒤마다): vLLM을 깨우고(POST /wake_up) 질문 하나로 확인한다
- check: 답이 그대로인지·처리 속도를 잰다(h2o_startup_modal.py와 같은 확인)
- diag : 되살린 서버가 왜 느린지 보려는 측정(동시 요청 수별 속도, GPU 사용률, 프로세스별 CPU 시간, 그림 없는 요청, vLLM 내부 시간)
- diag_plain: 스냅샷을 쓰지 않는 보통 컨테이너에서 같은 서버를 띄워 diag와 같은 측정을 한다(기준)
예제와 다른 점: vLLM을 직접 띄우지 않고 H2O 저장소의 serve.sh를 쓴다(패치 2개, FP8, shim 포함). KV 캐시 크기는 줄이지 않았다.
스냅샷은 배포한 앱에서만 만들어진다(`modal run`으로는 안 됨). 배포 뒤 첫 한두 번은 처음부터 뜨고 그다음부터 되살아난다.

실행:
  MODAL_PROFILE=<modal-profile> modal deploy experiments/h2o_startup/h2o_snap_modal.py
  MODAL_PROFILE=<modal-profile> <modal이 깔린 파이썬> experiments/h2o_startup/snap_client.py [--diag]   # check 또는 diag를 부르고 걸린 시간을 찍는다
  MODAL_PROFILE=<modal-profile> modal run experiments/h2o_startup/h2o_snap_modal.py::diag_plain          # 기준
  MODAL_PROFILE=<modal-profile> modal app stop moment-finder-h2o-snap --yes                              # 끝나면 배포를 내린다

변형 (2026-11-14 추가, 그림 요청이 느린 것을 고쳐 보려는 것. VARIANTS 참고)
- 배포할 때 H2O_VARIANT=<이름>을 주면 서버 옵션·환경이 바뀌고 앱 이름이 moment-finder-h2o-snap-<이름>이 된다(base는 이름 그대로).
  변형마다 앱이 따로라서 스냅샷도 따로다. 컨테이너에는 Modal의 env=로 같은 값이 들어가고, start가 둘이 같은지 확인한다
  H2O_VARIANT=fix MODAL_PROFILE=<modal-profile> modal deploy experiments/h2o_startup/h2o_snap_modal.py
  MODAL_PROFILE=<modal-profile> <modal이 깔린 파이썬> experiments/h2o_startup/snap_client.py --variant fix --lite --calls 3 --tag fix
  MODAL_PROFILE=<modal-profile> modal app stop moment-finder-h2o-snap-fix --yes
- diag(lite=True): 처음 보는 그림만으로 동시 16건을 두 번(img(100,220), img(220,300)) 재고, 답이 g512_cache와 같은지 200곳에서 본다.
  그림 처리 캐시를 켠 변형에서도 앞 단계에서 본 그림이 다시 나오지 않게 고른 것(전체 diag는 단계끼리 그림이 겹친다)
- diag에 컨테이너 성질(_fingerprint: CPU 플래그, 시스템 호출·페이지 폴트·스레드 왕복 비용)과 프로세스별 user/sys 시간·페이지 폴트·
  문맥 전환, vLLM 명령줄을 더했다. 컨테이너마다 속도가 다른 까닭을 가려 보려는 것
"""
import os

import modal

# 변형: args = VLLM_ARGS 뒤에 붙일 vLLM 옵션(serve.sh에서 --mm-processor-cache-gb 0보다 뒤에 와서 이긴다),
#       env = serve.sh에 줄 환경변수(None이면 지운다), cpu = 컨테이너 CPU 요청(물리 코어)
# 근거(vLLM 0.30.0 소스): 그림 전처리는 API 서버마다 일꾼 하나짜리 실행기에서 차례로 돈다(renderers/base.py의 _mm_executor).
#   --api-server-count 2: 그 줄을 둘로. 이때 그림 처리 캐시는 API 서버 안에만 두는 processor_only가 된다(multimodal/registry.py)
#   --mm-processor-cache-gb 4: 같은 그림을 다시 처리하지 않는다. shim은 질문마다 같은 그림 4장을 따로 보내므로(요청 1건 = vLLM 요청 4건) 효과가 크다.
#     API 서버 하나에서 켜면 엔진 쪽과 짝을 맞추는 lru 캐시가 되는데, 0.31.0 전에는 요청이 준 uuid와 두 캐시의 비우는 순서 어긋남이 겹치면
#     엔진이 멈출 수 있다(GHSA-p92p-rxj5-7p2x. 답이 틀리는 문제는 아님. GHSA-ph3r-5jfg-f84f는 0.28.0에서 고쳐짐). H2O 카드는 첫 그림에서 HTTP 500을 보고 껐다
#   OMP_NUM_THREADS: 그림 전처리의 torch 스레드 수가 이 값이다(utils/torch_utils.py set_default_torch_num_threads). Modal이 cpu=8에 맞춰 8로 둔다
VARIANTS = {
    "base": {},
    "fix": {"args": "--api-server-count 2 --mm-processor-cache-gb 4"},
    "fix_omp1": {"args": "--api-server-count 2 --mm-processor-cache-gb 4", "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    "omp1": {"env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    "asc2": {"args": "--api-server-count 2"},
    "cache": {"args": "--mm-processor-cache-gb 4"},
    "shm": {"args": "--mm-processor-cache-gb 4 --mm-processor-cache-type shm"},
    "malloc": {"env": {"MALLOC_MMAP_THRESHOLD_": "33554432", "MALLOC_TRIM_THRESHOLD_": "17179869184"}},
    "cpu16": {"cpu": 16},
}
VARIANT = os.environ.get("H2O_VARIANT", "base")
if VARIANT not in VARIANTS:
    raise ValueError(f"H2O_VARIANT는 {sorted(VARIANTS)} 중 하나")
V = VARIANTS[VARIANT]

app = modal.App("moment-finder-h2o-snap" if VARIANT == "base" else f"moment-finder-h2o-snap-{VARIANT}")
data = modal.Volume.from_name("mf-mc-dense")
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache", create_if_missing=True)

GRAPH = 512
VLLM_ARGS = "--max-num-batched-tokens 8192 --quantization fp8 --enable-sleep-mode" + (" " + V["args"] if V.get("args") else "")
TICKS = 20
CHECK_VIDEOS = ("mc_08", "ch_04")
SENT = {
    "inv": "the player's inventory screen is open, the one that shows the player's character next to the armor slots",
    "chest": "a chest's storage screen is open, the one titled Chest or Large Chest that shows the chest's rows of item slots above the player's inventory rows",
}
HEAD = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). "
ASK = {"show": "Do these frames show the following: {s}?", "any": "Does at least one of these frames show the following: {s}?"}
WARM = {"state": "warm-up", "questions": {"d": {"type": "noul", "instructions": "Is this a warm-up?", "criteria": {"true": "yes", "false": "no"}}}}

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
    .env({"VLLM_SERVER_DEV_MODE": "1"})             # 잠자기·깨우기 주소를 연다
    # Modal 예제는 TORCHINDUCTOR_COMPILE_THREADS=1도 넣으라고 한다. 넣었더니 스냅샷 없이도 처리 속도가 초당 16.5건에서 13.6건으로
    # 떨어져서(2026-11-12, h2o_startup_modal.py --compile-threads 1) 뺐다
)
SPEC = dict(image=gpu_image, gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models, "/cache": cache_vol}, timeout=20 * 60,
            cpu=V.get("cpu", 8), memory=V.get("memory", 32768), env={"H2O_VARIANT": VARIANT})


def _code(url, method="GET"):
    import urllib.error
    import urllib.request
    try:
        return urllib.request.urlopen(urllib.request.Request(url, method=method, data=b"" if method == "POST" else None), timeout=120).status
    except urllib.error.HTTPError as err:
        return err.code
    except Exception:
        return 0


def _post(payload, timeout=300):
    import json
    import urllib.request
    req = urllib.request.Request("http://127.0.0.1:8741/v1/systemone", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _start_server():
    """serve.sh를 띄우고 준비될 때까지 기다린다. (프로세스, 준비까지 초, 캐시를 되살렸나)"""
    import hashlib
    import os
    import subprocess
    import time
    t0 = time.time()
    home = "/tmp/h2ohome"
    os.makedirs(home, exist_ok=True)
    key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]
    tar_path = f"/cache/vllm_cache_{key}.tar"
    restored = os.path.exists(tar_path)
    if restored:
        subprocess.run(["tar", "-xf", tar_path, "-C", home], check=True)
    if os.environ.get("H2O_VARIANT", "base") != VARIANT:   # 컨테이너의 변형이 배포할 때의 것과 다르면 멈춘다
        raise RuntimeError(f"변형 불일치: 모듈 {VARIANT}, 환경 {os.environ.get('H2O_VARIANT')}")
    env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
           "MAX_PIXELS": "2000000", "SHIM_MAX_PIXELS": "2000000", "SHIM_QUESTION_THREADS": "512", "PORT": "8741"}
    for k, v in (V.get("env") or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    proc = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=open("/tmp/serve.log", "w"), stderr=subprocess.STDOUT, env=env)
    while _code("http://127.0.0.1:8741/health") != 200:
        if proc.poll() is not None or time.time() - t0 > 15 * 60:
            print(open("/tmp/serve.log").read()[-5000:])
            raise RuntimeError("서버가 뜨지 못함")
        time.sleep(1)
    ready = round(time.time() - t0, 1)
    _post(WARM)
    if not restored:   # 컴파일 캐시를 볼륨에 (tar 하나로)
        subprocess.run(["tar", "-cf", tar_path + ".part", "-C", home, ".cache/vllm"], check=True)
        os.replace(tar_path + ".part", tar_path)
        cache_vol.commit()
    return proc, ready, restored


def _save_result(tag, obj):
    """결과를 볼륨 results/에 쓴다. 같은 이름이 이미 있으면 덮어쓰지 않고 _2, _3…을 붙인다(앞선 실험의 결과를 지키려는 것)."""
    import json
    os.makedirs("/cache/results", exist_ok=True)
    path, i = f"/cache/results/{tag}.json", 2
    while os.path.exists(path):
        path, i = f"/cache/results/{tag}_{i}.json", i + 1
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    cache_vol.commit()
    return path


def _load_frames():
    import tarfile
    frames = {}
    for vid in CHECK_VIDEOS:
        with tarfile.open(f"/data/packs/{vid}.tar") as tar:
            frames[vid] = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    return frames


def _image_body(frames, vid, k):
    import base64
    import json
    last = json.loads(frames[vid]["_done.json"])["last"]
    ticks = [min(max(TICKS * k + t, 0), last) for t in (-5, 5, 15, 25)]
    uris = ["data:image/jpeg;base64," + base64.b64encode(frames[vid][f"{n:05d}.jpg"]).decode() for n in ticks]
    return {"state": {"frames": [{"seconds": round(n / TICKS, 2), "frame": u} for n, u in zip(ticks, uris)]},
            "questions": {f"{c}_{a}": {"type": "noul", "instructions": HEAD + q.format(s=s)} for c, s in SENT.items() for a, q in ASK.items()}}


def _text_body(i):
    """그림 없는 요청: 길이가 그림 요청과 비슷한 글(약 1,200토큰)과 질문 4개. 요청마다 글이 다르다."""
    words = ("ticket", "payment", "failed", "customer", "since", "checkout", "returns", "error", "after", "update", "region", "invoice")
    text = " ".join(f"{words[(i * 7 + j * 3) % len(words)]}{(i + j) % 97}" for j in range(900))
    return {"state": {"record": text, "id": i},
            "questions": {f"q{n}": {"type": "noul", "instructions": f"Does `record` mention a problem with {w}?"} for n, w in enumerate(("payments", "checkout", "invoices", "regions"))}}


def _fingerprint():
    """컨테이너의 성질(snap_cpu_probe.py의 것을 줄인 것). 컨테이너마다 속도가 다른 까닭을 가려 보려는 것. 몇 초 걸린다."""
    import hashlib
    import mmap
    import queue
    import resource
    import threading
    import time

    import numpy as np
    out = {}
    flags = set()
    try:
        flags = set(next(ln for ln in open("/proc/cpuinfo") if ln.startswith("flags")).split(":", 1)[1].split())
    except Exception:
        pass
    out["flags_n"], out["flags_hash"] = len(flags), hashlib.sha1(" ".join(sorted(flags)).encode()).hexdigest()[:10]
    out["flags_key"] = {k: k in flags for k in ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "sha_ni")}

    def timed(fn, n):
        t = time.perf_counter()
        fn(n)
        return time.perf_counter() - t

    def py_loop(n):
        s = 0
        for i in range(n):
            s += i

    def sysc(n):
        f = os.getppid
        for _ in range(n):
            f()

    def clock(n):
        f = time.perf_counter_ns
        for _ in range(n):
            f()
    out["py_loop_ms_2m"] = round(timed(py_loop, 2_000_000) * 1000, 1)
    out["syscall_us"] = round(timed(sysc, 200_000) / 200_000 * 1e6, 3)
    out["clock_ns_per_call"] = round(timed(clock, 500_000) / 500_000 * 1e9, 1)
    size = 256 << 20
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t = time.perf_counter()
    mm = mmap.mmap(-1, size)
    a = np.frombuffer(mm, dtype=np.uint8)
    a[::4096] = 1
    dt = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    del a
    mm.close()
    out["pagefault_us_per_4k"], out["minflt_per_4k"] = round(dt / (size // 4096) * 1e6, 3), round((r1.ru_minflt - r0.ru_minflt) / (size // 4096), 3)
    q1, q2, n = queue.SimpleQueue(), queue.SimpleQueue(), 20_000

    def echo():
        for _ in range(n):
            q2.put(q1.get())
    th = threading.Thread(target=echo)
    th.start()
    t = time.perf_counter()
    for i in range(n):
        q1.put(i)
        q2.get()
    out["thread_pingpong_us"] = round((time.perf_counter() - t) / n * 1e6, 2)
    th.join()
    return out


def _proc_stats():
    """프로세스 이름별로 지금까지 쓴 user·sys CPU 초, 페이지 폴트, 문맥 전환, 스레드 수."""
    import glob
    tick, out = os.sysconf("SC_CLK_TCK"), {}
    for path in glob.glob("/proc/[0-9]*/stat"):
        try:
            raw = open(path).read()
            name = raw[raw.index("(") + 1:raw.rindex(")")]
            f = raw[raw.rindex(")") + 2:].split()
            vcsw = ivcsw = 0
            for ln in open(path[:-4] + "status"):
                if ln.startswith("voluntary_ctxt_switches"):
                    vcsw = int(ln.split()[1])
                elif ln.startswith("nonvoluntary_ctxt_switches"):
                    ivcsw = int(ln.split()[1])
            row = out.setdefault(name, [0.0, 0.0, 0, 0, 0, 0, 0])
            for i, v in enumerate((int(f[11]) / tick, int(f[12]) / tick, int(f[7]), int(f[9]), vcsw, ivcsw, int(f[17]))):
                row[i] += v
        except Exception:
            pass
    return out


def _vllm_processes():
    """떠 있는 vLLM·shim 프로세스의 이름, 명령줄 앞부분, OMP_NUM_THREADS(변형이 제대로 들어갔는지 보려는 것)."""
    import glob
    rows = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(path, "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            if "vllm" not in cmd and "shim" not in cmd:
                continue
            raw = open(path[:-7] + "stat").read()
            env = dict(kv.split("=", 1) for kv in open(path[:-7] + "environ", "rb").read().decode(errors="replace").split("\0") if "=" in kv)
            rows.append({"name": raw[raw.index("(") + 1:raw.rindex(")")], "cmd": cmd[:400], "omp": env.get("OMP_NUM_THREADS")})
        except Exception:
            pass
    return rows


def _diag(tag, lite=False):
    """서버가 준비된 컨테이너 안에서 돈다. 어디서 시간이 드는지 볼 값들을 모아 돌려준다.
    lite: 처음 보는 그림만으로 동시 16건을 두 번 재고(그림 처리 캐시를 켠 변형과도 견줄 수 있게), 답을 g512_cache와 견준다."""
    import concurrent.futures as cf
    import json
    import re
    import subprocess
    import threading
    import time
    import urllib.request
    frames = _load_frames()
    cpu_seconds = _proc_stats

    def metrics():
        try:
            text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=20).read().decode()
        except Exception:
            return {}
        out = {}
        for line in text.splitlines():
            if line.startswith("#") or "_bucket" in line:
                continue
            m = re.match(r"(vllm:[a-z_]+?)(?:\{[^}]*\})?\s+([0-9.e+-]+)$", line)
            if m and any(w in m.group(1) for w in ("time", "latency", "prefix_cache", "mm_cache", "tokens_total", "request_success", "preemptions")):
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
        return out

    def run(bodies, workers):
        """bodies를 동시 workers개로 보낸다. 속도, 서버가 잰 시간, 부른 쪽이 잰 시간, GPU 사용률, 프로세스별 CPU 초, vLLM 내부 시간."""
        util, stop = [], threading.Event()

        def sample():
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
            while not stop.is_set():
                line = p.stdout.readline()
                if line.strip().isdigit():
                    util.append(int(line))
            p.terminate()

        def one(body):
            t = time.time()
            res = _post(body)
            return time.time() - t, (res.get("usage") or {}).get("latency_ms") or 0.0

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        c0, m0, t0 = cpu_seconds(), metrics(), time.time()
        with cf.ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(one, bodies))
        wall = time.time() - t0
        c1, m1 = cpu_seconds(), metrics()
        stop.set()
        z = [0.0] * 7
        d = {k: [b - a for a, b in zip(c0.get(k, z), v)] for k, v in c1.items()}
        busy = {k: v for k, v in d.items() if v[0] + v[1] > 0.05 * wall}
        cpu = {k: round((v[0] + v[1]) / wall, 2) for k, v in busy.items()}   # 프로세스가 쓴 코어 수(이전과 같은 값)
        n = len(got)
        detail = {k: {"user_cores": round(v[0] / wall, 2), "sys_cores": round(v[1] / wall, 2), "cpu_s_per_req": round((v[0] + v[1]) / n, 3),
                      "minflt_per_req": round(v[2] / n, 1), "majflt": int(v[3]), "vcsw_per_req": round(v[4] / n, 1), "ivcsw_per_req": round(v[5] / n, 1),
                      "threads": int(c1[k][6])} for k, v in busy.items()}
        vm = {k.replace("vllm:", ""): round(m1[k] - m0.get(k, 0), 3) for k in m1 if m1[k] - m0.get(k, 0)}
        client = sorted(g[0] for g in got)
        return {"workers": workers, "n": n, "requests_per_s": round(n / wall, 1), "client_s_median": round(client[len(client) // 2], 3),
                "shim_ms_mean": round(sum(g[1] for g in got) / n, 1), "gpu_util_mean": round(sum(util) / len(util), 1) if util else None,
                "gpu_util_samples": len(util), "cpu_cores_by_process": cpu, "process_detail": detail, "vllm": vm}

    img = lambda lo, hi: [_image_body(frames, v, k) for v in CHECK_VIDEOS for k in range(lo, hi)]
    out = {"tag": tag, "variant": VARIANT, "vllm_args": VLLM_ARGS, "lite": lite, "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
           "cpu_model": next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None),
           "cgroup_cpu_max": open("/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists("/sys/fs/cgroup/cpu.max") else None,
           "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS") if os.environ.get(k)},
           "processes": _vllm_processes()}
    out["fp"] = _fingerprint()                               # 서버가 놀 때 잰다
    if lite:
        run(img(0, 20), 8)                                   # 예열 40건
        out["image_w1"] = run(img(20, 30), 1)                # 한 건씩 20건
        out["image_w16"] = run(img(100, 220), 16)            # 240건: 전체 diag의 image_w16과 같은 묶음, 앞에서 본 그림이 없다
        out["image_w16_b"] = run(img(220, 300), 16)          # 160건: 역시 처음 보는 그림. 같은 컨테이너 안에서 다시 잰 값
        items = [(v, k) for v in CHECK_VIDEOS for k in range(100)]

        def ask(it):
            res = _post(_image_body(frames, *it))
            return it, {q: float(a["noul"]) for q, a in res["answers"].items()}
        with cf.ThreadPoolExecutor(16) as ex:
            got = dict(ex.map(ask, items))
        answers = {f"{it[0]}|{it[1]}|{q}": round(p, 5) for it in items for q, p in got[it].items()}
        path = "/cache/results/g512_cache.json"
        if os.path.exists(path):                             # 답이 그대로인가(같은 설정끼리도 800개 중 3~7개가 0.5를 넘나든다)
            prev = json.load(open(path)).get("answers") or {}
            both = [k for k in answers if k in prev]
            dd = [abs(answers[k] - prev[k]) for k in both]
            out["versus"] = {"run": "g512_cache", "answers": len(both), "max_diff": round(max(dd), 4), "mean_diff": round(sum(dd) / len(dd), 4),
                             "flips": sum((answers[k] >= 0.5) != (prev[k] >= 0.5) for k in both)}
        out["answers"] = answers
        out["gpu_memory"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        return out
    run(img(0, 20), 8)                                       # 예열 40건
    out["image_w1"] = run(img(20, 30), 1)                    # 한 건씩 20건
    out["image_w4"] = run(img(30, 100), 4)                   # 140건
    out["image_w16"] = run(img(100, 220), 16)                # 240건
    out["image_w32"] = run(img(60, 180), 32)                 # 240건
    out["image_w16_again"] = run(img(180, 300), 16)          # 240건: 시간이 지나면 돌아오는가
    out["text_w16"] = run([_text_body(i) for i in range(240)], 16)
    out["gpu_memory"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out


@app.cls(scaledown_window=5, enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot": True}, **SPEC)
class H2OSnap:
    @modal.enter(snap=True)
    def start(self):
        import time
        self.server, self.ready_seconds, self.cache_restored = _start_server()
        t = time.time()
        self.sleep_code = _code("http://127.0.0.1:8000/sleep?level=1", "POST")
        self.sleep_seconds = round(time.time() - t, 1)
        if self.sleep_code != 200:   # 깨어 있는 엔진(GPU 메모리 통째)을 스냅샷에 담지 않도록 여기서 멈춘다(변형의 옵션이 잠자기와 안 맞을 때)
            print(open("/tmp/serve.log").read()[-5000:], flush=True)
            raise RuntimeError(f"잠자기 실패: HTTP {self.sleep_code} (변형 {VARIANT})")
        self.t_snap = time.time()
        print(f"start: 변형 {VARIANT}, 준비 {self.ready_seconds}초, 캐시 되살림 {self.cache_restored}, 잠자기 {self.sleep_code} ({self.sleep_seconds}초)", flush=True)

    @modal.enter(snap=False)
    def wake(self):
        import time
        t = time.time()
        self.since_snap = round(t - self.t_snap, 1)     # 크면 스냅샷에서 되살아난 것, 몇 초면 방금 start를 돈 컨테이너
        self.wake_code = _code("http://127.0.0.1:8000/wake_up", "POST")
        self.wake_error = None
        try:
            while _code("http://127.0.0.1:8741/health") != 200:
                if self.server.poll() is not None or time.time() - t > 300:
                    raise RuntimeError("깨운 뒤 서버가 준비되지 않음")
                time.sleep(0.5)
            self.first_answer = _post(WARM)["answers"]["d"]["noul"]
        except Exception as err:
            self.wake_error = f"{type(err).__name__}: {err}"[:300]
        self.wake_seconds = round(time.time() - t, 1)
        print(f"wake: {self.wake_code}, {self.wake_seconds}초, 스냅샷 뒤 {self.since_snap}초, 오류 {self.wake_error}", flush=True)

    def _state(self, tag):
        return {"tag": tag, "ready_seconds_at_start": self.ready_seconds, "cache_restored_at_start": self.cache_restored, "sleep_code": self.sleep_code,
                "sleep_seconds": self.sleep_seconds, "since_snap": self.since_snap, "restored_from_snapshot": self.since_snap > 30,
                "wake_code": self.wake_code, "wake_seconds": self.wake_seconds, "wake_error": self.wake_error}

    @modal.method()
    def check(self, full: bool = False, compare: str = "g512_cache", tag: str = ""):
        import concurrent.futures as cf
        import json
        import os
        import time
        t_check = time.time()
        out = self._state(tag)
        if self.wake_error:
            out["serve_log_tail"] = open("/tmp/serve.log").read()[-3000:]
            return out
        frames = _load_frames()

        def ask(it):
            res = _post(_image_body(frames, *it))
            return it, {q: float(a["noul"]) for q, a in res["answers"].items()}, (res.get("usage") or {}).get("latency_ms")

        items = [(v, k) for v in CHECK_VIDEOS for k in range(100)]
        lat = sorted(ms for _, _, ms in [ask(it) for it in items[:10]] if ms)
        with cf.ThreadPoolExecutor(16) as ex:
            got = dict((it, p) for it, p, _ in ex.map(ask, items))
            if full:
                t = time.time()
                load = list(ex.map(ask, [(v, k) for v in CHECK_VIDEOS for k in range(60, 300)]))
                out["requests_per_s_16"] = round(len(load) / (time.time() - t), 1)
        answers = {f"{it[0]}|{it[1]}|{q}": round(p, 5) for it in items for q, p in got[it].items()}
        path = f"/cache/results/{compare}.json"
        if compare and os.path.exists(path):
            prev = json.load(open(path)).get("answers") or {}
            both = [k for k in answers if k in prev]
            d = [abs(answers[k] - prev[k]) for k in both]
            out["versus"] = {"run": compare, "answers": len(both), "max_diff": round(max(d), 4), "mean_diff": round(sum(d) / len(d), 4),
                             "flips": sum((answers[k] >= 0.5) != (prev[k] >= 0.5) for k in both)}
        out["single_ms_median"] = lat[len(lat) // 2] if lat else None
        out["check_seconds"] = round(time.time() - t_check, 1)
        if tag:
            out["saved_to"] = _save_result(tag, {**out, "answers": answers})
        return out

    @modal.method()
    def diag(self, tag: str = "", lite: bool = False):
        import json
        import os
        import time
        t = time.time()
        out = self._state(tag)
        if self.wake_error:
            out["serve_log_tail"] = open("/tmp/serve.log").read()[-3000:]
            return out
        out.update(_diag(tag, lite))
        out["check_seconds"] = round(time.time() - t, 1)
        if tag:
            out["saved_to"] = _save_result(tag, out)
        return out

    @modal.exit()
    def stop(self):
        self.server.terminate()


@app.function(**SPEC)
def diag_plain(tag: str = "diag_plain", sleep_wake: bool = False, lite: bool = False):
    """스냅샷을 쓰지 않는 보통 컨테이너에서 같은 서버를 띄워 같은 측정을 한다. sleep_wake면 재웠다 깨운 뒤에 잰다. 변형은 H2O_VARIANT로."""
    import json
    import os
    import time
    server, ready, restored = _start_server()
    extra = {}
    if sleep_wake:
        extra["sleep_code"] = _code("http://127.0.0.1:8000/sleep?level=1", "POST")
        time.sleep(3)
        extra["wake_code"] = _code("http://127.0.0.1:8000/wake_up", "POST")
        while _code("http://127.0.0.1:8741/health") != 200:
            time.sleep(0.5)
        _post(WARM)
    out = {"ready_seconds_at_start": ready, "cache_restored_at_start": restored, "restored_from_snapshot": False, **extra, **_diag(tag, lite)}
    server.terminate()
    out["saved_to"] = _save_result(tag if VARIANT == "base" or tag != "diag_plain" else f"diag_plain_{VARIANT}", out)
    print(json.dumps(out, ensure_ascii=False))
    return out
