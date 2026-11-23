"""스냅샷을 켠 컨테이너에서 CPU 쪽 그림 처리가 왜 느린지, GPU 없이 싸게 가려 보는 탐침 (2026-11-14 작성, 아직 돌리지 않음).

까닭 (README "스냅샷 앱이 느린 까닭"과 vLLM 0.30.0 소스에서)
- vLLM 0.30.0의 API 서버는 그림 전처리(HF 이미지 프로세서)를 일꾼 하나짜리 실행기(`_mm_executor`, max_workers=1)에서
  한 건씩 차례로 돌린다. 그 안의 torch 연산은 OMP_NUM_THREADS(Modal이 cpu=8에 맞춰 8로 둔다)개 스레드로 나뉜다
- 그래서 그림 요청의 처리 속도는 "한 건 전처리에 걸리는 시간"에 묶인다. 스냅샷 컨테이너에서 이 시간이 1.3~1.5배 길어진 것으로 보인다
- 이 탐침은 그 한 건 전처리(base64 풀기 → JPEG 풀기 → 해시 → HF 이미지 프로세서 → 직렬화)를 GPU 없이 그대로 재고,
  컨테이너의 성질(CPU 플래그, 시스템 호출·페이지 폴트·스레드 깨우기·시계 비용, 쓸 수 있는 코어 수)도 함께 잰다

견주는 것
- 보통 컨테이너(probe_plain) 대 스냅샷을 켠 컨테이너(ProbeSnap): 같은 이미지(h2o_snap_modal.py와 같은 층), cpu=8
- 스냅샷 컨테이너 안에서: 스냅샷에 들어간 프로세스(되살린 프로세스) 대 되살린 뒤 새로 띄운 프로세스
- 스레드 수 1/2/8, 새 프로세스에서는 OMP_WAIT_POLICY=PASSIVE, glibc malloc 문턱값을 바꾼 것

실행 (Phase 2에서만. 스냅샷은 배포한 앱에서만 만들어진다)
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal deploy experiments/h2o_startup/snap_cpu_probe.py
  MODAL_PROFILE=<modal-profile> <modal이 깔린 파이썬> experiments/h2o_startup/snap_cpu_probe.py --snap-calls 5
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal app stop moment-finder-snap-probe --yes
  (<modal이 깔린 파이썬>: ~/.local/bin/modal 첫 줄의 파이썬)
결과: 볼륨 mf-vllm-cache의 results/probe_*.json, 그리고 표준 출력(한 줄에 JSON 하나)
"""
import os
import sys
import time

import modal

APP_NAME = "moment-finder-snap-probe"
app = modal.App(APP_NAME)
data = modal.Volume.from_name("mf-mc-dense")
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache", create_if_missing=True)

# h2o_snap_modal.py의 이미지와 같은 층(이미 빌드된 것을 그대로 쓴다)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "patch")
    .uv_pip_install("vllm==0.30.0", "transformers==5.20.2", "huggingface_hub", "pillow")
    .env({"VLLM_SERVER_DEV_MODE": "1"})
)
SPEC = dict(image=gpu_image, volumes={"/data": data, "/models": models, "/cache": cache_vol}, cpu=8, memory=8192,
            timeout=15 * 60, scaledown_window=5)
TICKS = 20
KS = (100, 101, 150, 151, 200, 201, 250, 251)   # 그림 요청 8벌(이웃한 k는 2장이 겹친다: 실제 요청과 같은 모양)
MAX_PIXELS = 2000000
CHILD_ENVS = {                                   # 되살린 뒤 새로 띄운 프로세스에 줄 환경 변화
    "same": {},
    "passive": {"OMP_WAIT_POLICY": "PASSIVE"},
    "malloc": {"MALLOC_MMAP_THRESHOLD_": "33554432", "MALLOC_TRIM_THRESHOLD_": "17179869184"},
}


def _load_uris():
    """mc_08 묶음에서 그림 요청 8벌(4장씩)을 base64 문자열로. 셈은 h2o_snap_modal._image_body와 같다."""
    import base64
    import json
    import tarfile
    with tarfile.open("/data/packs/mc_08.tar") as tar:
        members = {m.name: m for m in tar.getmembers()}
        last = json.loads(tar.extractfile(members["_done.json"]).read())["last"]
        sets = []
        for k in KS:
            ticks = [min(max(TICKS * k + t, 0), last) for t in (-5, 5, 15, 25)]
            sets.append([base64.b64encode(tar.extractfile(members[f"{n:05d}.jpg"]).read()).decode() for n in ticks])
    return sets


def _load_processor():
    """vLLM이 부르는 것과 같은 HF 이미지 프로세서(Qwen2VLImageProcessorFast)와, vLLM이 넘기는 인자.
    vLLM 0.30.0은 mm_device_do_normalize(기본 켬) 때문에 do_normalize/do_rescale을 끄고, serve.sh는 max_pixels를 넘긴다."""
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("/models/h2o")
    return proc


def _hasher():
    try:
        import blake3   # vLLM의 기본 해시(mm_hasher_algorithm=blake3)
        return lambda b: blake3.blake3(b).hexdigest()
    except Exception:
        import hashlib
        return lambda b: hashlib.sha256(b).hexdigest()


def _pick_kwargs(proc, uris):
    """이 transformers가 받는 인자 조합을 고른다(앞의 것일수록 vLLM과 같다)."""
    tries = ({"do_normalize": False, "do_rescale": False, "max_pixels": MAX_PIXELS}, {"do_normalize": False, "do_rescale": False}, {})
    for kw in tries:
        try:
            _mm_call(proc, uris[0], kw, _hasher())
            return kw
        except Exception:
            continue
    raise RuntimeError("이미지 프로세서를 부를 수 없음")


def _mm_call(proc, uris, kw, hasher, stages=None):
    """vLLM 요청 하나의 CPU 일: base64 풀기 → JPEG 풀기 → 해시 → HF 이미지 프로세서(4장) → 직렬화."""
    import base64
    import io
    import pickle

    import numpy as np
    from PIL import Image
    t0 = time.perf_counter()
    imgs = []
    for u in uris:
        im = Image.open(io.BytesIO(base64.b64decode(u)))
        imgs.append(im.convert("RGB"))
    t1 = time.perf_counter()
    for im in imgs:
        hasher(np.asarray(im).tobytes())
    t2 = time.perf_counter()
    out = proc(images=imgs, return_tensors="pt", **kw)
    t3 = time.perf_counter()
    pv = out["pixel_values"]
    pickle.dumps(pv.numpy(), protocol=5)
    t4 = time.perf_counter()
    if stages is not None:
        for name, a, b in (("decode", t0, t1), ("hash", t1, t2), ("hf", t2, t3), ("serialize", t3, t4)):
            stages.setdefault(name, []).append((b - a) * 1000)
    return tuple(pv.shape)


def _mm_bench(proc, uri_sets, kw, threads, n=40):
    """한 스레드에서 차례로 n건(vLLM API 서버의 _mm_executor와 같은 모양). 건당 시간과 CPU·폴트·문맥 전환."""
    import resource
    import statistics

    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    hasher = _hasher()
    try:
        for i in range(5):
            _mm_call(proc, uri_sets[i % len(uri_sets)], kw, hasher)
        stages, wall = {}, []
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t_all = time.perf_counter()
        for i in range(n):
            t = time.perf_counter()
            shape = _mm_call(proc, uri_sets[i % len(uri_sets)], kw, hasher, stages)
            wall.append((time.perf_counter() - t) * 1000)
        span = time.perf_counter() - t_all
        r1 = resource.getrusage(resource.RUSAGE_SELF)
    finally:
        torch.set_num_threads(old)
    q = sorted(wall)
    return {"threads": threads, "n": n, "ms_median": round(statistics.median(wall), 2), "ms_p90": round(q[int(0.9 * (n - 1))], 2),
            "stage_ms_median": {k: round(statistics.median(v), 2) for k, v in stages.items()},
            "user_ms_per_call": round((r1.ru_utime - r0.ru_utime) * 1000 / n, 2), "sys_ms_per_call": round((r1.ru_stime - r0.ru_stime) * 1000 / n, 2),
            "cores_busy": round(((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) / span, 2),
            "minflt_per_call": round((r1.ru_minflt - r0.ru_minflt) / n, 1), "vcsw_per_call": round((r1.ru_nvcsw - r0.ru_nvcsw) / n, 1),
            "ivcsw_per_call": round((r1.ru_nivcsw - r0.ru_nivcsw) / n, 1), "pixel_values_shape": list(shape)}


def _fingerprint():
    """컨테이너의 성질. 숫자는 모두 한 스레드에서 잰 것."""
    import hashlib
    import mmap
    import queue
    import resource
    import threading

    import numpy as np
    out = {}
    flags = set()
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("flags"):
                flags = set(ln.split(":", 1)[1].split())
                break
    except Exception:
        pass
    out["flags_n"] = len(flags)
    out["flags_hash"] = hashlib.sha1(" ".join(sorted(flags)).encode()).hexdigest()[:10]
    out["flags_key"] = {k: k in flags for k in ("avx2", "fma", "avx512f", "avx512bw", "avx512_vnni", "avx512_bf16", "amx_tile", "sha_ni", "vaes", "pclmulqdq")}
    out["cpu_count"], out["affinity"] = os.cpu_count(), len(os.sched_getaffinity(0))
    files = {}
    for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/kernel/mm/transparent_hugepage/enabled",
              "/sys/kernel/mm/transparent_hugepage/shmem_enabled"):
        try:
            files[p] = open(p).read().strip()[:80]
        except Exception:
            files[p] = None
    out["files"] = files
    out["env"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_WAIT_POLICY", "MALLOC_MMAP_THRESHOLD_", "MALLOC_ARENA_MAX")}
    try:
        import torch
        out["torch_cpu_capability"], out["torch_threads"] = torch.backends.cpu.get_cpu_capability(), torch.get_num_threads()
    except Exception as err:
        out["torch_error"] = str(err)[:100]
    try:
        feats = np._core._multiarray_umath.__cpu_features__
        out["numpy_features"] = {k: feats.get(k) for k in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_SPR")}
    except Exception:
        pass

    def timed(fn, reps):
        t = time.perf_counter()
        fn(reps)
        return time.perf_counter() - t

    def py_loop(n):
        s = 0
        for i in range(n):
            s += i
    out["py_loop_ms_2m"] = round(timed(py_loop, 2_000_000) * 1000, 1)                      # 해석기 속도(호스트 기준값)

    def clock(n):
        f = time.perf_counter_ns
        for _ in range(n):
            f()
    out["clock_ns_per_call"] = round(timed(clock, 500_000) / 500_000 * 1e9, 1)            # 시계 읽기(vDSO인가 시스템 호출인가)

    def sysc(n):
        f = os.getppid
        for _ in range(n):
            f()
    out["syscall_us"] = round(timed(sysc, 200_000) / 200_000 * 1e6, 3)                     # 시스템 호출 한 번

    size, page = 256 << 20, 4096                                                              # 새 메모리 첫 접근(페이지 폴트)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t = time.perf_counter()
    mm = mmap.mmap(-1, size)
    a = np.frombuffer(mm, dtype=np.uint8)
    a[::page] = 1
    dt = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    del a
    mm.close()
    out["pagefault_us_per_4k"] = round(dt / (size // page) * 1e6, 3)
    out["minflt_per_4k"] = round((r1.ru_minflt - r0.ru_minflt) / (size // page), 3)

    t = time.perf_counter()                                                                   # malloc → 해제를 되풀이(64 MiB)
    for _ in range(8):
        b = np.ones((64 << 20) // 8)
        del b
    out["np_alloc64mib_ms"] = round((time.perf_counter() - t) / 8 * 1000, 2)

    src = np.ones((256 << 20) // 8)
    dst = np.zeros_like(src)
    t = time.perf_counter()
    for _ in range(4):
        np.copyto(dst, src)
    out["memcpy_gbs"] = round(4 * (256 << 20) / (time.perf_counter() - t) / 1e9, 2)
    del src, dst

    q1, q2, n = queue.SimpleQueue(), queue.SimpleQueue(), 20_000                            # 스레드 깨우기 왕복

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

    try:                                                                                      # OMP 나눠 돌리기(포크·조인) 비용
        import torch
        old = torch.get_num_threads()
        x, y = torch.randn(1 << 20), torch.randn(1 << 20)
        z = torch.empty_like(x)
        for th_n in (1, 8):
            torch.set_num_threads(th_n)
            for _ in range(20):
                torch.add(x, y, out=z)
            t = time.perf_counter()
            for _ in range(300):
                torch.add(x, y, out=z)
            out[f"torch_add_1m_us_t{th_n}"] = round((time.perf_counter() - t) / 300 * 1e6, 1)
        torch.set_num_threads(1)
        m = torch.randn(512, 512)
        t = time.perf_counter()
        for _ in range(20):
            m @ m
        out["matmul512_ms_t1"] = round((time.perf_counter() - t) / 20 * 1000, 2)              # SIMD 기준값
        torch.set_num_threads(old)
    except Exception as err:
        out["torch_bench_error"] = str(err)[:100]
    return out


def _cpu_capacity(nproc=16, secs=3.0):
    """바쁜 프로세스 nproc개를 secs초 돌려, 실제로 받은 코어 수를 잰다(Modal의 기본 상한은 요청 + 16코어)."""
    import resource
    import subprocess
    code = f"import time\nt = time.time()\nwhile time.time() - t < {secs}:\n    pass\n"
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.time()
    ps = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(nproc)]
    for p in ps:
        p.wait()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return {"procs": nproc, "cores_received": round(cpu / secs, 1), "wall_s": round(time.time() - t, 1)}


def _child(uris_path):
    """되살린 뒤(또는 보통 컨테이너에서) 새로 띄운 프로세스: 같은 측정."""
    import json
    uri_sets = json.load(open(uris_path))
    t = time.time()
    proc = _load_processor()
    kw = _pick_kwargs(proc, uri_sets)
    load_s = round(time.time() - t, 1)
    res = {"load_s": load_s, "kwargs": kw, "fp": _fingerprint(), "mm": [_mm_bench(proc, uri_sets, kw, th) for th in (1, 8)]}
    print("CHILD_JSON " + json.dumps(res), flush=True)


def _probe(kind, proc, kw, uri_sets, since_snap, tag):
    import json
    import subprocess
    t0 = time.time()
    out = {"kind": kind, "tag": tag, "since_snap": since_snap, "restored_from_snapshot": bool(since_snap and since_snap > 30), "kwargs": kw}
    out["fp"] = _fingerprint()
    out["capacity"] = _cpu_capacity(16)
    out["inproc"] = [_mm_bench(proc, uri_sets, kw, th) for th in (1, 2, 8)]
    path = "/tmp/probe_uris.json"
    json.dump(uri_sets, open(path, "w"))
    for name, extra in CHILD_ENVS.items():
        env = {**os.environ, **extra}
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path], env=env, capture_output=True, text=True, timeout=600,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("CHILD_JSON ")), None)
        out[f"child_{name}"] = json.loads(line[len("CHILD_JSON "):]) if line else {"error": (r.stderr or r.stdout)[-1500:]}
    out["probe_seconds"] = round(time.time() - t0, 1)
    if tag:   # 같은 이름이 있으면 덮어쓰지 않고 _2, _3…을 붙인다
        os.makedirs("/cache/results", exist_ok=True)
        path, i = f"/cache/results/probe_{tag}.json", 2
        while os.path.exists(path):
            path, i = f"/cache/results/probe_{tag}_{i}.json", i + 1
        with open(path, "w") as f:
            json.dump(out, f, ensure_ascii=False)
        cache_vol.commit()
        out["saved_to"] = path
    return out


def _setup():
    uri_sets = _load_uris()
    proc = _load_processor()
    kw = _pick_kwargs(proc, uri_sets)
    _mm_bench(proc, uri_sets, kw, 8, n=10)        # 서버가 스냅샷 전에 예열하는 것처럼 한 번 돌려 둔다
    return proc, kw, uri_sets


@app.cls(enable_memory_snapshot=True, **SPEC)
class ProbeSnap:
    @modal.enter(snap=True)
    def load(self):
        self.proc, self.kw, self.uri_sets = _setup()
        self.t_snap = time.time()

    @modal.enter(snap=False)
    def after_restore(self):
        self.since_snap = round(time.time() - self.t_snap, 1)   # 크면 되살린 컨테이너, 몇 초면 방금 스냅샷을 만든 컨테이너

    @modal.method()
    def probe(self, tag: str = ""):
        return _probe("snap", self.proc, self.kw, self.uri_sets, self.since_snap, tag)


@app.function(**SPEC)
def probe_plain(tag: str = ""):
    proc, kw, uri_sets = _setup()
    return _probe("plain", proc, kw, uri_sets, None, tag)


def _summary(out):
    """한 줄 요약: 건당 ms(스레드 8/1), 새 프로세스 건당 ms, 페이지 폴트·시스템 호출·스레드 왕복, 받은 코어."""
    pick = lambda rows, th: next((r["ms_median"] for r in rows if r["threads"] == th), None)
    fp = out.get("fp", {})
    s = {"kind": out.get("kind"), "tag": out.get("tag"), "since_snap": out.get("since_snap"), "restored": out.get("restored_from_snapshot"),
         "inproc_ms_t8": pick(out.get("inproc", []), 8), "inproc_ms_t1": pick(out.get("inproc", []), 1),
         "child_ms_t8": pick(out.get("child_same", {}).get("mm", []), 8), "child_ms_t1": pick(out.get("child_same", {}).get("mm", []), 1),
         "passive_ms_t8": pick(out.get("child_passive", {}).get("mm", []), 8), "malloc_ms_t8": pick(out.get("child_malloc", {}).get("mm", []), 8),
         "flags": fp.get("flags_hash"), "avx512f": fp.get("flags_key", {}).get("avx512f"), "py_loop_ms": fp.get("py_loop_ms_2m"),
         "pagefault_us": fp.get("pagefault_us_per_4k"), "syscall_us": fp.get("syscall_us"), "clock_ns": fp.get("clock_ns_per_call"),
         "pingpong_us": fp.get("thread_pingpong_us"), "cores_received": out.get("capacity", {}).get("cores_received")}
    return s


def _client(argv):
    """로컬에서: 보통 컨테이너 1번 → 스냅샷 컨테이너 N번 → 보통 컨테이너 1번. 부를 때마다 이 앱의 컨테이너가 다 꺼진 것을 확인한다."""
    import json

    from snap_client import ContainerWatch, app_id_of, wait_no_containers
    calls = int(argv[argv.index("--snap-calls") + 1]) if "--snap-calls" in argv else 5
    aid = app_id_of(APP_NAME)
    plain = modal.Function.from_name(APP_NAME, "probe_plain")
    Snap = modal.Cls.from_name(APP_NAME, "ProbeSnap")
    plan = [("plain", 1)] + [("snap", i + 1) for i in range(calls)] + [("plain", 2)]
    restored = 0
    for kind, i in plan:
        if kind == "snap" and restored >= 3:       # 되살린 컨테이너 셋을 쟀으면 남은 스냅샷 호출은 건너뛴다
            continue
        waited = wait_no_containers(aid)
        tag = f"{kind}{i}"
        with ContainerWatch(aid) as watch:
            t = time.time()
            out = plain.remote(tag=tag) if kind == "plain" else Snap().probe.remote(tag=tag)
            out["client_wall_seconds"] = round(time.time() - t, 1)
        out["containers_seen"], out["waited_for_scaledown_s"] = sorted(watch.seen), waited
        restored += bool(out.get("restored_from_snapshot"))
        print(json.dumps(out, ensure_ascii=False), flush=True)
        print("SUMMARY " + json.dumps(_summary(out), ensure_ascii=False), file=sys.stderr, flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child(sys.argv[sys.argv.index("--child") + 1])
    else:
        _client(sys.argv[1:])
