# @title H2O-Lightning-4B serve: start serve.sh (stock vLLM, one-way reading) and wait until it reports ready
"""Run after h2o_setup.py in the same kernel. The server keeps running after this cell returns.

One-way reading is the card's documented fallback for stock vLLM; the recommended two-way reading needs
H2O's patched vLLM build and was not installed here.
"""
import shutil
import signal
import urllib.request

# H2O_MODE (set in the kernel before this cell; default "stock"):
#   "stock"    vLLM 0.30.0 as installed. The card says this engine runs about one decision per scheduler step under load.
#   "patched"  the card's "Serving at scale" setup for stock vLLM: its two vLLM 0.30.0 patches (VLLM_PATCHES=1) and
#              --max-num-batched-tokens 8192. Both modes read one-way (TWO_WAY=0).
# SHIM_QUESTION_THREADS is raised from 64 so the shim does not cap runs with many requests in flight.
H2O_MODE = globals().get("H2O_MODE", "stock")
LOG = Path(f"/content/h2o_serve_{H2O_MODE}.log")
proc = globals().get("H2O_PROC")
if proc is not None and proc.poll() is None and globals().get("H2O_RUNNING_MODE") != H2O_MODE:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # a server in the other mode is running: stop it first
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    time.sleep(5)
if proc is None or proc.poll() is not None:
    env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], TWO_WAY="0", SHIM_QUESTION_THREADS="512")
    if H2O_MODE == "patched":
        if shutil.which("patch") is None:
            subprocess.run("apt-get install -y -q patch", shell=True, check=True, capture_output=True)
        env.update(VLLM_PATCHES="1", VLLM_ARGS="--max-num-batched-tokens 8192")
    H2O_PROC = subprocess.Popen(["bash", str(MODEL_DIR / "serve.sh")], stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                                env=env, cwd="/content", start_new_session=True)
    H2O_T0 = time.perf_counter()
    H2O_RUNNING_MODE = H2O_MODE
    H2O_META = dict(H2O_META, mode=H2O_MODE, serve_env={k: env[k] for k in ("TWO_WAY", "SHIM_QUESTION_THREADS", "VLLM_PATCHES",
                                                                            "VLLM_ARGS") if k in env})

deadline = time.perf_counter() + 1500
ready = False
while time.perf_counter() < deadline:
    text = LOG.read_text(errors="replace") if LOG.exists() else ""
    if "H2O-Lightning-4B ready" in text:
        ready = True
        break
    if H2O_PROC.poll() is not None:
        break
    time.sleep(10)

lines = LOG.read_text(errors="replace").splitlines()
print("\n".join(line[:220] for line in lines if "vLLM patch" in line))
print("\n".join(line[:220] for line in lines[-25:]))
print("ready" if ready else f"NOT READY (server exit code {H2O_PROC.poll()})",
      f"after {time.perf_counter() - H2O_T0:.0f}s")
if ready:
    body = json.dumps({"state": "warm-up", "questions": {"d": {"type": "noul", "instructions": "Is this a warm-up?",
                                                               "criteria": {"true": "yes", "false": "no"}}}}).encode()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8741/v1/systemone", body,
                                                      {"Content-Type": "application/json"}), timeout=60)
    print(json.dumps(json.load(r))[:600])
    print(subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip())
