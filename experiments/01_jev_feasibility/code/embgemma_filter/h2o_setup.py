# @title H2O-Lightning-4B setup: vLLM 0.30.0 in its own venv, download the weights, unpack the frames
"""Run first on the L4 session. Follows the model card's "Run it" steps (stock vLLM, one-way reading).

Needs /content/eg/frames.tgz. Leaves the model in /content/h2o-lightning-4b and vLLM in /content/h2o-venv.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/content/eg")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
VENV = Path("/content/h2o-venv")
MODEL_DIR = Path("/content/h2o-lightning-4b")


def sh(cmd, check=True):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    print(f"$ {cmd[:110]}  -> exit {p.returncode} in {time.perf_counter() - t0:.0f}s")
    for line in tail:
        print("   ", line[:200])
    if check and p.returncode != 0:
        raise RuntimeError(cmd)
    return p.stdout


gpu = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader").strip()
driver_major = int(gpu.split(",")[1].strip().split(".")[0])
print("python", sys.version.split()[0], "| disk free GB", round(shutil.disk_usage("/content").free / 1e9, 1))

with tarfile.open(WORK / "frames.tgz") as tar:
    tar.extractall(WORK / "frames", filter="data")

if shutil.which("uv") is None:
    sh(f"{sys.executable} -m pip install -q uv")
if not (VENV / "bin" / "vllm").exists():
    sh(f"uv venv {VENV}")
    pins = '"vllm==0.30.0" "torchcodec==0.17.0" "transformers==5.20.2"'
    if driver_major < 580:  # the card: older drivers need the CUDA 12.9 build
        pins = ('"vllm==0.30.0+cu129" "torchcodec==0.16.0+cu129" "transformers==5.20.2" '
                "--extra-index-url https://wheels.vllm.ai/0.30.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129")
    sh(f"uv pip install --python {VENV}/bin/python {pins}")
sh(f"{VENV}/bin/python -c \"import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)\"")

if not (MODEL_DIR / "model.safetensors").exists():
    sh(f"{VENV}/bin/hf download h2oai/h2o-lightning-4b --local-dir {MODEL_DIR} --exclude 'assets/*'")
sh(f"ls -la {MODEL_DIR} | head -40; du -sh {MODEL_DIR}")
H2O_META = {"gpu": gpu, "driver_major": driver_major}
print("setup done", json.dumps(H2O_META))
