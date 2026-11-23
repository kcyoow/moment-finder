# @title setup: install deps, unpack data, load EmbeddingGemma 2 (text + vision), sanity check
"""Run first in the Colab kernel. Later scripts reuse WORK, RES, model, DEVICE, frames, load_image.

Needs /content/eg/frames.tgz (v*/NNNNNN.jpg + frames.csv per video) and /content/eg/examples.tgz.
"""
import csv
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/content/eg")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "sentence-transformers[image]>=6.1.0", "transformers"], check=True)
print(f"pip install: {time.perf_counter() - t0:.0f}s")

for name in ("frames", "examples"):
    with tarfile.open(WORK / f"{name}.tgz") as tar:
        tar.extractall(WORK / name, filter="data")

import numpy as np  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_ID = "google/embeddinggemma-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.perf_counter()
# Text + vision only (audio tower not loaded). float32 for the accuracy runs.
model = SentenceTransformer(MODEL_ID, config_kwargs={"audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32}, device=DEVICE)
load_s = time.perf_counter() - t0


def load_image(path):
    return PILImage.open(path).convert("RGB")


VIDEOS = ["v1_chaeunwoo", "v2_goahsung", "v3_jungkyungho", "v4_choiyoojung"]
frames = []  # (video, idx, seconds, path), idx is 1-based like the Jev predictions
for v in VIDEOS:
    rows = list(csv.DictReader(open(WORK / "frames" / v / "frames.csv")))
    for idx, row in enumerate(rows, 1):
        frames.append((v, idx, float(row["seconds"]), WORK / "frames" / v / row["file"]))

META = {
    "model": MODEL_ID, "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    "model_load_s": round(load_s, 1), "n_frames": len(frames),
}
print(json.dumps(META, indent=1))

# Sanity check: the documented text->image path works and gives sensible ordering.
q = model.encode(["a person making a finger heart gesture", "a parked car"],
                 prompt_name="SearchQuery", normalize_embeddings=True)
img = model.encode([{"image": load_image(frames[23][3])}], normalize_embeddings=True)  # v1 #24, a clear finger heart
print("embedding dim", img.shape, "| sim(finger heart), sim(parked car) =", np.round(img @ q.T, 3).tolist())
print("setup ready")
