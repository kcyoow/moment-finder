# @title lib: shared helpers (run first in the Colab kernel)
"""Shared helpers for the Jev feasibility experiments.

Run once per kernel with `colab exec -s <session> -f lib.py`; later scripts reuse
`client`, `call`, `prep_image` from the kernel namespace. Every API call is appended
to /content/jev/results/calls.jsonl (no API key is ever printed or logged).
"""
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

WORK = Path("/content/jev")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)

_key_file = Path("/content/.tskey")
if _key_file.exists():
    os.environ["TYPESAFE_API_KEY"] = _key_file.read_text().strip()
    _key_file.unlink()
assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY missing"

from PIL import Image as PILImage, ImageOps  # noqa: E402
from typesafe_sdk import Image, Noul, NoulCriteria, TypeSafeAPIError, TypeSafeClient  # noqa: E402

MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
client = TypeSafeClient(model=MODEL)
CALLS = RES / "calls.jsonl"
_lock = threading.Lock()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def prep_image(src, long_side=1024, quality=90) -> bytes:
    """EXIF-rotate, RGB, shrink so the long side <= long_side, JPEG bytes."""
    img = ImageOps.exif_transpose(PILImage.open(src)).convert("RGB")
    img.thumbnail((long_side, long_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def call(tag, state, questions, meta=None, images=True):
    """One system_one call; returns a flat record and appends it to calls.jsonl."""
    t0 = time.perf_counter()
    response, error = None, None
    try:
        response = client.system_one(state=state, questions=questions,
                                     extra_headers=IMAGES if images else None)
    except TypeSafeAPIError as err:
        error = f"{type(err).__name__}: {err}"
    rec = {"tag": tag, "time": round(time.time(), 3),
           "latency": round(time.perf_counter() - t0, 3), "error": error, **(meta or {})}
    if response is not None:
        rec.update(nouls={k: v.noul for k, v in response.nouls.items()},
                   input_tokens=response.usage.input_tokens,
                   output_tokens=response.usage.output_tokens,
                   request_id=response.request_id,
                   model=getattr(response, "model", None))
    with _lock, CALLS.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


print("lib ready | model", MODEL, "| results", RES)
