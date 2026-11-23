"""moment-finder (Hugging Face Space, ZeroGPU). 이 파일은 모델 둘을 올리고 GPU 함수 둘을 만든다. 흐름과 화면은 engine.py에 있다.

요청 → 중간 LLM(Qwen3.8-27B-FP8)이 계획을 쓰고 "이렇게 찾을까요?" 되묻는다 → 판정 모델(H2O-Lightning-4B)이 1초마다 프레임 4장을 보고 답한다
→ 후보 6개까지 → 판정 모델이 후보마다 프레임 8장을 다시 본다(예/아니오 + 고르기) → 통과한 것만 클립 (engine.py, plan.py).

판정 모델: h2oai/h2o-lightning-4b 판 659fc412(Modal 실험과 같은 판). vLLM 대신 transformers로, 그 판의 shim(h2o_lightning_shim.py)이 만드는
프롬프트를 그대로 쓰고 질문마다 한 번 계산해 답 자리의 라벨 로짓을 읽는다(head는 fp32). Modal은 vLLM fp8, 여기는 bf16이라 확률이 조금 다를 수 있다.
중간 LLM: Qwen/Qwen3.8-27B-FP8을 FP8 그대로 올린다(transformers FineGrainedFP8 + kernels-community/finegrained-fp8). 생각 모드는 끄고 검색 도구는 없다.
"""
import spaces  # ZeroGPU: torch보다 먼저

import base64
import importlib.metadata as md
import importlib.util
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, FineGrainedFP8Config, StoppingCriteria,
                          StoppingCriteriaList)


import engine as E

MODEL_ID = "h2oai/h2o-lightning-4b"
REV = "659fc412de8486dbc2477b0d1cb4121e97e72471"          # Modal 실험(pipeline/mf_modal.py)과 같은 판


def _ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "없음"


print("[env]", json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                           "spaces": _ver("spaces"), "kernels": _ver("kernels"), "triton": _ver("triton"), "accelerate": _ver("accelerate"),
                           "fla": _ver("flash-linear-attention"), "causal_conv1d": _ver("causal-conv1d")}), flush=True)

# ---------------------------------------------------------------- 판정 모델과 그 판의 shim
SHIM_DIR = snapshot_download(MODEL_ID, revision=REV, allow_patterns=["*.py", "serve_config.json", "label_rows.lightning",
                                                                      "tokenizer_contract.json", "tokenizer/*"])
_spec = importlib.util.spec_from_file_location("h2o_lightning_shim", str(Path(SHIM_DIR) / "h2o_lightning_shim.py"))
shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim)
_cfg = json.load(open(Path(SHIM_DIR) / "serve_config.json"))
_cfg["_config_dir"] = SHIM_DIR
CONTRACT = shim.Contract(_cfg)

processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REV)
TOKENIZER_NOTE = "processor 기본"
try:   # serve.sh는 모델이 배운 Qwen2Tokenizer(tokenizer/ 폴더)를 쓴다. 같은 것을 쓴다
    processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, subfolder="tokenizer")
    TOKENIZER_NOTE = f"tokenizer/ ({type(processor.tokenizer).__name__})"
except Exception as e:  # noqa: BLE001
    TOKENIZER_NOTE = f"processor 기본 (tokenizer/를 읽지 못함: {type(e).__name__})"
tokenizer = processor.tokenizer
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16).to("cuda").eval()


def _label_ids():
    """라벨마다 답 자리의 토큰 하나(shim이 vLLM 토크나이저로 하는 확인과 같다)."""
    probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(len(CONTRACT.labels))}}
    text = CONTRACT.question_prompt(shim.render_state({"x": 1}, CONTRACT.state_compact), probe)[0]
    base = tokenizer(text, add_special_tokens=False).input_ids
    ids = {}
    for label in CONTRACT.labels:
        full = tokenizer(text + " " + label, add_special_tokens=False).input_ids
        assert len(full) == len(base) + 1 and full[: len(base)] == base, f"label {label!r} is not one token"
        ids[label] = full[-1]
    return ids


CONTRACT.label_ids = _label_ids()
SIZE = {"shortest_edge": processor.image_processor.size["shortest_edge"], "longest_edge": CONTRACT.max_pixels}

# ---------------------------------------------------------------- 중간 LLM: Qwen3.8-27B-FP8
WRITER_ID = "Qwen/Qwen3.8-27B-FP8"
WRITER_REV = "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"
# 모델 카드의 생각 끔(instruct) 값. 카드의 presence_penalty 1.5는 transformers generate에 없어 빠진다. 같은 요청이면 같은 질문이 나오게 씨앗을 고정한다
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
SEED = 1234
WRITER_MAX_NEW = 1400                                     # Gemini가 쓴 답은 이 토크나이저로 743~1,138토큰이었다


def _writer_quant_config():
    """체크포인트의 FP8 설정에서 `...mlp.gate`(MoE에서 남은 항목)를 뺀다. transformers는 건너뛸 이름을 앞부분 일치로도 보므로
    (5.20.2 quantizers_utils.should_convert_module의 re.match) `mlp.gate`가 `mlp.gate_proj`까지 잡아, 그 층이 FP8 축척 없이
    bf16 Linear로 남아 답이 망가진다. 이 판의 체크포인트 목록으로 확인: 잘못 걸리는 FP8 층은 gate_proj 65개뿐이다."""
    q = dict(AutoConfig.from_pretrained(WRITER_ID, revision=WRITER_REV).quantization_config)
    skip = q.get("modules_to_not_convert") or []
    q["modules_to_not_convert"] = [s for s in skip if not re.search(r"\.mlp\.gate$", s)]
    return FineGrainedFP8Config(**q), len(skip) - len(q["modules_to_not_convert"])


from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel  # noqa: E402

load_finegrained_fp8_kernel()       # 첫 GPU 호출 안에서 내려받다 실패하지 않게 시작할 때 받아 둔다


def _pin_fp8_autotune(num_warps=4, num_stages=2):
    """FP8 행렬곱 커널은 (N, K, BLOCK_SIZE_M)마다 설정 12가지(num_warps 2·4·8·16 × num_stages 2·3·4)를 컴파일해 재 보고 고른다
    (kernels-community/finegrained-fp8 matmul.py의 triton.autotune). 이 모델에는 그런 모양이 16개쯤이라 새 프로세스의 첫 호출이
    컴파일에 1분 넘게 쓴다(2026-11-18 Space 시험: 첫 토큰까지 37~66초). 설정을 하나로 묶으면 Triton은 재지 않고 그대로 쓴다
    (triton 3.7 Autotuner.run: 설정이 하나면 벤치마크를 건너뜀). num_stages는 2: 이 GPU(RTX PRO 6000, sm_120)의 블록당 공유 메모리 한도가
    101,376바이트라 128×128 타일에 num_stages 3(sglang 기본값)은 114,688바이트가 필요해 OutOfResources로 멈췄다(2026-11-18).
    자동 조정은 이런 설정을 건너뛰지만 하나로 묶으면 건너뛸 수 없다."""
    import sys
    import triton
    from triton.runtime.autotuner import Autotuner
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or "finegrained_fp8" not in name.replace("-", "_"):
            continue
        for obj in list(vars(mod).values()):
            if isinstance(obj, Autotuner) and len(obj.configs) > 1:
                obj.configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]
                n += 1
    return n


FP8_PINNED = _pin_fp8_autotune()
print(f"[writer] FP8 커널 자동 조정 {FP8_PINNED}곳을 설정 하나(num_warps 4, num_stages 2)로 묶음", flush=True)
_qcfg, _dropped = _writer_quant_config()
writer_tok = AutoTokenizer.from_pretrained(WRITER_ID, revision=WRITER_REV)
writer = AutoModelForImageTextToText.from_pretrained(WRITER_ID, revision=WRITER_REV, dtype="auto", device_map="cuda",
                                                     quantization_config=_qcfg).eval()
_lin = {}
for _n, _m in writer.named_modules():
    if _n.startswith("model.language_model.layers.") and _n.rsplit(".", 1)[-1].endswith(("_proj", "in_proj_qkv", "in_proj_z")):
        _lin[type(_m).__name__] = _lin.get(type(_m).__name__, 0) + 1
WRITER_NOTE = (f"{WRITER_ID}@{WRITER_REV[:8]}: 건너뛸 목록에서 뺀 항목 {_dropped}개, 언어 모델 선형층 {_lin}, "
               f"emulated capability {torch.cuda.get_device_capability()}")
print("[writer]", WRITER_NOTE, flush=True)
EOS_IDS = set(writer.generation_config.eos_token_id if isinstance(writer.generation_config.eos_token_id, list)
              else [writer.generation_config.eos_token_id])


@torch.inference_mode()
def _label_logits(inputs, ids):
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    h = model.model(**inputs, use_cache=False).last_hidden_state[0, -1].float()
    w = model.lm_head.weight[torch.tensor(ids, device=h.device)].float()
    return (w @ h).cpu().tolist()


def _decide(state, questions):
    """/v1/systemone 한 건(한 방향 읽기, 그림 경로)을 shim과 같은 순서로: 그림 꺼내기 → 기록 글 → 질문마다 프롬프트 → 라벨 로짓 → 답."""
    st, images = shim.extract_images({"state": state, "questions": questions}, CONTRACT.max_images)
    pil = [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB") for u in images]
    full_txt = shim.render_state(st, CONTRACT.state_compact)
    answers, ntok = {}, 0
    for qid, q in questions.items():
        _m, text, names, descs, labels = CONTRACT.image_prompt(full_txt, q, images, qid)
        inputs = processor(text=[text], images=pil, size=SIZE, return_tensors="pt")
        ntok += int(inputs["input_ids"].shape[1])
        logits = _label_logits(dict(inputs), [CONTRACT.label_ids[l] for l in labels])
        answers[qid] = shim.format_answer(q.get("type", "choice"), names, descs, logits, CONTRACT.image_temperature,
                                          CONTRACT.noul_floor, shim.noul_threshold(q))
    return answers, ntok


E.GPU["writer_name"] = f"{WRITER_ID}@{WRITER_REV[:8]} (transformers FP8, 생각 끔, 검색 없음)"


@spaces.GPU(duration=E.judge_duration)
def judge_batch(batch):
    """위치 여러 개를 한 GPU 호출에서 묻는다. batch = {"base_state", "questions", "items": [{"k", "frames": [b64 jpg], "seconds", ("state")}]}.
    item에 state가 있으면(다시 확인) Modal의 요청과 같은 순서 {**state, "frames"}로, 없으면(1초마다 훑기) 이전과 같은 {"frames", **base_state}로 넣는다."""
    t0, out, toks = time.time(), [], []
    for it in batch["items"]:
        fr = [{"seconds": s, "frame": "data:image/jpeg;base64," + b} for b, s in zip(it["frames"], it["seconds"])]
        state = {**it["state"], "frames": fr} if it.get("state") else {"frames": fr, **batch["base_state"]}
        ans, ntok = _decide(state, batch["questions"])
        out.append({"k": it["k"], "answers": ans})
        toks.append(ntok)
    return {"records": out, "seconds": round(time.time() - t0, 2), "reads": len(batch["items"]) * len(batch["questions"]),
            "tokens_median": int(np.median(toks)) if toks else 0}


class _Stamps(StoppingCriteria):
    """멈추게 하지는 않고, 새 토큰이 나올 때마다 시각만 적는다(첫 토큰까지 = 글 읽기, 그 뒤 = 한 토큰씩 쓰기)."""
    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores, **kwargs):
        self.times.append(time.time())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@spaces.GPU(duration=E.writer_duration)
def writer_generate(messages, max_new, prefix=""):
    """대화 한 벌을 받아 생각 끈 답 하나를 만든다. 정해 둔 GPU 시간 안에서 끝나게 max_time을 건다.
    prefix가 있으면 그 뒤에 이어 쓴다(앞 호출이 시간 제한에 걸린 경우). 돌려주는 text는 prefix까지 넣은 답 전체다."""
    t0 = time.time()
    max_new = int(max_new)
    text = writer_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False) + (prefix or "")
    enc = writer_tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    room = max(15.0, E.writer_duration(messages, max_new) - E.WRATE["overhead"] - 3)
    torch.manual_seed(SEED)
    stamps = _Stamps()
    t1 = time.time()
    with torch.inference_mode():
        out = writer.generate(**enc, max_new_tokens=max_new, max_time=room, stopping_criteria=StoppingCriteriaList([stamps]), **SAMPLING)
    torch.cuda.synchronize()
    new = out[0, enc["input_ids"].shape[1]:]
    n_new = int(new.shape[0])
    stop = "eos" if n_new and int(new[-1]) in EOS_IDS else ("length" if n_new >= max_new else "time")
    first = (stamps.times[0] - t1) if stamps.times else None
    decode = (len(stamps.times) - 1) / (stamps.times[-1] - stamps.times[0]) if len(stamps.times) > 1 else None
    res = {"text": (prefix or "") + writer_tok.decode(new, skip_special_tokens=True), "prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": n_new,
           "stop": stop, "gen_seconds": round(time.time() - t1, 2), "seconds": round(time.time() - t0, 2),
           "first_token_seconds": round(first, 2) if first is not None else None, "decode_tok_s": round(decode, 2) if decode else None,
           "room": round(room, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    print("[writer-gen]", json.dumps({k: v for k, v in res.items() if k != "text"}), flush=True)
    return res


E.GPU.update(judge=judge_batch, write=writer_generate)
demo = E.build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
