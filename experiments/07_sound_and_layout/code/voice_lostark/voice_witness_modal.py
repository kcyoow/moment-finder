"""목소리 쪽 "증인" 모델 다섯을 Modal GPU에서 돌린다(조사 C가 권한 것. pipeline/README.md 25절).

입력: voice_modal.py가 볼륨 mf-voice에 남긴 목소리 줄기(/<src>/vocals_16k.wav)와 말하는 구간(/<src>/vad_vocals.json).
증인마다 따로 try로 감싸서, 하나가 실패해도 나머지는 남긴다. 결과는 /witness_<src>/<이름>.json.

1. 화자 나누기 — nvidia/Nemotron-3-Diarization(OpenMDW-1.1). transformers 5.18부터 들어 있다. 긴 녹음은 forward가 스스로 나눠 묻는다(offline)
2. 웃음 — zencastr/laughter-detection(MIT). 카드의 audio-classification pipeline을 4초 창·2초 이동으로 돌린다(카드가 잰 창과 같다)
3. 짧은 발성(비명·헉·한숨 등) — laion/vocalburst-locator model_v2.pt(Apache-2.0). 카드의 inference.py와 v2 권장 후처리(0.5/0.1/0.1)
4. 흥분도(각성도·지배감·정서가) — audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim(CC BY-NC-SA 4.0, 연구용). 말하는 구간마다
5. SenseVoiceSmall(FunASR 모델 라이선스) — 말하는 구간마다 한국어 받아쓰기 + 감정 태그 + 소리 사건 태그. torchaudio를 요구해서 이미지를 따로 쓴다

돌리는 법(프로젝트 폴더에서):
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/voice_lostark/voice_witness_modal.py --src out_test   # 앞 2분 시험
  MODAL_PROFILE=<modal-profile> ~/.local/bin/modal run --detach experiments/voice_lostark/voice_witness_modal.py --src out        # 전체
"""
import json, time

import modal

app = modal.App("mf-voice-witness")
vol = modal.Volume.from_name("mf-voice")
GPUS = ["L4", "A10", "T4"]
main_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
              .pip_install("torch", "transformers==5.20.2", "numpy", "scipy", "soundfile", "librosa", "huggingface_hub", "onnxruntime"))
sense_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "libsndfile1")
               .pip_install("torch==2.8.0", "torchaudio==2.8.0", "funasr", "numpy", "soundfile", "huggingface_hub"))
SR = 16000


def _segments(vts, max_len=8.0):
    """VAD 구간을 max_len초 이하로 자른다."""
    out = []
    for s in vts:
        a = s["start"]
        while a < s["end"] - 0.2:
            b = min(s["end"], a + max_len); out.append((round(a, 2), round(b, 2))); a = b
    return out


@app.function(image=main_image, gpu=GPUS, volumes={"/v": vol}, timeout=30 * 60, cpu=4, memory=16384)
def witnesses_main(src: str):
    import os, sys, traceback
    import numpy as np, soundfile as sf, torch

    t0 = time.time(); log = {}; out_dir = f"/v/witness_{src}"; os.makedirs(out_dir, exist_ok=True)
    wav, sr = sf.read(f"/v/{src}/vocals_16k.wav", dtype="float32"); assert sr == SR
    vts = json.load(open(f"/v/{src}/vad_vocals.json")); segs = _segments(vts)
    n = len(wav) // SR

    def save(name, obj):
        json.dump(obj, open(f"{out_dir}/{name}.json", "w"), ensure_ascii=False); vol.commit()
        log[name] = round(time.time() - t0, 1); print(name, log[name], flush=True)

    # 1. 화자 나누기
    try:
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        proc = AutoProcessor.from_pretrained("nvidia/Nemotron-3-Diarization")
        dm = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization").to("cuda").eval()
        inp = proc(wav, sampling_rate=SR).to("cuda", dtype=dm.dtype)
        with torch.inference_mode():
            logits = dm(**inp).logits
        spk = proc.extract_speaker_dict(logits, inp.attention_mask)[0]
        p = logits.sigmoid()[0].float().cpu().numpy()                       # (프레임, 8), 10ms
        per_sec = p[: n * 100].reshape(n, 100, -1).mean(1)
        save("diarization", {"segments": [{k: (round(float(v), 2) if k != "Speaker" else int(v)) for k, v in s.items()} for s in spk],
                             "per_second": [[round(float(x), 3) for x in row] for row in per_sec]})
        del dm; torch.cuda.empty_cache()
    except Exception:
        save("diarization", {"error": traceback.format_exc()[-1500:]})

    # 2. 웃음(4초 창, 2초 이동)
    try:
        from transformers import pipeline
        clf = pipeline("audio-classification", model="zencastr/laughter-detection", trust_remote_code=True, device=0)
        starts = list(range(0, max(1, n - 3), 2)); scores = []
        for i in range(0, len(starts), 64):
            batch = [{"raw": wav[s * SR:(s + 4) * SR], "sampling_rate": SR} for s in starts[i:i + 64]]
            for r in clf(batch, top_k=None):
                scores.append(next((x["score"] for x in r if x["label"] == "laughter"), 0.0))
        per_sec = np.zeros(n)
        for s, v in zip(starts, scores):
            per_sec[s:s + 4] = np.maximum(per_sec[s:s + 4], v)
        save("laughter", {"window": [4, 2], "per_second": [round(float(v), 3) for v in per_sec]})
        del clf; torch.cuda.empty_cache()
    except Exception:
        save("laughter", {"error": traceback.format_exc()[-1500:]})

    # 3. 짧은 발성
    try:
        from huggingface_hub import hf_hub_download
        code = hf_hub_download("laion/vocalburst-locator", "inference.py"); sys.path.insert(0, os.path.dirname(code))
        import inference as vb
        ck = hf_hub_download("laion/vocalburst-locator", "model_v2.pt")
        model, fe, device = vb.load_model("cuda", checkpoint=ck)
        ev = vb.detect_vocal_bursts(f"/v/{src}/vocals_16k.wav", model=model, fe=fe, device=device, threshold=0.50, merge_gap=0.10, min_dur=0.10)
        save("vocalburst", {"events": [{k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in e.items()} for e in ev]})
        del model; torch.cuda.empty_cache()
    except Exception:
        save("vocalburst", {"error": traceback.format_exc()[-1500:]})

    # 4. 흥분도(말하는 구간마다)
    try:
        import torch.nn as nn
        from transformers import Wav2Vec2FeatureExtractor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size); self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config); self.config = config
                self.wav2vec2 = Wav2Vec2Model(config); self.classifier = RegressionHead(config); self.post_init()

            def forward(self, input_values):
                h = self.wav2vec2(input_values)[0].mean(1)
                return h, self.classifier(h)

        mid = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        fx = Wav2Vec2FeatureExtractor.from_pretrained(mid); em = EmotionModel.from_pretrained(mid).to("cuda").eval()
        rows = []
        with torch.no_grad():
            for a, b in segs:
                x = wav[int(a * SR): int(b * SR)]
                if len(x) < SR // 4: continue
                y = fx(x, sampling_rate=SR)["input_values"][0].reshape(1, -1)
                _, o = em(torch.from_numpy(np.asarray(y, dtype=np.float32)).to("cuda"))
                ar, do, va = [round(float(v), 3) for v in o[0].cpu()]
                rows.append({"s": a, "e": b, "arousal": ar, "dominance": do, "valence": va})
        save("arousal", {"labels_order": "arousal, dominance, valence (card)", "segments": rows})
        del em; torch.cuda.empty_cache()
    except Exception:
        save("arousal", {"error": traceback.format_exc()[-1500:]})
    return log


@app.function(image=sense_image, gpu=GPUS, volumes={"/v": vol}, timeout=30 * 60, cpu=4, memory=16384)
def witness_sensevoice(src: str):
    import os, re, traceback
    import numpy as np, soundfile as sf

    t0 = time.time(); out_dir = f"/v/witness_{src}"; os.makedirs(out_dir, exist_ok=True)
    try:
        from funasr import AutoModel
        wav, sr = sf.read(f"/v/{src}/vocals_16k.wav", dtype="float32")
        segs = _segments(json.load(open(f"/v/{src}/vad_vocals.json")), max_len=15.0)
        m = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cuda:0", disable_update=True)
        rows = []
        for i in range(0, len(segs), 32):
            chunk = segs[i:i + 32]
            res = m.generate(input=[wav[int(a * SR): int(b * SR)] for a, b in chunk], cache={}, language="ko", use_itn=True, batch_size=32)
            for (a, b), r in zip(chunk, res):
                raw = r.get("text", "")
                tags = re.findall(r"<\|([^|]+)\|>", raw)
                rows.append({"s": a, "e": b, "raw": raw, "tags": tags, "text": re.sub(r"<\|[^|]+\|>", "", raw).strip()})
        res = {"segments": rows, "seconds": round(time.time() - t0, 1)}
    except Exception:
        res = {"error": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(f"{out_dir}/sensevoice.json", "w"), ensure_ascii=False); vol.commit()
    print("sensevoice", res.get("seconds"), "error" if "error" in res else len(res["segments"]), flush=True)
    return {"sensevoice": res.get("seconds"), "ok": "error" not in res}


@app.local_entrypoint()
def main(src: str = "out_test"):
    a = witnesses_main.spawn(src); b = witness_sensevoice.spawn(src)
    print(json.dumps({"main": a.get(), "sensevoice": b.get()}, ensure_ascii=False, indent=1))
