"""목소리가 든 로스트아크 녹화(45분)의 소리를 Modal GPU에서 초마다 글로 바꿀 재료로 만든다.

한 함수가 차례로 하고, 끝나면 컨테이너가 스스로 꺼진다(L4 GPU 하나).
1. 볼륨의 소리 파일(녹화의 AAC를 다시 인코딩하지 않고 옮긴 것)을 ffmpeg로 푼다
2. Demucs htdemucs(4.1.0)로 목소리 줄기(vocals)와 나머지(게임 소리)를 나눈다
3. Whisper large-v3-turbo로 받아쓴다. 목소리 줄기와 원래 섞인 소리 둘 다(한국어, 순차 장문 방식)
4. 초마다 크기(dB)를 잰다: 목소리 줄기, 섞인 소리, 게임 소리
5. AST(AudioSet 527종)로 초마다 소리 이름을 낸다. 게임 소리 줄기는 24-1절과 같은 "사건 소리" 점수,
   목소리 줄기는 웃음·외침 같은 반응 소리 점수
결과: 볼륨의 out/(시험이면 out_test/)에 per_second.json, asr_*.json, vocals_16k.wav

돌리는 법(프로젝트 폴더에서, 사용자가 값과 함께 승인한 뒤에만):
  export MODAL_PROFILE=<modal-profile>; M=~/.local/bin/modal
  $M volume create mf-voice
  $M volume put mf-voice experiments/voice_lostark/data/audio.m4a /audio.m4a
  $M run experiments/voice_lostark/voice_modal.py --seconds 120     # 시험(앞 2분)
  $M run experiments/voice_lostark/voice_modal.py                   # 전체
  $M volume get mf-voice /out experiments/voice_lostark/results/
  $M volume delete mf-voice --yes                                   # 목소리 파일을 남기지 않는다
"""
import json, subprocess, time

import modal

app = modal.App("mf-voice")
vol = modal.Volume.from_name("mf-voice", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")
         .pip_install("torch", "demucs==4.1.0", "transformers==5.20.2", "silero-vad==6.2.3", "numpy", "scipy", "soundfile"))

WHISPER = "openai/whisper-large-v3-turbo"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
# 24-1절(pipeline/tools/sound_try.py)과 같은 "사건 소리" 낱말
EVENT_KEYS = ["explosion", "burst", "bang", "boom", "smash", "crash", "breaking", "shatter", "thump", "thud", "slam", "whoosh", "hiss", "gunshot", "scream", "shout",
              "yell", "cheer", "applause", "clapping", "laugh", "gasp", "groan", "grunt", "whoop", "roar", "crowd", "squeal", "splash", "clatter", "clink", "clang",
              "ding", "beep", "knock", "tap", "click", "crack", "pop", "whistl", "alarm", "buzzer", "drop", "rattle", "squeak"]
# 목소리 줄기에서 보는 반응 소리
REACT_KEYS = ["laugh", "giggle", "chuckle", "shout", "yell", "scream", "whoop", "cheer", "gasp", "groan", "sigh", "screech", "battle cry", "wail"]


@app.function(image=image, gpu=["L4", "A10", "T4"], volumes={"/v": vol}, timeout=45 * 60, cpu=4, memory=32768)   # 2026-11-17 L4만 청하니 8분 동안 배정되지 않았다
def process(seconds: int = 0):
    import numpy as np, soundfile as sf, torch
    from scipy.signal import resample_poly

    t0 = time.time(); log = {}

    def mark(k):
        log[k] = round(time.time() - t0, 1); print(k, log[k], flush=True)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("gpu", gpu, flush=True)
    out_dir = "/v/out_test" if seconds else "/v/out"
    subprocess.run(["mkdir", "-p", out_dir], check=True)

    def decode(sr, ch):
        cmd = ["ffmpeg", "-v", "error", "-i", "/v/audio.m4a"] + (["-t", str(seconds)] if seconds else []) + ["-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, ch).T.copy()

    mix44 = decode(44100, 2); mix16 = decode(SR, 1)[0]
    n = len(mix16) // SR
    mark("decode")

    # 2. 목소리 나누기
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    dm = get_model("htdemucs"); dm.eval()
    x = torch.from_numpy(mix44); ref = x.mean(0)
    xn = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        stems = apply_model(dm, xn[None], device="cuda", shifts=1, split=True, overlap=0.25, progress=False)[0]
    stems = stems * (ref.std() + 1e-8) + ref.mean()
    vocals44 = stems[dm.sources.index("vocals")].numpy()
    del stems, xn
    voc16 = resample_poly(vocals44.mean(0), 160, 441).astype(np.float32)[: len(mix16)]
    game16 = (mix16[: len(voc16)] - voc16).astype(np.float32)
    sf.write(f"{out_dir}/vocals_16k.wav", voc16, SR, subtype="PCM_16")
    del dm; torch.cuda.empty_cache()
    mark("demucs")

    # 2-1. 말하는 구간(Silero VAD, 목소리 줄기). 받아쓰기가 말 없는 곳에 지어낸 글을 거르는 데 쓴다
    from silero_vad import get_speech_timestamps, load_silero_vad
    vts = get_speech_timestamps(torch.from_numpy(voc16), load_silero_vad(), sampling_rate=SR, return_seconds=True)
    speech = np.zeros(n)
    for s in vts:
        for k in range(int(s["start"]), min(n, int(np.ceil(s["end"])))):
            speech[k] += max(0.0, min(s["end"], k + 1) - max(s["start"], k))
    json.dump(vts, open(f"{out_dir}/vad_vocals.json", "w"))

    def vad_frac(a, b):
        b = b if b is not None else a + 5.0
        if b <= a: return 0.0
        return round(sum(max(0.0, min(b, s["end"]) - max(a, s["start"])) for s in vts) / (b - a), 3)
    mark("vad")

    def sec_db(w):
        w = w[: n * SR].reshape(n, SR)
        return [round(float(v), 2) for v in 20 * np.log10(np.sqrt((w ** 2).mean(1)) + 1e-9)]

    # 3. 받아쓰기(순차 장문 방식. 모델 카드가 권하는 설정)
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device=0)
    gen = {"language": "korean", "task": "transcribe", "condition_on_prev_tokens": False, "compression_ratio_threshold": 1.35,
           "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "logprob_threshold": -1.0, "no_speech_threshold": 0.6}
    asr_out = {}
    for name, w in (("vocals", voc16), ("mix", mix16)):
        r = asr({"raw": w, "sampling_rate": SR}, return_timestamps=True, generate_kwargs=gen)
        segs = [{"s": c["timestamp"][0], "e": c["timestamp"][1], "text": c["text"].strip(), "vad": vad_frac(c["timestamp"][0], c["timestamp"][1])}
                for c in r.get("chunks", [])]
        asr_out[name] = segs
        json.dump(segs, open(f"{out_dir}/asr_{name}.json", "w"), ensure_ascii=False, indent=0)
        mark(f"asr_{name}")
    del asr; torch.cuda.empty_cache()

    # 5. 소리 이름(AST): 1초마다 그 초를 가운데 둔 2초
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(AST_ID)
    am = ASTForAudioClassification.from_pretrained(AST_ID).to("cuda").eval()
    labels = {int(i): l for i, l in am.config.id2label.items()}
    ev = [i for i, l in labels.items() if any(k in l.lower() for k in EVENT_KEYS)]
    re_ = [i for i, l in labels.items() if any(k in l.lower() for k in REACT_KEYS)]

    def probs(w):
        segs = [w[int(max(0, (k + 0.5 - 1.0) * SR)): int(min(len(w), (k + 1.5) * SR))] for k in range(n)]
        out = []
        for i in range(0, n, 32):
            feats = fe(segs[i:i + 32], sampling_rate=SR, return_tensors="pt")["input_values"].to("cuda")
            with torch.no_grad():
                out.append(torch.sigmoid(am(feats).logits).float().cpu().numpy())
        return np.concatenate(out)

    def top(P, k=3):
        return [[[labels[int(j)], round(float(P[t, j]), 3)] for j in np.argsort(-P[t])[:k]] for t in range(len(P))]

    Pg = probs(game16); rise = Pg - np.median(Pg, axis=0)
    Pv = probs(voc16); rise_v = Pv - np.median(Pv, axis=0)
    mark("ast")

    res = {"seconds": n, "source": "2023-09-17 19-08-32.mkv (audio track copied)", "gpu": gpu, "timing": log,
           "db": {"vocals": sec_db(voc16), "mix": sec_db(mix16), "game": sec_db(game16)},
           "vad_speech": [round(float(v), 2) for v in speech], "vad_segments": len(vts),
           "game_event": {"score": [round(float(v), 4) for v in rise[:, ev].max(1)], "label": [labels[ev[j]] for j in rise[:, ev].argmax(1)], "top": top(Pg)},
           "voice_react": {"score": [round(float(v), 4) for v in rise_v[:, re_].max(1)], "label": [labels[re_[j]] for j in rise_v[:, re_].argmax(1)],
                           "speech_p": [round(float(v), 3) for v in Pv[:, [i for i, l in labels.items() if l == "Speech"][0]]], "top": top(Pv)},
           "asr_segments": {k: len(v) for k, v in asr_out.items()}}
    json.dump(res, open(f"{out_dir}/per_second.json", "w"), ensure_ascii=False)
    vol.commit()
    mark("done")
    return {"seconds": n, "gpu": gpu, "timing": log, "asr_segments": res["asr_segments"],
            "asr_vocals_head": asr_out["vocals"][:8], "asr_mix_head": asr_out["mix"][:8]}


@app.local_entrypoint()
def main(seconds: int = 0):
    print(json.dumps(process.remote(seconds), ensure_ascii=False, indent=1))
