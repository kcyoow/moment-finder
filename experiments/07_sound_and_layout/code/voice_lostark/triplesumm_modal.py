"""TripleSumm(ICLR 2026, MIT)의 MoSu 체크포인트로 45분 로스트아크 녹화의 초마다 "다시 볼 만한 정도"를 낸다.

TripleSumm은 YouTube "Most Replayed"(재시청)로 학습한 영상 요약 모델이다. 화면·글·소리 특징을 1초 단위로 받아 초마다 0~1 점수를 낸다.
TRIBE처럼 "사람이 어디에 반응할까"를 대신 재는 증인으로 쓰고, 그 점수를 판정 모델(H2O)에 글로 넘기려는 것이다.

특징은 논문 부록 B.4를 그대로 따른다(https://arxiv.org/html/2603.01169):
- 화면: 1초에 한 장, openai/clip-vit-large-patch14의 768차원 영상 특징(시각 투영까지. 정규화하지 않음)
- 소리: 각 초 t의 [t-5, t+5] 10초를 MIT/ast-finetuned-audioset-10-10-0.4593에 넣은 [CLS] 출력(768). 섞인 원래 소리를 쓴다
- 글: 원래는 영어 자막 문장의 RoBERTa-base [CLS]. 우리 받아쓰기는 한국어라 쓰지 않고, 논문이 "글이 없는 초"에 쓰는
  기본 벡터(<pad> 토큰을 RoBERTa에 넣은 [CLS])로 모든 초를 채운다(이 점이 원래 쓰임과 다르다)
모델 설정은 저장소의 configs/mosu.yaml과 utils/config.py 기본값(max_seq_len 10000)이다. 모델 코드는 triplesumm/models(원본 커밋은 SOURCE_COMMIT).

돌리는 법(프로젝트 폴더에서, 소유자 승인 뒤):
  export MODAL_PROFILE=<modal-profile>; M=~/.local/bin/modal
  $M volume put mf-voice experiments/voice_lostark/data/frames_1fps.tar /frames_1fps.tar
  $M run --detach experiments/voice_lostark/triplesumm_modal.py
  $M volume get mf-voice /triplesumm experiments/voice_lostark/results/
"""
import io, json, subprocess, tarfile, time

import modal

app = modal.App("mf-triplesumm")
vol = modal.Volume.from_name("mf-voice")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")
         .pip_install("torch", "transformers==5.20.2", "numpy", "pillow", "huggingface_hub")
         .add_local_dir("experiments/voice_lostark/triplesumm", "/root/triplesumm"))
SR = 16000


@app.function(image=image, gpu=["L4", "A10", "T4"], volumes={"/v": vol}, timeout=30 * 60, cpu=4, memory=16384)
def run():
    import sys
    import numpy as np, torch
    from PIL import Image
    sys.path.insert(0, "/root/triplesumm")
    from models import TripleSumm
    from huggingface_hub import hf_hub_download
    from transformers import (ASTFeatureExtractor, ASTModel, AutoTokenizer, CLIPImageProcessor, CLIPModel, RobertaModel)

    t0 = time.time(); log = {}

    def mark(k):
        log[k] = round(time.time() - t0, 1); print(k, log[k], flush=True)

    # 화면: 1초에 한 장
    with tarfile.open("/v/frames_1fps.tar") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.name.endswith(".jpg"))
        imgs = [Image.open(io.BytesIO(tar.extractfile(n).read())).convert("RGB") for n in names]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "/v/audio.m4a", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    wav = np.frombuffer(raw, np.float32)
    n = min(len(imgs), len(wav) // SR)
    imgs = imgs[:n]
    mark(f"load n={n}")

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vis = []
    with torch.no_grad():
        for i in range(0, n, 64):
            px = proc(images=imgs[i:i + 64], return_tensors="pt")["pixel_values"].to("cuda")
            pooled = clip.vision_model(pixel_values=px).pooler_output       # get_image_features와 같은 계산(버전 차이를 피해 직접)
            vis.append(clip.visual_projection(pooled).float().cpu())
    vis = torch.cat(vis); assert vis.shape == (n, 768), vis.shape
    del clip; torch.cuda.empty_cache(); mark("clip")

    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to("cuda").eval()
    aud = []
    with torch.no_grad():
        for i in range(0, n, 32):
            segs = []
            for t in range(i, min(n, i + 32)):
                a, b = (t - 5) * SR, (t + 5) * SR
                seg = wav[max(0, a): max(0, min(len(wav), b))]
                if a < 0: seg = np.concatenate([np.zeros(-a, np.float32), seg])      # 앞쪽 모자란 만큼 0
                segs.append(seg)
            feats = fe(segs, sampling_rate=SR, return_tensors="pt")["input_values"].to("cuda")
            aud.append(ast(input_values=feats).last_hidden_state[:, 0].float().cpu())
    aud = torch.cat(aud); assert aud.shape == (n, 768), aud.shape
    del ast; torch.cuda.empty_cache(); mark("ast")

    tok = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    rob = RobertaModel.from_pretrained("FacebookAI/roberta-base").eval()
    with torch.no_grad():
        pad_vec = rob(**tok("<pad>", return_tensors="pt")).last_hidden_state[0, 0]
    txt = pad_vec[None].repeat(n, 1)
    mark("text(pad)")

    model = TripleSumm(visual_dim=768, text_dim=768, audio_dim=768, input_dim=128, hidden_dim=192, num_model_layers=2, num_mst_layers=2,
                       num_cmf_layers=2, num_heads=4, dropout=0.1, window_size=[5, 15, 45, 0], max_seq_len=10000, get_attn_weights=True)
    ck = hf_hub_download("smkim37/TripleSumm", "best_model_ckpt_mosu.pth")
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    out = {}
    with torch.no_grad():
        mask = torch.ones(1, n, dtype=torch.bool)
        s, attn = model(vis[None], txt[None], aud[None], mask)
        out["full"] = s[0].tolist()
        attn_info = []
        for w in attn:
            attn_info.append(list(w.shape))
        last = attn[-1] if attn else None
        if last is not None and last.dim() >= 3 and last.shape[-1] == 4:
            m = last.reshape(-1, n, 4) if last.shape[-2] == n else None
            if m is not None:
                out["attn_last_mean"] = m.mean(0).tolist()        # 초마다 [fusion, visual, text, audio]에 준 주의(가능할 때만)
    res = {"seconds": n, "timing": log, "attn_shapes": attn_info, "checkpoint": "smkim37/TripleSumm best_model_ckpt_mosu.pth",
           "features": {"visual": "clip-vit-large-patch14 1fps (grid frame t=k+0.25)", "audio": "AST [CLS] on [t-5,t+5] of mixed audio",
                        "text": "RoBERTa-base [CLS] of <pad> for every second (no transcript)"},
           "score": [round(float(v), 5) for v in out["full"]], "attn_last_mean": out.get("attn_last_mean")}
    subprocess.run(["mkdir", "-p", "/v/triplesumm"], check=True)
    json.dump(res, open("/v/triplesumm/scores.json", "w"))
    vol.commit(); mark("done")
    sc = np.array(res["score"])
    return {"seconds": n, "timing": log, "attn_shapes": attn_info, "score_min": float(sc.min()), "score_max": float(sc.max()),
            "score_mean": float(sc.mean()), "top10_seconds": [int(i) for i in np.argsort(-sc)[:10]]}


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), ensure_ascii=False, indent=1))
