"""소리로 고른 순간을 잘라 보여 주기 (이 Mac에서만 돈다)."""
import base64, csv, glob, html, json, subprocess, time
from pathlib import Path
import numpy as np, torch, imageio_ffmpeg
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ROOT = Path("<repo>")
OUT = ROOT / "reports" / "sound_try"; (OUT / "clips").mkdir(parents=True, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
MID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR, WIN, TOP, GAP, PRE, POST = 16000, 2.0, 5, 8, 3, 3
KEYS = ["explosion", "burst", "bang", "boom", "smash", "crash", "breaking", "shatter", "thump", "thud", "slam", "whoosh", "hiss", "gunshot", "scream", "shout", "yell",
        "cheer", "applause", "clapping", "laugh", "gasp", "groan", "grunt", "whoop", "roar", "crowd", "squeal", "splash", "clatter", "clink", "clang", "ding", "beep",
        "knock", "tap", "click", "crack", "pop", "whistl", "alarm", "buzzer", "drop", "rattle", "squeak"]
t0 = time.time()
fe = ASTFeatureExtractor.from_pretrained(MID); model = ASTForAudioClassification.from_pretrained(MID).eval()
labels = {int(i): l for i, l in model.config.id2label.items()}
ev = [i for i, l in labels.items() if any(k in l.lower() for k in KEYS)]
print("model ready", round(time.time() - t0), "s | event classes", len(ev), flush=True)

def run_video(r): return json.load(open(ROOT / "pipeline" / "runs" / r / "run.json"))["video"]
GROUPS = {"index": [("robot_ok", "로봇 영상 · 사탕을 컵에 넣음 (65초)", run_video("web_1116_200731_2"), {"기록해 둔 '컵에 들어간 순간'": 59}),
                    ("robot_fail", "로봇 영상 · 집으려는 시도 없음 (72초)", run_video("web_1116_200730_1"), {})], "tvsum_local": []}
src = str(ROOT / "experiments/highlight_auto/data/tvsum_src")
mp4 = {Path(p).stem: p for p in glob.glob(src + "/**/*.mp4", recursive=True)}
info = next((t for t in glob.glob(src + "/**/*.tsv", recursive=True) if "info" in Path(t).name.lower()), None)
seen = set()
if info:
    for row in csv.reader(open(info, encoding="utf-8"), delimiter="\t"):
        if len(row) >= 3 and row[0] in ("PK", "FM") and row[0] not in seen and row[1] in mp4:
            seen.add(row[0]); GROUPS["tvsum_local"].append((f"tvsum_{row[0]}", f"TVSum {row[0]} · {row[2]}", mp4[row[1]], {}))
print("tvsum info", bool(info), "picked", sorted(seen), flush=True)

import os
if os.environ.get("SV"):      # 영상 하나만: SV=영상 경로, SK=이름(영문), ST=제목, SMAX=앞에서 몇 초까지, STOP=몇 곳
    GROUPS = {os.environ.get("SK", "extra"): [(os.environ.get("SK", "extra"), os.environ.get("ST", "영상"), os.environ["SV"], {})]}
    TOP = int(os.environ.get("STOP", TOP))
SMAX = int(os.environ.get("SMAX", 0))
def wav(path):
    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
def probs(w):
    n = int(len(w) / SR); segs = [w[int(max(0, (k + 0.5 - WIN / 2) * SR)): int(min(len(w), (k + 0.5 + WIN / 2) * SR))] for k in range(n)]; out = []
    for i in range(0, n, 16):
        x = fe(segs[i:i + 16], sampling_rate=SR, return_tensors="pt")["input_values"]
        with torch.no_grad(): out.append(torch.sigmoid(model(x).logits).float().numpy())
    return np.concatenate(out)
mmss = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"
CSS = "body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;background:#eef0f2;color:#14171c;margin:0;padding:28px 16px}main{max-width:900px;margin:0 auto}h1{font-size:24px}h2{font-size:18px;margin:34px 0 6px}p{color:#454d58;margin:6px 0}.row{display:grid;grid-template-columns:330px 1fr;gap:16px;background:#fff;border:1px solid #d3d7dd;border-radius:10px;padding:12px;margin:10px 0}video{width:100%;border-radius:6px;background:#000}b.t{font:600 17px ui-monospace,Menlo,monospace}.tag{display:inline-block;background:#e4e7ea;border-radius:5px;padding:1px 8px;margin:2px 4px 2px 0;font-size:13px}svg{width:100%;height:auto;background:#fff;border:1px solid #d3d7dd;border-radius:8px}.m{color:#5d6672;font-size:13.5px}@media(max-width:700px){.row{grid-template-columns:1fr}}"
for group, vids in GROUPS.items():
    body = ""
    for key, title, video, marks in vids:
        w = wav(video); w = w[:SR * SMAX] if SMAX else w; P = probs(w); n = len(P); rise = P - np.median(P, axis=0); score = rise[:, ev].max(1)
        order = np.argsort(-score); picks = []
        for t in order:
            if all(abs(int(t) - p) >= GAP for p in picks): picks.append(int(t))
            if len(picks) == TOP: break
        json.dump({"seconds": n, "score": [round(float(s), 4) for s in score], "top_event": [labels[ev[j]] for j in rise[:, ev].argmax(1)], "picks": picks},
                  open(OUT / f"{key}_scores.json", "w"), ensure_ascii=False)
        mk = {m: (s, int((score > score[max(0, s - 1): s + 2].max()).sum()) + 1) for m, s in marks.items()}
        pts = " ".join(f"{10 + k / max(1, n - 1) * 880:.1f},{70 - min(1, float(score[k]) / max(0.05, float(score.max()))) * 58:.1f}" for k in range(n))
        dots = "".join(f'<circle cx="{10 + p / max(1, n - 1) * 880:.1f}" cy="8" r="4" fill="#c47d0a"/><text x="{10 + p / max(1, n - 1) * 880:.1f}" y="92" font-size="11" text-anchor="middle" fill="#454d58">{mmss(p)}</text>' for p in picks)
        body += f'<h2>{html.escape(title)}</h2><p class="m">아래 선은 1초마다의 소리 점수(사건 소리가 평소보다 얼마나 올랐나). 점은 고른 다섯 곳.</p><svg viewBox="0 0 900 98"><polyline points="{pts}" fill="none" stroke="#2a78d6" stroke-width="1.4"/>{dots}</svg>'
        for m, (s, rk) in mk.items(): body += f'<p>{html.escape(m)} {mmss(s)}: 소리 점수로는 {n}초 가운데 <b>{rk}번째</b>.</p>'
        line = []
        for r, t in enumerate(picks, 1):
            start = max(0, t - PRE); clip = OUT / "clips" / f"{key}_{r}.mp4"
            subprocess.run([FF, "-v", "error", "-y", "-ss", str(start), "-i", str(video), "-t", str(PRE + POST + 1), "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(clip)], check=True)
            top = sorted(ev, key=lambda i: -rise[t, i])[:3]; overall = labels[int(P[t].argmax())]
            tags = "".join(f'<span class="tag">{html.escape(labels[i])} {P[t, i]:.2f}</span>' for i in top)
            body += (f'<div class="row"><video controls preload="metadata" src="clips/{clip.name}"></video><div><b class="t">{mmss(t)}</b> '
                     f'<span class="m">(클립은 {mmss(start)}부터 {PRE + POST + 1}초) · 점수 {score[t]:.2f}</span><p>오른 사건 소리: {tags}</p><p class="m">그 초에 가장 큰 소리 종류: {html.escape(overall)} {P[t].max():.2f}</p></div></div>')
            line.append((mmss(t), round(float(score[t]), 2), labels[top[0]], clip.stat().st_size // 1024))
        print(key, "| n", n, "| picks", line, "| marks", mk, flush=True)
    head = ("소리로 고른 순간" if group == "index" else "소리로 고른 순간 · " + os.environ.get("ST", "TVSum (이 Mac에서만 볼 것)"))
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{head}</title><style>{CSS}</style></head><body><main><h1>{head}</h1>'
            '<p>화면은 보지 않고 소리만으로 고른 것이다. 소리 모델(AST, AudioSet 527종)이 1초마다 소리 종류의 확률을 내고, 사건 소리(부딪힘, 딸깍, 환호, 박수 같은 것)가 그 영상의 평소보다 가장 크게 오른 초 다섯을 골랐다. 클립은 그 초의 앞뒤 3초다. 소리를 켜고 본다.</p>'
            f'{body}</main></body></html>')
    (OUT / f"{group}.html").write_text(page, encoding="utf-8"); print(group, "html KB", len(page) // 1024, flush=True)
print("done", round(time.time() - t0), "s")
