# @title embed windows: 3-second clips as VIDEO input, scored at the centre frame
"""Run after embed_frames.py in the same kernel (reuses Q, q_names, EMB, frames).

Each centre frame i gets a clip made of frames i-1, i, i+1 (1 fps mp4). The clip is embedded through
the model's video path (default: 1 fps sampling, 140 tokens per frame) and again at 280 tokens per frame.
For a like-for-like baseline the mean of the three IMAGE embeddings (280 tokens) is scored too, which
separates "video input" from "averaging over 3 seconds".
Writes /content/eg/results/window_scores.json.
"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchcodec"], check=True)

CLIPS = WORK / "clips"
CLIPS.mkdir(exist_ok=True)
by_video = {}
for row, (v, idx, sec, path) in enumerate(frames):
    by_video.setdefault(v, []).append((idx, row, path))

windows = []  # (video, centre idx, rows of the 3 frames, clip path)
for v, items in by_video.items():
    n = len(items)
    for idx, row, path in items:
        if idx == 1 or idx == n:
            continue
        clip = CLIPS / f"{v}_{idx:06d}.mp4"
        if not clip.exists():
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1", "-start_number", str(idx - 1),
                 "-i", str(path.parent / "%06d.jpg"), "-frames:v", "3", "-c:v", "libx264", "-crf", "10",
                 "-pix_fmt", "yuv420p", "-r", "1", str(clip)], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr[-400:]
        windows.append((v, idx, [row - 1, row, row + 1], clip))
print(len(windows), "clips")

win_records, win_timing = [], {}
W = {}
for tokens in (140, 280):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    W[tokens] = model.encode([{"video": str(w[3])} for w in windows], batch_size=8, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False,
                             processing_kwargs={"video": {"max_soft_tokens": tokens}})
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    win_timing[tokens] = {"seconds": round(dt, 2), "clips_per_s": round(len(windows) / dt, 2)}
    print(f"video {tokens} tok/frame: {len(windows)} clips in {dt:.1f}s ({len(windows) / dt:.1f} clips/s)")

E280 = EMB[280]
for k, (v, idx, rows, _) in enumerate(windows):
    mean3 = E280[rows].mean(axis=0)
    mean3 /= np.linalg.norm(mean3)
    rec = {"video": v, "idx": idx, "scores": {}}
    for name, vec in (("video140", W[140][k]), ("video280", W[280][k]), ("image_mean3", mean3), ("image_centre", E280[rows[1]])):
        s = vec @ Q.T
        rec["scores"][name] = {q: float(s[j]) for j, q in enumerate(q_names)}
    win_records.append(rec)

# How many frames the video path really used (token count check on one clip).
try:
    proc_out = model[0].processor(text="<|video|>", videos=[str(windows[0][3])], return_tensors="pt")
    print("one clip ->", {k: tuple(v.shape) for k, v in proc_out.items() if hasattr(v, "shape")})
except Exception as err:  # diagnostic only
    print("processor probe skipped:", type(err).__name__, str(err)[:200])

(RES / "window_scores.json").write_text(json.dumps({"meta": META, "timing": win_timing, "records": win_records},
                                                   ensure_ascii=False))
print("wrote", RES / "window_scores.json", len(win_records), "records")
