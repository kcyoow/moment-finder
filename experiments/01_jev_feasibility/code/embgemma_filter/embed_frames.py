# @title embed frames: 243 frames x 3 vision-token budgets, 4 text queries, example-photo queries
"""Run after colab_setup.py in the same kernel. Writes /content/eg/results/frame_scores.json
and frame_emb_<budget>.npy (row order = `frames`).

Scores are cosine similarities (embeddings are L2-normalized).
"""
QUERIES = {
    # what a user would type, translated to English
    "en_plain": "a person making a finger heart gesture",
    # best-effort description, same content as the Jev criteria `true` text (PRIMARY, fixed before seeing results)
    "en_detail": ("a person making the Korean finger heart gesture, with the thumb and index finger crossed "
                  "so their tips form a small heart"),
    # the user's own Korean sentence
    "ko_plain": "인물이 손하트 하는 모습",
    # the main confusable gesture, used for a contrast score
    "en_two_hand": "a person forming one big heart shape with both hands together",
}
EXAMPLES = {"pos": ["pfin002", "pfin024", "pfin038"], "neg": ["nhan001", "ncro004", "nok004"]}
BUDGETS = [280, 70, 1120]  # 280 = model default per image
BATCH = {70: 32, 280: 16, 1120: 4}


def encode_images(images, budget, batch_size):
    return model.encode([{"image": im} for im in images], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False,
                        processing_kwargs={"image": {"max_soft_tokens": budget}})


q_names = list(QUERIES)
Q = model.encode([QUERIES[k] for k in q_names], prompt_name="SearchQuery", normalize_embeddings=True,
                 convert_to_numpy=True)
frame_images = [load_image(p) for _, _, _, p in frames]
ex_images = {k: [load_image(WORK / "examples" / f"{i}.jpg") for i in ids] for k, ids in EXAMPLES.items()}

records, timing, EMB = [], {}, {}
for budget in BUDGETS:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    E = encode_images(frame_images, budget, BATCH[budget])
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    timing[budget] = {"seconds": round(dt, 2), "frames_per_s": round(len(frames) / dt, 2), "batch": BATCH[budget]}
    EMB[budget] = E
    np.save(RES / f"frame_emb_{budget}.npy", E)
    S = E @ Q.T
    ex_pos = E @ encode_images(ex_images["pos"], budget, 4).T
    ex_neg = E @ encode_images(ex_images["neg"], budget, 4).T
    for i, (v, idx, sec, _) in enumerate(frames):
        scores = {k: float(S[i, j]) for j, k in enumerate(q_names)}
        scores["en_detail_minus_two_hand"] = scores["en_detail"] - scores["en_two_hand"]
        scores["ex_pos_max"] = float(ex_pos[i].max())
        scores["ex_pos_mean"] = float(ex_pos[i].mean())
        scores["ex_pos_minus_neg"] = float(ex_pos[i].max() - ex_neg[i].max())
        records.append({"video": v, "idx": idx, "seconds": sec, "budget": budget, "scores": scores})
    print(f"budget {budget}: {len(frames)} frames in {dt:.1f}s ({len(frames) / dt:.1f} frames/s), emb {E.shape}")

out = {"meta": META, "queries": QUERIES, "examples": EXAMPLES, "timing": timing, "records": records}
(RES / "frame_scores.json").write_text(json.dumps(out, ensure_ascii=False))
print("wrote", RES / "frame_scores.json", len(records), "records")
print("query-query cosine:", np.round(Q @ Q.T, 2).tolist())
