"""오픈 결정 모델을 채점할 평가 묶음을 만든다. Jev 요청은 없다(받아 둔 답을 함께 담는다). 표준 라이브러리만 쓴다.

Jev 크레딧이 끝나 가서 판정 모델을 오픈 모델로 바꾸려 한다. 이 묶음을 Colab에 올려 후보 모델을 돌리고(open_eval_colab.py),
결과를 내려받아 정답·Jev 답과 견준다(open_eval.py).

과제 (고정 시드로 뽑음)
- mc_gui   : 마인크래프트 프레임 한 장. 창이 열린 300 + 닫힌 300. 질문 2개(창이 열림 / 인벤토리만)
- toast    : 화면 오른쪽 위 모서리 그림. Jev가 켠 것(눈으로 확인한 실제 알림) 전부 + 알림에서 10초 넘게 떨어진 300장. 질문 2개
- fight    : 원본 프레임. 싸움 사건 둘레 프레임 전부 + 견줄 프레임 300장(hl2_run.py의 frames 단계와 같은 프레임). 질문 1개
- charades : 4장 묶음. 새 영상 30개의 모든 위치(한 곳 찾기를 그대로 채점). 위치마다 그 영상의 질문 전부
- qv       : QVHighlights 프레임 한 장. 클립 20개에서 30장씩. 질문 1개
항목: {id, task, video, t, images[경로], kind(one|win4|corner), questions{질문 id: 문장}, gt{질문 id: 0/1 또는 null}, jev{질문 id: 확률}}
      fight 항목에는 role(event = 싸움 사건 둘레, control = 견줄 프레임), charades 항목에는 seconds(프레임 4장의 시각)가 더 있다
질문은 Jev에 보낸 문장의 알맹이(sentence)만 담는다. 모델에 맞는 틀은 open_eval_colab.py가 씌운다.

읽는 곳: experiments/frame_limit 의 data/, results/ (고치지 않는다)
실행: python3 experiments/open_model/open_pack.py
결과: pack/manifest.json, pack/parts/open_pack.zip.NN(45MB씩), pack/parts/sha256.txt
"""
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"  # 프레임과 받아 둔 답이 있는 곳
OUT = HERE / "pack"
SEED, PART = 20261111, 45 * 1024 * 1024
TOAST = {"advancement": "a notification box whose title reads 'Advancement Made!'",
         "recipe": "a notification box whose title reads 'New Recipes Unlocked!'"}


def read(path):
    return json.loads((SRC / path).read_text())


def inside(t, windows, slack=0.0):
    return any(s - slack <= t <= e + slack for s, e in windows)


def main():
    rng = random.Random(SEED)
    items, files = [], {}

    def add(src, task):
        """이미지를 묶음 안의 경로로 옮겨 적는다(같은 파일은 한 번만)."""
        dest = f"img/{task}/{Path(src).parent.name}__{Path(src).name}"
        files[dest] = SRC / src
        return dest

    # mc_gui
    mc = read("data/mc_items.json")["items"]
    frames = read("data/mc_frames.json")
    win = {(it["video"], it["kind"]): it for it in mc}
    recs = [r for r in read("results/mc_preds.json")["records"] if not r["error"]]
    opened = [r for r in recs if inside(r["t"], win[(r["video"], "gui")]["windows"])]
    closed = [r for r in recs if not inside(r["t"], win[(r["video"], "gui")]["windows"])]
    for r in sorted(rng.sample(opened, 300) + rng.sample(closed, 300), key=lambda r: (r["video"], r["t"])):
        g, i = win[(r["video"], "gui")], win[(r["video"], "inv")]
        items.append({"id": f"mc_gui/{r['video']}/{r['t']}", "task": "mc_gui", "video": r["video"], "t": r["t"], "kind": "one",
                      "images": [add(frames[r["video"]]["frames"][f"{r['t']:.2f}"], "mc_gui")],
                      "questions": {"gui": g["sentence"], "inv": i["sentence"]},
                      "gt": {"gui": int(inside(r["t"], g["windows"])), "inv": int(inside(r["t"], i["windows"]))},
                      "jev": {"gui": r["nouls"][g["id"]], "inv": r["nouls"][i["id"]]}})

    # toast
    toasts = read("data/mc_toasts.json")["toasts"]
    corner = [r for r in read("results/hl2_toast-corner.json")["records"] if not r["error"]]
    hot = [r for r in corner if r["advancement"] >= 0.5]
    far = [r for r in corner if r["advancement"] < 0.5 and not any(x["video"] == r["video"] and abs(r["t"] - x["t"]) <= 10 for x in toasts)]
    for r in sorted(hot + rng.sample(far, 300), key=lambda r: (r["video"], r["t"])):
        items.append({"id": f"toast/{r['video']}/{r['t']}", "task": "toast", "video": r["video"], "t": r["t"], "kind": "corner",
                      "images": [add(r["file"], "toast")], "questions": dict(TOAST),
                      "gt": {"advancement": int(r["advancement"] >= 0.5), "recipe": None},
                      "jev": {"advancement": r["advancement"], "recipe": r["recipe"]}})

    # fight
    fight_text = read("results/hl2_frames.json")["meta"]["events"]["fight"]
    for r in sorted((r for r in read("results/hl2_frames.json")["records"] if not r["error"]), key=lambda r: (r["video"], r["t"])):
        items.append({"id": f"fight/{r['video']}/{r['t']}", "task": "fight", "video": r["video"], "t": r["t"], "kind": "one",
                      "role": r["role"], "images": [add(r["file"], "fight")], "questions": {"fight": fight_text},
                      "gt": {"fight": None}, "jev": {"fight": r["fight"]}})

    # charades (4장 묶음, 모든 위치)
    sta = read("data/tl_sta2_items.json")["items"]
    sf = read("data/sta2_frames.json")
    by_video = {}
    for it in sta:
        by_video.setdefault(it["video"], []).append(it)
    jev = {(r["video"], r["t"]): r["nouls"] for r in read("results/full_sta2_preds.json")["records"] if r["frames"] == "win4"}
    for video in sorted(by_video):
        v = sf[video]
        for t in v["centers"]:
            imgs = [add(v["frames"][f"{round(t + o, 2):.2f}"], "charades") for o in v["window_offsets"]]
            qs = by_video[video]
            items.append({"id": f"charades/{video}/{t}", "task": "charades", "video": video, "t": t, "kind": "win4", "images": imgs,
                          "seconds": [round(t + o, 2) for o in v["window_offsets"]],  # Jev에 보낸 상태에 프레임마다 적었던 시각
                          "questions": {q["id"]: q["sentence"] for q in qs},
                          "gt": {q["id"]: int(inside(t, q["windows"])) for q in qs},
                          "jev": {q["id"]: jev[(video, t)][q["id"]] for q in qs}})

    # qv
    qv = read("data/qvm_items.json")["items"]
    qf = read("data/qvm_frames.json")
    jq = {(r["video"], r["t"]): r["nouls"] for r in read("results/full_qvm_preds.json")["records"]}
    for it in qv:
        v = qf[it["video"]]
        for t in sorted(rng.sample(v["centers"], 30)):
            items.append({"id": f"qv/{it['video']}/{t}", "task": "qv", "video": it["video"], "t": t, "kind": "one",
                          "images": [add(v["frames"][f"{t:.2f}"], "qv")], "questions": {it["id"]: it["sentence"]},
                          "gt": {it["id"]: int(inside(t, it["windows"]))}, "jev": {it["id"]: jq[(it["video"], t)][it["id"]]}})

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "parts").mkdir(parents=True)
    note = ("open_pack.py가 만든 평가 묶음. jev는 받아 둔 Jev(jev-1.14.0) 확률, gt는 정답(없으면 null). "
            "Charades·QVHighlights 프레임은 비상업 연구용 자료에서 온 것으로, 실험에만 쓰고 다시 배포하지 않는다")
    manifest = {"_note": note, "seed": SEED, "items": items}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    zip_path = OUT / "open_pack.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEG는 더 줄지 않으므로 묶기만 한다
        z.write(OUT / "manifest.json", "manifest.json")
        for dest, src in sorted(files.items()):
            z.write(src, dest)
    data = zip_path.read_bytes()
    parts = [data[i:i + PART] for i in range(0, len(data), PART)]
    for n, chunk in enumerate(parts):
        (OUT / "parts" / f"open_pack.zip.{n:02d}").write_bytes(chunk)
    zip_path.unlink()
    digest = hashlib.sha256(data).hexdigest()
    (OUT / "parts" / "sha256.txt").write_text(digest + "\n")
    count = {}
    for it in items:
        count[it["task"]] = count.get(it["task"], 0) + 1
    print(f"항목 {len(items)}개 {count}, 이미지 {len(files)}장, 묶음 {len(data) / 1e6:.0f}MB를 {len(parts)}조각으로. sha256 {digest[:16]}…")
    print("질문 수(항목 × 질문):", {t: sum(len(it['questions']) for it in items if it['task'] == t) for t in count})


if __name__ == "__main__":
    main()
