"""프레임 넣는 모양 실험(B·C·D)의 타일 그림과 요청 묶음을 만든다. 돈은 들지 않는다. 질문은 2차 화면 질문(types/proposal_r2.json) 그대로이고,
첫 문장(프레임 설명)만 모양에 맞게 바꾼다. A(지금 방식)는 results/h2o_out/types_frames_r2.json을 그대로 쓴다.

  B  맥락 타일: [앞 4·3·2·1초 전 2×2 타일, 가운데 원본 −0.25초, 가운데 원본 +0.25초, 뒤 1·2·3·4초 후 2×2 타일]. 화소 수는 A와 같다(640×360 넷)
  C  촘촘한 타일: 0.125초 간격 16장(가운데 ±1초)을 2×2 타일 4장으로. 화소 수는 A와 같다
  D  직전 값: A와 같은 프레임 4장 + state에 같은 질문들에 대한 직전 3초의 A 답(글). 이것에 끌려가는지 본다
  E  사슬(소유자의 원래 뜻): A와 같은 프레임 4장 + 같은 실행이 바로 앞 초에 낸 답. A를 먼저 돌리지 않고 한 번만 돌린다(16토막)
실행: uv run --no-project --with numpy --with pillow python -I experiments/voice_lostark/layout_prep.py B D   (C는 8fps 프레임이 다 뽑힌 뒤)
"""
import io, json, sys, tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA, SPECS = ROOT / "data", ROOT / "specs"
sys.path.insert(0, str(ROOT.parents[1] / "pipeline"))
from mf import ABOUT  # noqa: E402

P = json.load(open(ROOT / "types/proposal_r2.json"))
n = json.load(open(ROOT / "results/out/per_second.json"))["seconds"]
GAME = "Lost Ark (online action RPG). This recording is a party raid; the recording player is in the party."
PREFIX = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first)."
ABOUT_B = ("`frames` holds 4 images in time order. Image 1 is a 2x2 grid of 4 earlier frames (4, 3, 2 and 1 seconds before now; read left to right, then top to bottom). "
           "Images 2 and 3 are the current moment, 0.5 seconds apart. Image 4 is a 2x2 grid of 4 later frames (1, 2, 3 and 4 seconds after now).")
ABOUT_C = ("`frames` holds 4 images in time order; each image is a 2x2 grid of 4 consecutive frames (read left to right, then top to bottom), "
           "so the 16 frames cover 2 seconds at 0.125-second spacing, earliest first.")


def questions(about):
    qs = {}
    for t in P["types"]:
        q = json.loads(json.dumps(t["screen_question"]))
        assert q["instructions"].startswith(PREFIX), t["key"]
        q["instructions"] = (about + q["instructions"][len(PREFIX):])
        if len(q["instructions"]) > 600:
            raise SystemExit(f"{t['key']}: instructions too long")
        qs[t["key"]] = q
    return qs


def grid_name(t):
    g = min(max(int(round((t - 0.25) / 0.5)), 0), 2 * n - 1)
    return f"{g + 1:06d}.jpg", g


def tile(paths):
    im = Image.new("RGB", (640, 360))
    for i, p in enumerate(paths):
        f = Image.open(p).convert("RGB").resize((320, 180))
        im.paste(f, ((i % 2) * 320, (i // 2) * 180))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88); return b.getvalue()


def dump(name, qs, items, about):
    json.dump({"name": name, "questions": qs, "base_state": {"game": GAME, "viewer_request": "highlight moments of this Lost Ark raid recording", "about": about},
               "items": items, "est_seconds": n / 10}, open(SPECS / f"{name}.json", "w"), ensure_ascii=False)
    print(name, len(items))


arms = sys.argv[1:]
if "B" in arms:
    gd = DATA / "frames_grid"
    with tarfile.open(DATA / "tiles_B.tar", "w") as tar:
        items = []
        for k in range(n):
            c = k + 0.5
            pre = [gd / grid_name(c - d - 0.25)[0] for d in (4, 3, 2, 1)]
            post = [gd / grid_name(c + d - 0.25)[0] for d in (1, 2, 3, 4)]
            for nm, ps in ((f"B_pre_{k:05d}.jpg", pre), (f"B_post_{k:05d}.jpg", post)):
                raw = tile(ps); ti = tarfile.TarInfo(nm); ti.size = len(raw); tar.addfile(ti, io.BytesIO(raw))
            items.append({"k": k, "state": {}, "frame_names": [f"B_pre_{k:05d}.jpg", grid_name(c - 0.25)[0], grid_name(c + 0.25)[0], f"B_post_{k:05d}.jpg"],
                          "seconds": [round(c - 2.5, 2), round(c - 0.25, 2), round(c + 0.25, 2), round(c + 2.5, 2)]})
    dump("layout_B", questions(ABOUT_B), items, ABOUT_B)
if "C" in arms:
    e8 = DATA / "frames_8fps"; m8 = len(list(e8.glob("*.jpg")))
    with tarfile.open(DATA / "tiles_C.tar", "w") as tar:
        items = []
        for k in range(n):
            c = k + 0.5
            ts = [c - 0.9375 + 0.125 * j for j in range(16)]
            idx = [min(max(int(round((t - 0.0625) / 0.125)), 0), m8 - 1) for t in ts]
            names = []
            for q in range(4):
                nm = f"C_{k:05d}_{q}.jpg"; raw = tile([e8 / f"{i + 1:06d}.jpg" for i in idx[4 * q: 4 * q + 4]])
                ti = tarfile.TarInfo(nm); ti.size = len(raw); tar.addfile(ti, io.BytesIO(raw)); names.append(nm)
            items.append({"k": k, "state": {}, "frame_names": names, "seconds": [round(ts[4 * q], 3) for q in range(4)]})
    dump("layout_C", questions(ABOUT_C), items, ABOUT_C)
if "D" in arms:
    A = {r["k"]: r["answers"] for r in json.load(open(ROOT / "results/h2o_out/types_frames_r2.json"))["records"] if r and "answers" in r}
    keys = [t["key"] for t in P["types"]]
    items = []
    for k in range(n):
        c = k + 0.5
        prev = {q: [round(A[j][q]["noul"], 2) for j in range(k - 3, k) if j in A] for q in keys}
        g = [grid_name(c + o)[1] for o in (-0.75, -0.25, 0.25, 0.75)]
        items.append({"k": k, "state": {"earlier_answers": {"note": "Probabilities this model gave to the same questions for the previous 3 seconds of the video (oldest first). They may be wrong.", **prev}},
                      "frames": g, "seconds": [round(c + o, 2) for o in (-0.75, -0.25, 0.25, 0.75)]})
    dump("layout_D", {t["key"]: t["screen_question"] for t in P["types"]}, items, ABOUT)
if "E" in arms:
    # 사슬: A의 요청 묶음(types_frames_r2) 그대로 + 같은 실행이 바로 앞 초에 낸 답. 한 번만 돌린다.
    # A와 다른 것은 앞 답 하나뿐이게 A 묶음을 복사한다(B·C·D는 이 파일의 GAME 문장을 써서 A의 GAME 문장과 달랐다. 25-5절).
    # 영상을 16토막으로 나눠 토막 안은 차례로, 토막끼리는 동시에 묻는다(토막 시작 16곳은 앞 답 없음). 답을 넣는 일은 h2o_voice_modal.py가 한다
    sp = json.load(open(SPECS / "types_frames_r2.json")); sp["name"] = "layout_E"
    sp["chain"] = {"chunks": 16, "key": "previous_answers",
                   "note": "Probabilities this model gave to the same questions for the previous second of the video (its 4 frames overlap the first 2 frames here). They may be wrong."}
    json.dump(sp, open(SPECS / "layout_E.json", "w"), ensure_ascii=False)
    print("layout_E", len(sp["items"]))
