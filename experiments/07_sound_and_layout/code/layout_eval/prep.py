"""프레임 넣는 모양 B·C·D·E를 마인크래프트 12편·로봇 2편에서 다시 잰다(pipeline/README.md 25-7절). 돈은 들지 않는다.

A는 이미 받아 둔 pipeline/runs/gem_mc_*·gem_robot_*의 답이다(같은 판정 모델 판, fp8, 한 방향 읽기). 여기서는 그 요청(requests.json)의
state·질문·프레임을 그대로 두고 모양만 바꾼다. 로스트아크(25-5절)에서 B·C·D의 GAME 문장이 A와 달랐던 실수가 없게, state는 A의 것을 복사한다.

  B  [앞 4·3·2·1초 2×2 타일, 가운데 −0.25초, +0.25초, 뒤 1·2·3·4초 2×2 타일]. 프레임 설명 문장(about, 질문 첫 문장)만 바꾼다
  C  0.125초 간격 16장(가운데 ±1초)을 2×2 타일 4장으로(data/f8, frames8.py). 프레임 설명 문장만 바꾼다
  D  A와 같은 프레임 + state.earlier_answers = 같은 질문에 대한 직전 3초의 A 답
  E  A와 같은 프레임 + 같은 실행이 바로 앞 초에 낸 답(사슬). 영상마다 약 150초씩 토막을 나눠 토막 안은 차례로, 토막끼리는 동시에

묶음 하나 = 모양 × (마인크래프트 | 로봇). 항목의 k = 영상 번호 × 10000 + 그 영상의 k
쓰는 것: data/eval_grid.tar(모든 영상의 격자 프레임, 이름 앞에 영상 이름), data/tiles_B.tar, data/tiles_C.tar, specs/lay_<모양>_<mc|robot>.json, specs/index.json
실행: uv run --no-project --with pillow python -I experiments/layout_eval/prep.py
"""
import io, json, tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parents[1] / "pipeline/runs"
DATA, SPECS = ROOT / "data", ROOT / "specs"
SETS = {"mc": [f"gem_mc_{i:02d}" for i in range(1, 13)], "robot": ["gem_robot_green", "gem_robot_red"]}
PREFIX = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first)."
ABOUT_B = ("`frames` holds 4 images in time order. Image 1 is a 2x2 grid of 4 earlier frames (4, 3, 2 and 1 seconds before now; read left to right, then top to bottom). "
           "Images 2 and 3 are the current moment, 0.5 seconds apart. Image 4 is a 2x2 grid of 4 later frames (1, 2, 3 and 4 seconds after now).")
ABOUT_C = ("`frames` holds 4 images in time order; each image is a 2x2 grid of 4 consecutive frames (read left to right, then top to bottom), "
           "so the 16 frames cover 2 seconds at 0.125-second spacing, earliest first.")
NOTE_D = ("Answers this model gave to the same questions for the previous 3 seconds of the video (oldest first; yes/no questions: probability of yes; "
          "score questions: expected level, counting from 0). They may be wrong.")
NOTE_E = ("Answers this model gave to the same questions for the previous second of the video (its 4 frames overlap the first 2 frames here; yes/no questions: "
          "probability of yes; score questions: expected level, counting from 0). They may be wrong.")
CHUNK = 150


def read(p):
    return json.load(open(p, encoding="utf-8"))


def relabel(questions, about):
    qs = json.loads(json.dumps(questions))
    for k, q in qs.items():
        if q["instructions"].startswith(PREFIX):
            q["instructions"] = about + q["instructions"][len(PREFIX):]
    return qs


def tile(raws):
    im = Image.new("RGB", (640, 360))
    for i, r in enumerate(raws):
        im.paste(Image.open(io.BytesIO(r)).convert("RGB").resize((320, 180)), ((i % 2) * 320, (i // 2) * 180))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88); return b.getvalue()


def add(tar, name, raw):
    ti = tarfile.TarInfo(name); ti.size = len(raw); tar.addfile(ti, io.BytesIO(raw))


def val(a):
    return round(a["noul"], 2) if "noul" in a else round(a["score"], 2)


index = {}
with tarfile.open(DATA / "eval_grid.tar", "w") as tg, tarfile.open(DATA / "tiles_B.tar", "w") as tb, tarfile.open(DATA / "tiles_C.tar", "w") as tc:
    for sname, names in SETS.items():
        req0 = read(RUNS / names[0] / "requests.json")
        items = {a: [] for a in "BCDE"}; bounds = [0]
        for vi, nm in enumerate(names):
            req = read(RUNS / nm / "requests.json")
            assert req["state"] == req0["state"] and req["questions"] == req0["questions"], nm
            with tarfile.open(RUNS / nm / "pack.tar") as tp:
                grid = {Path(m.name).name: tp.extractfile(m).read() for m in tp.getmembers() if m.isfile() and m.name.endswith(".jpg")}
            ng = len(grid)
            for b, raw in grid.items():
                add(tg, f"{nm}_{b}", raw)
            gname = lambda t: f"{min(max(int(round((t - 0.25) / 0.5)), 0), ng - 1) + 1:06d}.jpg"
            f8 = sorted((DATA / "f8" / nm).glob("*.jpg")); n8 = len(f8)
            A = {r["k"]: r["answers"] for r in read(RUNS / nm / "results.json")["records"] if "error" not in r}
            pos = req["positions"]
            for p in pos:
                k, c = p["k"], 0.5 * (p["seconds"][1] + p["seconds"][2])   # 가운데 = 두 번째·세 번째 프레임 사이
                gk = vi * 10000 + k
                base = {"k": gk, "frame_names": [f"{nm}_{x}" for x in p["frames"]], "seconds": p["seconds"]}
                # B
                pre = [grid[gname(c - d - 0.25)] for d in (4, 3, 2, 1)]; post = [grid[gname(c + d - 0.25)] for d in (1, 2, 3, 4)]
                add(tb, f"{nm}_B_pre_{k:05d}.jpg", tile(pre)); add(tb, f"{nm}_B_post_{k:05d}.jpg", tile(post))
                items["B"].append({"k": gk, "state": {}, "frame_names": [f"{nm}_B_pre_{k:05d}.jpg", f"{nm}_{gname(c - 0.25)}", f"{nm}_{gname(c + 0.25)}", f"{nm}_B_post_{k:05d}.jpg"],
                                   "seconds": [round(c - 2.5, 2), round(c - 0.25, 2), round(c + 0.25, 2), round(c + 2.5, 2)]})
                # C
                ts = [c - 0.9375 + 0.125 * j for j in range(16)]
                idx = [min(max(int(round(t / 0.125)), 0), n8 - 1) for t in ts]
                cn = []
                for q in range(4):
                    nmq = f"{nm}_C_{k:05d}_{q}.jpg"; add(tc, nmq, tile([f8[i].read_bytes() for i in idx[4 * q: 4 * q + 4]])); cn.append(nmq)
                items["C"].append({"k": gk, "state": {}, "frame_names": cn, "seconds": [round(ts[4 * q], 3) for q in range(4)]})
                # D
                prev = {q: [val(A[j][q]) for j in range(k - 3, k) if j in A] for q in req["questions"]}
                items["D"].append({**base, "state": {"earlier_answers": {"note": NOTE_D, **prev}}})
                # E
                items["E"].append({**base, "state": {}})
            n0 = bounds[-1]; m = len(pos); parts = max(1, round(m / CHUNK))
            bounds += [n0 + round(m * (i + 1) / parts) for i in range(parts)]
            print(sname, nm, "위치", len(pos), "격자", ng, "8fps", n8, flush=True)
        qa = req0["questions"]
        specs = {"B": (relabel(qa, ABOUT_B), {**req0["state"], "about": ABOUT_B}),
                 "C": (relabel(qa, ABOUT_C), {**req0["state"], "about": ABOUT_C}),
                 "D": (qa, req0["state"]), "E": (qa, req0["state"])}
        for a, (qs, st) in specs.items():
            sp = {"name": f"lay_{a}_{sname}", "questions": qs, "base_state": st, "items": items[a], "est_seconds": len(items[a]) / 7}
            if a == "E":
                sp["chain"] = {"key": "previous_answers", "note": NOTE_E, "bounds": bounds}
            json.dump(sp, open(SPECS / f"lay_{a}_{sname}.json", "w"), ensure_ascii=False)
            longest = max(len(q["instructions"]) for q in qs.values())
            print(f"  lay_{a}_{sname}: {len(items[a])}건, 가장 긴 질문 글 {longest}자" + (f", 토막 {len(bounds) - 1}" if a == "E" else ""), flush=True)
        index[sname] = {"names": names, "bounds": bounds}
json.dump(index, open(SPECS / "index.json", "w"), ensure_ascii=False, indent=1)
