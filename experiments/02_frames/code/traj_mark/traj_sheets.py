"""눈으로 볼 비교 그림(PNG, 최대 6장): 한 줄 = 한 구간, 왼쪽부터 plain | 논문식 경로(img/) | 카메라를 뺀 경로(img_comp/, 우리 변경).

고르는 규칙(그림을 보기 전에 정함)
1 camera_pickup : sta_RX7TC "카메라를 집어 든다"(주석 9~11초). README의 예시였던 8~9초 구간(072)과 그 앞뒤
2 glass_table   : sta_OKHVL "물컵을 탁자에 놓는다"(주석 1~3초). README의 예시였던 0.98~1.96초 구간(012)과 그 앞뒤
3 still_inside  : 카메라가 거의 멈춘(처음→끝 < 너비 1%) 동작 구간 안의 구간 가운데 U가 큰 것, 영상마다 하나씩 4개
4 moving_inside : 동작 구간 안에서 카메라가 가장 많이 움직인 것, 영상마다 하나씩 4개
5 moving_outside: 동작 구간 밖에서 카메라가 가장 많이 움직인 것, 영상마다 하나씩 4개
6 random        : 경로가 하나라도 있는 구간에서 무작위 4개(난수 0)

실행: <Pillow가 있는 파이썬> traj_sheets.py   (index_comp.json을 읽는다) → sheets/<번호>_<이름>.png
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ITEMS = HERE.parent / "frame_limit" / "data" / "tl_sta_items.json"
GAP, HEAD = 8, 54


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def labels_for(x, labels):
    out = []
    for q in labels.get(x["video"], []):
        for s, e in q["windows"]:
            ov = max(0.0, min(x["t"], e) - max(x["t0"], s)) / max(x["t"] - x["t0"], 1e-9)
            if ov > 0:
                out.append((ov, f"{q['sentence']} [{s:g}-{e:g}s]"))
    return [t for _, t in sorted(out, reverse=True)]


def inside(x, labels):
    wins = [w for q in labels.get(x["video"], []) for w in q["windows"]]
    ov = sum(max(0.0, min(x["t"], e) - max(x["t0"], s)) for s, e in wins) / max(x["t"] - x["t0"], 1e-9)
    return ov >= 0.5, ov == 0


def one_per_video(rows, key, n=4):
    out, seen = [], set()
    for x in sorted(rows, key=key, reverse=True):
        if x["video"] not in seen:
            out.append(x)
            seen.add(x["video"])
        if len(out) == n:
            break
    return out


def wrap(d, text, f, width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if cur and d.textlength(trial, font=f) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def sheet(rows, title, labels, path):
    f_title, f_cap, f_head = font(20), font(15), font(14)
    blocks = []
    for x in rows:
        ims = [Image.open(HERE / x[k]).convert("RGB") for k in ("plain", "path", "comp")]
        w, h = ims[0].size
        width = 3 * w + 2 * GAP
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lab = labels_for(x, labels)
        cam = f"camera {x['cam'] * 100:.1f}% of width, RANSAC agree {x['inlier']:.2f}" if x["inlier"] is not None else f"camera {x['cam'] * 100:.1f}%"
        info = (f"{x['video']} {x['t0']:.2f}-{x['t']:.2f}s (frame {x['frame']:03d}) | {cam} | marks {x['paths_raw']} -> {x['paths']}"
                f" | U {x['U']:.3f} R {x['R']:.3f}")
        lines = [(t, "black") for t in wrap(probe, info, f_cap, width - 8)]
        lab_text = ("label: " + " / ".join(lab)) if lab else "label: none (outside all labelled windows)"
        lines += [(t, (160, 0, 0) if lab else (90, 90, 90)) for t in wrap(probe, lab_text, f_cap, width - 8)]
        head = 18 * len(lines) + 20
        block = Image.new("RGB", (width, h + head + 8), "white")
        d = ImageDraw.Draw(block)
        for k, (t, c) in enumerate(lines):
            d.text((4, 2 + 18 * k), t, fill=c, font=f_cap)
        for i, (im, name) in enumerate(zip(ims, ("plain", "paper-style (raw)", "compensated (ours)"))):
            d.text((i * (w + GAP) + 2, head - 16), name, fill=(0, 0, 160), font=f_head)
            block.paste(im, (i * (w + GAP), head + 4))
        blocks.append(block)
    W = max(b.width for b in blocks)
    H = 40 + sum(b.height + 12 for b in blocks)
    out = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(out).text((6, 8), title, fill="black", font=f_title)
    y = 40
    for b in blocks:
        out.paste(b, (0, y))
        y += b.height + 12
    out.save(path, optimize=True)
    return out.size


def main():
    items = json.loads((HERE / "index_comp.json").read_text())["items"]
    labels = {}
    for q in json.loads(ITEMS.read_text())["items"]:
        labels.setdefault(q["video"], []).append(q)
    by = {(x["video"], x["frame"]): x for x in items}
    rx = [x for x in items if x["video"] == "sta_RX7TC" and 7.5 < x["t"] <= 11.5]          # 끝이 8, 9, 10, 11초인 구간
    ok = [x for x in items if x["video"] == "sta_OKHVL" and x["t"] <= 4.0]                 # 0~3.9초의 네 구간
    assert by[("sta_RX7TC", 72)] in rx and by[("sta_OKHVL", 12)] in ok
    ins = [x for x in items if inside(x, labels)[0]]
    outs = [x for x in items if inside(x, labels)[1]]
    rnd = random.Random(0).sample([x for x in items if x["paths_raw"] or x["over"]], 4)   # over > 0 = 끝 프레임 규칙 전의 "보정 경로 있음"
    plan = [("1_camera_pickup", "RX7TC picks up camera (label 9-11s); 072 = README example", rx),
            ("2_glass_table", "OKHVL puts glass on table (label 1-3s); 012 = README example", ok),
            ("3_still_inside", "camera still (<1% width), inside a label, top raw motion",
             one_per_video([x for x in ins if x["cam"] < 0.01], key=lambda x: x["U"])),
            ("4_moving_inside", "largest camera motion inside a label", one_per_video(ins, key=lambda x: x["cam"])),
            ("5_moving_outside", "largest camera motion outside all labels", one_per_video(outs, key=lambda x: x["cam"])),
            ("6_random", "random segments with any mark (seed 0)", sorted(rnd, key=lambda x: (x["video"], x["frame"])))]
    (HERE / "sheets").mkdir(exist_ok=True)
    for name, title, rows in plan:
        size = sheet(rows, title, labels, HERE / "sheets" / f"{name}.png")
        print(name, size, [(x["video"], x["frame"]) for x in rows])


if __name__ == "__main__":
    main()
