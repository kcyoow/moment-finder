"""보완 후보 두 가지를 우리 영상으로 그려 보는 예시 그림을 만든다. Jev 요청은 없다.

예시 1: 칸을 줄이고 키운 격자 + 창 겹치기 (T2SGrid 방식)
  - 지금의 60칸 시트와, 같은 영상을 12칸(4×3) 격자로 만들고 창을 7칸씩 옮겨 5칸이 겹치게 한 것을 나란히 보인다
예시 2: 움직임이 큰 구간만 촘촘히 보기
  - 0.5초 간격 프레임 사이의 변화량(작게 줄인 흑백 프레임의 평균 절댓값 차이)을 재고,
    같은 장수를 변화량이 큰 곳에 더 많이 배정한다(변화량에 바닥값을 더한 가중치의 누적 분포를 같은 간격으로 자름)

예시 3: 점 추적 모델로 움직임을 재고 궤적을 그려 넣기 (Motion-as-Prompt 방식)
  - CoTracker3로 낸 움직임 에너지와 고른 프레임, 그리고 앞 표본 프레임 이후의 궤적을 그린 프레임을 보인다
  - motion_track.py의 결과(data/motion/pass1·pass2)가 있어야 만들어진다
예시 4: 화면 전체 이동을 맞춘 뒤의 프레임 차이 (내 제안. 논문에 나온 방법이 아니다)
  - 손으로 들고 찍은 영상에서, 그냥 뺀 차이와 전체 이동을 맞추고 뺀 차이를 나란히 보인다
예시 3, 4는 실험(motion_sheets.py)이 고른 프레임을 그대로 그린다. 예시 2는 실험보다 먼저 그린 것이라 실험과 두 가지가 다르다
(0.5초 간격 프레임에서 고름, 지수 0.5 대신 바닥값을 더함).

실행: <venv>/bin/python experiments/frame_limit/scripts/motion/examples_make.py   (Pillow, numpy 필요. 글꼴은 macOS의 AppleSDGothicNeo)
결과: examples/ex1_grid12.jpg, examples/ex1_window<k>.jpg(실제로 보낼 크기의 시트), examples/ex2_motion_sampling.jpg,
      examples/ex3_point_tracking.jpg, examples/ex4_shift_compensated.jpg
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import motion_lib as M
import sheet_run as R

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
OUT = HERE / "examples"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # dataviz 검사기로 확인한 두 색 (밝은 바탕, 인접 쌍 통과)
VIDEO_1, VIDEO_2 = "sta_MJYTA", "sta_RFH6M"
VIDEO_3, VIDEO_4 = "sta_RFH6M", "sta_RX7TC"   # 예시 3은 카메라가 고정된 영상, 예시 4는 손으로 들고 찍어 화면이 흔들리는 영상
BAND = "#efeee9"
KOREAN = {"person puts a towel over the shoulder": "수건을 어깨에 걸침", "a person opens the refrigerator": "냉장고를 엶",
          "A person closed the refrigerator door": "냉장고 문을 닫음", "A person took food out of the refrigerator": "음식을 꺼냄",
          "A person peels a banana": "바나나 껍질을 벗김", "A person picks up the camera on the table": "탁자 위의 카메라를 집음"}
WINDOW, STRIDE = 12, 7


def font(size, bold=False):
    want = "Bold" if bold else "Regular"
    for index in range(12):
        try:
            f = ImageFont.truetype(FONT, size, index=index)
        except OSError:
            break
        if f.getname()[1] == want:
            return f
    return ImageFont.truetype(FONT, size)


def fit(im, width=None, height=None):
    scale = width / im.width if width else height / im.height
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)


def example_1():
    frames = json.loads((HERE / "data" / "sta_frames.json").read_text())[VIDEO_1]
    centers = frames["centers"]
    paths = [frames["frames"][f"{t:.2f}"] for t in centers]
    with Image.open(HERE / paths[0]) as im:
        aspect = im.width / im.height
    ids = R.tile_ids(60)
    lay60, lay12 = R.choose_layout(aspect, 60), R.choose_layout(aspect, WINDOW)
    now = R.compose(paths, ids[:len(paths)], lay60)
    rows_used = -(-len(paths) // lay60["cols"])
    now = now.crop((0, 0, now.width, rows_used * lay60["tile_h"]))
    windows = []
    for k, start in enumerate(range(0, len(paths) - WINDOW + STRIDE, STRIDE), 1):
        chunk = list(range(start, min(start + WINDOW, len(paths))))
        sheet = R.compose([paths[i] for i in chunk], [ids[i] for i in chunk], lay12)
        sheet.save(OUT / f"ex1_window{k}.jpg", quality=90)
        windows.append((chunk, sheet))

    width, pad = 1800, 28
    a = fit(now, width=width - 2 * pad)
    w_each = (width - 3 * pad) // 2
    shown = []
    for chunk, sheet in windows[:2]:
        im = sheet.copy()
        d = ImageDraw.Draw(im)
        other = set(windows[1][0]) if chunk is windows[0][0] else set(windows[0][0])
        for slot, i in enumerate(chunk):  # 두 창에 모두 들어가는 칸에 테두리
            if i in other:
                x0, y0 = (slot % lay12["cols"]) * lay12["tile_w"], (slot // lay12["cols"]) * lay12["tile_h"]
                d.rectangle((x0 + 3, y0 + 3, x0 + lay12["tile_w"] - 4, y0 + lay12["tile_h"] - 4), outline=ORANGE, width=7)
        shown.append(fit(im, width=w_each))
    title, label, note = font(40, True), font(28, True), font(24)
    h = pad + 56 + 44 + a.height + 36 + 44 + 34 + shown[0].height + pad + 40
    canvas = Image.new("RGB", (width, h), SURFACE)
    d = ImageDraw.Draw(canvas)
    y = pad
    d.text((pad, y), "예시 1. 칸을 줄이고 키운 격자에 창을 겹치기", fill=INK, font=title)
    y += 62
    d.text((pad, y), f"지금: 60칸 격자 한 장. 칸 하나 {lay60['tile_w']}×{lay60['tile_h']}px ({len(paths)}초 영상이라 {len(paths)}칸만 참)", fill=INK, font=label)
    y += 44
    canvas.paste(a, (pad, y))
    y += a.height + 36
    d.text((pad, y), f"바꾼 것: 12칸 격자 여러 장. 칸 하나 {lay12['tile_w']}×{lay12['tile_h']}px (넓이 약 {lay12['tile_w'] * lay12['tile_h'] / (lay60['tile_w'] * lay60['tile_h']):.0f}배). "
                     f"창을 {STRIDE}칸씩 옮겨 {WINDOW - STRIDE}칸이 겹침", fill=INK, font=label)
    y += 44
    d.text((pad, y), f"주황 테두리 = 두 창에 모두 들어가는 칸. 이 영상은 창 {len(windows)}장이 필요함 (그림에는 앞의 두 장)", fill=INK2, font=note)
    y += 34
    for k, im in enumerate(shown):
        canvas.paste(im, (pad + k * (w_each + pad), y))
    y += shown[0].height + 10
    for k, (chunk, _) in enumerate(windows[:2]):
        d.text((pad + k * (w_each + pad), y), f"창 {k + 1}: {centers[chunk[0]]:g}~{centers[chunk[-1]]:g}초", fill=INK2, font=note)
    canvas.save(OUT / "ex1_grid12.jpg", quality=90)
    return {"video": VIDEO_1, "tile_now": (lay60["tile_w"], lay60["tile_h"]), "tile_new": (lay12["tile_w"], lay12["tile_h"]),
            "windows": len(windows), "sheet_size": windows[0][1].size}


def motion_energy(frames):
    """0.5초 간격 프레임(x.25초, x.75초) 사이의 변화량."""
    ts = sorted(float(t) for t in frames["frames"] if abs((float(t) * 4) % 2 - 1) < 1e-6)
    small = [np.asarray(Image.open(HERE / frames["frames"][f"{t:.2f}"]).convert("L").resize((64, 96)), dtype=np.float32) for t in ts]
    return ts, [float(np.abs(b - a).mean()) for a, b in zip(small, small[1:])]


def pick_by_motion(ts, energy, count):
    """같은 장수를 변화량이 큰 곳에 더 많이: (변화량 + 바닥값)의 누적 분포를 같은 간격으로 자른다."""
    e = np.array([energy[0]] + [(a + b) / 2 for a, b in zip(energy, energy[1:])] + [energy[-1]])  # 프레임마다 앞뒤 구간의 평균
    weight = e + np.median(e)  # 바닥값: 움직임이 없는 곳도 가끔은 본다
    cdf = np.cumsum(weight) / weight.sum()
    chosen = []
    for q in (np.arange(count) + 0.5) / count:
        k = int(np.searchsorted(cdf, q))
        while k in chosen and k + 1 < len(ts):
            k += 1
        chosen.append(min(k, len(ts) - 1))
    return sorted(set(chosen))


def example_2():
    frames = json.loads((HERE / "data" / "sta_frames.json").read_text())[VIDEO_2]
    items = [it for it in json.loads((HERE / "data" / "tl_sta_items.json").read_text())["items"] if it["video"] == VIDEO_2]
    ts, energy = motion_energy(frames)
    uniform = frames["centers"]
    chosen = pick_by_motion(ts, energy, len(uniform))
    picked = [ts[k] for k in chosen]
    duration = frames["duration"]

    width, pad, left = 1800, 28, 300
    plot_w, plot_h = width - left - pad - 20, 300
    title, label, note, small = font(40, True), font(28, True), font(24), font(21)
    zoom_end = 6.0
    strip_h = 250
    h = pad + 62 + 40 + 60 + plot_h + 50 + 2 * 54 + 30 + 2 * (44 + strip_h) + pad + 20
    canvas = Image.new("RGB", (width, h), SURFACE)
    d = ImageDraw.Draw(canvas)
    x_of = lambda t: left + t / duration * plot_w
    y = pad
    d.text((pad, y), "예시 2. 움직임이 큰 구간만 촘촘히 보기", fill=INK, font=title)
    y += 62
    d.text((pad, y), "0.5초마다 프레임이 얼마나 달라졌는지 재고, 같은 장수를 많이 달라진 곳에 더 배정한다 (카메라가 고정된 30초 영상)", fill=INK2, font=note)
    y += 40
    # 정답 구간 표시 (회색 띠와 글)
    top = y + 60
    for it in items:
        s, e = it["windows"][0]
        d.rectangle((x_of(s), top, x_of(e), top + plot_h), fill="#efeee9")
    names = {"person puts a towel over the shoulder": "수건을 어깨에 걸침", "a person opens the refrigerator": "냉장고를 엶",
             "A person closed the refrigerator door": "냉장고 문을 닫음", "A person took food out of the refrigerator": "음식을 꺼냄"}
    front = [it for it in items if it["windows"][0][0] < 5]
    d.text((x_of(0) + 4, y + 26), f"0~{max(it['windows'][0][1] for it in front):g}초: "
           + ", ".join(names.get(it["sentence"], it["sentence"]) for it in front), fill=INK2, font=small)
    for it in items:
        if it["windows"][0][0] >= 5:
            s, e = it["windows"][0]
            d.text((x_of(s) - 30, y + 26), f"{s:g}~{e:g}초: {names.get(it['sentence'], it['sentence'])}", fill=INK2, font=small)
    # 축과 눈금
    peak = max(energy)
    for frac in (0.5, 1.0):
        gy = top + plot_h - frac * (plot_h - 10)
        d.line((left, gy, left + plot_w, gy), fill=GRID, width=1)
    d.line((left, top + plot_h, left + plot_w, top + plot_h), fill=INK2, width=1)
    d.text((pad, top + plot_h // 2 - 30), "프레임 사이의", fill=INK2, font=small)
    d.text((pad, top + plot_h // 2 - 4), "변화량", fill=INK2, font=small)
    d.text((pad, top + plot_h // 2 + 22), "(0.5초 간격)", fill=INK2, font=small)
    bar_w = plot_w / duration * 0.5 - 2
    for t, e in zip(ts, energy):  # 구간 [t, t+0.5]
        x0 = x_of(t) + 1
        bh = e / peak * (plot_h - 10)
        d.rounded_rectangle((x0, top + plot_h - bh, x0 + bar_w, top + plot_h), radius=4, fill=BLUE,
                            corners=(True, True, False, False))
    for sec in range(0, int(duration) + 1, 5):
        d.line((x_of(sec), top + plot_h, x_of(sec), top + plot_h + 6), fill=INK2, width=1)
        d.text((x_of(sec) - 12, top + plot_h + 10), f"{sec}초", fill=INK2, font=small)
    y = top + plot_h + 50
    # 표본 위치 두 줄
    for name, times, color in ((f"지금: 1초마다 한 장 ({len(uniform)}장)", uniform, INK2),
                               (f"바꾼 것: 같은 {len(picked)}장을 움직임 따라", picked, ORANGE)):
        d.text((pad, y + 8), name, fill=INK, font=small)
        d.line((left, y + 22, left + plot_w, y + 22), fill=GRID, width=1)
        for t in times:
            d.ellipse((x_of(t) - 6, y + 16, x_of(t) + 6, y + 28), fill=color, outline=SURFACE, width=2)
        y += 54
    y += 30
    # 앞부분을 확대해 실제 프레임으로
    for name, times in ((f"지금, 처음 {zoom_end:g}초에 들어가는 프레임", [t for t in uniform if t <= zoom_end]),
                        (f"바꾼 것, 처음 {zoom_end:g}초에 들어가는 프레임", [t for t in picked if t <= zoom_end])):
        d.text((pad, y), f"{name}: {len(times)}장", fill=INK, font=label)
        y += 44
        x = pad
        for t in times:
            im = fit(Image.open(HERE / frames["frames"][f"{t:.2f}"]).convert("RGB"), height=strip_h - 30)
            if x + im.width > width - pad:
                break
            canvas.paste(im, (x, y))
            d.text((x + 2, y + im.height + 2), f"{t:g}초", fill=INK2, font=small)
            x += im.width + 6
        y += strip_h
    canvas.save(OUT / "ex2_motion_sampling.jpg", quality=90)
    first = sum(t <= zoom_end for t in picked)
    return {"video": VIDEO_2, "frames_total": len(picked), "uniform_in_zoom": sum(t <= zoom_end for t in uniform),
            "picked_in_zoom": first, "largest_gap_seconds": round(max(b - a for a, b in zip(picked, picked[1:])), 2)}


def energy_chart(d, box, duration, times, energy, items, lines, peak=None):
    """움직임 에너지 막대 그림 한 칸. 회색 띠는 정답 구간. 프레임 t의 막대는 앞 프레임과 t 사이에 놓는다."""
    left, top, plot_w, plot_h = box
    x_of = lambda t: left + t / duration * plot_w
    small = font(21)
    for it in items:
        s, e = it["windows"][0]
        d.rectangle((x_of(s), top, x_of(e), top + plot_h), fill=BAND)
    peak = peak or max(energy)
    for frac in (0.5, 1.0):
        gy = top + plot_h - frac * (plot_h - 10)
        d.line((left, gy, left + plot_w, gy), fill=GRID, width=1)
    d.line((left, top + plot_h, left + plot_w, top + plot_h), fill=INK2, width=1)
    step = times[1] - times[0]
    for t, e in zip(times, energy):
        x1 = x_of(t)
        x0 = max(left, x1 - plot_w / duration * step + 1.5)
        bh = min(e / peak, 1.0) * (plot_h - 10)
        if bh >= 1:
            d.rectangle((x0, top + plot_h - bh, x1, top + plot_h), fill=BLUE)
    for k, text in enumerate(lines):
        d.text((28, top + plot_h // 2 - 13 * len(lines) + 26 * k), text, fill=INK2, font=small)
    for sec in range(0, int(duration) + 1, 5):
        d.line((x_of(sec), top + plot_h, x_of(sec), top + plot_h + 6), fill=INK2, width=1)
        d.text((x_of(sec) - 12, top + plot_h + 10), f"{sec}초", fill=INK2, font=small)
    return x_of


def dot_row(d, x_of, box, y, name, times, color, hollow=()):
    left, _, plot_w, _ = box
    d.text((28, y + 8), name, fill=INK, font=font(21))
    d.line((left, y + 22, left + plot_w, y + 22), fill=GRID, width=1)
    for t in times:
        if t in hollow:  # 균등하게 깐 프레임은 속을 비운다
            d.ellipse((x_of(t) - 6, y + 16, x_of(t) + 6, y + 28), fill=SURFACE, outline=color, width=3)
        else:
            d.ellipse((x_of(t) - 6, y + 16, x_of(t) + 6, y + 28), fill=color, outline=SURFACE, width=2)


def band_note(items):
    return "회색 띠 = 정답 구간: " + ", ".join(
        f"{it['windows'][0][0]:g}~{it['windows'][0][1]:g}초 {KOREAN.get(it['sentence'], it['sentence'])}"
        for it in sorted(items, key=lambda it: it["windows"][0]))


def example_3():
    select = json.loads((HERE / "data" / "motion_select.json").read_text())[VIDEO_3]
    index = HERE / "data" / "sheets_mtrackmark.json"
    if "track" not in select or not index.exists():
        return "추적 결과가 없어 만들지 않음"
    sheet = next(s for s in json.loads(index.read_text())["sheets"] if s["video"] == VIDEO_3)
    frames = json.loads((HERE / "data" / "sta_frames.json").read_text())[VIDEO_3]
    items = [it for it in json.loads((HERE / "data" / "tl_sta_items.json").read_text())["items"] if it["video"] == VIDEO_3]
    times, energy = select["times"], select["track"]["energy"]
    picked = [times[k] for k in select["track"]["indices"]]
    anchors = {times[k] for k in select["track"]["anchors"]}
    duration = frames["duration"]
    width, pad, left = 1800, 28, 300
    box = (left, 0, width - left - pad - 20, 260)
    strip_h = 330
    tiles = [t for t in sheet["tiles"] if "marked_" in t["frame"]]
    first = next((n for n, t in enumerate(sheet["tiles"]) if "marked_" in t["frame"]), 0)
    shown = sheet["tiles"][first:first + 7]
    h = pad + 62 + 2 * 34 + 20 + box[3] + 50 + 2 * 54 + 30 + 44 + 34 + strip_h + pad
    canvas = Image.new("RGB", (width, h), SURFACE)
    d = ImageDraw.Draw(canvas)
    note = font(24)
    y = pad
    d.text((pad, y), "예시 3. 점 추적 모델로 움직임을 재고, 궤적을 그려 넣기", fill=INK, font=font(40, True))
    y += 62
    d.text((pad, y), "화면에 점 100개를 찍어 따라가고(CoTracker3), 카메라 움직임을 뺀 점들의 속력·가속·방향 전환을 더해 움직임 에너지를 낸다", fill=INK2, font=note)
    y += 34
    d.text((pad, y), band_note(items), fill=INK2, font=note)
    y += 34 + 20
    box = (left, y, box[2], box[3])
    x_of = energy_chart(d, box, duration, times, energy, items, ["움직임 에너지", "(8fps, 0~1)"])
    y += box[3] + 50
    dot_row(d, x_of, box, y, f"지금: 1초마다 한 장 ({len(frames['centers'])}장)", frames["centers"], INK2)
    y += 54
    dot_row(d, x_of, box, y, f"바꾼 것: 같은 {len(picked)}장", picked, ORANGE, hollow=anchors)
    y += 54 + 30
    d.text((pad, y), f"고른 프레임에 그려 넣은 궤적 ({shown[0]['t']:.1f}~{shown[-1]['t']:.1f}초). 선 = 앞 프레임 이후에 점이 움직인 길, 끝의 점 = 지금 위치", fill=INK, font=font(28, True))
    y += 44
    d.text((pad, y), f"속이 빈 점 = 균등하게 깐 프레임(4분의 1), 채운 점 = 에너지가 높아 고른 프레임. 궤적이 그려진 칸은 {len(tiles)}/{len(sheet['tiles'])}개", fill=INK2, font=note)
    y += 34
    x = pad
    for t in shown:
        im = fit(Image.open(HERE / t["frame"]).convert("RGB"), height=strip_h - 30)
        if x + im.width > width - pad:
            break
        canvas.paste(im, (x, y))
        d.text((x + 2, y + im.height + 2), f"{t['t']:.2f}초", fill=INK2, font=font(21))
        x += im.width + 6
    canvas.save(OUT / "ex3_point_tracking.jpg", quality=90)
    return {"video": VIDEO_3, "frames_total": len(picked), "anchors": len(anchors), "tiles_with_marks": len(tiles)}


def example_4():
    select = json.loads((HERE / "data" / "motion_select.json").read_text())[VIDEO_4]
    frames = json.loads((HERE / "data" / "sta_frames.json").read_text())[VIDEO_4]
    items = [it for it in json.loads((HERE / "data" / "tl_sta_items.json").read_text())["items"] if it["video"] == VIDEO_4]
    times = select["times"]
    plain, fixed = np.array(select["mdiff"]["energy"]), np.array(select["mshift"]["energy"])
    by_diff = [times[k] for k in select["mdiff"]["indices"]]
    by_shift = [times[k] for k in select["mshift"]["indices"]]
    duration = frames["duration"]
    worst = int(np.argmax(plain - fixed))  # 맞추기 전과 뒤의 차이가 가장 큰 프레임 쌍
    dy, dx = select["mshift"]["shifts"][worst]
    video, ts, _ = M.timeline(VIDEO_4)
    a, b = video[worst - 1].astype(np.float32), video[worst].astype(np.float32)
    hh, ww = a.shape[:2]
    ys, xs = slice(max(dy, 0), hh + min(dy, 0)), slice(max(dx, 0), ww + min(dx, 0))
    yp, xp = slice(max(-dy, 0), hh + min(-dy, 0)), slice(max(-dx, 0), ww + min(-dx, 0))
    raw_diff = np.abs(b - a).mean(axis=2)
    fixed_diff = np.zeros_like(raw_diff)
    fixed_diff[ys, xs] = np.abs(b[ys, xs] - a[yp, xp]).mean(axis=2)
    as_image = lambda x: Image.fromarray(np.clip(x * 3, 0, 255).astype(np.uint8)).convert("RGB")
    pictures = [(f"{ts[worst - 1]:.3f}초", Image.fromarray(video[worst - 1])), (f"{ts[worst]:.3f}초", Image.fromarray(video[worst])),
                (f"그냥 뺀 차이 (평균 {plain[worst]:.1f})", as_image(raw_diff)), (f"전체 이동을 맞추고 뺀 차이 (평균 {fixed[worst]:.1f})", as_image(fixed_diff))]

    width, pad, left = 1800, 28, 300
    plot_w, plot_h = width - left - pad - 20, 200
    strip_h = 420
    note = font(24)
    h = pad + 62 + 3 * 34 + 20 + 2 * (plot_h + 56) + 3 * 54 + 30 + 44 + 34 + strip_h + 40 + pad
    canvas = Image.new("RGB", (width, h), SURFACE)
    d = ImageDraw.Draw(canvas)
    y = pad
    d.text((pad, y), "예시 4. 화면 전체 이동을 맞춘 뒤의 프레임 차이로 고르기 (제안)", fill=INK, font=font(40, True))
    y += 62
    d.text((pad, y), "손으로 들고 찍으면 화면 전체가 흔들려 프레임 차이가 커진다. 두 프레임이 가장 잘 겹치는 가로·세로 이동을 구해 맞춘 뒤에 뺀다", fill=INK2, font=note)
    y += 34
    d.text((pad, y), "(추적 모델 없이 계산. 회전과 확대는 맞추지 못함. 이 영상은 손으로 들고 찍어 3초쯤부터 화면이 계속 흔들린다)", fill=INK2, font=note)
    y += 34
    d.text((pad, y), band_note(items), fill=INK2, font=note)
    y += 34 + 20
    peak = float(plain.max())
    box = (left, y, plot_w, plot_h)
    x_of = energy_chart(d, box, duration, times, plain, items, ["그냥 뺀 차이", "(예시 2의 방식)"], peak)
    y += plot_h + 56
    box = (left, y, plot_w, plot_h)
    energy_chart(d, box, duration, times, fixed, items, ["전체 이동을", "맞추고 뺀 차이", "(위와 같은 눈금)"], peak)
    y += plot_h + 56
    dot_row(d, x_of, box, y, f"지금: 1초마다 한 장 ({len(frames['centers'])}장)", frames["centers"], INK2)
    y += 54
    dot_row(d, x_of, box, y, f"그냥 뺀 차이로 고른 {len(by_diff)}장", by_diff, BLUE)
    y += 54
    dot_row(d, x_of, box, y, f"맞추고 뺀 차이로 고른 {len(by_shift)}장", by_shift, ORANGE)
    y += 54 + 30
    d.text((pad, y), f"차이가 가장 많이 줄어든 프레임 쌍: 화면 전체가 세로 {abs(dy)}화소, 가로 {abs(dx)}화소 움직임 (긴 변 512화소 기준)", fill=INK, font=font(28, True))
    y += 44
    d.text((pad, y), "차이 그림은 밝을수록 많이 달라진 곳이다(3배로 밝힘). 맞추고 빼면 배경의 윤곽이 사라지고 실제로 움직인 것만 남는다", fill=INK2, font=note)
    y += 34
    x = pad
    for name, im in pictures:
        im = fit(im, height=strip_h)
        canvas.paste(im, (x, y))
        d.text((x + 2, y + im.height + 6), name, fill=INK2, font=font(21))
        x += max(im.width, 400) + 24
    canvas.save(OUT / "ex4_shift_compensated.jpg", quality=90)
    return {"video": VIDEO_4, "mean_plain": round(float(plain.mean()), 2), "mean_fixed": round(float(fixed.mean()), 2),
            "pair": [round(float(ts[worst - 1]), 3), round(float(ts[worst]), 3)], "shift_dy_dx": [dy, dx],
            "same_frames": len(set(by_diff) & set(by_shift)), "frames_total": len(by_shift)}


def main():
    OUT.mkdir(exist_ok=True)
    print("예시 1:", example_1())
    print("예시 2:", example_2())
    print("예시 3:", example_3())
    print("예시 4:", example_4())


if __name__ == "__main__":
    main()
