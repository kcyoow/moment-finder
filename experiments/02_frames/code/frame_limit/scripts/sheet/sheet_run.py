"""60장 묶음(접촉 시트) 시험: 이미 뽑아 둔 프레임을 이미지 한 장에 여러 칸으로 묶어 Jev에 묻는다.

jevreel(DeepAleph, MIT)의 방식을 따라 직접 구현했다. 묶는 규칙과 질문 문구는 그 저장소의
sheets.py·search.py를 읽고 맞췄다(한 변 1568px 안에서 칸이 가장 커지는 격자, 칸 왼쪽 위에 T01…,
"Does any tile of `sheet` show this: …" + "Which tile of `sheet` shows this most clearly: …").

요청 두 가지 (시트 × 요청문마다 각각 1건)
- pick  : jevreel 방식. "이 시트에 있는가"(Noul) + "어느 칸인가"(Choice, 보기는 칸 이름과 none)
- tiles : 칸마다 "이 칸에 있는가"(Noul)를 한 요청에 모두 넣는다

프레임은 한 장 조건에서 쓴 것과 같은 파일이다(Charades 1초 간격, QVHighlights 2초 간격).
시트에 60장이 안 차도 60칸 격자를 그대로 써서 칸 크기를 같게 둔다.

실행: <venv>/bin/python experiments/frame_limit/scripts/sheet/sheet_run.py   (환경변수 TYPESAFE_API_KEY 필요)
      --dry-run     시트만 만들고 요청 수와 예상 비용을 출력한다. 보내지 않는다
      --smoke       첫 요청 두 가지만 보내고 응답을 출력한다 (저장하지 않음)
      --tiles 60    시트 한 장에 넣는 프레임 수 (기본 60). 결과 파일 이름에 붙는다
      --tags sta,qv,qv2   묶음 고르기
      --workers 6   동시에 보내는 요청 수
      --prefix tl_  data/tl_<묶음>_items.json의 질문을 이미 만든 시트에 묻고 results/tl_sheet<tiles>_preds.json에 쓴다
      --stride 2    판정 위치를 하나 걸러 쓴다(1초 간격 자료를 2초 간격으로). 파일 이름에 _s2가 붙는다
      --blend       칸마다 프레임 한 장 대신 그 시각 둘레의 4프레임(4장 조건의 프레임)을 겹친 그림을 넣는다. 이름에 _blend가 붙는다
      --out NAME    결과 파일 이름을 따로 정한다(results/NAME_preds.json). 다른 묶음의 결과 파일과 겹치지 않게 할 때 쓴다
      --relate state|question   칸 사이의 시간 관계를 JSON에 적는다. state는 상태의 칸 목록에 앞 칸·뒤 칸을 넣고 앞뒤와 견줘
                    판단하라고 알린다. question은 칸마다 질문에 앞 칸·뒤 칸을 넣는다(칸마다 방식에만). 이름에 _rel…이 붙는다
                    wording은 대조군으로, question과 문구·구조는 같고 앞 칸·뒤 칸만 뺀다
      --kinds tiles 묻는 방법 고르기 (기본 pick,tiles)
      --enhance levels|sharpen|png   시트에 넣기 전의 이미지 처리. levels는 프레임마다 밝기·대비 보정, sharpen은 칸 크기로 줄인 뒤
                    윤곽 살리기, png는 시트를 손실 없이 저장. 이름에 _enh…이 붙는다
      --index NAME  이미 만들어 둔 시트 목록 data/sheets_NAME.json을 그대로 쓴다(motion_sheets.py가 만든 변형: grid12, mdiff, mshift,
                    mtrack, mtrackmark. bundle_sheets.py가 만든 grid12x4). 결과는 results/<prefix>sheet_NAME_preds.json. 시간 간격이 고르지 않은 시트는 그 사실을
                    설명에 적고 칸의 시각을 소수 첫째 자리까지 적는다. 궤적을 그린 시트는 궤적의 뜻을 `overlay`에 적는다
      --tint        겹치기 전에 프레임마다 시간 순서대로 다른 색(빨강, 노랑, 청록, 파랑)을 입히고, 그 뜻을 설명에 적는다. 이름에 _tint가 붙는다
결과: data/sheets<tiles>/…jpg, data/sheets<tiles>.json(시트와 칸의 목록), results/sheet<tiles>_preds.json
      결과 파일이 있으면 성공한 요청은 건너뛰고 나머지만 보낸다
"""
import argparse
import concurrent.futures as cf
import io
import json
import math
import os
import time
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
PRICE_PER_MTOK = 0.042
MAX_SIDE, MIN_SIDE = 1568, 280
SHEET_TOKENS_GUESS = 6000  # jevreel이 잰 값. 보내기 전 어림에만 쓴다


def choose_layout(aspect, tiles):
    """한 변 MAX_SIDE 안에서 tiles칸이 가장 크게 들어가는 격자. 같은 크기면 빈칸이 적은 쪽."""
    best = None
    for cols in range(1, tiles + 1):
        rows = math.ceil(tiles / cols)
        w = math.floor(min(MAX_SIDE / cols, MAX_SIDE * aspect / rows))
        h = math.floor(w / aspect)
        if w < 1 or h < 1:
            continue
        rank = (w * h, -(cols * rows - tiles))
        if best is None or rank > best[0]:
            best = (rank, {"cols": cols, "rows": rows, "tile_w": w, "tile_h": h})
    return best[1]


def tile_ids(count):
    width = max(2, len(str(count)))
    return [f"T{i + 1:0{width}d}" for i in range(count)]


# 시간 순서대로 입히는 색. 값은 프레임의 빨강·초록·파랑에 곱하는 배수이고, 네 프레임의 배수를 더하면 채널마다 4가 된다.
# 그래서 네 프레임에서 그대로인 곳은 원래 색 그대로 나오고, 움직인 곳에만 그 시각의 색이 남는다.
TINTS = [("red", (2.0, 0.4, 0.4)), ("yellow", (1.6, 1.6, 0.0)), ("cyan", (0.0, 1.6, 1.6)), ("blue", (0.4, 0.4, 2.0))]


def overlay(paths, tint=False):
    """여러 프레임을 같은 비중으로 겹친 그림(다중 노출). 움직인 부분은 겹쳐 보이고 멈춘 부분은 또렷하다.

    tint면 프레임마다 시간 순서대로 다른 색을 입힌 뒤 겹친다(빨강 → 노랑 → 청록 → 파랑).
    """
    if tint:
        import numpy as np
        assert len(paths) == len(TINTS)
        total = None
        for path, (_, gain) in zip(paths, TINTS):
            frame = np.asarray(PILImage.open(HERE / path).convert("RGB"), dtype=np.float32) * np.array(gain, dtype=np.float32)
            total = frame if total is None else total + frame
        return PILImage.fromarray(np.clip(total / len(paths) + 0.5, 0, 255).astype(np.uint8))
    out = None
    for n, path in enumerate(paths, 1):
        im = PILImage.open(HERE / path).convert("RGB")
        out = im if out is None else PILImage.blend(out, im, 1 / n)  # n번째 프레임을 1/n만큼 섞으면 평균이 된다
    return out


def variant_of(stride, blend, tint=False, enhance="none"):
    return ((f"_s{stride}" if stride > 1 else "") + ("_tint" if tint else "_blend" if blend else "")
            + ("" if enhance == "none" else f"_enh{enhance}"))


def fix_levels(im):
    """밝기·대비 보정: 가장 어두운 1%와 가장 밝은 1%를 잘라 범위를 펴고(색 균형은 유지), 그래도 어두우면 감마로 밝힌다."""
    im = ImageOps.autocontrast(im, cutoff=1, preserve_tone=True)
    mean = ImageStat.Stat(im.convert("L")).mean[0]
    if mean < 90:  # 평균 밝기를 110쯤으로 끌어올린다. 너무 세지 않게 감마는 0.4까지만
        gamma = max(0.4, math.log(110 / 255) / math.log(max(mean, 1) / 255))
        im = im.point([round(255 * (v / 255) ** gamma) for v in range(256)] * 3)
    return im


def compose(paths, ids, lay, tint=False, enhance="none"):
    canvas = PILImage.new("RGB", (lay["cols"] * lay["tile_w"], lay["rows"] * lay["tile_h"]))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, lay["tile_h"] // 8))
    for slot, (path, tid) in enumerate(zip(paths, ids)):
        x0, y0 = (slot % lay["cols"]) * lay["tile_w"], (slot // lay["cols"]) * lay["tile_h"]
        im = overlay(path, tint) if isinstance(path, list) else PILImage.open(HERE / path).convert("RGB")
        if enhance == "levels":
            im = fix_levels(im)
        im.thumbnail((lay["tile_w"], lay["tile_h"]), PILImage.Resampling.LANCZOS)
        if enhance == "sharpen":  # 줄인 뒤에 윤곽을 살린다
            im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))
        canvas.paste(im, (x0 + (lay["tile_w"] - im.width) // 2, y0 + (lay["tile_h"] - im.height) // 2))
        left, top, right, bottom = draw.textbbox((x0 + 3, y0 + 2), tid, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 2), tid, fill=(255, 255, 255), font=font)
    if min(canvas.size) < MIN_SIDE:  # Jev는 한 변 280px 미만을 받지 않는다
        padded = PILImage.new("RGB", (max(canvas.width, MIN_SIDE), max(canvas.height, MIN_SIDE)))
        padded.paste(canvas, (0, 0))
        canvas = padded
    return canvas


def build_sheets(tags, tiles, stride=1, blend=False, tint=False, enhance="none"):
    """영상마다 프레임을 시간 순서대로 tiles장씩 묶어 시트를 만든다. 돌려주는 값: 시트 목록.

    stride가 2면 판정 위치를 하나 걸러 쓴다. blend면 칸마다 그 시각 둘레의 프레임(window_offsets)을 겹친 그림을 넣는다.
    tint면 겹치기 전에 프레임마다 시간 순서대로 다른 색을 입힌다.
    """
    blend = blend or tint
    name = f"sheets{tiles}{variant_of(stride, blend, tint, enhance)}"
    out_dir = HERE / "data" / name
    sheets = []
    for tag in tags:
        frames = json.loads((HERE / "data" / f"{tag}_frames.json").read_text())
        items = json.loads((HERE / "data" / f"{tag}_items.json").read_text())["items"]
        for video in sorted({it["video"] for it in items}):
            v = frames[video]
            centers = v["centers"]
            paths = [v["frames"][f"{t:.2f}"] for t in centers]
            around = [[v["frames"][f"{round(t + o, 2):.2f}"] for o in v["window_offsets"]] for t in centers] if blend else None
            with PILImage.open(HERE / paths[0]) as im:
                aspect = im.width / im.height
            lay = choose_layout(aspect, tiles)
            picked = list(range(0, len(centers), stride))
            step = centers[stride] - centers[0]
            for s, start in enumerate(range(0, len(picked), tiles)):
                chunk = picked[start:start + tiles]
                ids = tile_ids(tiles)[: len(chunk)]
                file = f"data/{name}/{tag}/{video}/sheet-{s}.{'png' if enhance == 'png' else 'jpg'}"
                (HERE / file).parent.mkdir(parents=True, exist_ok=True)
                image = compose([around[k] if blend else paths[k] for k in chunk], ids, lay, tint, enhance)
                if enhance == "png":  # 손실 없이 저장한다
                    image.save(HERE / file, format="PNG")
                else:
                    image.save(HERE / file, format="JPEG", quality=90)
                sheets.append({"tag": tag, "video": video, "sheet": s, "file": file, "layout": lay,
                               "size": list(image.size), "step_seconds": step, "blend": blend, "tint": tint,
                               "tiles": [{"id": tid, "pos": k, "t": centers[k], "frame": paths[k],
                                          **({"frames": around[k]} if blend else {})}
                                         for tid, k in zip(ids, chunk)]})
    path = HERE / "data" / f"{name}.json"
    kept = [x for x in json.loads(path.read_text())["sheets"] if x["tag"] not in tags] if path.exists() else []
    index = {"_note": "sheet_run.py가 만든 시트 목록. pos는 영상의 위치 번호(0부터), t는 초.",
             "tiles_per_sheet": tiles, "stride": stride, "blend": blend, "tint": tint, "enhance": enhance,
             "sheets": kept + sheets}
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    return sheets, out_dir


def clock(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def neighbours(sheet):
    ids = [t["id"] for t in sheet["tiles"]]
    return {tid: (ids[k - 1] if k else None, ids[k + 1] if k + 1 < len(ids) else None) for k, tid in enumerate(ids)}


def state_of(sheet, image, relate="none"):
    n, step = len(sheet["tiles"]), sheet["step_seconds"]
    if sheet.get("uneven"):  # 움직임을 따라 고른 프레임: 간격이 고르지 않다는 것과 칸마다의 시각을 알린다
        state = {"layout": (f"`sheet` is a contact sheet of {n} frames from one video, in time order from left to right and "
                            "top to bottom. The frames are not evenly spaced in time: more frames were taken where there is "
                            "more motion, so neighbouring tiles can be less than a second or a few seconds apart. Each tile "
                            "has its ID (T01, T02, ...) in its top-left corner. `tiles` maps each ID to its time in the "
                            "video (minutes:seconds).")}
        if sheet.get("marks"):  # Motion-as-Prompt의 안내문(prompt-v2)을 시트의 칸에 맞게 고쳐 씀
            state["overlay"] = ("Some tiles of `sheet` have motion trajectories drawn on moving points as an overlay: each "
                                "colored line shows the path a tracked point moved along between the previous tile and that "
                                "tile, ending in a filled dot at the point's current position. The line colors are only to "
                                "tell different points apart; they carry no other meaning. These overlays are visual aids "
                                "for motion, not real objects in the scene. Use the trajectories to judge how things move.")
        state["tiles"] = {t["id"]: f"{int(t['t']) // 60}:{t['t'] % 60:04.1f}" for t in sheet["tiles"]}
        state["sheet"] = image
        return state
    if sheet.get("bundle"):  # 칸마다 프레임 한 장 대신 그 시각 둘레의 프레임 여러 장을 2×2로 나란히 넣은 시트 (bundle_sheets.py)
        return {"layout": (f"`sheet` is a contact sheet of {n} tiles from one video, {step:g} second(s) apart, in time order "
                           "from left to right and top to bottom. Tiles are separated by black lines. Each tile is a 2x2 "
                           f"block of {sheet['bundle']} consecutive frames taken within about 1.5 seconds around its time, in "
                           "time order inside the block: top-left (earliest), top-right, bottom-left, bottom-right (latest). "
                           "Each tile has its ID (T01, T02, ...) in its top-left corner. `tiles` maps each ID to its time "
                           "in the video."),
                "tiles": {t["id"]: clock(t["t"]) for t in sheet["tiles"]},
                "sheet": image}
    if relate == "state":  # 칸마다 앞 칸·뒤 칸을 적고, 앞뒤와 견줘 판단하라고 미리 알린다
        near = neighbours(sheet)
        return {"layout": (f"`sheet` is a contact sheet of {n} frames from one video, {step:g} second(s) apart, in time order "
                           "from left to right and top to bottom. Each tile has its ID (T01, T02, ...) in its top-left "
                           "corner. `tiles` lists the tiles in time order with each tile's time and the IDs of the tiles "
                           "just before and after it. Consecutive tiles are consecutive moments of the same scene: to judge "
                           "whether an action is happening in a tile, compare it with the tile before and the tile after it."),
                "tiles": [{"id": t["id"], "time": clock(t["t"]), "previous": near[t["id"]][0], "next": near[t["id"]][1]}
                          for t in sheet["tiles"]],
                "sheet": image}
    if sheet.get("tint"):
        names = [name for name, _ in TINTS]
        return {"layout": (f"`sheet` is a contact sheet of {n} tiles from one video, {step:g} second(s) apart, in time order "
                           "from left to right and top to bottom. Each tile is a multiple exposure of 4 frames taken within "
                           "about 2 seconds around its time. Before being overlaid, the frames were tinted by time order: "
                           f"the earliest {names[0]}, the second {names[1]}, the third {names[2]}, the latest {names[3]}. "
                           "Things that stay still keep their normal colors. Things that move leave colored copies or "
                           f"fringes, running from {names[0]} (where it was first) through {names[1]} and {names[2]} to "
                           f"{names[3]} (where it was last). Each tile has its ID (T01, T02, ...) in its top-left corner. "
                           "`tiles` maps each ID to its time in the video."),
                "tiles": {t["id"]: clock(t["t"]) for t in sheet["tiles"]},
                "sheet": image}
    if sheet.get("blend"):
        return {"layout": (f"`sheet` is a contact sheet of {n} tiles from one video, {step:g} second(s) apart, in time order "
                           "from left to right and top to bottom. Each tile is a multiple exposure: 4 frames taken within "
                           "about 2 seconds around its time are overlaid, so things that move appear as ghosted or blurred "
                           "copies and things that stay still look sharp. Each tile has its ID (T01, T02, ...) in its "
                           "top-left corner. `tiles` maps each ID to its time in the video."),
                "tiles": {t["id"]: clock(t["t"]) for t in sheet["tiles"]},
                "sheet": image}
    return {"layout": (f"`sheet` is a contact sheet of {n} frames from one video, {step:g} second(s) apart, in time order "
                       "from left to right and top to bottom. Each tile has its ID (T01, T02, ...) in its top-left "
                       "corner. `tiles` maps each ID to its time in the video."),
            "tiles": {t["id"]: clock(t["t"]) for t in sheet["tiles"]},
            "sheet": image}


def questions_of(kind, sheet, sentence, relate="none"):
    from typesafe_sdk import Choice, Noul
    ids = [t["id"] for t in sheet["tiles"]]
    if kind == "tiles" and relate == "wording":  # 대조군: 질문 문구와 구조만 같고 앞 칸·뒤 칸은 적지 않는다
        return {tid: Noul(instructions={"question": f"Is this happening at `tile` of `sheet`: {sentence}", "tile": tid})
                for tid in ids}
    if kind == "tiles" and relate == "question":  # 질문마다 앞 칸·뒤 칸을 적고 세 칸의 변화로 판단하라고 한다
        near, step = neighbours(sheet), sheet["step_seconds"]
        return {tid: Noul(instructions={
            "question": f"Is this happening at `tile` of `sheet`: {sentence}",
            "tile": tid, "previous": near[tid][0], "next": near[tid][1],
            "how_to_read": (f"`previous`, `tile` and `next` are three consecutive frames, {step:g} second(s) apart. "
                            "Judge from the change across them, not from `tile` alone.")}) for tid in ids}
    if kind == "pick":
        return {"shown": Noul(instructions=f"Does any tile of `sheet` show this: {sentence}"),
                "tile": Choice(instructions=f"Which tile of `sheet` shows this most clearly: {sentence}",
                               criteria={**dict.fromkeys(ids), "none": "No tile matches."})}
    return {tid: Noul(instructions=f"Does tile {tid} of `sheet` show this: {sentence}") for tid in ids}


def run(client, job, relate="none"):
    from typesafe_sdk import Image, TypeSafeAPIError
    sheet, item, kind = job
    rec = {"tag": sheet["tag"], "item": item["id"], "video": sheet["video"], "sheet": sheet["sheet"], "kind": kind,
           "error": None, "attempts": 0}
    state = state_of(sheet, Image((HERE / sheet["file"]).read_bytes()), relate)
    qs = questions_of(kind, sheet, item["sentence"], relate)
    resp = None
    for wait in (0, 3, 8, 20):
        time.sleep(wait)
        rec["attempts"] += 1
        t0 = time.perf_counter()
        try:
            resp = client.system_one(state=state, questions=qs, extra_headers=IMAGES)
            rec["error"] = None
            break
        except TypeSafeAPIError as err:
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
            if "BadRequest" in type(err).__name__ or "Authentication" in type(err).__name__:
                break
    if resp is None:
        return rec
    rec["latency"] = round(time.perf_counter() - t0, 3)
    rec["nouls"] = {qid: a.noul for qid, a in resp.nouls.items()}
    if kind == "pick":
        a = resp.choices["tile"]
        rec["choice"] = {"choice": a.choice, "confidence": a.confidence, "probabilities": dict(a.probabilities)}
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id, model=getattr(resp, "model", None))
    return rec


def key(r):
    return (r["tag"], r["item"], r["sheet"], r["kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=60)
    ap.add_argument("--tags", default="sta,qv,qv2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--tint", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--relate", default="none", choices=["none", "state", "question", "wording"])
    ap.add_argument("--kinds", default="pick,tiles")
    ap.add_argument("--enhance", default="none", choices=["none", "levels", "sharpen", "png"])
    ap.add_argument("--index", default=None)
    args = ap.parse_args()
    tags = args.tags.split(",")
    variant = variant_of(args.stride, args.blend, args.tint, args.enhance)
    index_path = HERE / "data" / f"sheets{args.tiles}{variant}.json"
    have = {s["tag"] for s in json.loads(index_path.read_text())["sheets"]} if index_path.exists() else set()
    if args.index:  # 따로 만들어 둔 시트 목록을 그대로 쓴다
        sheets = [s for s in json.loads((HERE / "data" / f"sheets_{args.index}.json").read_text())["sheets"] if s["tag"] in tags]
    elif set(tags) <= have:  # 이미 만든 시트는 다시 만들지 않는다
        sheets = [s for s in json.loads(index_path.read_text())["sheets"] if s["tag"] in tags]
    else:
        sheets, _ = build_sheets(tags, args.tiles, args.stride, args.blend, args.tint, args.enhance)
    jobs = []
    for tag in tags:
        items = json.loads((HERE / "data" / f"{args.prefix}{tag}_items.json").read_text())["items"]
        for it in items:
            for sh in sheets:
                if sh["tag"] == tag and sh["video"] == it["video"]:
                    jobs += [(sh, it, kind) for kind in args.kinds.split(",")]
    related = "" if args.relate == "none" else f"_rel{args.relate}"
    default_name = f"{args.prefix}sheet_{args.index}" if args.index else f"{args.prefix}sheet{args.tiles}{variant}{related}"
    out_path = HERE / "results" / f"{args.out or default_name}_preds.json"
    done = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())["records"]
        wanted = {(j[0]["tag"], j[1]["id"], j[0]["sheet"], j[2]) for j in jobs}
        lost = [r for r in existing if key(r) not in wanted]
        if lost:  # 이번에 고르지 않은 묶음의 기록이 지워지지 않게 멈춘다
            raise SystemExit(f"{out_path.name}에 이번 실행에 없는 기록 {len(lost)}건이 있다. --out으로 다른 이름을 준다")
        done = {key(r): r for r in existing if not r["error"]}
    todo = [j for j in jobs if (j[0]["tag"], j[1]["id"], j[0]["sheet"], j[2]) not in done]
    n_tiles = sum(len(s["tiles"]) for s in sheets)
    guess = len(todo) * (SHEET_TOKENS_GUESS + 1500)
    sizes = sorted({(s["layout"]["cols"], s["layout"]["rows"], s["layout"]["tile_w"], s["layout"]["tile_h"]) for s in sheets})
    print(f"시트 {len(sheets)}장(프레임 {n_tiles}장), 격자(열, 행, 칸 너비, 칸 높이): {sizes}")
    print(f"요청 {len(jobs)}건 중 보낼 것 {len(todo)}건. 어림: 입력 약 {guess:,}토큰, 약 ${guess * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient
    if args.smoke:
        with TypeSafeClient(model=MODEL) as client:
            for job in todo[:2]:
                rec = run(client, job, args.relate)
                print(json.dumps(rec, ensure_ascii=False)[:1500])
        return
    t0 = time.perf_counter()
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        fresh = {key(r): r for r in pool.map(lambda j: run(client, j, args.relate), todo)}
    wall = time.perf_counter() - t0
    recs = [done.get(k) or fresh[k] for k in ((j[0]["tag"], j[1]["id"], j[0]["sheet"], j[2]) for j in jobs)]
    errors = [r for r in recs if r["error"]]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "tiles_per_sheet": args.tiles, "requests": len(recs), "errors": len(errors),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "wall_seconds_this_run": round(wall, 1), "sent_this_run": len(todo), "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, {wall:.1f}초")
    for kind in ("pick", "tiles"):
        t = [r["input_tokens"] for r in recs if not r["error"] and r["kind"] == kind]
        if t:
            print(f"  {kind}: 요청당 평균 {sum(t) / len(t):,.0f}토큰")
    for e in errors[:5]:
        print("  오류:", e["item"], e["sheet"], e["kind"], e["error"])


if __name__ == "__main__":
    main()
