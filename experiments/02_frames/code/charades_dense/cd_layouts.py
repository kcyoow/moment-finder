"""charades_dense의 모양, 문장, 그림 만들기, 요청 본문. 이 Mac(준비·미리 세기)과 Modal 컨테이너(묻기)가 함께 쓴다.

모듈 맨 위에서는 표준 라이브러리만 불러온다(Pillow는 그림을 만드는 함수 안에서만). 결과를 보기 전에 정했고, 본 뒤에는 고치지 않는다.

위치 k(초)는 앞선 실험(prompt_study)의 위치 그대로다. 정답은 그 시각 k가 정답 구간 안인가.
프레임은 0.125초 격자(0.25, 0.375, …, K+0.75초)에서 뽑았다. 파일 이름은 밀리초(000250.jpg = 0.25초).
위치 k의 "1초"는 [k−0.5, k+0.5)이고, 그 안의 격자 프레임 8장은 k−0.5, k−0.375, …, k+0.375초다.
"""
import base64
import io
import json
import math

TICK = 0.125
CAP = 3_700_000            # 서버의 그림 한도(MAX_PIXELS = SHIM_MAX_PIXELS). 보내는 그림이 모두 이보다 작아 서버가 줄이지 않는다
                           # (가장 큰 것: 4:3 영상의 grid8 원본 크기 3096×1160 = 3,591,360화소. H2O 기본 한도는 1,638,400, 마인크래프트 실행은 2,000,000)
GAP = 8                    # grid8·sheet32에서 프레임 사이의 검은 띠(화소). mc_dense와 같다
SHEET32_PIXELS = 2_000_000 # sheet32 한 장의 화소 상한(띠 포함). 마인크래프트 실행의 그림 한도와 같은 값
SHEET60_SIDE = 1568        # jevreel 0.2.0의 시트 한 변(media.MAX_SIDE)
SHEET60_TILES = 60
PIXELS_PER_TOKEN = 32 * 32 # H2O(Qwen3.5 그림 탑): 16px 조각을 2×2로 합쳐 토큰 하나
MIN_PIXELS = 65_536        # H2O preprocessor_config의 shortest_edge


def fname(t):
    return f"{round(t * 1000):06d}.jpg"


# ---- 문장 (mc_dense·prompt_study의 것을 그대로 쓴 곳은 그렇게 적었다)
HEAD_SEP = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). "   # prompt_study base 그대로
HEAD_GRID8 = ("`grid` is one picture that holds 8 consecutive frames from a video, about 0.125 seconds apart, laid out in 2 rows of 4 in time order: "
              "top row left to right (earliest first), then bottom row left to right (latest last). ")                 # mc_dense 그대로
HEAD_GRID8_WIDE = HEAD_GRID8.replace("about 0.125 seconds apart", "about 0.25 seconds apart")   # 덧붙임(2026-11-14): 간격만 바꿈
SHOW = "Do these frames show the following: {s}?"                        # prompt_study base·mc_dense show 그대로
ANY = "Does at least one of these frames show the following: {s}?"       # mc_dense any 그대로
HEAD_S32 = ("`sheet` is one picture that holds {n} consecutive frames from a video, about 0.125 seconds apart, laid out in {r} rows of 8 "
            "in time order: top row left to right (earliest first), then each next row left to right (latest last). Each row is one second. "
            "`rows` gives the time of each frame in the video, in seconds, one list per row. ")                          # 새로 씀(grid8 문장을 늘림)
ROW = "Do the 8 frames in row {j} of `sheet` (row 1 is the top row) show the following: {s}?"                            # 새로 씀
JR_LAYOUT = "Tiles are frames 1 second apart, in time order: left to right, then top to bottom. Each tile's ID is printed in its top-left corner."  # jevreel 0.2.0 states.layout_note
JR_SHOWN = "Does any tile of `sheet` show this: {s}"                    # jevreel 0.2.0 search._sheet_questions
JR_PICK = "Which tile of `sheet` shows this most clearly: {s}"          # 같은 곳
JR_NONE = "No tile matches."                                            # 같은 곳
T60_LAYOUT = ("`sheet` is a contact sheet of {n} frames from one video, 1 second(s) apart, in time order from left to right and top to bottom. "
              "Each tile has its ID (T01, T02, ...) in its top-left corner. `tiles` maps each ID to its time in the video.")   # frame_limit sheet_run.py 그대로
T60_Q = "Does tile {tid} of `sheet` show this: {s}"                     # sheet_run.py의 칸마다 물음 그대로
# Q3 (경로 그림). 물음 문장은 (a)~(c)에서 같다. 경로 설명 한 줄은 경로가 그려진 그림에만, 물음 바로 앞에 넣는다
# (MaP 공식 코드 track_marker.legend_text: 물음 바로 앞(프레임 뒤)에 두고, 그린 경로가 없으면 빈 글. 논문 본문은 "a corresponding text prompt"라고만 적고 글을 싣지 않았다)
HEAD_ONE = "`frame` is one frame taken from a video. "                   # mc_dense ONE_Q 그대로
Q_ONE = "Does this frame show the following: {s}?"                      # 같은 곳
LEGEND = ("The image has motion trajectories drawn on some moving object points: each line shows the path the object point moved along "
          "during the 1 second before this frame, and each object point's line has its own single color with a larger filled dot of that "
          "color marking where the point is in this frame. Use these trajectories to reason about how things move. ")
# ↑ legend_text(prompt-v1, color_mode="row", span="frame")에서 "between the previous frame and this frame"만
#   "during the 1 second before this frame"으로 바꿨다(앞 프레임이 없는 한 장 변형이라서). (b)와 (c)에 같은 글을 쓴다
HEAD_Q3SEP = "`frames` holds 4 consecutive frames from a video, 0.25 seconds apart, in time order (earliest first). "   # mc_dense sep 머리(gap 0.25)

# ---- 모양. 적힌 순서대로 돈다(중요한 것부터. 시간이 모자라면 뒤의 것부터 건너뛴다)
GRID_OFFSETS = tuple((j - 4) * TICK for j in range(8))   # −0.5 … +0.375
WIDE_OFFSETS = tuple((2 * j - 7) * TICK for j in range(8))   # −0.875, −0.625, … +0.875 (0.25초 간격, 폭 1.75초). 덧붙임(2026-11-14)
FIRST_TICK = 0.25          # 묶음에 있는 첫 프레임. 마지막은 그 영상의 마지막 위치 + 0.75초
LAYOUTS = {
    "sep4":          {"kind": "sep", "offsets": (-0.75, -0.25, 0.25, 0.75), "what": "지금 방법: 0.5초 간격 4장을 따로(prompt_study base와 같은 프레임·문장)"},
    "grid8_small":   {"kind": "grid8", "scale": 0.7071, "what": "그 1초의 8장을 4×2 한 그림으로, 화소를 절반으로 줄여 4장 따로와 같은 화소 수"},
    "q3_plain":      {"kind": "q3one", "img": "plain", "what": "Q3 (a) 구간의 끝 프레임 한 장(traj_mark img/, 긴 변 512)"},
    "q3_path":       {"kind": "q3one", "img": "path", "what": "Q3 (b) 같은 프레임 + 논문식 경로(traj_mark img/). 한 장만 넣는 것은 우리 변형"},
    "q3_comp":       {"kind": "q3one", "img": "comp", "what": "Q3 (c) 같은 프레임 + 카메라를 뺀 경로(traj_mark img_comp/, 우리 변형)"},
    "q3_sep4":       {"kind": "q3sep", "what": "Q3 (d) 같은 1초를 고르게 덮는 4장을 따로(같은 프레임 줄기, 같은 크기)"},
    "sheet32":       {"kind": "sheet32", "per_sheet": 4, "what": "4초(위치 4곳)의 32장을 8×4 한 그림으로(한 줄 = 1초). 창마다 있는가 + 줄마다 있는가. 우리 변형"},
    "sheet60_jr":    {"kind": "sheet60", "ask": "jr", "what": "1초 간격 60칸 시트 + jevreel 0.2.0 1단계의 상태·물음(있는가 + 어느 칸). 실제 jevreel이 아니라 1단계를 따라 만든 것"},
    "sheet60_tiles": {"kind": "sheet60", "ask": "tiles", "what": "같은 시트 + 칸마다 있는가(frame_limit 시트 실험의 상태·물음). 우리 변형"},
    "grid8":         {"kind": "grid8", "scale": 1.0, "what": "8장을 원본 크기 그대로 4×2 한 그림(토큰 약 2배). 참고용"},
    # 덧붙임(2026-11-14): "한 그림"과 "좁은 시간 폭"을 가르려고. sep4(1.5초)보다 조금 넓은 1.75초를 0.25초 간격 8장으로
    "grid8_wide_small": {"kind": "grid8", "scale": 0.7071, "offsets": WIDE_OFFSETS, "head": HEAD_GRID8_WIDE,
                         "what": "k±0.125 … k±0.875초 8장(0.25초 간격, 1.75초 폭)을 4×2 한 그림으로, 0.7071배로 줄임(화소 합 ≈ 4장)"},
}
WORKERS = {"sep4": 16, "grid8_small": 16, "grid8": 16, "sheet32": 12, "sheet60_jr": 8, "sheet60_tiles": 2,
           "q3_plain": 16, "q3_path": 16, "q3_comp": 16, "q3_sep4": 16, "grid8_wide_small": 16}   # 동시 요청 수. 큰 시트는 읽기가 많아 줄인다
Q3_KINDS = ("q3one", "q3sep")


def clock(t):
    t = max(0, int(t))
    return f"{t // 60}:{t % 60:02d}"


# ---- 묶음(요청 하나에 들어가는 위치들)
def units(items, layout):
    """items: 위치 목록(같은 영상끼리 시각 순). 돌려주는 값: 요청 단위 목록 [{"id", "layout", "set", "video", "items": [위치…]}]."""
    kind = LAYOUTS[layout]["kind"]
    if kind in Q3_KINDS:   # items = Q3 구간 목록. 구간 하나가 요청 하나
        return [{"id": f"{layout}/{s['video']}/{s['frame']}", "layout": layout, "set": s["set"], "video": s["video"], "items": [s]}
                for s in sorted(items, key=lambda x: (x["video"], x["frame"]))]
    by_video = {}
    for it in items:
        by_video.setdefault(it["video"], []).append(it)
    out = []
    for video in sorted(by_video):
        its = sorted(by_video[video], key=lambda x: x["t"])
        if kind in ("sep", "grid8"):
            groups = [[it] for it in its]
        elif kind == "sheet32":
            n = LAYOUTS[layout]["per_sheet"]
            groups = [its[i:i + n] for i in range(0, len(its), n)]
        else:
            assert len(its) <= SHEET60_TILES, f"{video}: 위치 {len(its)}곳은 시트 한 장에 들어가지 않는다"
            groups = [its]
        for g in groups:
            out.append({"id": f"{layout}/{video}/{g[0]['t']:g}", "layout": layout, "set": g[0]["set"], "video": video, "items": g,
                        "range": [FIRST_TICK, its[-1]["t"] + 0.75]})   # 묶음에 있는 프레임의 범위
    return out


def ticks_of(unit):
    """요청에 들어가는 프레임의 시각(초) 목록, 그림에 놓이는 순서대로."""
    spec = LAYOUTS[unit["layout"]]
    ts = [it["t"] for it in unit["items"]]
    if spec["kind"] == "sep":
        return [round(ts[0] + o, 3) for o in spec["offsets"]]
    if spec["kind"] in ("grid8", "sheet32"):
        out = [round(t + o, 3) for t in ts for o in spec.get("offsets", GRID_OFFSETS)]
        if "range" in unit:   # 묶음 밖의 시각은 가장 가까운 있는 프레임으로 당겨 쓴다(mc_dense와 같은 규칙). 앞선 모양들은 범위 밖으로 나가지 않는다
            lo, hi = unit["range"]
            out = [min(max(x, lo), hi) for x in out]
        return out
    return list(ts)   # sheet60: 위치마다 그 시각의 한 장(1초 간격)


# ---- 그림
def _open(b):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def _jpeg(im, quality):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def s32_scale(w, h):
    """sheet32에서 프레임을 줄이는 배율: 8열×4행과 띠가 SHEET32_PIXELS 안에 드는 가장 큰 값(0.001 단위)."""
    s = 1.0
    while s > 0.05:
        tw, th = round(w * s), round(h * s)
        if (8 * tw + 7 * GAP) * (4 * th + 3 * GAP) <= SHEET32_PIXELS:
            return s
        s = round(s - 0.001, 3)
    raise ValueError("배율을 찾지 못함")


def choose_layout(aspect, tiles=SHEET60_TILES, max_side=SHEET60_SIDE):
    """jevreel 0.2.0 sheets.choose_layout과 같다: 한 변 max_side 안에서 칸이 가장 큰 격자, 같으면 빈칸이 적은 쪽."""
    best = None
    for cols in range(1, tiles + 1):
        rows = math.ceil(tiles / cols)
        tw = math.floor(min(max_side / cols, max_side * aspect / rows))
        th = math.floor(tw / aspect)
        if tw < 1 or th < 1:
            continue
        rank = (tw * th, -(cols * rows - tiles))
        if best is None or rank > best[0]:
            best = (rank, {"cols": cols, "rows": rows, "tile_w": tw, "tile_h": th})
    return best[1]


def grid_picture(frames, scale, cols, rows):
    """frames: JPEG 바이트(시간 순). cols×rows로 붙이고 사이에 GAP 띠. mc_dense _picture와 같은 방식(LANCZOS, JPEG 95)."""
    from PIL import Image
    ims = [_open(b) for b in frames]
    w, h = ims[0].size
    if scale != 1.0:
        w, h = round(w * scale), round(h * scale)
        ims = [im.resize((w, h), Image.LANCZOS) for im in ims]
    used_rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * w + (cols - 1) * GAP, used_rows * h + (used_rows - 1) * GAP), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (w + GAP), (i // cols) * (h + GAP)))
    return _jpeg(sheet, 95)


def sheet60_picture(frames, ids):
    """jevreel 0.2.0 sheets.compose와 같다: 60칸 격자(빈칸은 검정), 칸에 맞춰 줄여 가운데 놓고, 왼쪽 위에 칸 이름. JPEG 90."""
    from PIL import Image, ImageDraw, ImageFont
    ims = [_open(b) for b in frames]
    lay = choose_layout(ims[0].width / ims[0].height)
    canvas = Image.new("RGB", (lay["cols"] * lay["tile_w"], lay["rows"] * lay["tile_h"]))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, lay["tile_h"] // 8))
    for slot, (im, tid) in enumerate(zip(ims, ids)):
        x0, y0 = (slot % lay["cols"]) * lay["tile_w"], (slot // lay["cols"]) * lay["tile_h"]
        if im.size != (lay["tile_w"], lay["tile_h"]):
            im.thumbnail((lay["tile_w"], lay["tile_h"]), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (lay["tile_w"], lay["tile_h"]))
            tile.paste(im, ((lay["tile_w"] - im.width) // 2, (lay["tile_h"] - im.height) // 2))
            im = tile
        canvas.paste(im, (x0, y0))
        left, top, right, bottom = draw.textbbox((x0 + 3, y0 + 2), tid, font=font)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 2), tid, fill=(255, 255, 255), font=font)
    return _jpeg(canvas, 90)


def tile_ids(n):
    return [f"T{i + 1:02d}" for i in range(n)]


def pictures(unit, get):
    """요청에 넣을 그림들(JPEG 바이트 목록). get(video, 파일 이름) → 바이트."""
    spec, video = LAYOUTS[unit["layout"]], unit["video"]
    if spec["kind"] == "q3one":
        return [get(video, f"q3/{unit['items'][0]['frame']:03d}_{spec['img']}.jpg")]
    if spec["kind"] == "q3sep":
        return [get(video, f"q3/tl{n:03d}.jpg") for n in unit["items"][0]["d_frames"]]
    frames = [get(video, fname(t)) for t in ticks_of(unit)]
    if spec["kind"] == "sep":
        return frames
    if spec["kind"] == "grid8":
        return [grid_picture(frames, spec["scale"], 4, 2)]
    if spec["kind"] == "sheet32":
        w, h = _open(frames[0]).size
        return [grid_picture(frames, s32_scale(w, h), 8, len(unit["items"]))]
    return [sheet60_picture(frames, tile_ids(len(frames)))]


# ---- 요청 본문
def body(unit, pics):
    """(요청 본문, 질문 키 → [질문 id, 무엇, 위치 시각]). 무엇: pos(그 위치), window(창 전체), row(그 줄 = 그 위치), shown(시트에 있는가), pick(어느 칸), tile(그 칸 = 그 위치)."""
    spec = LAYOUTS[unit["layout"]]
    uris = ["data:image/jpeg;base64," + base64.b64encode(p).decode() for p in pics]
    its = unit["items"]
    qs = its[0]["questions"]          # 같은 영상의 위치는 질문이 같다
    keys, questions = {}, {}

    def noul(key, text, qid, what, t):
        questions[key] = {"type": "noul", "instructions": text}
        keys[key] = [qid, what, t]

    if spec["kind"] == "q3one":   # 무엇: seg(그 구간), 세 번째 값은 구간의 끝 프레임 번호
        seg = its[0]
        legend = LEGEND if spec["img"] != "plain" and seg["paths_" + spec["img"]] > 0 else ""
        state = {"frame": uris[0]}
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"q{j}", HEAD_ONE + legend + Q_ONE.format(s=s), qid, "seg", seg["frame"])
    elif spec["kind"] == "q3sep":
        seg = its[0]
        state = {"frames": [{"seconds": round(s, 2), "frame": u} for s, u in zip(seg["d_seconds"], uris)]}
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"q{j}", HEAD_Q3SEP + SHOW.format(s=s), qid, "seg", seg["frame"])
    elif spec["kind"] == "sep":
        secs = ticks_of(unit)
        state = {"frames": [{"seconds": round(s, 2), "frame": u} for s, u in zip(secs, uris)]}
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"q{j}", HEAD_SEP + SHOW.format(s=s), qid, "pos", its[0]["t"])
    elif spec["kind"] == "grid8":
        state = {"seconds": ticks_of(unit), "grid": uris[0]}
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"q{j}", spec.get("head", HEAD_GRID8) + SHOW.format(s=s), qid, "pos", its[0]["t"])
    elif spec["kind"] == "sheet32":
        secs = ticks_of(unit)
        state = {"rows": [secs[i:i + 8] for i in range(0, len(secs), 8)], "sheet": uris[0]}
        head = HEAD_S32.format(n=len(secs), r=len(its))
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"w{j}", head + ANY.format(s=s), qid, "window", None)
            for r, it in enumerate(its, 1):
                noul(f"r{r}q{j}", head + ROW.format(j=r, s=s), qid, "row", it["t"])
    elif spec["ask"] == "jr":
        ids = tile_ids(len(its))
        state = {"window": f"{clock(its[0]['t'])}-{clock(its[-1]['t'] + 1)}", "sheet": uris[0], "layout": JR_LAYOUT}
        for j, (qid, s) in enumerate(qs.items()):
            noul(f"s{j}", JR_SHOWN.format(s=s), qid, "shown", None)
            questions[f"c{j}"] = {"type": "choice", "instructions": JR_PICK.format(s=s), "criteria": {**dict.fromkeys(ids), "none": JR_NONE}}
            keys[f"c{j}"] = [qid, "pick", None]
    else:
        ids = tile_ids(len(its))
        state = {"layout": T60_LAYOUT.format(n=len(its)), "tiles": {tid: clock(it["t"]) for tid, it in zip(ids, its)}, "sheet": uris[0]}
        for j, (qid, s) in enumerate(qs.items()):
            for tid, it in zip(ids, its):
                noul(f"{tid}q{j}", T60_Q.format(tid=tid, s=s), qid, "tile", it["t"])
    return {"state": state, "questions": questions}, keys


def parse_answers(keys, out):
    """서버 답 → ([질문 id, 무엇, 위치 시각(또는 구간 번호), 확률] 목록, {질문 id: {선택지: 확률}}). 답이 빠졌거나 범위 밖이면 오류."""
    ans, rows, choices = out["answers"], [], {}
    for k, (qid, what, x) in keys.items():
        a = ans[k]
        if a["type"] == "choice":
            choices[qid] = {o: round(float(p), 6) for o, p in a["probabilities"].items()}
            if abs(sum(choices[qid].values()) - 1) > 0.01:
                raise ValueError(f"{k}: 선택지 확률의 합이 1이 아니다")
        else:
            p = float(a["noul"])
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{k}: 확률 {p}")
            rows.append([qid, what, x, round(p, 6)])
    return rows, choices


def tile_time(unit):
    """sheet60의 칸 이름 → 위치 시각."""
    return {tid: it["t"] for tid, it in zip(tile_ids(len(unit["items"])), unit["items"])}


# ---- 토큰 어림
def smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=CAP):
    """Qwen2-VL 계열 그림 처리기의 크기 맞춤(transformers smart_resize와 같은 규칙). (높이, 너비)."""
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


SYSTEM = "You are a decision engine. You read a record and answer one question about it by choosing exactly one option. Reply with a single letter."
CHARS_PER_TOKEN = 3.4   # 글 토큰 어림. cd_prepare.py dryrun이 받아 둔 기준 답의 usage로 맞춘 값을 찍는다


def _labels(n):
    base = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
    return (base + ["x"] * max(0, n - len(base)))[:n]


def estimate(payload, sizes):
    """(H2O의 usage.input_tokens 어림, processed_input_tokens 어림 = 질문마다 다시 읽는 몫까지, 그림 토큰).
    sizes: 그림마다 (너비, 높이). 공유 머리(시스템 + 그림 + 기록)는 한 번, 질문마다 꼬리를 더한다(shim 설명과 같다)."""
    img = 0
    for w, h in sizes:
        hb, wb = smart_resize(h, w)
        img += hb * wb // PIXELS_PER_TOKEN + 2
    state = json.loads(json.dumps(payload["state"]))
    n = [0]

    def strip(x):
        if isinstance(x, str) and x.startswith("data:image"):
            n[0] += 1
            return f"[image {n[0]}]"
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    record = json.dumps(strip(state), ensure_ascii=False, separators=(",", ":"))
    head = img + (len(SYSTEM) + len("record: ") + len(record) + 60) / CHARS_PER_TOKEN
    tails = []
    for q in payload["questions"].values():
        if q["type"] == "noul":
            opts = ["true: the statement holds", "false: it does not"]
        else:
            opts = [f"{k}: {v if v is not None else k}" for k, v in q["criteria"].items()]
        text = "question: " + q["instructions"] + "\noptions:\n" + "\n".join(f"{l}) {o}" for l, o in zip(_labels(len(opts)), opts))
        tails.append(len(text) / CHARS_PER_TOKEN)
    return round(head + sum(tails)), round(len(tails) * head + sum(tails)), img
