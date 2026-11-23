"""확인 단계를 그림 한 장으로: 시트가 낸 후보 k곳의 4장 묶음을 한 이미지에 모두 넣고 Jev에 한 번만 묻는다.

지금의 확인(pipeline_run.py)은 후보마다 요청을 따로 보내 후보가 늘면 토큰도 는다. 여기서는 후보가 몇 곳이든 그림 한 장 값만 든다.
대신 후보가 늘수록 프레임 하나가 작아진다. 따른 논문은 없다(가까운 것: 여러 프레임을 한 장의 패널로 넣는 Video Panels).

정한 것 (답을 보기 전에 정함)
- 후보: 시트 칸마다(Noul) 확률이 높은 순으로 k곳, 이미 고른 칸의 바로 옆 칸은 건너뜀 (pipeline_run.py와 같음)
- 후보마다 프레임 4장: 그 시각 −0.75, −0.25, +0.25, +0.75초 (앞뒤 1초의 묶음은 넣지 않는다)
- 배치: 후보 하나가 한 줄(세로 영상이면 세로줄, 위 → 아래가 시간 순서. 가로 영상이면 가로줄, 왼쪽 → 오른쪽).
  두 배치 중 프레임이 더 커지는 쪽을 쓴다. 후보는 시간 순서로 놓고 C1, C2, …로 부른다(시트에서의 순위는 보이지 않는다)
- 질문: 후보마다 "이 클립의 4장에 보이는가"(Noul) + "어느 클립이 가장 뚜렷한가"(Choice, 보기에 none 포함)를 한 요청에
- 답: Noul이 가장 높은 후보(주), Choice의 몫이 가장 큰 후보(보조)

실행: <venv>/bin/python experiments/frame_limit/scripts/pipeline/cand_sheet.py --k 6   (환경변수 TYPESAFE_API_KEY 필요)
      --tag sta --sheet-preds results/tl_sheet60_preds.json   --dry-run / --smoke / --workers 6
결과: results/cand_<묶음>_k<k>.json, data/cand_sheets/<묶음>_k<k>/<항목>.jpg(보낸 그림)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import io
import json
import math
import os
import time
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from pipeline_run import candidates, tile_probs

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
MODEL = "jev-1.14.0"
IMAGES = {"TypeSafe-Danger-Zone-Images": "true"}
PRICE_PER_MTOK = 0.042
MAX_SIDE, GAP, PER_CLIP = 1568, 6, 4


def layout_for(aspect, k):
    """후보 하나를 세로줄로 둘지 가로줄로 둘지: 프레임이 더 커지는 쪽."""
    best = None
    for by_column, cols, rows in ((True, k, PER_CLIP), (False, PER_CLIP, k)):
        w = math.floor(min((MAX_SIDE - GAP * (cols + 1 if by_column else 2)) / cols,
                           (MAX_SIDE - GAP * (2 if by_column else rows + 1)) * aspect / rows))
        h = math.floor(w / aspect)
        if best is None or w * h > best["w"] * best["h"]:
            best = {"by_column": by_column, "cols": cols, "rows": rows, "w": w, "h": h}
    return best


def compose(clips, lay):
    """clips: 후보마다 프레임 4장의 경로(시간 순서). 후보 사이에는 검은 띠를 두고 첫 프레임 위에 C1, C2, …를 적는다."""
    w, h, by_col = lay["w"], lay["h"], lay["by_column"]
    k = len(clips)
    size = (k * w + GAP * (k + 1), PER_CLIP * h + 2 * GAP) if by_col else (PER_CLIP * w + 2 * GAP, k * h + GAP * (k + 1))
    canvas = PILImage.new("RGB", size)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, h // 12))
    for c, paths in enumerate(clips):
        for f, path in enumerate(paths):
            im = PILImage.open(HERE / path).convert("RGB")
            im.thumbnail((w, h), PILImage.Resampling.LANCZOS)
            x0 = GAP + c * (w + GAP) if by_col else GAP + f * w
            y0 = GAP + f * h if by_col else GAP + c * (h + GAP)
            canvas.paste(im, (x0 + (w - im.width) // 2, y0 + (h - im.height) // 2))
            if f == 0:
                label = f"C{c + 1}"
                left, top, right, bottom = draw.textbbox((x0 + 3, y0 + 2), label, font=font)
                draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
                draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def build_job(item, v, probs, k, out_dir):
    centers, offsets = v["centers"], v["window_offsets"]
    cand = sorted(candidates(probs, k))  # 시간 순서
    clips = [[v["frames"][f"{round(centers[p] + o, 2):.2f}"] for o in offsets] for p in cand]
    with PILImage.open(HERE / clips[0][0]) as im:
        lay = layout_for(im.width / im.height, len(cand))
    image = compose(clips, lay)
    path = out_dir / f"{item['id']}.jpg"
    image.save(path, format="JPEG", quality=90)
    return {"item": item, "cand": cand, "lay": lay, "file": str(path.relative_to(HERE)), "size": list(image.size)}


def clock(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def run(client, job, frames):
    from typesafe_sdk import Choice, Image, Noul, TypeSafeAPIError
    item, lay = job["item"], job["lay"]
    labels = [f"C{i + 1}" for i in range(len(job["cand"]))]
    line, order = ("column", "top to bottom") if lay["by_column"] else ("row", "left to right")
    centers = frames[item["video"]]["centers"]
    rec = {"item": item["id"], "video": item["video"], "cand": job["cand"], "labels": labels, "by_column": lay["by_column"],
           "frame_size": [lay["w"], lay["h"]], "image_size": job["size"], "error": None, "attempts": 0}
    state = {"layout": (f"`sheet` shows {len(labels)} short clips taken from one video. Each {line} is one clip: {PER_CLIP} consecutive "
                        f"frames, 0.5 seconds apart, in time order from {order}. Clips are separated by black lines and each clip's "
                        "label (C1, C2, ...) is printed on its first frame. `clips` maps each label to the clip's time in the video."),
             "clips": {lab: clock(centers[p]) for lab, p in zip(labels, job["cand"])},
             "sheet": Image((HERE / job["file"]).read_bytes())}
    s = item["sentence"]
    qs = {lab: Noul(instructions=f"Do the {PER_CLIP} frames of clip {lab} in `sheet` show the following: {s}?") for lab in labels}
    qs["pick"] = Choice(instructions=f"Which clip of `sheet` shows the following most clearly: {s}",
                        criteria={**dict.fromkeys(labels), "none": "No clip shows it."})
    resp = None
    for wait in (0, 3, 8, 20):
        time.sleep(wait)
        rec["attempts"] += 1
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
    rec["nouls"] = {lab: resp.nouls[lab].noul for lab in labels}
    rec["choice"] = dict(resp.choices["pick"].probabilities)
    rec.update(input_tokens=resp.usage.input_tokens, request_id=resp.request_id)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sta")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sheet-preds", default="results/tl_sheet60_preds.json")
    ap.add_argument("--sheet-index", default="data/sheets60.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    items = json.loads((HERE / "data" / f"tl_{args.tag}_items.json").read_text())["items"]
    frames = json.loads((HERE / "data" / f"{args.tag}_frames.json").read_text())
    stage1 = tile_probs(items, args.tag, args.sheet_index, args.sheet_preds)
    out_dir = HERE / "data" / "cand_sheets" / f"{args.tag}_k{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [build_job(it, frames[it["video"]], stage1[it["id"]]["probs"], args.k, out_dir) for it in items]
    out_path = HERE / "results" / f"cand_{args.tag}_k{args.k}.json"
    done = {r["item"]: r for r in json.loads(out_path.read_text())["records"] if not r["error"]} if out_path.exists() else {}
    todo = [j for j in jobs if j["item"]["id"] not in done]
    pixels = sum(j["size"][0] * j["size"][1] for j in todo)
    guess = pixels / 310 + 400 * len(todo)  # 받아 둔 요청에서 토큰당 약 310화소였다
    sizes = sorted({(j["lay"]["w"], j["lay"]["h"], "세로줄" if j["lay"]["by_column"] else "가로줄") for j in jobs})
    print(f"질문 {len(items)}개, 요청 {len(jobs)}건 중 보낼 것 {len(todo)}건. 후보 {args.k}곳 × {PER_CLIP}장, 프레임 크기 {sizes}. "
          f"어림: 입력 약 {guess:,.0f}토큰, 약 ${guess * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient
    if args.smoke:
        with TypeSafeClient(model=MODEL) as client:
            print(todo[0]["file"])
            print(json.dumps(run(client, todo[0], frames), ensure_ascii=False)[:900])
        return
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        fresh = {r["item"]: r for r in pool.map(lambda j: run(client, j, frames), todo)}
    recs = [done.get(j["item"]["id"]) or fresh[j["item"]["id"]] for j in jobs]
    errors = [r for r in recs if r["error"]]
    tokens = sum(r.get("input_tokens", 0) for r in recs)
    meta = {"model": MODEL, "tag": args.tag, "k": args.k, "sheet_preds": args.sheet_preds, "requests": len(recs), "errors": len(errors),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
            "stage1_tokens": {k: v["tokens"] for k, v in stage1.items()}, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, 요청당 평균 {tokens / len(recs):,.0f}토큰")
    for e in errors[:5]:
        print("  오류:", e["item"], e["error"])


if __name__ == "__main__":
    main()
