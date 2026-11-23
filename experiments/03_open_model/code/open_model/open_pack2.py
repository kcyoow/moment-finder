"""둘째 평가 묶음: 60칸 시트와, 0.5초 어긋난 위치의 4장 묶음. Jev 요청은 없다. 표준 라이브러리만 쓴다.

"요청 수를 줄이면(시트로 먼저 훑기) 또는 늘리면(0.5초 간격) d1-2B가 어떻게 되나"를 보려는 것이다.
- sheet_sta2 : Charades 30개 영상의 시트 30장 × 그 영상의 질문 (요청 100개, 요청마다 칸 60개에 예/아니오)
- sheet_mc   : 마인크래프트 시트 60장 × 질문 2개(창이 열림 / 인벤토리만) (요청 120개)
- sheet_qvm  : QVHighlights 시트 60장 × 질문 1개 (요청 60개)
- charades_h : Charades의 0.5초 어긋난 위치(1.5, 2.5, …)의 4장 묶음. 그림은 첫 묶음(open_pack.py)에 이미 들어 있다
시트 항목: {id, task, video, sheet, item, kind: "sheet", images[시트 그림], layout(Jev에 보낸 설명), tiles{칸: 시각 글자}, tile_t{칸: 초},
           questions{칸: 문장}, gt{칸: 0/1}, jev{칸: 확률}}

읽는 곳: experiments/frame_limit 의 data/, results/ (고치지 않는다)
실행: python3 experiments/open_model/open_pack2.py
결과: pack/manifest2.json, pack/parts2/open_pack2.zip.NN, pack/parts2/sha256.txt
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"
OUT = HERE / "pack"
PART = 45 * 1024 * 1024
SHEETS = {"sta2": ("data/tl_sta2_items.json", "results/tl_sheet60_sta2_preds.json"),
          "mc": ("data/mc_items.json", "results/mc_sheet60_preds.json"),
          "qvm": ("data/qvm_items.json", "results/sheet60_qvm_preds.json")}


def read(path):
    return json.loads((SRC / path).read_text())


def inside(t, windows):
    return any(s <= t <= e for s, e in windows)


def clock(t):  # sheet_run.py와 같다
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    items, files = [], {}
    sheets = {(s["tag"], s["video"], str(s["sheet"])): s for s in read("data/sheets60.json")["sheets"]}
    for tag, (item_file, pred_file) in SHEETS.items():
        by_id = {it["id"]: it for it in read(item_file)["items"]}
        for r in read(pred_file)["records"]:
            if r["kind"] != "tiles" or r["error"]:
                continue
            sheet, it = sheets[(tag, r["video"], str(r["sheet"]))], by_id[r["item"]]
            dest = f"img/sheet_{tag}/{r['video']}__{Path(sheet['file']).name}"
            files[dest] = SRC / sheet["file"]
            n, step = len(sheet["tiles"]), sheet["step_seconds"]
            layout = (f"`sheet` is a contact sheet of {n} frames from one video, {step:g} second(s) apart, in time order "
                      "from left to right and top to bottom. Each tile has its ID (T01, T02, ...) in its top-left "
                      "corner. `tiles` maps each ID to its time in the video.")  # sheet_run.state_of의 기본 문장 그대로
            tiles = sheet["tiles"]
            items.append({"id": f"sheet_{tag}/{r['video']}/{r['sheet']}/{it['id']}", "task": f"sheet_{tag}", "video": r["video"],
                          "sheet": str(r["sheet"]), "item": it["id"], "kind": "sheet", "images": [dest], "layout": layout,
                          "tiles": {t["id"]: clock(t["t"]) for t in tiles}, "tile_t": {t["id"]: t["t"] for t in tiles},
                          "questions": {t["id"]: it["sentence"] for t in tiles},
                          "gt": {t["id"]: int(inside(t["t"], it["windows"])) for t in tiles},
                          "jev": {t["id"]: r["nouls"][t["id"]] for t in tiles}})

    # 0.5초 어긋난 위치의 4장 묶음. 그림은 첫 묶음에 있는 것을 가리킨다
    first = {p for it in json.loads((OUT / "manifest.json").read_text())["items"] for p in it["images"]}
    by_video = {}
    for it in read("data/tl_sta2_items.json")["items"]:
        by_video.setdefault(it["video"], []).append(it)
    frames = read("data/sta2h_frames.json")
    for r in sorted((r for r in read("results/full_sta2h_preds.json")["records"] if not r["error"]), key=lambda r: (r["video"], r["t"])):
        v, t = frames[r["video"]], r["t"]
        secs = [round(t + o, 2) for o in v["window_offsets"]]
        imgs = [f"img/charades/{Path(v['frames'][f'{s:.2f}']).parent.name}__{Path(v['frames'][f'{s:.2f}']).name}" for s in secs]
        if not all(p in first for p in imgs):
            raise SystemExit(f"첫 묶음에 없는 그림: {r['video']} {t}")
        qs = by_video[r["video"]]
        items.append({"id": f"charades_h/{r['video']}/{t}", "task": "charades_h", "video": r["video"], "t": t, "kind": "win4",
                      "images": imgs, "seconds": secs, "questions": {q["id"]: q["sentence"] for q in qs},
                      "gt": {q["id"]: int(inside(t, q["windows"])) for q in qs}, "jev": {q["id"]: r["nouls"][q["id"]] for q in qs}})

    parts_dir = OUT / "parts2"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()
    note = "open_pack2.py가 만든 둘째 묶음(시트, 0.5초 어긋난 위치). 그림 일부는 첫 묶음의 것을 가리킨다. 실험에만 쓰고 다시 배포하지 않는다"
    (OUT / "manifest2.json").write_text(json.dumps({"_note": note, "items": items}, ensure_ascii=False) + "\n")
    zip_path = OUT / "open_pack2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(OUT / "manifest2.json", "manifest2.json")
        for dest, src in sorted(files.items()):
            z.write(src, dest)
    data = zip_path.read_bytes()
    zip_path.unlink()
    for n in range(0, len(data), PART):
        (parts_dir / f"open_pack2.zip.{n // PART:02d}").write_bytes(data[n:n + PART])
    (parts_dir / "sha256.txt").write_text(hashlib.sha256(data).hexdigest() + "\n")
    count = {}
    for it in items:
        count[it["task"]] = count.get(it["task"], 0) + 1
    print(f"항목 {len(items)}개 {count}, 시트 그림 {len(files)}장, {len(data) / 1e6:.0f}MB")


if __name__ == "__main__":
    main()
