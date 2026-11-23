"""실제 jevreel(PyPI 설치본, DeepAleph, MIT)을 우리 영상과 질문으로 돌려 기준선을 낸다.

지금까지 "jevreel 방식"이라고 부른 것은 그 1단계를 우리가 따라 만든 것이었다. 이 스크립트는 도구 자체를 부른다.
- 설정은 기본값 그대로다(1fps, 시트 60칸, 이웃 시트 2칸 겹침, 검증 상위 6개 창, 재질문 끔)
- 색인은 시트만 만든다(timeline=False, transcription="never"). 기본 질문과 음성 인식은 찾기에 쓰이지 않아 비용에서 뺀다
- 찾기는 reel.ask(video, 질문, threshold=0.0)으로 부른다. 기준을 0으로 두면 jevreel이 순위를 매긴 모든 창이 돌아와서,
  "가장 순위가 높은 한 곳"을 볼 수 있다. 기본 기준 0.5로 보고되는지는 돌려받은 확률로 따로 본다
- jevreel이 쓰는 ffmpeg·ffprobe는 PATH에서 찾는다. 이 Mac에는 없어서 실행할 때 PATH 앞에 임시 폴더를 둔다
  (ffmpeg는 imageio-ffmpeg의 실행 파일, ffprobe는 길이·크기·회전·소리 유무만 내는 작은 대체 스크립트)

실행: PATH=<ffmpeg·ffprobe가 있는 폴더>:$PATH JEVREEL_HOME=<색인을 둘 폴더> <jevreel venv>/bin/python scripts/pipeline/jevreel_run.py --tag sta
      --dry-run   보내지 않고 영상·질문 수와 어림을 출력한다
      --smoke     첫 영상의 첫 질문만 보내고 답을 출력한다 (저장하지 않음)
      --force     시트를 다시 만든다(영상 정보를 다시 읽게 할 때)
      JEVREEL_CACHE=0으로 돌린다. jevreel은 같은 요청의 답을 캐시에서 꺼내면 토큰을 0으로 세어서, 비용을 재려면 캐시를 꺼야 한다
      (환경변수 TYPESAFE_API_KEY 필요. 묶음: sta = 설정을 정한 Charades 12개 영상, sta2 = 확인용 새 영상 30개)
결과: results/jevreel_real_<묶음>.json. 이미 받은 질문은 건너뛴다
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
PRICE_PER_MTOK = 0.042


def video_file(tag, video):
    """영상 파일의 위치는 묶음의 항목 파일(data/<묶음>_items.json)에 적힌 것을 쓴다."""
    return HERE / json.loads((HERE / "data" / f"{tag}_items.json").read_text())["videos"][video]["file"]


def items_of(tag):
    path = HERE / "data" / f"tl_{tag}_items.json"
    if not path.exists():
        path = HERE / "data" / f"{tag}_items.json"
    return json.loads(path.read_text())["items"]


def hit_dict(h):
    return {"at": h.at, "probability": h.probability, "shown": h.shown, "screened": h.screened, "verified": h.verified,
            "anchor": h.anchor, "sheet": h.sheet, "strong": h.strong,
            "also": [{"at": m.at, "probability": m.probability} for m in h.also]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sta")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    items = items_of(args.tag)
    by_video = {}
    for it in items:
        by_video.setdefault(it["video"], []).append(it)
    out_path = HERE / "results" / f"jevreel_real_{args.tag}.json"
    saved = json.loads(out_path.read_text()) if out_path.exists() else {"meta": {}, "records": []}
    done = {r["item"]: r for r in saved["records"] if not r.get("error")}
    todo = sum(it["id"] not in done for it in items)
    print(f"영상 {len(by_video)}개, 질문 {len(items)}개 중 보낼 것 {todo}개. 질문마다 시트 요청(영상의 분 수만큼) + 검증 요청 1건 안팎. "
          f"30초 영상이면 질문당 2건·약 15,000토큰으로 어림: 약 {todo * 15000:,}토큰, 약 ${todo * 15000 * PRICE_PER_MTOK / 1e6:.3f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    assert os.environ.get("JEVREEL_HOME"), "JEVREEL_HOME을 정해 색인이 임시 폴더에 가게 한다"
    import jevreel
    from jevreel import Reel
    records, index_info = dict(done), dict(saved["meta"].get("index", {}))

    def save():
        recs = [records[it["id"]] for it in items if it["id"] in records]
        tokens = sum(r.get("input_tokens", 0) for r in recs)
        meta = {"jevreel_version": jevreel.__version__, "asks": len(recs), "errors": sum(bool(r.get("error")) for r in recs),
                "requests": sum(r.get("requests", 0) for r in recs), "cached": sum(r.get("cached", 0) for r in recs),
                "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4),
                "settings": "기본값(fps 1, 60칸, 겹침 2, verify_top 6, recheck 끔), threshold 0.0, timeline 없음, 캐시 끔",
                "index": index_info, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        out_path.write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
        return meta

    async with Reel() as reel:
        for video, its in by_video.items():
            if all(it["id"] in records for it in its) and not args.smoke:
                continue
            indexed = await reel.index(str(video_file(args.tag, video)), timeline=False, transcription="never", force=args.force)
            index_info[video] = {"id": indexed.id, "sheets": indexed.sheets, "layout": indexed.layout, "duration": indexed.duration,
                                 "index_requests": indexed.requests}

            async def ask(it):
                t0 = time.perf_counter()
                try:
                    r = await reel.ask(indexed.id, it["sentence"], threshold=0.0)
                except Exception as err:  # 한 질문이 멈춰도 나머지는 이어서 받는다
                    return {"tag": args.tag, "item": it["id"], "video": video, "error": f"{type(err).__name__}: {err}"[:300]}
                return {"tag": args.tag, "item": it["id"], "video": video, "hits": [hit_dict(h) for h in r.hits],
                        "windows_checked": r.windows_checked, "requests": r.requests, "cached": r.cached,
                        "input_tokens": r.input_tokens, "seconds": round(time.perf_counter() - t0, 2)}
            pending = [it for it in its if it["id"] not in records]
            if args.smoke:
                rec = await ask(pending[0] if pending else its[0])
                print(json.dumps(index_info[video], ensure_ascii=False))
                print(json.dumps(rec, ensure_ascii=False)[:1500])
                return
            for rec in await asyncio.gather(*(ask(it) for it in pending)):
                records[rec["item"]] = rec
            save()
            print(f"{video}: 시트 {indexed.sheets}장({indexed.layout}), 길이 {indexed.duration}, 질문 {len(its)}개", flush=True)
    meta = save()
    print(f"질문 {meta['asks']}개(오류 {meta['errors']}개), 요청 {meta['requests']}건(캐시 {meta['cached']}건), 입력 토큰 {meta['input_tokens']:,}, "
          f"약 ${meta['cost_usd']}, 질문당 평균 {meta['input_tokens'] / max(meta['asks'], 1):,.0f}토큰")


if __name__ == "__main__":
    asyncio.run(main())
