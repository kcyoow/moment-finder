""""모두 찾기"의 확인 단계: 시트가 낸 덩어리마다 가운데 프레임 한 장을 원본 크기로 물어 걸러낸다.

시트의 칸마다 답(Noul)이 0.5 이상인 위치를 뽑고, 이어진 위치를 한 덩어리로 묶는다. 덩어리마다 가운데 위치의 프레임 한 장을
"한 장" 질문(loc_run.py의 문구 그대로)으로 묻는다. 0.5 미만이면 그 덩어리를 버리는 것은 채점 쪽(mc_all_eval.py)에서 한다.
따른 논문은 없다. 작업 22에서 받아 둔 답으로 계산해 본 방식을 실제 요청으로 돌리는 것이다.

실행: <venv>/bin/python experiments/frame_limit/scripts/pipeline/runs_verify.py --tag qvm   (환경변수 TYPESAFE_API_KEY 필요)
      --sheet-preds results/sheet60_qvm_preds.json   --dry-run   --workers 6
결과: results/runs_<묶음>.json (덩어리마다 항목, 시각, 답, 토큰)
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

from pipeline_run import tile_probs

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CUT = 0.5
PRICE_PER_MTOK = 0.042


def runs_of(flags):
    """이어진 True의 덩어리마다 위치 번호의 목록."""
    out, cur = [], []
    for k, f in enumerate(flags):
        if f:
            cur.append(k)
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qvm")
    ap.add_argument("--sheet-preds", default=None)
    ap.add_argument("--sheet-index", default="data/sheets60.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    preds = args.sheet_preds or f"results/sheet60_{args.tag}_preds.json"
    items = json.loads((HERE / "data" / f"{args.tag}_items.json").read_text())["items"]
    frames = json.loads((HERE / "data" / f"{args.tag}_frames.json").read_text())
    stage1 = tile_probs(items, args.tag, args.sheet_index, preds)
    jobs = []
    for it in items:
        centers = frames[it["video"]]["centers"]
        for g in runs_of([p >= CUT for p in stage1[it["id"]]["probs"]]):
            jobs.append((it, centers[g[len(g) // 2]], g))
    sizes = sorted(len(g) for _, _, g in jobs)
    print(f"질문 {len(items)}개, 덩어리 {len(jobs)}개(질문당 {len(jobs) / len(items):.1f}개, 길이 중앙값 {sizes[len(sizes) // 2] if sizes else 0}칸, "
          f"한 칸짜리 {sum(s == 1 for s in sizes)}개). 요청 {len(jobs)}건, 어림: 약 {len(jobs) * 1300:,}토큰, 약 ${len(jobs) * 1300 * PRICE_PER_MTOK / 1e6:.4f}")
    if args.dry_run:
        return
    assert os.environ.get("TYPESAFE_API_KEY"), "TYPESAFE_API_KEY가 없습니다"
    from typesafe_sdk import TypeSafeClient

    from loc_run import MODEL, run
    with TypeSafeClient(model=MODEL) as client, cf.ThreadPoolExecutor(args.workers) as pool:
        got = list(pool.map(lambda j: run(client, (j[0], j[1], "one", "plain"), frames), jobs))
    recs = [{"item": it["id"], "video": it["video"], "t": t, "run": [g[0], g[-1]], "noul": r.get("noul"), "error": r["error"],
             "input_tokens": r.get("input_tokens", 0)} for (it, t, g), r in zip(jobs, got)]
    errors = [r for r in recs if r["error"]]
    tokens = sum(r["input_tokens"] for r in recs)
    meta = {"model": MODEL, "tag": args.tag, "sheet_preds": preds, "cut": CUT, "requests": len(recs), "errors": len(errors),
            "input_tokens": tokens, "cost_usd": round(tokens * PRICE_PER_MTOK / 1e6, 4), "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (HERE / "results" / f"runs_{args.tag}.json").write_text(json.dumps({"meta": meta, "records": recs}, ensure_ascii=False, indent=1) + "\n")
    print(f"요청 {len(recs)}건, 오류 {len(errors)}건, 입력 토큰 {tokens:,}, 약 ${meta['cost_usd']}, 버려질 덩어리 {sum((r['noul'] or 0) < CUT for r in recs)}개")
    for e in errors[:5]:
        print("  오류:", e["item"], e["error"])


if __name__ == "__main__":
    main()
