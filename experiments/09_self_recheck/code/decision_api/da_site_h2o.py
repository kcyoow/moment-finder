"""사이트의 받아 둔 실행 여섯(README 5-1절의 표)을 복사해, 다시 확인 단계만 판정 모델(H2O)이 스스로 하게 실제로 돌린다. README.md 13절.

원래 실행은 건드리지 않는다. 복사본 pipeline/runs/h2o_<이름>에 쓴다. 확인 질문 여섯을 먼저 함께 쓰고(Gemini 4 Hydrogen, 영상은 주지 않는다),
그 뒤에 올려 둔 판정 모델 앱에 차례로 묻는다(GPU가 떠 있는 동안 기다리는 시간을 줄이려는 것).

실행: MODAL_PROFILE=<modal-profile> uv run --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg --with modal==1.5.5 python -I experiments/decision_api/da_site_h2o.py
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import mf  # noqa: E402
import mf_recheck as RC  # noqa: E402

RUNS = ["web_1116_200731_2", "web_1116_200730_1", "web_1116_201838_3", "web_1116_201839_4", "web_1116_041133_3", "web_1116_043326_3"]


def hits(cj):
    return [{"start": c["start"], "end": c["end"], "p": c.get("p"), "cand": c.get("cand")} for c in cj["clips"] if c.get("match", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="복사본과 앞서 쓴 확인 질문을 그대로 쓴다(Gemini를 부르지 않는다)")
    ap.add_argument("--rule", default="both", choices=["both", "noul"])
    ap.add_argument("--app", default=None)
    args = ap.parse_args()
    plan = []
    for name in RUNS:
        src, new = mf.RUNS / name, "h2o_" + name
        dst = mf.RUNS / new
        if not args.reuse:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        cj, info, job = mf.read(src / "clips.json"), mf.read(src / "run.json"), mf.read(src / "job.json")
        plan.append({"name": name, "new": new, "request": job["request"], "duration": info["duration"], "size": [info["width"], info["height"]],
                     "hydrogen": {"candidates": cj["verify"]["candidates"], "matched": cj["verify"]["matched"], "clips": hits(cj),
                                  "p_by_cand": {str(c["cand"]): c["p"] for c in cj["clips"] if "cand" in c}}})
    print(f"실행 {len(plan)}개. 확인 질문{'을 다시 쓰는' if args.reuse else '을 쓰는'} 중…", flush=True)
    t0 = time.time()
    if args.reuse:
        for p in plan:
            p["question"] = mf.read(mf.RUNS / p["new"] / "verify_question" / "question.json")
    else:
      with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(RC.write_question, p["new"], p["request"], [f"길이 {mf.mmss(p['duration'])}", f"화면 {p['size'][0]}×{p['size'][1]}"]) for p in plan]
        for p, f in zip(plan, futs):
            p["question"] = f.result()
    for p in plan:
            print(f"  {p['name']} | {p['request']} | ${p['question']['usage']['usd_estimate']} | {p['question']['noul']['instructions']}", flush=True)
    print(f"확인 질문 {time.time() - t0:.0f}초, 합 ${sum(p['question']['usage']['usd_estimate'] for p in plan):.2f}. 판정 모델에 묻는 중…", flush=True)
    for p in plan:
        t1 = time.time()
        v = RC.recheck_h2o(p["new"], p["request"], question=p["question"], on_stage=lambda text, frac=0.0: None, rule=args.rule, app=args.app)
        shutil.rmtree(mf.RUNS / p["new"] / "posters", ignore_errors=True)
        mf.stage_render(argparse.Namespace(name=p["new"], top=0))
        cj, verdict = mf.read(mf.RUNS / p["new"] / "clips.json"), mf.read(mf.RUNS / p["new"] / "verify" / "verdict.json")
        job = mf.read(mf.RUNS / p["new"] / "job.json")                       # 사이트의 목록에서 원래 실행과 가려 보이게 한다
        job.update(id=p["new"], title=str(job.get("title") or "").replace(" · H2O 확인", "") + " · H2O 확인")
        for s in job.get("steps", []):
            if s.get("key") == "verify":
                s["label"] = "다시 확인 · H2O"
        mf.write(mf.RUNS / p["new"] / "job.json", job)
        p["h2o"] = {"candidates": v["candidates"], "matched": v["matched"], "clips": hits(cj), "seconds": round(time.time() - t1, 1),
                    "by_cand": {str(c["id"]): c for c in verdict["candidates"]}, "ready_seconds": (verdict.get("judge", {}).get("a") or {}).get("ready_seconds")}
        print(f"  {p['name']} ({p['h2o']['seconds']}초): Hydrogen {p['hydrogen']['matched']}/{p['hydrogen']['candidates']} 통과 {[(round(c['start']), round(c['end'])) for c in p['hydrogen']['clips']]}"
              f" | H2O {v['matched']}/{v['candidates']} 통과 {[(round(c['start']), round(c['end']), c['p']) for c in p['h2o']['clips']]}", flush=True)
    (HERE / "results" / f"site_h2o_{args.rule}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    print("\n후보마다 (Hydrogen의 p → H2O의 p [예/아니오, 고르기])")
    for p in plan:
        print(f"-- {p['name']} | {p['request']}")
        for cid, h in sorted(p["h2o"]["by_cand"].items(), key=lambda kv: int(kv[0])):
            print(f"     후보 {cid}: Hydrogen {p['hydrogen']['p_by_cand'].get(cid)} → H2O {h['p']:.2f} [{h['p_noul']:.2f}, {h['p_choice'] if h['p_choice'] is None else round(h['p_choice'], 2)}]")


if __name__ == "__main__":
    main()
