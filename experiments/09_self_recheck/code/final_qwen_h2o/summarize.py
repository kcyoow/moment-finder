"""results/<이름>/status.json을 읽어 표 하나로 정리하고(results/summary.json), Space에 넣을 예시(space/examples/<이름>/)를 만든다.

  python3 experiments/final_qwen_h2o/summarize.py            # 표
  python3 experiments/final_qwen_h2o/summarize.py --examples robot_ok robot_none   # 예시도 만든다

클립이 맞는지: 클립의 절반 이상이 맞는 구간 안이거나, 맞는 구간의 절반 이상을 덮으면 맞음(README 2절).
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = HERE.parent.parent / "space"
TRUTH = {"robot_ok": [(42, 62)], "robot_none": [], "mc_diamond": [(87, 114)], "mc_creeper": [(154, 175), (191, 211), (212, 233)], "mc_fight": [], "mc_mixed": []}
TITLE = {"robot_ok": "로봇 실습 65초 · 성공 장면이 한 번 있는 영상", "robot_none": "로봇 실습 72초 · 성공 장면이 없는 영상"}


# 구간 겹침으로는 가르지 못해 프레임을 직접 열어 본 것(results/<이름>/cand_<순위>_*.jpg). (이름, 시작, 끝) → (맞음?, 본 것)
SEEN = {("mc_creeper", 166.0, 185.0): (True, "굴 틈으로 크리퍼가 보인다(위 줄 네 장과 아래 첫 장)"),
        ("mc_diamond", 286.0, 300.0): (False, "용암 옆에서 돌과 자갈을 캔다. 다이아몬드 광석은 없다"),
        ("mc_diamond", 6.0, 26.0): (False, "어두운 굴에서 돌을 캔다. 다이아몬드 광석은 없다")}


def right(c, truth, name=None):
    if (name, c["start"], c["end"]) in SEEN:
        return SEEN[(name, c["start"], c["end"])][0]
    for a, b in truth:
        ov = min(c["end"], b) - max(c["start"], a)
        if ov > 0 and (ov >= 0.5 * (c["end"] - c["start"]) or ov >= 0.5 * (b - a)):
            return True
    return False


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    args = sys.argv[1:]
    base = HERE / (args[args.index("--dir") + 1] if "--dir" in args else "results")     # results(첫 실행) 또는 results_v2(고친 뒤)
    out, tot = {}, {"shown": 0, "right": 0, "gpu": 0.0}
    for name, truth in TRUTH.items():
        f = base / name / "status.json"
        if not f.exists():
            print(f"{name}: (아직 없음)")
            continue
        st = json.loads(f.read_text(encoding="utf-8"))
        r, d = st["result"], st["result"]["details"]
        cands = d["candidates"]
        hit = [c for c in cands if c.get("match")]
        ok = [right(c, truth, name) for c in hit]
        v2 = base / name / "plan_v2.json"                       # 고친 뒤: 계획을 새로 쓴 시간은 plan_v2.json에 있다
        wplan = json.loads(v2.read_text(encoding="utf-8")) if v2.exists() else d["plan"]
        gw = sum(c["seconds"] for c in wplan.get("writer_calls", []))
        gj = sum(c["gpu_seconds"] for c in d.get("judge_calls") or [])
        gr = (d.get("recheck_call") or {}).get("gpu_seconds", 0.0)
        row = {"request": r["request"], "truth": [f"{mmss(a)}~{mmss(b)}" for a, b in truth] or "없음", "candidates": len(cands),
               "shown": [f"{mmss(c['start'])}~{mmss(c['end'])} ({c['p']:.2f})" for c in hit], "shown_right": sum(ok), "shown_wrong": len(ok) - sum(ok),
               "rejected": [f"{mmss(c['start'])}~{mmss(c['end'])} (예/아니오 {c['p_noul']:.2f}, 고르기 {c['p_choice']:.2f}{', 맞는 구간과 겹침' if right(c, truth, name) else ''})" for c in cands if not c.get("match")],
               "format_retries": d["plan"].get("format_retries"), "checked_by_eye": {f"{mmss(s)}~{mmss(e)}": v[1] for (n, s, e), v in SEEN.items() if n == name},
               "gpu_seconds": {"writer": round(gw, 1), "judge": round(gj, 1), "recheck": round(gr, 1), "function_total": round(gw + gj + gr, 1)},
               "wall_seconds": st.get("wall_seconds")}
        out[name] = row
        tot["shown"] += len(hit); tot["right"] += sum(ok); tot["gpu"] += gw + gj + gr
        print(f"{name}: 맞는 구간 {row['truth']} | 보여 준 클립 {row['shown']} → 맞음 {row['shown_right']}, 틀림 {row['shown_wrong']} | 빠진 후보 {len(row['rejected'])} | "
              f"GPU(함수 안) 쓰기 {gw:.0f} + 판정 {gj:.0f} + 다시 확인 {gr:.0f}초")
        for x in row["rejected"]:
            print("     빠짐:", x)
    out["_total"] = {"shown": tot["shown"], "right": tot["right"], "wrong": tot["shown"] - tot["right"], "gpu_function_seconds": round(tot["gpu"], 1)}
    print("합계:", out["_total"])
    (base / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if "--examples" in args:
        for name in args[args.index("--examples") + 1:]:
            st = json.loads((HERE / "results" / name / "status.json").read_text(encoding="utf-8"))
            r = st["result"]
            dst = SPACE / "examples" / name
            dst.mkdir(parents=True, exist_ok=True)
            for c in r["clips"]:
                shutil.copy(HERE / "results" / name / c["file"], dst / c["file"])
            ex = {"title": TITLE.get(name, name), **{k: r[k] for k in ("request", "proposal", "summary", "clips", "columns", "rows", "curve", "log", "details")}}
            (dst / "example.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
            print("예시:", dst)


if __name__ == "__main__":
    main()
