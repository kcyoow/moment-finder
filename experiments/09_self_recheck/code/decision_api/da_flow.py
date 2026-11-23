"""H2O 스스로 다시 확인하기를 흐름 전체로 센다. README.md 12절. 받아 둔 답만 쓴다(돈이 들지 않는다).

세는 법은 pipeline/mf_verify.py의 evaluate와 같다(p ≥ 0.5인 후보를 남긴다. 20절의 묶음은 p가 높은 순으로, 예산 15% 안에서).
실행: uv run --offline --no-project --with numpy --with scipy --with pillow --with imageio-ffmpeg python -I experiments/decision_api/da_flow.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import mf_verify as M  # noqa: E402

DIRS = {"verify_runs": "18절의 후보 (합친 점수에서 5개)", "verify_rr": "20절의 후보 (돌아가며 6개. 사이트의 방식)"}


def main():
    by = {}
    for r in json.loads((HERE / "results" / "h2o.json").read_text())["records"]:
        by.setdefault(r["uid"], {})[r["arm"]] = r["p"]["noul_def"]
    out = {"keep_p": M.KEEP_P, "checkers": ["hydrogen", "h2o4", "h2o8"], "groups": {}, "items": {}}
    rows = []
    for d, label in DIRS.items():
        for ip in sorted((ROOT / "pipeline" / d).glob("*/item.json")):
            item, verdict = json.loads(ip.read_text()), json.loads((ip.parent / "verdict.json").read_text())
            uid = lambda c: f"{d}/{item['name']}#{c['id']}"
            ps = {"hydrogen": {c["id"]: c["p"] for c in verdict["candidates"]},
                  "h2o4": {c["id"]: (by[uid(c)]["a"] + by[uid(c)]["b"]) / 2 for c in item["candidates"]},
                  "h2o8": {c["id"]: by[uid(c)]["8"] for c in item["candidates"]}}
            ev_t = [e["t"] for e in item["events"] if "t" in e]
            ev_iv = [(e["start"], e["end"]) for e in item["events"] if "start" in e]
            has = lambda a, b: any(a <= t < b for t in ev_t) or any(min(b, e1) - max(a, e0) >= 1.0 for e0, e1 in ev_iv)
            n_in = lambda cs: sum(any(c["start"] <= t < c["end"] for c in cs) for t in ev_t) + sum(any(min(c["end"], e1) - max(c["start"], e0) >= 1.0 for c in cs) for e0, e1 in ev_iv)
            count = lambda cs: {"clips": len(cs), "with_event": sum(has(c["start"], c["end"]) for c in cs), "events": n_in(cs)}
            now = [{"start": a, "end": b} for a, b in item["now_clips"]]
            rec = {"dir": d, "group": item["group"], "name": item["name"], "request": item["request"], "events": len(item["events"]), "now": count(now),
                   "perfect": count([c for c in item["candidates"] if has(c["start"], c["end"])])}
            for k, p in ps.items():
                kept = M.keep_clips(item, {"candidates": [{"id": i, "p": v} for i, v in p.items()]})
                rec[k] = {**count(kept), "kept": [{"id": c["id"], "start": c["start"], "end": c["end"], "p": round(p[c["id"]], 2), "has_event": bool(has(c["start"], c["end"])) if item["events"] else None} for c in kept],
                          "p": {i: round(v, 2) for i, v in p.items()}}
            rows.append(rec)
            out["items"][f"{d}/{item['name']}"] = rec
    for d, label in DIRS.items():
        for g in sorted({r["group"] for r in rows if r["dir"] == d}):
            rs = [r for r in rows if r["dir"] == d and r["group"] == g]
            agg = {"label": label, "items": len(rs), "events": sum(r["events"] for r in rs)}
            for k in ("now", "hydrogen", "h2o4", "h2o8", "perfect"):
                agg[k] = {m: sum(r[k][m] for r in rs) for m in ("clips", "with_event", "events")}
                if k not in ("now", "perfect"):
                    agg[k]["none_returned"] = sum(r[k]["clips"] == 0 for r in rs)
            out["groups"][f"{d}/{g}"] = agg
    (HERE / "results" / "flow.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for key, a in out["groups"].items():
        print(f"\n== {key} · {a['label']} · 항목 {a['items']}, 표시된 사건 {a['events']}")
        for k, name in (("now", "확인 없음(지금)"), ("hydrogen", "Hydrogen"), ("h2o4", "H2O 4장 둘"), ("h2o8", "H2O 8장"), ("perfect", "완벽한 확인")):
            v = a[k]
            print(f"   {name:14s} 클립 {v['clips']:3d} · 사건 든 클립 {v['with_event']:3d} · 클립에 든 사건 {v['events']:3d}" + (f" · '없음' {v['none_returned']}편" if "none_returned" in v else ""))
    print("\n== 로봇과 웹 실행: 영상마다 남긴 후보")
    for key, r in out["items"].items():
        if r["group"] in ("robot_Ra", "web"):
            print(f"-- {key} | {r['request']}")
            for k in ("hydrogen", "h2o4", "h2o8"):
                print(f"     {k:9s} p {r[k]['p']} → 남김 {[(c['id'], M.mmss(c['start']) + '~' + M.mmss(c['end'])) for c in r[k]['kept']] or '없음'}")


if __name__ == "__main__":
    main()
