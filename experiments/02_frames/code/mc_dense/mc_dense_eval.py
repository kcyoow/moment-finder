"""mc_dense_modal.py의 답을 채점한다. 표준 라이브러리만 쓴다. 규칙은 결과를 보기 전에 정했다(2026-11-12).

읽는 것: results/<모양>.json(답), results/truth.json(창 구간과 종류). 둘 다 Modal 볼륨 mf-mc-dense에서 받아 온다.

정의
- 기준은 둘이다. inv = 인벤토리 창이 열려 있다, chest = 상자 창이 열려 있다
- 위치 k = 영상의 k초부터 1초([k, k+1)). 모든 모양이 같은 위치에 답한다
- 정답 구간: 그 기준의 종류인 창 구간([첫 틱, 마지막 틱], 20분의 1초 단위). 구간의 끝은 마지막 틱 + 0.05초로 본다
- 위치가 "맞음": 정답 구간이 [k, k+1)과 겹친다. "앞뒤 0.5초까지 맞음": [k−0.5, k+1.5)와 겹친다
- 상자 기준에서, 상자와 같은 모양의 다른 보관함(통, 엔더 상자 등 kind=container)과 겹치는 위치는 세지 않는다
- 잡음: 확률 0.5 이상
- 구간을 "찾음": 그 구간과 겹치는 위치 가운데 잡은 것이 하나라도 있다. 길이별(0.5초 미만, 0.5~1초, 1초 이상)로 나눠 센다
- 정밀도: 잡은 위치 중 맞음의 비율(엄격), 앞뒤 0.5초까지 맞음의 비율
- 헛짚음의 종류: 잡았는데 맞지 않은 위치가 다른 종류의 창과 겹치는가(다른 창), 아무 창도 없는가(창 없음)
- AUC: 위치를 맞음/아님으로 가르는 정도(0.5와 무관)
- 비용: 요청당 토큰의 중앙값(= 영상 1초당 토큰)
- 차이의 구간: 지금 방법(sep4_half, 물음 show)과의 "찾은 구간 비율" 차이를 영상째 다시 뽑아(2,000번) 낸다

실행: python3 mc_dense_eval.py [--try N]  → results/mc_dense_eval.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "open_model"))
from open_eval import auc  # noqa: E402

SEED, BOOT, CUT, TICK = 20261112, 2000, 0.5, 0.05
BASE = ("sep4_half", "show")
BINS = (("0.5초 미만", 0, 0.5), ("0.5~1초", 0.5, 1.0), ("1초 이상", 1.0, 1e9))


def overlaps(k, a, b, lo=0.0, hi=1.0):
    return a < k + hi and b + TICK > k + lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--try", dest="try_n", type=int, default=0)
    ap.add_argument("--small", action="store_true", help="작은 시험: sep4_half.json(전체)과 *_small.json을 읽고, 모든 모양에 답이 있는 위치만 채점한다")
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    res = HERE / "results"
    truth = {v["vid"]: v for v in json.loads((res / "truth.json").read_text())["videos"]}
    suffix = f"_try{args.try_n}" if args.try_n else ""
    out_tag = "_small" if args.small else suffix

    runs = {}   # (모양, 물음) → {기준: {(vid, k): p}}, 그리고 메타
    meta = {}
    for path in sorted(res.glob("*.json")):
        stem = path.stem
        if stem.startswith(("truth", "mc_dense_eval", "chosen")):
            continue
        if args.small:
            if not (stem.endswith("_small") or stem == BASE[0]):
                continue
            name = stem[:-6] if stem.endswith("_small") else stem
        elif suffix:
            if not stem.endswith(suffix):
                continue
            name = stem[:-len(suffix)]
        else:
            if "_try" in stem or stem.endswith("_small"):
                continue
            name = stem
        d = json.loads(path.read_text())
        meta[name] = d["meta"]
        for r in d["records"]:
            if "error" in r:
                continue
            for q, p in r["probs"].items():
                c, _, a = q.partition("_")
                runs.setdefault((name, a or "show"), {}).setdefault(c, {})[(r["vid"], r["k"])] = p
    if args.small:   # 모든 모양에 답이 있는 위치만 남긴다
        common = None
        for by_c in runs.values():
            for got in by_c.values():
                common = set(got) if common is None else common & set(got)
        for by_c in runs.values():
            for c in by_c:
                by_c[c] = {k: v for k, v in by_c[c].items() if k in common}
        print("작은 시험: 함께 채점하는 위치", len(common or []))
    if BASE not in runs:
        print("기준 모양의 답이 없다:", BASE, "| 있는 것:", sorted(runs))

    def score(got, c, vids):
        """got: {(vid, k): p}. 반환: 영상별 조각(다시 뽑기에 쓴다)과 합친 값."""
        per = {}
        for vid in vids:
            v = truth[vid]
            pos_iv = [(it["start"], it["end"]) for it in v["intervals"] if it["kind"] == c]
            skip_iv = [(it["start"], it["end"]) for it in v["intervals"] if c == "chest" and it["kind"] == "container"]
            other_iv = [(it["start"], it["end"]) for it in v["intervals"] if it["kind"] != c and (it["start"], it["end"]) not in skip_iv]
            rows, found = [], [False] * len(pos_iv)
            for k in range(v["seconds"]):
                if (vid, k) not in got or any(overlaps(k, a, b) for a, b in skip_iv):
                    continue
                p = got[(vid, k)]
                y = any(overlaps(k, a, b) for a, b in pos_iv)
                near = y or any(overlaps(k, a, b, -0.5, 1.5) for a, b in pos_iv)
                oth = any(overlaps(k, a, b) for a, b in other_iv)
                rows.append((p, y, near, oth))
                if p >= CUT:
                    for i, (a, b) in enumerate(pos_iv):
                        if overlaps(k, a, b):
                            found[i] = True
            per[vid] = {"rows": rows, "iv": [(b - a + TICK, f) for (a, b), f in zip(pos_iv, found)]}
        return per

    def summary(per, vids):
        rows = [r for v in vids for r in per[v]["rows"]]
        iv = [x for v in vids for x in per[v]["iv"]]
        fl = [r for r in rows if r[0] >= CUT]
        out = {"positions": len(rows), "positive_positions": sum(r[1] for r in rows), "flagged": len(fl),
               "precision": (sum(r[1] for r in fl) / len(fl)) if fl else None, "precision_near": (sum(r[2] for r in fl) / len(fl)) if fl else None,
               "false_other_window": sum(1 for r in fl if not r[2] and r[3]), "false_no_window": sum(1 for r in fl if not r[2] and not r[3]),
               "auc": auc([(r[0], int(r[1])) for r in rows]), "intervals": len(iv), "found": sum(f for _, f in iv),
               "found_rate": (sum(f for _, f in iv) / len(iv)) if iv else None, "by_length": {}}
        for label, lo, hi in BINS:
            sub = [f for length, f in iv if lo <= length < hi]
            out["by_length"][label] = [sum(sub), len(sub)]
        short = [f for length, f in iv if length < 1.0]
        out["found_short"], out["short"] = sum(short), len(short)
        return out

    out = {"base": list(BASE), "cut": CUT, "meta": {k: {x: m.get(x) for x in ("requests", "errors", "tokens_median", "wall_seconds", "spec")} for k, m in meta.items()}, "criteria": {}}
    sets = {"all": sorted(truth), "old": sorted(v for v in truth if v.startswith("mc_")), "new": sorted(v for v in truth if v.startswith("ch_"))}
    for c in ("inv", "chest"):
        out["criteria"][c] = {}
        for sname, vids in sets.items():
            if not vids or BASE not in runs or c not in runs[BASE]:
                continue
            vids = [v for v in vids if any(key[0] == v for key in runs[BASE][c])]
            if not vids:
                continue
            base_per = score(runs[BASE][c], c, vids)
            table = {}
            for (layout, a), by_c in sorted(runs.items()):
                if c not in by_c:
                    continue
                per = score(by_c[c], c, vids)
                s = summary(per, vids)
                s["tokens_median"] = meta[layout].get("tokens_median")
                if (layout, a) != BASE and args.boot and s["intervals"]:
                    rng, d_all, d_short = random.Random(SEED), [], []
                    for _ in range(args.boot):
                        pick = rng.choices(vids, k=len(vids))
                        for key, dest, test in (("all", d_all, lambda L: True), ("short", d_short, lambda L: L < 1.0)):
                            m = [f for v in pick for L, f in per[v]["iv"] if test(L)]
                            b = [f for v in pick for L, f in base_per[v]["iv"] if test(L)]
                            if m:
                                dest.append(sum(m) / len(m) - sum(b) / len(b))
                    for key, vals in (("found_rate_diff", d_all), ("found_short_rate_diff", d_short)):
                        if vals:
                            vals.sort()
                            s[key + "_interval"] = [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]
                table[f"{layout}|{a}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
            out["criteria"][c][sname] = {"videos": len(vids), "variants": table}
    (res / f"mc_dense_eval{out_tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")

    for c, by_set in out["criteria"].items():
        for sname, t in by_set.items():
            print(f"\n[{c} / {sname}] 영상 {t['videos']}개")
            print(f"{'모양|물음':24s}{'토큰':>6s}{'찾은 구간':>10s}{'<0.5초':>8s}{'0.5~1초':>8s}{'1초 이상':>8s}{'잡은 위치':>8s}{'정밀':>7s}{'±0.5정밀':>8s}{'다른창':>6s}{'창없음':>6s}{'AUC':>7s}  짧은 구간 찾는 비율 차이[95%]")
            for name, s in t["variants"].items():
                bl = s["by_length"]
                f = lambda x: "   -  " if x is None else f"{x:6.3f}"
                print(f"{name:24s}{str(s['tokens_median']):>6s}{s['found']:>6d}/{s['intervals']:<3d}" + "".join(f"{bl[b][0]:>5d}/{bl[b][1]:<2d}" for b, _, _ in BINS)
                      + f"{s['flagged']:>8d} {f(s['precision'])} {f(s['precision_near'])}{s['false_other_window']:>6d}{s['false_no_window']:>6d} {f(s['auc'])}  {s.get('found_short_rate_diff_interval', '')}")


if __name__ == "__main__":
    main()
