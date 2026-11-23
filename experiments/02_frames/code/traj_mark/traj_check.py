"""모델 없이 하는 확인: 카메라를 뺀 뒤 남는 움직임(R)이 동작 구간 안에서 더 큰가, 빼기 전(U)보다 그 차이가 뚜렷한가.

정의는 README의 "모델 없이 하는 확인"에 점수를 내기 전에 적어 둔 것과 같다.
- 동작 구간: ../frame_limit/data/tl_sta_items.json(hold 묶음 44개 질문, TimeLens-Bench)에서 영상마다 모든 구간의 합
- 1초 구간이 그 합과 50% 이상 겹치면 안, 0이면 밖, 그 사이는 뺀다
- 영상마다 AUC(안 하나 > 밖 하나일 확률, 같으면 반). 안이나 밖이 3개 미만인 영상은 요약에서 뺀다
- 읽는 법: R의 AUC 평균 ≥ 0.60이고 영상의 2/3 이상에서 > 0.5면 "신호 있음". 영상의 2/3 이상에서 R의 AUC > U의 AUC면 "보정이 낫게 한다"
- 덧붙여(판정에 안 씀): 4초 이하 구간만 안으로 친 것, 참고 방법(ls, aff, hom), 3% 넘은 점 수(over, over_raw와 견줌)
- 그림을 본 뒤 덧붙인 참고값(판정에 안 씀): U_end, R_end = 끝 프레임에서 보이는 점만으로 잰 U, R

실행: python3 traj_check.py   (traj_comp.py가 만든 index_comp.json을 읽는다) → check_comp.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = HERE.parent / "frame_limit" / "data" / "tl_sta_items.json"
MIN_SIDE = 3
SHORT = 4.0
BASE = {"R": "U", "R_ls": "U", "R_aff": "U", "R_hom": "U", "over": "over_raw", "R_end": "U_end"}   # 무엇과 견주나


def union(windows):
    out = []
    for s, e in sorted(windows):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(t0, t, windows):
    return sum(max(0.0, min(t, e) - max(t0, s)) for s, e in windows) / max(t - t0, 1e-9)


def auc(inside, outside):
    if not inside or not outside:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in inside for b in outside)
    return wins / (len(inside) * len(outside))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run(segs, labels, keys, short_only):
    per_video = {}
    for video in sorted(labels):
        all_w = union([w for q in labels[video] for w in q["windows"]])
        in_w = union([w for q in labels[video] for w in q["windows"] if not short_only or w[1] - w[0] <= SHORT])
        rows = [x for x in segs if x["video"] == video]
        inside = [x for x in rows if in_w and overlap(x["t0"], x["t"], in_w) >= 0.5]
        outside = [x for x in rows if overlap(x["t0"], x["t"], all_w) == 0]
        rec = {"n_in": len(inside), "n_out": len(outside), "eligible": len(inside) >= MIN_SIDE and len(outside) >= MIN_SIDE}
        for k in keys:
            a = auc([x[k] for x in inside], [x[k] for x in outside])
            rec[k] = None if a is None else round(a, 3)
        rec["median_in"] = {k: round(sorted(x[k] for x in inside)[len(inside) // 2], 4) for k in ("U", "R")} if inside else None
        rec["median_out"] = {k: round(sorted(x[k] for x in outside)[len(outside) // 2], 4) for k in ("U", "R")} if outside else None
        per_video[video] = rec
    ok = [v for v in per_video.values() if v["eligible"]]
    summary = {"videos": len(ok)}
    for k in keys:
        summary[f"mean_{k}"] = round(mean([v[k] for v in ok]), 3) if ok else None
        summary[f"above_half_{k}"] = sum(v[k] > 0.5 for v in ok)
    for k, base in BASE.items():
        if k in keys:
            summary[f"{k}_beats_{base}"] = sum(v[k] > v[base] for v in ok)
    return per_video, summary


def main():
    segs = json.loads((HERE / "index_comp.json").read_text())["items"]
    labels = {}
    for q in json.loads(ITEMS.read_text())["items"]:
        labels.setdefault(q["video"], []).append(q)
    keys = ["U", "R", "R_ls", "R_aff", "R_hom", "over_raw", "over", "U_end", "R_end"]
    result = {}
    for name, short in (("main", False), ("short_windows", True)):
        per_video, summary = run(segs, labels, keys, short)
        result[name] = {"per_video": per_video, "summary": summary}
        print(f"\n== {name} (안 = {'4초 이하 구간만' if short else '모든 구간의 합'}) ==")
        print(f"{'영상':10s} 안 밖  " + " ".join(f"{k:>8s}" for k in keys))
        for video, v in per_video.items():
            cells = " ".join(f"{'-' if v[k] is None else format(v[k], '.2f'):>8s}" for k in keys)
            print(f"{video:10s} {v['n_in']:2d} {v['n_out']:2d}{' ' if v['eligible'] else '*'} {cells}")
        print("요약(* 표시 영상 제외):", json.dumps(summary, ensure_ascii=False))
        if name == "main":
            n = summary["videos"]
            signal = summary["mean_R"] >= 0.60 and summary["above_half_R"] >= 2 * n / 3
            better = summary["R_beats_U"] >= 2 * n / 3
            verdict = ("동작에 대한 신호가 있다" if signal else "뚜렷한 신호 없음") + (", 보정이 신호를 낫게 한다" if better else ", 보정이 신호를 낫게 한다고 할 수 없다")
            result["verdict"] = verdict
            print("미리 정한 규칙에 따른 판정:", verdict)
    # 덧붙여: 카메라가 거의 멈춘 구간(처음→끝 이동 < 너비 1%)에서 보정 전후로 고른 점이 얼마나 같은가.
    # 같은 고르기 규칙(끝 프레임에서 보이는 점만)으로 견준 것이 보정만의 효과다. img/와 견준 값은 규칙 차이까지 들어간다
    still = [x for x in segs if x["cam"] < 0.01]

    def jac(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b)) if (a or b) else None
    result["still_camera"] = {"segments": len(still),
                              "jaccard_same_rule": round(mean([jac(x["ids_raw_end"], x["ids_comp"]) for x in still]), 3),
                              "jaccard_vs_img": round(mean([jac(x["ids_raw"], x["ids_comp"]) for x in still]), 3)}
    print("\n카메라가 거의 멈춘 구간:", result["still_camera"])
    (HERE / "check_comp.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
