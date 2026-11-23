"""찾기 흐름별로 정확도와 영상 하나당 시간을 d1-2B와 Jev에 같은 식으로 낸다. 표준 라이브러리만 쓴다. Jev 요청은 없다.

Charades 30개 영상, 질문 100개. 맞음 = 고른 위치가 정답 구간 안(엄격한 기준). 최고 확률이 같으면 앞선 위치를 고른다.
흐름
- 시트만      : 시트에서 확률이 가장 높은 칸의 위치
- 확인 k      : 시트에서 확률이 높은 칸 k개만 4장 묶음으로 확인해 그중 묶음 확률이 가장 높은 곳 (요청 수를 줄이는 흐름)
- 1초 간격    : 1초마다 4장 묶음, 최고 위치
- 1초+시트    : 묶음 확률과 그 시각의 칸 확률의 평균
- 0.5초 간격  : 1초 위치 + 0.5초 어긋난 위치
- 0.5초+시트  : 위와 같되 칸 확률과 평균(0.5초 위치는 앞뒤 두 칸의 평균)
시간 = 그 흐름에 든 요청들의 시간 합을 영상마다 내어 평균한 것. d1은 L4에서 하나씩 처리한 시간, Jev는 받아 둔 응답 시간(API 왕복)의 합이다.
Jev는 요청을 동시에 보낼 수 있어서 실제 걸린 시간은 이 합보다 훨씬 짧았다(보고서 참고). 시트 요청은 질문마다 하나, 묶음 요청은 위치마다 하나(그 영상의 질문 전부).

읽는 것: pack/manifest*.json, results/d1-2b__full_video.json, d1-2b__half_video.json, d1-2b__sheet.json(, d1-2b__sheet_set64.json),
        ../frame_limit/results 의 Jev 응답 시간
실행: python3 experiments/open_model/flow_eval.py
결과: results/flow_eval.json
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "frame_limit"
CUT, BOOT, SEED = 0.5, 2000, 20261111
FLOWS = ["시트만", "확인 1", "확인 3", "확인 5", "1초 간격", "1초+시트", "0.5초 간격", "0.5초+시트"]


def read(path):
    return json.loads(path.read_text())


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def auc(pairs):
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    r = ranks([p for p, _ in pairs])
    return round((sum(ri for ri, (_, y) in zip(r, pairs) if y) - pos * (pos + 1) / 2) / (pos * neg), 3)


def answers(name):
    path = HERE / "results" / f"{name}.json"
    if not path.exists():
        return None
    return {r["id"]: r for r in read(path)["records"] if not r["error"]}


def median(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 3) if xs else None


def flows(pos1, pos2, tiles, gt, t_pos, t_sheet, videos):
    """pos1/pos2: {(영상, t): {질문: 확률}}, tiles: {(영상, 질문): {t: 확률}}, gt: {(영상, t, 질문): 0/1}. → 흐름별 (질문별 맞음, 영상별 시간)"""
    hits = {f: {} for f in FLOWS}
    secs = {f: dict.fromkeys(videos, 0.0) for f in FLOWS}
    for v in videos:
        ones = sorted(t for (vv, t) in pos1 if vv == v)
        halves = sorted(t for (vv, t) in pos2 if vv == v)
        qs = sorted(q for (vv, q) in tiles if vv == v)
        sheet_time = sum(t_sheet[(v, q)] for q in qs)
        one_time = sum(t_pos[(v, t)] for t in ones)
        half_time = sum(t_pos[(v, t)] for t in halves)
        for f, s in (("시트만", sheet_time), ("1초 간격", one_time), ("1초+시트", one_time + sheet_time),
                     ("0.5초 간격", one_time + half_time), ("0.5초+시트", one_time + half_time + sheet_time)):
            secs[f][v] = s
        checked = {k: set() for k in (1, 3, 5)}
        for q in qs:
            tile = tiles[(v, q)]

            def near(t):  # 그 시각의 칸 확률(0.5초 위치는 앞뒤 두 칸의 평균). 칸이 없으면 None
                if t in tile:
                    return tile[t]
                side = [tile[x] for x in (round(t - 0.5, 2), round(t + 0.5, 2)) if x in tile]
                return sum(side) / len(side) if side else None

            def best(cands, score):
                top = max(cands, key=lambda t: (score(t), -t))
                return gt[(v, top, q)]
            plain = lambda t: (pos1.get((v, t)) or pos2[(v, t)])[q]
            mixed = lambda t: plain(t) if near(t) is None else (plain(t) + near(t)) / 2
            in_ones = [t for t in sorted(tile) if (v, t) in pos1]
            hits["시트만"][(v, q)] = best(in_ones, lambda t: tile[t])
            hits["1초 간격"][(v, q)] = best(ones, plain)
            hits["1초+시트"][(v, q)] = best(ones, mixed)
            hits["0.5초 간격"][(v, q)] = best(ones + halves, plain)
            hits["0.5초+시트"][(v, q)] = best(ones + halves, mixed)
            order = sorted(in_ones, key=lambda t: (-tile[t], t))
            for k in (1, 3, 5):
                hits[f"확인 {k}"][(v, q)] = best(order[:k], plain)
                checked[k] |= set(order[:k])
        for k in (1, 3, 5):
            secs[f"확인 {k}"][v] = sheet_time + sum(t_pos[(v, t)] for t in checked[k])
    return hits, secs


def interval(by_video):
    rng, videos, vals = random.Random(SEED), sorted(by_video), []
    for _ in range(BOOT):
        rows = [d for v in rng.choices(videos, k=len(videos)) for d in by_video[v]]
        vals.append(sum(rows) / len(rows))
    vals.sort()
    return [round(vals[int(0.025 * BOOT)], 3), round(vals[int(0.975 * BOOT) - 1], 3)]


def main():
    m1 = [it for it in read(HERE / "pack" / "manifest.json")["items"] if it["task"] == "charades"]
    m2 = read(HERE / "pack" / "manifest2.json")["items"]
    half_items = [it for it in m2 if it["task"] == "charades_h"]
    sheet_items = [it for it in m2 if it["task"] == "sheet_sta2"]
    videos = sorted({it["video"] for it in m1})
    gt = {(it["video"], it["t"], q): y for it in m1 + half_items for q, y in it["gt"].items()}
    duration = sum(v["duration"] for v in read(SRC / "data" / "sta2h_frames.json").values())
    out = {"_note": "flow_eval.py의 결과. 시간은 영상 하나당 요청 시간의 합(초)의 평균", "videos": len(videos), "video_seconds_total": round(duration, 1)}

    d_one, d_half, d_sheet = answers("d1-2b__full_video"), answers("d1-2b__half_video"), answers("d1-2b__sheet")
    j_one = {(r["video"], r["t"]): r for r in read(SRC / "results" / "full_sta2_preds.json")["records"] if r["frames"] == "win4" and not r["error"]}
    j_half = {(r["video"], r["t"]): r for r in read(SRC / "results" / "full_sta2h_preds.json")["records"] if not r["error"]}
    j_sheet = {(r["video"], r["item"]): r for r in read(SRC / "results" / "tl_sheet60_sta2_preds.json")["records"] if r["kind"] == "tiles" and not r["error"]}
    sides = {"jev": (
        {(it["video"], it["t"]): it["jev"] for it in m1}, {(it["video"], it["t"]): it["jev"] for it in half_items},
        {(it["video"], it["item"]): {it["tile_t"][k]: p for k, p in it["jev"].items()} for it in sheet_items},
        {**{k: r["latency"] for k, r in j_one.items()}, **{k: r["latency"] for k, r in j_half.items()}},
        {k: r["latency"] for k, r in j_sheet.items()})}
    if d_one and d_half and d_sheet:
        sides["d1-2b"] = (
            {(it["video"], it["t"]): d_one[it["id"]]["probs"] for it in m1}, {(it["video"], it["t"]): d_half[it["id"]]["probs"] for it in half_items},
            {(it["video"], it["item"]): {it["tile_t"][k]: p for k, p in d_sheet[it["id"]]["probs"].items()} for it in sheet_items},
            {**{(it["video"], it["t"]): d_one[it["id"]]["seconds"] for it in m1}, **{(it["video"], it["t"]): d_half[it["id"]]["seconds"] for it in half_items}},
            {(it["video"], it["item"]): d_sheet[it["id"]]["seconds"] for it in sheet_items})
    results = {name: flows(a[0], a[1], a[2], gt, a[3], a[4], videos) for name, a in sides.items()}
    print(f"Charades {len(videos)}개 영상(합 {duration:.0f}초), 질문 {len(next(iter(results.values()))[0]['1초 간격'])}개")
    print(f"{'흐름':12s}" + "".join(f"{n + ' 맞음':>14s}{n + ' 초/영상':>16s}{'배속':>7s}" for n in results) + ("   d1−Jev 차이(구간)" if "d1-2b" in results else ""))
    out["flows"] = {}
    for f in FLOWS:
        row, line = {}, f"{f:12s}"
        for name, (hits, secs) in results.items():
            per = sum(secs[f].values()) / len(videos)
            row[name] = {"hits": sum(hits[f].values()), "questions": len(hits[f]), "seconds_per_video": round(per, 2),
                         "times_realtime": round(duration / sum(secs[f].values()), 1)}
            line += f"{row[name]['hits']:>10d}/{row[name]['questions']}{per:>16.2f}{row[name]['times_realtime']:>7.1f}"
        if "d1-2b" in results:
            by = {}
            for (v, q), h in results["d1-2b"][0][f].items():
                by.setdefault(v, []).append(h - results["jev"][0][f][(v, q)])
            diff = sum(d for ds in by.values() for d in ds) / sum(len(ds) for ds in by.values())
            row["d1_minus_jev"] = [round(diff, 3)] + interval(by)
            line += f"   {diff:+.2f} {row['d1_minus_jev'][1:]}"
        out["flows"][f] = row
        print(line)

    # 시트 자체: 칸 단위
    out["sheets"] = {}
    for name in ("d1-2b__sheet", "d1-2b__sheet_set64"):
        got = answers(name)
        if not got:
            continue
        for task in ("sheet_sta2", "sheet_mc", "sheet_qvm"):
            mine = [it for it in m2 if it["task"] == task and it["id"] in got]
            res = {"requests": len(mine), "seconds_median": median(got[it["id"]]["seconds"] for it in mine)}
            for side, pick in (("d1", lambda it: got[it["id"]]["probs"]), ("jev", lambda it: it["jev"])):
                pairs, found, runs, outside = [], 0, 0, 0
                for it in mine:
                    p, ids, run = pick(it), sorted(it["gt"]), []
                    pairs += [(p[k], it["gt"][k]) for k in ids]
                    outside += sum(p[k] >= CUT and not it["gt"][k] for k in ids)
                    for k in ids + [None]:  # 정답인 칸이 이어진 덩어리를 구간 하나로 본다
                        if k is not None and it["gt"][k]:
                            run.append(p[k])
                        elif run:
                            runs += 1
                            found += max(run) >= CUT
                            run = []
                res[side] = {"tile_auc": auc(pairs), "intervals_found": f"{found}/{runs}", "tiles_on_outside": outside,
                             "tiles_on": sum(p >= CUT for p, _ in pairs), "tiles": len(pairs)}
            out["sheets"][f"{name}/{task}"] = res
            print(f"{name:20s}{task:11s} 요청 {res['requests']:>3d}, 요청당 {res['seconds_median']}초 | d1: AUC {res['d1']['tile_auc']}, 구간 {res['d1']['intervals_found']}, "
                  f"밖에서 켜짐 {res['d1']['tiles_on_outside']} | Jev: AUC {res['jev']['tile_auc']}, 구간 {res['jev']['intervals_found']}, 밖에서 켜짐 {res['jev']['tiles_on_outside']}")
    lat = {"jev_bundle_median": median(r["latency"] for r in j_one.values()), "jev_sheet_median": median(r["latency"] for r in j_sheet.values())}
    if "d1-2b" in sides:
        lat.update(d1_bundle_median=median(d_one[it["id"]]["seconds"] for it in m1), d1_sheet_median=median(d_sheet[it["id"]]["seconds"] for it in sheet_items))
    out["request_seconds"] = lat
    print("요청당 시간(중앙값, 초):", lat)
    probe = HERE / "results" / "speed_probe.json"
    if probe.exists():
        out["speed_probe"] = read(probe)
    (HERE / "results" / "flow_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
