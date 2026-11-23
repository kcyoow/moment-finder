"""확정한 찾기 흐름에 "시트의 칸 대신 원본 크기 프레임으로 위치마다 묻기"를 적용한 결과의 채점. Jev 요청은 없다.
(pipeline_run·mc_all_eval을 불러 쓰므로 venv의 파이썬으로 돌린다)

가. Charades 새 영상 30개, 질문 99개, 한 곳 찾기 (results/full_sta2_preds.json: 위치마다 한 장과 4장 묶음)
    답 = 확률이 가장 높은 위치. 주 설정 = 4장 묶음. 보조 = 한 장, 둘의 평균, 4장과 시트 칸 확률의 평균
    첫 결과를 본 뒤 더한 것: 4장 묶음을 0.5초 어긋난 위치에도 물어 0.5초 간격으로 만든 것(results/full_sta2h_preds.json)
    채점은 validate_eval.py와 같다(앞뒤 0.5초가 주, 구간 안이 보조, 영상 단위 다시 뽑기). 견줄 값: 실제 jevreel, 시트만, 2단계
나. QVHighlights 20개 클립, 구간 84개, 모두 찾기 (results/full_qvm_preds.json: 위치마다 한 장)
    낸 시각 = 0.5 이상인 위치. 보조 = 시트 칸과 한 장이 모두 0.5 이상인 위치. 지표는 mc_all_eval.py와 같다
    정답 표시에 빈 곳이 있으므로, 앞서 눈으로 본 86곳(results/wrong_look_qvm.json)에서 한 장이 0.5 이상인 비율도 판정별로 센다

실행: <venv>/bin/python experiments/frame_limit/scripts/pipeline/full_eval.py
결과: 표 출력, results/full_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import mc_all_eval as M
from lev_probe_eval import boot
from pipeline_run import tile_probs

HERE = M.HERE
SLACK, CUT = 0.5, 0.5
BINS = [("2초 이하", 0, 2), ("3~5초", 3, 5), ("6초 이상", 6, 10 ** 9)]


def read(path):
    return json.loads((HERE / path).read_text())


def inside(t, it, slack=0.0):
    return any(s - slack <= t <= e + slack for s, e in it["windows"])


def charades(out):
    frames = read("data/sta2_frames.json")
    everything = read("data/tl_sta2_items.json")["items"]
    items = [it for it in everything if any(y := [inside(t, it) for t in frames[it["video"]]["centers"]]) and not all(y)]
    videos = [it["video"] for it in items]
    probs = {}  # (조건, 항목) → {시각: 확률}
    tokens = {"one": {}, "win4": {}, "win4h": {}}
    for path, rename in (("results/full_sta2_preds.json", {}), ("results/full_sta2h_preds.json", {"win4": "win4h"})):
        for r in read(path)["records"]:  # 둘째 파일은 0.5초 어긋난 위치(1.5, 2.5, …초)의 4장 묶음
            assert not r["error"]
            cond = rename.get(r["frames"], r["frames"])
            tokens[cond][r["video"]] = tokens[cond].get(r["video"], 0) + r["input_tokens"]
            for item, p in r["nouls"].items():
                probs.setdefault((cond, item), {})[r["t"]] = p
    stage1 = tile_probs(items, "sta2", "data/sheets60.json", "results/tl_sheet60_sta2_preds.json")
    real = {r["item"]: r for r in read("results/jevreel_real_sta2.json")["records"]}
    old = read("results/validate_sta2.json")["rows"]
    n_q = {v: sum(it["video"] == v for it in everything) for v in set(videos)}  # 한 요청에 그 영상의 질문을 모두 넣었다

    def pick(scores):
        return max(sorted(scores), key=scores.__getitem__)

    answers = {"실제 jevreel 0.2.0": [(real[it["id"]]["hits"][0]["at"] if real[it["id"]]["hits"] else None, real[it["id"]]["input_tokens"]) for it in items]}
    rows = {"원본 4장 묶음, 위치마다 (주 설정)": [], "원본 한 장, 위치마다": [], "한 장과 4장의 평균": [], "4장과 시트 칸의 평균": [],
            "원본 4장 묶음, 0.5초 간격": [], "4장 묶음 0.5초 간격과 시트 칸의 평균": []}
    for it in items:
        c = frames[it["video"]]["centers"]
        w4, one = probs[("win4", it["id"])], probs[("one", it["id"])]
        sheet = dict(zip(c, stage1[it["id"]]["probs"]))
        t4, t1 = tokens["win4"][it["video"]] / n_q[it["video"]], tokens["one"][it["video"]] / n_q[it["video"]]
        rows["원본 4장 묶음, 위치마다 (주 설정)"].append((pick(w4), t4))
        rows["원본 한 장, 위치마다"].append((pick(one), t1))
        rows["한 장과 4장의 평균"].append((pick({t: (w4[t] + one[t]) / 2 for t in c}), t4 + t1))
        rows["4장과 시트 칸의 평균"].append((pick({t: (w4[t] + sheet[t]) / 2 for t in c}), t4 + stage1[it["id"]]["tokens"]))
        dense = {**w4, **probs[("win4h", it["id"])]}
        th = tokens["win4h"][it["video"]] / n_q[it["video"]]
        tile = lambda t: sheet[t] if t in sheet else (sheet[round(t - 0.5, 2)] + sheet[round(t + 0.5, 2)]) / 2  # 사이 위치는 이웃 두 칸의 평균
        rows["원본 4장 묶음, 0.5초 간격"].append((pick(dense), t4 + th))
        rows["4장 묶음 0.5초 간격과 시트 칸의 평균"].append((pick({t: (p + tile(t)) / 2 for t, p in dense.items()}), t4 + th + stage1[it["id"]]["tokens"]))
    answers.update(rows)
    score = {name: {"loose": [float(t is not None and inside(t, it, SLACK)) for (t, _), it in zip(r, items)],
                    "strict": [float(t is not None and inside(t, it)) for (t, _), it in zip(r, items)]} for name, r in answers.items()}
    base = "실제 jevreel 0.2.0"
    print(f"===== 가. Charades 새 영상 30개, 질문 {len(items)}개: 한 곳 찾기 =====")
    print(f"{'방식':30s}{'맞힘(±0.5초)':>11s}{'맞힘(구간 안)':>10s}{'질문당 토큰':>10s}   jevreel과의 차이: ±0.5초 / 구간 안 (95% 구간)")
    out["charades"] = {"items": len(items), "rows": {}, "earlier": {k: [v["hit_loose"], v["hit_strict"], v["tokens"]] for k, v in old.items()}}
    for name, r in answers.items():
        tok = sum(x[1] for x in r) / len(r)
        row = {"hit_loose": int(sum(score[name]["loose"])), "hit_strict": int(sum(score[name]["strict"])), "tokens": round(tok)}
        diff = ""
        if name != base:
            for key in ("loose", "strict"):
                row[f"diff_{key}"] = list(boot(score[name][key], score[base][key], videos)[:3])
            diff = (f"{row['diff_loose'][0]:+.3f} [{row['diff_loose'][1]:+.3f}, {row['diff_loose'][2]:+.3f}] / "
                    f"{row['diff_strict'][0]:+.3f} [{row['diff_strict'][1]:+.3f}, {row['diff_strict'][2]:+.3f}]")
        out["charades"]["rows"][name] = row
        print(f"{name:30s}{row['hit_loose']:>8d}/{len(items)}{row['hit_strict']:>7d}/{len(items)}{row['tokens']:>12,d}   {diff}")
    print("  앞서 잰 값: " + ", ".join(f"{k} {v['hit_loose']}·{v['hit_strict']}" for k, v in old.items() if k != base))
    print("  정답 구간 길이별 맞힌 비율(±0.5초)")
    out["charades"]["by_length"] = {}
    for label, lo, hi in BINS:
        keep = [k for k, it in enumerate(items) if lo <= it["windows"][0][1] - it["windows"][0][0] <= hi]
        cell = {name: round(sum(score[name]["loose"][k] for k in keep) / len(keep), 2) for name in (base, "원본 4장 묶음, 위치마다 (주 설정)", "원본 4장 묶음, 0.5초 간격")}
        out["charades"]["by_length"][label] = {"items": len(keep), **cell}
        print(f"    {label}({len(keep)}개): " + ", ".join(f"{n} {v:.2f}" for n, v in cell.items()))


def qvh(out):
    items = read("data/qvm_items.json")["items"]
    frames = read("data/qvm_frames.json")
    earlier = M.reported(items, "qvm")
    one, tokens = {}, {}
    for r in read("results/full_qvm_preds.json")["records"]:
        assert not r["error"]
        tokens[r["video"]] = tokens.get(r["video"], 0) + r["input_tokens"]
        for item, p in r["nouls"].items():
            one.setdefault(item, {})[r["t"]] = p
    sheet_hot = {it["id"]: set(earlier[M.OURS_A][it["id"]][0]) for it in items}
    methods = dict(earlier)
    methods["원본 한 장, 위치마다"] = {it["id"]: ([t for t in frames[it["video"]]["centers"] if one[it["id"]][t] >= CUT], tokens[it["video"]]) for it in items}
    methods["시트 칸과 한 장이 모두 0.5 이상"] = {it["id"]: ([t for t in frames[it["video"]]["centers"] if one[it["id"]][t] >= CUT and t in sheet_hot[it["id"]]],
                                                tokens[it["video"]] + earlier[M.OURS_A][it["id"]][1]) for it in items}
    spans = [(it, w) for it in items for w in it["windows"]]
    truth = {it["id"]: {t for t in frames[it["video"]]["centers"] if any(M.near(t, w, 0.0) for w in it["windows"])} for it in items}
    total = sum(len(v) for v in truth.values())
    base_tokens = sum(methods[M.BASE][it["id"]][1] for it in items) / len(items)
    found = {}
    print(f"\n===== 나. QVHighlights 20개 클립, 구간 {len(spans)}개: 모두 찾기 =====")
    print(f"{'방식':30s}{'찾은 구간':>9s}{'낸 시각(맞음)':>14s}{'맞는 비율':>8s}{'정답 위치를 덮음 · 낸 위치 중 정답 (F1)':>30s}{'물음당 토큰':>12s}   jevreel과의 차이")
    out["qvh"] = {"rows": {}}
    for name, rows in methods.items():
        if name == M.EVERYTHING:
            continue
        times = [(t, it) for it in items for t in rows[it["id"]][0]]
        right = sum(any(M.near(t, w) for w in it["windows"]) for t, it in times)
        found[name] = [float(any(M.near(t, w) for t in rows[it["id"]][0])) for it, w in spans]
        tok = sum(rows[it["id"]][1] for it in items) / len(items)
        row = {"found": int(sum(found[name])), "reported": len(times), "right": right, "precision": round(right / max(len(times), 1), 3),
               "tokens": round(tok), "token_ratio": round(tok / base_tokens, 2)}
        cover = ""
        if not name.startswith("실제 jevreel"):
            hit = sum(t in truth[it["id"]] for t, it in times)
            row.update(covered=round(hit / total, 3), position_precision=round(hit / max(len(times), 1), 3), f1=round(2 * hit / (total + len(times)), 3))
            cover = f"{row['covered']:.3f} · {row['position_precision']:.3f} ({row['f1']:.3f})"
        diff = ""
        if name != M.BASE:
            row["diff_found"] = list(boot(found[name], found[M.BASE], [it["video"] for it, _ in spans])[:3])
            diff = f"{row['diff_found'][0]:+.3f} [{row['diff_found'][1]:+.3f}, {row['diff_found'][2]:+.3f}]"
        out["qvh"]["rows"][name] = row
        print(f"{name:30s}{row['found']:>6d}/{len(spans)}{len(times):>9d}({right}){row['precision']:>9.3f}{cover:>34s}{row['tokens']:>10,d} ({row['token_ratio']:.2f}배)   {diff}")
    # 앞서 눈으로 본 곳(시트만이 낸 정답 밖 덩어리의 가운데)에서 한 장의 답
    by_id = {it["id"]: it for it in items}
    seen = [p for p in read("results/wrong_look_qvm.json")["places"] if p["kind"] == "ours"]
    cell = {}
    for v, label in (("Y", "질문의 장면으로 본 곳"), ("N", "아니라고 본 곳"), ("U", "못 가린 곳")):
        ps = [p for p in seen if p["verdict"] == v and p["t"] in one[p["item"]]]
        cell[label] = [sum(one[p["item"]][p["t"]] >= CUT for p in ps), len(ps)]
    out["qvh"]["by_eye"] = cell
    print("  앞서 눈으로 본 정답 밖 86곳에서 원본 한 장이 0.5 이상인 수: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in cell.items()))


def main():
    out = {}
    charades(out)
    qvh(out)
    for name in ("full_sta2", "full_sta2h", "full_qvm"):
        m = read(f"results/{name}_preds.json")["meta"]
        out[f"cost_{name}"] = {"requests": m["saved"], "input_tokens": m["input_tokens"], "usd": m["cost_usd"]}
        print(f"비용 {name}: {m['saved']}건, {m['input_tokens']:,}토큰, ${m['cost_usd']}")
    (HERE / "results" / "full_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
