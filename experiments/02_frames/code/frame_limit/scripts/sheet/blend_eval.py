"""시트의 칸마다 프레임 한 장 대신 4프레임을 겹친 그림을 넣었을 때를 견준다. 표준 라이브러리만 쓴다.

겹침: 그 시각 둘레의 4프레임(0.5초 간격, 1.5초 폭. 4장 조건에서 쓴 프레임)을 같은 비중으로 평균 낸 것.
색 겹침: 겹치기 전에 프레임마다 시간 순서대로 색(빨강, 노랑, 청록, 파랑)을 입힌 것. 멈춘 곳은 원래 색으로 남는다.
시트의 격자, 칸 크기, 질문 문구는 한 장 시트와 같고, 겹친 시트에는 칸이 어떤 그림인지(색의 뜻 포함) 설명을 더했다.

견주는 묶음
- Charades 12개 영상, 고친 질문(TimeLens-Bench) 44개: 1초 간격(기존 시트와 짝)과 2초 간격(하나 걸러)
- QVHighlights 본 10개 영상(2초 간격): 고친 질문 11개와 원래 질문 10개

보는 것 (실행 전에 정한 것)
- 주 지표: 칸마다 방식의 항목별 AUC에서 겹친 것 − 한 장, 영상 단위로 다시 뽑은 95% 구간
- 함께: 고르기 방식, 최고점 적중, 같은 위치에서의 Jev 한 장·4장(이미 있는 확률), 구간 길이별

실행: python3 experiments/frame_limit/scripts/sheet/blend_eval.py
결과: 표 출력, results/blend_eval.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from lev_probe_eval import boot
from qv_eval import auc

HERE = E.HERE
BINS = [("2초 이하", 0, 2), ("3~5초", 3, 5), ("6초 이상", 6, 10 ** 9)]


def read(path):
    return json.loads((HERE / path).read_text())


def tl_items(tag):
    """고친 질문 항목에 위치, 정답 여부, Jev 한 장·4장 확률을 붙인다."""
    items = read(f"data/tl_{tag}_items.json")["items"]
    frames = read(f"data/{tag}_frames.json")
    table = {}
    for r in read("results/tl_preds.json")["records"]:
        if r["tag"] == f"tl_{tag}" and not r["error"]:
            for item_id, p in r["nouls"].items():
                table[(item_id, r["frames"], r["t"])] = p
    for it in items:
        it["ts"] = frames[it["video"]]["centers"]
        it["y"] = [any(s <= t <= e for s, e in it["windows"]) for t in it["ts"]]
        it["length"] = it["windows"][0][1] - it["windows"][0][0]
        for cond in ("one", "win4"):
            it[cond] = [table[(it["id"], cond, t)] for t in it["ts"]]
    return items


def sheet_scores(items, tag, index_file, preds_file):
    """항목마다 시트가 덮는 위치 번호와 그 위치의 점수(고르기 = 있는가 × 칸 확률, 칸마다 = 칸의 예 확률)."""
    sheets = [s for s in read(index_file)["sheets"] if s["tag"] == tag]
    data = read(preds_file)
    assert data["meta"]["errors"] == 0, f"{preds_file}: 오류가 남은 요청이 있다"
    recs = {(r["tag"], r["item"], r["sheet"], r["kind"]): r for r in data["records"]}
    out = {}
    for it in items:
        pos, pick, tiles = [], [], []
        for sh in sorted((s for s in sheets if s["video"] == it["video"]), key=lambda s: s["sheet"]):
            a, b = recs[(tag, it["id"], sh["sheet"], "pick")], recs[(tag, it["id"], sh["sheet"], "tiles")]
            for t in sh["tiles"]:
                assert abs(t["t"] - it["ts"][t["pos"]]) < 1e-6, f"{it['id']}: 시각이 다르다"
                pos.append(t["pos"])
                pick.append(a["nouls"]["shown"] * a["choice"]["probabilities"][t["id"]])
                tiles.append(b["nouls"][t["id"]])
        out[it["id"]] = {"pos": pos, "pick": pick, "tiles": tiles}
    return out


def compare(title, items, tag, variants, has_win4=True):
    """variants: {이름: (시트 목록 파일, 결과 파일)}. 첫 번째가 기준(한 장)이고, 모든 시트가 덮는 위치는 같아야 한다."""
    scores = {name: sheet_scores(items, tag, *files) for name, files in variants.items()}
    names = list(variants)
    base = names[0]
    videos = [it["video"] for it in items]
    rows = {}
    for it in items:
        pos = scores[base][it["id"]]["pos"]
        y = [it["y"][k] for k in pos]
        series = {"Jev 한 장": [it["one"][k] for k in pos]}
        if has_win4 and it.get("win4"):
            series["Jev 4장"] = [it["win4"][k] for k in pos]
        for name in names:
            assert scores[name][it["id"]]["pos"] == pos, f"{it['id']}: 시트마다 위치가 다르다"
            series[f"시트 {name}, 칸마다"] = scores[name][it["id"]]["tiles"]
            series[f"시트 {name}, 고르기"] = scores[name][it["id"]]["pick"]
        for name, ps in series.items():
            rows.setdefault(name, []).append({"auc": auc(ps, y), "hit": y[max(range(len(ps)), key=ps.__getitem__)]})
    npos = sum(len(scores[base][it["id"]]["pos"]) for it in items)
    print(f"\n===== {title}: 질문 {len(items)}개, 영상 {len(set(videos))}개, 위치 {npos}개 =====")
    order = ["Jev 한 장", "Jev 4장"] + [f"시트 {n}, 칸마다" for n in names] + [f"시트 {n}, 고르기" for n in names]
    res = {}
    for name in [n for n in order if n in rows]:
        vals = [r["auc"] for r in rows[name] if r["auc"] is not None]
        res[name] = {"auc": round(sum(vals) / len(vals), 3), "auc_items": len(vals),
                     "peak_hit": sum(r["hit"] for r in rows[name]), "items": len(items)}
        print(f"  {name:20s} AUC {res[name]['auc']:.3f}   최고점이 정답 안 {res[name]['peak_hit']}/{len(items)}")
    pairs = []
    for kind in ("칸마다", "고르기"):
        pairs += [(f"시트 {n}, {kind}", f"시트 {base}, {kind}") for n in names[1:]]
        pairs += [(f"시트 {names[k]}, {kind}", f"시트 {names[j]}, {kind}") for j in range(1, len(names)) for k in range(j + 1, len(names))]
    pairs += [(f"시트 {n}, 칸마다", "Jev 한 장") for n in names[1:]]
    pairs += [("Jev 4장", f"시트 {n}, 칸마다") for n in names[1:] + [base]]
    diffs = {}
    print("  차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
    for x, y_ in pairs:
        if x in rows and y_ in rows:
            m, lo, hi, n, g = boot([r["auc"] for r in rows[x]], [r["auc"] for r in rows[y_]], videos)
            diffs[f"{x} − {y_}"] = [m, lo, hi, n]
            print(f"    {x} − {y_}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n}개)")
    by_length = {}
    if all("length" in it for it in items):
        print(f"  구간 길이별 ({base} 시트와의 차이)")
        for label, lo_, hi_ in BINS:
            keep = [k for k, it in enumerate(items) if lo_ <= it["length"] <= hi_]
            cell = {}
            for name in names[1:]:
                for kind in ("칸마다", "고르기"):
                    xs = [rows[f"시트 {name}, {kind}"][k]["auc"] for k in keep]
                    ys = [rows[f"시트 {base}, {kind}"][k]["auc"] for k in keep]
                    use = [(p, q, videos[k]) for p, q, k in zip(xs, ys, keep) if p is not None and q is not None]
                    if len({u[2] for u in use}) >= 3:
                        m, lo, hi, n, g = boot([u[0] for u in use], [u[1] for u in use], [u[2] for u in use])
                        cell[f"{name}, {kind}"] = {"plain": round(sum(u[1] for u in use) / len(use), 3),
                                                    "other": round(sum(u[0] for u in use) / len(use), 3),
                                                    "diff": [m, lo, hi], "items": n}
            by_length[label] = cell
            for key, c in cell.items():
                print(f"    {label:8s} {key}: {c['plain']:.3f} → {c['other']:.3f} ({c['diff'][0]:+.3f} [{c['diff'][1]:+.3f}, {c['diff'][2]:+.3f}], {c['items']}개)")
    return {"title": title, "items": len(items), "positions": npos, "rows": res, "diff": diffs, "by_length": by_length}


def full_size(items, tag):
    """겹친 그림을 원본 크기 한 장으로 물었을 때 (overlay_run.py). 칸이 작아서인지 그림 자체의 한계인지 가른다."""
    data = read("results/tl_overlay_preds.json")
    assert data["meta"]["errors"] == 0
    table = {}
    for r in data["records"]:
        if r["tag"] == f"tl_{tag}":
            for item_id, p in r["nouls"].items():
                table[(item_id, r["kind"], r["t"])] = p
    videos = [it["video"] for it in items]
    series = {"Jev 한 장": [it["one"] for it in items], "Jev 4장": [it["win4"] for it in items],
              "겹친 그림 한 장": [[table[(it["id"], "avg", t)] for t in it["ts"]] for it in items],
              "색 겹친 그림 한 장": [[table[(it["id"], "tint", t)] for t in it["ts"]] for it in items]}
    aucs = {name: [auc(ps, it["y"]) for ps, it in zip(rows, items)] for name, rows in series.items()}
    print(f"\n===== Charades, 1초 간격, 고친 질문: 겹친 그림을 원본 크기 한 장으로 (질문 {len(items)}개) =====")
    res = {}
    for name, rows in series.items():
        vals = [a for a in aucs[name] if a is not None]
        hit = sum(it["y"][max(range(len(ps)), key=ps.__getitem__)] for ps, it in zip(rows, items))
        res[name] = {"auc": round(sum(vals) / len(vals), 3), "peak_hit": hit, "items": len(items)}
        print(f"  {name:16s} AUC {res[name]['auc']:.3f}   최고점이 정답 안 {hit}/{len(items)}")
    diffs = {}
    print("  차이와 95% 구간 (항목 AUC, 영상 단위로 다시 뽑음)")
    for a, b in (("겹친 그림 한 장", "Jev 한 장"), ("색 겹친 그림 한 장", "Jev 한 장"), ("색 겹친 그림 한 장", "겹친 그림 한 장"),
                 ("Jev 4장", "겹친 그림 한 장"), ("Jev 4장", "색 겹친 그림 한 장")):
        m, lo, hi, n, g = boot(aucs[a], aucs[b], videos)
        diffs[f"{a} − {b}"] = [m, lo, hi, n]
        print(f"    {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n}개)")
    by_length = {}
    print("  구간 길이별 AUC 평균")
    for label, lo_, hi_ in BINS:
        keep = [k for k, it in enumerate(items) if lo_ <= it["length"] <= hi_]
        row = {}
        for name in series:
            vals = [aucs[name][k] for k in keep if aucs[name][k] is not None]
            row[name] = round(sum(vals) / len(vals), 3) if vals else None
        by_length[label] = {"items": len(keep), **row}
        print(f"    {label:8s}({len(keep):2d}개) " + ", ".join(f"{n} {v:.3f}" for n, v in row.items() if v is not None))
    tokens = {}
    for r in data["records"]:
        tokens.setdefault(r["kind"], []).append(r["input_tokens"])
    res_tokens = {k: round(sum(v) / len(v)) for k, v in tokens.items()}
    print(f"  요청 하나의 입력 토큰: 겹친 그림 {res_tokens['avg']:,}, 색 겹친 그림 {res_tokens['tint']:,} (영상의 질문을 한 요청에 묶음)")
    return {"rows": res, "diff": diffs, "by_length": by_length, "tokens_per_request": res_tokens, "run": data["meta"]}


def main():
    out = {}
    sta, qv = tl_items("sta"), tl_items("qv")
    def files(prefix, variant):
        return (f"data/sheets60{variant}.json", f"results/{prefix}sheet60{variant}_preds.json")

    def three(prefix, stride=""):
        return {"한 장": files(prefix, stride), "겹침": files(prefix, f"{stride}_blend"), "색 겹침": files(prefix, f"{stride}_tint")}

    out["sta_1s"] = compare("Charades, 1초 간격, 고친 질문", sta, "sta", three("tl_"))
    out["sta_2s"] = compare("Charades, 2초 간격, 고친 질문", sta, "sta", three("tl_", "_s2"))
    out["qv_refined"] = compare("QVHighlights 본 영상, 2초 간격, 고친 질문", qv, "qv", three("tl_"))
    out["qv_original"] = compare("QVHighlights 본 영상, 2초 간격, 원래 질문", E.bundle("qv"), "qv", three(""))
    out["sta_full_size"] = full_size(sta, "sta")
    tokens = {}
    for name, f in (("한 장 시트(1초)", "tl_sheet60_preds.json"), ("겹친 시트(1초)", "tl_sheet60_blend_preds.json"),
                    ("색 겹친 시트(1초)", "tl_sheet60_tint_preds.json"), ("한 장 시트(2초)", "tl_sheet60_s2_preds.json"),
                    ("겹친 시트(2초)", "tl_sheet60_s2_blend_preds.json"), ("색 겹친 시트(2초)", "tl_sheet60_s2_tint_preds.json")):
        recs = [r for r in read(f"results/{f}")["records"] if r["tag"] == "sta"]
        tokens[name] = round(sum(r["input_tokens"] for r in recs) / len(recs))
    print("\nCharades 시트 요청 하나의 입력 토큰: " + ", ".join(f"{k} {v:,}" for k, v in tokens.items()))
    out["tokens_per_request_sta"] = tokens
    (HERE / "results" / "blend_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
