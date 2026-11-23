"""후보를 뽑아 원본 프레임으로 확인하는 2단계 흐름을, 이미 받아 둔 답만으로 흉내 낸다. 새 요청은 없다. 표준 라이브러리만 쓴다.

자료: Charades 12개 영상, TimeLens-Bench의 고친 질문 43개(동작). 영상마다 60칸 시트 한 장.
쓰는 답 (모두 저장돼 있음)
- 시트 칸마다(Noul), 시트 고르기(있는가 + Choice): results/tl_sheet60_preds.json
- 위치마다 원본 크기 한 장 / 4장을 함께: results/tl_preds.json (1초 위치마다 있음)

흉내 내는 흐름
- 시트만            : 칸마다의 가장 높은 칸 / 고르기의 가장 높은 칸
- jevreel식 확인    : 고르기의 가장 높은 칸과 둘째 칸(몫 0.15 이상)의 둘레를 원본 한 장씩 확인, 가장 높은 프레임을 답으로
                      실제 jevreel은 앞뒤 1.5초를 0.5초 간격 7장으로 본다. 여기서는 받아 둔 답이 1초 위치뿐이라
                      앞뒤 1초의 3장으로만 흉내 낸다(정확도는 어림, 토큰은 7장으로 셈)
- 칸 후보 + 여러 장 : 칸마다 확률이 높은 순으로 k곳(이웃 칸은 하나로 침)을 후보로, 후보의 앞뒤 r초 위치를 원본 4장으로 확인,
                      확인 확률이 가장 높은 위치를 답으로
- 4장 전부          : 모든 위치를 원본 4장으로 (가장 비싼 기준)

보는 것 (계산 전에 정함): 가리킨 한 점이 정답 안인 수, 움직임을 봐야 하는 15개 중 맞힌 수, 확인 범위에 정답이 들어온 수(상한),
        질문 하나의 입력 토큰(시트 1건 + 확인 요청들. 확인 요청의 토큰은 받아 둔 요청의 실제 값이고 영상의 질문 3.7개가 함께 든 값이라 조금 크게 잡힌다)

실행: python3 experiments/frame_limit/scripts/pipeline/pipeline_sim.py
결과: 표 출력, results/pipeline_sim.json
"""
import sys
from pathlib import Path

sys.path[:0] = [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # scripts/의 다른 주제 폴더에서 불러 쓴다

import json

import siglip_eval as E
from blend_eval import read, tl_items

HERE = E.HERE
ALSO_MIN = 0.15


def top(ps):
    return max(range(len(ps)), key=ps.__getitem__)


def around(p, r, n):
    return [q for q in range(p - r, p + r + 1) if 0 <= q < n]


def candidates(ps, k):
    """확률이 높은 순으로 k곳. 이미 고른 칸의 바로 옆 칸은 건너뛴다."""
    chosen = []
    for p in sorted(range(len(ps)), key=lambda q: -ps[q]):
        if all(abs(p - c) > 1 for c in chosen):
            chosen.append(p)
        if len(chosen) == k:
            break
    return chosen


def load():
    """저장된 답을 읽는다. 돌려주는 값: 항목, 칸마다 확률, 고르기 몫, 시트 요청의 토큰, 원본 프레임 요청의 토큰."""
    items = [it for it in tl_items("sta") if any(it["y"]) and not all(it["y"])]
    sheets = {s["video"]: s for s in read("data/sheets60.json")["sheets"] if s["tag"] == "sta"}
    recs = {(r["item"], r["kind"]): r for r in read("results/tl_sheet60_preds.json")["records"] if r["tag"] == "sta"}
    frame_tokens = {}
    for r in read("results/tl_preds.json")["records"]:
        if r["tag"] == "tl_sta" and not r["error"]:
            frame_tokens.setdefault((r["video"], r["frames"]), []).append(r["input_tokens"])
    frame_tokens = {k: sum(v) / len(v) for k, v in frame_tokens.items()}
    tile_p, pick_share, sheet_tok = {}, {}, {}
    for it in items:
        sh = sheets[it["video"]]
        a, b = recs[(it["id"], "pick")], recs[(it["id"], "tiles")]
        tile_p[it["id"]] = [b["nouls"][t["id"]] for t in sh["tiles"]]
        pick_share[it["id"]] = [a["choice"]["probabilities"][t["id"]] for t in sh["tiles"]]
        sheet_tok[it["id"]] = {"pick": a["input_tokens"], "tiles": b["input_tokens"]}
        assert [t["pos"] for t in sh["tiles"]] == list(range(len(it["ts"])))
    return items, tile_p, pick_share, sheet_tok, frame_tokens


def two_stage(data, k, r, verify="win4"):
    """칸 후보 k곳의 앞뒤 r초 위치를 원본 프레임(verify 조건)으로 확인했을 때. 돌려주는 값: 가리킨 위치, 토큰, 범위에 정답이 있었나."""
    items, tile_p, _, sheet_tok, frame_tokens = data
    picks, toks, cov = [], [], []
    for it in items:
        look = sorted({q for p in candidates(tile_p[it["id"]], k) for q in around(p, r, len(it["ts"]))})
        picks.append(max(look, key=lambda q: it[verify][q]))
        cov.append(any(it["y"][q] for q in look))
        toks.append(sheet_tok[it["id"]]["tiles"] + len(look) * frame_tokens[(it["video"], verify)])
    return picks, toks, cov


def main():
    data = load()
    items, tile_p, pick_share, sheet_tok, frame_tokens = data
    motion = {it["id"] for it in items if not it["y"][top(it["one"])] and it["y"][top(it["win4"])]}
    rows = {}

    def add(name, picks, tokens, covered=None):
        """picks: 항목마다 가리킨 위치 번호. tokens: 항목마다 든 입력 토큰."""
        hit = [bool(it["y"][p]) for it, p in zip(items, picks)]
        rows[name] = {"hit": sum(hit), "motion_hit": sum(h for h, it in zip(hit, items) if it["id"] in motion),
                      "covered": None if covered is None else sum(covered), "tokens": round(sum(tokens) / len(tokens))}

    add("시트만, 고르기 (jevreel 1단계)", [top(pick_share[it["id"]]) for it in items], [sheet_tok[it["id"]]["pick"] for it in items])
    add("시트만, 칸마다", [top(tile_p[it["id"]]) for it in items], [sheet_tok[it["id"]]["tiles"] for it in items])

    # jevreel식: 가장 높은 칸 + 둘째 칸(몫 0.15 이상), 둘레를 원본 한 장씩
    picks, toks, cov = [], [], []
    for it in items:
        share = pick_share[it["id"]]
        order = sorted(range(len(share)), key=lambda k: -share[k])
        spots = [order[0]] + ([order[1]] if share[order[1]] >= ALSO_MIN else [])
        look = sorted({q for p in spots for q in around(p, 1, len(share))})
        picks.append(max(look, key=lambda q: it["one"][q]))
        cov.append(any(it["y"][q] for q in look))
        toks.append(sheet_tok[it["id"]]["pick"] + len(spots) * 7 * frame_tokens[(it["video"], "one")])
    add("jevreel식: 고른 칸 둘레를 한 장씩 확인 (어림)", picks, toks, cov)

    for verify, label in (("one", "한 장씩"), ("win4", "4장을 함께")):
        for k in (1, 2, 3, 4, 6):
            for r in (0, 1):
                picks, toks, cov = two_stage(data, k, r, verify)
                add(f"칸 후보 {k}곳, 앞뒤 {r}초를 {label} 확인", picks, toks, cov)
    add("모든 위치를 한 장씩 (시트 없음)", [top(it["one"]) for it in items], [len(it["ts"]) * frame_tokens[(it["video"], "one")] for it in items])
    add("모든 위치를 4장씩 (시트 없음)", [top(it["win4"]) for it in items], [len(it["ts"]) * frame_tokens[(it["video"], "win4")] for it in items])

    print(f"질문 {len(items)}개, 그중 움직임을 봐야 하는 것 {len(motion)}개. 영상 길이 22~38초(시트 한 장)")
    print(f"{'흐름':40s}{'맞힘':>8s}{'움직임':>8s}{'범위에 정답':>9s}{'질문당 토큰':>10s}{'$/질문 1,000개':>13s}")
    for name, r in rows.items():
        cov = "" if r["covered"] is None else f"{r['covered']}"
        print(f"{name:40s}{r['hit']:>5d}/{len(items)}{r['motion_hit']:>5d}/{len(motion)}{cov:>9s}{r['tokens']:>12,d}{r['tokens'] * 0.042 / 1e3:>12.2f}")
    (HERE / "results" / "pipeline_sim.json").write_text(json.dumps({"items": len(items), "motion_items": len(motion), "rows": rows},
                                                                   ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
