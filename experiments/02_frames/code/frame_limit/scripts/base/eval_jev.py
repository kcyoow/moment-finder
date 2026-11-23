"""run_jev.py 결과를 채점한다. 표준 라이브러리만 쓴다.

- 예·아니오(Noul): 확률 0.5 이상이면 "yes". sep4는 4장의 확률을 평균한다
- 객관식(Choice): 확률이 가장 높은 보기. sep4는 4장의 확률 분포를 평균한다
표본이 작으므로 비율보다 맞힌 개수를 본다.

실행: python3 experiments/frame_limit/scripts/base/eval_jev.py
결과: 표 출력, results/jev_eval.json
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
CONDITIONS = ["one", "sep4", "joint4", "joint8"]
LABEL = {"one": "한 장", "sep4": "4장 따로", "joint4": "4장 함께", "joint8": "8장 함께"}
TEMPORAL = {"direction", "attribute_change", "speed", "order"}


def predictions(data):
    """(영상, 조건, 질문 id) -> 예측. noul은 평균 확률, choice는 평균 분포."""
    acc = defaultdict(list)
    for r in data["records"]:
        if r["error"]:
            continue
        for qid, a in r["answers"].items():
            acc[(r["video"], r["condition"], qid)].append(a)
    out = {}
    for key, answers in acc.items():
        if "noul" in answers[0]:
            out[key] = sum(a["noul"] for a in answers) / len(answers)
        else:
            opts = answers[0]["probabilities"].keys()
            out[key] = {o: sum(a["probabilities"][o] for a in answers) / len(answers) for o in opts}
    return out


def main():
    data = json.loads((HERE / "results" / "jev_preds.json").read_text())
    failed = [(r["video"], r["condition"], r["frame_index"]) for r in data["records"] if r["error"]]
    if failed:
        print(f"주의: 실패한 요청 {len(failed)}건이 남아 있어 아래 수치가 불완전합니다:", failed)
    pred = predictions(data)
    rows = []  # 질문 하나 × 조건 하나
    for video, items in data["items"].items():
        for it in items:
            for c in CONDITIONS:
                p = pred.get((video, c, it["id"]))
                if p is None:
                    continue
                if it["type"] == "noul":
                    guess, score = ("yes" if p >= 0.5 else "no"), round(p, 3)
                else:
                    guess, score = max(p, key=p.get), {k: round(x, 3) for k, x in p.items()}
                rows.append({"video": video, "qid": it["id"], "type": it["type"], "dim": it["dim"],
                             "question": it["question"], "truth": it["truth"], "condition": c,
                             "guess": guess, "score": score, "correct": guess == it["truth"],
                             "reverse": video.endswith("_reverse")})

    def table(title, subset, groups):
        print(f"\n{title}")
        print(f"{'구분':28s}" + "".join(f"{LABEL[c]:>12s}" for c in CONDITIONS) + f"{'문항':>6s}")
        summary = {}
        for name, keep in groups:
            sel = [r for r in subset if keep(r)]
            n = len({(r["video"], r["qid"]) for r in sel})
            if not n:
                continue
            cells = {c: sum(r["correct"] for r in sel if r["condition"] == c) for c in CONDITIONS}
            print(f"{name:28s}" + "".join(f"{cells[c]:>9d}/{n:<3d}" for c in CONDITIONS) + f"{n:>5d}")
            summary[name] = {"n": n, **cells}
        return summary

    noul = [r for r in rows if r["type"] == "noul"]
    choice = [r for r in rows if r["type"] == "choice"]
    result = {"meta": data["meta"]}
    result["tempcompass"] = table("TempCompass 예·아니오 (맞힌 개수/문항 수)", noul, [
        ("전체", lambda r: True),
        ("시간 정보 질문", lambda r: r["dim"] in TEMPORAL),
        ("  방향", lambda r: r["dim"] == "direction"),
        ("  변화", lambda r: r["dim"] == "attribute_change"),
        ("  속도", lambda r: r["dim"] == "speed"),
        ("동작 종류 질문", lambda r: r["dim"] == "action"),
        ("원본 영상", lambda r: not r["reverse"]),
        ("역재생 영상", lambda r: r["reverse"]),
        ("  역재생 중 방향·변화", lambda r: r["reverse"] and r["dim"] in ("direction", "attribute_change")),
    ])
    yes_share = sum(r["truth"] == "yes" for r in noul if r["condition"] == "one") / max(1, len(noul) // len(CONDITIONS))
    print(f"(참고: 정답이 'yes'인 비율 {yes_share:.2f}. 모두 한쪽으로 찍으면 {max(yes_share, 1 - yes_share):.2f})")
    result["tempcompass"]["yes_share"] = round(yes_share, 3)

    # 영상별: 방향·변화 질문만. 원본과 역재생은 프레임이 같고 순서만 반대다
    print("\nTempCompass 영상별 방향·변화 질문 (맞힌 개수/문항 수)")
    per_video = {}
    for video in data["items"]:
        sel = [r for r in noul if r["video"] == video and r["dim"] in ("direction", "attribute_change")]
        n = len({r["qid"] for r in sel})
        if not n:
            continue
        cells = {c: sum(r["correct"] for r in sel if r["condition"] == c) for c in CONDITIONS}
        print(f"{video[12:]:28s}" + "".join(f"{cells[c]:>9d}/{n:<3d}" for c in CONDITIONS))
        per_video[video] = {"n": n, **cells}
    result["tempcompass"]["per_video_direction_change"] = per_video

    # 같은 질문을 원본과 역재생에 물었을 때 정답이 "yes"인 쪽의 확률이 더 높은가.
    # 0.5 기준으로 자르면 안 보이는 차이를 본다
    by_key = {(r["video"], r["question"], r["condition"]): r for r in noul
              if r["dim"] in ("direction", "attribute_change")}
    paired = {c: [0, 0] for c in CONDITIONS}
    for (video, question, c), r in by_key.items():
        if video.endswith("_reverse"):
            continue
        other = by_key.get((video + "_reverse", question, c))
        if other is None or other["truth"] == r["truth"]:
            continue
        yes, no = (r, other) if r["truth"] == "yes" else (other, r)
        paired[c][1] += 1
        paired[c][0] += yes["score"] > no["score"]

    def auc(sel):
        pos = [r["score"] for r in sel if r["truth"] == "yes"]
        neg = [r["score"] for r in sel if r["truth"] == "no"]
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return round(wins / (len(pos) * len(neg)), 3) if pos and neg else None

    temporal = [r for r in noul if r["dim"] in ("direction", "attribute_change")]
    aucs = {c: auc([r for r in temporal if r["condition"] == c]) for c in CONDITIONS}
    print("\n방향·변화 질문을 확률로 보면")
    print(f"{'구분':28s}" + "".join(f"{LABEL[c]:>12s}" for c in CONDITIONS))
    print(f"{'같은 질문, 원본 대 역재생':28s}" + "".join(f"{paired[c][0]:>9d}/{paired[c][1]:<3d}" for c in CONDITIONS))
    print(f"{'AUC (0.5면 구분 못 함)':28s}" + "".join(f"{aucs[c]:>12.2f}" for c in CONDITIONS))
    result["tempcompass"]["paired_same_question"] = {c: {"higher_for_yes": paired[c][0], "n": paired[c][1]}
                                                     for c in CONDITIONS}
    result["tempcompass"]["auc_direction_change"] = aucs

    print("\nTVBench 문항별 (정답 / 조건별 선택. 맞으면 O)")
    for video, items in data["items"].items():
        for it in items:
            if it["type"] != "choice":
                continue
            marks = []
            for c in CONDITIONS:
                r = next((x for x in choice if x["video"] == video and x["qid"] == it["id"] and x["condition"] == c), None)
                marks.append("없음" if r is None else ("O " if r["correct"] else "X ") + it["options"][r["guess"]][:22])
            print(f"- [{it['dim']}] {it['question'][:70]}\n    정답: {it['options'][it['truth']][:40]} | " + " | ".join(marks))

    tasks = sorted({r["dim"] for r in choice})
    result["tvbench"] = table("TVBench 객관식 (맞힌 개수/문항 수)", choice,
                              [("전체", lambda r: True)] + [(f"  {t}", lambda r, t=t: r["dim"] == t) for t in tasks])
    n_opts = [len(it["options"]) for items in data["items"].values() for it in items if it["type"] == "choice"]
    chance = sum(1 / n for n in n_opts)
    print(f"(참고: 무작위로 찍을 때 기대 정답 수 {chance:.1f}/{len(n_opts)})")
    result["tvbench"]["chance_expected"] = round(chance, 2)

    result["rows"] = rows
    (HERE / "results" / "jev_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    m = data["meta"]
    print(f"\n요청 {m['requests']}건, 오류 {m['errors']}건, 입력 토큰 {m['input_tokens']:,}, 약 ${m['cost_usd']}")


if __name__ == "__main__":
    main()
