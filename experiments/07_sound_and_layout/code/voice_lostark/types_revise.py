"""2차: 1차 제안(types/proposal.json)의 화면 질문을 고친다. 종류와 목소리 질문은 그대로 둔다(1차와 견주려고).

Gemini에게 주는 것: 1차 종류와 화면 질문, 그리고 종류마다 판정 모델이 45분 가운데 몇 %의 초에 "예"라고 답했는지(정답이 아니라 넓이만).
1차 다시 확인에서 Gemini가 본 것(장면 설명)은 주지 않는다. 그것까지 주면 이 영상에 맞춰 질문을 쓰게 되어 2차를 따로 잴 수 없다.
더하는 것: 종류마다 다시 확인 단계에 줄 한국어 정의(verify_ko). 소유자의 규칙 "실패 화면·재시작 투표도 전멸로 친다"를 넣는다.
쓰는 것: types/proposal_r2.json. 돈이 든다(Gemini 한 번, 약 $0.1).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "pipeline"))
import mf_writer as W  # noqa: E402

TY = ROOT / "types"
P = json.load(open(TY / "proposal.json"))
F = json.load(open(ROOT / "results/h2o_out/types_frames.json"))
recs = [r for r in F["records"] if r and "answers" in r]
rate = {t["key"]: round(100 * sum(r["answers"][t["key"]]["noul"] > 0.5 for r in recs) / len(recs), 1) for t in P["types"]}
first = [{"key": t["key"], "name_ko": t["name_ko"], "why_ko": t["why_ko"], "screen_question": t["screen_question"], "yes_rate_percent": rate[t["key"]]} for t in P["types"]]
BRIEF = """# 할 일: 너무 넓은 화면 질문을 좁히고, 다시 확인 단계에 줄 정의를 쓰기

1차에서 네가 쓴 화면 질문을 판정 모델이 45분짜리 로스트아크 레이드 녹화의 1초마다 물었다. 아래 `yes_rate_percent`는 판정 모델이 "예"(확률 0.5 넘음)라고 답한 초의 비율이다.
하이라이트는 드문 순간이므로, 한 종류가 녹화의 10%를 넘게 "예"라면 그 질문은 평범한 전투까지 잡고 있는 것이다.

종류마다 할 것:
1. `screen_question`을 다시 쓴다. **그 종류에서만 화면에 나타나는 구별 단서**(특정 글자·배너·UI 창·아이콘·키 입력 표시·화면 전체 연출 등)를 짚고,
   평범한 전투(스킬 이펙트, 보스 공격, 장판, 체력 바 변화)는 "아니오"로 가게 쓴다. 단서를 검색으로 확인해도 된다.
   yes_rate가 이미 낮은(3% 이하) 질문은 그대로 두거나 조금만 고친다. 모양과 규칙은 1차와 같다(noul, 영어 ASCII, 각 300자 이하,
   instructions는 "`frames` holds 4 consecutive frames ..."로 시작).
2. `verify_ko`를 쓴다: 다른 모델이 후보 구간의 프레임 8장을 보고 "이 종류의 장면이 담겼나"를 판단할 때 받을 한국어 정의 한두 문장.
   무엇이 이 종류이고 무엇은 아닌지(헷갈리는 경우 포함)를 적는다.
   **사용자의 규칙: "공략에 실패하였습니다" 같은 실패 화면이나 재시작 투표 창이 보이면 전멸로 친다.** 전멸 종류의 정의에 이것을 넣고,
   다른 종류(예: 웃긴 실수)의 정의에는 "평범한 전멸·실패 화면만으로는 이 종류가 아니다"처럼 겹침을 가른다.

종류의 key와 이름은 바꾸지 않는다. 목소리 질문은 다루지 않는다.

## 내놓을 것 (JSON 하나만)

```json
{"types": [{"key": "...", "screen_question": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}}, "verify_ko": "...", "change_note": "<무엇을 왜 바꿨는지 한 문장>"}]}
```
"""
KEY = re.compile(r"^[a-z0-9_]{1,40}$")


def check(d):
    errs = []
    ts = d.get("types") if isinstance(d, dict) else None
    if not isinstance(ts, list) or [t.get("key") for t in ts] != [t["key"] for t in P["types"]]:
        return ["types must keep the same keys in the same order: " + ", ".join(t["key"] for t in P["types"])]
    for t in ts:
        q = t.get("screen_question")
        if not (isinstance(q, dict) and q.get("type") == "noul" and isinstance(q.get("criteria"), dict) and set(q["criteria"]) == {"true", "false"}):
            errs.append(f"{t['key']}: screen_question must be a noul question with criteria true/false"); continue
        for s in [q.get("instructions", "")] + list(q["criteria"].values()):
            if not isinstance(s, str) or not s.strip() or len(s) > 300 or any(ord(c) > 126 or ord(c) < 32 for c in s):
                errs.append(f"{t['key']}: text empty, too long, or non-ASCII"); break
        if not isinstance(t.get("verify_ko"), str) or not t["verify_ko"].strip():
            errs.append(f"{t['key']}: verify_ko missing")
    return errs


guide = (ROOT.parents[1] / "experiments/highlight_auto/writer/guide_for_writer.md").read_text(encoding="utf-8")
user = (f"# 안내서 (판정 모델에 보낼 질문을 짜는 법)\n\n{guide}\n\n---\n\n{BRIEF}\n\n---\n\n# 1차에서 쓴 것과 판정 모델의 '예' 비율\n\n```json\n"
        f"{json.dumps(first, ensure_ascii=False, indent=1)}\n```\n\n## 검색 도구\n\n주어졌다.\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")
(TY / "prompt_r2.txt").write_text(user, encoding="utf-8")
body = {"contents": [{"role": "user", "parts": [{"text": user}]}], "tools": [{"googleSearch": {}}],
        "generationConfig": {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": "HIGH"}}}
usage, errs, data, gq = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}, ["(not called)"], None, []
for a in range(3):
    resp = W.call(body, W.PROJECT, W.MODEL)
    u = resp.get("usageMetadata", {})
    for k, src in (("prompt", "promptTokenCount"), ("output", "candidatesTokenCount"), ("thoughts", "thoughtsTokenCount")):
        usage[k] += u.get(src, 0)
    usage["calls"] += 1
    (TY / f"response_r2_{a + 1}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    text = W.text_of(resp); gq += W.grounding(resp)["queries"]
    try:
        data = W.parse_json(text); errs = check(data)
    except Exception as e:
        data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
    if not errs: break
    body["contents"] += [{"role": "model", "parts": [{"text": text}]},
                         {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
if errs:
    raise SystemExit(f"형식 오류가 남았다: {errs}")
usd = usage["prompt"] / 1e6 * W.PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * W.PRICE_OUT
new = {t["key"]: t for t in data["types"]}
P2 = dict(P, round=2, yes_rate_r1=rate, search_queries_r2=gq, usage_r2={**usage, "usd_estimate": round(usd, 4)},
          types=[dict(t, screen_question=new[t["key"]]["screen_question"], verify_ko=new[t["key"]]["verify_ko"], change_note=new[t["key"]].get("change_note"))
                 for t in P["types"]])
(TY / "proposal_r2.json").write_text(json.dumps(P2, ensure_ascii=False, indent=1), encoding="utf-8")
print("usd", round(usd, 4), "| searches", len(gq))
for t in P2["types"]:
    print(f"\n[{t['key']}] 1차 예 {rate[t['key']]}% | {t['change_note']}\n  화면: {t['screen_question']['instructions'][-160:]}\n  예: {t['screen_question']['criteria']['true'][:200]}\n  정의: {t['verify_ko'][:200]}")
