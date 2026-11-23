"""막연한 "하이라이트" 요청에서, 중간 LLM(Gemini 4 Hydrogen, 검색 켬)이 그 게임에서 하이라이트로 꼽히는 순간의 종류를 찾아
"어떤 하이라이트를 찾을까요?"라는 되묻기와 함께 제안한다. 종류마다 판정 모델(H2O)에 물을 화면 질문과 목소리 질문을 같이 쓴다.
영상·그림·받아쓰기는 주지 않는다(질문 쓰기 단계와 같다). 돈이 든다(Gemini 한 번, 약 $0.1).

쓰는 것: types/proposal.json(제안), types/prompt.txt, types/response_N.json
실행: uv run --no-project python -I experiments/voice_lostark/types_propose.py
"""
import json, re, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parents[1] / "pipeline"
sys.path.insert(0, str(PIPE))
import mf_writer as W  # noqa: E402  Vertex 부르기·답 읽기는 사이트의 질문 쓰기와 같은 함수

OUT = ROOT / "types"; OUT.mkdir(exist_ok=True)
REQUEST = "로스트아크 영상인데 하이라이트 뽑아줘"
FACTS = ["길이 45:34, 1920×1080 60fps, 플레이어 한 사람이 녹화한 것",
         "파티 레이드를 하는 영상으로 보인다(확인하지 않음). 같은 사람의 전날 녹화는 군단장 레이드 카멘이었다",
         "플레이어의 한국어 목소리가 들어 있다(파티원과 대화). 목소리는 받아쓰기·크기·웃음 같은 측정값으로 바뀌어 판정 모델에 글로 들어갈 수 있다",
         "사용자는 하이라이트에 극적인 전멸 장면도 들어간다고 했다"]
BRIEF = """# 할 일: 막연한 하이라이트 요청을 "찾을 순간의 종류" 몇 개로 나누고, 사용자에게 되물을 선택지로 내놓기

사용자는 긴 게임 녹화에서 "하이라이트"를 뽑아 달라고만 했다. 무엇을 하이라이트로 볼지는 사람마다 다르다.
너는 영상을 보지 않는다. 검색 도구로 이 게임(그리고 이 콘텐츠)에서 시청자와 플레이어가 하이라이트로 꼽는 순간이 무엇인지 확인하고,
그 가운데 **이 녹화에서 찾을 만한 종류 5~7개**를 고른다. 사용자는 이 목록을 보고 고르기만 한다. 각 종류마다 대표 장면 하나를 찾아 보여 줄 것이다.

종류마다 쓸 것:
- `key`: 영어 소문자·숫자·밑줄(40자 이하)
- `name_ko`: 사용자에게 보일 짧은 한국어 이름 (예: "전멸 장면")
- `why_ko`: 왜 하이라이트로 꼽히는지 한 문장(검색에서 찾은 근거를 짧게)
- `screen_question`: 판정 모델에 물을 예/아니오 질문 하나. 판정 모델은 **0.5초 간격 연속 프레임 4장(`frames`)만** 본다.
  모양: {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}}. instructions는 "`frames` holds 4 consecutive frames ..."로 시작한다.
  화면에 보이는 것(UI, 글자, 효과, 캐릭터 상태)으로 판단할 수 있게 쓴다. 영어, ASCII 글자만, 각 300자 이하.
- `voice_question`: 플레이어 목소리 기록(`voice`: 받아쓴 한국어 문장, 평소 대비 크기, 웃음·흥분 측정)으로 물을 예/아니오 질문 하나.
  그 종류를 목소리로 알아볼 단서가 없으면 null. 모양은 위와 같고 instructions는 "`voice` summarizes ..."로 시작한다. 받아쓰기는 한국어이므로
  알아볼 한국어 낱말 예를 criteria에 적어도 된다(이때만 한국어 허용).
- `voice_hint`: 이 종류를 찾는 데 목소리가 "essential" / "helpful" / "not_needed" 가운데 무엇이라고 보는지(참고용. 판정 모델이 따로 정한다)

또 `ask_ko`: 사용자에게 보일 되묻기 한 문장 (예: "어떤 하이라이트를 찾을까요? 여러 개 골라도 됩니다")
`looked_up`: 검색으로 확인한 것 한두 문장. 확인하지 못한 것은 확인하지 못했다고 쓴다.

## 내놓을 것 (JSON 하나만)

```json
{"ask_ko": "...", "looked_up": "...", "types": [{"key": "...", "name_ko": "...", "why_ko": "...", "voice_hint": "helpful",
  "screen_question": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}},
  "voice_question": null}]}
```
"""
KEY = re.compile(r"^[a-z0-9_]{1,40}$")


def check(d):
    errs = []
    ts = d.get("types") if isinstance(d, dict) else None
    if not isinstance(ts, list) or not 5 <= len(ts) <= 7:
        return ["types must be a list of 5 to 7 items"]
    for t in ts:
        k = t.get("key", "")
        if not KEY.match(k): errs.append(f"{k}: bad key")
        for f in ("name_ko", "why_ko", "voice_hint"):
            if not isinstance(t.get(f), str) or not t[f].strip(): errs.append(f"{k}: {f} missing")
        for qn, ascii_only in (("screen_question", True), ("voice_question", False)):
            q = t.get(qn)
            if q is None and qn == "voice_question": continue
            if not (isinstance(q, dict) and q.get("type") == "noul" and isinstance(q.get("criteria"), dict) and set(q["criteria"]) == {"true", "false"}):
                errs.append(f"{k}: {qn} must be a noul question with criteria true/false"); continue
            for s in [q.get("instructions", "")] + list(q["criteria"].values()):
                if not isinstance(s, str) or not s.strip() or len(s) > 300 or (ascii_only and any(ord(c) > 126 or ord(c) < 32 for c in s)):
                    errs.append(f"{k}: {qn} text empty, too long, or non-ASCII"); break
    for f in ("ask_ko", "looked_up"):
        if not isinstance(d.get(f), str): errs.append(f"{f} missing")
    return errs


guide = (ROOT.parents[1] / "experiments/highlight_auto/writer/guide_for_writer.md").read_text(encoding="utf-8")
user = (f"# 안내서 (판정 모델에 보낼 질문을 짜는 법)\n\n{guide}\n\n---\n\n{BRIEF}\n\n---\n\n# 이번 일\n\n## 사용자의 요청 (그대로)\n\n{REQUEST}\n\n"
        "## 영상에 대한 사실\n\n" + "\n".join(f"- {f}" for f in FACTS) + "\n\n## 검색 도구\n\n주어졌다.\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")
(OUT / "prompt.txt").write_text(user, encoding="utf-8")
body = {"contents": [{"role": "user", "parts": [{"text": user}]}], "tools": [{"googleSearch": {}}],
        "generationConfig": {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": "HIGH"}}}
usage, errs, data, attempts = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}, ["(not called)"], None, []
for a in range(3):
    resp = W.call(body, W.PROJECT, W.MODEL)
    u = resp.get("usageMetadata", {})
    for k, src in (("prompt", "promptTokenCount"), ("output", "candidatesTokenCount"), ("thoughts", "thoughtsTokenCount")):
        usage[k] += u.get(src, 0)
    usage["calls"] += 1
    (OUT / f"response_{a + 1}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    text = W.text_of(resp)
    try:
        data = W.parse_json(text); errs = check(data)
    except Exception as e:
        data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
    attempts.append(W.grounding(resp))
    if not errs: break
    body["contents"] += [{"role": "model", "parts": [{"text": text}]},
                         {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
if errs:
    raise SystemExit(f"형식 오류가 남았다: {errs}")
usd = usage["prompt"] / 1e6 * W.PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * W.PRICE_OUT
out = {"model": resp.get("modelVersion", W.MODEL), "written_at": date.today().isoformat(), "request": REQUEST, "facts": FACTS,
       "search_queries": [q for g in attempts for q in g["queries"]], "search_sources": attempts[-1]["sources"],
       "usage": {**usage, "usd_estimate": round(usd, 4)}, **data}
(OUT / "proposal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("ask_ko", "looked_up", "search_queries", "usage")}, ensure_ascii=False, indent=1))
for t in out["types"]:
    print(f"\n[{t['key']}] {t['name_ko']} (목소리: {t['voice_hint']}) — {t['why_ko']}\n  화면: {t['screen_question']['instructions'][:200]}")
    if t.get("voice_question"): print(f"  목소리: {t['voice_question']['instructions'][:200]}")
