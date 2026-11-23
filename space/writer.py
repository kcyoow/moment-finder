"""중간 LLM이 받을 글과 답 검사. GPU·모델 없이 돌아가는 부분만 모았다(app.py가 쓰고, 이 Mac에서 따로 시험한다).

pipeline/mf_writer.py와 같은 안내서(guide_for_writer.md, writer_brief.md는 그 사본)와 같은 규칙(check_questions)이다.
다른 점: 검색 도구가 없고(그렇게 알린다), 사이트처럼 viewer_request 칸을 더하고, GPU 시간 때문에 한국어 설명 칸을 짧게 쓰라고 덧붙인다.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
# pipeline/mf_writer.py의 SITE_EXTRA와 같은 글
SITE_EXTRA = ("\n\n## 덧붙임 (웹사이트용)\n\nJSON에 `viewer_request` 칸을 하나 더한다: 사용자의 요청을 영어 한 구절로 옮긴 것이다 "
              "(예: `the moments where the robot puts a candy into the cup`). 뜻을 넓히거나 좁히지 않는다. ASCII 글자만 쓴다.")
SPACE_EXTRA = ("\n\n## 덧붙임 (Space용)\n\nGPU 시간이 짧다. `understanding`, `kind`, `limits`는 각각 한 문장으로 짧게 쓴다. "
               "검색 도구는 없으므로 `looked_up`에는 '검색하지 않음'이라고 쓴다.")
FIX_MESSAGE = "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- "


def build_prompt(request, facts=()):
    """mf_writer.write()가 만드는 글과 같은 순서: 안내서 → 할 일과 형식 → 이번 요청과 사실 → 검색 도구 없음."""
    guide = (HERE / "guide_for_writer.md").read_text(encoding="utf-8")
    brief = (HERE / "writer_brief.md").read_text(encoding="utf-8") + SITE_EXTRA + SPACE_EXTRA
    fact_lines = "\n".join(f"- {f}" for f in facts) or "- (없음)"
    return (f"# 안내서 (판정 모델에 보낼 요청을 짜는 법)\n\n{guide}\n\n---\n\n{brief}\n\n---\n\n# 이번 일\n\n"
            f"## 사용자의 요청 (그대로)\n\n{request}\n\n## 영상에 대한 사실\n\n{fact_lines}\n\n"
            f"## 검색 도구\n\n주어지지 않았다. 검색하지 않는다.\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")


def parse_json(text):
    """mf_writer.parse_json과 같다: ```json 울타리가 있으면 그 안, 없으면 처음 { 부터 마지막 } 까지."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def check_questions(qs, lo=3, hi=5):
    """mf_writer.check_questions와 같은 규칙(질문 3~5개, noul·score, `frames`를 가리킴, ASCII, 300자 이하)."""
    errs = []
    if not isinstance(qs, dict) or not lo <= len(qs) <= hi:
        return [f"questions must be an object with {lo} to {hi} questions (got {len(qs) if isinstance(qs, dict) else type(qs).__name__})"]
    for key, q in qs.items():
        if not KEY_RE.match(key) or key == "worth":
            errs.append(f"{key}: key must be lowercase letters, digits or underscores (max 40), and not 'worth'")
        if not isinstance(q, dict) or q.get("type") not in ("noul", "score"):
            errs.append(f"{key}: type must be noul or score")
            continue
        ins, c = q.get("instructions", ""), q.get("criteria")
        texts = [ins]
        if not isinstance(ins, str) or not ins.strip() or "`frames`" not in ins:
            errs.append(f"{key}: instructions is empty or does not point at `frames`")
        for bad in ("`viewer_request`", "`game`", "`about`"):
            if isinstance(ins, str) and bad in ins:
                errs.append(f"{key}: instructions must not point at {bad}")
        if q["type"] == "noul":
            if not isinstance(c, dict) or set(c) != {"true", "false"}:
                errs.append(f"{key}: a noul question needs criteria with exactly 'true' and 'false'")
            else:
                texts += list(c.values())
        elif not isinstance(c, list) or not 3 <= len(c) <= 5:
            errs.append(f"{key}: a score question needs criteria as a list of 3 to 5 level descriptions (lowest first)")
        else:
            texts += c
        for t in texts:
            if not isinstance(t, str) or not t.strip() or len(t) > 300 or any(ord(ch) > 126 or ord(ch) < 32 for ch in t):
                errs.append(f"{key}: a text is empty, longer than 300 characters, or has non-ASCII characters: {str(t)[:60]!r}")
                break
    return errs


# 영어 칸에 섞이기 쉬운 문장부호를 같은 뜻의 ASCII로 바꾼다(뜻은 그대로). 다른 비ASCII 글자는 그대로 두어 검사에 걸리게 한다
ASCII_MAP = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...",
             "\u00a0": " ", "\u00d7": "x", "\u2212": "-", "\u2192": "->", "\u2264": "<=", "\u2265": ">="}


FRAMES_RE = re.compile(r"(?<!`)\b[Ff]rames\b(?!`)")


def normalize_ascii(data):
    """questions의 영어 글과 viewer_request에서 ASCII_MAP의 글자만 바꾼다. instructions에 `frames` 표기가 없고 frames라는 낱말만 있으면
    처음 하나를 `frames`로 감싼다(2026-11-18 Space 시험: 내용은 같은데 이것만 빠져 형식 다시 쓰기에 GPU 70초가 더 들었다). 바꾼 것 목록을 돌려준다."""
    changed = set()

    def fix(t):
        if not isinstance(t, str):
            return t
        for a, b in ASCII_MAP.items():
            if a in t:
                changed.add(a)
                t = t.replace(a, b)
        return t
    if isinstance(data.get("viewer_request"), str):
        data["viewer_request"] = fix(data["viewer_request"])
    qs = data.get("questions")
    if isinstance(qs, dict):
        for q in qs.values():
            if not isinstance(q, dict):
                continue
            q["instructions"] = fix(q.get("instructions"))
            ins = q["instructions"]
            if isinstance(ins, str) and "`frames`" not in ins and FRAMES_RE.search(ins):
                q["instructions"] = FRAMES_RE.sub("`frames`", ins, count=1)
                changed.add("frames->`frames`")
            c = q.get("criteria")
            if isinstance(c, dict):
                q["criteria"] = {k: fix(v) for k, v in c.items()}
            elif isinstance(c, list):
                q["criteria"] = [fix(v) for v in c]
    return sorted(changed)


def check_reply(text):
    """답 글 하나를 읽어 (data, 형식 오류 목록)을 돌려준다. mf_writer.write()의 검사와 같다(viewer_request 칸 포함).
    검사 전에 영어 칸의 굽은 따옴표·긴 대시 같은 문장부호만 ASCII로 바꾼다(바꾼 글자는 data["_ascii_fixed"])."""
    try:
        data = parse_json(text)
    except Exception as e:  # noqa: BLE001
        return None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
    if not isinstance(data, dict):
        return None, ["the reply is not one valid JSON object: not an object"]
    data["_ascii_fixed"] = normalize_ascii(data)
    errs = check_questions(data.get("questions"))
    vr = data.get("viewer_request")
    if not isinstance(vr, str) or not vr.strip() or len(vr) > 300 or any(ord(ch) > 126 or ord(ch) < 32 for ch in vr):
        errs.append("viewer_request must be one short ASCII English phrase (max 300 characters)")
    return data, errs
