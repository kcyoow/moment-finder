"""중간 LLM(Qwen)이 한 번에 쓰는 "계획": 1초마다 물을 질문(writer.py와 같은 규칙) + 다시 확인용 정의(check) + 사용자에게 되물을 말.

GPU·모델 없이 돌아가는 부분만 모았다(engine.py가 쓰고, 이 Mac에서 따로 시험한다).
다시 확인용 정의의 규칙은 pipeline/verify_question_brief.md, 검사는 pipeline/mf_recheck.py의 check_question과 같다.
"""
import json
import re

import writer as WR

KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")

# writer.py의 글 뒤에 붙인다. 질문 수를 줄이고(GPU 시간), 같은 답에 다시 확인용 정의를 함께 쓰게 한다
PLAN_EXTRA = """

## 덧붙임 (대화와 다시 확인)

이번에는 질문을 **3개만** 쓴다(요청이 "A거나 B거나 C" 꼴로 갈래가 넷 이상일 때만 4개). GPU 시간이 짧다.

네 답은 두 군데에 쓰인다.

1. 사용자에게 "이렇게 찾을까요?" 하고 되묻는다. `understanding`, `limits`와 아래 `check`의 `ko` 칸이 그대로 사용자에게 보인다. 사람이 읽기 쉬운 한국어로 쓴다.
2. 네 질문으로 1초마다 훑어 후보 구간을 몇 개 고른 뒤, **같은 판정 모델이 후보마다 그림 8장을 다시 본다**(그 구간의 어느 1초에서 연속 4장, 구간 전체에서 고르게 4장). 훑는 단계는 자주 헛짚고, 비슷해 보이는 다른 장면을 고르기도 한다. 그때 쓸 정의를 `check` 칸에 쓴다.

JSON에 `check` 칸을 하나 더한다.

```json
"check": {
  "target": {"key": "<이름>", "ko": "<한국어로 짧게>", "is": "<그림에 무엇이 보이면 찾는 장면인가>"},
  "confusers": [
    {"key": "<이름>", "ko": "<한국어로 짧게>", "is": "<그림에 무엇이 보이면 이것인가>"}
  ],
  "noul": {"instructions": "<Do these frames show ...? 꼴의 예/아니오 질문>", "criteria": {"true": "<예의 정의>", "false": "<아니오의 정의>"}}
}
```

- `target`: 사용자가 찾는 장면 하나. 요청보다 넓히지도 좁히지도 않는다. 요청이 "성공한 장면"이면 시도만 한 것은 target이 아니다
- `confusers`: **3개 또는 4개.** 그 종류의 영상에서 찾는 장면과 비슷해 보이지만 찾는 장면이 아닌 것. 훑는 단계가 잘못 고를 만한 것을 고른다
- confusers 가운데 **하나는 반드시 "찾는 대상만 빠진 같은 동작"** 으로 쓴다. 그 종류의 영상에서 가장 흔하게 나오는 평범한 장면이다(예: 찾는 것이 "골이 들어가는 장면"이면 "골 없이 공을 몰고 가거나 패스하는 장면", 찾는 것이 "전을 뒤집는 순간"이면 "뒤집지 않고 팬 위에서 익히는 장면"). 이 선택지가 없으면 판정 모델은 평범한 장면을 모두 찾는 장면으로 고른다
- 요청이 "A거나 B거나" 꼴로 갈래가 여럿이면, target의 `is`에 갈래마다 화면에서 보이는 모습을 적고, confusers에는 **갈래마다** 그 갈래의 평범한 닮은 장면을 넣는다
- `is`에는 그것이 **화면에서 어떻게 보이는지**(색, 모양, 자리, 움직임)를 적는다. 이름만 적지 않는다. target과 confuser를 가르는 눈에 보이는 차이가 글에 드러나야 한다
- 판정 모델은 target, confusers, 그리고 우리가 덧붙이는 "어느 것도 아님" 가운데 하나를 고른다. 그래서 선택지들은 **서로 겹치지 않게** 쓴다. "어느 것도 아님"은 쓰지 않는다
- 요청이 막연하면(예: "하이라이트") 그 종류의 영상을 보는 사람이 하이라이트로 여길 장면을 target으로 잡고, 평범하게 이어지는 장면들을 confusers로 잡는다
- 정의는 그림에 보이는 것으로 적는다. 소리, 의도, 그림 밖의 일은 적지 않는다
- `is`, `instructions`, `criteria`는 영어, ASCII 글자만, 한 칸에 300자 이하. `ko`는 한국어 30자 이하. `key`는 영어 소문자·숫자·밑줄, 서로 달라야 하고 `none_of_these`는 쓰지 않는다
- `check` 안의 글에서는 `frames` 표기를 쓰지 않아도 된다("these frames"라고 쓴다)
"""

REVISE = ("사용자가 위 계획을 보고 이렇게 말했다(그대로):\n\n{feedback}\n\n이 말을 반영해 같은 모양의 JSON 전체를 다시 낸다. "
          "사용자가 말하지 않은 부분은 바꾸지 않는다.")


def build_messages(request, facts=(), feedback=(), prev_reply=None):
    """Qwen에게 보낼 대화. 처음에는 글 하나, 사용자가 고쳐 말하면 앞의 답과 그 말을 이어 붙인다."""
    base = WR.build_prompt(request, facts)
    head, tail = base.rsplit("\n\n---\n\n# 이번 일", 1)
    msgs = [{"role": "user", "content": head + PLAN_EXTRA + "\n\n---\n\n# 이번 일" + tail}]
    if feedback and prev_reply:
        msgs += [{"role": "assistant", "content": prev_reply}, {"role": "user", "content": REVISE.format(feedback="\n".join(feedback))}]
    return msgs


def _ascii_ok(t, limit=300):
    return isinstance(t, str) and bool(t.strip()) and len(t) <= limit and not any(ord(ch) > 126 or ord(ch) < 32 for ch in t)


def _fix_ascii(t):
    if isinstance(t, str):
        for a, b in WR.ASCII_MAP.items():
            t = t.replace(a, b)
    return t


def check_block(chk):
    """다시 확인용 정의의 형식 검사(mf_recheck.check_question과 같은 규칙. ko 칸은 없어도 넘긴다)."""
    if not isinstance(chk, dict):
        return ["check must be an object with target, confusers and noul"]
    errs = []
    conf = chk.get("confusers")
    if not isinstance(conf, list) or not 2 <= len(conf) <= 4:
        errs.append("check.confusers must be a list of 2 to 4 items")
    keys = []
    for o in [chk.get("target")] + (conf if isinstance(conf, list) else []):
        if not isinstance(o, dict) or not KEY_RE.match(str(o.get("key", ""))) or o.get("key") == "none_of_these":
            errs.append(f"check: target and each confuser need a key of lowercase letters, digits or underscores (max 40), not 'none_of_these': {str(o)[:60]}")
            continue
        keys.append(o["key"])
        o["is"] = _fix_ascii(o.get("is"))
        if not _ascii_ok(o["is"]):
            errs.append(f"check.{o['key']}: 'is' is empty, longer than 300 characters, or has non-ASCII characters")
    if len(set(keys)) != len(keys):
        errs.append("check: keys of target and confusers must all differ")
    n = chk.get("noul")
    if isinstance(n, dict):
        n["instructions"] = _fix_ascii(n.get("instructions"))
        if isinstance(n.get("criteria"), dict):
            n["criteria"] = {k: _fix_ascii(v) for k, v in n["criteria"].items()}
    c = n.get("criteria") if isinstance(n, dict) else None
    if not isinstance(n, dict) or not _ascii_ok(n.get("instructions")) or not isinstance(c, dict) or set(c) != {"true", "false"} or not all(_ascii_ok(v) for v in c.values()):
        errs.append("check.noul needs ASCII 'instructions' and criteria with exactly 'true' and 'false' (each max 300 characters)")
    return errs


def point_at_frames(text):
    """질문 글에 `frames`라는 말이 아예 없으면 앞에 붙여 준다. 내용은 그대로인데 이것만 빠져 형식 다시 쓰기에 GPU 2분이 더 드는 일을 막는다
    (2026-11-23 Space 시험: 질문 셋 가운데 하나가 "Does the cup ...?" 꼴이었다). 고친 질문의 이름을 돌려준다."""
    try:
        data = WR.parse_json(text)
    except Exception:  # noqa: BLE001
        return text, []
    fixed = []
    qs = data.get("questions") if isinstance(data, dict) else None
    for key, q in (qs.items() if isinstance(qs, dict) else []):
        ins = q.get("instructions") if isinstance(q, dict) else None
        if isinstance(ins, str) and ins.strip() and "`frames`" not in ins and not WR.FRAMES_RE.search(ins):
            q["instructions"] = "Looking at `frames` together: " + ins.strip()
            fixed.append(key)
    return (json.dumps(data, ensure_ascii=False) if fixed else text), fixed


def check_plan(text):
    """Qwen의 답 하나 → (data, 형식 오류 목록). 질문·viewer_request는 writer.check_reply, check 칸은 check_block."""
    text, fixed = point_at_frames(text)
    data, errs = WR.check_reply(text)
    if data is None:
        return None, errs
    data["_frames_added"] = fixed
    return data, errs + check_block(data.get("check"))


def _ko(o):
    return (o.get("ko") or "").strip() or o["key"].replace("_", " ")


def proposal_text(plan):
    """사용자에게 되물을 말(한국어). Qwen이 쓴 칸을 그대로 늘어놓는다."""
    chk = plan["check"]
    lines = [f"이렇게 이해했습니다: {(plan.get('understanding') or '').strip()}", "",
             f"- **찾을 장면:** {_ko(chk['target'])}",
             f"- **헷갈려서 빼려는 장면:** {', '.join(_ko(c) for c in chk['confusers'])}",
             f"- **1초마다 볼 것:** {', '.join(plan['questions'])} (질문 {len(plan['questions'])}개)"]
    if (plan.get("limits") or "").strip():
        lines.append(f"- **놓칠 수 있는 것:** {plan['limits'].strip()}")
    lines += ["", "이대로 찾을까요? 맞으면 **이대로 찾기**를 누르고, 다르면 아래에 고칠 점을 적어 보내 주세요."]
    return "\n".join(lines)


CHOICE_ASK = "Which option best describes what happens in this segment?"
CHOICE_NONE = "None of the other options is visible in these frames."


def recheck_questions(chk):
    """다시 확인 때 묻는 셋: 예/아니오 하나, 고르기 둘(선택지 순서만 반대. H2O는 끝 선택지 쪽으로 기울어서 두 순서를 함께 묻는다)."""
    opts = [(chk["target"]["key"], chk["target"]["is"])] + [(c["key"], c["is"]) for c in chk["confusers"]] + [("none_of_these", CHOICE_NONE)]
    return {"verify": {"type": "noul", "instructions": chk["noul"]["instructions"], "criteria": chk["noul"]["criteria"]},
            "choice_fwd": {"type": "choice", "instructions": CHOICE_ASK, "criteria": dict(opts)},
            "choice_rev": {"type": "choice", "instructions": CHOICE_ASK, "criteria": dict(reversed(opts))}}


def to_plan(data, request, facts, writer_name, feedback=(), raw=""):
    """검사를 통과한 답 → 저장할 계획."""
    return {"writer": writer_name, "request": request, "facts": list(facts), "feedback": list(feedback),
            "understanding": data.get("understanding"), "kind": data.get("kind"), "limits": data.get("limits"),
            "viewer_request": data["viewer_request"].strip(), "questions": data["questions"], "check": data["check"], "raw_reply": raw}


def parse_plan_json(text):
    """붙여 넣은 계획 JSON(시험용: Qwen을 다시 부르지 않고 앞서 쓴 계획을 그대로 쓴다)."""
    p = json.loads(text)
    errs = WR.check_questions(p.get("questions")) + check_block(p.get("check"))
    if errs or not isinstance(p.get("viewer_request"), str):
        raise ValueError(f"계획 JSON의 형식이 맞지 않는다: {errs[:3]}")
    return p
