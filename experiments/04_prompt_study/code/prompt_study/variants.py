"""시험할 변형. 2026-11-12에 결과를 보기 전에 정했고, 본 뒤에는 고치지 않는다.

변형 하나 = state 모양(layout) + 질문 문장(q) + criteria(없으면 H2O 서버의 기본값 "the statement holds" / "it does not").
{s}에는 질문 문장(예: "A person opens a backpack")이 그대로 들어간다.
어디서 온 변형인지는 why에 적었다. 출처의 문장을 그대로 쓸 수 없어 바꾼 곳도 거기 적었다. 조사 정리는 REPORT.md.
"""
import re

HEAD = "`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). "

# state 모양. frames: 어떤 프레임 4장을 쓰나(win = 그 시각 −0.75, −0.25, +0.25, +0.75초 / wide = 1초 간격 4장), state: (시각, 그림) → state
LAYOUTS = {
    # 지금까지 쓴 모양. H2O가 읽는 글: record: {"frames":[{"seconds":0.25,"frame":"[image 1]"}, …]}
    "ts": {"frames": "win", "state": lambda secs, uris: {"frames": [{"seconds": s, "frame": u} for s, u in zip(secs, uris)]}},
    # 시각을 뺀 것. H2O가 읽는 글: record: {"frames":["[image 1]","[image 2]","[image 3]","[image 4]"]}
    "bare": {"frames": "win", "state": lambda secs, uris: {"frames": list(uris)}},
    # 1초 간격 4장(3초 폭). state의 꼴은 ts와 같다
    "wide": {"frames": "wide", "state": lambda secs, uris: {"frames": [{"seconds": s, "frame": u} for s, u in zip(secs, uris)]}},
}

NEAR = {"true": "The frames show the action itself happening.",
        "false": "The action itself is not happening in these frames, even if the person or the objects involved are visible."}

BASE = "base"
VARIANTS = {
    "base": {"layout": "ts", "criteria": None, "q": HEAD + "Do these frames show the following: {s}?",
             "why": "지금까지 쓴 문장(open_model의 JEV_Q['win4']). 기준"},
    # ---- 질문 문장만 바꾼 것
    "short": {"layout": "ts", "criteria": None, "q": "Does the video clip in `frames` show {s}?",
              "why": "가장 짧은 꼴. FV-Action(arXiv 2608.08315)의 고정 문장 'Does this clip show {q}?'와 VQAScore의 꼴. 판정 모델이라 'Answer Yes or No'는 뺐고, "
                     "state의 이름을 가리키게 'in `frames`'를 넣었다. 두 논문 모두 가벼운 문장 손질은 차이가 없다고 보고"},
    "statement": {"layout": "ts", "criteria": None, "q": "The 4 consecutive video frames in `frames` show the following: {s}.",
                  "why": "질문 대신 판정할 문장으로. TypeSafe 문서('or as a statement for the model to judge')와 H2O 카드의 예시, H2O 기본 선택지 'the statement holds'에 맞춘 꼴"},
    "strict_q": {"layout": "ts", "criteria": None, "q": HEAD + "Is the following clearly happening in these frames: {s}?",
                 "why": "기준 문장에서 물음만 엄하게('clearly happening'). REZE의 'CLEARLY occurring'에서 낱말만 가져온 최소 변경"},
    "strict_reze": {"layout": "ts", "criteria": {"true": "Yes", "false": "No"},
                    "q": "You are a strict video action verifier. Your default answer is 'No'. Answer 'Yes' ONLY if you are highly confident the described action is "
                         "clearly and actively occurring in the frames. If there is any doubt, answer 'No'.\nAction: {s}\nIs this action CLEARLY occurring in these 4 frames?",
                    "why": "REZE(arXiv 2608.04480) 부록 A.3의 문장. 원문은 system과 user로 나뉘는데 H2O는 system이 고정이라 둘을 instructions에 이어 붙였다. "
                           "뺀 것: system의 마지막 문장 'Output: a single word …'(H2O는 글자 하나로 답한다)와 user 끝의 'Answer:'(서버가 붙인다). "
                           "선택지가 Yes/No로 읽히게 criteria를 Yes/No로 두었다. 원문 수치: QVHighlights mAP 43.56(중립 문장 41.35, 뜻만 같은 다른 문장 39.19)"},
    "excl_q": {"layout": "ts", "criteria": None,
               "q": HEAD + "Do these frames show the following: {s}? Seeing the person or the objects involved without the action itself does not count.",
               "why": "비슷해 보이지만 아닌 경우를 질문에 적기. OpenAI Decisions 안내('Ignore shadows …' 예시)와 OpenRouter 안내의 권고. "
                      "문장은 dev의 헛짚음 유형(물건만 보여도 잡음)을 보고 내가 썼다"},
    # ---- criteria만 바꾼 것 (질문은 기준 문장)
    "crit_yn": {"layout": "ts", "criteria": {"true": "yes", "false": "no"}, "q": HEAD + "Do these frames show the following: {s}?",
                "why": "H2O 카드의 예시가 쓰는 criteria. strict_reze가 criteria도 바꾸므로 그 효과를 따로 보려는 것"},
    "crit_desc": {"layout": "ts", "criteria": {"true": "The frames show this happening.", "false": "The frames do not show this happening."},
                  "q": HEAD + "Do these frames show the following: {s}?",
                  "why": "예/아니오의 뜻을 풀어 적기. TypeSafe: criteria는 질문의 연장으로 쓰고, 넣은 것과 뺀 것을 견줘 볼 것"},
    "crit_near": {"layout": "ts", "criteria": NEAR, "q": HEAD + "Do these frames show the following: {s}?",
                  "why": "아니오 쪽에 헷갈리는 경우를 적기(TypeSafe: 경계가 미묘하면 criteria에, 내 가이드 초안의 권고). excl_q와 같은 내용을 선택지 쪽에 둔 것"},
    # ---- 둘을 겹친 것
    "strict_near": {"layout": "ts", "criteria": NEAR, "q": HEAD + "Is the following clearly happening in these frames: {s}?",
                    "why": "strict_q + crit_near"},
    # ---- state를 바꾼 것 (요청을 따로 보낸다)
    "no_ts": {"layout": "bare", "criteria": None, "q": HEAD + "Do these frames show the following: {s}?",
              "why": "프레임마다 적던 시각을 뺌. TypeSafe: 필요 없는 것은 state에서 빼고, 숫자 표현에 약하다. "
                     "FV-Action은 시각을 적으면 나빠졌다고 보고했지만 그것은 시각을 글로 생성하게 하는 쪽의 결과"},
    "wide": {"layout": "wide", "criteria": None,
             "q": "`frames` holds 4 consecutive frames from a video, 1 second apart, in time order (earliest first). Do these frames show the following: {s}?",
             "why": "문장이 아니라 프레임 간격을 넓힌 것(1.5초 폭 → 3초 폭). FV-Action 표 12: Charades에서 프레임 3→5장·폭 2배로 R@0.5 56.7→62.3. "
                    "REZE도 3초에 6장. 조사에서 효과가 가장 크게 보고된 축이라 견줄 기준으로 하나 넣는다"},
}

# 받은 답을 섞어 만드는 것(요청을 더 보내지 않는다). how: mean = 확률의 평균, min = 가장 낮은 확률
COMBOS = {
    "mean_base_reze": {"how": "mean", "of": ["base", "strict_reze"], "why": "중립 문장과 엄한 문장의 평균"},
    "mean_para4": {"how": "mean", "of": ["base", "short", "statement", "strict_q"], "why": "바꿔 쓴 문장 4개의 평균. 조사에서 직접 잰 근거는 찾지 못했다"},
    "min_base_near": {"how": "min", "of": ["base", "crit_near"], "why": "둘 다 높아야 잡는다"},
}

# 질문을 두 묶음으로 나눠 보기 위한 규칙(거칠다). 순간의 변화를 가리키는 동사가 있으면 change, 없으면 other
CHANGE = re.compile(r"\b(open|tak|took|put|pick|grab|throw|threw|turn|remov|pull|push|start|begin|began|stop|awaken|wak|woke)\w*|\bclos(e|es|ed|ing)\b"
                    r"|\b(stands?|stood|sits?|sat|gets?|got|lies|lay|lays)\s+(up|down)", re.I)  # closet, place(명사)가 걸리지 않게


def kind(sentence):
    return "change" if CHANGE.search(sentence) else "other"
