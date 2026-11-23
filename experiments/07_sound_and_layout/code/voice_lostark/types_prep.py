"""Gemini가 제안한 하이라이트 종류(types/proposal.json)를 판정 모델(H2O) 요청 묶음으로 바꾼다. 돈은 들지 않는다.

  types_gate   — 종류마다 한 번, 글만: 이 종류를 찾는 데 목소리가 필요한가(voice_role). 판정 모델이 정한다
  types_frames — 모든 초, 프레임만: 종류마다의 화면 질문(한 요청에 질문 여럿)
  types_voice  — 말하는 초, 글만: 종류마다의 목소리 질문(목소리 질문이 있는 종류만)
목소리 기록과 프레임 격자는 voice_prep.py와 같은 것을 쓴다.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import voice_prep as VP  # noqa: E402  (불러올 때 25-2절의 묶음을 같은 내용으로 다시 쓴다)

R2 = "--r2" in sys.argv          # 2차: 고친 화면 질문만 묶는다(types_revise.py). 종류·목소리 질문·필요 여부는 1차 답을 그대로 쓴다
P = json.load(open(ROOT / ("types/proposal_r2.json" if R2 else "types/proposal.json")))
types = P["types"]
if R2:
    VP.dump("types_frames_r2", {t["key"]: t["screen_question"] for t in types}, [{"k": k, "state": {}, "frames": VP.fr[k][0], "seconds": VP.fr[k][1]} for k in range(VP.n)],
            {"game": VP.GAME, "viewer_request": "highlight moments of this Lost Ark raid recording", "about": VP.ABOUT}, VP.n / 10)
    sys.exit(0)
gate_q = {"voice_role": {"type": "choice",
                         "instructions": "A viewer wants clips of the highlight type in `highlight_type` from a long gameplay recording that contains the player's spoken Korean voice "
                                         "(summarized in `voice_overview`; the voice reaches you only as transcripts and audio measurements). How much does finding this type depend on the player's voice, compared with the picture alone?",
                         "criteria": {"essential": "This type is defined by the player's or party's reaction or words; the picture alone cannot find it.",
                                      "helpful": "The picture can find it, but the voice would help confirm or rank the moments.",
                                      "not_needed": "This type is a visible event on screen; the voice adds little."}}}
ov = {"speaking_share": f"{float((VP.vad > 0.3).mean()) * 100:.0f}% of seconds", "sample_lines": VP.sample_lines}
gate_items = [{"k": i, "state": {"highlight_type": {"name": t["name_ko"], "description": t["why_ko"]}, "voice_overview": ov}} for i, t in enumerate(types)]
VP.dump("types_gate", gate_q, gate_items, {"game": VP.GAME}, 5)
fq = {t["key"]: t["screen_question"] for t in types}
VP.dump("types_frames", fq, [{"k": k, "state": {}, "frames": VP.fr[k][0], "seconds": VP.fr[k][1]} for k in range(VP.n)],
        {"game": VP.GAME, "viewer_request": "highlight moments of this Lost Ark raid recording", "about": VP.ABOUT}, VP.n / 10)
vq = {t["key"]: t["voice_question"] for t in types if t.get("voice_question")}
VP.dump("types_voice", vq, [{"k": k, "state": {"voice": VP.voice_record(k)}} for k in VP.speech_secs], {"game": VP.GAME}, len(VP.speech_secs) / 30)
print("types", [t["key"] for t in types], "| voice questions", list(vq))
