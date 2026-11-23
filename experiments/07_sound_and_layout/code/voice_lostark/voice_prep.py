"""측정 모델들의 결과를 판정 모델(H2O)에 넘길 요청 묶음(spec)으로 바꾼다. 돈은 들지 않는다(이 Mac에서 파일만 만든다).

읽는 것(results/): out/(소리 처리), witness_out/(증인 다섯), triplesumm/scores.json
만드는 것(specs/, h2o_voice_modal.py가 읽는 모양):
  gate     — 요청 단위, 글만: 이 요청에 목소리가 필요한가(voice_role). 요청 셋(본 요청 + 대조 둘)
  voice    — 말하는 초마다, 글만: 받아쓰기·크기·웃음 등을 읽고 반응·전멸·성공·잡담을 판단 (조사 A의 방식)
  cards    — 모든 초, 글만: 받아쓰기 없이 측정값(크기, 웃음, 발성, 흥분도, 화자, 재시청 모델, 게임 소리)만 읽고 판단 (조사 B의 방식)
  frames   — 모든 초, 프레임만: 사이트와 같은 화면 판정(24-3절의 질문 넷 + 전멸 질문 + 막연한 질문). 기준선
  mix_vf / mix_fv — 400곳, 프레임 + 목소리 기록: 같은 내용을 넣는 자리만 바꾼다(목소리가 프레임 앞 / 뒤). VideoRouteAudit의 "넣는 자리" 시험
돌리는 법: uv run --no-project --with numpy python -I experiments/voice_lostark/voice_prep.py
"""
import json, random, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "specs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parents[1] / "pipeline"))
from mf import ABOUT, OFFSETS, Q_VAGUE, positions_for  # noqa: E402  사이트와 같은 위치·문구

REQUEST = "the highlight moments of this Lost Ark raid video, including dramatic party wipes"
GAME = "Lost Ark (online action RPG). This recording is a party raid against the Legion Commander Kamen; the recording player is in the party and talks with party members in Korean."
per = json.load(open(RES / "out/per_second.json")); n = per["seconds"]
asr = [s for s in json.load(open(RES / "out/asr_vocals.json")) if s["text"] and s["s"] is not None]
vad = np.array(per["vad_speech"]); vdb = np.array(per["db"]["vocals"])


def load(p):
    try:
        j = json.load(open(p)); return None if "error" in j else j
    except FileNotFoundError:
        return None


W = {k: load(RES / f"witness_out/{k}.json") for k in ("diarization", "laughter", "vocalburst", "arousal", "sensevoice")}
TS = load(RES / "triplesumm/scores.json")
print({k: (v is not None) for k, v in W.items()}, "triplesumm", TS is not None)

# 초마다의 측정값
speech_med = float(np.median(vdb[vad > 0.3])) if (vad > 0.3).any() else -40.0
loud_ratio = 10 ** ((vdb - speech_med) / 20)
laugh = np.array(W["laughter"]["per_second"]) if W["laughter"] else np.zeros(n)
burst = np.zeros(n)
for e in (W["vocalburst"] or {}).get("events", []):
    burst[int(e["start"]): int(e["end"]) + 1] = np.maximum(burst[int(e["start"]): int(e["end"]) + 1], e["confidence"])
arousal = np.full(n, np.nan)
for s in (W["arousal"] or {}).get("segments", []):
    arousal[int(s["s"]): int(s["e"]) + 1] = s["arousal"]
ar_valid = arousal[~np.isnan(arousal)]
ar_pct = np.array([(ar_valid < a).mean() if not np.isnan(a) else np.nan for a in arousal]) if len(ar_valid) else np.full(n, np.nan)
emo = [""] * n
for s in (W["sensevoice"] or {}).get("segments", []):
    tag = next((t for t in s["tags"] if t in ("HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED")), "")
    ev = next((t for t in s["tags"] if t in ("Laughter", "Cry", "Applause", "Cough", "Sneeze", "Breath")), "")
    for k in range(int(s["s"]), min(n, int(s["e"]) + 1)):
        emo[k] = ", ".join(x for x in (tag.lower(), ev.lower()) if x)
spk = [None] * n; main_spk = None
if W["diarization"]:
    talk = {}
    for s in W["diarization"]["segments"]:
        talk[s["Speaker"]] = talk.get(s["Speaker"], 0) + s["End"] - s["Start"]
    main_spk = max(talk, key=talk.get) if talk else None
    for k, row in enumerate(W["diarization"]["per_second"][:n]):
        if max(row) > 0.5: spk[k] = int(np.argmax(row))
ts_pct = None
if TS:
    sc = np.array(TS["score"][:n]); ts_pct = np.array([(sc < v).mean() for v in sc])
game_ev = per["game_event"]


def pct_word(p):
    return f"top {max(1, int(round((1 - p) * 100)))}% of this video" if p >= 0.5 else f"bottom {max(1, int(round(p * 100)))}% of this video"


def voice_record(k, with_lines=True):
    lo, hi = k - 3, k + 5
    rec = {}
    if with_lines:
        rec["lines"] = [{"from": f"{s['s'] - k:+.1f}s", "text": s["text"]} for s in asr if s["e"] is not None and s["s"] < hi and s["e"] > lo and s["vad"] >= 0.2][:6]
    w = slice(max(0, k - 1), min(n, k + 2))
    rec["speaking"] = f"{float(vad[w].mean()) * 100:.0f}% of the 3 seconds around now"
    if vad[w].max() > 0.2:
        rec["loudness"] = f"{float(loud_ratio[w].max()):.1f}x the player's usual speaking level"
        if not np.isnan(ar_pct[k]): rec["vocal_arousal"] = pct_word(float(ar_pct[k]))
        if spk[k] is not None and main_spk is not None:
            rec["speaker"] = "the most frequent speaker (likely the recording player)" if spk[k] == main_spk else f"another voice (speaker {spk[k]})"
    sounds = []
    if laugh[w].max() > 0.5: sounds.append(f"laughter ({float(laugh[w].max()):.2f})")
    if burst[w].max() > 0: sounds.append(f"short vocal burst such as a scream, gasp or sigh ({float(burst[w].max()):.2f})")
    if emo[k]: sounds.append(f"speech emotion/event tag: {emo[k]}")
    if sounds: rec["vocal_sounds"] = sounds
    return rec


def card(k):
    c = voice_record(k, with_lines=False)
    lo, hi = k - 3, k + 5
    c["words_spoken_nearby"] = sum(len(s["text"].split()) for s in asr if s["e"] is not None and s["s"] < hi and s["e"] > lo and s["vad"] >= 0.2)
    if ts_pct is not None: c["replay_model"] = f"TripleSumm replay-likelihood score is in the {pct_word(float(ts_pct[k]))}"
    if game_ev["score"][k] > 0.15: c["game_sound"] = f"{game_ev['label'][k]} ({game_ev['score'][k]:.2f} above usual)"
    return c


def noul(instr, t, f):
    return {"type": "noul", "instructions": instr, "criteria": {"true": t, "false": f}}


VOICE_INTRO = ("`voice` summarizes what the recording player said and how they sounded around this moment of a Lost Ark raid recording. "
               "`lines` are automatically transcribed Korean speech (may contain errors) with start times relative to now; the other fields come from audio models. ")
Q_VOICE = {
    "player_reacts": noul(VOICE_INTRO + "Is the player reacting strongly to something that just happened in the game?",
                          "Clear excitement, relief, panic, frustration, laughter, shouting or cursing about a game event.",
                          "Calm talk, routine callouts, silence, or reactions unrelated to the game."),
    "says_wipe": noul(VOICE_INTRO + "Does the speech indicate that the party wiped, someone died, or the attempt failed?",
                      "Words such as 죽었다, 전멸, 망했다, 다시, 재시작, or clear dismay about dying/failing.",
                      "No sign of death, wipe or failure."),
    "says_success": noul(VOICE_INTRO + "Does the speech indicate a success: a mechanic cleared, a big hit, the boss downed or the raid cleared?",
                         "Words such as 클리어, 잡았다, 성공, 나이스, 됐다, or clear joy about succeeding.",
                         "No sign of success."),
    "off_topic": noul(VOICE_INTRO + "Is the talk unrelated to the current gameplay?",
                      "Small talk or other topics not about what is happening in the raid.", "Talk about the raid, or no talk."),
}
CARD_INTRO = ("`witnesses` lists what several audio and video models measured at this moment of a Lost Ark raid recording "
              "(no transcript is given). Each value is a model output and may be wrong. ")
Q_CARDS = {"highlight_from_witnesses": noul(CARD_INTRO + "Would a viewer making a highlight video of this raid (including dramatic wipes) want this moment?",
                                            "The measurements point to a strong reaction or a notable game event.",
                                            "The measurements look like ordinary play, calm talk or silence.")}
writer = json.load(open(ROOT.parents[1] / "pipeline/runs/web_1117_023931_1/writer.json"))
Q_FRAMES = {"worth": {"type": "noul", "instructions": Q_VAGUE}, **writer["questions"],
            "wipe_or_fail": noul("`frames` holds 4 consecutive frames from a video, 0.5 seconds apart, in time order (earliest first). Do these frames show a party wipe or failure?",
                                 "Characters dying or lying dead, a failure banner (for example the Korean text 공략에 실패하였습니다), or a restart/retry vote window.",
                                 "Combat or movement continues normally, or a success/clear screen.")}
Q_MIX = {**Q_FRAMES, "highlight_with_voice": noul("Consider both `frames` (4 frames, 0.5 seconds apart) and `voice` (what the player said and how they sounded). "
                                                  "Would this moment belong in a highlight video of this raid (including dramatic wipes)?",
                                                  "A notable game event or a strong player reaction is visible or audible.", "Ordinary play or calm talk.")}

pos = positions_for(n)
grid = lambda t: min(max(int(round((t - 0.25) / 0.5)), 0), 2 * n - 1)
fr = {p["k"]: ([grid(t) for t in p["times"]], p["times"]) for p in pos}
speech_secs = [k for k in range(n) if vad[max(0, k - 2): k + 3].max() > 0.2]


def dump(name, questions, items, base, est):
    json.dump({"name": name, "questions": questions, "base_state": base, "items": items, "est_seconds": est}, open(OUT / f"{name}.json", "w"), ensure_ascii=False)
    print(name, len(items), "items")


rng = random.Random(7)
sample_lines = [s["text"] for s in rng.sample([s for s in asr if s["vad"] >= 0.3], 15)]
gate_q = {"voice_role": {"type": "choice", "instructions": "A viewer asked for `viewer_request` from a long gameplay recording that contains the player's spoken voice (summarized in `voice_overview`). How much does finding the requested moments depend on the player's voice, compared with the picture alone?",
                         "criteria": {"essential": "The request is about the player's own reactions or words; the picture alone cannot find it.",
                                      "helpful": "The picture can find it, but the player's reactions would help confirm or rank moments.",
                                      "not_needed": "The request is about something visible on screen; the voice adds little."}}}
gate_items = [{"k": i, "state": {"viewer_request": r, "voice_overview": {"speaking_share": f"{float((vad > 0.3).mean()) * 100:.0f}% of seconds", "sample_lines": sample_lines}}}
              for i, r in enumerate([REQUEST, "the moments where the boss is staggered or knocked down", "the moments where I (the player) laughed or got angry"])]
dump("gate", gate_q, gate_items, {"game": GAME}, 5)
dump("voice", Q_VOICE, [{"k": k, "state": {"voice": voice_record(k)}} for k in speech_secs], {"game": GAME}, len(speech_secs) / 30)
dump("cards", Q_CARDS, [{"k": k, "state": {"witnesses": card(k)}} for k in range(n)], {"game": GAME}, n / 30)
base_f = {"game": GAME, "viewer_request": REQUEST, "about": ABOUT}
dump("frames", Q_FRAMES, [{"k": k, "state": {}, "frames": fr[k][0], "seconds": fr[k][1]} for k in range(n)], base_f, n / 10)
sub = sorted(set(rng.sample(speech_secs, min(200, len(speech_secs))) + rng.sample(range(n), 200)))
for name, first in (("mix_fv", True), ("mix_vf", False)):
    dump(name, Q_MIX, [{"k": k, "state": {"voice": voice_record(k)}, "frames": fr[k][0], "seconds": fr[k][1], "frames_first": first} for k in sub], base_f, len(sub) / 10)
ex = speech_secs[len(speech_secs) // 2]
print("example voice record", ex, json.dumps(voice_record(ex), ensure_ascii=False)[:600])
print("example card", ex, json.dumps(card(ex), ensure_ascii=False)[:400])
