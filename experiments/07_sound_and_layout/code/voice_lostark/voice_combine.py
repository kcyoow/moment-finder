"""H2O의 답(results/h2o_out/)을 합쳐 후보를 고르고, 방식끼리 견주고, 소유자가 가린 채 볼 클립 페이지를 만든다. 돈은 들지 않는다.

후보 목록(사이트의 mf_recheck.candidates와 같은 고르기: make_clips(budget=0.9) + round_robin):
  F  화면만(기준선)   : 합친 점수, 질문 하나하나, 막연한 질문에서 돌아가며 6개 — 사이트와 같은 방식
  V  목소리 읽기(A)   : 반응×(1−잡담), 전멸을 말함, 성공을 말함에서 돌아가며 4개
  W  증인 카드(B)     : 측정값만 읽은 하이라이트 판단에서 4개
  T  TripleSumm        : 재시청 모델 점수에서 3개(견주기용 증인)
  최종: F 6개 + V를 gate의 voice_role이 정한 몫(essential 4 / helpful 2 / not_needed 0)
돌리는 법: uv run --no-project --with numpy --with imageio-ffmpeg python -I experiments/voice_lostark/voice_combine.py [--clips]
"""
import json, random, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT.parents[1] / "pipeline"))
import mf_core as C  # noqa: E402
from mf import positions_for  # noqa: E402

VIDEO = Path.home() / "Downloads/2023-09-17 19-08-32.mkv"
n = json.load(open(RES / "out/per_second.json"))["seconds"]
pos = positions_for(n); t0 = np.array([p["t0"] for p in pos]); t1 = np.array([p["t1"] for p in pos])
mmss = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"


def load(name):
    p = RES / f"h2o_out/{name}.json"
    if not p.exists():
        return None, None
    j = json.load(open(p)); spec = json.load(open(ROOT / f"specs/{name}.json"))
    by_k = {r["k"]: r.get("answers") for r in j["records"] if r and "answers" in r}
    return j["meta"], {"spec": spec, "by_k": by_k}


def per_second(d, key, default=0.0):
    out = np.full(n, default)
    for k, a in d["by_k"].items():
        if isinstance(k, int) and k < n and key in a:
            out[k] = C.answer_value(a[key])
    return out


def clips_from(lists, m):
    return C.round_robin([C.make_clips(x, t0, t1, n, budget=0.9) for x in lists], m)


summary, cand = {}, {}
for name in ("gate", "voice", "cards", "frames", "mix_fv", "mix_vf"):
    meta, _ = load(name)
    summary[name] = {k: meta.get(k) for k in ("requests", "answered", "errors", "wall_seconds", "tokens_median")} if meta else None
print(json.dumps(summary, ensure_ascii=False))

# 요청 단위: 목소리가 필요한가
_, G = load("gate")
gate = {}
if G:
    for it in G["spec"]["items"]:
        a = G["by_k"].get(it["k"], {}).get("voice_role")
        gate[it["state"]["viewer_request"]] = a["probabilities"] if a else None
print("voice_role", json.dumps(gate, ensure_ascii=False))
main_role = max(gate[next(iter(gate))], key=gate[next(iter(gate))].get) if gate and gate[next(iter(gate))] else "helpful"
slots = {"essential": 4, "helpful": 2, "not_needed": 0}[main_role]

# F: 화면만(사이트와 같은 합치기)
_, F = load("frames")
fkeys = [k for k in F["spec"]["questions"] if k != "worth"]
answers = [F["by_k"].get(k) for k in range(n)]
comb = C.combine(answers, fkeys, vague_key="worth")
f_lists = [comb["final"]] + [comb["rank"][k] for k in fkeys] + [comb["vague"]]
cand["F"] = clips_from(f_lists, 6)
for c in cand["F"]:
    c["from"] = (["합친 점수"] + fkeys + ["막연한 질문"])[c["source"]]

# V: 목소리 읽기
_, V = load("voice")
react, offt = per_second(V, "player_reacts"), per_second(V, "off_topic")
v_react = react * (1 - offt); v_wipe = per_second(V, "says_wipe"); v_succ = per_second(V, "says_success")
cand["V"] = clips_from([v_react, v_wipe, v_succ], 4)
for c in cand["V"]:
    c["from"] = ["반응×(1−잡담)", "전멸을 말함", "성공을 말함"][c["source"]]

# W: 증인 카드
_, W = load("cards")
w_score = per_second(W, "highlight_from_witnesses")
cand["W"] = clips_from([w_score], 4)
# T: TripleSumm
ts = np.array(json.load(open(RES / "triplesumm/scores.json"))["score"][:n])
cand["T"] = clips_from([ts], 3)

# 최종: F + V 몫
final = list(cand["F"])
for c in cand["V"]:
    if len([x for x in final if x.get("arm") == "V"]) >= slots: break
    if all(min(c["end"], o["end"]) - max(c["start"], o["start"]) <= 0.5 * min(c["length"], o["length"]) for o in final):
        final.append(dict(c, arm="V"))
summary["voice_role_main"] = main_role; summary["voice_slots"] = slots

# 견주기
def corr(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) if ok.sum() > 3 else None

speech = np.array([k in V["by_k"] for k in range(n)])
summary["speech_positions"] = int(speech.sum())
summary["corr"] = {"V_react vs W_cards (speech secs)": corr(v_react[speech], w_score[speech]),
                   "V_react vs F_final": corr(v_react, comb["final"]), "W_cards vs F_final": corr(w_score, comb["final"]),
                   "TripleSumm vs F_final": corr(ts, comb["final"]), "TripleSumm vs W_cards": corr(ts, w_score),
                   "says_wipe vs frames wipe_or_fail": corr(v_wipe, comb["raw"]["wipe_or_fail"])}
# 전멸: 화면의 전멸 정점 뒤 15초 안에 말로 전멸을 말했나
fw = comb["raw"]["wipe_or_fail"]; peaks = [k for k in range(n) if fw[k] > 0.7 and fw[k] == fw[max(0, k - 10): k + 11].max()]
summary["wipe_peaks_frames"] = [mmss(k) for k in peaks][:20]
summary["wipe_peaks_followed_by_wipe_talk_within_15s"] = sum(1 for k in peaks if v_wipe[k: k + 16].max() > 0.5)

# 넣는 자리 시험: 같은 382곳에서 화면 질문의 답이 목소리 글 때문에 얼마나 움직였나
_, MF = load("mix_fv"); _, MV = load("mix_vf")
place = {}
if MF and MV:
    ks = sorted(set(MF["by_k"]) & set(MV["by_k"]) & set(F["by_k"]))
    for q in fkeys + ["worth"]:
        base = np.array([C.answer_value(F["by_k"][k][q]) for k in ks])
        fv = np.array([C.answer_value(MF["by_k"][k][q]) for k in ks]); vf = np.array([C.answer_value(MV["by_k"][k][q]) for k in ks])
        scale = 3.0 if "score" in F["by_k"][ks[0]][q] else 1.0
        place[q] = {"mean_abs_shift_frames_first": round(float(np.mean(np.abs(fv - base)) / scale), 4),
                    "mean_abs_shift_voice_first": round(float(np.mean(np.abs(vf - base)) / scale), 4)}
        if scale == 1.0:
            place[q]["cross_0.5_frames_first"] = int(((fv > 0.5) != (base > 0.5)).sum()); place[q]["cross_0.5_voice_first"] = int(((vf > 0.5) != (base > 0.5)).sum())
    hv_fv = np.array([C.answer_value(MF["by_k"][k]["highlight_with_voice"]) for k in ks]); hv_vf = np.array([C.answer_value(MV["by_k"][k]["highlight_with_voice"]) for k in ks])
    place["highlight_with_voice"] = {"corr_fv_vf": corr(hv_fv, hv_vf), "mean_fv": round(float(hv_fv.mean()), 3), "mean_vf": round(float(hv_vf.mean()), 3),
                                     "corr_with_V_react": corr(hv_vf, v_react[ks]), "corr_with_F_final": corr(hv_vf, comb["final"][ks])}
    place["n"] = len(ks)
summary["placement"] = place

def brief(cs):
    return [{"span": f"{mmss(c['start'])}~{mmss(c['end'])}", "from": c.get("from"), "peak": c.get("peak")} for c in cs]
summary["candidates"] = {k: brief(v) for k, v in cand.items()}
summary["final"] = [dict(brief([c])[0], arm=c.get("arm", "F")) for c in final]
json.dump(summary, open(RES / "combine_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: summary[k] for k in summary if k not in ("candidates", "final")}, ensure_ascii=False, indent=1))
print(json.dumps(summary["candidates"], ensure_ascii=False))

# 가린 채 볼 클립 페이지(이 Mac 안에서만)
if "--clips" in sys.argv:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    pool = []
    for arm in ("F", "V", "W", "T"):
        for c in cand[arm]:
            hit = next((p for p in pool if min(c["end"], p["end"]) - max(c["start"], p["start"]) > 0.5 * min(c["end"] - c["start"], p["end"] - p["start"])), None)
            if hit: hit["arms"].append(arm)
            else: pool.append({"start": c["start"], "end": min(c["end"], c["start"] + 30), "arms": [arm]})
    random.Random(17).shuffle(pool)
    out = ROOT.parents[1] / "reports/voice_try"; (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = ""
    for i, p in enumerate(pool, 1):
        f = out / "clips" / f"c{i:02d}.mp4"
        subprocess.run([FF, "-v", "error", "-y", "-ss", f"{p['start']:.2f}", "-i", str(VIDEO), "-t", f"{p['end'] - p['start']:.2f}", "-vf", "scale=-2:540",
                        "-c:v", "h264_videotoolbox", "-b:v", "2500k", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(f)], check=True)
        rows += (f'<div class="row"><b>{i}</b><video controls preload="metadata" src="clips/{f.name}"></video>'
                 f'<span class="m">{mmss(p["start"])}부터 {int(p["end"] - p["start"])}초</span></div>')
        p["id"] = i
    json.dump(pool, open(RES / "blind_key.json", "w"), ensure_ascii=False, indent=1)   # 어느 방식에서 나왔는지(페이지에는 싣지 않는다)
    css = ("body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;background:#eef0f2;color:#14171c;margin:0;padding:24px 16px}main{max-width:820px;margin:0 auto}"
           ".row{display:grid;grid-template-columns:36px 1fr;gap:6px 12px;background:#fff;border:1px solid #d3d7dd;border-radius:10px;padding:12px;margin:10px 0}"
           "video{width:100%;border-radius:6px;background:#000}b{font:600 20px ui-monospace,Menlo,monospace}.m{grid-column:2;color:#5d6672;font-size:13px}")
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>로스트아크 후보 클립</title>'
            f'<style>{css}</style></head><body><main><h1>로스트아크 45분 · 후보 클립 {len(pool)}개</h1>'
            '<p>여러 방식이 고른 후보를 섞고 어느 방식에서 나왔는지 가렸다. 소리를 켜고 보고, 번호마다 넣는다 / 애매 / 뺀다를 알려 주면 된다.</p>'
            f'{rows}</main></body></html>')
    (out / "index.html").write_text(html, encoding="utf-8")
    print("clips", len(pool), "->", out / "index.html")
