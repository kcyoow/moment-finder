"""종류마다: 판정 모델이 정한 목소리 필요 여부(gate)로 화면·목소리 확률을 섞어 후보 3개 → Gemini가 프레임으로 다시 확인 → 대표 장면 하나.
그 다음 "어떤 하이라이트를 찾을까요?" 확인 페이지(종류마다 한 장면)를 만든다.

  섞기: not_needed → 화면만, helpful → 0.6×화면 + 0.4×목소리, essential → 0.3×화면 + 0.7×목소리 (목소리가 없는 초는 목소리 0)
  다시 확인: pipeline/mf_verify.py의 save_item·ask_one 그대로(후보마다 프레임 8장). 요청 글은 그 종류의 이름과 설명
실행: uv run --no-project --with numpy --with imageio-ffmpeg python -I experiments/voice_lostark/types_combine.py [--verify] [--page]
  --verify가 있을 때만 Gemini를 부른다(종류마다 한 번, 돈이 든다)
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parents[1] / "pipeline"
sys.path.insert(0, str(PIPE))
import mf_core as C  # noqa: E402
import mf_verify as MV  # noqa: E402
from mf import positions_for  # noqa: E402

VIDEO = Path.home() / "Downloads/2023-09-17 19-08-32.mkv"
RES, TY = ROOT / "results", ROOT / "types"
R2 = "--r2" in sys.argv     # 2차: 고친 화면 질문(types_frames_r2)과 확인용 정의(verify_ko). 필요 여부·목소리 답은 1차 그대로
ARM = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else None   # 프레임 넣는 모양 실험: B, C, D (layout_prep.py). 질문·정의는 2차 것
SUF = f"_L{ARM}" if ARM else ("_r2" if R2 else "")
FRAMES = f"layout_{ARM}" if ARM else f"types_frames{SUF}"
P = json.load(open(TY / ("proposal_r2.json" if ARM else f"proposal{SUF}.json"))); types = P["types"]
n = json.load(open(RES / "out/per_second.json"))["seconds"]
pos = positions_for(n); t0 = np.array([p["t0"] for p in pos]); t1 = np.array([p["t1"] for p in pos])
W = {"not_needed": (1.0, 0.0), "helpful": (0.6, 0.4), "essential": (0.3, 0.7)}
mmss = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"


def answers(name):
    j = json.load(open(RES / f"h2o_out/{name}.json"))
    return {r["k"]: r["answers"] for r in j["records"] if r and "answers" in r}, j["meta"]


G, gm = answers("types_gate"); Fa, fm = answers(FRAMES); Va, vm = answers("types_voice")
print("meta", {k: (m["requests"], m["errors"], m["wall_seconds"]) for k, m in (("gate", gm), ("frames", fm), ("voice", vm))})
summary = {"ask_ko": P["ask_ko"], "looked_up": P["looked_up"], "types": []}
for i, t in enumerate(types):
    k = t["key"]
    probs = G[i]["voice_role"]["probabilities"]; role = max(probs, key=probs.get)
    sf = np.array([C.answer_value(Fa[s][k]) if s in Fa else np.nan for s in range(n)])
    sv = np.array([C.answer_value(Va[s][k]) if (s in Va and k in Va[s]) else 0.0 for s in range(n)])
    wf, wv = W[role] if t.get("voice_question") else (1.0, 0.0)
    score = wf * np.nan_to_num(sf, nan=0.0) + wv * sv
    clips = C.make_clips(score, t0, t1, n, budget=0.9)[:3]
    for c in clips:
        c["from"] = f"{k} ({role})"
    summary["types"].append({"key": k, "name_ko": t["name_ko"], "why_ko": t["why_ko"], "voice_role": {a: round(b, 3) for a, b in probs.items()}, "role": role,
                             "weights": [wf, wv], "frames_above_0.5": int((sf > 0.5).sum()), "voice_above_0.5": int((sv > 0.5).sum()),
                             "candidates": [{"start": c["start"], "end": c["end"], "peak_t": c["peak_t"], "peak": c["peak"], "length": c["length"], "rank": c["rank"]} for c in clips]})
    print(f"[{k}] {t['name_ko']}: role={role} {summary['types'][-1]['voice_role']} | 화면>0.5 {summary['types'][-1]['frames_above_0.5']}초, 목소리>0.5 {summary['types'][-1]['voice_above_0.5']}초 | 후보",
          [f"{mmss(c['start'])}~{mmss(c['end'])} ({c['peak']:.2f})" for c in clips])

if "--verify" in sys.argv:
    total = 0.0
    for ty in summary["types"]:
        d = RES / f"types_verify{SUF}" / ty["key"]
        vk = next((t.get("verify_ko") for t in types if t["key"] == ty["key"]), None)
        req = f"로스트아크 레이드 녹화에서 '{ty['name_ko']}' 장면을 찾아줘. {vk}" if vk else f"로스트아크 레이드 녹화에서 '{ty['name_ko']}' 장면을 찾아줘 ({ty['why_ko']})"
        clips = [dict(c) for c in ty["candidates"]]
        old = d / "verdict.json"
        if old.exists() and [(c["start"], c["end"]) for c in json.load(open(d / "item.json"))["candidates"]] == [(c["start"], c["end"]) for c in clips]:
            v = json.load(open(old)); print(f"  [{ty['key']}] 이전 다시 확인 답을 씀(같은 후보, 다시 부르지 않음)")   # 멈췄다 이어 할 때 같은 값을 두 번 내지 않는다
        else:
            MV.save_item(ty["key"], VIDEO, req, float(n), (1920, 1080), clips, d=d)
            v = MV.ask_one(ty["key"], d=d)
            total += v["usage"]["usd_estimate"]
        best = max(v["candidates"], key=lambda c: c["p"])
        ty["verify"] = {"candidates": v["candidates"], "note": v.get("note"), "usd": v["usage"]["usd_estimate"]}
        ty["pick"] = {"id": best["id"], "p": best["p"], "see": best["see"], **{x: ty["candidates"][best["id"] - 1][x] for x in ("start", "end", "peak_t")}}
        print(f"  확인 [{ty['key']}] 대표 {mmss(ty['pick']['start'])}~{mmss(ty['pick']['end'])} p={best['p']:.2f} | {best['see'][:90]}")
    summary["verify_usd"] = round(total, 4)
    print("Gemini 확인 값 어림 $", round(total, 4))
json.dump(summary, open(RES / f"types_summary{SUF}.json", "w"), ensure_ascii=False, indent=1)

if "--page" in sys.argv:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    out = ROOT.parents[1] / "reports/voice_try"; (out / f"types{SUF}").mkdir(parents=True, exist_ok=True)
    rows = ""
    for ty in summary["types"]:
        pk = ty.get("pick") or dict(ty["candidates"][0], p=None, see="(다시 확인 전)")
        a = max(0.0, pk["start"]); b = min(pk["end"], a + 30)
        f = out / f"types{SUF}" / f"{ty['key']}.mp4"
        subprocess.run([FF, "-v", "error", "-y", "-ss", f"{a:.2f}", "-i", str(VIDEO), "-t", f"{b - a:.2f}", "-vf", "scale=-2:540", "-c:v", "h264_videotoolbox",
                        "-b:v", "2500k", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(f)], check=True)
        role_ko = {"essential": "꼭 필요", "helpful": "도움", "not_needed": "필요 없음"}[ty["role"]]
        found = "찾음" if (pk.get("p") or 0) >= 0.5 else "확실한 장면을 찾지 못함(가장 가까운 후보)"
        rows += (f'<section><h2>{ty["name_ko"]}</h2><p class="m">{ty["why_ko"]}</p>'
                 f'<video controls preload="metadata" src="types{SUF}/{f.name}"></video>'
                 f'<p><b>{mmss(a)}부터 {int(b - a)}초</b> · {found}' + (f' (Gemini 확률 {pk["p"]:.2f})' if pk.get("p") is not None else "") + '</p>'
                 f'<p class="m">Gemini가 본 것: {pk.get("see", "")}</p>'
                 f'<p class="m">판정 모델(H2O)이 정한 목소리 필요 여부: <b>{role_ko}</b> (꼭 {ty["voice_role"]["essential"]:.2f} / 도움 {ty["voice_role"]["helpful"]:.2f} / 없음 {ty["voice_role"]["not_needed"]:.2f})</p></section>')
    css = ("body{font:15px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;background:#eef0f2;color:#14171c;margin:0;padding:24px 16px}main{max-width:820px;margin:0 auto}"
           "section{background:#fff;border:1px solid #d3d7dd;border-radius:10px;padding:14px;margin:14px 0}h2{font-size:18px;margin:0 0 4px}"
           "video{width:100%;border-radius:6px;background:#000;margin:8px 0}.m{color:#5d6672;font-size:13.5px;margin:4px 0}")
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>하이라이트 종류 확인</title>'
            f'<style>{css}</style></head><body><main><h1>{summary["ask_ko"]}</h1>'
            f'<p class="m">요청: "{P["request"]}". 중간 LLM이 검색으로 이 게임의 하이라이트 종류를 찾아 아래처럼 나눴다. 종류마다 대표 장면 하나다. 맞으면 고르고, 아니면 빼면 된다.</p>'
            f'<p class="m">검색으로 확인한 것: {P["looked_up"]}</p>{rows}</main></body></html>')
    (out / f"types{SUF}.html").write_text(html, encoding="utf-8")
    print("page ->", out / f"types{SUF}.html")
