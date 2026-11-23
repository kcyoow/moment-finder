"""다시 확인 단계: 판정 모델이 고른 후보의 프레임을 Gemini가 보고, 찾는 장면이 담겼을 확률을 준다. pipeline/README.md 22절.

후보를 뽑는 법, 프레임, Gemini에 주는 글은 18·20절에서 잰 것과 같다(mf_verify.py의 함수를 그대로 쓴다). 여기는 실행 폴더 하나에 그것을 잇는다.

  python mf_recheck.py <실행 이름> [--fake]     # runs/<이름>/의 받아 둔 답으로 후보를 뽑아 확인하고, clips.json을 고쳐 쓰고, 클립을 다시 자른다 (--fake가 아니면 Gemini 값이 든다)
  python mf_recheck.py <실행 이름> --h2o        # Gemini가 프레임을 보는 대신 판정 모델(H2O)이 스스로 다시 확인한다 (2026-11-23. experiments/decision_api/README.md 10·12·13절)
                                               # 올려 둔 판정 모델 앱을 부른다(MODAL_PROFILE, --with modal 필요). 확인 질문은 Gemini가 요청만 보고 쓴다

적는 것: runs/<이름>/verify/{item.json, frames/, prompt.txt, response.json, verdict.json}, runs/<이름>/clips.json
실행: uv run --offline --no-project --with numpy --with pillow --with imageio-ffmpeg python -I pipeline/mf_recheck.py <이름>
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf  # noqa: E402
import mf_core as C  # noqa: E402
import mf_verify as M  # noqa: E402
import mf_writer as W  # noqa: E402

N_CAND, KEEP_P = 6, 0.5
FAKE_P = (0.94, 0.72, 0.31, 0.12, 0.08, 0.04)

# ---- 판정 모델(H2O)이 스스로 다시 확인하기 (experiments/decision_api/README.md 10·12절에서 잰 방식)
# 후보의 프레임 8장을 4장씩 두 번(연속 4장, 고르게 4장) 같은 판정 모델에 "확인 질문"으로 묻고 두 답을 평균한다.
# 확인 질문은 Gemini 4 Hydrogen이 요청만 보고 쓴다(영상과 그림은 주지 않는다). 프레임은 Gemini로 가지 않는다.
H2O_MODEL = "H2O-Lightning-4B (판정 모델이 스스로 다시 봄)"
LEAD_A = "These 4 frames are consecutive frames, 0.5 seconds apart, from one second of one candidate segment of a longer video."
LEAD_B = "These 4 frames are spread evenly across one candidate segment of a longer video, in time order."
CHOICE_ASK = "Which option best describes what happens in this segment?"
CHOICE_NONE = "None of the other options is visible in these frames."
SERVE_APP_H2O = "moment-finder-serve2"          # 고르기 질문을 받는 앱(mf_serve2_modal.py). 환경 변수 MF_SERVE_APP로 바꿀 수 있다
VQ_KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
FAKE_QUESTION = {"writer": "흉내 (Gemini를 부르지 않음)", "understanding": "(흉내) 꾸민 확인 질문입니다.", "viewer_request": "the scenes where the robot successfully puts a candy into a cup",
                 "target": {"key": "robot_puts_candy_in_cup", "is": "The robot moves a candy to a cup and drops it so the candy stays inside the cup."},
                 "confusers": [{"key": "failed_drop_outside_cup", "is": "The robot drops a candy but it lands outside the cup."},
                               {"key": "idle_or_retracting", "is": "The robot stays still or pulls back with no candy held."}],
                 "noul": {"instructions": "Do these frames show the robot putting a candy into a cup successfully?",
                          "criteria": {"true": "The robot drops a candy and it stays inside the cup.", "false": "No candy is put into a cup successfully."}},
                 "search_queries": [], "format_retries": 0, "usage": {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0, "usd_estimate": 0.0}}


def candidates(run):
    """받아 둔 답 → 후보 클립. 목록 여럿(합친 점수, 질문 하나하나, 막연한 질문)에서 돌아가며 N_CAND개."""
    pos, qj, comb, t0, t1, _ = mf.load_scores(run)
    info = mf.read(run / "run.json")
    keys = list(qj["criteria_keys"])
    lists = [("합친 점수", comb["final"])] + [(k, comb["rank"][k]) for k in keys]
    if keys and "vague" in comb:
        lists.append(("막연한 질문", comb["vague"]))
    picked = C.round_robin([C.make_clips(x, t0, t1, info["duration"], budget=0.9) for _, x in lists], N_CAND)
    ranks = {k: comb["rank"][k] for k in keys}
    out = []
    for i, c in enumerate(picked, 1):
        c = dict(c)
        c["from"] = lists[c.pop("source")][0]
        c["why"] = C.why(c, ranks, t0, t1)
        c["id"] = i
        out.append(c)
    return out, info


def renumber(text, to_rank):
    """Gemini가 적은 글의 '후보 N'(Gemini에 준 차례)을 화면의 순위로 바꾼다. 화면에는 순위만 보이기 때문이다. 뒤따르는 조사는 새 숫자에 맞춘다."""
    if not text:
        return text

    def fix(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if any(n not in to_rank for n in nums):
            return m.group(0)
        new = sorted(to_rank[n] for n in nums)
        part = m.group(2) or ""
        if part:
            open_end = str(new[-1])[-1] in "2459"                      # 이·사·오·구: 받침이 없다
            part = {"은": "는", "는": "는", "이": "가", "가": "가", "을": "를", "를": "를", "과": "와", "와": "와"}[part] if open_end else \
                   {"은": "은", "는": "은", "이": "이", "가": "이", "을": "을", "를": "을", "과": "과", "와": "과"}[part]
        return "후보 " + ", ".join(map(str, new)) + part
    return re.sub(r"후보\s*(\d+(?:\s*(?:,|과|와|및|·)\s*\d+)*)(?:(은|는|이|가|을|를|과|와)(?![가-힣]))?", fix, text)


def union_length(clips):
    total, end = 0.0, -1.0
    for c in sorted(clips, key=lambda c: c["start"]):
        total += max(0.0, c["end"] - max(c["start"], end))
        end = max(end, c["end"])
    return total


def mark_skipped(name, reason):
    """확인하지 못했다고 적는다. 클립은 확인 전의 것 그대로다."""
    run = mf.RUNS / name
    cj = mf.read(run / "clips.json")
    if "unverified" in cj:
        cj["clips"] = cj.pop("unverified")
        total = sum(c["length"] for c in cj["clips"])
        cj.update(total_length=round(total, 2), share=round(total / cj["duration"], 4))
    cj["verify"] = {"status": "skipped", "reason": reason}
    mf.write(run / "clips.json", cj)
    return cj["verify"]


def recheck(name, request=None, fake=False, on_stage=None):
    """runs/<name>/의 답으로 후보를 뽑아 Gemini에 확인시키고 clips.json을 고쳐 쓴다. 돌려주는 것: clips.json의 verify."""
    run = mf.RUNS / name
    say = on_stage or (lambda text, frac=0.0: None)
    cj = mf.read(run / "clips.json")
    base = cj.get("unverified") or cj["clips"]                      # 확인 전의 클립(다시 돌려도 처음 것을 지킨다)
    cands, info = candidates(run)
    if not cands:
        return mark_skipped(name, "후보가 없다")
    if not request:
        for f, k in (("job.json", "request"), ("writer.json", "request"), ("run.json", "viewer_request")):
            if (run / f).exists() and mf.read(run / f).get(k):
                request = mf.read(run / f)[k]
                break
    size = (info["width"], info["height"]) if "width" in info else (lambda p: (p["width"], p["height"]))(mf.probe(info["video"]))
    d = run / "verify"
    shutil.rmtree(d, ignore_errors=True)                            # 후보가 바뀌면 프레임도 바뀐다. 이 단계가 만든 것만 지운다
    say(f"후보 {len(cands)}개의 프레임을 뽑는 중", 0.1)
    M.save_item(name, info["video"], request, info["duration"], size, cands, d=d)
    say(f"Gemini가 후보 {len(cands)}개의 프레임 {len(cands) * 8}장을 보는 중", 0.35)
    if fake:
        verdict = {"name": name, "model": "흉내 (Gemini를 부르지 않음)", "thinking": None, "format_retries": 0,
                   "usage": {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0, "usd_estimate": 0.0}, "note": "흉내 모드라서 프레임을 보지 않았습니다. 값은 꾸민 것입니다.",
                   "candidates": [{"id": c["id"], "p": FAKE_P[i % len(FAKE_P)], "see": "(흉내) 프레임을 보지 않고 꾸민 값입니다."} for i, c in enumerate(cands)]}
        mf.write(d / "verdict.json", verdict)
    else:
        verdict = M.ask_one(name, d=d)
    by = {v["id"]: v for v in verdict["candidates"]}
    clips = [{**{k: c[k] for k in ("start", "end", "length", "peak_t", "peak", "why", "from")}, "cand": c["id"],
              "p": round(by[c["id"]]["p"], 3), "see": by[c["id"]]["see"], "match": by[c["id"]]["p"] >= KEEP_P} for c in cands]
    clips.sort(key=lambda c: (not c["match"], -c["p"], c["start"]))  # 통과한 것 먼저, 그 안에서는 p가 높은 순
    for r, c in enumerate(clips, 1):
        c["rank"] = r
    to_rank = {c["cand"]: c["rank"] for c in clips}
    for c in clips:
        c["see"] = renumber(c["see"], to_rank)
    note = renumber(verdict.get("note"), to_rank)
    hit = [c for c in clips if c["match"]]
    total = union_length(hit)
    cj.update(clips=clips, unverified=base, total_length=round(total, 2), share=round(total / cj["duration"], 4),
              verify={"status": "ok", "model": verdict["model"], "keep_p": KEEP_P, "note": note, "note_raw": verdict.get("note"), "candidates": len(clips), "matched": len(hit),
                      "usage": verdict["usage"], "fake": bool(fake)})
    mf.write(run / "clips.json", cj)
    say("확인을 마쳤다", 1.0)
    return cj["verify"]


# =====================================================================  판정 모델이 스스로 다시 확인하기
def _ascii_ok(t):
    return isinstance(t, str) and bool(t.strip()) and len(t) <= 300 and all(32 <= ord(ch) <= 126 for ch in t)


def check_question(data):
    """확인 질문의 형식 검사(experiments/decision_api/da_write.py의 check와 같다)."""
    if not isinstance(data, dict):
        return ["the reply must be one JSON object"]
    errs = []
    if not _ascii_ok(data.get("viewer_request")):
        errs.append("viewer_request must be one short ASCII English phrase (max 300 characters)")
    conf = data.get("confusers")
    if not isinstance(conf, list) or not 2 <= len(conf) <= 4:
        errs.append("confusers must be a list of 2 to 4 items")
    keys = []
    for o in [data.get("target")] + (conf if isinstance(conf, list) else []):
        if not isinstance(o, dict) or not VQ_KEY_RE.match(str(o.get("key", ""))) or o.get("key") == "none_of_these":
            errs.append(f"each of target and confusers needs a key of lowercase letters, digits or underscores (max 40), not 'none_of_these': {str(o)[:60]}")
            continue
        keys.append(o["key"])
        if not _ascii_ok(o.get("is")):
            errs.append(f"{o['key']}: 'is' is empty, longer than 300 characters, or has non-ASCII characters")
    if len(set(keys)) != len(keys):
        errs.append("keys of target and confusers must all differ")
    n = data.get("noul")
    c = n.get("criteria") if isinstance(n, dict) else None
    if not isinstance(n, dict) or not _ascii_ok(n.get("instructions")) or not isinstance(c, dict) or set(c) != {"true", "false"} or not all(_ascii_ok(v) for v in c.values()):
        errs.append("noul needs ASCII 'instructions' and criteria with exactly 'true' and 'false' (each max 300 characters)")
    return errs


def write_question(name, request, facts=(), out=None, thinking="HIGH", fake=False):
    """확인 질문을 쓰게 한다(Gemini 4 Hydrogen, 영상과 그림은 주지 않는다). 받는 글: 안내서, verify_question_brief.md, 요청, 영상에 대한 사실.
    돌려주는 것: {"viewer_request", "target", "confusers", "noul", "understanding", "usage", …}. out 폴더에 question.json, prompt.txt, response_N.json을 남긴다."""
    out = Path(out) if out else mf.RUNS / name / "verify_question"
    out.mkdir(parents=True, exist_ok=True)
    if fake:
        spec = {**FAKE_QUESTION, "request": request}
        mf.write(out / "question.json", spec)
        return spec
    guide = (HERE.parent / "experiments" / "highlight_auto" / "writer" / "guide_for_writer.md").read_text(encoding="utf-8")
    brief = (HERE / "verify_question_brief.md").read_text(encoding="utf-8")
    fact_lines = "\n".join(f"- {f}" for f in facts) or "- (없음)"
    user = (f"# 안내서 (판정 모델에 보낼 요청을 짜는 법)\n\n{guide}\n\n---\n\n{brief}\n\n---\n\n# 이번 일\n\n"
            f"## 사용자의 요청 (그대로)\n\n{request}\n\n## 영상에 대한 사실\n\n{fact_lines}\n\n"
            f"## 검색 도구\n\n주어졌다.\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")
    (out / "prompt.txt").write_text(user, encoding="utf-8")
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": thinking}}, "tools": [{"googleSearch": {}}]}
    usage, queries, data, errs, resp = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}, [], None, ["(아직 부르지 않음)"], None
    for attempt in range(3):
        resp = W.call(body, W.PROJECT, W.MODEL)
        u = resp.get("usageMetadata", {})
        usage = {"prompt": usage["prompt"] + u.get("promptTokenCount", 0), "output": usage["output"] + u.get("candidatesTokenCount", 0),
                 "thoughts": usage["thoughts"] + u.get("thoughtsTokenCount", 0), "calls": usage["calls"] + 1}
        (out / f"response_{attempt + 1}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
        queries += W.grounding(resp)["queries"]
        text = W.text_of(resp)
        try:
            data = W.parse_json(text)
            errs = check_question(data)
        except Exception as e:
            data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
        if not errs:
            break
        body["contents"] += [{"role": "model", "parts": [{"text": text}]},
                             {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
    if errs:
        raise RuntimeError(f"확인 질문에 형식 오류가 남았다: {errs}")
    usd = usage["prompt"] / 1e6 * W.PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * W.PRICE_OUT
    spec = {"request": request, "facts": list(facts), "writer": f"{resp.get('modelVersion', W.MODEL)} (Vertex, thinking {thinking}, search on)",
            "search_queries": queries, "format_retries": usage["calls"] - 1, "usage": {**usage, "usd_estimate": round(usd, 4)},
            **{k: data.get(k) for k in ("understanding", "looked_up", "viewer_request", "target", "confusers", "noul")}}
    mf.write(out / "question.json", spec)
    return spec


def choice_questions(question):
    """고르기 질문 둘(선택지 순서만 반대). 선택지: 찾는 장면, 헷갈리는 것들, 어느 것도 아님. H2O는 끝 선택지 쪽으로 기울어서 두 순서를 함께 묻는다(README 10-3절)."""
    opts = [(question["target"]["key"], question["target"]["is"])] + [(c["key"], c["is"]) for c in question["confusers"]] + [("none_of_these", CHOICE_NONE)]
    return {"choice_fwd": {"type": "choice", "instructions": CHOICE_ASK, "criteria": dict(opts)},
            "choice_rev": {"type": "choice", "instructions": CHOICE_ASK, "criteria": dict(reversed(opts))}}


def verify_packs(cands, d, questions):
    """후보의 프레임을 요청 둘로 묶는다: a = 가장 확신한 1초의 연속 4장, b = 구간 전체에서 고르게 4장. 모양은 mf.py prepare의 묶음·요청 글과 같다(올려 둔 판정 모델 앱이 그대로 읽는다)."""
    out = {}
    for tag, lead, key in (("a", LEAD_A, "times_a"), ("b", LEAD_B, "times_b")):
        buf, pos = io.BytesIO(), []
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for c in cands:
                names = [f"c{c['id']}_{tag}{j}.jpg" for j in range(1, 5)]
                for n in names:
                    raw = (Path(d) / "frames" / n).read_bytes()
                    ti = tarfile.TarInfo(n)
                    ti.size = len(raw)
                    tar.addfile(ti, io.BytesIO(raw))
                pos.append({"k": c["id"], "frames": names, "seconds": c[key]})
        raw = buf.getvalue()
        out[tag] = (raw, {"state": {"about": lead}, "questions": questions, "pack_sha256": hashlib.sha256(raw).hexdigest(), "positions": pos})
    return out


def h2o_judge_remote(name, packs, target=None, app=None):
    """올려 둔 판정 모델 앱에 확인 요청 둘을 묻는다. 돌려주는 것: ({"a": {후보 id: {"noul": p, "choice": 찾는 장면이 받은 확률(두 순서의 평균) 또는 None}}, "b": {…}}, 앱이 준 요약).
    고르기 질문은 moment-finder-serve2만 받는다(먼저 올려 둔 moment-finder-serve는 예/아니오와 점수만 받는다)."""
    import modal
    vol = modal.Volume.from_name("mf-pipeline", create_if_missing=True)
    Judge = modal.Cls.from_name(app or os.environ.get("MF_SERVE_APP") or SERVE_APP_H2O, "Judge")
    ps, metas = {}, {}
    for tag, (raw, req) in packs.items():
        sub = f"{name}-v{tag}"
        with vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO(raw), f"/{sub}/pack.tar")
            b.put_file(io.BytesIO(json.dumps(req).encode()), f"/{sub}/requests.json")
        for ev in Judge().run.remote_gen(sub):
            if ev.get("stage") == "done":
                bad = [r["error"] for r in ev["records"] if "error" in r]
                if bad:
                    raise RuntimeError(f"판정 모델이 확인 요청에 답하지 못했다: {bad[0]}")
                ps[tag] = {}
                for r in ev["records"]:
                    a = r["answers"]
                    ch = [float(a[k]["probabilities"][target]) for k in ("choice_fwd", "choice_rev") if k in a] if target else []
                    ps[tag][r["k"]] = {"noul": float(a["verify"]["noul"]), "choice": sum(ch) / len(ch) if ch else None}
                metas[tag] = ev["meta"]
        if tag not in ps:
            raise RuntimeError("판정 모델이 확인 요청의 답을 끝까지 돌려주지 않았다")
    return ps, metas


def recheck_h2o(name, request=None, question=None, facts=(), judge=None, fake=False, on_stage=None, rule="both", app=None):
    """runs/<name>/의 답으로 후보를 뽑아 판정 모델에 확인 질문으로 다시 묻고 clips.json을 고쳐 쓴다. 돌려주는 것: clips.json의 verify.
    question: write_question이 돌려준 것(없으면 여기서 쓴다). judge: (name, packs, target, app) → (확률, 요약). 없으면 올려 둔 앱을 부른다.
    rule: "both" = 예/아니오 질문과 고르기 질문이 둘 다 0.5 이상이어야 통과(값은 둘 중 작은 것. 고르기를 받는 앱이 있어야 한다), "noul" = 예/아니오 질문만."""
    run = mf.RUNS / name
    say = on_stage or (lambda text, frac=0.0: None)
    cj = mf.read(run / "clips.json")
    base = cj.get("unverified") or cj["clips"]
    cands, info = candidates(run)
    if not cands:
        return mark_skipped(name, "후보가 없다")
    if not request:
        for f, k in (("job.json", "request"), ("writer.json", "request"), ("run.json", "viewer_request")):
            if (run / f).exists() and mf.read(run / f).get(k):
                request = mf.read(run / f)[k]
                break
    size = (info["width"], info["height"]) if "width" in info else (lambda p: (p["width"], p["height"]))(mf.probe(info["video"]))
    d = run / "verify"
    shutil.rmtree(d, ignore_errors=True)
    say(f"후보 {len(cands)}개의 프레임을 뽑는 중", 0.1)
    M.save_item(name, info["video"], request, info["duration"], size, cands, d=d)
    item = mf.read(d / "item.json")
    if question is None:
        say("확인 질문을 쓰는 중 (영상은 보여 주지 않는다)", 0.25)
        question = write_question(name, request, facts or [f"길이 {mf.mmss(info['duration'])}", f"화면 {size[0]}×{size[1]}"], fake=fake)
    noul = {"type": "noul", "instructions": question["noul"]["instructions"], "criteria": question["noul"]["criteria"]}
    both = rule == "both"
    questions = {"verify": noul, **(choice_questions(question) if both else {})}
    say(f"판정 모델이 후보 {len(cands)}개를 확인 질문으로 다시 보는 중 (4장씩 두 번)", 0.45)
    t0 = time.time()
    if fake:
        ps = {t: {c["id"]: {"noul": FAKE_P[i % len(FAKE_P)], "choice": FAKE_P[i % len(FAKE_P)] if both else None} for i, c in enumerate(cands)} for t in ("a", "b")}
        metas = {}
    else:
        ps, metas = (judge or h2o_judge_remote)(name, verify_packs(item["candidates"], d, questions), question["target"]["key"] if both else None, app)
    seconds = round(time.time() - t0, 1)
    rows = {}
    for c in cands:
        a, b = ps["a"][c["id"]], ps["b"][c["id"]]
        yn = (a["noul"] + b["noul"]) / 2
        ch = (a["choice"] + b["choice"]) / 2 if both else None
        rows[c["id"]] = {"id": c["id"], "p": round(min(yn, ch) if both else yn, 4), "p_noul": round(yn, 4), "p_choice": None if ch is None else round(ch, 4),
                         "noul_consecutive": round(a["noul"], 4), "noul_spread": round(b["noul"], 4),
                         "choice_consecutive": None if ch is None else round(a["choice"], 4), "choice_spread": None if ch is None else round(b["choice"], 4)}
    mf.write(d / "verdict.json", {"name": name, "by": "h2o", "rule": rule, "model": H2O_MODEL, "question": question, "seconds": seconds, "judge": metas, "fake": bool(fake),
                                  "candidates": [rows[c["id"]] for c in cands]})
    see = lambda r: (f"예/아니오 {r['p_noul']:.2f} · 고르기 {r['p_choice']:.2f}" if both else f"연속 4장 {r['noul_consecutive']:.2f} · 고르게 4장 {r['noul_spread']:.2f}")
    clips = [{**{k: c[k] for k in ("start", "end", "length", "peak_t", "peak", "why", "from")}, "cand": c["id"], "p": round(rows[c["id"]]["p"], 3),
              "see": see(rows[c["id"]]), "match": rows[c["id"]]["p"] >= KEEP_P} for c in cands]
    clips.sort(key=lambda c: (not c["match"], -c["p"], c["start"]))
    for r, c in enumerate(clips, 1):
        c["rank"] = r
    hit = [c for c in clips if c["match"]]
    total = union_length(hit)
    cj.update(clips=clips, unverified=base, total_length=round(total, 2), share=round(total / cj["duration"], 4),
              verify={"status": "ok", "by": "h2o", "rule": rule, "model": H2O_MODEL, "keep_p": KEEP_P, "note": None, "candidates": len(clips), "matched": len(hit),
                      "question": {"instructions": noul["instructions"], "criteria": noul["criteria"], "writer": question.get("writer"), "understanding": question.get("understanding"),
                                   "options": [{"key": k, "is": v} for k, v in questions["choice_fwd"]["criteria"].items()] if both else None},
                      "usage": question.get("usage"), "h2o_seconds": seconds, "fake": bool(fake)})
    mf.write(run / "clips.json", cj)
    say("확인을 마쳤다", 1.0)
    return cj["verify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fake", action="store_true", help="Gemini를 부르지 않고 꾸민 값으로(화면 확인용)")
    ap.add_argument("--h2o", action="store_true", help="Gemini 대신 판정 모델(H2O)이 스스로 다시 확인한다. 올려 둔 판정 모델 앱을 부른다(GPU 값이 든다)")
    ap.add_argument("--rule", default="both", choices=["both", "noul"], help="--h2o일 때: both = 예/아니오와 고르기가 둘 다 0.5 이상이어야 통과(moment-finder-serve2), noul = 예/아니오만")
    ap.add_argument("--app", default=None, help="--h2o일 때 부를 판정 모델 앱 이름(기본: MF_SERVE_APP 또는 moment-finder-serve2)")
    args = ap.parse_args()
    say = lambda text, frac=0.0: print("  ·", text, flush=True)
    v = recheck_h2o(args.name, fake=args.fake, on_stage=say, rule=args.rule, app=args.app) if args.h2o else recheck(args.name, fake=args.fake, on_stage=say)
    run = mf.RUNS / args.name
    if v["status"] == "ok":
        shutil.rmtree(run / "posters", ignore_errors=True)          # 순위가 바뀌었으니 순위로 이름 붙인 그림은 다시 만든다
        mf.stage_render(argparse.Namespace(name=args.name, top=0))
    cj = mf.read(run / "clips.json")
    print(f"{args.name}: {v}")
    for c in cj["clips"]:
        if "p" in c:
            print(f"  #{c['rank']} {mf.mmss(c['start'])}~{mf.mmss(c['end'])} p={c['p']:.2f} {'통과' if c['match'] else '낮음'} [{c['from']}] {c['see'][:110]}")


if __name__ == "__main__":
    main()
