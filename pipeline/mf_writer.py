"""중간 LLM: 사용자의 짧은 요청을 판정 모델에 물을 질문으로 바꾼다. 영상과 그림은 주지 않는다. pipeline/README.md 12절.

  Gemini 4 Hydrogen (gemini-4-hydrogen, 2026-11-12 공개)을 Vertex(Gemini Enterprise Agent Platform)의 generateContent로 부른다.
  인증은 이 Mac의 gcloud 로그인(access token), 프로젝트는 --project. 키나 토큰을 파일에 쓰지 않는다.
  --search를 주면 Google 검색 도구(tools: googleSearch)를 붙인다. 검색어와 출처는 답의 groundingMetadata에서 꺼내 함께 적는다.

중간 LLM이 받는 글: writer_brief.md(할 일과 형식), experiments/highlight_auto/writer/guide_for_writer.md(질문 쓰는 안내서. 실험 결과를 드러내는 두 항목을 지운 사본),
                  그리고 사용자의 요청과 영상에 대한 사실 몇 줄. 그 밖은 주지 않는다.
쓰는 것: --out 폴더에 criteria.json(파이프라인이 읽는 기준 파일), prompt.txt(보낸 글), response.json(받은 답 전체)

실행: python3 pipeline/mf_writer.py --name N --request "<요청>" [--fact "<사실 한 줄>" …] [--search] [--thinking HIGH]
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MODEL = "gemini-4-hydrogen"
PROJECT = "<gcp-project>"
PRICE_IN, PRICE_OUT = 1.50, 7.50          # 100만 토큰당 달러 (cloud.google.com/model-garden/pricing, 2026-11-16에 읽음)
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def check_questions(qs, lo=3, hi=5):
    """mf.py의 check_questions와 같은 규칙(질문 수만 3~5). mf.py는 numpy가 있어야 불러올 수 있어 여기에 따로 둔다."""
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


def call(body, project, model, timeout=600):
    token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
    if not token:
        raise RuntimeError("gcloud 로그인이 없다(gcloud auth print-access-token이 비었다)")
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Vertex 오류 HTTP {e.code}: {e.read()[:600].decode(errors='replace')}")


def text_of(resp):
    return "".join(p.get("text", "") for p in resp["candidates"][0].get("content", {}).get("parts", []) if not p.get("thought"))


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def grounding(resp):
    g = resp["candidates"][0].get("groundingMetadata") or {}
    src = []
    for ch in g.get("groundingChunks", []):
        w = ch.get("web") or {}
        if w:
            src.append({"title": w.get("title"), "domain": w.get("domain"), "uri": w.get("uri")})
    return {"queries": g.get("webSearchQueries", []), "sources": src}


SITE_EXTRA = ("\n\n## 덧붙임 (웹사이트용)\n\nJSON에 `viewer_request` 칸을 하나 더한다: 사용자의 요청을 영어 한 구절로 옮긴 것이다 "
              "(예: `the moments where the robot puts a candy into the cup`). 뜻을 넓히거나 좁히지 않는다. ASCII 글자만 쓴다.")


def write(name, request, facts=(), search=True, thinking="HIGH", project=PROJECT, model=MODEL, out=None, with_viewer_request=False, quiet=False):
    """중간 LLM에게 질문을 쓰게 하고 기준(dict)을 돌려준다. out 폴더에 criteria.json, prompt.txt, response_N.json을 남긴다."""
    out = Path(out) if out else HERE / "writer_runs" / name
    out.mkdir(parents=True, exist_ok=True)
    brief = (HERE / "writer_brief.md").read_text(encoding="utf-8") + (SITE_EXTRA if with_viewer_request else "")
    guide = (ROOT / "experiments" / "highlight_auto" / "writer" / "guide_for_writer.md").read_text(encoding="utf-8")
    fact_lines = "\n".join(f"- {f}" for f in facts) or "- (없음)"
    user = (f"# 안내서 (판정 모델에 보낼 요청을 짜는 법)\n\n{guide}\n\n---\n\n{brief}\n\n---\n\n# 이번 일\n\n"
            f"## 사용자의 요청 (그대로)\n\n{request}\n\n## 영상에 대한 사실\n\n{fact_lines}\n\n"
            f"## 검색 도구\n\n{'주어졌다.' if search else '주어지지 않았다. 검색하지 않는다.'}\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")
    (out / "prompt.txt").write_text(user, encoding="utf-8")
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": thinking}}}
    if search:
        body["tools"] = [{"googleSearch": {}}]
    usage = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}
    resp, data, errs, attempts = None, None, ["(아직 부르지 않음)"], []
    for attempt in range(3):
        resp = call(body, project, model)
        u = resp.get("usageMetadata", {})
        usage = {"prompt": usage["prompt"] + u.get("promptTokenCount", 0), "output": usage["output"] + u.get("candidatesTokenCount", 0),
                 "thoughts": usage["thoughts"] + u.get("thoughtsTokenCount", 0), "calls": usage["calls"] + 1}
        (out / f"response_{attempt + 1}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
        text = text_of(resp)
        try:
            data = parse_json(text)
            errs = check_questions(data.get("questions"))
            vr = data.get("viewer_request")
            if with_viewer_request and (not isinstance(vr, str) or not vr.strip() or len(vr) > 300 or any(ord(ch) > 126 or ord(ch) < 32 for ch in vr)):
                errs.append("viewer_request must be one short ASCII English phrase (max 300 characters)")
        except Exception as e:
            data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
        attempts.append({"attempt": attempt + 1, "format_errors": errs, "grounding": grounding(resp)})
        if not errs:
            break
        # 형식만 고치게 한다(내용은 바꾸지 말라고 한다). experiments/highlight_auto와 같은 방식
        body["contents"] += [{"role": "model", "parts": [{"text": text}]},
                             {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
    if errs:
        raise RuntimeError(f"중간 LLM의 답에 형식 오류가 남았다: {errs}")
    g = attempts[-1]["grounding"]
    all_queries = [q for a in attempts for q in a["grounding"]["queries"]]
    usd = usage["prompt"] / 1e6 * PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * PRICE_OUT
    crit = {"writer": f"{resp.get('modelVersion', model)} (Vertex, thinking {thinking}, search {'on' if search else 'off'})",
            "written_at": date.today().isoformat(), "request": request, "facts": list(facts), "search_tool": search,
            "search_queries": all_queries, "search_sources": g["sources"], "understanding": data.get("understanding"), "looked_up": data.get("looked_up"),
            "kind": data.get("kind"), "limits": data.get("limits"), "viewer_request": data.get("viewer_request"), "format_retries": len(attempts) - 1,
            "usage": {**usage, "usd_estimate": round(usd, 4)}, "questions": data["questions"]}
    (out / "criteria.json").write_text(json.dumps(crit, ensure_ascii=False, indent=1), encoding="utf-8")
    if not quiet:
        print(json.dumps({k: crit[k] for k in ("writer", "understanding", "looked_up", "kind", "limits", "search_queries", "format_retries", "usage")}, ensure_ascii=False, indent=1))
        print("출처", len(g["sources"]), "개:", [s.get("domain") or s.get("title") for s in g["sources"]][:12])
        for k, q in crit["questions"].items():
            print(f"\n[{k}] ({q['type']}) {q['instructions']}")
            if q["type"] == "noul":
                print("   예:", q["criteria"]["true"], "\n   아니오:", q["criteria"]["false"])
            else:
                for i, c in enumerate(q["criteria"]):
                    print(f"   {i}: {c}")
    return crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--request", required=True, help="사용자가 한 요청 그대로")
    ap.add_argument("--fact", action="append", default=[], help="영상에 대한 사실 한 줄(여러 번 줄 수 있다). 영상의 내용을 본 사람만 아는 것은 넣지 않는다")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--thinking", default="HIGH", choices=["LOW", "MEDIUM", "HIGH", "XHIGH"])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--viewer-request", dest="viewer_request", action="store_true")
    args = ap.parse_args()
    try:
        write(args.name, args.request, args.fact, args.search, args.thinking, args.project, args.model, args.out, args.viewer_request)
    except RuntimeError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
