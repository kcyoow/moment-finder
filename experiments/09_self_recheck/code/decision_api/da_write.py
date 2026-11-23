"""쓰는 쪽(Gemini 4 Hydrogen)에게 요청마다 확인 질문을 쓰게 한다. README.md 3절.

영상과 그림은 주지 않는다. 받는 글: 안내서(guide_for_writer.md), choice_brief.md, 사용자의 요청, 영상의 길이와 화면 크기.
Vertex 호출과 설정은 pipeline/mf_writer.py의 것을 그대로 쓴다(이 Mac의 gcloud 로그인, 생각 수준 HIGH, 검색 도구 붙임).

  python3 da_write.py --dry-run          # 보내지 않고 보낼 글의 길이만
  python3 da_write.py [--only ma,robot] [--budget 1.0]
쓰는 것: specs/<이름>.json (질문), specs/<이름>_prompt.txt, specs/<이름>_response_N.json
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import mf_writer as W  # noqa: E402

SPECS = HERE / "specs"
MC = ["영상의 길이 5:00", "화면 640x360 (가로)"]
REQS = {
    "ma": ("마크 영상인데 하이라이트 좀 뽑아줘", MC),
    "mb": ("마크 다이아 캐는 영상이야. 중요한 장면 뽑아줘", MC),
    "fight": ("마크 영상인데 싸우는 장면 뽑아줘", MC),
    "danger": ("마크 영상인데 다이아 캐거나 크리퍼 만나거나 위험한 장면 뽑아줘", MC),
    "robot": ("로봇이 사탕 컵에 넣는 거 성공한 장면 찾아줘", ["영상의 길이 1분~15분 (여러 편)", "화면 1080x1920 (세로)"]),
}
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def ascii_ok(t):
    return isinstance(t, str) and t.strip() and len(t) <= 300 and all(32 <= ord(ch) <= 126 for ch in t)


def check(data):
    if not isinstance(data, dict):
        return ["the reply must be one JSON object"]
    errs = []
    if not ascii_ok(data.get("viewer_request")):
        errs.append("viewer_request must be one short ASCII English phrase (max 300 characters)")
    opts = [data.get("target")] + (data.get("confusers") if isinstance(data.get("confusers"), list) else [])
    if not isinstance(data.get("confusers"), list) or not 2 <= len(data["confusers"]) <= 4:
        errs.append("confusers must be a list of 2 to 4 items")
    keys = []
    for o in opts:
        if not isinstance(o, dict) or not KEY_RE.match(str(o.get("key", ""))) or o.get("key") == "none_of_these":
            errs.append(f"each of target and confusers needs a key of lowercase letters, digits or underscores (max 40), not 'none_of_these': {str(o)[:60]}")
            continue
        keys.append(o["key"])
        if not ascii_ok(o.get("is")):
            errs.append(f"{o['key']}: 'is' is empty, longer than 300 characters, or has non-ASCII characters")
    if len(set(keys)) != len(keys):
        errs.append("keys of target and confusers must all differ")
    n = data.get("noul")
    c = n.get("criteria") if isinstance(n, dict) else None
    if not isinstance(n, dict) or not ascii_ok(n.get("instructions")) or not isinstance(c, dict) or set(c) != {"true", "false"} or not all(ascii_ok(v) for v in c.values()):
        errs.append("noul needs ASCII 'instructions' and criteria with exactly 'true' and 'false' (each max 300 characters)")
    return errs


def prompt_of(request, facts):
    guide = (ROOT / "experiments" / "highlight_auto" / "writer" / "guide_for_writer.md").read_text(encoding="utf-8")
    brief = (HERE / "choice_brief.md").read_text(encoding="utf-8")
    fact_lines = "\n".join(f"- {f}" for f in facts)
    return (f"# 안내서 (판정 모델에 보낼 요청을 짜는 법)\n\n{guide}\n\n---\n\n{brief}\n\n---\n\n# 이번 일\n\n"
            f"## 사용자의 요청 (그대로)\n\n{request}\n\n## 영상에 대한 사실\n\n{fact_lines}\n\n"
            f"## 검색 도구\n\n주어졌다.\n\n위의 '내놓을 것' 모양의 JSON 하나만 낸다.")


def write_one(name, thinking="HIGH"):
    request, facts = REQS[name]
    user = prompt_of(request, facts)
    (SPECS / f"{name}_prompt.txt").write_text(user, encoding="utf-8")
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 32768, "thinkingConfig": {"thinkingLevel": thinking}}, "tools": [{"googleSearch": {}}]}
    usage, queries, data, errs, resp = {"prompt": 0, "output": 0, "thoughts": 0, "calls": 0}, [], None, ["(아직 부르지 않음)"], None
    for attempt in range(3):
        resp = W.call(body, W.PROJECT, W.MODEL)
        u = resp.get("usageMetadata", {})
        usage = {"prompt": usage["prompt"] + u.get("promptTokenCount", 0), "output": usage["output"] + u.get("candidatesTokenCount", 0),
                 "thoughts": usage["thoughts"] + u.get("thoughtsTokenCount", 0), "calls": usage["calls"] + 1}
        (SPECS / f"{name}_response_{attempt + 1}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
        queries += W.grounding(resp)["queries"]
        text = W.text_of(resp)
        try:
            data = W.parse_json(text)
            errs = check(data)
        except Exception as e:
            data, errs = None, [f"the reply is not one valid JSON object: {type(e).__name__}"]
        if not errs:
            break
        body["contents"] += [{"role": "model", "parts": [{"text": text}]},
                             {"role": "user", "parts": [{"text": "The reply has format errors. Fix only the format and keep the content. Return the whole JSON again.\n- " + "\n- ".join(errs)}]}]
    if errs:
        raise RuntimeError(f"{name}: 형식 오류가 남았다: {errs}")
    usd = usage["prompt"] / 1e6 * W.PRICE_IN + (usage["output"] + usage["thoughts"]) / 1e6 * W.PRICE_OUT
    spec = {"name": name, "request": request, "facts": facts, "writer": f"{resp.get('modelVersion', W.MODEL)} (Vertex, thinking {thinking}, search on)",
            "search_queries": queries, "format_retries": usage["calls"] - 1, "usage": {**usage, "usd_estimate": round(usd, 4)},
            **{k: data.get(k) for k in ("understanding", "looked_up", "viewer_request", "target", "confusers", "noul")}}
    (SPECS / f"{name}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    SPECS.mkdir(exist_ok=True)
    names = [n for n in REQS if not args.only or n in args.only.split(",")]
    todo = [n for n in names if not (SPECS / f"{n}.json").exists()]
    spent = sum(json.loads((SPECS / f"{n}.json").read_text())["usage"]["usd_estimate"] for n in REQS if (SPECS / f"{n}.json").exists())
    print(f"부를 것 {len(todo)}개 {todo} (이미 있는 것 {len(names) - len(todo)}개, 지금까지 ${spent:.2f})")
    if args.dry_run:
        for n in todo:
            print(f"  {n}: 보낼 글 {len(prompt_of(*REQS[n])):,}자 | {REQS[n][0]}")
        return
    with ThreadPoolExecutor(5) as ex:
        for n, fut in [(n, ex.submit(write_one, n)) for n in todo]:
            try:
                s = fut.result()
            except Exception as e:
                print(f"  {n}: 실패 {type(e).__name__}: {str(e)[:300]}")
                continue
            spent += s["usage"]["usd_estimate"]
            print(f"\n== {n} | {s['request']} | 검색 {len(s['search_queries'])}번, 다시 쓰기 {s['format_retries']}번, ${s['usage']['usd_estimate']}")
            print("   이해:", s["understanding"])
            print("   viewer_request:", s["viewer_request"])
            print(f"   target [{s['target']['key']}]: {s['target']['is']}")
            for c in s["confusers"]:
                print(f"   confuser [{c['key']}]: {c['is']}")
            print("   noul:", s["noul"]["instructions"], "| 예:", s["noul"]["criteria"]["true"], "| 아니오:", s["noul"]["criteria"]["false"])
    print(f"\n합계 ${spent:.2f}" + ("  (멈추는 선을 넘었다)" if spent > args.budget else ""))


if __name__ == "__main__":
    main()
