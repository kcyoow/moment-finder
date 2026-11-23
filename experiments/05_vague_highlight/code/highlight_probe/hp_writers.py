"""highlight_probe B 단계: 글쓴이의 팔마다 안내와 입력 폴더를 만들고, 받은 기준을 검사한다 (이 Mac, 돈 없음).

단계
  build    : writers/<팔>/ 에 WRITER_BRIEF.md, writer_input.json, guide_for_writer.md 를 만든다. data/p_id_map.json(P 팔의 가린 id), writers/manifest.json(sha256)
  check    : writers/<팔>/criteria.json 의 형식 검사(내용은 보지 않는다)와 입력 파일이 그대로인지(sha256)
  keywords : README 8절에서 미리 정한 낱말 확인(마인크래프트 G, GP, P)
  cost     : 실제 질문 수로 본 실행(C 단계)의 시간과 비용을 다시 센다 → data/main_run_cost.json

실행: uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_probe/hp_writers.py <단계>

팔(README 6절)
  T    TVSum 50개: 제목, 범주, 길이
  TP   TVSum 50개: 제목, 범주, 길이 + 훑어보기 요약
  P50  TVSum 50개: 길이 + 훑어보기 요약만. id는 v01~v50(섞음). 안내에 자료 이름을 적지 않는다
  G    마인크래프트: 게임 설명 한 줄과 요청. 12개 영상에 한 벌
  GP   마인크래프트 12개: 게임 설명 한 줄과 요청 + 영상마다의 요약. 영상마다 한 벌
  P12  마인크래프트 12개: 영상마다의 요약만. id는 v01~v12(섞음). 안내에 게임 이름을 적지 않는다
  (P50, P12 의 안내서 사본은 3절의 보기 한 곳에 나오는 게임 이름만 가렸다. 나머지 네 팔은 바이트 그대로다)

안내의 글은 highlight_auto/writer/WRITER_BRIEF.md 에서 옮겼고, 팔마다 다른 곳(받는 것, 요청, 상태, 내놓을 id)만 바꿨다.
안내서(guide_for_writer.md)는 highlight_auto 의 것을 복사한다(T, TP, G, GP 는 바이트 그대로).
다른 실험 폴더에는 쓰지 않는다. 기준의 내용은 고치거나 고르지 않는다.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import hashlib  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hp_prepare as H  # noqa: E402  (읽기·쓰기 도우미, 속도·값의 식, highlight_auto 경로)

WRITERS = HERE / "writers"
ID_MAP = HERE / "data" / "p_id_map.json"
ARMS = ["T", "TP", "P50", "G", "GP", "P12"]
TV_ARMS, MC_ARMS = ["T", "TP", "P50"], ["G", "GP", "P12"]
SEED_TV, SEED_MC = 20261115, 20261116
WORDS = ["craft", "tool", "pickaxe", "inventory", "furnace", "workbench", "smelt"]     # README 8절
REQ_TV_GENERIC = "the most important moments of this video"
REQ_MC_GENERIC = "the most exciting and important moments of this video"
ABOUT = "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first)."
KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    wi = H.read(H.HA / "writer" / "writer_input.json")
    if wi.get("status") != "final":
        raise SystemExit("highlight_auto 의 writer_input.json 이 final 이 아니다")
    prof = H.read(HERE / "data" / "profiles.json")
    tv = wi["tvsum"]
    mc_ids = sorted(k for k in prof if k.startswith("mc_"))
    if len(tv) != 50 or len(mc_ids) != 12 or any(x["id"] not in prof for x in tv):
        raise SystemExit("영상 수가 다르거나 요약이 빠졌다")
    return wi, prof, tv, mc_ids


def duration_of(text):
    m = re.search(r"over this (\d+:\d\d) video", text)
    if not m:
        raise SystemExit("요약 글의 머리말에서 길이를 찾지 못했다")
    mm, ss = m.group(1).split(":")
    return m.group(1), int(mm) * 60 + int(ss)


def id_map(tv, mc_ids):
    a = [x["id"] for x in tv]
    random.Random(SEED_TV).shuffle(a)
    b = list(mc_ids)
    random.Random(SEED_MC).shuffle(b)
    return {"what": "P 팔에서 글쓴이에게 보여 준 id → 실제 id. 글쓴이의 폴더 밖에 둔다. 순서는 고정한 씨앗으로 섞었다",
            "seeds": {"P50": SEED_TV, "P12": SEED_MC},
            "P50": {f"v{i + 1:02d}": v for i, v in enumerate(a)},
            "P12": {f"v{i + 1:02d}": v for i, v in enumerate(b)}}


def inputs(arm, wi, prof, tv, mc_ids, idm):
    mc = wi["minecraft"]
    if arm == "T":
        return {"status": "final", "status_note": wi["status_note"], "tvsum": tv}
    if arm == "TP":
        return {"status": "final", "status_note": wi["status_note"],
                "tvsum": [{**x, "probe_summary": prof[x["id"]]["text"]} for x in tv]}
    if arm == "P50":
        by = {x["id"]: x for x in tv}
        return {"status": "final", "viewer_request": REQ_TV_GENERIC,
                "videos": [{"id": k, "duration": by[v]["duration"], "duration_seconds": by[v]["duration_seconds"], "probe_summary": prof[v]["text"]}
                           for k, v in idm["P50"].items()]}
    if arm == "G":
        return {"status": "final", "minecraft": mc}
    if arm == "GP":
        vids = []
        for v in mc_ids:
            d, s = duration_of(prof[v]["text"])
            vids.append({"id": v, "duration": d, "duration_seconds": s, "probe_summary": prof[v]["text"]})
        return {"status": "final", "game": mc["game"], "viewer_request": mc["viewer_request"], "videos": vids}
    if arm == "P12":
        vids = []
        for k, v in idm["P12"].items():
            d, s = duration_of(prof[v]["text"])
            vids.append({"id": k, "duration": d, "duration_seconds": s, "probe_summary": prof[v]["text"]})
        return {"status": "final", "viewer_request": REQ_MC_GENERIC, "videos": vids}
    raise SystemExit(arm)


def want_ids(arm, tv, mc_ids, idm):
    return {"T": [x["id"] for x in tv], "TP": [x["id"] for x in tv], "P50": list(idm["P50"]),
            "G": ["minecraft"], "GP": list(mc_ids), "P12": list(idm["P12"])}[arm]


# ---------------------------------------------------------------------  안내 글
SUMMARY_SECTION = """## 훑어보기 요약 (`probe_summary`)

- 무엇인가: 판정 모델이 그 영상의 위치 몇 곳(영상마다 12~30곳, 처음부터 끝까지 고르게)을 먼저 보고, 모든 영상에 똑같이 묻는 일반 질문 9개에 답한 것을 코드가 글로 옮긴 것이다. 사람이 쓴 글이 아니고, 영상에 맞춰 고른 질문도 아니다
- 수치는 그 위치들에서의 확률 평균이다. 대략의 힌트이지 사실이 아니다. 틀릴 수 있다
- 정해 둔 질문 밖의 내용(무엇이 나오는지의 구체적인 이름 등)은 요약에 나오지 않는다. 본 위치 사이에 지나간 장면도 나오지 않는다
- 요약을 어떻게 쓸지는 네가 정한다
"""

ANSWERS_SECTION = """## 답을 어떻게 쓰나 (정해져 있다)

- 예/아니오(noul) 질문은 "예"일 확률을, 점수(score) 질문은 단계의 기댓값을 쓴다
- 한 영상 안에서 질문마다 그 값을 순위로 바꾼다. 그리고 질문들의 순위를 **같은 무게로 평균** 낸다. 높을수록 그 요청에 더 맞는 순간으로 본다
- 그래서 지킬 것:
  - **모든 질문은 "예" 또는 높은 단계가 "이 요청에 더 맞음"이 되게 쓴다.** 반대 방향 질문(예: "지루한 장면인가")은 쓰지 않는다
  - **같은 것을 두 번 묻지 않는다.** 같은 무게로 평균 내므로, 두 번 물으면 그 면이 두 배 무게를 얻는다
  - 자르는 기준 값은 쓰지 않는다. 순위만 쓰기 때문이다(안내서 6절의 4번 "답을 읽는 법"과 2번 "보여 줄 것"은 이 실험에서 정해져 있으니 쓰지 않는다)
  - 그 영상의 거의 모든 위치에서 "예"이거나 거의 모든 위치에서 "아니오"일 질문은 순위를 가르지 못한다
"""

RULES_HEAD = """## 질문의 규칙

- {unit} 질문 **3개 이상 6개 이하**
- 종류는 두 가지만 쓴다
  - `noul`(예/아니오): `criteria`에 `true`와 `false` 설명을 **둘 다** 쓴다. 헷갈릴 만한 것의 이름은 `false` 쪽에 적는다(안내서 2절)
  - `score`(정도): `criteria`에 단계 설명을 **3~5개**, 낮은 단계(덜 맞음)부터 적는다. 첫 단계는 "해당 없음"에 해당하게 쓴다
  - `choice`와 `box`는 쓰지 않는다
- `instructions`와 모든 설명은 **영어**로 쓴다. 한국어를 쓰지 않는다
- 질문 하나에 판단 하나(안내서 2절)
- 그림에서 눈에 보이는 것으로 쓴다. 소리, 말, 음악, 대사는 묻지 않는다
- 움직임이나 변화를 묻는 질문은 4장을 함께 보는 문장으로 쓴다. 안내서 2절의 "4장 함께" 틀이 기본이다
- 엄한 말(clearly, strict, "default answer is No" 같은 것)을 넣지 않는다(안내서 2절, 7절)
- 길이: `instructions` 300자 이하, 설명 한 줄마다 300자 이하
- 질문의 이름(키)은 판정 모델이 보지 못한다. 문장만으로 뜻이 통하게 쓴다. 키는 영어 소문자, 숫자, 밑줄만 쓰고 40자 이하로 한다. `worth`라는 키는 쓰지 않는다
- {vague}
"""

FRAMES_JSON = """ "about": "`frames` holds 4 consecutive frames from the video, 0.5 seconds apart, in time order (earliest first).",
 "frames": [{"seconds": 10.25, "frame": "<그림>"}, {"seconds": 10.75, "frame": "<그림>"},
            {"seconds": 11.25, "frame": "<그림>"}, {"seconds": 11.75, "frame": "<그림>"}]}"""


def brief(arm, n_ids, mc):
    tvsum, summary = arm in TV_ARMS, arm in ("TP", "P50", "GP", "P12")
    L = ["# 기준 쓰기 안내 (criteria writer에게)", "", "## 너의 일", ""]
    if arm == "G":
        L += ["판정 모델에 물을 구체적인 질문을 3~6개 쓴다. 같은 종류의 영상 12개에 똑같이 쓸 질문 **한 벌**이다."]
    elif arm in ("GP", "P12"):
        L += [f"영상마다, 판정 모델에 물을 구체적인 질문을 3~6개 쓴다. 영상은 {n_ids}개다. 영상마다 한 벌씩 쓴다(영상끼리 같아야 할 필요도, 달라야 할 필요도 없다)."]
    else:
        L += [f"영상마다, 판정 모델에 물을 구체적인 질문을 3~6개 쓴다. 영상은 {n_ids}개다."]
    L += ["결과는 이 문서와 같은 폴더의 한 파일 `criteria.json`에 쓴다. 형식은 아래 \"내놓을 것\"을 따른다.", "",
          "## 네가 받는 것 (이것만 쓴다)", ""]
    got = {"T": "`writer_input.json`: 영상마다 id, 제목(title), 범주(category), 길이(duration)",
           "TP": "`writer_input.json`: 영상마다 id, 제목(title), 범주(category), 길이(duration), 훑어보기 요약(probe_summary)",
           "P50": "`writer_input.json`: 영상마다 id, 길이(duration), 훑어보기 요약(probe_summary). 제목과 범주는 주어지지 않는다. 시청자의 요청(viewer_request)은 모든 영상에 같다",
           "G": "`writer_input.json`: 시청자의 요청(viewer_request)과 게임 설명(game) 한 줄씩뿐이다",
           "GP": "`writer_input.json`: 시청자의 요청(viewer_request)과 게임 설명(game) 한 줄씩, 그리고 영상마다 id, 길이(duration), 훑어보기 요약(probe_summary)",
           "P12": "`writer_input.json`: 영상마다 id, 길이(duration), 훑어보기 요약(probe_summary). 무슨 영상인지는 주어지지 않는다. 시청자의 요청(viewer_request)은 모든 영상에 같다"}[arm]
    L += [f"- {got}",
          "- `guide_for_writer.md`: 판정 모델에 보낼 요청을 짜는 안내서. **이 안내서를 따른다**",
          "- 이 문서", "",
          "이 폴더의 이 세 파일 말고는 어떤 파일도 열지 않는다(이 폴더 밖의 파일, 폴더 목록도 보지 않는다). 인터넷에서 영상이나 제목을 찾아보지 않는다. 영상을 보지 않는다.",
          "너는 이 실험의 결과, 정답, 다른 사람이 쓴 기준을 보지 못했다. 찾아보지도 않는다.",
          "`writer_input.json`의 `status`가 `\"final\"`이 아니면 일을 시작하지 않고 그렇게 알린다.", ""]
    if summary:
        L += [SUMMARY_SECTION]
    L += ["## 시청자의 요청", ""]
    if arm in ("T", "TP"):
        L += ["- `the most important moments of the video titled \"<title>\"` (`<title>`은 writer_input.json의 제목 그대로)"]
    elif arm == "P50":
        L += [f"- `{REQ_TV_GENERIC}` (writer_input.json의 `viewer_request` 그대로. 모든 영상에 같다)"]
    elif arm == "P12":
        L += [f"- `{REQ_MC_GENERIC}` (writer_input.json의 `viewer_request` 그대로. 모든 영상에 같다)"]
    else:
        L += ["- writer_input.json의 `viewer_request` 그대로"]
    L += ["", "네 질문은 이 요청을, 그림에서 눈에 보이는 좁은 기준 몇 개로 나눈 것이어야 한다.",
          "요청에 없는 조건으로 장면을 좁히지 않는다(안내서 2절).", "",
          "## 판정 모델이 보는 것", "",
          "판정 모델은 H2O-Lightning-4B다. 글을 쓰지 못한다. 질문마다 확률만 낸다. 질문끼리 서로 보지 못한다. **소리를 듣지 못한다.**", "",
          "위치 하나에 요청 하나를 보낸다. 요청 하나에는 그림 4장과 글 몇 줄(상태, state)이 들어 있고, 질문 여러 개를 함께 묻는다.", ""]
    if tvsum:
        L += ["- 그림: 그 위치를 가운데에 두고 0.5초 간격으로 뽑은 4장(약 1.5초 폭), 시간 순서. 원본 크기이고 긴 변이 640px 이하다(480×360~640×360)",
              "- 2초마다 위치 하나. 상태는 아래와 같다", "", "```json"]
        if arm == "P50":
            L += ['{"viewer_request": "' + REQ_TV_GENERIC + '",']
        else:
            L += ['{"viewer_request": "the most important moments of the video titled \\"<title>\\"",']
        L += [FRAMES_JSON, "```", ""]
        hidden = "`viewer_request`, `about`"
    else:
        L += ["- 그림: 그 위치를 가운데에 두고 0.5초 간격으로 뽑은 4장(약 1.5초 폭), 시간 순서. 640×360이다",
              "- 1초마다 위치 하나. 상태는 아래와 같다", "", "```json"]
        if arm == "P12":
            L += ['{"viewer_request": "' + REQ_MC_GENERIC + '",']
            hidden = "`viewer_request`, `about`"
        else:
            L += ['{"game": "' + mc["game"] + '",', ' "viewer_request": "' + mc["viewer_request"] + '",']
            hidden = "`viewer_request`, `game`, `about`"
        L += [FRAMES_JSON, "```", ""]
    L += [f"**네 질문은 `frames`만 가리킨다.** {hidden}을 질문에서 가리키지 않는다.", "", ANSWERS_SECTION]
    unit = "질문 한 벌에" if arm == "G" else "영상마다"
    vague = {"T": "제목이 모호해서 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다",
             "TP": "제목이 모호해서 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다",
             "P50": "받은 것만으로 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다",
             "G": "요청이 모호해서 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다",
             "GP": "요청이 모호해서 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다",
             "P12": "받은 것만으로 기준의 종류를 가를 수 없으면, 물어볼 수 없으니 가장 그럴듯한 뜻으로 쓰고 그 사실을 `limits`에 적는다"}[arm]
    L += [RULES_HEAD.format(unit=unit, vague=vague)]
    key = '"minecraft"' if arm == "G" else '"<영상 id>"'
    L += ["## 내놓을 것: `criteria.json` (이 폴더에)", "", "```json", "{",
          ' "writer_model": "<모델 이름과 판>",', ' "written_at": "<YYYY-MM-DD>",', ' "videos": {',
          f"  {key}: {{",
          '   "kind": "<기준의 종류와 그렇게 본 이유, 한두 문장 (안내서 6절의 1번)>",',
          '   "questions": {',
          '    "<key>": {"type": "noul",',
          '              "instructions": "<영어 문장, `frames`를 가리킴>",',
          '              "criteria": {"true": "<예일 때>", "false": "<아니오일 때, 헷갈리는 것의 이름>"}},',
          '    "<key>": {"type": "score",',
          '              "instructions": "<영어 문장, `frames`를 가리킴>",',
          '              "criteria": ["<단계 0: 해당 없음>", "<단계 1>", "<단계 2>", "<단계 3>"]}',
          "   },",
          '   "limits": "<이 기준에서 놓칠 만한 것, 한두 문장 (안내서 6절의 5번)>"',
          "  }", " }", "}", "```", ""]
    if arm == "G":
        L += ["- `videos`에는 `minecraft` 하나만 있어야 한다"]
    else:
        L += [f"- `videos`에는 writer_input.json의 id {n_ids}개가 모두 있어야 한다(id는 받은 그대로 쓴다)"]
    L += ["- 파일은 올바른 JSON이어야 한다(주석 없음)",
          "- 받은 쪽은 형식만 검사한다(질문 수, 종류, 칸, 길이, 영어, `frames`를 가리키는지). 형식이 틀리면 틀린 곳의 목록을 한 번 돌려준다. 그때는 형식만 고치고 내용은 바꾸지 않는다", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------  build
GUIDE_EXAMPLE = '(예: `"game": "Minecraft survival gameplay, first-person view"`)'
GUIDE_EXAMPLE_HIDDEN = "(예: 무슨 영상인지 적은 한 줄)"


def guide_for(arm, guide):
    """안내서는 highlight_auto 의 것을 바이트 그대로 쓴다. 다만 P 팔(P50, P12)의 사본에서는 3절의 보기 한 곳에 나오는 게임 이름을 가린다.
    P 팔은 무슨 영상인지 알려 주지 않는 팔인데, 그 보기가 마인크래프트를 알려 주기 때문이다. 글쓴이를 띄우기 전에 정했다(2026-11-15)."""
    if arm not in ("P50", "P12"):
        return guide
    text = guide.decode("utf-8")
    if text.count(GUIDE_EXAMPLE) != 1:
        raise SystemExit("안내서에서 가릴 보기를 정확히 한 곳 찾지 못했다")
    return text.replace(GUIDE_EXAMPLE, GUIDE_EXAMPLE_HIDDEN).encode("utf-8")


def stage_build(args):
    wi, prof, tv, mc_ids = sources()
    idm = id_map(tv, mc_ids)
    H.write(ID_MAP, idm)
    guide = (H.HA / "writer" / "guide_for_writer.md").read_bytes()
    man = {"what": "글쓴이에게 준 파일의 sha256. check 단계가 그대로인지 본다", "guide_source": "experiments/highlight_auto/writer/guide_for_writer.md",
           "guide_sha256": hashlib.sha256(guide).hexdigest(), "arms": {}}
    for arm in ARMS:
        d = WRITERS / arm
        if (d / "criteria.json").exists() and not args.force:
            raise SystemExit(f"{arm}: criteria.json 이 이미 있다. 입력을 다시 만들지 않는다(--force 로만)")
        d.mkdir(parents=True, exist_ok=True)
        ids = want_ids(arm, tv, mc_ids, idm)
        H.write(d / "writer_input.json", inputs(arm, wi, prof, tv, mc_ids, idm))
        (d / "WRITER_BRIEF.md").write_text(brief(arm, len(ids), wi["minecraft"]), encoding="utf-8")
        (d / "guide_for_writer.md").write_bytes(guide_for(arm, guide))
        man["arms"][arm] = {"ids": len(ids), **{f: sha(d / f) for f in ("WRITER_BRIEF.md", "writer_input.json", "guide_for_writer.md")}}
        print(arm, len(ids), "개,", (d / "writer_input.json").stat().st_size, "바이트")
    H.write(WRITERS / "manifest.json", man)
    leak_check(wi, tv)


def leak_check(wi, tv):
    """P 팔의 폴더에 제목, 범주, 실제 id, 자료 이름, 게임 이름이 없는지 본다."""
    bad = []
    names = [x["title"] for x in tv] + sorted({x["category"] for x in tv}) + [x["id"] for x in tv]
    for arm, extra in (("P50", ["TVSum", "tvsum", "Minecraft", "minecraft"]), ("P12", ["Minecraft", "minecraft", "mc_0", "mc_1", "TVSum", "tvsum"])):
        for f in ("WRITER_BRIEF.md", "writer_input.json", "guide_for_writer.md"):
            text = (WRITERS / arm / f).read_text(encoding="utf-8")
            for n in names + extra:
                if n in text:
                    bad.append(f"{arm}/{f}: {n!r}")
    for arm in ("T", "G"):
        if "probe_summary" in (WRITERS / arm / "writer_input.json").read_text(encoding="utf-8"):
            bad.append(f"{arm}: 요약이 들어 있다")
    print("새는 곳:", bad if bad else "없음")
    if bad:
        raise SystemExit(1)


# ---------------------------------------------------------------------  check
def check_criteria(crit, want):
    """highlight_auto/ha_prepare.py 의 check_criteria 와 같은 규칙. 영상 id의 집합만 인자로 받는다. 내용은 보지 않는다."""
    errs = []
    if not isinstance(crit, dict) or not isinstance(crit.get("videos"), dict):
        return ["맨 위에 videos 객체가 없다"]
    for k in ("writer_model", "written_at"):
        if not crit.get(k):
            errs.append(f"맨 위의 {k}가 없다")
    want, got = set(want), set(crit["videos"])
    if want - got:
        errs.append(f"빠진 영상: {sorted(want - got)}")
    if got - want:
        errs.append(f"모르는 영상: {sorted(got - want)}")
    for vid, e in crit["videos"].items():
        qs = e.get("questions") if isinstance(e, dict) else None
        if not isinstance(qs, dict):
            errs.append(f"{vid}: questions가 없다")
            continue
        if not 3 <= len(qs) <= 6:
            errs.append(f"{vid}: 질문 수 {len(qs)} (3~6이어야 함)")
        for key, q in qs.items():
            where = f"{vid}.{key}"
            if not KEY_RE.match(key) or key == "worth":
                errs.append(f"{where}: 키는 영어 소문자·숫자·밑줄 40자 이하, 'worth'는 쓸 수 없다")
            if not isinstance(q, dict) or q.get("type") not in ("noul", "score"):
                errs.append(f"{where}: type은 noul 또는 score")
                continue
            ins = q.get("instructions", "")
            texts = [ins]
            if not isinstance(ins, str) or not ins.strip() or len(ins) > 300:
                errs.append(f"{where}: instructions가 비었거나 300자를 넘는다")
            if not isinstance(ins, str) or "`frames`" not in ins:
                errs.append(f"{where}: instructions가 `frames`를 가리키지 않는다")
            for bad in ("`viewer_request`", "`game`", "`about`"):
                if isinstance(ins, str) and bad in ins:
                    errs.append(f"{where}: instructions가 {bad}를 가리킨다")
            c = q.get("criteria")
            if q["type"] == "noul":
                if not isinstance(c, dict) or set(c) != {"true", "false"} or not all(isinstance(x, str) and x.strip() for x in c.values()):
                    errs.append(f"{where}: noul의 criteria는 true와 false 설명 둘")
                else:
                    texts += list(c.values())
            else:
                if not isinstance(c, list) or not 3 <= len(c) <= 5 or not all(isinstance(x, str) and x.strip() for x in c):
                    errs.append(f"{where}: score의 criteria는 단계 설명 3~5개의 목록")
                else:
                    texts += c
            for t in texts:
                if isinstance(t, str) and (len(t) > 300 or any(ord(ch) > 126 or (ord(ch) < 32) for ch in t)):
                    errs.append(f"{where}: 300자를 넘거나 영어(ASCII) 밖의 글자가 있다: {t[:60]!r}")
                    break
    return errs


def load_all():
    wi, prof, tv, mc_ids = sources()
    idm = H.read(ID_MAP)
    out = {}
    for arm in ARMS:
        p = WRITERS / arm / "criteria.json"
        out[arm] = (H.read(p) if p.exists() else None, want_ids(arm, tv, mc_ids, idm))
    return out, idm


def stage_check(args):
    from collections import Counter
    man = H.read(WRITERS / "manifest.json")
    allc, _ = load_all()
    ok = True
    for arm in ARMS:
        crit, want = allc[arm]
        moved = [f for f in ("WRITER_BRIEF.md", "writer_input.json", "guide_for_writer.md") if sha(WRITERS / arm / f) != man["arms"][arm][f]]
        extra = sorted(p.name for p in (WRITERS / arm).iterdir() if p.name not in ("WRITER_BRIEF.md", "writer_input.json", "guide_for_writer.md", "criteria.json"))
        if crit is None:
            print(f"{arm}: criteria.json 없음")
            ok = False
            continue
        errs = check_criteria(crit, want)
        n = Counter(len(v["questions"]) for v in crit["videos"].values() if isinstance(v.get("questions"), dict))
        types = Counter(q.get("type") for v in crit["videos"].values() for q in v.get("questions", {}).values() if isinstance(q, dict))
        print(f"{arm}: 영상 {len(crit['videos'])}, 질문 수 분포 {dict(sorted(n.items()))}, 종류 {dict(types)}, 글쓴이 {crit.get('writer_model')!r}, 형식 오류 {len(errs)}, 바뀐 입력 {moved}, 그 밖의 파일 {extra}")
        for e in errs[:40]:
            print("   -", e)
        ok = ok and not errs and not moved
    print("모두 통과" if ok else "통과하지 못한 팔이 있다")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------  keywords
def texts_of(entry):
    out = []
    for q in entry["questions"].values():
        out.append(q["instructions"])
        c = q["criteria"]
        out += list(c.values()) if isinstance(c, dict) else list(c)
    return " ".join(out).lower()


def stage_keywords(args):
    allc, _ = load_all()
    res = {"what": "README 8절의 미리 정한 낱말 확인. 질문 문장(instructions)과 설명(criteria)에서만 찾는다. 대소문자 무시, 부분 일치", "words": WORDS, "arms": {}}
    for arm in MC_ARMS:
        crit, _ = allc[arm]
        if crit is None:
            continue
        per = {vid: [w for w in WORDS if w in texts_of(e)] for vid, e in crit["videos"].items()}
        res["arms"][arm] = {"videos": len(per), "videos_with_any": sum(1 for v in per.values() if v), "per_video": per}
        print(f"{arm}: {res['arms'][arm]['videos_with_any']} / {len(per)}  ", {k: v for k, v in per.items() if v})
    H.write(HERE / "data" / "keyword_check.json", res)


# ---------------------------------------------------------------------  cost
def stage_cost(args):
    allc, idm = load_all()
    P = H.P
    tv_items = H.read(H.HA / "data" / "tvsum_items.json")["videos"]
    fsum = H.read(H.HA / "data" / "frames_summary.json")["packs"]
    q_tv = {vid: 0 for vid in tv_items}
    per_arm = {}
    for arm in TV_ARMS:
        crit, _ = allc[arm]
        if crit is None:
            raise SystemExit(f"{arm}: criteria.json 없음")
        back = idm["P50"] if arm == "P50" else None
        n = 0
        for vid, e in crit["videos"].items():
            real = back[vid] if back else vid
            q_tv[real] += len(e["questions"])
            n += len(e["questions"]) * len(tv_items[real]["positions"])
        per_arm[arm] = {"questions_total": sum(len(e["questions"]) for e in crit["videos"].values()), "question_reads": n}
    tv_sec = 0.0
    for vid, it in tv_items.items():
        w, h = (int(x) for x in fsum[vid]["sizes"][0].split("x"))
        tv_sec += len(it["positions"]) / H.rate(q_tv[vid], 4 * w * h)

    def mc_seconds(arms):
        tot, reads = 0.0, {}
        ids = sorted(idm["P12"].values())
        back = idm["P12"]
        for v in ids:
            q = 0
            for arm in arms:
                crit, _ = allc[arm]
                if arm == "G":
                    k = len(crit["videos"]["minecraft"]["questions"])
                elif arm == "P12":
                    k = len(crit["videos"][next(a for a, b in back.items() if b == v)]["questions"])
                else:
                    k = len(crit["videos"][v]["questions"])
                q += k
                reads[arm] = reads.get(arm, 0) + 300 * k
            tot += 300 / H.rate(q, 4 * 640 * 360)
        return tot, reads
    out = {"what": "본 실행(C 단계)의 시간과 비용. 글쓴이가 실제로 쓴 질문 수로 셌다. 식은 highlight_auto README 11절과 같다(초당 요청 = min(16, 5,500만 ÷ (질문 수 × 4장의 화소 합)))",
           "price_per_second": round(P.PRICE, 8), "tvsum": {"arms": per_arm, "ask_seconds": round(tv_sec, 1)}, "scenarios": {}}
    for name, arms in (("base_GP_P12", ["GP", "P12"]), ("with_G", ["G", "GP", "P12"])):
        mc_sec, reads = mc_seconds(arms)
        one = H.window(tv_sec + mc_sec + H.SANITY_SECONDS)
        tvw, mcw = H.window(tv_sec + H.SANITY_SECONDS), H.window(mc_sec + H.SANITY_SECONDS)
        out["scenarios"][name] = {"minecraft_arms": arms, "minecraft_question_reads": reads, "minecraft_ask_seconds": round(mc_sec, 1),
                                  "one_function": one,
                                  "two_functions": {"tvsum": tvw, "minecraft": mcw, "fast_usd": round(tvw["fast_usd"] + mcw["fast_usd"], 2), "slow_usd": round(tvw["slow_usd"] + mcw["slow_usd"], 2)},
                                  "split_needed": one["slow_seconds"] > 2300}
    out["note"] = "A 단계에서 고르기 질문 9개짜리 요청은 이 식이 내다본 속도의 86%(TVSum), 71%(마인크래프트)로 돌았다. 본 실행의 질문은 noul·score라 highlight_auto 본 실행과 같은 꼴이다"
    H.write(HERE / "data" / "main_run_cost.json", out)
    import json
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "check", "keywords", "cost"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"build": stage_build, "check": stage_check, "keywords": stage_keywords, "cost": stage_cost}[args.stage](args)


if __name__ == "__main__":
    main()
