"""highlight_probe C 단계(본 실행)의 요청 글과 이 Mac에서의 확인 (돈 없음).

단계
  requests : 글쓴이 여섯의 기준 → Modal에 올릴 요청 글 data/main_requests.json
  check    : 요청 글로 본문을 실제 프레임으로 만들어 보고(묶음마다 몇 건), 상태의 칸과 질문 수를 확인한다

실행: uv run --offline --no-project --with numpy --with pillow python -I experiments/highlight_probe/hp_main.py <단계>

묶음(요청 하나에 그림 4장과 상태, 그리고 그 상태를 함께 쓰는 팔들의 질문)
  tv_titled : TVSum 6,316곳. 상태는 highlight_auto와 같다(viewer_request에 제목). 질문은 T(키 앞에 t__)와 TP(tp__)
  tv_plain  : TVSum 6,316곳. 상태의 viewer_request는 제목 없는 문장. 질문은 P50
  mc_game   : 마인크래프트 3,600곳. 상태는 highlight_auto와 같다(game, viewer_request). 질문은 G(g__)와 GP(gp__)
  mc_plain  : 마인크래프트 3,600곳. 상태에 game이 없고 viewer_request는 게임 이름 없는 문장. 질문은 P12

P 팔을 따로 묻는 까닭: 질문은 `frames`만 가리키지만 판정 모델은 상태 전체를 본다. 제목이나 게임 이름이 든 상태로 P의 질문을 물으면
"사람이 알려 주지 않은 경우"가 아니게 된다(README 6절, 15-1).
질문의 키는 판정 모델에 보이지 않는다(안내의 규칙). 키 앞의 t__ 같은 표시는 답을 팔로 나누는 데만 쓴다.

올라가는 글: 제목(viewer_request 안), 질문. TVSum 점수와 범주, 길이는 넣지 않는다.
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tarfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hp_prepare as H  # noqa: E402
import hp_writers as W  # noqa: E402

P, HP = H.P, H.HP
OUT = HERE / "data" / "main_requests.json"
PREFIX = {"T": "t__", "TP": "tp__", "G": "g__", "GP": "gp__"}


def build():
    allc, idm = W.load_all()
    for arm, (crit, want) in allc.items():
        errs = W.check_criteria(crit, want) if crit else ["criteria.json 없음"]
        if errs:
            raise SystemExit(f"{arm}: 형식 오류 {errs[:3]}")
    C = {arm: crit["videos"] for arm, (crit, _) in allc.items()}
    items = H.read(H.HA / "data" / "tvsum_items.json")["videos"]
    fsum = H.read(H.HA / "data" / "frames_summary.json")["packs"]
    titles = {r["video_id"]: r["title"] for r in P.load_info()}
    back50, back12 = idm["P50"], idm["P12"]
    mc_ids = sorted(back12.values())

    def pre(arm, qs):
        return {PREFIX[arm] + k: q for k, q in qs.items()}
    tv_titled = {vid: {"viewer_request": P.TV_REQUEST.format(title=titles[vid]),
                       "questions": {**pre("T", C["T"][vid]["questions"]), **pre("TP", C["TP"][vid]["questions"])}} for vid in items}
    tv_plain = {real: {"viewer_request": W.REQ_TV_GENERIC, "questions": dict(C["P50"][v]["questions"])} for v, real in back50.items()}
    mc_game = {v: {"questions": {**pre("G", C["G"]["minecraft"]["questions"]), **pre("GP", C["GP"][v]["questions"])}} for v in mc_ids}
    mc_plain = {real: {"questions": dict(C["P12"][v]["questions"])} for v, real in back12.items()}
    if set(tv_plain) != set(items) or set(mc_plain) != set(mc_ids):
        raise SystemExit("P 팔의 id 대응이 맞지 않는다")
    positions = {vid: [{"k": p["k"], "frames": [P.fname(i) for i in p["frames"]], "seconds": p["seconds"]} for p in it["positions"]] for vid, it in items.items()}

    def tv_seconds(arm):
        s = 0.0
        for vid in items:
            w, h = (int(x) for x in fsum[vid]["sizes"][0].split("x"))
            s += len(positions[vid]) / H.rate(len(arm[vid]["questions"]), 4 * w * h)
        return round(s, 1)

    def mc_seconds(arm):
        return round(sum(300 / H.rate(len(arm[v]["questions"]), 4 * 640 * 360) for v in mc_ids), 1)
    n_tv = sum(len(p) for p in positions.values())
    out = {"note": "highlight_probe C 단계의 요청 글. TVSum 점수, 범주, 길이는 없다. 제목은 tv_titled의 viewer_request 글로만 들어 있다",
           "about": P.ABOUT,
           "arms": {"tv_titled": {"videos": tv_titled}, "tv_plain": {"videos": tv_plain},
                    "mc_game": {"game": P.MC_GAME, "viewer_request": P.MC_REQUEST, "videos": mc_game},
                    "mc_plain": {"viewer_request": W.REQ_MC_GENERIC, "videos": mc_plain}},
           "positions": positions, "packs": {vid: fsum[vid]["sha256"] for vid in items},
           "prefix": PREFIX,
           "plan": {"sanity": {"requests": 200, "est_seconds": round(200 / 16, 1)},
                    "tv_titled": {"requests": n_tv, "est_seconds": tv_seconds(tv_titled)}, "tv_plain": {"requests": n_tv, "est_seconds": tv_seconds(tv_plain)},
                    "mc_game": {"requests": 3600, "est_seconds": mc_seconds(mc_game)}, "mc_plain": {"requests": 3600, "est_seconds": mc_seconds(mc_plain)},
                    "model": "초당 요청 = min(16, 5,500만 ÷ (요청당 질문 수 × 4장의 화소 합)) (highlight_auto README 11절)"}}
    return out, C


def stage_requests(args):
    out, _ = build()
    H.write(OUT, out)
    keys = [k for a in out["arms"].values() for v in a["videos"].values() for k in v["questions"]]
    print(json.dumps(out["plan"], ensure_ascii=False, indent=1))
    print("파일", OUT.stat().st_size, "바이트, 가장 긴 키", max(len(k) for k in keys), "자")


def tar_get(path, names):
    with tarfile.open(path) as tar:
        return {n: tar.extractfile(tar.getmember(n)).read() for n in names}


def stage_check(args):
    import hp_main_modal as M   # noqa: E402  (본문을 만드는 같은 함수를 쓴다. modal이 있어야 불러진다)
    req = H.read(OUT)
    built, C = build()
    assert built == req, "요청 글이 지금의 기준과 다르다(requests를 다시 돌린다)"
    M.REQ.clear()
    M.REQ.update(req)
    mc_items = HP.load_items()["full"]
    bad = []
    titles = {r["video_id"]: r["title"] for r in P.load_info()}
    old = H.read(H.HA / "data" / "requests.json")
    # --- TVSum: 영상마다 첫·가운데·끝 위치
    n_built = 0
    for vid, pos in req["positions"].items():
        picks = [pos[0], pos[len(pos) // 2], pos[-1]]
        M.FRAMES[vid] = tar_get(P.PACKS / f"{vid}.tar", sorted({n for p in picks for n in p["frames"]}))
        for p in picks:
            it = {"id": f"{vid}@{p['k']}", "video": vid, "k": p["k"], "pos": p}
            for arm in ("tv_titled", "tv_plain"):
                body, n_img = M._build((arm, it))
                pay = json.loads(body)
                n_built += 1
                st = pay["state"]
                if list(st) != ["viewer_request", "about", "frames"] or n_img != 4 or len(st["frames"]) != 4:
                    bad.append((arm, vid, "상태의 칸"))
                if pay["questions"] != req["arms"][arm]["videos"][vid]["questions"]:
                    bad.append((arm, vid, "질문"))
                if arm == "tv_titled":
                    ref = P.tv_body(old["tvsum"]["videos"][vid], p, lambda v, n: M.FRAMES[v][n], vid)["state"]     # highlight_auto의 상태와 글자까지 같아야 한다
                    if json.dumps(st) != json.dumps(ref):
                        bad.append((arm, vid, "상태가 highlight_auto와 다르다"))
                else:
                    text = json.dumps({"vr": st["viewer_request"], "about": st["about"], "q": pay["questions"]})
                    if st["viewer_request"] != W.REQ_TV_GENERIC or titles[vid] in text:
                        bad.append((arm, vid, "제목이 샌다"))
        del M.FRAMES[vid]
    # --- 마인크래프트: 영상마다 첫·가운데·끝 위치
    by_video = {}
    for it in mc_items:
        by_video.setdefault(it["video"], []).append(it)
    for vid, its in by_video.items():
        picks = [its[0], its[len(its) // 2], its[-1]]
        M.FRAMES[vid] = tar_get(H.HH / "data" / "packs" / f"{vid}.tar", sorted({HP.fname(t) for it in picks for t in it["times"]}))
        for it in picks:
            for arm in ("mc_game", "mc_plain"):
                body, n_img = M._build((arm, it))
                pay = json.loads(body)
                n_built += 1
                st = pay["state"]
                want = ["game", "viewer_request", "about", "frames"] if arm == "mc_game" else ["viewer_request", "about", "frames"]
                if list(st) != want or n_img != 4:
                    bad.append((arm, vid, "상태의 칸"))
                if pay["questions"] != req["arms"][arm]["videos"][vid]["questions"]:
                    bad.append((arm, vid, "질문"))
                if arm == "mc_game":
                    ref = HP.body(it, lambda v, n: M.FRAMES[v][n])["state"]
                    if json.dumps(st) != json.dumps(ref):
                        bad.append((arm, vid, "상태가 highlight_h2o와 다르다"))
                else:
                    text = json.dumps({"vr": st["viewer_request"], "about": st["about"], "q": pay["questions"]})
                    if st["viewer_request"] != W.REQ_MC_GENERIC or "inecraft" in text:
                        bad.append((arm, vid, "게임 이름이 샌다"))
        del M.FRAMES[vid]
    # --- 질문을 읽는 횟수가 비용 파일과 같은가
    cost = H.read(HERE / "data" / "main_run_cost.json")
    reads = {"T": 0, "TP": 0, "P50": 0, "G": 0, "GP": 0, "P12": 0}
    for vid, pos in req["positions"].items():
        qs = req["arms"]["tv_titled"]["videos"][vid]["questions"]
        reads["T"] += len(pos) * sum(k.startswith("t__") for k in qs)
        reads["TP"] += len(pos) * sum(k.startswith("tp__") for k in qs)
        reads["P50"] += len(pos) * len(req["arms"]["tv_plain"]["videos"][vid]["questions"])
    for vid in by_video:
        qs = req["arms"]["mc_game"]["videos"][vid]["questions"]
        reads["G"] += 300 * sum(k.startswith("g__") for k in qs)
        reads["GP"] += 300 * sum(k.startswith("gp__") for k in qs)
        reads["P12"] += 300 * len(req["arms"]["mc_plain"]["videos"][vid]["questions"])
    want = {**{a: cost["tvsum"]["arms"][a]["question_reads"] for a in ("T", "TP", "P50")}, **cost["scenarios"]["with_G"]["minecraft_question_reads"]}
    if reads != want:
        bad.append(("reads", reads, want))
    # --- 답 읽기(parse)가 맞는 답은 받고 틀린 답은 거르는가
    qs = {"a": {"type": "noul"}, "b": {"type": "score", "criteria": ["x", "y", "z"]}}
    ok = {"answers": {"a": {"noul": 0.25}, "b": {"score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}}}
    assert M.parse(ok, qs) == {"a": {"noul": 0.25}, "b": {"score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}}
    for broken in ({"answers": {"a": {"noul": 1.5}, "b": ok["answers"]["b"]}},
                   {"answers": {"a": {"noul": 0.5}, "b": {"score": 2.0, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}}},
                   {"answers": {"a": {"noul": 0.5}, "b": {"score": 1.0, "probabilities": {"0": 0.5, "1": 0.5}}}}):
        try:
            M.parse(broken, qs)
            bad.append(("parse", "틀린 답을 받았다", broken))
        except (ValueError, KeyError):
            pass
    print("만들어 본 본문", n_built, "개, 읽는 횟수", reads)
    print("틀린 곳:", bad[:10] if bad else "없음")
    if bad:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["requests", "check"])
    args = ap.parse_args()
    {"requests": stage_requests, "check": stage_check}[args.stage](args)


if __name__ == "__main__":
    main()
