"""최종 시험을 Space(kcyoow/moment-finder)의 API로 돌린다. 화면과 같은 job을 단계마다 부른다(/ask → /step … → /confirm → /step …).

  uv run --no-project --with gradio_client --with huggingface_hub python experiments/final_qwen_h2o/run_space.py [이름 ...]

단계마다 받은 것을 results/<이름>/에 바로 적는다(계획 plan.json, 판정 답 answers.json, 끝난 상태 status.json, 클립 clip_N.mp4).
다시 부르면 끝난 것은 건너뛰고, 계획이나 판정 답이 남아 있으면 그것을 그대로 넣어 GPU를 다시 쓰지 않는다.
Hugging Face 토큰은 이 Mac에 로그인해 둔 것을 쓴다(GPU 한도가 그 계정에서 나간다).
"""
import json
import shutil
import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file
from huggingface_hub import get_token

HERE = Path(__file__).resolve().parent
UPLOADS = HERE.parent.parent / "pipeline" / "uploads"
SPACE = "kcyoow/moment-finder"
ROBOT = "로봇이 사탕 컵에 넣는 거 성공한 장면 찾아줘"
CASES = {"robot_ok": ("web_1116_200731_2.mp4", ROBOT), "robot_none": ("web_1116_200730_1.mp4", ROBOT),
         "mc_diamond": ("web_1116_201838_3.mp4", "마크 영상인데 다이아 캐는 장면 찾아줘"),
         "mc_creeper": ("web_1116_201839_4.mp4", "마크 영상인데 크리퍼 나오는 장면 찾아줘"),
         "mc_fight": ("web_1116_041133_3.mp4", "마크 영상인데 싸우는 장면 뽑아줘"),
         "mc_mixed": ("web_1116_043326_3.mp4", "마크 영상인데 다이아 캐거나 크리퍼 만나거나 위험한 장면 뽑아줘")}
WORK = ("write", "judge", "recheck")


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def steps(client, job, out, st):
    """일하는 단계가 남은 동안 /step을 부른다. 멈추면 세 번까지 다시 부르고, 한도 문제면 그만둔다."""
    fails = 0
    while st["stage"] in WORK:
        t = time.time()
        st, files = client.predict(job, api_name="/step")
        print(f"   [{st['stage']}{' 멈춤' if st.get('paused') else ''}] {time.time() - t:5.1f}초 · {st['log'][-1][:150] if st['log'] else ''}", flush=True)
        if st.get("answers"):
            write(out / "answers.json", st["answers"])
        if st.get("plan"):
            write(out / "plan.json", st["plan"])
        if st.get("paused"):
            fails += 1
            err = (st.get("error") or "").lower()
            if fails > 3 or "quota" in err or "exceeded" in err:
                return st, None
            time.sleep(5)
        else:
            fails = 0
        if st["stage"] == "done":
            return st, files
    return st, None


def run(client, name):
    video, request = CASES[name]
    out = HERE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("stage") == "done":
        print(f"{name}: 이미 끝남", flush=True)
        return True
    plan = (out / "plan.json").read_text(encoding="utf-8") if (out / "plan.json").exists() else ""
    answers = (out / "answers.json").read_text(encoding="utf-8") if (out / "answers.json").exists() else ""
    t0 = time.time()
    print(f"{name}: 올리는 중 ({video})" + (" · 앞서 쓴 계획을 넣음" if plan else "") + (" · 앞서 받은 판정 답을 넣음" if answers else ""), flush=True)
    job, st = client.predict(handle_file(str(UPLOADS / video)), request, "", "", plan, api_name="/ask")
    st, _ = steps(client, job, out, st)
    if st["stage"] != "confirm":
        print(f"{name}: 계획을 받지 못함: {st.get('error')}", flush=True)
        write(out / "status_failed.json", st)
        return False
    write(out / "plan.json", st["plan"])
    st = client.predict(job, answers, api_name="/confirm")
    print(f"   [confirm] {st['log'][-1][:150]}", flush=True)
    st, files = steps(client, job, out, st)
    if st["stage"] != "done":
        print(f"{name}: 끝까지 가지 못함: {st.get('error')}", flush=True)
        write(out / "status_failed.json", st)
        return False
    for i, f in enumerate(files or [], 1):
        shutil.copy(f, out / f"clip_{i}.mp4")
        st["result"]["clips"][i - 1]["file"] = f"clip_{i}.mp4"
    st.pop("answers", None)
    st["wall_seconds"] = round(time.time() - t0, 1)
    write(out / "status.json", st)
    r = st["result"]
    print(f"{name}: {r['summary']} (전체 {st['wall_seconds']:.0f}초)", flush=True)
    for row in r["rows"]:
        print("   ", row, flush=True)
    return True


def main():
    names = sys.argv[1:] or list(CASES)
    client = Client(SPACE, token=get_token(), verbose=False)
    for n in names:
        if not run(client, n):
            print("멈춤: 남은 것은 돌리지 않는다", flush=True)
            break


if __name__ == "__main__":
    main()
