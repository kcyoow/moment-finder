"""고친 안내 글(헷갈리는 장면 3~4개, 하나는 "찾는 대상만 빠진 같은 동작")로 다시 잰다. README 6절.

  uv run --no-project --with gradio_client --with huggingface_hub python experiments/final_qwen_h2o/run_space_v2.py mc_diamond robot_ok ...

하루 GPU 한도가 얼마 남지 않아, 후보를 다시 뽑지 않는다. 영상마다:
  1) Qwen이 고친 안내 글로 계획을 새로 쓴다(GPU 약 2분) → results_v2/<이름>/plan_v2.json
  2) 첫 실행의 질문·판정 답(results/<이름>/)은 그대로 두고, 다시 확인용 정의(check)만 새 것으로 바꿔 다시 확인 단계만 돌린다(GPU 몇 초)
그래서 후보는 첫 실행과 같고, 달라지는 것은 다시 확인에 쓰는 정의뿐이다.
"""
import json
import shutil
import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file
from huggingface_hub import get_token

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_space as R  # noqa: E402


def run(client, name):
    video, request = R.CASES[name]
    old, out = HERE / "results" / name, HERE / "results_v2" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        print(f"{name}: 이미 끝남", flush=True)
        return True
    vf = handle_file(str(R.UPLOADS / video))
    if not (out / "plan_v2.json").exists():
        print(f"{name}: 고친 안내 글로 계획을 새로 쓰는 중", flush=True)
        job, st = client.predict(vf, request, "", "", "", api_name="/ask")
        st, _ = R.steps(client, job, out, st)
        if st["stage"] != "confirm":
            print(f"{name}: 계획을 받지 못함: {st.get('error')}", flush=True)
            return False
        R.write(out / "plan_v2.json", st["plan"])
        if (out / "plan.json").exists():
            (out / "plan.json").unlink()
    v2 = json.loads((out / "plan_v2.json").read_text(encoding="utf-8"))
    merged = {**json.loads((old / "plan.json").read_text(encoding="utf-8")), "check": v2["check"]}
    answers = (old / "answers.json").read_text(encoding="utf-8")
    t0 = time.time()
    job, st = client.predict(vf, request, "", "", json.dumps(merged, ensure_ascii=False), api_name="/ask")
    st = client.predict(job, answers, api_name="/confirm")
    st, files = R.steps(client, job, out, st)
    if st["stage"] != "done":
        print(f"{name}: 끝까지 가지 못함: {st.get('error')}", flush=True)
        return False
    for i, f in enumerate(files or [], 1):
        shutil.copy(f, out / f"clip_{i}.mp4")
        st["result"]["clips"][i - 1]["file"] = f"clip_{i}.mp4"
    st.pop("answers", None)
    st["wall_seconds"] = round(time.time() - t0, 1)
    for f in ("plan.json", "answers.json"):
        if (out / f).exists():
            (out / f).unlink()
    R.write(out / "status.json", st)
    print(f"{name}: {st['result']['summary']}", flush=True)
    for row in st["result"]["rows"]:
        print("   ", row, flush=True)
    return True


def main():
    client = Client(R.SPACE, token=get_token(), verbose=False)
    for n in sys.argv[1:]:
        if not run(client, n):
            print("멈춤: 남은 것은 돌리지 않는다", flush=True)
            break


if __name__ == "__main__":
    main()
