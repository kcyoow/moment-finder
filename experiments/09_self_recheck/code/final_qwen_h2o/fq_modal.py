"""고친 안내 글로 다시 재는 것을 Modal에서 돈다. Space의 코드(space/)를 그대로 불러 같은 흐름(engine.py의 job 단계)을 돈다. README 6절.

Hugging Face의 하루 GPU 한도를 더 쓰지 않으려고 옮겼다(2026-11-23). 환경은 Space의 로그에 찍힌 것에 맞췄다
(torch 2.13.0 + CUDA 13, transformers 5.20.2, kernels 0.17.x, triton 3.7.1, accelerate 1.16.0, causal-conv1d 1.7.0). GPU도 같은 RTX PRO 6000이다.
다른 점: ZeroGPU가 아니라 GPU가 계속 붙어 있다(spaces.GPU는 Space 밖에서는 아무 일도 하지 않는다).

영상마다 run_space_v2.py와 같은 일을 한다:
  1) Qwen이 고친 안내 글로 계획을 새로 쓴다 → plan_v2.json
  2) 첫 실행의 질문·판정 답은 그대로 두고 다시 확인용 정의(check)만 새 것으로 바꿔, 다시 확인 단계만 돌린다 → status.json, clip_N.mp4

볼륨 (끝나면 둘 다 지운다)
  mf-final-hf (/hf)     : Hugging Face 캐시(H2O-Lightning-4B 9GB, Qwen3.8-27B-FP8 30GB). fetch가 받는다
  mf-final-data (/data) : 영상, 첫 실행의 계획과 판정 답. main이 올린다

실행 (저장소 맨 위에서):
  MODAL_PROFILE=<modal-profile> modal run experiments/final_qwen_h2o/fq_modal.py::check_env     # 패키지가 맞게 깔렸나 (CPU)
  MODAL_PROFILE=<modal-profile> modal run experiments/final_qwen_h2o/fq_modal.py::fetch         # 가중치 받기 (CPU)
  MODAL_PROFILE=<modal-profile> modal run experiments/final_qwen_h2o/fq_modal.py --names mc_mixed,mc_creeper,robot_ok,robot_none,mc_fight
  MODAL_PROFILE=<modal-profile> modal volume delete mf-final-data --yes ; MODAL_PROFILE=<modal-profile> modal volume delete mf-final-hf --yes

값: RTX PRO 6000 + 8코어 + 64GiB = 초당 약 $0.00109. 함수의 상한 2,400초 = $2.6.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
app = modal.App("moment-finder-final-test")
hf = modal.Volume.from_name("mf-final-hf", create_if_missing=True)
data = modal.Volume.from_name("mf-final-data", create_if_missing=True)

H2O, H2O_REV = "h2oai/h2o-lightning-4b", "659fc412de8486dbc2477b0d1cb4121e97e72471"
QWEN, QWEN_REV = "Qwen/Qwen3.8-27B-FP8", "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"
ROBOT = "로봇이 사탕 컵에 넣는 거 성공한 장면 찾아줘"
CASES = {"robot_ok": ("web_1116_200731_2.mp4", ROBOT), "robot_none": ("web_1116_200730_1.mp4", ROBOT),
         "mc_diamond": ("web_1116_201838_3.mp4", "마크 영상인데 다이아 캐는 장면 찾아줘"),
         "mc_creeper": ("web_1116_201839_4.mp4", "마크 영상인데 크리퍼 나오는 장면 찾아줘"),
         "mc_fight": ("web_1116_041133_3.mp4", "마크 영상인데 싸우는 장면 뽑아줘"),
         "mc_mixed": ("web_1116_043326_3.mp4", "마크 영상인데 다이아 캐거나 크리퍼 만나거나 위험한 장면 뽑아줘")}
TIMEOUT = 2400

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")          # Triton이 커널을 컴파일할 때 C 컴파일러를 쓴다
    .uv_pip_install("torch==2.13.0", "torchvision", "transformers==5.20.2", "accelerate==1.16.0", "kernels>=0.17.2,<0.18", "tqdm<4.70", "triton==3.7.1",
                    "spaces==0.52.1", "gradio==6.29.1", "pillow", "numpy", "pandas", "imageio-ffmpeg", "huggingface_hub",
                    "causal-conv1d @ https://wheels.astral.sh/artifacts/b8b7d03b1cafb306f9e89d8d195ece8c296417d8be00b55c4b809b018f77a99b/"
                    "causal_conv1d-1.7.0+cu.13.0.torch.2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl")
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(ROOT / "space"), "/root/space", ignore=["examples/**", "__pycache__/**", "*.pyc"])
)


@app.function(image=image, timeout=600, cpu=2)
def check_env():
    """GPU 없이: 패키지가 Space와 같은 판으로 깔렸는지, 컴파일러가 있는지만 본다."""
    import importlib.metadata as md
    import shutil
    out = {n: md.version(n) for n in ("torch", "transformers", "kernels", "triton", "accelerate", "spaces", "gradio", "causal-conv1d", "tqdm")}
    import torch
    out.update(cuda_build=torch.version.cuda, gcc=shutil.which("gcc"))
    import spaces  # noqa: F401
    import gradio  # noqa: F401
    import kernels  # noqa: F401
    from transformers import FineGrainedFP8Config  # noqa: F401
    print(json.dumps(out))
    return out


@app.function(image=image, volumes={"/hf": hf}, timeout=3600, cpu=8)
def fetch():
    """두 모델을 Hugging Face 캐시(볼륨)에 받아 둔다. GPU가 떠 있는 동안 받지 않으려는 것."""
    import os
    import time
    from huggingface_hub import snapshot_download
    t0 = time.time()
    for repo, rev in ((H2O, H2O_REV), (QWEN, QWEN_REV)):
        for attempt in range(5):
            try:
                p = snapshot_download(repo, revision=rev, max_workers=16)
                break
            except Exception as err:  # noqa: BLE001
                print(f"{repo} 받기 실패 {attempt + 1}/5: {str(err)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError(f"{repo}를 받지 못함")
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p, followlinks=True) for f in fs)
        print(f"{repo}: {size / 1e9:.1f}GB, {time.time() - t0:.0f}초", flush=True)
    hf.commit()
    return round(time.time() - t0)


@app.function(image=image, gpu="RTX-PRO-6000", volumes={"/hf": hf, "/data": data}, timeout=TIMEOUT, cpu=8, memory=65536)
def run(names: list):
    """Space의 app.py를 그대로 불러(모델 둘이 GPU에 올라간다) engine의 job 단계를 돈다. 돌려주는 것: {이름: {plan_v2, status, clips: {파일: bytes}}}"""
    import os
    import sys
    import time
    t0 = time.time()
    sys.path.insert(0, "/root/space")
    os.chdir("/root/space")
    import app  # noqa: F401 — 모델 올리기
    import engine as E
    print(f"[modal] 모델 올림: {time.time() - t0:.0f}초", flush=True)
    out = {}

    def drain(job):
        while E.more(job):
            job = E.job_step(job["id"])
            print(f"   [{job['stage']}{' 멈춤' if job.get('paused') else ''}] {job['log'][-1][:200]}", flush=True)
        return job

    for name in names:
        video, request = CASES[name]
        path = f"/data/videos/{video}"
        first = json.load(open(f"/data/first/{name}/plan.json", encoding="utf-8"))
        answers = open(f"/data/first/{name}/answers.json", encoding="utf-8").read()
        t1 = time.time()
        print(f"{name}: 고친 안내 글로 계획을 새로 쓰는 중", flush=True)
        job = drain(E.job_ask("", path, request))
        if job["stage"] != "confirm":
            out[name] = {"error": job.get("error"), "log": job["log"]}
            print(f"{name}: 계획을 받지 못함: {job.get('error')}", flush=True)
            continue
        v2 = job["plan"]
        merged = {**first, "check": v2["check"]}
        job = E.job_ask("", path, request, "", json.dumps(merged, ensure_ascii=False))
        job = drain(E.job_confirm(job["id"], answers))
        if job["stage"] != "done":
            out[name] = {"plan_v2": v2, "error": job.get("error"), "log": job["log"]}
            print(f"{name}: 끝까지 가지 못함: {job.get('error')}", flush=True)
            continue
        st = E.status_of(job)
        clips = {}
        for i, c in enumerate(st["result"]["clips"], 1):
            clips[f"clip_{i}.mp4"] = open(c["file"], "rb").read()
            c["file"] = f"clip_{i}.mp4"
        st["wall_seconds"] = round(time.time() - t1, 1)
        out[name] = {"plan_v2": v2, "status": st, "clips": clips}
        os.makedirs("/data/out", exist_ok=True)                      # 함수가 중간에 끊겨도 끝난 것은 볼륨에 남는다
        json.dump({"plan_v2": v2, "status": st}, open(f"/data/out/{name}.json", "w", encoding="utf-8"), ensure_ascii=False)
        data.commit()
        print(f"{name}: {st['result']['summary']} ({time.time() - t1:.0f}초)", flush=True)
        for row in st["result"]["rows"]:
            print("   ", row, flush=True)
    print(f"[modal] 전체 {time.time() - t0:.0f}초", flush=True)
    return {"seconds": round(time.time() - t0, 1), "cases": out}


@app.local_entrypoint()
def main(names: str = "mc_mixed,mc_creeper,robot_ok,robot_none,mc_fight"):
    todo = [n for n in names.split(",") if n and not (HERE / "results_v2" / n / "status.json").exists()]
    if not todo:
        print("할 것이 없다(모두 results_v2/에 있다)")
        return
    with data.batch_upload(force=True) as b:
        for n in todo:
            b.put_file(str(ROOT / "pipeline" / "uploads" / CASES[n][0]), f"/videos/{CASES[n][0]}")
            for f in ("plan.json", "answers.json"):
                b.put_file(str(HERE / "results" / n / f), f"/first/{n}/{f}")
    print("올림:", todo, flush=True)
    res = run.remote(todo)
    for n, r in res["cases"].items():
        d = HERE / "results_v2" / n
        d.mkdir(parents=True, exist_ok=True)
        if r.get("plan_v2"):
            (d / "plan_v2.json").write_text(json.dumps(r["plan_v2"], ensure_ascii=False, indent=1), encoding="utf-8")
        if r.get("status"):
            for f, raw in r["clips"].items():
                (d / f).write_bytes(raw)
            r["status"]["ran_on"] = "Modal RTX PRO 6000 (Space의 코드 그대로)"
            (d / "status.json").write_text(json.dumps(r["status"], ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            (d / "failed.json").write_text(json.dumps({k: v for k, v in r.items() if k != "clips"}, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "results_v2" / "modal_run.json").write_text(json.dumps({"seconds": res["seconds"], "names": todo}, ensure_ascii=False), encoding="utf-8")
    print(f"GPU 함수 {res['seconds']}초", flush=True)
