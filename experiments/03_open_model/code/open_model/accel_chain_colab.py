"""Colab에서 d1-2b의 가속 변형을 차례로 돌린다(open_eval_colab.py를 변형마다 한 번씩). 가벼운 것부터.

모든 변형은 같은 항목(과제마다 3번째 영상마다 하나, 그 영상의 항목 전부)에서 돈다. 변형마다 최대 25분.
실행 (Colab): cd /content/open && nohup python accel_chain_colab.py > log_accel_chain.txt 2>&1 &
결과: results/d1-2b__<변형>.json, log_accel_<변형>.txt, 끝나면 log_accel_done.txt
"""
import subprocess
import sys
import time

BASE = [sys.executable, "open_eval_colab.py", "--model", "d1-2b", "--pack", "/content/open/pack", "--out", "/content/open/results", "--subset", "3"]
RUNS = [("base", []), ("fp16", ["--dtype", "fp16"]), ("compile", ["--compile"]), ("batch8", ["--batch", "8"]),
        ("compile_batch8", ["--compile", "--batch", "8"]), ("scale075", ["--scale", "0.75"]), ("scale050", ["--scale", "0.5"]),
        ("video", ["--video", "--tasks", "charades"]), ("4bit", ["--quant", "4bit"]), ("8bit", ["--quant", "8bit"])]

for tag, extra in RUNS:
    if extra[:1] == ["--quant"]:
        dry = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "bitsandbytes"], capture_output=True, text=True).stdout
        if "torch-" in dry.split("Would install")[-1]:
            print(tag, "건너뜀: bitsandbytes를 깔면 torch가 바뀐다", flush=True)
            continue
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
    t0 = time.time()
    with open(f"log_accel_{tag}.txt", "w") as log:
        try:
            code = subprocess.run(BASE + ["--tag", tag] + extra, stdout=log, stderr=subprocess.STDOUT, timeout=1500).returncode
        except subprocess.TimeoutExpired:
            code = "25분 넘겨 중단"
    print(tag, "끝", code, round(time.time() - t0), "초", flush=True)
open("log_accel_done.txt", "w").write("DONE\n")
