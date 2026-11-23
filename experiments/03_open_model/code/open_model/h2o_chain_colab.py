"""Colab에서 H2O-Lightning-4B를 평가 묶음에 차례로 돌린다. 서버(모델 저장소의 serve.sh: vLLM + shim)를 필요하면 다시 띄운다.

첫 실행(2026-11-12)에서는 영상의 1/3을 하나씩 돌린 것(base)까지는 됐고, 8개를 동시에 보내던 중 4장 묶음에서 엔진이 GPU 메모리
부족으로 죽었다(L4 22GB, GPU_UTIL 기본 0.90). GPU_UTIL만 0.70으로 낮추면 vLLM이 시작을 거부한다(동시 처리 자리 256개를 못 채움).
그래서 GPU_UTIL 0.80, 동시 처리 자리 32개(--max-num-seqs), MAX_LEN 16384(한 번에 읽는 토큰 수의 한도이기도 하다)로 띄우고,
실행마다 서버가 살아 있는지 보고 죽었으면 다시 띄운다.

순서: 한 방향 읽기(하나씩, 8개 동시) → 양방향 8개 동시(이어 하기) → 전체 묶음(양방향, 4개 동시) → FP8 양자화로 서버를 다시 띄워 두 가지.
실행 (Colab): cd /content/open && nohup python h2o_chain_colab.py [--only 이름,이름] > log_h2o_chain2.txt 2>&1 &
결과: results/h2o-4b__<변형>.json, results/h2o-4b.json, 끝나면 log_h2o_done.txt
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

EVAL = [sys.executable, "open_eval_colab.py", "--model", "h2o-4b", "--pack", "/content/open/pack", "--out", "/content/open/results"]
SUB = ["--subset", "3"]
BASE = {"GPU_UTIL": "0.80", "MAX_LEN": "16384"}
PLAIN, FP8 = {**BASE, "VLLM_ARGS": "--max-num-seqs 32"}, {**BASE, "VLLM_ARGS": "--max-num-seqs 32 --quantization fp8"}
RUNS = [("oneway", PLAIN, SUB + ["--tag", "oneway", "--attention", "causal"]),
        ("oneway_c8", PLAIN, SUB + ["--tag", "oneway_c8", "--attention", "causal", "--batch", "8"]),
        ("c8", PLAIN, SUB + ["--tag", "c8", "--batch", "8"]),
        ("full", PLAIN, ["--batch", "4"]),
        ("fp8", FP8, SUB + ["--tag", "fp8"]),
        ("fp8_oneway_c8", FP8, SUB + ["--tag", "fp8_oneway_c8", "--attention", "causal", "--batch", "8"]),
        # 둘째 차례(--only로 골라 돌림): 가장 빠른 구성에 그림 줄이기를 얹고, 전체 묶음으로 다시 잰다
        ("fp8_oneway_c8_s075", FP8, SUB + ["--tag", "fp8_oneway_c8_s075", "--attention", "causal", "--batch", "8", "--scale", "0.75"]),
        ("fp8_oneway_c8_s050", FP8, SUB + ["--tag", "fp8_oneway_c8_s050", "--attention", "causal", "--batch", "8", "--scale", "0.5"]),
        ("full_fp8_oneway", FP8, ["--tag", "full_fp8_oneway", "--attention", "causal", "--batch", "8"]),
        ("full_fp8_oneway_s050", FP8, ["--tag", "full_fp8_oneway_s050", "--attention", "causal", "--batch", "8", "--scale", "0.5"])]
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
MARK = "h2o_server_env.json"  # 지금 떠 있는 서버가 어떤 설정인지 적어 둔다(스크립트를 다시 시작해도 서버를 다시 띄우지 않게)
current = json.load(open(MARK)) if os.path.exists(MARK) else None


def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8741/health", timeout=3).status == 200
    except Exception:
        return False


def ensure_server(env):
    """env대로 뜬 서버가 살아 있게 한다. 설정이 바뀌었거나 죽었으면 다시 띄우고 준비될 때까지(최대 20분) 기다린다."""
    global current
    if current == env and healthy():
        return True
    subprocess.run("pkill -f 'vllm serv[e]'; pkill -f 'h2o_lightning_shi[m]'; pkill -f 'h2o-lightning-4b/serve.s[h]'; sleep 8", shell=True)
    log = open(f"log_h2o_serve_{int(time.time())}.txt", "w")
    proc = subprocess.Popen("bash -c '. /content/vl/bin/activate && exec bash /content/h2o-lightning-4b/serve.sh'", shell=True, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    t0 = time.time()
    while time.time() - t0 < 1200:
        if healthy():
            current = env
            json.dump(env, open(MARK, "w"))
            print("서버 준비", env, round(time.time() - t0), "초", flush=True)
            return True
        if proc.poll() is not None:  # 시작하다 끝났으면 더 기다리지 않는다
            print("서버 시작 실패", env, "→", log.name, flush=True)
            return False
        time.sleep(5)
    print("서버가 20분 안에 준비되지 않음", env, flush=True)
    return False


for tag, env, extra in RUNS:
    if ONLY and tag not in ONLY:
        continue
    if not ensure_server(env):
        print(tag, "건너뜀(서버 없음)", flush=True)
        continue
    t0 = time.time()
    with open(f"log_h2o_{tag}.txt", "w") as log:
        try:
            code = subprocess.run(EVAL + extra, stdout=log, stderr=subprocess.STDOUT, timeout=2400).returncode
        except subprocess.TimeoutExpired:
            code = "40분 넘겨 중단"
    print(tag, "끝", code, round(time.time() - t0), "초", "| 서버", "살아 있음" if healthy() else "죽음", flush=True)
open("log_h2o_done.txt", "w").write("DONE\n")
