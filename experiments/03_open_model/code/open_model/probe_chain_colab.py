"""Colab에서 "d1-2B가 Jev보다 빠를 수 있나" 확인용 실행을 차례로 돌린다. 가벼운 것부터.

1. speed_probe_colab.py (요청 하나의 시간 내역과 동시 처리)
2. 60칸 시트(둘째 묶음의 sheet_*), 3. 같은 것을 set_size 64로, 4. 0.5초 어긋난 위치의 4장 묶음(영상 입력)
실행 (Colab): cd /content/open && nohup python probe_chain_colab.py > log_probe_chain.txt 2>&1 &
결과: results/speed_probe.json, results/d1-2b__sheet.json, d1-2b__sheet_set64.json, d1-2b__half_video.json, 끝나면 log_probe_done.txt
"""
import subprocess
import sys
import time

EVAL = [sys.executable, "open_eval_colab.py", "--model", "d1-2b", "--pack", "/content/open/pack", "--out", "/content/open/results", "--manifest", "manifest2.json"]
SHEETS = ["--tasks", "sheet_sta2,sheet_mc,sheet_qvm"]
RUNS = [("probe", [sys.executable, "speed_probe_colab.py", "--pack", "/content/open/pack", "--out", "/content/open/results"]),
        ("sheet", EVAL + SHEETS + ["--tag", "sheet", "--show"]),
        ("sheet_set64", EVAL + SHEETS + ["--tag", "sheet_set64", "--set-size", "64"]),
        ("half_video", EVAL + ["--tasks", "charades_h", "--video", "--tag", "half_video"])]
for tag, cmd in RUNS:
    t0 = time.time()
    with open(f"log_probe_{tag}.txt", "w") as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1500).returncode
        except subprocess.TimeoutExpired:
            code = "25분 넘겨 중단"
    print(tag, "끝", code, round(time.time() - t0), "초", flush=True)
open("log_probe_done.txt", "w").write("DONE\n")
