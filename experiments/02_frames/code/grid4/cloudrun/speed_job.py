"""Cloud Run 작업(job) 안에서 도는 속도 시험의 시작 스크립트. speed_job.sh가 base64로 넘긴다.

이 Mac에서 보내면 집 회선의 업로드 속도(약 10Mbps)가 먼저 막혀 GPU의 처리량을 잴 수 없었다(2026-11-12).
그래서 같은 지역(us-central1)의 Cloud Run 작업에서 요청을 보낸다. 프레임과 스크립트는 비공개 버킷의 speed_pack.tar에서 받는다.
환경 변수: BUCKET(버킷 이름), TARGET(H2O 서비스 주소), SPEED_ARGS(grid4_run.py에 더 넘길 인자)
"""
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

META = "http://metadata.google.internal/computeMetadata/v1/"


def meta(path):
    return urllib.request.urlopen(urllib.request.Request(META + path, headers={"Metadata-Flavor": "Google"}), timeout=10).read().decode()


token = json.loads(meta("instance/service-accounts/default/token"))["access_token"]
url = f"https://storage.googleapis.com/storage/v1/b/{os.environ['BUCKET']}/o/speed_pack.tar?alt=media"
with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + token}), timeout=600) as r, open("/tmp/p.tar", "wb") as f:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
tarfile.open("/tmp/p.tar").extractall("/w")
os.chdir("/w")
print("묶음 받음", os.path.getsize("/tmp/p.tar") // 1000000, "MB, CPU", os.cpu_count(), flush=True)
cmd = [sys.executable, "experiments/grid4/grid4_run.py", "--url", os.environ["TARGET"], "--auth", "metadata", "--speed", "--tag", "cloud"] + os.environ.get("SPEED_ARGS", "").split()
rc = subprocess.run(cmd).returncode
out = "experiments/grid4/results/h2o-rtx__speed_cloud.json"
if os.path.exists(out):
    print("RESULT_JSON " + json.dumps(json.load(open(out)), ensure_ascii=False), flush=True)
sys.exit(rc)
