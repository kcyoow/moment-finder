"""pipeline: 웹사이트용 판정 모델 서버. Modal(<modal-profile>)에 올려 두는(deploy) 앱이다.

mf_modal.py는 한 번 돌고 꺼지는 작업이고, 이것은 요청이 오면 컨테이너가 뜨고 한동안 요청이 없으면 스스로 꺼지는 앱이다.
컨테이너가 뜰 때 H2O-Lightning-4B 서버(serve.sh)를 띄우고(약 3~5분), 떠 있는 동안에는 영상 여러 개를 바로 받는다.
서버 설정과 답 검사는 mf_modal.py와 같다.

올리기: MODAL_PROFILE=<modal-profile> modal deploy pipeline/mf_serve_modal.py
내리기: MODAL_PROFILE=<modal-profile> modal app stop moment-finder-serve --yes
값: 떠 있는 동안 초당 $0.001018(시간당 약 $3.66). 요청이 끊기고 SCALEDOWN초 뒤에 꺼지면 값이 나가지 않는다.

쓰는 법(이 Mac의 웹 서버가 한다):
  1) 볼륨 mf-pipeline에 <이름>/pack.tar, <이름>/requests.json을 올린다(mf.py prepare가 만든 것)
  2) Judge().run.remote_gen(이름)을 부르면 진행 상황을 하나씩 돌려주고 마지막에 답을 돌려준다
  3) 끝나면 볼륨의 <이름> 폴더를 지운다(프레임을 남기지 않는다)
"""
import modal

from mf_modal import GRAPH, H2O_REVISION, MAX_PIXELS, PORT, VLLM_ARGS, gpu_image, parse   # 같은 폴더의 mf_modal.py

app = modal.App("moment-finder-serve")
data = modal.Volume.from_name("mf-pipeline", create_if_missing=True)
models = modal.Volume.from_name("mf-models")
cache_vol = modal.Volume.from_name("mf-vllm-cache").read_only()
SCALEDOWN = 300          # 마지막 요청 뒤 이만큼(초) 지나면 컨테이너가 꺼진다
WORKERS = 16             # 판정 모델에 동시에 보내는 요청 수(컨테이너 전체)


@app.cls(image=gpu_image.add_local_python_source("mf_modal"), gpu="RTX-PRO-6000", volumes={"/data": data, "/models": models, "/cache": cache_vol},
         timeout=3600, scaledown_window=SCALEDOWN, max_containers=1, cpu=8, memory=32768)
@modal.concurrent(max_inputs=6)
class Judge:
    @modal.enter()
    def start(self):
        import hashlib
        import os
        import subprocess
        import threading
        import time
        import urllib.request
        t0 = time.time()
        if not (os.path.exists("/models/h2o/serve.sh") and open("/models/h2o/_complete").read().strip() == H2O_REVISION):
            raise RuntimeError("mf-models에 H2O 가중치(정한 판)가 없다")
        home = "/tmp/h2ohome"
        os.makedirs(home, exist_ok=True)
        key = hashlib.sha256(f"{GRAPH}|{VLLM_ARGS}".encode()).hexdigest()[:12]
        cache_tar = f"/cache/vllm_cache_{key}.tar"
        if os.path.exists(cache_tar):
            subprocess.run(["tar", "-xf", cache_tar, "-C", home], check=True)
        env = {**os.environ, "HOME": home, "TWO_WAY": "0", "VLLM_PATCHES": "1", "VLLM_ARGS": VLLM_ARGS, "CUDA_GRAPH_MAX_TOKENS": str(GRAPH),
               "MAX_PIXELS": str(MAX_PIXELS), "SHIM_MAX_PIXELS": str(MAX_PIXELS), "SHIM_QUESTION_THREADS": "512", "SHIM_TIMEOUT": "600", "PORT": str(PORT)}
        self.server = subprocess.Popen(["bash", "/models/h2o/serve.sh"], stdout=open("/tmp/serve.log", "w"), stderr=subprocess.STDOUT, env=env)
        while True:
            if self.server.poll() is not None or time.time() - t0 > 15 * 60:
                print(open("/tmp/serve.log").read()[-4000:])
                raise RuntimeError("판정 모델 서버가 뜨지 못했다")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=30).read()
                break
            except Exception:
                time.sleep(3)
        self.ready_seconds = round(time.time() - t0)
        self.gate = threading.Semaphore(WORKERS)
        self.vol_lock = threading.Lock()         # 볼륨을 다시 읽거나 적는 동안에는 이 컨테이너에 열린 파일이 없어야 한다(영상 여러 개를 한꺼번에 받으므로 줄을 세운다)
        print(f"판정 모델 서버 준비 {self.ready_seconds}초", flush=True)

    @modal.exit()
    def stop(self):
        self.server.terminate()

    @modal.method()
    def ping(self):
        return {"ready_seconds": self.ready_seconds}

    @modal.method()
    def run(self, name: str, keep: bool = False):
        """볼륨의 <name>을 묻는다. 진행 상황({"done", "total", "errors"})을 하나씩 돌려주고, 마지막에 {"records", "meta"}를 돌려준다."""
        import base64
        import concurrent.futures as cf
        import hashlib
        import io
        import json
        import shutil
        import tarfile
        import time
        import urllib.error
        import urllib.request
        t0 = time.time()
        with self.vol_lock:                      # 2026-11-16: 잠금 없이 셋을 한꺼번에 받았더니 둘이 "there are open files preventing the operation"으로 실패했다
            data.reload()
            with open(f"/data/{name}/requests.json") as f:
                req = json.load(f)
            with open(f"/data/{name}/pack.tar", "rb") as f:
                raw = f.read()
        if hashlib.sha256(raw).hexdigest() != req["pack_sha256"]:
            raise RuntimeError(f"{name}: 볼륨의 묶음이 요청 글의 해시와 다르다")
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        pos, questions = req["positions"], req["questions"]
        yield {"stage": "judge", "done": 0, "total": len(pos), "errors": 0, "ready_seconds": self.ready_seconds}

        def ask(i):
            q = pos[i]
            frames = [{"seconds": s, "frame": "data:image/jpeg;base64," + base64.b64encode(files[n]).decode()} for n, s in zip(q["frames"], q["seconds"])]
            body = json.dumps({"state": {**req["state"], "frames": frames}, "questions": questions}).encode()
            msg = ""
            for attempt in range(4):
                try:
                    with self.gate:
                        r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(r, timeout=300) as resp:
                            out = json.loads(resp.read())
                    return {"k": q["k"], "answers": parse(out, questions), "tokens": (out.get("usage") or {}).get("input_tokens")}
                except urllib.error.HTTPError as err:
                    msg = f"HTTP {err.code} {err.read()[:200].decode(errors='replace')}"
                except Exception as err:
                    msg = f"{type(err).__name__}: {err}"[:200]
                time.sleep(2 ** attempt)
            return {"k": q["k"], "error": msg}

        recs, errors, last = [None] * len(pos), 0, time.time()
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(ask, i): i for i in range(len(pos))}
            for n, fut in enumerate(cf.as_completed(futs), 1):
                rec = fut.result()
                recs[futs[fut]] = rec
                errors += "error" in rec
                if time.time() - last > 1.0 or n == len(pos):
                    last = time.time()
                    yield {"stage": "judge", "done": n, "total": len(pos), "errors": errors}
        if not keep:
            with self.vol_lock:
                shutil.rmtree(f"/data/{name}", ignore_errors=True)
                data.commit()
        wall = time.time() - t0
        yield {"stage": "done", "records": recs,
               "meta": {"name": name, "requests": len(pos), "answered": len(pos) - errors, "errors": errors, "wall_seconds": round(wall, 1),
                        "rate_per_second": round(len(pos) / max(wall, 1e-9), 2), "h2o_revision": H2O_REVISION, "ready_seconds": self.ready_seconds}}
