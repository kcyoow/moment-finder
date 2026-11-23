"""배포한 H2OSnap의 check(또는 --diag면 diag)를 부르고, 부른 쪽에서 본 전체 시간과 그 안에서 잰 값을 찍는다.
전체 시간 − 측정이 돈 시간 = 컨테이너가 뜨고(또는 스냅샷에서 되살아나고) 깨어나기까지의 시간.
실행: MODAL_PROFILE=<modal-profile> <modal이 깔린 파이썬> snap_client.py [--full | --diag | --lite] [--tag 이름] [--variant 이름] [--calls N]

2026-11-14 추가
- --variant: h2o_snap_modal.py의 변형(앱 이름 moment-finder-h2o-snap-<이름>, base는 moment-finder-h2o-snap)
- --lite: diag(lite=True)
- 부르기 전마다 이 앱의 컨테이너가 하나도 없을 때까지 기다린다(`modal container list --app-id <앱 ID> --json`).
  앱 목록에서 이름을 찾는 방식은 이름이 잘려 보여 실패한 적이 있어(README) 앱 ID로 본다.
  그래서 매 호출은 새 컨테이너(스냅샷을 만드는 컨테이너 또는 되살린 컨테이너)가 받는다. 되살렸는지는 결과의 since_snap(> 30초)으로 본다
- 부르는 동안 이 앱에 뜬 컨테이너 ID를 모아 containers_seen에 적는다(호출마다 다른 컨테이너였는지 확인용)
- --calls N: N번 되풀이(태그 뒤에 _1, _2…)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import modal

MODAL = shutil.which("modal") or os.path.expanduser("~/.local/bin/modal")


def app_name(variant):
    return "moment-finder-h2o-snap" if variant == "base" else f"moment-finder-h2o-snap-{variant}"


def app_id_of(name):
    """배포된 앱의 ID(이름이 정확히 같은 것)."""
    return modal.App.lookup(name).app_id


def running_containers(aid):
    """이 앱 ID로 떠 있는 컨테이너 ID 목록. MODAL_PROFILE은 부른 쪽 환경을 그대로 쓴다."""
    r = subprocess.run([MODAL, "container", "list", "--app-id", aid, "--json"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"modal container list 실패: {r.stderr[-300:]}")
    rows = json.loads(r.stdout or "[]")   # Modal 1.5.5: 키는 container_id, app_id, app_name, start_time
    return [row.get("container_id") or "?" for row in rows if row.get("app_id", aid) == aid]


def wait_no_containers(aid, timeout=240):
    """이 앱의 컨테이너가 다 꺼질 때까지 기다린다(scaledown_window=5초). 기다린 초를 돌려준다."""
    t = time.time()
    while True:
        cs = running_containers(aid)
        if not cs:
            return round(time.time() - t, 1)
        if time.time() - t > timeout:
            raise RuntimeError(f"{aid}: {timeout}초가 지나도 컨테이너 {len(cs)}개가 떠 있음 {cs}")
        time.sleep(3)


class ContainerWatch:
    """with 블록 동안 5초마다 이 앱의 컨테이너 ID를 모은다."""

    def __init__(self, aid, every=5.0):
        self.aid, self.every, self.seen, self._stop = aid, every, set(), threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.seen.update(c for c in running_containers(self.aid) if c)
            except Exception:
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=70)
        return False


def summary(out):
    """한 줄 요약(표준 오류로)."""
    w16 = out.get("image_w16") or {}
    api = {k: v for k, v in (w16.get("process_detail") or {}).items() if k == "vllm" or "api" in k.lower()}   # API 서버(들)
    return {"variant": out.get("variant"), "tag": out.get("tag"), "restored": out.get("restored_from_snapshot"), "since_snap": out.get("since_snap"),
            "cold_start_s": out.get("cold_start_seconds"), "w16": w16.get("requests_per_s"), "w16_b": (out.get("image_w16_b") or {}).get("requests_per_s"),
            "w16_again": (out.get("image_w16_again") or {}).get("requests_per_s"), "gpu": w16.get("gpu_util_mean"),
            "api_cpu_s_per_req": {k: v.get("cpu_s_per_req") for k, v in api.items()}, "flips": (out.get("versus") or {}).get("flips"),
            "rps_16_check": out.get("requests_per_s_16"), "wake_error": out.get("wake_error"), "containers": out.get("containers_seen")}


def main(argv):
    val = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
    variant, tag, calls = val("--variant", "base"), val("--tag", ""), int(val("--calls", "1"))
    name = app_name(variant)
    aid = app_id_of(name)
    H2OSnap = modal.Cls.from_name(name, "H2OSnap")
    for i in range(calls):
        t_tag = f"{tag}_{i + 1}" if tag and calls > 1 else tag
        waited = wait_no_containers(aid)
        with ContainerWatch(aid) as watch:
            t0 = time.time()
            if "--diag" in argv or "--lite" in argv:
                out = H2OSnap().diag.remote(tag=t_tag, lite="--lite" in argv)
            else:
                out = H2OSnap().check.remote(full="--full" in argv, tag=t_tag)
            wall = time.time() - t0
        out.pop("answers", None)
        out.update({"variant": out.get("variant", variant), "app_id": aid, "client_wall_seconds": round(wall, 1),
                    "waited_for_scaledown_s": waited, "containers_seen": sorted(watch.seen)})
        if "check_seconds" in out:
            out["cold_start_seconds"] = round(wall - out["check_seconds"], 1)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        print("SUMMARY " + json.dumps(summary(out), ensure_ascii=False), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
