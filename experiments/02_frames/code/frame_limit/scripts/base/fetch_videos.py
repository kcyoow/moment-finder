"""정지 프레임 한 장으로는 판정하기 어려운 영상 모음을 내려받는다.

출처는 세 곳이다.
- TVBench (Hugging Face `FunAILab/TVBench`): 한 장이나 글만으로는 못 풀게 만든 벤치마크
- TempCompass (Hugging Face `lmms-lab/TempCompass`): 같은 영상의 원본과 역재생 쌍. 어느 프레임을
  뽑아도 두 영상의 화면이 같아서, 한 장만 보는 판정으로는 둘을 구분할 수 없다
- Wikimedia Commons: CC 라이선스 영상 (동작이 움직임으로만 구분되는 것)

벤치마크의 zip은 수백 MB에서 수 GB라서 통째로 받지 않는다. HTTP Range 요청으로 zip 목록을 읽고
고른 파일만 꺼낸다. 표준 라이브러리만 쓴다.

실행: python3 experiments/frame_limit/scripts/base/fetch_videos.py
결과: experiments/frame_limit/data/videos/<출처>/<파일>, experiments/frame_limit/data/videos.json
"""
import hashlib
import io
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = "moment-finder/0.1 (video understanding research script; python-urllib)"
HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
OUT = HERE / "data" / "videos"
MANIFEST = HERE / "data" / "videos.json"

TVBENCH = "https://huggingface.co/datasets/FunAILab/TVBench/resolve/main/"
TEMPCOMPASS_ZIP = "https://huggingface.co/datasets/lmms-lab/TempCompass/resolve/main/tempcompass_videos.zip"
TEMPCOMPASS_QA = "https://raw.githubusercontent.com/llyx97/TempCompass/main/questions/yes_no.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# 과제 이름 -> (종류, 고른 파일들)
TVBENCH_PICKS = {
    "action_localization": ("언제 일어나나", ["LR9KL.mp4", "QOYH2.mp4", "98W87.mp4"]),
    "action_sequence": ("순서", ["NKCXF.mp4", "43FG9.mp4"]),
    "action_count": ("횟수", ["video_5852.mp4", "video_11586.mp4"]),
    "moving_direction": ("방향", ["video_12661.mp4", "video_12504.mp4"]),
    "unexpected_action": ("주관적 (재미있는·인상적인 부분)", ["H_A_31_2511_2980.mp4", "M_Z_10_5793_5923.mp4"]),
}
# 원본 id. 각 id마다 원본과 `_reverse`(역재생)를 함께 받는다
TEMPCOMPASS_PICKS = {
    "1091143117": "변화 (눈을 뜸 / 감음)",
    "1034419625": "방향 (왼쪽에서 오른쪽 / 반대)",
    "1111657885": "방향 (카메라 쪽으로 / 멀어짐)",
    "1083845854": "방향 (시계 방향 / 반대)",
}
# 파일 제목 -> (종류, 한 장으로 헷갈리는 상대)
COMMONS_PICKS = {
    "File:HandWaveExample.webm": ("움직임 자체", "손 흔들기 / 손을 들고만 있기"),
    "File:The Coal Minig Museum of Slovenia, opening and closing of wooden doors.webm":
        ("방향", "문이 열림 / 닫힘"),
    "File:Jumping jacks and burpees.webm": ("횟수·동작 구분", "팔 벌려 뛰기 / 버피, 몇 번 했나"),
    "File:Applaudissements à la Wikiconvention Francophone 2022.webm":
        ("움직임 자체", "수어 박수(손 흔들기) / 손을 들고만 있기"),
}

LICENSE_TVBENCH = ("데이터셋 카드는 CC BY 4.0, 저장소의 license_notice.txt는 Apache 2.0. "
                   "원본 영상은 Charades-STA, STAR, Perception Test, CLEVRER, FunQA에서 온 것")
LICENSE_TEMPCOMPASS = "CC BY-NC 4.0, 학술 연구 전용 (GitHub README). 원본 영상은 ShutterStock"


def http_get(url, headers=None, timeout=120):
    h = {"User-Agent": UA}
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


class HttpRangeFile(io.RawIOBase):
    """HTTP Range 요청으로 읽는 읽기 전용 파일. zip에서 일부 파일만 꺼낼 때 쓴다."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache, self.fetched = url, block, 0, {}, 0
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            lo, hi = i * self.block, min((i + 1) * self.block, self.size) - 1
            data = http_get(self.url, {"Range": f"bytes={lo}-{hi}"}, timeout=300)
            self.cache[i] = data
            self.fetched += len(data)
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.block)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def mp4_duration(path):
    """mp4의 mvhd 상자에서 길이(초)를 읽는다. 못 읽으면 None."""
    data = path.read_bytes()
    i = data.find(b"mvhd")
    if i < 0:
        return None
    version = data[i + 4]
    if version == 1:
        timescale, duration = struct.unpack(">IQ", data[i + 24:i + 36])
    else:
        timescale, duration = struct.unpack(">II", data[i + 16:i + 24])
    return round(duration / timescale, 2) if timescale else None


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def file_info(path):
    return {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fetch_tvbench(entries):
    for task, (kind, files) in TVBENCH_PICKS.items():
        ann = json.loads(http_get(TVBENCH + f"json/{task}.json"))
        z = None
        for name in files:
            dest = OUT / "tvbench" / task / name
            if not dest.exists():
                z = z or open_remote_zip(TVBENCH + f"video/{task}.zip")
                save(dest, z.read(f"{task}/{name}"))  # zipfile이 CRC를 확인한다
                print("받음", dest.relative_to(HERE))
            qa = [{k: e[k] for k in ("question", "candidates", "answer", "start", "end") if k in e}
                  for e in ann if e["video"] == name]
            entries.append({
                "id": f"tvbench_{task}_{Path(name).stem}", "source": "TVBench", "task": task, "kind": kind,
                "file": str(dest.relative_to(HERE)), "duration": mp4_duration(dest), **file_info(dest),
                "license": LICENSE_TVBENCH,
                "page": "https://huggingface.co/datasets/FunAILab/TVBench", "qa": qa,
            })


def fetch_tempcompass(entries):
    yes_no = json.loads(http_get(TEMPCOMPASS_QA))
    z = None
    for vid, kind in TEMPCOMPASS_PICKS.items():
        for name, variant in ((vid, "원본"), (vid + "_reverse", "역재생")):
            dest = OUT / "tempcompass" / f"{name}.mp4"
            if not dest.exists():
                z = z or open_remote_zip(TEMPCOMPASS_ZIP)
                save(dest, z.read(f"videos/{name}.mp4"))
                print("받음", dest.relative_to(HERE))
            entries.append({
                "id": f"tempcompass_{name}", "source": "TempCompass", "kind": kind, "variant": variant,
                "pair": f"tempcompass_{vid}" if variant == "역재생" else f"tempcompass_{vid}_reverse",
                "file": str(dest.relative_to(HERE)), "duration": mp4_duration(dest), **file_info(dest),
                "license": LICENSE_TEMPCOMPASS,
                "page": "https://huggingface.co/datasets/lmms-lab/TempCompass",
                "yes_no": yes_no.get(name, {}),
            })


def fetch_commons(entries):
    params = {"action": "query", "format": "json", "titles": "|".join(COMMONS_PICKS), "prop": "imageinfo",
              "iiprop": "url|size|mime|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist"}
    data = json.loads(http_get(COMMONS_API + "?" + urllib.parse.urlencode(params)))
    names = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
    for page in data["query"]["pages"].values():
        title = names.get(page["title"], page["title"])
        if "imageinfo" not in page:
            print("Commons에 없음:", title)
            continue
        kind, confusable = COMMONS_PICKS[title]
        ii = page["imageinfo"][0]
        # URL 끝에 추적용 쿼리(?utm_...)가 붙어 오므로 경로 부분만 파일 이름으로 쓴다
        dest = OUT / "commons" / urllib.parse.unquote(urllib.parse.urlparse(ii["url"]).path.rsplit("/", 1)[1])
        if not dest.exists():
            time.sleep(5)  # Wikimedia 요청 제한을 피하려고 천천히 받는다
            save(dest, http_get(ii["url"], timeout=600))
            print("받음", dest.relative_to(HERE))
        meta = ii.get("extmetadata", {})
        entries.append({
            "id": "commons_" + dest.stem[:40], "source": "Wikimedia Commons", "kind": kind,
            "confusable": confusable, "file": str(dest.relative_to(HERE)),
            "duration": round(float(ii["duration"]), 2) if ii.get("duration") else None,
            "w": ii.get("width"), "h": ii.get("height"), **file_info(dest),
            "license": meta.get("LicenseShortName", {}).get("value"),
            "artist": meta.get("Artist", {}).get("value"), "page": ii.get("descriptionurl"), "url": ii["url"],
        })


def main():
    entries, failed = [], []
    for name, fn in (("TVBench", fetch_tvbench), ("TempCompass", fetch_tempcompass), ("Commons", fetch_commons)):
        try:
            fn(entries)
        except Exception as e:  # noqa: BLE001  한 출처가 막혀도 나머지는 받는다
            failed.append(f"{name}: {e!r}")
            print("실패", name, repr(e), file=sys.stderr)
    note = ("정지 프레임 한 장으로는 판정하기 어려운 영상 모음. 벤치마크 영상은 실험용으로만 쓰고 "
            "다시 배포하지 않는다. 한 장 판정이 실제로 틀리는지는 아직 확인하지 않았다.")
    MANIFEST.write_text(json.dumps({"_note": note, "videos": entries}, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{len(entries)}개 기록 -> {MANIFEST.relative_to(HERE)}")
    if failed:
        print("받지 못한 출처:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
