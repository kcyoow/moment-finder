"""OpenAI VPT의 마인크래프트 플레이 기록(MIT)에서 5분짜리 조각을 골라 받는다. 표준 라이브러리만 쓴다.

기록 하나는 영상(mp4, 640x360, 초당 20장)과 틱마다 한 줄인 행동 기록(jsonl)으로 되어 있다.
행동 기록의 isGuiOpen(화면 창이 열려 있는가), isGuiInventory(그 창이 인벤토리인가)가 장면 찾기의 정답이 된다.

고르는 규칙 (받기 전에 정함. 영상 내용이나 모델의 답을 보고 고르지 않는다)
- 목록: 10.0판(다이아몬드 곡괭이 만들기 과제)의 색인
- 씨앗 20261111로 섞은 순서대로 보면서, 참여자마다 조각 하나만 고른다
- 행동 기록을 먼저 받아 두 조건을 본다: 길이가 295초 이상(끊기지 않은 5분 조각), 창이 열린 구간이 하나 이상
- 조건에 맞는 조각이 --count개가 되면 멈추고, 그 조각들의 영상을 받는다

실행: python3 -I experiments/frame_limit/scripts/mc/mc_fetch.py --dry-run   (행동 기록만 받아 고르고, 영상 크기를 합산해 출력)
      python3 -I experiments/frame_limit/scripts/mc/mc_fetch.py             (영상까지 받는다)
      --count 12   고를 조각 수
      --max-gb 2.3 영상 크기 합이 이 값을 넘으면 받지 않고 멈춘다
      --workers 6  함께 받는 영상 수 (연결 하나는 초당 0.6MB쯤으로 느리다)
결과: data/mc/<이름>.jsonl, data/mc/<이름>.mp4, data/mc/videos.json(고른 조각, 창이 열린 구간, 파일 크기)
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
OUT = HERE / "data" / "mc"
INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json"
SEED = 20261111
TICKS_PER_SECOND = 20
MIN_SECONDS = 295
NAME = re.compile(r"^data/(?P<ver>[\d.]+)/(?P<alias>.+)-(?P<session>[0-9a-f]{12})-(?P<date>\d{8})-(?P<time>\d{6})\.mp4$")


def fetch(url, path):
    """파일을 받는다. 받다 만 조각(.part)이 있으면 그 뒤부터 이어받는다."""
    if path.exists() and path.stat().st_size > 0:
        return path.stat().st_size
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):  # 끊기면 이어받기로 다시 시도한다
        have = tmp.stat().st_size if tmp.exists() else 0
        # 이 저장소는 x-ms-version을 함께 보내야 이어받기(Range)를 받아 준다. 없으면 처음부터 다시 보낸다
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "x-ms-version": "2019-12-12"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp, "ab" if have and resp.status == 206 else "wb") as f:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
            break
        except urllib.error.HTTPError as err:
            if err.code == 416:  # 이미 끝까지 받은 조각
                break
            if attempt == 4 or err.code in (403, 404):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
    tmp.rename(path)
    return path.stat().st_size


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def runs(flags):
    """참인 연속 구간을 [시작 틱, 끝 틱(포함)]으로 묶는다."""
    out, start = [], None
    for k, f in enumerate(flags):
        if f and start is None:
            start = k
        elif not f and start is not None:
            out.append([start, k - 1])
            start = None
    if start is not None:
        out.append([start, len(flags) - 1])
    return out


def read_actions(path):
    gui, inv = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                gui.append(bool(d["isGuiOpen"]))
                inv.append(bool(d["isGuiInventory"]))
    return gui, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--max-gb", type=float, default=2.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fetch(INDEX_URL, OUT / "index_10xx.json")
    index = json.loads((OUT / "index_10xx.json").read_text())
    base = index["basedir"]
    paths = [p for p in index["relpaths"] if NAME.match(p)]
    print(f"색인: 조각 {len(index['relpaths'])}개 중 이름 형식에 맞는 것 {len(paths)}개, "
          f"참여자 {len({NAME.match(p)['alias'] for p in paths})}명")
    order = sorted(paths)
    random.Random(SEED).shuffle(order)
    chosen, seen, looked = [], set(), 0
    for rel in order:
        m = NAME.match(rel)
        if m["alias"] in seen:
            continue
        name = Path(rel).stem
        looked += 1
        try:
            fetch(base + rel[:-4] + ".jsonl", OUT / f"{name}.jsonl")
        except Exception as err:  # 행동 기록이 없는 조각은 건너뛴다
            print(f"  건너뜀 {name}: 행동 기록을 받지 못함 ({type(err).__name__})")
            continue
        gui, inv = read_actions(OUT / f"{name}.jsonl")
        seconds = len(gui) / TICKS_PER_SECOND
        gui_runs = runs(gui)
        if seconds < MIN_SECONDS or not gui_runs:
            print(f"  건너뜀 {name}: 길이 {seconds:.0f}초, 창이 열린 구간 {len(gui_runs)}개")
            (OUT / f"{name}.jsonl").unlink()
            continue
        seen.add(m["alias"])
        to_seconds = lambda rs: [[round(a / TICKS_PER_SECOND, 2), round((b + 1) / TICKS_PER_SECOND, 2)] for a, b in rs]
        chosen.append({"name": name, "alias": m["alias"], "session": m["session"], "url": base + rel,
                       "ticks": len(gui), "seconds": round(seconds, 2),
                       "gui_open": to_seconds(gui_runs), "gui_inventory": to_seconds(runs(inv)),
                       "gui_share": round(sum(gui) / len(gui), 3), "inventory_share": round(sum(inv) / len(inv), 3)})
        if len(chosen) == args.count:
            break
    print(f"행동 기록을 본 조각 {looked}개 중 {len(chosen)}개를 골랐다")
    total = 0
    for c in chosen:
        c["bytes"] = size_of(c["url"])
        total += c["bytes"]
        lengths = sorted(round(b - a, 1) for a, b in c["gui_open"])
        print(f"  {c['name']}: {c['seconds']:.0f}초, 영상 {c['bytes'] / 1e6:.0f}MB, 창이 열린 구간 {len(lengths)}개"
              f"(화면의 {c['gui_share']:.0%}, 가장 짧은 것 {lengths[0]}초, 중앙값 {lengths[len(lengths) // 2]}초)")
    print(f"영상 크기 합 {total / 1e9:.2f}GB")
    (OUT / "videos.json").write_text(json.dumps({
        "_note": "mc_fetch.py가 고른 조각. gui_open, gui_inventory는 [시작 초, 끝 초] 목록(행동 기록의 틱을 20으로 나눈 값).",
        "source": "https://github.com/openai/Video-Pre-Training (contractor data 10.0, MIT)",
        "seed": SEED, "videos": chosen}, ensure_ascii=False, indent=1) + "\n")
    if args.dry_run:
        return
    if total / 1e9 > args.max_gb:
        raise SystemExit(f"영상 크기 합이 {args.max_gb}GB를 넘어서 받지 않는다. --count를 줄이거나 --max-gb를 올린다")
    def grab(c):
        got = fetch(c["url"], OUT / f"{c['name']}.mp4")
        assert got == c["bytes"], f"{c['name']}: 받은 크기가 다르다 ({got} != {c['bytes']})"
        return c["name"], got

    with cf.ThreadPoolExecutor(args.workers) as pool:  # 연결 하나는 느려서 여러 개를 함께 받는다
        for k, (name, got) in enumerate(pool.map(grab, chosen), 1):
            print(f"  받음 {k}/{len(chosen)} {name}.mp4 {got / 1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
