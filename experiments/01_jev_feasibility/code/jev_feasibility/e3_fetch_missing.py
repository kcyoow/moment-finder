# @title E3 prep: unpack locally-downloaded Commons thumbnails, fetch the rest politely, tar the new ones
import csv
import tarfile
import urllib.error
import urllib.request

UA = "moment-finder-eval/0.1 (personal research script)"
IMGS = WORK / "imgs"
if not IMGS.exists():
    with tarfile.open(WORK / "imgs109.tgz") as tar:
        tar.extractall(WORK, filter="data")
rows = list(csv.DictReader(open(WORK / "manifest.csv")))
new = []
for row in rows:
    dest = IMGS / f"{row['id']}.jpg"
    if dest.exists():
        continue
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(row["thumburl"], headers={"User-Agent": UA}), timeout=60) as r:
                dest.write_bytes(r.read())
            new.append(dest)
            break
        except urllib.error.HTTPError as err:
            if err.code in (429, 503):
                time.sleep(min(60, int(err.headers.get("Retry-After") or 10 * (attempt + 1))))
                continue
            print("skip", row["id"], err.code)
            break
    time.sleep(1.0)
with tarfile.open(WORK / "imgs_new.tgz", "w:gz") as tar:
    for p in new:
        tar.add(p, arcname=f"imgs/{p.name}")
print(f"rows {len(rows)} | fetched {len(new)} | total files {len(list(IMGS.glob('*.jpg')))}")
