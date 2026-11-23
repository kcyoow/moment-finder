"""pipeline/runs를 이 Mac 안에서만(127.0.0.1) 내놓는 작은 서버. 보기 페이지(index.html)의 클립 영상을 재생하고 감을 수 있게 한다.

파일을 직접 여는 것(file://)으로는 앱의 브라우저 창에서 영상이 재생되지 않아서 쓴다. 영상의 구간 요청(Range)에 답한다.
실행: python3 pipeline/serve.py [포트]   (기본 8765. 앱에서는 .claude/launch.json의 highlight-clips로 띄운다)
"""
import html
import http.server
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "runs"


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):          # 조용히
        pass

    def listing(self):
        rows = []
        for d in sorted(p for p in ROOT.iterdir() if (p / "index.html").exists()):
            try:
                info, cj = json.loads((d / "run.json").read_text(encoding="utf-8")), json.loads((d / "clips.json").read_text(encoding="utf-8"))
                note = f"영상 {mmss(info['duration'])} · 클립 {len(cj['clips'])}개 · {html.escape(str(info.get('viewer_request') or ''))}"
            except Exception:
                note = ""
            rows.append(f'<li><a href="/{d.name}/index.html">{html.escape(d.name)}</a><br><small>{note}</small></li>')
        body = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>하이라이트 실행 목록</title><style>body{font-family:-apple-system,system-ui,sans-serif;margin:24px auto;max-width:760px;padding:0 16px;"
                "background:#fff;color:#111}li{margin:14px 0}small{color:#666}</style></head><body><h1>하이라이트 실행 목록</h1><ul>"
                + "".join(rows) + "</ul></body></html>").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            return self.listing()
        path = self.translate_path(self.path)
        rng = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        a, b = rng.group(1), rng.group(2)
        start = int(a) if a else max(0, size - int(b or 0))
        end = min(int(b), size - 1) if (a and b) else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as srv:
        print(f"http://127.0.0.1:{port}/  (폴더 {ROOT})", flush=True)
        srv.serve_forever()
