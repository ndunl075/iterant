"""Probe-only logging forwarder: 127.0.0.1:11435 -> 127.0.0.1:11434, appends each exchange to a jsonl file."""
import http.server
import json
import sys
import urllib.request

LOG = sys.argv[1]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        req = urllib.request.Request(
            "http://127.0.0.1:11434" + self.path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            r = urllib.request.urlopen(req, timeout=1800)
            data, code, ct = r.read(), r.status, r.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e:
            data, code, ct = e.read(), e.code, "application/json"
        with open(LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "path": self.path,
                        "req": body.decode("utf-8", "replace"),
                        "status": code,
                        "resp": data.decode("utf-8", "replace"),
                    }
                )
                + "\n"
            )
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            r = urllib.request.urlopen("http://127.0.0.1:11434" + self.path)
            data, code = r.read(), r.status
        except Exception:
            data, code = b"{}", 502
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", 11435), H).serve_forever()
