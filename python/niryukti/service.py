"""Local authenticated synchronous HTTP API; subprocess-isolated solver requests."""
import hmac
import json
import os
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import Model, __version__
from .report import render_report

def make_server(host="127.0.0.1", port=8090, *, token=None, workers=2, max_time=300):
    token = token or os.environ.get("NIRYUKTI_API_TOKEN")
    if not token:
        raise ValueError("Set NIRYUKTI_API_TOKEN before starting the API service")
    if workers < 1 or workers > 16 or not (0 < max_time <= 3600):
        raise ValueError("Invalid API resource limits")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number): raise ValueError("Nonfinite JSON number")
        return number
    capacity = threading.BoundedSemaphore(workers)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log credentials/model bodies.
        def send(self, status, data, content_type="application/json"):
            body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + token).encode("utf-8")):
                self.send(401, {"error":"Bearer token required"}); return False
            return True
        def do_GET(self):
            if not self.authorized(): return
            if self.path != "/v1/health": return self.send(404, {"error":"Unknown route"})
            self.send(200, {"service":"NIRYUKTI", "version":__version__, "mode":"synchronous", "max_parallel_solves":workers})
        def do_POST(self):
            if not self.authorized(): return
            if self.path not in ("/v1/solve", "/v1/report"):
                return self.send(404, {"error":"Unknown route"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    return self.send(413, {"error":"JSON payload limit: 8 MiB"})
                self.connection.settimeout(30)
                data = json.loads(self.rfile.read(length), parse_float=finite_float, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                if not isinstance(data, dict): raise ValueError("Expected JSON object")
                if self.path == "/v1/report":
                    if set(data) - {"result","title"}: raise ValueError("Unknown report fields")
                    return self.send(200, render_report(data["result"], title=str(data.get("title","NIRYUKTI solve report"))), "text/html; charset=utf-8")
                if set(data) - {"model", "options"}: raise ValueError("Unknown solve fields")
                if not isinstance(data.get("model"), dict): raise ValueError("Provide an inline native JSON model")
                options = data.get("options", {})
                if not isinstance(options, dict) or set(options) - {"device","method","tol","time_limit","threads","iterations"}:
                    raise ValueError("Unsupported solve options")
                if not 0 < float(options.get("time_limit",60)) <= max_time:
                    raise ValueError("Time limit exceeds API resource policy")
                if not 1 <= int(options.get("threads",1)) <= 64:
                    raise ValueError("Threads must be 1..64")
                if not capacity.acquire(blocking=False):
                    return self.send(429, {"error":"All solver workers are occupied"})
                try:
                    model = Model(); model.data = data["model"]
                    result = model.solve(**options)
                    self.send(200, result)  # Limit/unsupported outcomes retain their solver status.
                finally: capacity.release()
            except (ValueError, TypeError, OverflowError, KeyError, RuntimeError, OSError) as error:
                self.send(400, {"error":str(error)[:2048]})
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
