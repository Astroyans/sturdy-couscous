from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

try:
    from websockets.sync.server import serve as ws_serve
except ImportError:  # pragma: no cover
    ws_serve = None


def get_local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


class ExpoLANServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        http_port: int = 8000,
        ws_port: int = 8001,
    ) -> None:
        self.host = host
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)

        self._lock = threading.Lock()
        self._latest_json_bytes: bytes = b'{"node":"drone_cv","status":"warming_up"}'
        self._goal_lock = threading.Lock()
        self._pending_goal_cmd: Optional[dict[str, Any]] = None

        self._ws_clients_lock = threading.Lock()
        self._ws_clients: set[Any] = set()

        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._ws_server: Any = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    def publish(self, payload: dict[str, Any]) -> None:
        raw_str = json.dumps(payload, separators=(",", ":"))
        raw_bytes = raw_str.encode("utf-8")
        with self._lock:
            self._latest_json_bytes = raw_bytes

        with self._ws_clients_lock:
            clients = list(self._ws_clients)

        dead = []
        for ws in clients:
            try:
                ws.send(raw_str)
            except Exception:
                dead.append(ws)

        if dead:
            with self._ws_clients_lock:
                for ws in dead:
                    self._ws_clients.discard(ws)

    def pop_goal_command(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            cmd = self._pending_goal_cmd
            self._pending_goal_cmd = None
            return cmd

    def _push_goal_command(self, raw_msg: Any) -> bool:
        try:
            data = json.loads(raw_msg) if isinstance(raw_msg, (str, bytes)) else raw_msg
            if not isinstance(data, dict):
                return False
            if any(k in data for k in ("x_m", "y_m", "norm_x", "norm_y", "clear")):
                with self._goal_lock:
                    self._pending_goal_cmd = data
                return True
        except Exception:
            pass
        return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        parent = self

        class Handler(BaseHTTPRequestHandler):
            def _send_cors(self, status: int = 200, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def do_OPTIONS(self) -> None:
                self._send_cors(204)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/") or "/"
                if path in ("/", "/health", "/api/health"):
                    self._send_cors(200)
                    self.wfile.write(b'{"ok":true,"node":"drone_cv"}')
                elif path in ("/api/latest", "/telemetry", "/api/telemetry"):
                    with parent._lock:
                        body = parent._latest_json_bytes
                    self._send_cors(200)
                    self.wfile.write(body)
                else:
                    self._send_cors(404)
                    self.wfile.write(b'{"error":"not_found"}')

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path in ("/api/goal", "/goal"):
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    ok = parent._push_goal_command(raw)
                    self._send_cors(200 if ok else 400)
                    resp = {"ok": ok}
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                else:
                    self._send_cors(404)
                    self.wfile.write(b'{"error":"not_found"}')

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._http_server = ThreadingHTTPServer((self.host, self.http_port), Handler)
        self.http_port = int(self._http_server.server_address[1])
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        )
        self._http_thread.start()

        if ws_serve is not None and self.ws_port >= 0:
            def _ws_handler(ws: Any) -> None:
                with parent._ws_clients_lock:
                    parent._ws_clients.add(ws)
                try:
                    with parent._lock:
                        initial = parent._latest_json_bytes.decode("utf-8")
                    ws.send(initial)
                    for message in ws:
                        parent._push_goal_command(message)
                except Exception:
                    pass
                finally:
                    with parent._ws_clients_lock:
                        parent._ws_clients.discard(ws)

            try:
                self._ws_server = ws_serve(_ws_handler, self.host, self.ws_port)
                sock = self._ws_server.socket
                if sock is not None:
                    self.ws_port = int(sock.getsockname()[1])
                self._ws_thread = threading.Thread(
                    target=self._ws_server.serve_forever, daemon=True
                )
                self._ws_thread.start()
            except Exception:
                self._ws_server = None

    def stop(self) -> None:
        self._running = False
        if self._ws_server is not None:
            try:
                self._ws_server.shutdown()
            except Exception:
                pass
            self._ws_server = None
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
            self._http_server = None
