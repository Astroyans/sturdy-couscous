from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

try:
    from websockets.sync.server import serve as ws_serve
except ImportError:  # pragma: no cover
    ws_serve = None

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def get_local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def _encode_ws_frame(payload_bytes: bytes, opcode: int = 0x1) -> bytes:
    length = len(payload_bytes)
    first_byte = 0x80 | (opcode & 0x0F)
    if length < 126:
        header = struct.pack("!BB", first_byte, length)
    elif length < 65536:
        header = struct.pack("!BBH", first_byte, 126, length)
    else:
        header = struct.pack("!BBQ", first_byte, 127, length)
    return header + payload_bytes


def _read_exact(rfile: Any, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = rfile.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _read_ws_frame(rfile: Any) -> Optional[tuple[int, bytes]]:
    head = _read_exact(rfile, 2)
    if head is None:
        return None
    b1, b2 = head[0], head[1]
    opcode = b1 & 0x0F
    masked = (b2 & 0x80) != 0
    length = b2 & 0x7F
    if length == 126:
        ext = _read_exact(rfile, 2)
        if ext is None:
            return None
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = _read_exact(rfile, 8)
        if ext is None:
            return None
        length = struct.unpack("!Q", ext)[0]
    if length > 4 * 1024 * 1024:
        return None
    mask_key = _read_exact(rfile, 4) if masked else b""
    if masked and mask_key is None:
        return None
    payload = _read_exact(rfile, length) if length > 0 else b""
    if payload is None:
        return None
    if masked and mask_key:
        payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
    return opcode, payload


class ExpoLANServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        http_port: int = 8765,
        ws_port: Optional[int] = None,
    ) -> None:
        self.host = host
        self.http_port = int(http_port)
        self._separate_ws_port: Optional[int] = (
            None if ws_port is None else int(ws_port)
        )
        self.ws_port = (
            self.http_port
            if self._separate_ws_port is None
            else self._separate_ws_port
        )

        self._lock = threading.Lock()
        self._latest_json_bytes: bytes = (
            b'{"node":"drone_cv","node_id":"drone_cv","status":"warming_up"}'
        )
        self._latest_map_state_bytes: bytes = b"{}"
        self._goal_lock = threading.Lock()
        self._pending_goal_cmd: Optional[dict[str, Any]] = None

        self._ws_clients_lock = threading.Lock()
        self._ws_clients: set[Any] = set()
        self._raw_ws_clients: dict[socket.socket, threading.Lock] = {}

        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._ws_server: Any = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    def publish(
        self,
        payload: dict[str, Any],
        map_state: Optional[dict[str, Any]] = None,
    ) -> None:
        raw_str = json.dumps(payload, separators=(",", ":"))
        raw_bytes = raw_str.encode("utf-8")
        map_bytes = (
            json.dumps(map_state, separators=(",", ":")).encode("utf-8")
            if map_state is not None
            else None
        )
        with self._lock:
            self._latest_json_bytes = raw_bytes
            if map_bytes is not None:
                self._latest_map_state_bytes = map_bytes

        with self._ws_clients_lock:
            clients = list(self._ws_clients)
            raw_clients = list(self._raw_ws_clients.items())

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

        if raw_clients:
            frame = _encode_ws_frame(raw_bytes, opcode=0x1)
            dead_raw = []
            for sock, wlock in raw_clients:
                try:
                    with wlock:
                        sock.sendall(frame)
                except Exception:
                    dead_raw.append(sock)
            if dead_raw:
                with self._ws_clients_lock:
                    for sock in dead_raw:
                        self._raw_ws_clients.pop(sock, None)

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
            protocol_version = "HTTP/1.1"

            def _send_cors(
                self,
                status: int = 200,
                content_type: str = "application/json",
                content_length: int = 0,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(content_length))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Connection", "close")
                self.end_headers()

            def do_OPTIONS(self) -> None:
                self._send_cors(204, content_length=0)

            def _handle_websocket_upgrade(self) -> None:
                ws_key = (self.headers.get("Sec-WebSocket-Key") or "").strip()
                if not ws_key:
                    body = b'{"error":"missing_websocket_key"}'
                    self._send_cors(400, content_length=len(body))
                    self.wfile.write(body)
                    return

                accept_raw = hashlib.sha1((ws_key + _WS_GUID).encode("ascii")).digest()
                accept_val = base64.b64encode(accept_raw).decode("ascii")

                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept_val)
                self.end_headers()
                self.wfile.flush()

                sock = self.request
                wlock = threading.Lock()
                with parent._ws_clients_lock:
                    parent._raw_ws_clients[sock] = wlock

                try:
                    with parent._lock:
                        initial = parent._latest_json_bytes
                    with wlock:
                        sock.sendall(_encode_ws_frame(initial, opcode=0x1))

                    while parent._running:
                        parsed = _read_ws_frame(self.rfile)
                        if parsed is None:
                            break
                        opcode, payload = parsed
                        if opcode == 0x8:
                            break
                        if opcode == 0x9:
                            with wlock:
                                sock.sendall(_encode_ws_frame(payload, opcode=0xA))
                        elif opcode == 0x1:
                            parent._push_goal_command(payload.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
                finally:
                    with parent._ws_clients_lock:
                        parent._raw_ws_clients.pop(sock, None)
                    self.close_connection = True

            def do_GET(self) -> None:
                upgrade_hdr = (self.headers.get("Upgrade") or "").lower()
                if upgrade_hdr == "websocket":
                    self._handle_websocket_upgrade()
                    return

                path = self.path.split("?", 1)[0].rstrip("/") or "/"
                if path in ("/", "/health", "/api/health"):
                    body = b'{"ok":true,"node":"drone_cv","node_id":"drone_cv"}'
                    self._send_cors(200, content_length=len(body))
                    self.wfile.write(body)
                elif path in (
                    "/map_state.json",
                    "/data/real_map_state.json",
                    "/api/map_state",
                ):
                    with parent._lock:
                        body = parent._latest_map_state_bytes
                    self._send_cors(200, content_length=len(body))
                    self.wfile.write(body)
                elif path in (
                    "/latest_expo_payload.json",
                    "/latest_drone_cv.json",
                    "/data/latest_expo_payload.json",
                    "/data/latest_drone_cv.json",
                    "/api/latest",
                    "/telemetry",
                    "/api/telemetry",
                ):
                    with parent._lock:
                        body = parent._latest_json_bytes
                    self._send_cors(200, content_length=len(body))
                    self.wfile.write(body)
                else:
                    body = b'{"error":"not_found"}'
                    self._send_cors(404, content_length=len(body))
                    self.wfile.write(body)

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path in ("/api/goal", "/goal"):
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    ok = parent._push_goal_command(raw)
                    body = json.dumps({"ok": ok}).encode("utf-8")
                    self._send_cors(200 if ok else 400, content_length=len(body))
                    self.wfile.write(body)
                else:
                    body = b'{"error":"not_found"}'
                    self._send_cors(404, content_length=len(body))
                    self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._http_server = ThreadingHTTPServer((self.host, self.http_port), Handler)
        self._http_server.daemon_threads = True
        self.http_port = int(self._http_server.server_address[1])
        if self._separate_ws_port is None or self._separate_ws_port == self.http_port:
            self.ws_port = self.http_port

        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        )
        self._http_thread.start()

        if (
            ws_serve is not None
            and self._separate_ws_port is not None
            and self._separate_ws_port >= 0
            and self._separate_ws_port != self.http_port
        ):
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
                self._ws_server = ws_serve(_ws_handler, self.host, self._separate_ws_port)
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
        with self._ws_clients_lock:
            raw_socks = list(self._raw_ws_clients.keys())
            self._raw_ws_clients.clear()
        for sock in raw_socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
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
