from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

import requests

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover
    ws_connect = None


class TelemetrySender:
    def __init__(
        self,
        url: str = "",
        transport: str = "http",
        timeout_s: float = 3.0,
        auth_token: Optional[str] = None,
        goal_url: str = "",
    ) -> None:
        self.url = url.strip()
        self.goal_url = goal_url.strip()
        self.transport = transport.lower().strip()
        self.timeout_s = timeout_s
        self.auth_token = auth_token

        self._cond = threading.Condition()
        self._pending_payload: Optional[dict[str, Any]] = None
        self._goal_lock = threading.Lock()
        self._pending_goal_cmd: Optional[dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.sent_count = 0
        self.failed_count = 0
        self.last_error: Optional[str] = None

    def pop_goal_command(self) -> Optional[dict[str, Any]]:
        with self._goal_lock:
            cmd = self._pending_goal_cmd
            self._pending_goal_cmd = None
            return cmd

    def _push_goal_command(self, raw_msg: Any) -> None:
        try:
            data = json.loads(raw_msg) if isinstance(raw_msg, str) else raw_msg
            if not isinstance(data, dict):
                return
            if any(k in data for k in ("x_m", "y_m", "norm_x", "norm_y", "clear")):
                with self._goal_lock:
                    self._pending_goal_cmd = data
        except Exception:
            return

    def start(self) -> None:
        if self.transport == "dry-run" or not self.url:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def enqueue(self, payload: dict[str, Any]) -> None:
        if self.transport == "dry-run" or not self.url:
            return
        with self._cond:
            self._pending_payload = payload
            self._cond.notify()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _worker_loop(self) -> None:
        if self.transport == "ws":
            self._run_ws_loop()
        else:
            self._run_http_loop()

    def _take_next_payload(self) -> Optional[dict[str, Any]]:
        with self._cond:
            while self._running and self._pending_payload is None:
                self._cond.wait(timeout=0.5)
            if not self._running:
                return None
            payload = self._pending_payload
            self._pending_payload = None
            return payload

    def _run_http_loop(self) -> None:
        session = requests.Session()
        headers = self._headers()
        last_goal_poll = 0.0
        try:
            while self._running:
                if self.goal_url and (time.perf_counter() - last_goal_poll) >= 0.5:
                    last_goal_poll = time.perf_counter()
                    try:
                        g_resp = session.get(
                            self.goal_url, headers=headers, timeout=self.timeout_s
                        )
                        if 200 <= g_resp.status_code < 300 and g_resp.text.strip():
                            self._push_goal_command(g_resp.json())
                    except Exception:
                        pass

                payload = self._take_next_payload()
                if payload is None:
                    continue
                try:
                    resp = session.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_s,
                    )
                    if 200 <= resp.status_code < 300:
                        self.sent_count += 1
                        self.last_error = None
                    else:
                        self.failed_count += 1
                        self.last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
                        time.sleep(0.5)
                except Exception as exc:
                    self.failed_count += 1
                    self.last_error = str(exc)
                    time.sleep(1.0)
        finally:
            session.close()

    def _run_ws_loop(self) -> None:
        if ws_connect is None:
            self.last_error = "Paquete 'websockets' no disponible"
            return

        extra_headers = {}
        if self.auth_token:
            extra_headers["Authorization"] = f"Bearer {self.auth_token}"

        while self._running:
            try:
                with ws_connect(
                    self.url,
                    additional_headers=extra_headers or None,
                    open_timeout=self.timeout_s,
                    close_timeout=1.0,
                ) as ws:
                    self.last_error = None
                    while self._running:
                        try:
                            incoming = ws.recv(timeout=0.01)
                            if incoming:
                                self._push_goal_command(incoming)
                        except TimeoutError:
                            pass
                        payload = self._take_next_payload()
                        if payload is None:
                            continue
                        ws.send(json.dumps(payload, separators=(",", ":")))
                        self.sent_count += 1
            except Exception as exc:
                self.failed_count += 1
                self.last_error = str(exc)
                time.sleep(1.0)
