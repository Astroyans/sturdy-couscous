from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Optional


class CentroMandoLink:
    def __init__(
        self,
        centro_mando_ip: str = "",
        centro_mando_port: int = 8767,
        astronaut_udp_ip: str = "",
        astronaut_udp_port: int = 8769,
        local_listen_port: int = 0,
    ) -> None:
        self.centro_mando_ip = centro_mando_ip.strip()
        self.centro_mando_port = int(centro_mando_port)
        self.astronaut_udp_ip = astronaut_udp_ip.strip()
        self.astronaut_udp_port = int(astronaut_udp_port)
        self.local_listen_port = int(local_listen_port)

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._incoming_quadrants: list[dict[str, Any]] = []
        self.active_scan_request: Optional[dict[str, Any]] = None
        self.link_mode: str = "LOCAL_ONLY"
        self.last_rx_time: float = 0.0
        self.tx_count: int = 0
        self.rx_count: int = 0

    def start(self) -> None:
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        except OSError:
            pass
        sock.bind(("0.0.0.0", self.local_listen_port))
        sock.settimeout(0.25)
        self.local_listen_port = int(sock.getsockname()[1])
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _rx_loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                raw, _addr = self._sock.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            msg_type = msg.get("type", "")
            self.rx_count += 1
            self.last_rx_time = time.time()

            with self._lock:
                if "link_mode" in msg:
                    self.link_mode = str(msg["link_mode"])
                if msg_type == "QUADRANT_DATA" and "quadrant" in msg:
                    self._incoming_quadrants.append(msg["quadrant"])
                    self.active_scan_request = None
                elif msg_type == "SCAN_REQUEST":
                    self.active_scan_request = {
                        "id": msg.get("id"),
                        "bounds": msg.get("bounds"),
                        "target_xy_m": msg.get("target_xy_m"),
                        "reason": msg.get("reason", "CACHE_REDUNDANT_SCAN_DRONE"),
                        "timestamp": msg.get("timestamp", time.time()),
                    }
                elif msg_type == "CLEAR_SCAN_REQUEST":
                    self.active_scan_request = None

    def pop_incoming_quadrants(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._incoming_quadrants)
            self._incoming_quadrants.clear()
            return items

    def get_active_scan_request(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self.active_scan_request is None:
                return None
            if (time.time() - float(self.active_scan_request.get("timestamp", 0.0))) > 15.0:
                self.active_scan_request = None
                return None
            return dict(self.active_scan_request)

    def send_to_centro_mando(
        self,
        prediction_dict: dict[str, Any],
        ruta_txt: str,
        scanned_patch: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.centro_mando_ip or self._sock is None:
            return False
        packet: dict[str, Any] = {
            "type": "PREDICTION_AND_MAP",
            "timestamp_ms": prediction_dict.get("timestamp_ms", int(time.time() * 1000)),
            "prediction": prediction_dict,
            "ruta_txt": ruta_txt.strip(),
        }
        if scanned_patch is not None:
            packet["scanned_patch"] = scanned_patch

        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        if len(raw) > 60000 and "scanned_patch" in packet:
            del packet["scanned_patch"]
            raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")

        try:
            self._sock.sendto(raw, (self.centro_mando_ip, self.centro_mando_port))
            self.tx_count += 1
            return True
        except OSError:
            return False

    def send_astronaut_udp(
        self,
        mode: str,
        planner: str,
        safe_corridor: bool,
        altitude_m: float,
        heading_deg: float,
        pose_xy: tuple[float, float],
        prediction_dict: dict[str, Any],
        waypoints_xy: list[tuple[float, float]],
        scan_request: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.astronaut_udp_ip or self._sock is None:
            return False

        step = max(1, len(waypoints_xy) // 6) if len(waypoints_xy) > 6 else 1
        sampled_wp = [
            [round(float(wx), 2), round(float(wy), 2)]
            for wx, wy in waypoints_xy[::step][:6]
        ]
        msg: dict[str, Any] = {
            "node": "drone_cv",
            "ts": int(time.time()),
            "mode": mode,
            "plan": planner,
            "safe": bool(safe_corridor),
            "alt": round(float(altitude_m), 2),
            "hdg": round(float(heading_deg), 1),
            "pose": [round(float(pose_xy[0]), 2), round(float(pose_xy[1]), 2)],
            "cur_q": prediction_dict.get("current_quadrant"),
            "next_q": prediction_dict.get("next_quadrant"),
            "wp": sampled_wp,
            "link": self.link_mode,
            "scan": scan_request.get("target_xy_m") if scan_request else None,
        }
        raw = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(raw) >= 1200:
            msg["wp"] = sampled_wp[:3]
            raw = json.dumps(msg, separators=(",", ":")).encode("utf-8")

        try:
            self._sock.sendto(raw, (self.astronaut_udp_ip, self.astronaut_udp_port))
            return True
        except OSError:
            return False
