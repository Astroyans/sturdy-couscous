from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from typing import Any, Optional

# Configuracion de red del Centro de Mando (editar aqui o pasar por CLI)
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8767
STATION_IP = ""
STATION_UDP_PORT = 8770
STATION_TIMEOUT_S = 0.8
SATELLITE_IP = ""
SATELLITE_TCP_PORT = 8888
CACHE_DIR = "captures/centro_mando_cache"


def _bounds_key(bounds: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        round(float(bounds["x_min"]), 2),
        round(float(bounds["x_max"]), 2),
        round(float(bounds["y_min"]), 2),
        round(float(bounds["y_max"]), 2),
    )


def _bounds_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (
        float(a["x_max"]) <= float(b["x_min"]) + 0.05
        or float(a["x_min"]) >= float(b["x_max"]) - 0.05
        or float(a["y_max"]) <= float(b["y_min"]) + 0.05
        or float(a["y_min"]) >= float(b["y_max"]) - 0.05
    )


class CentroMandoVehicleCache:
    def __init__(self, cache_dir: str = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self.pasado: Optional[dict[str, Any]] = None
        self.actual: Optional[dict[str, Any]] = None
        self.proximo: Optional[dict[str, Any]] = None
        self.dataset: dict[int, dict[str, Any]] = {}
        self._load_disk()

    def _load_disk(self) -> None:
        if not self.cache_dir or not os.path.isdir(self.cache_dir):
            return
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.cache_dir, fname), "r", encoding="utf-8") as f:
                    rec = json.load(f)
                qid = int(rec["id"])
                self.dataset[qid] = rec
            except Exception:
                continue

    def _save_record(self, rec: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            qid = int(rec["id"])
            path = os.path.join(self.cache_dir, f"q_{qid}.json")
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, separators=(",", ":"))
            os.replace(tmp, path)
        except OSError:
            pass

    def store_quadrant(self, rec: dict[str, Any]) -> None:
        qid = int(rec.get("id", int(time.time() * 1000)))
        rec["id"] = qid
        with self._lock:
            self.dataset[qid] = rec
        self._save_record(rec)

    def find_by_bounds(self, target_bounds: dict[str, Any]) -> Optional[dict[str, Any]]:
        with self._lock:
            candidates = [
                rec
                for rec in self.dataset.values()
                if "bounds" in rec and _bounds_overlap(rec["bounds"], target_bounds)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: int(r["id"]), reverse=True)
        return candidates[0]

    def rotate_on_advance(self, current_bounds: dict[str, Any]) -> bool:
        with self._lock:
            cur_key = _bounds_key(current_bounds)
            if self.actual is not None and _bounds_key(self.actual["bounds"]) == cur_key:
                return False
            if self.actual is not None:
                self.pasado = self.actual
            if self.proximo is not None and _bounds_key(self.proximo["bounds"]) == cur_key:
                self.actual = self.proximo
                self.proximo = None
            else:
                found = None
                for rec in sorted(self.dataset.values(), key=lambda r: int(r["id"]), reverse=True):
                    if "bounds" in rec and _bounds_overlap(rec["bounds"], current_bounds):
                        found = rec
                        break
                self.actual = found or {
                    "id": int(time.time() * 1000),
                    "bounds": current_bounds,
                }
            return True

    def is_redundant_with_actual(self, candidate: Optional[dict[str, Any]]) -> bool:
        if candidate is None:
            return True
        with self._lock:
            if self.actual is None:
                return False
            if int(candidate.get("id", -1)) == int(self.actual.get("id", -2)):
                return True
            return _bounds_key(candidate["bounds"]) == _bounds_key(self.actual["bounds"])


class CentroMandoServer:
    def __init__(
        self,
        listen_host: str = LISTEN_HOST,
        listen_port: int = LISTEN_PORT,
        station_ip: str = STATION_IP,
        station_udp_port: int = STATION_UDP_PORT,
        station_timeout_s: float = STATION_TIMEOUT_S,
        satellite_ip: str = SATELLITE_IP,
        satellite_tcp_port: int = SATELLITE_TCP_PORT,
        cache_dir: str = CACHE_DIR,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.station_ip = station_ip.strip()
        self.station_udp_port = int(station_udp_port)
        self.station_timeout_s = float(station_timeout_s)
        self.satellite_ip = satellite_ip.strip()
        self.satellite_tcp_port = int(satellite_tcp_port)

        self.cache = CentroMandoVehicleCache(cache_dir=cache_dir)
        self.link_mode = "STATION_DIRECT" if self.station_ip else "SATELLITE_MESH"

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _query_station_udp(self, next_bounds: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.station_ip:
            raise OSError("Estacion no configurada")
        req = json.dumps(
            {
                "cmd": "GET_QUADRANT",
                "bounds": next_bounds,
                "timestamp_ms": int(time.time() * 1000),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.station_timeout_s)
        try:
            sock.sendto(req, (self.station_ip, self.station_udp_port))
            raw, _ = sock.recvfrom(65507)
            resp = json.loads(raw.decode("utf-8"))
            self.link_mode = "STATION_DIRECT"
            if resp.get("found") and "quadrant" in resp:
                return resp["quadrant"]
            return None
        finally:
            sock.close()

    def _forward_to_satellite_tcp(self, ruta_line: str) -> Optional[dict[str, Any]]:
        if not self.satellite_ip:
            return None
        clean_line = (ruta_line.strip() + "\n").encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.station_timeout_s)
        try:
            sock.connect((self.satellite_ip, self.satellite_tcp_port))
            sock.sendall(clean_line)
            try:
                resp = sock.recv(4096).decode("utf-8", errors="ignore").strip()
                if resp.startswith("{"):
                    parsed = json.loads(resp)
                    if "quadrant" in parsed:
                        return parsed["quadrant"]
            except (socket.timeout, ValueError):
                pass
        except OSError:
            pass
        finally:
            sock.close()
        return None

    def handle_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        scanned = packet.get("scanned_patch")
        if isinstance(scanned, dict) and "occupancy_subgrid" in scanned:
            self.cache.store_quadrant(scanned)

        ruta_txt = str(packet.get("ruta_txt", "")).strip()
        pred = packet.get("prediction", {})
        cur_bounds = pred.get("current_quadrant")
        next_bounds = pred.get("next_quadrant")

        if isinstance(cur_bounds, dict):
            self.cache.rotate_on_advance(cur_bounds)

        candidate: Optional[dict[str, Any]] = None
        if isinstance(next_bounds, dict):
            if self.station_ip:
                try:
                    candidate = self._query_station_udp(next_bounds)
                    self.link_mode = "STATION_DIRECT"
                except (socket.timeout, OSError, ValueError):
                    self.link_mode = "SATELLITE_MESH"
                    if ruta_txt:
                        candidate = self._forward_to_satellite_tcp(ruta_txt)
            else:
                self.link_mode = "SATELLITE_MESH"
                if ruta_txt and self.satellite_ip:
                    candidate = self._forward_to_satellite_tcp(ruta_txt)

            if candidate is None:
                candidate = self.cache.find_by_bounds(next_bounds)

            if candidate is not None:
                self.cache.store_quadrant(candidate)
                self.cache.proximo = candidate

        if self.cache.is_redundant_with_actual(candidate):
            target_b = next_bounds or cur_bounds or {
                "x_min": -0.75,
                "x_max": 0.75,
                "y_min": 0.0,
                "y_max": 1.0,
            }
            cx = round(0.5 * (float(target_b["x_min"]) + float(target_b["x_max"])), 2)
            cy = round(0.5 * (float(target_b["y_min"]) + float(target_b["y_max"])), 2)
            return {
                "type": "SCAN_REQUEST",
                "id": int(time.time() * 1000),
                "bounds": target_b,
                "target_xy_m": [cx, cy],
                "reason": "CACHE_REDUNDANT_OR_MISSING_NEXT_QUADRANT",
                "link_mode": self.link_mode,
                "timestamp": time.time(),
            }

        if ruta_txt and self.satellite_ip and self.link_mode == "STATION_DIRECT":
            self._forward_to_satellite_tcp(ruta_txt)

        return {
            "type": "QUADRANT_DATA",
            "id": int(candidate["id"]),
            "quadrant": candidate,
            "link_mode": self.link_mode,
            "timestamp": time.time(),
        }

    def start(self) -> None:
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.listen_host, self.listen_port))
        sock.settimeout(0.25)
        self.listen_port = int(sock.getsockname()[1])
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                raw, addr = self._sock.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                pkt = json.loads(raw.decode("utf-8"))
                resp = self.handle_packet(pkt)
                out = json.dumps(resp, separators=(",", ":")).encode("utf-8")
                if len(out) <= 65000:
                    self._sock.sendto(out, addr)
            except Exception:
                continue

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Servidor RPi Centro de Mando: cache pasado/actual/proximo, Estacion y Satelite ESP32"
    )
    parser.add_argument("--listen-host", default=LISTEN_HOST)
    parser.add_argument("--listen-port", type=int, default=LISTEN_PORT)
    parser.add_argument("--station-ip", default=STATION_IP)
    parser.add_argument("--station-port", type=int, default=STATION_UDP_PORT)
    parser.add_argument("--satellite-ip", default=SATELLITE_IP)
    parser.add_argument("--satellite-port", type=int, default=SATELLITE_TCP_PORT)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    srv = CentroMandoServer(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        station_ip=args.station_ip,
        station_udp_port=args.station_port,
        satellite_ip=args.satellite_ip,
        satellite_tcp_port=args.satellite_port,
        cache_dir=args.cache_dir,
    )
    srv.start()
    print(
        f"[CENTRO-MANDO] Escuchando UDP en {args.listen_host}:{srv.listen_port} | "
        f"Estacion={args.station_ip or 'offline'} | "
        f"SateliteESP32={args.satellite_ip or 'offline'}:{args.satellite_port}"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[CENTRO-MANDO] Detenido.")
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
