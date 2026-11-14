from __future__ import annotations

import json
import socket
import struct
import threading
import time
import zlib
from typing import Any, Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import CameraIntrinsics, FrameBundle

PACKET_MAGIC = b"R3D1"
_HEADER_STRUCT = struct.Struct("!4sI")
_LEN_STRUCT = struct.Struct("!I")
_MAX_PACKET_BYTES = 16 * 1024 * 1024


def encode_raspi3_packet(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_bgr: Optional[np.ndarray] = None,
    timestamp: Optional[float] = None,
    imu_yaw: float = 0.0,
    imu_pitch: float = -90.0,
    imu_roll: float = 0.0,
    jpeg_quality: int = 75,
    zlib_level: int = 1,
) -> bytes:
    h, w = depth_m.shape[:2]
    depth_scale = float(intrinsics.depth_scale) if intrinsics.depth_scale > 0 else 0.001
    depth_units = np.clip(np.round(depth_m / depth_scale), 0, 65535).astype(np.uint16)
    depth_compressed = zlib.compress(depth_units.tobytes(), level=int(np.clip(zlib_level, 0, 9)))

    color_bytes = b""
    depth_only = color_bgr is None
    if color_bgr is not None:
        ch, cw = color_bgr.shape[:2]
        img = (
            cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            if (cw, ch) != (w, h)
            else color_bgr
        )
        ok, buf = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 30, 95))],
        )
        if not ok:
            raise RuntimeError("No se pudo codificar color JPEG para paquete Raspi3.")
        color_bytes = buf.tobytes()

    header = {
        "ts": float(timestamp if timestamp is not None else time.time()),
        "w": int(w),
        "h": int(h),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "depth_scale": depth_scale,
        "imu_yaw": float(imu_yaw),
        "imu_pitch": float(imu_pitch),
        "imu_roll": float(imu_roll),
        "depth_only": bool(depth_only),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    return (
        _HEADER_STRUCT.pack(PACKET_MAGIC, len(header_bytes))
        + header_bytes
        + _LEN_STRUCT.pack(len(color_bytes))
        + color_bytes
        + _LEN_STRUCT.pack(len(depth_compressed))
        + depth_compressed
    )


def decode_raspi3_packet(
    header_bytes: bytes,
    color_bytes: bytes,
    depth_bytes: bytes,
    default_pitch_deg: float = -90.0,
) -> tuple[FrameBundle, float]:
    meta: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
    w = int(meta["w"])
    h = int(meta["h"])
    depth_scale = float(meta.get("depth_scale", 0.001))

    raw_depth = zlib.decompress(depth_bytes)
    depth_u16 = np.frombuffer(raw_depth, dtype=np.uint16).reshape((h, w))
    depth_m = depth_u16.astype(np.float32) * depth_scale

    if color_bytes and not meta.get("depth_only", False):
        arr = np.frombuffer(color_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Frame JPEG corrupto recibido desde Raspi3.")
        if (decoded.shape[1], decoded.shape[0]) != (w, h):
            color_bgr = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            color_bgr = decoded
    else:
        norm = np.clip((depth_m - 0.25) / 3.0, 0.0, 1.0)
        gray = ((1.0 - norm) * 235.0).astype(np.uint8)
        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        color_bgr[depth_m <= 0.05] = (20, 20, 25)

    intrinsics = CameraIntrinsics(
        width=w,
        height=h,
        fx=float(meta.get("fx", w * 0.92)),
        fy=float(meta.get("fy", h * 1.22)),
        ppx=float(meta.get("ppx", w / 2.0)),
        ppy=float(meta.get("ppy", h / 2.0)),
        depth_scale=depth_scale,
    )

    sender_ts = float(meta.get("ts", time.time()))
    now = time.time()
    net_latency_ms = max(0.0, (now - sender_ts) * 1000.0)

    bundle = FrameBundle(
        timestamp=now,
        color_bgr=color_bgr,
        depth_m=depth_m,
        intrinsics=intrinsics,
        imu_yaw=float(meta.get("imu_yaw", 0.0)),
        imu_pitch=float(meta.get("imu_pitch", default_pitch_deg)),
        imu_roll=float(meta.get("imu_roll", 0.0)),
    )
    return bundle, net_latency_ms


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _configure_tcp_socket(sock: socket.socket, timeout_s: float = 5.0) -> None:
    sock.settimeout(timeout_s)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
    except OSError:
        pass


class Raspi3ReceiverCamera:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        listen_host: str = "0.0.0.0",
        port: int = 8766,
        remote_host: str = "",
        wait_timeout_s: float = 15.0,
    ) -> None:
        self.config = config or PipelineConfig()
        self.listen_host = listen_host
        self.port = int(port)
        self.remote_host = remote_host.strip()
        self.wait_timeout_s = float(wait_timeout_s)

        self.device_name = "Raspberry Pi 3 -> RealSense D415 (LAN)"
        self.usb_type = "LAN-TCP"
        self.connected = False
        self.net_latency_ms = 0.0
        self.rx_fps = 0.0
        self.rx_kbps = 0.0
        self.frames_received = 0
        self.last_rx_time = 0.0

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._latest_bundle: Optional[FrameBundle] = None
        self._server_sock: Optional[socket.socket] = None
        self._active_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _consume_connection(self, conn: socket.socket) -> None:
        _configure_tcp_socket(conn, timeout_s=5.0)
        self._active_sock = conn
        self.connected = True
        fps_t0 = time.perf_counter()
        fps_frames = 0
        fps_bytes = 0

        try:
            while self._running:
                head = _recv_exact(conn, _HEADER_STRUCT.size)
                if head is None:
                    break
                magic, header_len = _HEADER_STRUCT.unpack(head)
                if magic != PACKET_MAGIC or header_len <= 0 or header_len > 65536:
                    break

                header_bytes = _recv_exact(conn, header_len)
                if header_bytes is None:
                    break

                c_len_raw = _recv_exact(conn, _LEN_STRUCT.size)
                if c_len_raw is None:
                    break
                (color_len,) = _LEN_STRUCT.unpack(c_len_raw)
                if color_len > _MAX_PACKET_BYTES:
                    break
                color_bytes = _recv_exact(conn, color_len)
                if color_bytes is None:
                    break

                d_len_raw = _recv_exact(conn, _LEN_STRUCT.size)
                if d_len_raw is None:
                    break
                (depth_len,) = _LEN_STRUCT.unpack(d_len_raw)
                if depth_len <= 0 or depth_len > _MAX_PACKET_BYTES:
                    break
                depth_bytes = _recv_exact(conn, depth_len)
                if depth_bytes is None:
                    break

                bundle, latency_ms = decode_raspi3_packet(
                    header_bytes=header_bytes,
                    color_bytes=color_bytes,
                    depth_bytes=depth_bytes,
                    default_pitch_deg=self.config.nominal_camera_pitch_deg,
                )

                with self._lock:
                    self._latest_bundle = bundle
                self._frame_ready.set()

                self.frames_received += 1
                self.last_rx_time = time.time()
                self.net_latency_ms = latency_ms

                fps_frames += 1
                fps_bytes += _HEADER_STRUCT.size + header_len + 8 + color_len + depth_len
                dt = time.perf_counter() - fps_t0
                if dt >= 1.0:
                    self.rx_fps = fps_frames / dt
                    self.rx_kbps = (fps_bytes * 8.0 / 1000.0) / dt
                    fps_t0 = time.perf_counter()
                    fps_frames = 0
                    fps_bytes = 0
        except (OSError, ValueError, zlib.error, RuntimeError):
            pass
        finally:
            self.connected = False
            self._active_sock = None
            try:
                conn.close()
            except OSError:
                pass

    def _rx_loop(self) -> None:
        while self._running:
            if self.remote_host:
                sock: Optional[socket.socket] = None
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3.0)
                    sock.connect((self.remote_host, self.port))
                    self._consume_connection(sock)
                except OSError:
                    if sock is not None:
                        try:
                            sock.close()
                        except OSError:
                            pass
                    time.sleep(0.5)
            else:
                if self._server_sock is None:
                    break
                try:
                    conn, _addr = self._server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        time.sleep(0.1)
                    continue
                self._consume_connection(conn)

    def start(self, wait_for_first_frame: bool = True) -> None:
        if self._running:
            return
        if not self.remote_host:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.listen_host, self.port))
            srv.listen(2)
            srv.settimeout(0.5)
            self.port = int(srv.getsockname()[1])
            self._server_sock = srv

        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

        if wait_for_first_frame:
            if not self._frame_ready.wait(timeout=self.wait_timeout_s):
                self.stop()
                target = (
                    f"{self.remote_host}:{self.port}"
                    if self.remote_host
                    else f"0.0.0.0:{self.port}"
                )
                raise RuntimeError(
                    f"Timeout ({self.wait_timeout_s:.1f}s) esperando stream de la Raspi 3 en {target}."
                )

    def read_frame(self, timeout_ms: int = 2000) -> FrameBundle:
        if not self._frame_ready.wait(timeout=max(timeout_ms / 1000.0, 0.1)):
            raise RuntimeError("No han llegado frames desde la Raspberry Pi 3.")
        with self._lock:
            bundle = self._latest_bundle
        if bundle is None:
            raise RuntimeError("Buffer de frames de Raspi 3 vacío.")
        return FrameBundle(
            timestamp=time.time(),
            color_bgr=bundle.color_bgr.copy(),
            depth_m=bundle.depth_m.copy(),
            intrinsics=bundle.intrinsics,
            imu_yaw=bundle.imu_yaw,
            imu_pitch=bundle.imu_pitch,
            imu_roll=bundle.imu_roll,
        )

    def stop(self) -> None:
        self._running = False
        for sock_attr in ("_active_sock", "_server_sock"):
            sock = getattr(self, sock_attr)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                setattr(self, sock_attr, None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def __enter__(self) -> Raspi3ReceiverCamera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
