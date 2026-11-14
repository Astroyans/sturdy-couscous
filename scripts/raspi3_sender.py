from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import zlib

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None

PACKET_MAGIC = b"R3D1"
_HEADER_STRUCT = struct.Struct("!4sI")
_LEN_STRUCT = struct.Struct("!I")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emisor ligero RealSense D415 desde Raspberry Pi 3 hacia Laptop Rover (TCP binario)"
    )
    parser.add_argument(
        "--laptop-ip",
        required=True,
        help="IP LAN de la Laptop Rover ejecutando --source raspi3.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Puerto TCP de entrada en la Laptop Rover (por defecto 8766).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Ancho del stream en D415 (480 recomendado para USB 2.0 en Raspi 3).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=270,
        help="Alto del stream en D415 (270 recomendado para USB 2.0 en Raspi 3).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="FPS objetivo de captura en la Raspi 3.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=72,
        help="Calidad JPEG del canal RGB (30-95).",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Forzar envio solo de profundidad (maxima estabilidad en USB 2.0).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=-90.0,
        help="Angulo nominal de camara en el dron (-90.0 = cenital).",
    )
    return parser.parse_args(argv)


def start_d415(width: int, height: int, fps: int, depth_only: bool):
    if rs is None:
        raise RuntimeError("pyrealsense2 no esta instalado en la Raspberry Pi 3.")

    profiles = (
        [((width, height, min(fps, 15)), None), ((480, 270, 6), None)]
        if depth_only
        else [
            ((width, height, fps), (width, height, fps)),
            ((480, 270, 6), (480, 270, 6)),
            ((480, 270, 6), None),
        ]
    )

    last_err = None
    for d_spec, c_spec in profiles:
        pipe = None
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, d_spec[0], d_spec[1], rs.format.z16, d_spec[2])
            if c_spec is not None:
                cfg.enable_stream(rs.stream.color, c_spec[0], c_spec[1], rs.format.bgr8, c_spec[2])
            profile = pipe.start(cfg)
            sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(sensor.get_depth_scale())
            has_color = c_spec is not None
            align = rs.align(rs.stream.color) if has_color else None
            stream_t = rs.stream.color if has_color else rs.stream.depth
            intr = profile.get_stream(stream_t).as_video_stream_profile().get_intrinsics()
            return pipe, align, depth_scale, intr, has_color
        except Exception as exc:
            last_err = exc
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass
    raise RuntimeError(f"No se pudo iniciar RealSense D415 en Raspi 3: {last_err}")


def pack_frame(
    depth_u16: np.ndarray,
    intr,
    depth_scale: float,
    color_bgr: np.ndarray | None,
    pitch_deg: float,
    jpeg_quality: int,
) -> bytes:
    h, w = depth_u16.shape[:2]
    depth_bytes = zlib.compress(depth_u16.tobytes(), level=1)
    color_bytes = b""
    if color_bgr is not None:
        if (color_bgr.shape[1], color_bgr.shape[0]) != (w, h):
            color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(
            ".jpg",
            color_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(max(30, min(95, jpeg_quality)))],
        )
        if ok:
            color_bytes = buf.tobytes()

    header = {
        "ts": time.time(),
        "w": int(w),
        "h": int(h),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "depth_scale": float(depth_scale),
        "imu_yaw": 0.0,
        "imu_pitch": float(pitch_deg),
        "imu_roll": 0.0,
        "depth_only": color_bgr is None or len(color_bytes) == 0,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return (
        _HEADER_STRUCT.pack(PACKET_MAGIC, len(header_bytes))
        + header_bytes
        + _LEN_STRUCT.pack(len(color_bytes))
        + color_bytes
        + _LEN_STRUCT.pack(len(depth_bytes))
        + depth_bytes
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipe, align, depth_scale, intr, has_color = start_d415(
        args.width, args.height, args.fps, args.depth_only
    )
    print(
        f"[RPI3] D415 iniciada ({intr.width}x{intr.height}, color={has_color}) -> "
        f"enviando a {args.laptop_ip}:{args.port}"
    )

    try:
        while True:
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(4.0)
                sock.connect((args.laptop_ip, args.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[RPI3] Conectado a Laptop Rover {args.laptop_ip}:{args.port}")

                sent = 0
                t0 = time.perf_counter()
                while True:
                    frames = pipe.wait_for_frames(timeout_ms=3000)
                    active = align.process(frames) if align is not None else frames
                    d_frame = active.get_depth_frame()
                    if not d_frame:
                        continue
                    depth_u16 = np.asanyarray(d_frame.get_data()).copy()
                    c_frame = active.get_color_frame() if has_color else None
                    color_bgr = np.asanyarray(c_frame.get_data()).copy() if c_frame else None

                    pkt = pack_frame(
                        depth_u16=depth_u16,
                        intr=intr,
                        depth_scale=depth_scale,
                        color_bgr=color_bgr,
                        pitch_deg=args.pitch,
                        jpeg_quality=args.jpeg_quality,
                    )
                    sock.sendall(pkt)
                    sent += 1
                    if sent % 30 == 0:
                        dt = time.perf_counter() - t0
                        print(f"[RPI3] TX fps={30.0 / max(dt, 1e-3):.1f} pkt_kb={len(pkt) / 1024.0:.1f}")
                        t0 = time.perf_counter()
            except OSError as exc:
                print(f"[RPI3] Reconectando ({exc})...")
                time.sleep(1.0)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
    except KeyboardInterrupt:
        print("\n[RPI3] Emisor detenido.")
    finally:
        pipe.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
