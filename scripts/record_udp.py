from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.camera.raspi3_receiver import (
    _UDP_ACK_STRUCT,
    UDP_ACK_MAGIC,
    UdpFrameReassembler,
    decode_raspi3_packet,
    unpack_r3d1_payload,
)
from src.models import FrameBundle

_REC_LEN_STRUCT = struct.Struct("!I")


def iter_recorded_frames(recording_dir: str | Path) -> Iterator[FrameBundle]:
    """Lee un archivo frames_r3d1.bin grabado y produce FrameBundle por cada frame."""
    bin_path = Path(recording_dir) / "frames_r3d1.bin"
    with bin_path.open("rb") as f:
        while True:
            len_buf = f.read(_REC_LEN_STRUCT.size)
            if not len_buf or len(len_buf) < _REC_LEN_STRUCT.size:
                break
            (payload_len,) = _REC_LEN_STRUCT.unpack(len_buf)
            payload = f.read(payload_len)
            if len(payload) < payload_len:
                break
            hb, cb, db = unpack_r3d1_payload(payload)
            bundle, _ = decode_raspi3_packet(hb, cb, db)
            yield bundle


def record_udp_stream(
    host: str,
    port: int,
    out_dir: Path,
    duration_s: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "frames_r3d1.bin"
    jsonl_path = out_dir / "frames.jsonl"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    except OSError:
        pass
    sock.bind((host, port))
    sock.settimeout(0.5)

    reassembler = UdpFrameReassembler(max_inflight=4, frame_timeout_s=0.25)
    frames_saved = 0
    bytes_rx = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    last_frames = 0
    last_bytes = 0

    print(f"[REC] Escuchando UDP R3U1 en {host}:{port}")
    print(f"[REC] Guardando en: {out_dir}")
    print("[REC] Presiona Ctrl+C para detener y cerrar los archivos.")

    try:
        with bin_path.open("wb") as f_bin, jsonl_path.open("w", encoding="utf-8") as f_meta:
            while True:
                if duration_s > 0 and (time.perf_counter() - t_start) >= duration_s:
                    break
                try:
                    datagram, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                bytes_rx += len(datagram)
                completed = reassembler.feed_datagram(datagram)
                if completed is None:
                    continue

                frame_id, payload = completed
                try:
                    hb, cb, db = unpack_r3d1_payload(payload)
                    bundle, _ = decode_raspi3_packet(hb, cb, db)
                    meta = json.loads(hb.decode("utf-8"))
                except (ValueError, zlib.error, RuntimeError, json.JSONDecodeError):
                    continue

                f_bin.write(_REC_LEN_STRUCT.pack(len(payload)))
                f_bin.write(payload)

                valid = bundle.depth_m > 0.05
                d_min = float(bundle.depth_m[valid].min()) if np.any(valid) else 0.0
                d_max = float(bundle.depth_m[valid].max()) if np.any(valid) else 0.0

                record = {
                    "idx": frames_saved,
                    "frame_id": int(frame_id),
                    "rx_ts": time.time(),
                    "src_ip": addr[0],
                    "src_port": int(addr[1]),
                    "ts": meta.get("ts"),
                    "w": meta.get("w"),
                    "h": meta.get("h"),
                    "fx": meta.get("fx"),
                    "fy": meta.get("fy"),
                    "ppx": meta.get("ppx"),
                    "ppy": meta.get("ppy"),
                    "depth_scale": meta.get("depth_scale"),
                    "imu_yaw": meta.get("imu_yaw", 0.0),
                    "imu_pitch": meta.get("imu_pitch", -90.0),
                    "imu_roll": meta.get("imu_roll", 0.0),
                    "depth_only": bool(meta.get("depth_only", False)),
                    "color_len": len(cb),
                    "depth_len": len(db),
                    "depth_min_m": round(d_min, 4),
                    "depth_max_m": round(d_max, 4),
                }
                f_meta.write(json.dumps(record, separators=(",", ":")) + "\n")

                frames_saved += 1
                if frames_saved % 10 == 0:
                    f_bin.flush()
                    f_meta.flush()

                try:
                    loss_x100 = int(min(65535, round(reassembler.loss_pct * 100.0)))
                    ack = _UDP_ACK_STRUCT.pack(UDP_ACK_MAGIC, frame_id & 0xFFFFFFFF, loss_x100)
                    sock.sendto(ack, addr)
                except OSError:
                    pass

                now = time.perf_counter()
                dt = now - t_last_log
                if dt >= 1.0:
                    fps = (frames_saved - last_frames) / dt
                    kbps = ((bytes_rx - last_bytes) * 8.0 / 1000.0) / dt
                    print(
                        f"[REC] frames={frames_saved} | fps={fps:.1f} | "
                        f"loss={reassembler.loss_pct:.1f}% | depth={d_min:.2f}-{d_max:.2f}m | {kbps:.0f} kbps",
                        flush=True,
                    )
                    t_last_log = now
                    last_frames = frames_saved
                    last_bytes = bytes_rx
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"[REC] Grabacion finalizada: {frames_saved} frames guardados en {out_dir}")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Graba paquetes UDP R3U1/R3D1 en disco para uso posterior.")
    parser.add_argument("--host", default="0.0.0.0", help="IP local de escucha (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8768, help="Puerto UDP de escucha (default 8768)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duracion maxima en segundos (0 = ilimitado)")
    parser.add_argument("--out", default="", help="Carpeta de salida (default captures/udp_rec_<timestamp>)")
    parser.add_argument("--replay", default="", help="Ruta a una carpeta grabada para verificar/reproducir frames")
    args = parser.parse_args()

    if args.replay:
        count = 0
        for b in iter_recorded_frames(args.replay):
            count += 1
            if count <= 3 or count % 50 == 0:
                print(f"[REPLAY] frame={count} shape={b.depth_m.shape} pitch={b.imu_pitch}")
        print(f"[REPLAY] Total frames leidos: {count}")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT_DIR / "captures" / f"udp_rec_{stamp}"
    record_udp_stream(args.host, args.port, out_dir, duration_s=args.duration)


if __name__ == "__main__":
    main()
