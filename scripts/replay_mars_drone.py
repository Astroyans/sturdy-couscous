from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.record_udp import iter_recorded_frames
from src.camera.raspi3_receiver import encode_raspi3_packet, encode_udp_chunks
from src.models import CameraIntrinsics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulador de stream en vivo del dron sobre Marte a partir de grabacion R3D1."
    )
    parser.add_argument(
        "--recording",
        default="captures/udp_rec_20261115_042022",
        help="Carpeta con frames_r3d1.bin grabados del dron.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP destino donde escucha main_expo.py --source raspi3 (por defecto 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Puerto UDP destino (por defecto 8766).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frecuencia de envio de frames simulados (por defecto 12.0 FPS).",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.35,
        help="Proporcion minima de pixeles validos para seleccionar frames de la grabacion.",
    )
    parser.add_argument(
        "--target-altitude",
        type=float,
        default=1.65,
        help="Altitud nominal simulada sobre la superficie marciana en metros.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Numero maximo de frames a enviar (0 = bucle infinito para presentacion).",
    )
    return parser.parse_args(argv)


def _mars_synthesize_frame(
    raw_depth_m: np.ndarray,
    target_altitude_m: float,
    phase_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = raw_depth_m.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) - 0.5
    ny = (yy / max(h - 1, 1)) - 0.5

    # Relieve base de dunas marcianas y pendiente suave
    phase = phase_idx * 0.08
    ripples = 0.014 * np.sin(nx * 26.0 + ny * 8.0 + phase) + 0.025 * np.sin(nx * 5.0 - ny * 4.0)
    base_plane = float(target_altitude_m) + 0.04 * nx - 0.03 * ny + ripples

    valid = (raw_depth_m > 0.18) & (raw_depth_m < 4.5)
    if float(np.mean(valid)) >= 0.15:
        med = float(np.median(raw_depth_m[valid]))
        shifted = raw_depth_m + (target_altitude_m - med)
        shifted = np.clip(shifted, 0.35, target_altitude_m + 1.2)
        filled = np.where(valid, shifted, base_plane).astype(np.float32)
        depth_m = cv2.GaussianBlur(filled, (5, 5), 0.9)
    else:
        depth_m = base_plane.astype(np.float32)

    # Anadir una roca o crater marciano sutil si el frame es muy plano
    relief = base_plane - depth_m
    if float(np.std(relief)) < 0.02:
        cx = 0.22 * np.sin(phase * 0.5)
        cy = 0.12 * np.cos(phase * 0.4)
        dist_r = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        boulder = 0.22 * np.exp(-((dist_r / 0.09) ** 2))
        crater_r = np.sqrt((nx + 0.24) ** 2 + (ny + 0.10) ** 2)
        crater = -0.18 * np.exp(-((crater_r / 0.11) ** 2))
        depth_m = np.clip(depth_m - boulder - crater, 0.4, 3.5).astype(np.float32)
        relief = base_plane - depth_m

    # Textura RGB marciana alineada al relieve (BGR para OpenCV)
    gy, gx = np.gradient(depth_m)
    shade = np.clip(1.0 - (gx * 6.5 + gy * 4.5), 0.42, 1.45)

    grain = rng.normal(0.0, 4.5, size=(h, w)).astype(np.float32)
    ripple_tint = 10.0 * np.sin(nx * 32.0 + ny * 10.0 + phase)

    # Paleta oxido de hierro / basalto marciano
    r_ch = np.clip((178.0 + ripple_tint + relief * 110.0) * shade + grain, 28, 245)
    g_ch = np.clip((92.0 + ripple_tint * 0.5 + relief * 60.0) * shade + grain * 0.7, 18, 195)
    b_ch = np.clip((58.0 + ripple_tint * 0.2 + relief * 35.0) * shade + grain * 0.5, 14, 155)

    # Rocas oscuras basalticas en elevaciones altas y sombra en grietas
    rock_mask = relief > 0.06
    r_ch = np.where(rock_mask, r_ch * 0.72, r_ch)
    g_ch = np.where(rock_mask, g_ch * 0.70, g_ch)
    b_ch = np.where(rock_mask, b_ch * 0.74, b_ch)

    crevice_mask = relief < -0.06
    r_ch = np.where(crevice_mask, r_ch * 0.58, r_ch)
    g_ch = np.where(crevice_mask, g_ch * 0.52, g_ch)
    b_ch = np.where(crevice_mask, b_ch * 0.50, b_ch)

    color_bgr = np.stack([b_ch, g_ch, r_ch], axis=-1).astype(np.uint8)
    return color_bgr, depth_m


def build_mars_replay_frames(
    recording_dir: str | Path,
    min_valid_ratio: float = 0.35,
    target_altitude_m: float = 1.65,
    max_selected: int = 180,
) -> list[tuple[np.ndarray, np.ndarray, CameraIntrinsics]]:
    rec_path = Path(recording_dir)
    selected: list[tuple[np.ndarray, CameraIntrinsics]] = []
    fallback: list[tuple[np.ndarray, CameraIntrinsics]] = []

    if (rec_path / "frames_r3d1.bin").exists():
        for bundle in iter_recorded_frames(rec_path):
            d = bundle.depth_m
            ratio = float(np.mean((d > 0.18) & (d < 4.5)))
            if ratio >= min_valid_ratio:
                selected.append((d.copy(), bundle.intrinsics))
                if len(selected) >= max_selected:
                    break
            elif len(fallback) < 60 and ratio >= 0.05:
                fallback.append((d.copy(), bundle.intrinsics))

    source = selected if selected else fallback
    if not source:
        intr = CameraIntrinsics(width=480, height=270, fx=384.0, fy=384.0, ppx=240.0, ppy=135.0)
        source = [(np.full((270, 480), target_altitude_m, dtype=np.float32), intr) for _ in range(30)]

    rng = np.random.default_rng(2026)
    frames: list[tuple[np.ndarray, np.ndarray, CameraIntrinsics]] = []
    for idx, (raw_d, intr) in enumerate(source):
        color_bgr, mars_depth = _mars_synthesize_frame(
            raw_depth_m=raw_d,
            target_altitude_m=target_altitude_m,
            phase_idx=idx,
            rng=rng,
        )
        frames.append((color_bgr, mars_depth, intr))
    return frames


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[MARS-SIM] Cargando grabacion desde {args.recording}...")
    frames = build_mars_replay_frames(
        recording_dir=args.recording,
        min_valid_ratio=args.min_valid_ratio,
        target_altitude_m=args.target_altitude,
    )
    print(
        f"[MARS-SIM] Listos {len(frames)} frames marcianos | "
        f"Destino UDP {args.host}:{args.port} a {args.fps:.1f} FPS"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period_s = 1.0 / max(1.0, float(args.fps))
    sent = 0
    frame_id = 1
    n = len(frames)

    try:
        while args.max_frames <= 0 or sent < args.max_frames:
            idx = sent % n
            color_bgr, depth_m, intr = frames[idx]
            yaw = float(4.0 * np.sin(sent * 0.05))
            pitch = float(-90.0 + 1.5 * np.cos(sent * 0.04))
            roll = float(1.2 * np.sin(sent * 0.03))

            payload = encode_raspi3_packet(
                depth_m=depth_m,
                intrinsics=intr,
                color_bgr=color_bgr,
                timestamp=time.time(),
                imu_yaw=yaw,
                imu_pitch=pitch,
                imu_roll=roll,
                jpeg_quality=72,
                zlib_level=1,
            )
            chunks = encode_udp_chunks(payload, frame_id=frame_id, max_chunk_bytes=1200)
            for chunk in chunks:
                sock.sendto(chunk, (args.host, args.port))

            sent += 1
            frame_id += 1
            if sent == 1 or sent % 30 == 0:
                print(
                    f"[MARS-SIM] frame={sent} idx={idx}/{n} "
                    f"chunks={len(chunks)} payload_kb={len(payload)/1024:.1f}"
                )
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[MARS-SIM] Simulacion detenida por usuario.")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
