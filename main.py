from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from src.camera.raspi3_receiver import Raspi3ReceiverCamera
from src.camera.realsense_d415 import RealSenseD415Camera
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.pipeline import TerrainNavigationPipeline
from src.visualization.hud import HUDRenderer

# Configuracion de red para recepcion desde Raspi 3 / Laptop Puente
RASPI_LISTEN_HOST = "0.0.0.0"
RASPI_PORT = 8766
RASPI_REMOTE_IP = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rover D415 Automated Terrain Analysis & Persistent Global Path Planning"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "realsense", "raspi3", "synthetic"],
        default="auto",
        help="Fuente de camara: 'realsense' (D415 USB-C), 'raspi3' (LAN UDP/TCP), 'synthetic' (simulador 3D) o 'auto'.",
    )
    parser.add_argument(
        "--raspi-proto",
        choices=["udp", "tcp"],
        default="udp",
        help="Protocolo de recepcion desde Raspberry Pi 3: 'udp' (R3U1) o 'tcp'.",
    )
    parser.add_argument(
        "--raspi-port",
        type=int,
        default=RASPI_PORT,
        help="Puerto de entrada para recibir datos desde la Raspberry Pi 3.",
    )
    parser.add_argument(
        "--raspi-host",
        default=RASPI_REMOTE_IP,
        help="IP opcional de la Raspberry Pi 3 / Puente si se conecta por TCP cliente.",
    )
    parser.add_argument(
        "--planner",
        choices=["astar", "dijkstra"],
        default="astar",
        help="Algoritmo primario de ruta minima (ambos A* y Dijkstra se calculan automaticamente).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=-90.0,
        help="Angulo nominal de inclinacion de montaje de la camara en grados (ej. -90.0 vista aerea cenital).",
    )
    parser.add_argument(
        "--boulder-min-height",
        type=float,
        default=0.08,
        help="Altura minima en metros sobre el plano del suelo para clasificar BOULDER.",
    )
    parser.add_argument(
        "--crevice-min-depth",
        type=float,
        default=0.08,
        help="Profundidad minima en metros bajo el plano del suelo para clasificar CREVICE.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Numero de frames a procesar (0 = bucle continuo automatizado).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecutar sin abrir ventana grafica OpenCV.",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Guardar captura del HUD (.png) y payload JSON (.json) en --save-dir.",
    )
    parser.add_argument(
        "--save-dir",
        default="captures",
        help="Directorio donde se exporta continuamente latest_drone_cv.json y capturas.",
    )
    return parser.parse_args(argv)


def save_capture(hud_img, payload: dict, save_dir: str) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    img_path = os.path.join(save_dir, f"hud_{ts_ms}.png")
    json_path = os.path.join(save_dir, f"drone_cv_{ts_ms}.json")
    cv2.imwrite(img_path, hud_img)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return img_path, json_path


def export_live_telemetry(output, payload: dict, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, "latest_drone_cv.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    npz_path = os.path.join(save_dir, "latest_map_state.npz")
    np.savez_compressed(
        npz_path,
        occupancy_grid=output.navigation.occupancy_grid,
        cost_grid=output.navigation.cost_grid,
        height_map_m=output.terrain.height_map_m,
        slope_map_deg=output.terrain.slope_map_deg,
    )
    return latest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = PipelineConfig(
        nominal_camera_pitch_deg=args.pitch,
        boulder_min_height_m=args.boulder_min_height,
        crevice_min_depth_m=args.crevice_min_depth,
        default_planner=args.planner,
    )

    camera = None
    source_label = ""

    if args.source == "raspi3":
        try:
            r3_cam = Raspi3ReceiverCamera(
                config=config,
                listen_host=RASPI_LISTEN_HOST,
                port=args.raspi_port,
                remote_host=args.raspi_host,
                protocol=args.raspi_proto,
            )
            print(
                f"[INFO] Esperando stream de Raspberry Pi 3 "
                f"({args.raspi_proto.upper()} puerto {args.raspi_port})..."
            )
            r3_cam.start(wait_for_first_frame=True)
            camera = r3_cam
            source_label = r3_cam.device_name
            print(f"[INFO] Enlace Raspi 3 activo: {source_label}")
        except Exception as exc:
            print(f"[ERROR] No se pudo recibir desde Raspi 3: {exc}", file=sys.stderr)
            return 1

    if camera is None and args.source in ("auto", "realsense"):
        try:
            rs_cam = RealSenseD415Camera(config)
            rs_cam.start()
            camera = rs_cam
            source_label = f"{rs_cam.device_name} (USB {rs_cam.usb_type})"
            print(f"[INFO] Conectado a camara fisica: {source_label}")
        except Exception as exc:
            if args.source == "realsense":
                print(f"[ERROR] No se pudo iniciar RealSense D415: {exc}", file=sys.stderr)
                return 1
            print(f"[WARN] RealSense D415 no disponible ({exc}). Usando simulador 3D.")

    if camera is None:
        syn_cam = SyntheticTerrainCamera(config, terrain_incline_deg=6.0, animate=True)
        syn_cam.start()
        camera = syn_cam
        source_label = syn_cam.device_name
        print(f"[INFO] Fuente activa: {source_label}")

    pipeline = TerrainNavigationPipeline(config)
    hud_renderer = HUDRenderer(config)

    window_name = "Rover D415 Automated Persistent Map & Path Planning (NASA Space Apps 2026)"
    if not args.headless:
        def _on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and 490 <= x <= 910 and 40 <= y <= 660:
                nx = (x - 490) / 420.0
                ny = (y - 40) / 620.0
                gx, gy = pipeline.set_goal_normalized(nx, ny)
                print(f"[INFO] Destino fijado en mapa: ({gx:+.2f}m, {gy:+.2f}m)")
            elif event == cv2.EVENT_RBUTTONDOWN:
                pipeline.clear_goal()
                print("[INFO] Destino limpiado.")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _on_mouse)

    frame_count = 0
    last_hud = None
    last_payload = None
    last_output = None
    last_export_ts = 0.0

    try:
        while True:
            frame_bundle = camera.read_frame()
            output, boulder_mask, crevice_mask = pipeline.process_frame(frame_bundle)
            last_output = output
            last_payload = output.to_drone_cv_payload()
            last_hud = hud_renderer.render(
                output,
                boulder_mask,
                crevice_mask,
                plane_locked=pipeline.ground_analyzer.is_plane_locked,
                source_name=source_label,
            )

            now = time.time()
            if now - last_export_ts >= 0.5:
                export_live_telemetry(output, last_payload, args.save_dir)
                last_export_ts = now

            frame_count += 1

            if not args.headless:
                cv2.imshow(window_name, last_hud)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("p"):
                    new_p = pipeline.toggle_planner()
                    print(f"[INFO] Algoritmo primario cambiado a: {new_p.upper()}")
                elif key == ord("m"):
                    mode = pipeline.toggle_navigation_mode()
                    print(f"[INFO] Modo de mision cambiado a: {mode}")
                elif key == ord("r"):
                    pipeline.reset_global_map()
                    print("[INFO] Mapa global persistente reiniciado.")
                elif key == ord("w"):
                    p = pipeline.advance_pose_manually(dy_local_m=0.25)
                    print(f"[INFO] Avance Dron/Rover -> Pose Global: ({p.x_m:+.2f}m, {p.y_m:+.2f}m)")
                elif key == ord("s"):
                    img_p, json_p = save_capture(last_hud, last_payload, args.save_dir)
                    print(f"[INFO] Captura guardada en: {img_p} y {json_p}")
                elif key == ord("a"):
                    pipeline.advance_pose_manually(dx_local_m=-0.15)
                elif key == ord("d"):
                    pipeline.advance_pose_manually(dx_local_m=0.15)
                elif key == ord("c"):
                    locked = pipeline.toggle_plane_lock()
                    state_str = "FIJADO (Manual)" if locked else "DINAMICO (RANSAC)"
                    print(f"[INFO] Plano de referencia del suelo: {state_str}")
                elif key == ord("x"):
                    pipeline.clear_goal()
                    print("[INFO] Destino limpiado.")

            if args.frames > 0 and frame_count >= args.frames:
                break

        if last_payload is not None and last_output is not None:
            export_live_telemetry(last_output, last_payload, args.save_dir)

        if args.save_snapshot and last_hud is not None and last_payload is not None:
            img_p, json_p = save_capture(last_hud, last_payload, args.save_dir)
            print(f"[INFO] Snapshot guardado en: {img_p} y {json_p}")

        if last_payload is not None:
            print("\n=== Ultimo Payload Generado (Contrato JSON `drone_cv`) ===")
            print(json.dumps(last_payload, indent=2, ensure_ascii=False))

    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
