from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from src.camera.realsense_d415 import RealSenseD415Camera
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.pipeline import TerrainNavigationPipeline
from src.transport.client import TelemetrySender
from src.transport.media import build_expo_payload, build_map_state_payload
from src.transport.server import ExpoLANServer, get_local_lan_ip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rover D415 Real Headless Telemetry & Media Streamer for Expo Go (iPad A16) & Web Dashboard"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "realsense", "synthetic"],
        default="auto",
        help="Fuente de camara: 'realsense' (D415), 'synthetic' (simulador 3D) o 'auto'.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "ws", "dry-run"],
        default=os.environ.get("DRONE_CV_TRANSPORT", "http"),
        help="Protocolo de envio al backend externo: 'http' (POST JSON), 'ws' (WebSocket) o 'dry-run'.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("DRONE_CV_BACKEND_URL", ""),
        help="URL del backend externo (ej. http://192.168.1.50:3000/api/telemetry o ws://192.168.1.50:3000/ws).",
    )
    parser.add_argument(
        "--goal-url",
        default=os.environ.get("DRONE_CV_GOAL_URL", ""),
        help="URL HTTP GET opcional para recibir destino tocado por el astronauta en Expo Go.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("DRONE_CV_AUTH_TOKEN", ""),
        help="Token Bearer opcional para autenticacion con el backend.",
    )
    parser.add_argument(
        "--planner",
        choices=["astar", "dijkstra"],
        default="astar",
        help="Algoritmo primario de ruta minima (A* y Dijkstra se calculan ambos automaticamente).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=-90.0,
        help="Angulo nominal de inclinacion de montaje de la camara en grados (-90.0 = vista aerea cenital).",
    )
    parser.add_argument(
        "--port",
        "--http-port",
        dest="http_port",
        type=int,
        default=8765,
        help="Puerto unificado LAN (WebSocket + HTTP en el mismo puerto, por defecto 8765 para full-stack).",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="Puerto WebSocket separado opcional (por defecto None = comparte --port 8765 con HTTP).",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Desactivar el servidor LAN integrado hacia el iPad / Dashboard.",
    )
    parser.add_argument(
        "--boulder-min-height",
        type=float,
        default=0.08,
        help="Altura minima en metros sobre el suelo para clasificar BOULDER.",
    )
    parser.add_argument(
        "--crevice-min-depth",
        type=float,
        default=0.08,
        help="Profundidad minima en metros bajo el suelo para clasificar CREVICE.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Ancho de las 3 imagenes JPEG enviadas al iPad (RGB, profundidad y mapa de rutas).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Alto de las 3 imagenes JPEG enviadas al iPad.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="Calidad JPEG (30-95) para las 3 imagenes enviadas.",
    )
    parser.add_argument(
        "--raw-rgb",
        action="store_true",
        help="Enviar imagen RGB limpia sin anotaciones de bounding boxes.",
    )
    parser.add_argument(
        "--return-base",
        action="store_true",
        help="Iniciar directamente en modo RETURN_TO_BASE (retorno EVA).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Numero de frames a procesar (0 = bucle continuo automatizado).",
    )
    parser.add_argument(
        "--no-local-cache",
        action="store_true",
        help="Desactivar respaldo local continuo en captures/latest_*.json/.npz.",
    )
    return parser.parse_args(argv)


def create_and_start_camera(source: str, config: PipelineConfig):
    if source == "synthetic":
        cam = SyntheticTerrainCamera(config, terrain_incline_deg=6.0, animate=True)
        cam.start()
        print("[INFO] Iniciando camara sintetica 3D (modo simulacion)...")
        return cam, "synthetic"

    if source in ("auto", "realsense"):
        try:
            cam = RealSenseD415Camera(config)
            cam.start()
            print(f"[INFO] Conectado a camara fisica: {cam.device_name} (USB {cam.usb_type})")
            return cam, "realsense"
        except Exception as exc:
            if source == "realsense":
                raise RuntimeError(
                    f"No se pudo iniciar RealSense D415: {exc}"
                ) from exc
            print(f"[WARN] RealSense D415 no disponible ({exc}). Usando camara sintetica 3D...")

    cam = SyntheticTerrainCamera(config, terrain_incline_deg=6.0, animate=True)
    cam.start()
    return cam, "synthetic"


def _atomic_write_json(path: str, obj: dict, indent: int | None = None) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if indent is None:
            json.dump(obj, f, separators=(",", ":"))
        else:
            json.dump(obj, f, indent=indent)
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass


def export_local_cache(
    payload: dict,
    output,
    map_state: dict | None = None,
    captures_dir: str = "captures",
) -> None:
    os.makedirs(captures_dir, exist_ok=True)
    _atomic_write_json(os.path.join(captures_dir, "latest_expo_payload.json"), payload)

    light_payload = {k: v for k, v in payload.items() if k != "images"}
    _atomic_write_json(
        os.path.join(captures_dir, "latest_drone_cv.json"),
        light_payload,
        indent=2,
    )

    if map_state is not None:
        _atomic_write_json(os.path.join(captures_dir, "real_map_state.json"), map_state)

    tmp_npz = os.path.join(captures_dir, "latest_map_state.tmp.npz")
    final_npz = os.path.join(captures_dir, "latest_map_state.npz")
    np.savez_compressed(
        tmp_npz,
        occupancy_grid=output.navigation.occupancy_grid,
        cost_grid=output.navigation.cost_grid,
        height_map_m=output.terrain.height_map_m,
        slope_map_deg=output.terrain.slope_map_deg,
    )
    try:
        os.replace(tmp_npz, final_npz)
    except PermissionError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        nominal_camera_pitch_deg=args.pitch,
        boulder_min_height_m=args.boulder_min_height,
        crevice_min_depth_m=args.crevice_min_depth,
        default_planner=args.planner,
    )

    try:
        camera, active_source = create_and_start_camera(args.source, config)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    pipeline = TerrainNavigationPipeline(config=config)
    if args.return_base:
        pipeline.set_navigation_mode("RETURN_TO_BASE")
    sender = TelemetrySender(
        url=args.url,
        transport=args.transport,
        auth_token=args.token or None,
        goal_url=args.goal_url,
    )
    lan_server = (
        None
        if args.no_server
        else ExpoLANServer(http_port=args.http_port, ws_port=args.ws_port)
    )

    effective_transport = (
        args.transport if args.url and args.transport != "dry-run" else "dry-run"
    )
    print(
        f"[INFO] Nodo real drone_cv iniciado | Fuente: {active_source} | "
        f"Pitch: {args.pitch:.0f}deg | Transporte externo: {effective_transport}"
    )

    sender.start()
    if lan_server is not None:
        lan_server.start()
        lan_ip = get_local_lan_ip()
        print(
            f"[LAN->full-stack] Conectarse desde Dashboard Web / Expo Go (iPad A16):\n"
            f"  -> WebSocket Stream : ws://{lan_ip}:{lan_server.ws_port}\n"
            f"  -> HTTP Map State   : http://{lan_ip}:{lan_server.http_port}/map_state.json\n"
            f"  -> HTTP Latest JSON : http://{lan_ip}:{lan_server.http_port}/latest_expo_payload.json\n"
            f"  -> HTTP Click Goal  : POST http://{lan_ip}:{lan_server.http_port}/api/goal"
        )
    frame_count = 0

    try:
        while True:
            goal_cmd = sender.pop_goal_command()
            if goal_cmd is None and lan_server is not None:
                goal_cmd = lan_server.pop_goal_command()
            if goal_cmd is not None:
                if goal_cmd.get("clear"):
                    pipeline.clear_goal()
                elif "norm_x" in goal_cmd and "norm_y" in goal_cmd:
                    pipeline.set_goal_normalized(goal_cmd["norm_x"], goal_cmd["norm_y"])
                elif "x_m" in goal_cmd and "y_m" in goal_cmd:
                    pipeline.set_goal_world(goal_cmd["x_m"], goal_cmd["y_m"])

            frame = camera.read_frame()
            output, b_mask, c_mask = pipeline.process_frame(frame)
            payload = build_expo_payload(
                output=output,
                config=config,
                boulder_mask=b_mask,
                crevice_mask=c_mask,
                image_width=args.image_width,
                image_height=args.image_height,
                jpeg_quality=args.jpeg_quality,
                annotate_rgb=not args.raw_rgb,
            )
            map_state = build_map_state_payload(output=output, config=config)

            sender.enqueue(payload)
            if lan_server is not None:
                lan_server.publish(payload, map_state=map_state)
            if not args.no_local_cache:
                export_local_cache(payload, output, map_state=map_state)

            frame_count += 1
            if frame_count == 1 or frame_count % 10 == 0 or (
                args.frames > 0 and frame_count >= args.frames
            ):
                err_note = f" | err={sender.last_error}" if sender.last_error else ""
                live_obs = len(output.frame_obstacles) if output.frame_obstacles is not None else len(output.obstacles)
                print(
                    f"[EXPO] frame={frame_count} fps={output.edge_fps:.1f} "
                    f"obs_live={live_obs} obs_map={len(output.obstacles)} "
                    f"sent={sender.sent_count} fail={sender.failed_count}{err_note}"
                )

            if args.frames > 0 and frame_count >= args.frames:
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[INFO] Detenido por usuario.")
    finally:
        if lan_server is not None:
            lan_server.stop()
        sender.stop()
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
