from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

from src.camera.raspi3_receiver import Raspi3ReceiverCamera
from src.camera.realsense_d415 import RealSenseD415Camera
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.navigation.trajectory_predictor import (
    extract_quadrant_patch,
    format_esp32_ruta_txt,
    predict_trajectory,
)
from src.pipeline import TerrainNavigationPipeline
from src.transport.centro_mando_link import CentroMandoLink
from src.transport.client import TelemetrySender
from src.transport.media import build_expo_payload, build_map_state_payload
from src.transport.server import ExpoLANServer, get_local_lan_ip

# Configuracion de red: entrada Raspi 3 / Puente (8766) y salida Dashboard (8765 / 8766)
RASPI_LISTEN_HOST = "0.0.0.0"
RASPI_PORT = 8766
RASPI_REMOTE_IP = ""
EXPO_PORT = 8765

# IP y puerto de ingesta del Dashboard remoto (full-stack INBOUND_PORT)
DASHBOARD_IP = "172.20.10.7"
DASHBOARD_INGEST_PORT = 8766
DASHBOARD_TRANSPORT = "http"
DASHBOARD_URL = f"http://{DASHBOARD_IP}:{DASHBOARD_INGEST_PORT}/api/ingest"

# IP y puertos UDP para RPi Centro de Mando (8767) y Dispositivo Astronauta (8769)
CENTRO_MANDO_IP = ""
CENTRO_MANDO_PORT = 8767
ASTRONAUT_UDP_IP = ""
ASTRONAUT_UDP_PORT = 8769


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rover D415 Real Headless Telemetry & Media Streamer for Expo Go (iPad A16) & Web Dashboard"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "realsense", "raspi3", "synthetic"],
        default="auto",
        help="Fuente de camara: 'realsense' (D415 USB), 'raspi3' (LAN UDP/TCP puerto 8766), 'synthetic' o 'auto'.",
    )
    parser.add_argument(
        "--raspi-proto",
        choices=["udp", "tcp"],
        default="udp",
        help="Protocolo de recepcion desde Raspberry Pi 3: 'udp' (R3U1 baja latencia) o 'tcp'.",
    )
    parser.add_argument(
        "--raspi-port",
        type=int,
        default=RASPI_PORT,
        help="Puerto de entrada para recibir RGB-D desde la Raspberry Pi 3 (por defecto 8766).",
    )
    parser.add_argument(
        "--raspi-host",
        default=RASPI_REMOTE_IP,
        help="IP opcional de la Raspberry Pi 3 si la laptop se conecta como cliente TCP en lugar de escuchar.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "ws", "dry-run"],
        default=os.environ.get("DRONE_CV_TRANSPORT", DASHBOARD_TRANSPORT),
        help="Protocolo de envio al backend externo: 'http' (POST JSON), 'ws' (WebSocket) o 'dry-run'.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("DRONE_CV_BACKEND_URL", DASHBOARD_URL),
        help="URL del backend externo (ej. http://172.18.0.165:8766/api/ingest o ws://172.18.0.165:8766).",
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
        default=EXPO_PORT,
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
    parser.add_argument(
        "--centro-mando-ip",
        default=CENTRO_MANDO_IP,
        help="IP de la RPi Centro de mando (UDP 8767) para prediccion y cache.",
    )
    parser.add_argument(
        "--centro-mando-port",
        type=int,
        default=CENTRO_MANDO_PORT,
        help="Puerto UDP de la RPi Centro de mando (por defecto 8767).",
    )
    parser.add_argument(
        "--astronaut-udp-ip",
        default=ASTRONAUT_UDP_IP,
        help="IP del dispositivo del astronauta para paquetes UDP ligeros (<1200 B).",
    )
    parser.add_argument(
        "--astronaut-udp-port",
        type=int,
        default=ASTRONAUT_UDP_PORT,
        help="Puerto UDP del dispositivo del astronauta (por defecto 8769).",
    )
    return parser.parse_args(argv)


def create_and_start_camera(
    source: str,
    config: PipelineConfig,
    raspi_port: int = 8766,
    raspi_host: str = "",
    raspi_proto: str = "udp",
):
    if source == "raspi3":
        cam = Raspi3ReceiverCamera(
            config=config,
            listen_host=RASPI_LISTEN_HOST,
            port=raspi_port,
            remote_host=raspi_host,
            protocol=raspi_proto,
        )
        mode_desc = (
            f"cliente TCP -> {raspi_host}:{raspi_port}"
            if (raspi_proto == "tcp" and raspi_host)
            else f"escuchando {raspi_proto.upper()} en 0.0.0.0:{raspi_port}"
        )
        print(f"[INFO] Esperando stream de Raspberry Pi 3 ({mode_desc})...")
        cam.start(wait_for_first_frame=True)
        print(f"[INFO] Enlace Raspi 3 activo: {cam.device_name}")
        return cam, "raspi3"

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
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            if indent is None:
                json.dump(obj, f, separators=(",", ":"))
            else:
                json.dump(obj, f, indent=indent)
        os.replace(tmp, path)
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def export_local_cache(
    payload: dict,
    output,
    map_state: dict | None = None,
    captures_dir: str = "captures",
) -> None:
    try:
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

        tmp_npz = os.path.join(captures_dir, f"latest_map_state.{os.getpid()}.tmp.npz")
        final_npz = os.path.join(captures_dir, "latest_map_state.npz")
        np.savez_compressed(
            tmp_npz,
            occupancy_grid=output.navigation.occupancy_grid,
            cost_grid=output.navigation.cost_grid,
            height_map_m=output.terrain.height_map_m,
            slope_map_deg=output.terrain.slope_map_deg,
        )
        os.replace(tmp_npz, final_npz)
    except OSError:
        pass


class _AsyncCacheWriter:
    def __init__(self, captures_dir: str = "captures") -> None:
        self.captures_dir = captures_dir
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending: tuple | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, payload: dict, output, map_state: dict | None = None) -> None:
        snapshot = (
            payload,
            output.navigation.occupancy_grid.copy(),
            output.navigation.cost_grid.copy(),
            output.terrain.height_map_m.copy(),
            output.terrain.slope_map_deg.copy(),
            map_state,
        )
        with self._lock:
            self._pending = snapshot
        self._event.set()

    def _run(self) -> None:
        while self._running:
            if not self._event.wait(timeout=0.5):
                continue
            self._event.clear()
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue
            payload, occ, cost, height_m, slope_deg, map_state = item
            try:
                os.makedirs(self.captures_dir, exist_ok=True)
                _atomic_write_json(
                    os.path.join(self.captures_dir, "latest_expo_payload.json"), payload
                )
                light_payload = {k: v for k, v in payload.items() if k != "images"}
                _atomic_write_json(
                    os.path.join(self.captures_dir, "latest_drone_cv.json"),
                    light_payload,
                    indent=2,
                )
                if map_state is not None:
                    _atomic_write_json(
                        os.path.join(self.captures_dir, "real_map_state.json"), map_state
                    )
                tmp_npz = os.path.join(self.captures_dir, f"latest_map_state.{os.getpid()}.tmp.npz")
                final_npz = os.path.join(self.captures_dir, "latest_map_state.npz")
                np.savez_compressed(
                    tmp_npz,
                    occupancy_grid=occ,
                    cost_grid=cost,
                    height_map_m=height_m,
                    slope_map_deg=slope_deg,
                )
                os.replace(tmp_npz, final_npz)
            except OSError:
                pass

    def stop(self) -> None:
        self._running = False
        self._event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        nominal_camera_pitch_deg=args.pitch,
        boulder_min_height_m=args.boulder_min_height,
        crevice_min_depth_m=args.crevice_min_depth,
        default_planner=args.planner,
    )

    try:
        camera, active_source = create_and_start_camera(
            args.source,
            config,
            raspi_port=args.raspi_port,
            raspi_host=args.raspi_host,
            raspi_proto=args.raspi_proto,
        )
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
    centro_link = CentroMandoLink(
        centro_mando_ip=args.centro_mando_ip,
        centro_mando_port=args.centro_mando_port,
        astronaut_udp_ip=args.astronaut_udp_ip,
        astronaut_udp_port=args.astronaut_udp_port,
    )
    centro_link.start()
    cache_writer = None if args.no_local_cache else _AsyncCacheWriter()
    if cache_writer is not None:
        cache_writer.start()
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
    last_cm_tx_ts = 0.0

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

            for ext_q in centro_link.pop_incoming_quadrants():
                if isinstance(ext_q, dict) and "bounds" in ext_q and "occupancy_subgrid" in ext_q:
                    pipeline.inject_external_quadrant(ext_q["bounds"], ext_q["occupancy_subgrid"])

            frame = camera.read_frame()
            output, b_mask, c_mask = pipeline.process_frame(frame)
            prediction = predict_trajectory(output)
            scan_req = centro_link.get_active_scan_request()
            ruta_txt = format_esp32_ruta_txt(
                mode=output.navigation.navigation_mode,
                planner=output.navigation.planner_used,
                safe_corridor=output.navigation.safe_corridor,
                altitude_m=output.terrain.altitude_m,
                heading_deg=output.navigation.recommended_heading_deg,
                timestamp_ms=prediction.timestamp_ms,
                waypoints_xy=output.navigation.path_world_xy,
                scan_alert_id=scan_req.get("id") if scan_req else None,
            )

            now_cm = time.perf_counter()
            if (now_cm - last_cm_tx_ts) >= 0.5:
                last_cm_tx_ts = now_cm
                scanned_patch = extract_quadrant_patch(
                    occupancy_grid=output.navigation.occupancy_grid,
                    cost_grid=output.navigation.cost_grid,
                    bounds=prediction.current_bounds,
                    resolution_m=config.grid_resolution_m,
                    y_min_m=pipeline.grid_builder.y_min_m,
                    grid_width_m=config.grid_width_m,
                )
                centro_link.send_to_centro_mando(
                    prediction_dict=prediction.to_dict(),
                    ruta_txt=ruta_txt,
                    scanned_patch=scanned_patch,
                )

            centro_link.send_astronaut_udp(
                mode=output.navigation.navigation_mode,
                planner=output.navigation.planner_used,
                safe_corridor=output.navigation.safe_corridor,
                altitude_m=output.terrain.altitude_m,
                heading_deg=output.navigation.recommended_heading_deg,
                pose_xy=(output.rover_pose_xy_yaw[0], output.rover_pose_xy_yaw[1]),
                prediction_dict=prediction.to_dict(),
                waypoints_xy=output.navigation.path_world_xy,
                scan_request=scan_req,
            )

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
            payload["data"]["trajectory_prediction"] = prediction.to_dict()
            payload["data"]["centro_mando_link"] = centro_link.link_mode
            if scan_req is not None:
                payload["data"]["scan_request"] = scan_req
            map_state = build_map_state_payload(output=output, config=config)

            sender.enqueue(payload)
            if lan_server is not None:
                lan_server.publish(payload, map_state=map_state)
            if cache_writer is not None:
                cache_writer.enqueue(payload, output, map_state=map_state)

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
        if cache_writer is not None:
            cache_writer.stop()
        centro_link.stop()
        sender.stop()
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
