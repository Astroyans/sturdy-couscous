import json
import socket
import threading
import time

import numpy as np

from scripts.centro_mando_server import CentroMandoServer
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.models import GRID_BOULDER, GRID_FREE, GRID_UNKNOWN
from src.navigation.trajectory_predictor import (
    extract_quadrant_patch,
    format_esp32_ruta_txt,
    predict_trajectory,
)
from src.pipeline import TerrainNavigationPipeline
from src.transport.centro_mando_link import CentroMandoLink


def test_trajectory_prediction_and_esp32_ruta_txt_format():
    cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
    cam = SyntheticTerrainCamera(cfg)
    cam.start()
    frame = cam.read_frame()
    cam.stop()

    pipeline = TerrainNavigationPipeline(cfg)
    pipeline.set_goal_normalized(0.5, 0.15)
    output, _, _ = pipeline.process_frame(frame)

    pred = predict_trajectory(output)
    d = pred.to_dict()
    assert "current_quadrant" in d
    assert "next_quadrant" in d
    assert "predicted_pose_3s" in d
    assert len(d["remaining_quadrants"]) >= 1

    line = format_esp32_ruta_txt(
        mode=output.navigation.navigation_mode,
        planner=output.navigation.planner_used,
        safe_corridor=output.navigation.safe_corridor,
        altitude_m=output.terrain.altitude_m,
        heading_deg=output.navigation.recommended_heading_deg,
        timestamp_ms=pred.timestamp_ms,
        waypoints_xy=output.navigation.path_world_xy,
    )
    assert line.endswith("\n")
    assert len(line.encode("utf-8")) <= 500
    assert line.startswith("RUTA_V1|")


def test_inject_external_quadrant_fills_unknown_cells_without_overwriting_live():
    cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
    pipeline = TerrainNavigationPipeline(cfg)
    rows, cols = pipeline.grid_builder.grid_shape

    sub_occ = np.full((20, 30), GRID_FREE, dtype=np.uint8)
    sub_occ[8:12, 12:16] = GRID_BOULDER
    bounds = {"x_min": -0.75, "x_max": 0.75, "y_min": 1.0, "y_max": 2.0}

    filled = pipeline.inject_external_quadrant(bounds, sub_occ)
    assert filled == 20 * 30

    cam = SyntheticTerrainCamera(cfg)
    cam.start()
    frame = cam.read_frame()
    cam.stop()
    output, _, _ = pipeline.process_frame(frame)
    grid = output.navigation.occupancy_grid
    assert grid.shape == (rows, cols)
    assert np.count_nonzero(grid != GRID_UNKNOWN) >= filled


def test_centro_mando_station_fallback_to_satellite_and_scan_request(tmp_path):
    station_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    station_sock.bind(("127.0.0.1", 0))
    station_port = station_sock.getsockname()[1]
    station_sock.settimeout(0.2)

    sat_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sat_srv.bind(("127.0.0.1", 0))
    sat_port = sat_srv.getsockname()[1]
    sat_srv.listen(2)
    sat_srv.settimeout(0.2)

    astro_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    astro_sock.bind(("127.0.0.1", 0))
    astro_port = astro_sock.getsockname()[1]
    astro_sock.settimeout(1.5)

    stop_ev = threading.Event()
    station_enabled = [True]
    sat_received_lines: list[str] = []

    def _station_worker():
        while not stop_ev.is_set():
            try:
                raw, addr = station_sock.recvfrom(4096)
            except socket.timeout:
                continue
            if not station_enabled[0]:
                continue
            req = json.loads(raw.decode("utf-8"))
            b = req["bounds"]
            resp = {
                "found": True,
                "quadrant": {
                    "id": int(time.time() * 1000) + 10,
                    "bounds": b,
                    "explored_ratio": 0.95,
                    "occupancy_subgrid": [[0] * 10 for _ in range(10)],
                },
            }
            station_sock.sendto(json.dumps(resp).encode("utf-8"), addr)

    def _sat_worker():
        while not stop_ev.is_set():
            try:
                conn, _ = sat_srv.accept()
            except socket.timeout:
                continue
            with conn:
                data = conn.recv(1024).decode("utf-8", errors="ignore")
                if data:
                    sat_received_lines.append(data)

    st_thread = threading.Thread(target=_station_worker, daemon=True)
    sa_thread = threading.Thread(target=_sat_worker, daemon=True)
    st_thread.start()
    sa_thread.start()

    cm = CentroMandoServer(
        listen_host="127.0.0.1",
        listen_port=0,
        station_ip="127.0.0.1",
        station_udp_port=station_port,
        station_timeout_s=0.2,
        satellite_ip="127.0.0.1",
        satellite_tcp_port=sat_port,
        cache_dir=str(tmp_path / "cm_cache"),
    )
    cm.start()

    link = CentroMandoLink(
        centro_mando_ip="127.0.0.1",
        centro_mando_port=cm.listen_port,
        astronaut_udp_ip="127.0.0.1",
        astronaut_udp_port=astro_port,
    )
    link.start()

    try:
        cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
        cam = SyntheticTerrainCamera(cfg)
        cam.start()
        frame = cam.read_frame()
        cam.stop()

        pipeline = TerrainNavigationPipeline(cfg)
        pipeline.set_goal_normalized(0.5, 0.15)
        output, _, _ = pipeline.process_frame(frame)
        pred = predict_trajectory(output)
        ruta_txt = format_esp32_ruta_txt(
            mode=output.navigation.navigation_mode,
            planner=output.navigation.planner_used,
            safe_corridor=output.navigation.safe_corridor,
            altitude_m=output.terrain.altitude_m,
            heading_deg=output.navigation.recommended_heading_deg,
            timestamp_ms=pred.timestamp_ms,
            waypoints_xy=output.navigation.path_world_xy,
        )
        patch = extract_quadrant_patch(
            occupancy_grid=output.navigation.occupancy_grid,
            cost_grid=output.navigation.cost_grid,
            bounds=pred.current_bounds,
            resolution_m=cfg.grid_resolution_m,
            y_min_m=pipeline.grid_builder.y_min_m,
            grid_width_m=cfg.grid_width_m,
        )

        # 1. Estacion responde -> QUADRANT_DATA llega a Laptop Bernie
        assert link.send_to_centro_mando(pred.to_dict(), ruta_txt, scanned_patch=patch)
        deadline = time.time() + 2.0
        incoming = []
        while time.time() < deadline and not incoming:
            incoming = link.pop_incoming_quadrants()
            if not incoming:
                time.sleep(0.02)
        assert len(incoming) == 1
        assert link.link_mode == "STATION_DIRECT"

        # 2. Interferencia con Estacion + cuadrante desconocido -> fallback a Satelite ESP32 y SCAN_REQUEST
        station_enabled[0] = False
        pred_missing = pred.to_dict()
        pred_missing["next_quadrant"] = {
            "x_min": 4.5,
            "x_max": 6.0,
            "y_min": 5.0,
            "y_max": 6.0,
        }
        assert link.send_to_centro_mando(pred_missing, ruta_txt)
        deadline = time.time() + 2.5
        scan_req = None
        while time.time() < deadline and scan_req is None:
            scan_req = link.get_active_scan_request()
            if scan_req is None:
                time.sleep(0.02)
        assert scan_req is not None
        assert scan_req["target_xy_m"] == [5.25, 5.5]
        assert link.link_mode == "SATELLITE_MESH"
        assert len(sat_received_lines) >= 1
        assert sat_received_lines[0].startswith("RUTA_V1|")

        # 3. Paquete UDP ligero al dispositivo del astronauta (< 1200 B)
        assert link.send_astronaut_udp(
            mode=output.navigation.navigation_mode,
            planner=output.navigation.planner_used,
            safe_corridor=output.navigation.safe_corridor,
            altitude_m=output.terrain.altitude_m,
            heading_deg=output.navigation.recommended_heading_deg,
            pose_xy=(output.rover_pose_xy_yaw[0], output.rover_pose_xy_yaw[1]),
            prediction_dict=pred.to_dict(),
            waypoints_xy=output.navigation.path_world_xy,
            scan_request=scan_req,
        )
        astro_raw, _ = astro_sock.recvfrom(2048)
        assert len(astro_raw) < 1200
        astro_msg = json.loads(astro_raw.decode("utf-8"))
        assert astro_msg["node"] == "drone_cv"
        assert astro_msg["scan"] == [5.25, 5.5]
    finally:
        stop_ev.set()
        link.stop()
        cm.stop()
        st_thread.join(timeout=1.0)
        sa_thread.join(timeout=1.0)
        station_sock.close()
        sat_srv.close()
        astro_sock.close()


def test_expo_server_accepts_nested_request_route_target():
    from src.transport.server import ExpoLANServer

    srv = ExpoLANServer(http_port=0)
    ok = srv._push_goal_command(
        json.dumps(
            {
                "action": "request_route",
                "unit": "meters",
                "target": {"x_m": 1.25, "y_m": -0.75},
            }
        )
    )
    assert ok is True
    cmd = srv.pop_goal_command()
    assert cmd == {"x_m": 1.25, "y_m": -0.75}
