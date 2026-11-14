import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.pipeline import TerrainNavigationPipeline
from src.transport.client import TelemetrySender
from src.transport.media import build_expo_payload
from src.transport.server import ExpoLANServer


def test_build_expo_payload_contains_telemetry_and_three_jpegs():
    cfg = PipelineConfig()
    cam = SyntheticTerrainCamera(cfg)
    cam.start()
    frame = cam.read_frame()
    cam.stop()

    pipeline = TerrainNavigationPipeline(cfg)
    output, b_mask, c_mask = pipeline.process_frame(frame)

    payload = build_expo_payload(
        output=output,
        config=cfg,
        boulder_mask=b_mask,
        crevice_mask=c_mask,
        image_width=640,
        image_height=480,
        jpeg_quality=80,
    )

    assert payload["node"] == "drone_cv"
    assert "altitude_m" in payload["data"]
    assert "navigation" in payload["data"]
    assert "astar" in payload["data"]["navigation"]
    assert "dijkstra" in payload["data"]["navigation"]

    images = payload["images"]
    assert images["format"] == "jpeg_base64"
    assert images["width"] == 640
    assert images["height"] == 480

    for key in ("rgb", "depth", "route_map"):
        uri = images[key]
        assert uri.startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(uri.split(",", 1)[1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (480, 640, 3)


def test_telemetry_sender_http_delivers_payload():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received.append(json.loads(body.decode("utf-8")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    sender = TelemetrySender(
        url=f"http://127.0.0.1:{port}/api/telemetry",
        transport="http",
        timeout_s=2.0,
    )
    sender.start()
    try:
        sender.enqueue({"node": "drone_cv", "data": {"altitude_m": 1.2}})
        deadline = time.time() + 3.0
        while time.time() < deadline and sender.sent_count == 0:
            time.sleep(0.05)
        assert sender.sent_count == 1
        assert sender.failed_count == 0
        assert len(received) == 1
        assert received[0]["node"] == "drone_cv"
    finally:
        sender.stop()
        server.shutdown()
        server.server_close()


def test_telemetry_sender_does_not_block_when_backend_offline():
    sender = TelemetrySender(
        url="http://127.0.0.1:59999/api/telemetry",
        transport="http",
        timeout_s=0.3,
    )
    sender.start()
    try:
        t0 = time.perf_counter()
        sender.enqueue({"node": "drone_cv", "test": 1})
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2
        time.sleep(0.5)
        assert sender.failed_count >= 1
    finally:
        sender.stop()


def test_expo_lan_server_serves_latest_and_accepts_goal_for_ipad():
    import requests

    srv = ExpoLANServer(host="127.0.0.1", http_port=0, ws_port=-1)
    srv.start()
    try:
        srv.publish({"node": "drone_cv", "data": {"altitude_m": 5.2}})
        r1 = requests.get(f"http://127.0.0.1:{srv.http_port}/api/latest", timeout=2.0)
        assert r1.status_code == 200
        assert r1.json()["data"]["altitude_m"] == 5.2

        r2 = requests.post(
            f"http://127.0.0.1:{srv.http_port}/api/goal",
            json={"norm_x": 0.4, "norm_y": 0.25},
            timeout=2.0,
        )
        assert r2.status_code == 200
        cmd = srv.pop_goal_command()
        assert cmd == {"norm_x": 0.4, "norm_y": 0.25}
    finally:
        srv.stop()


def test_nadir_aerial_view_and_on_demand_goal_flow():
    cfg = PipelineConfig(
        width=320,
        height=240,
        nominal_camera_pitch_deg=-90.0,
        boulder_min_height_m=0.05,
        crevice_min_depth_m=0.05,
        min_cluster_pixels=40,
    )
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=4.0)
    cam.start()
    frame = cam.read_frame()
    cam.stop()

    pipeline = TerrainNavigationPipeline(cfg)
    out_idle, _, _ = pipeline.process_frame(frame)
    rows, cols = out_idle.navigation.occupancy_grid.shape
    assert out_idle.navigation.start_cell == (rows // 2, cols // 2)
    assert out_idle.navigation.path_world_xy == []

    pipeline.set_goal_normalized(0.5, 0.15)
    out_routed, _, _ = pipeline.process_frame(frame)
    assert len(out_routed.navigation.astar_path_world_xy) >= 5
    assert len(out_routed.navigation.dijkstra_path_world_xy) >= 5

    pipeline.clear_goal()
    out_cleared, _, _ = pipeline.process_frame(frame)
    assert out_cleared.navigation.path_world_xy == []
