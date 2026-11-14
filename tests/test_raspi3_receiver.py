import socket
import threading
import time

import numpy as np

from src.camera.raspi3_receiver import (
    Raspi3ReceiverCamera,
    decode_raspi3_packet,
    encode_raspi3_packet,
)
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.pipeline import TerrainNavigationPipeline


def test_raspi3_packet_roundtrip_rgb_and_depth_only():
    cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
    cam = SyntheticTerrainCamera(cfg)
    cam.start()
    frame = cam.read_frame()
    cam.stop()

    pkt = encode_raspi3_packet(
        depth_m=frame.depth_m,
        intrinsics=frame.intrinsics,
        color_bgr=frame.color_bgr,
        imu_yaw=12.5,
        imu_pitch=-89.2,
        imu_roll=1.4,
        jpeg_quality=80,
    )
    assert pkt[:4] == b"R3D1"

    header_len = int.from_bytes(pkt[4:8], "big")
    pos = 8
    header_bytes = pkt[pos : pos + header_len]
    pos += header_len
    c_len = int.from_bytes(pkt[pos : pos + 4], "big")
    pos += 4
    color_bytes = pkt[pos : pos + c_len]
    pos += c_len
    d_len = int.from_bytes(pkt[pos : pos + 4], "big")
    pos += 4
    depth_bytes = pkt[pos : pos + d_len]

    decoded, latency_ms = decode_raspi3_packet(header_bytes, color_bytes, depth_bytes)
    assert latency_ms >= 0.0
    assert decoded.depth_m.shape == (240, 320)
    assert decoded.color_bgr.shape == (240, 320, 3)
    assert abs(decoded.imu_yaw - 12.5) < 1e-4
    assert abs(decoded.imu_pitch - (-89.2)) < 1e-4
    assert np.allclose(decoded.depth_m, frame.depth_m, atol=0.002)

    pkt_depth_only = encode_raspi3_packet(
        depth_m=frame.depth_m,
        intrinsics=frame.intrinsics,
        color_bgr=None,
    )
    h2_len = int.from_bytes(pkt_depth_only[4:8], "big")
    p2 = 8
    h2_bytes = pkt_depth_only[p2 : p2 + h2_len]
    p2 += h2_len
    c2_len = int.from_bytes(pkt_depth_only[p2 : p2 + 4], "big")
    assert c2_len == 0
    p2 += 4
    d2_len = int.from_bytes(pkt_depth_only[p2 : p2 + 4], "big")
    p2 += 4
    d2_bytes = pkt_depth_only[p2 : p2 + d2_len]
    decoded_only, _ = decode_raspi3_packet(h2_bytes, b"", d2_bytes)
    assert decoded_only.color_bgr.shape == (240, 320, 3)


def test_raspi3_receiver_server_mode_feeds_pipeline():
    cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
    syn = SyntheticTerrainCamera(cfg)
    syn.start()
    sample = syn.read_frame()
    syn.stop()

    receiver = Raspi3ReceiverCamera(
        config=cfg,
        listen_host="127.0.0.1",
        port=0,
        wait_timeout_s=5.0,
    )

    def _mock_raspi3_sender():
        time.sleep(0.15)
        pkt = encode_raspi3_packet(
            depth_m=sample.depth_m,
            intrinsics=sample.intrinsics,
            color_bgr=sample.color_bgr,
            imu_pitch=-90.0,
        )
        with socket.create_connection(("127.0.0.1", receiver.port), timeout=2.0) as s:
            s.sendall(pkt)
            time.sleep(0.05)
            s.sendall(pkt)
            time.sleep(0.2)

    tx_thread = threading.Thread(target=_mock_raspi3_sender, daemon=True)
    receiver.start(wait_for_first_frame=False)
    tx_thread.start()
    try:
        deadline = time.time() + 4.0
        while time.time() < deadline and receiver.frames_received < 1:
            time.sleep(0.02)
        assert receiver.frames_received >= 1
        frame = receiver.read_frame(timeout_ms=1000)
        pipeline = TerrainNavigationPipeline(cfg)
        output, _, _ = pipeline.process_frame(frame)
        payload = output.to_drone_cv_payload()
        assert payload["node"] == "drone_cv"
        assert payload["data"]["altitude_m"] > 0.0
    finally:
        receiver.stop()
        tx_thread.join(timeout=1.0)
