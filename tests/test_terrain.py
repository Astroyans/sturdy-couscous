import numpy as np

from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.preprocessing.filters import preprocess_depth
from src.terrain.ground_plane import GroundPlaneAnalyzer


def test_ground_plane_ransac_recovers_incline_and_altitude():
    cfg = PipelineConfig(width=320, height=240)
    target_incline_deg = 7.5
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=target_incline_deg, add_noise=False)
    frame = cam.read_frame()

    filtered_depth, valid_mask = preprocess_depth(frame.depth_m, cfg)
    analyzer = GroundPlaneAnalyzer(cfg)
    terrain = analyzer.analyze(filtered_depth, valid_mask, frame.intrinsics)

    assert abs(terrain.pitch_deg - target_incline_deg) < 1.5
    assert abs(terrain.max_incline_deg - target_incline_deg) < 2.0
    assert terrain.altitude_m > 0.20

    assert float(np.max(terrain.height_map_m)) > 0.08
    assert float(np.min(terrain.height_map_m)) < -0.08


def test_plane_lock_and_unlock():
    cfg = PipelineConfig(width=160, height=120)
    analyzer = GroundPlaneAnalyzer(cfg)
    assert not analyzer.is_plane_locked

    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    analyzer.lock_plane(normal, 0.50)
    assert analyzer.is_plane_locked

    analyzer.unlock_plane()
    assert not analyzer.is_plane_locked


def test_altitude_uses_deep_tof_rays_ignoring_foreground_table():
    cfg = PipelineConfig(width=320, height=240, nominal_camera_pitch_deg=-90.0)
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=0.0, add_noise=False)
    frame = cam.read_frame()

    depth_m = np.full((240, 320), 1.15, dtype=np.float32)
    depth_m[:135, :] = 0.27

    filtered_depth, valid_mask = preprocess_depth(depth_m, cfg)
    analyzer = GroundPlaneAnalyzer(cfg)
    terrain = analyzer.analyze(filtered_depth, valid_mask, frame.intrinsics)

    assert terrain.altitude_m > 1.05
    assert float(np.median(terrain.height_map_m[:100, :])) > 0.70

