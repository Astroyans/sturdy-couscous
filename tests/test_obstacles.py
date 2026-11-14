"""Pruebas unitarias para Task 4 (Fase 3): detección de `BOULDER` y `CREVICE` con DBSCAN."""

from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.obstacles.detector import ObstacleDetector
from src.preprocessing.filters import preprocess_depth
from src.terrain.ground_plane import GroundPlaneAnalyzer


def test_obstacle_detector_identifies_boulder_and_crevice():
    cfg = PipelineConfig(width=320, height=240, min_cluster_pixels=40)
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=5.0, add_noise=False)
    frame = cam.read_frame()

    filtered_depth, valid_mask = preprocess_depth(frame.depth_m, cfg)
    terrain = GroundPlaneAnalyzer(cfg).analyze(
        filtered_depth, valid_mask, frame.intrinsics
    )
    detector = ObstacleDetector(cfg)
    obstacles, boulder_mask, crevice_mask = detector.detect(terrain)

    types_found = {obs.type for obs in obstacles}
    assert "BOULDER" in types_found
    assert "CREVICE" in types_found

    boulders = [o for o in obstacles if o.type == "BOULDER"]
    crevices = [o for o in obstacles if o.type == "CREVICE"]

    # El boulder sintético está a la izquierda (bearing < 0) y a ~1.28 m
    b = boulders[0]
    assert b.bearing_deg < -5.0
    assert 0.9 < b.distance_m < 1.7
    assert 0.08 < b.magnitude_m < 0.22
    assert "height_m" in b.to_contract_dict()

    # El crevice sintético está a la derecha (bearing > 0) y a ~1.48 m
    c = crevices[0]
    assert c.bearing_deg > 5.0
    assert 1.0 < c.distance_m < 1.9
    assert 0.07 < c.magnitude_m < 0.22
    assert "depth_m" in c.to_contract_dict()

    assert boulder_mask.any()
    assert crevice_mask.any()
