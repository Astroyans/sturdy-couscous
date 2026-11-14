import numpy as np

from src.config import PipelineConfig
from src.models import (
    CameraIntrinsics,
    FrameBundle,
    NavigationResult,
    Obstacle,
    PipelineOutput,
    TerrainAnalysis,
)


def test_obstacle_contract_serialization():
    boulder = Obstacle(
        id=1,
        type="BOULDER",
        distance_m=1.854,
        bearing_deg=-15.04,
        magnitude_m=0.421,
        bbox_2d=(100, 120, 50, 60),
        centroid_3d=(-0.48, 1.79, 0.21),
        pixel_count=450,
    )
    crevice = Obstacle(
        id=2,
        type="CREVICE",
        distance_m=3.102,
        bearing_deg=8.01,
        magnitude_m=0.654,
        bbox_2d=(300, 200, 80, 40),
        centroid_3d=(0.43, 3.07, -0.32),
        pixel_count=520,
    )

    b_dict = boulder.to_contract_dict()
    assert b_dict == {
        "id": 1,
        "type": "BOULDER",
        "distance_m": 1.85,
        "bearing_deg": -15.0,
        "height_m": 0.42,
    }
    assert "depth_m" not in b_dict

    c_dict = crevice.to_contract_dict()
    assert c_dict == {
        "id": 2,
        "type": "CREVICE",
        "distance_m": 3.1,
        "bearing_deg": 8.0,
        "depth_m": 0.65,
    }
    assert "height_m" not in c_dict


def test_pipeline_output_to_drone_cv_payload():
    cfg = PipelineConfig()
    intrinsics = CameraIntrinsics.default_d415(cfg.width, cfg.height)
    frame = FrameBundle(
        timestamp=1731456003.0,
        color_bgr=np.zeros((60, 80, 3), dtype=np.uint8),
        depth_m=np.ones((60, 80), dtype=np.float32),
        intrinsics=intrinsics,
        imu_yaw=92.0,
    )
    terrain = TerrainAnalysis(
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        plane_d=-2.45,
        altitude_m=2.45,
        pitch_deg=4.5,
        roll_deg=-1.2,
        max_incline_deg=14.8,
        height_map_m=np.zeros((60, 80), dtype=np.float32),
        slope_map_deg=np.zeros((60, 80), dtype=np.float32),
        points_3d=np.zeros((60, 80, 3), dtype=np.float32),
        valid_mask=np.ones((60, 80), dtype=bool),
    )
    obstacles = [
        Obstacle(
            id=1,
            type="BOULDER",
            distance_m=1.85,
            bearing_deg=-15.0,
            magnitude_m=0.42,
            bbox_2d=(10, 10, 20, 20),
            centroid_3d=(-0.4, 1.8, 0.42),
        )
    ]
    nav = NavigationResult(
        occupancy_grid=np.zeros((20, 20), dtype=np.uint8),
        cost_grid=np.ones((20, 20), dtype=np.float32),
        safe_corridor=True,
    )
    output = PipelineOutput(
        frame=frame,
        preprocessed_depth_m=frame.depth_m,
        terrain=terrain,
        obstacles=obstacles,
        navigation=nav,
        edge_fps=28.5,
    )

    payload = output.to_drone_cv_payload()
    assert payload["node"] == "drone_cv"
    assert payload["timestamp"] == 1731456003
    assert payload["data"]["altitude_m"] == 2.45
    assert payload["data"]["imu"] == {"yaw": 92.0, "pitch": 4.5, "roll": -1.2}
    assert payload["data"]["max_incline_deg"] == 14.8
    assert payload["data"]["safe_corridor"] is True
    assert len(payload["data"]["obstacles"]) == 1
    assert payload["data"]["edge_fps"] == 28.5
