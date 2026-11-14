import os
import tempfile

import numpy as np

from main import main
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.pipeline import TerrainNavigationPipeline
from src.visualization.hud import HUDRenderer


def test_full_pipeline_and_hud_rendering():
    cfg = PipelineConfig(
        width=320,
        height=240,
        min_cluster_pixels=40,
        boulder_min_height_m=0.05,
        crevice_min_depth_m=0.05,
    )
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=6.0)
    pipeline = TerrainNavigationPipeline(cfg)
    hud = HUDRenderer(cfg)

    frame = cam.read_frame()
    output_idle, _, _ = pipeline.process_frame(frame)
    assert output_idle.navigation.path_world_xy == []

    pipeline.set_goal_normalized(0.5, 0.2)
    output, boulder_mask, crevice_mask = pipeline.process_frame(frame)
    assert len(output.navigation.path_world_xy) >= 2
    canvas = hud.render(output, boulder_mask, crevice_mask, source_name=cam.device_name)

    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8

    payload = output.to_drone_cv_payload()
    assert payload["node"] == "drone_cv"
    assert isinstance(payload["data"]["safe_corridor"], bool)
    assert len(payload["data"]["obstacles"]) >= 2

    assert pipeline.toggle_planner() == "dijkstra"
    assert pipeline.toggle_plane_lock() is True
    output2, _, _ = pipeline.process_frame(frame)
    assert output2.navigation.planner_used == "dijkstra"
    assert pipeline.toggle_plane_lock() is False


def test_cli_synthetic_headless_and_snapshot():
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = main(
            [
                "--source",
                "synthetic",
                "--planner",
                "astar",
                "--frames",
                "3",
                "--headless",
                "--save-snapshot",
                "--save-dir",
                tmpdir,
            ]
        )
        assert rc == 0
        files = os.listdir(tmpdir)
        assert any(f.endswith(".png") for f in files)
        assert any(f.endswith(".json") for f in files)
