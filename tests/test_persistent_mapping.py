import numpy as np

from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.models import GRID_BOULDER, GRID_CREVICE, GRID_UNKNOWN
from src.pipeline import TerrainNavigationPipeline


def test_persistent_map_accumulates_across_multiple_frames_and_poses():
    cfg = PipelineConfig(
        width=320,
        height=240,
        min_cluster_pixels=40,
        boulder_min_height_m=0.05,
        crevice_min_depth_m=0.05,
        grid_width_m=4.0,
        grid_depth_m=5.0,
        persistent_mapping=True,
    )
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=5.0)
    pipeline = TerrainNavigationPipeline(cfg)

    frame1 = cam.read_frame()
    out1, _, _ = pipeline.process_frame(frame1)
    explored_area_1 = out1.navigation.explored_area_m2
    obs_count_1 = len(out1.obstacles)
    assert obs_count_1 >= 2

    # Avanzar el dron/rover 2.0 metros hacia adelante:
    # los obstaculos del frame 1 quedan atras de la camara, pero DEBEN persistir en el mapa global
    pipeline.advance_pose_manually(dy_local_m=2.0)
    frame2 = cam.read_frame()
    out2, _, _ = pipeline.process_frame(frame2)

    explored_area_2 = out2.navigation.explored_area_m2
    assert explored_area_2 > explored_area_1
    assert out2.traversed_distance_m >= 2.0
    assert len(out2.obstacles) >= obs_count_1
    assert len(out2.navigation.trajectory_cells) >= 2

    # Verificar que la malla global conserva los obstaculos tanto en la zona inicial como en la nueva
    grid = out2.navigation.occupancy_grid
    assert np.any(grid[:35, :] == GRID_BOULDER)
    assert np.any(grid[35:, :] == GRID_BOULDER)

    # Probar modo de retorno de emergencia a la Base (RETURN_TO_BASE) con Dijkstra y A*
    pipeline.toggle_navigation_mode()
    out_ret_astar, _, _ = pipeline.process_frame(frame2)
    assert out_ret_astar.navigation.navigation_mode == "RETURN_TO_BASE"
    assert out_ret_astar.navigation.goal_cell == out_ret_astar.navigation.start_cell

    pipeline.toggle_planner()
    out_ret_dijkstra, _, _ = pipeline.process_frame(frame2)
    assert out_ret_dijkstra.navigation.planner_used == "dijkstra"
    assert len(out_ret_dijkstra.navigation.path_cells) > 15
    assert int(np.sum(grid != GRID_UNKNOWN)) > 0
