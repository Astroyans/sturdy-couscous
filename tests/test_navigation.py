import numpy as np

from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.models import GRID_BOULDER, GRID_CREVICE
from src.navigation.occupancy_grid import OccupancyGridBuilder
from src.navigation.path_planner import PathPlanner
from src.obstacles.detector import ObstacleDetector
from src.preprocessing.filters import preprocess_depth
from src.terrain.ground_plane import GroundPlaneAnalyzer


def test_occupancy_grid_and_astar_vs_dijkstra_avoid_obstacles():
    cfg = PipelineConfig(
        width=320,
        height=240,
        min_cluster_pixels=40,
        boulder_min_height_m=0.05,
        crevice_min_depth_m=0.05,
    )
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=5.0)
    frame = cam.read_frame()

    filtered_depth, valid_mask = preprocess_depth(frame.depth_m, cfg)
    terrain = GroundPlaneAnalyzer(cfg).analyze(
        filtered_depth, valid_mask, frame.intrinsics
    )
    obstacles, boulder_mask, crevice_mask = ObstacleDetector(cfg).detect(terrain)

    builder = OccupancyGridBuilder(cfg)
    grid, cost_grid, safe_corridor = builder.build(
        terrain, obstacles, boulder_mask, crevice_mask
    )

    assert np.any(grid == GRID_BOULDER)
    assert np.any(grid == GRID_CREVICE)

    planner = PathPlanner(cfg)
    res_idle = planner.plan(grid, cost_grid, safe_corridor, algorithm="astar")
    assert res_idle.path_cells == []

    goal_xy = (0.0, 4.0)
    res_astar = planner.plan(
        grid, cost_grid, safe_corridor, algorithm="astar", goal_world_xy=goal_xy
    )
    res_dijkstra = planner.plan(
        grid, cost_grid, safe_corridor, algorithm="dijkstra", goal_world_xy=goal_xy
    )

    assert len(res_astar.path_cells) > 10
    assert len(res_dijkstra.path_cells) > 10

    for r, c in res_astar.path_cells:
        assert grid[r, c] not in (GRID_BOULDER, GRID_CREVICE)

    for r, c in res_dijkstra.path_cells:
        assert grid[r, c] not in (GRID_BOULDER, GRID_CREVICE)

    assert res_astar.explored_nodes <= res_dijkstra.explored_nodes
