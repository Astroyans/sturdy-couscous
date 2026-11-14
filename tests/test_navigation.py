"""Pruebas unitarias para Task 5 (Fase 4): Malla de Ocupación 2D y rutas A* / Dijkstra."""

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
    cfg = PipelineConfig(width=320, height=240, min_cluster_pixels=40)
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
    res_astar = planner.plan(grid, cost_grid, safe_corridor, algorithm="astar")
    res_dijkstra = planner.plan(grid, cost_grid, safe_corridor, algorithm="dijkstra")

    assert len(res_astar.path_cells) > 10
    assert len(res_dijkstra.path_cells) > 10

    # Ninguna celda de la ruta óptima puede pisar un BOULDER o CREVICE
    for r, c in res_astar.path_cells:
        assert grid[r, c] not in (GRID_BOULDER, GRID_CREVICE)

    for r, c in res_dijkstra.path_cells:
        assert grid[r, c] not in (GRID_BOULDER, GRID_CREVICE)

    # A* debe explorar igual o menos nodos que Dijkstra para llegar a la misma meta
    assert res_astar.explored_nodes <= res_dijkstra.explored_nodes
