"""Orquestador central del pipeline de análisis topográfico y navegación (Fases 1 a 5)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.models import FrameBundle, PipelineOutput, TerrainAnalysis
from src.navigation.occupancy_grid import OccupancyGridBuilder
from src.navigation.path_planner import PathPlanner
from src.obstacles.detector import ObstacleDetector
from src.preprocessing.filters import preprocess_depth
from src.terrain.ground_plane import GroundPlaneAnalyzer


class TerrainNavigationPipeline:
    """Integra preprocesamiento, plano RANSAC, obstáculos DBSCAN y navegación A*/Dijkstra."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.ground_analyzer = GroundPlaneAnalyzer(self.config)
        self.obstacle_detector = ObstacleDetector(self.config)
        self.grid_builder = OccupancyGridBuilder(self.config)
        self.path_planner = PathPlanner(self.config)
        self.planner_algorithm: str = self.config.default_planner
        self._ema_fps: float = 0.0
        self._last_terrain: Optional[TerrainAnalysis] = None

    def toggle_planner(self) -> str:
        """Alterna en vivo entre los algoritmos `'astar'` y `'dijkstra'`."""
        self.planner_algorithm = (
            "dijkstra" if self.planner_algorithm == "astar" else "astar"
        )
        return self.planner_algorithm

    def toggle_plane_lock(self) -> bool:
        """Fija o libera el plano de referencia del suelo/mesa usando la última medición."""
        if self.ground_analyzer.is_plane_locked:
            self.ground_analyzer.unlock_plane()
            return False
        if self._last_terrain is not None:
            self.ground_analyzer.lock_plane(
                self._last_terrain.plane_normal,
                self._last_terrain.plane_d,
            )
            return True
        return False

    def process_frame(
        self,
        frame: FrameBundle,
    ) -> tuple[PipelineOutput, np.ndarray, np.ndarray]:
        """Ejecuta las Fases 1 a 4 sobre un `FrameBundle` y mide los FPS de procesamiento."""
        t0 = time.perf_counter()

        # Fase 1: Preprocesamiento Gaussiano + Morfológico
        filtered_depth_m, valid_mask = preprocess_depth(frame.depth_m, self.config)

        # Fase 2: Deproyección 3D, plano del suelo (RANSAC) e inclinación
        terrain = self.ground_analyzer.analyze(
            filtered_depth_m,
            valid_mask,
            frame.intrinsics,
            camera_pitch_deg=frame.imu_pitch,
        )
        self._last_terrain = terrain

        # Fase 3: Segmentación de obstáculos BOULDER y CREVICE (Morfología + DBSCAN)
        obstacles, boulder_mask, crevice_mask = self.obstacle_detector.detect(terrain)

        # Fase 4: Malla de ocupación 2D y planificación de ruta (A* / Dijkstra)
        occupancy_grid, cost_grid, safe_corridor = self.grid_builder.build(
            terrain, obstacles, boulder_mask, crevice_mask
        )
        navigation = self.path_planner.plan(
            occupancy_grid,
            cost_grid,
            safe_corridor,
            algorithm=self.planner_algorithm,
        )

        dt = max(time.perf_counter() - t0, 1e-4)
        inst_fps = 1.0 / dt
        if self._ema_fps <= 0.0:
            self._ema_fps = inst_fps
        else:
            self._ema_fps = 0.8 * self._ema_fps + 0.2 * inst_fps

        output = PipelineOutput(
            frame=frame,
            preprocessed_depth_m=filtered_depth_m,
            terrain=terrain,
            obstacles=obstacles,
            navigation=navigation,
            edge_fps=self._ema_fps,
        )
        return output, boulder_mask, crevice_mask
