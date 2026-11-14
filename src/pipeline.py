from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.models import FrameBundle, PipelineOutput, TerrainAnalysis
from src.navigation.occupancy_grid import OccupancyGridBuilder
from src.navigation.path_planner import PathPlanner
from src.obstacles.detector import ObstacleDetector
from src.odometry.visual_odometry import Pose2D, RGBDVisualOdometry
from src.preprocessing.filters import preprocess_depth
from src.terrain.ground_plane import GroundPlaneAnalyzer


class TerrainNavigationPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.ground_analyzer = GroundPlaneAnalyzer(self.config)
        self.obstacle_detector = ObstacleDetector(self.config)
        self.grid_builder = OccupancyGridBuilder(self.config)
        self.path_planner = PathPlanner(self.config, grid_builder=self.grid_builder)
        self.visual_odometry = RGBDVisualOdometry(self.config)
        self.planner_algorithm: str = self.config.default_planner
        self.navigation_mode: str = "EXPLORE_FORWARD"
        self.target_goal_xy: Optional[tuple[float, float]] = None
        self._ema_fps: float = 0.0
        self._last_terrain: Optional[TerrainAnalysis] = None

    def toggle_planner(self) -> str:
        self.planner_algorithm = (
            "dijkstra" if self.planner_algorithm == "astar" else "astar"
        )
        return self.planner_algorithm

    def toggle_navigation_mode(self) -> str:
        self.navigation_mode = (
            "RETURN_TO_BASE"
            if self.navigation_mode == "EXPLORE_FORWARD"
            else "EXPLORE_FORWARD"
        )
        return self.navigation_mode

    def toggle_plane_lock(self) -> bool:
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

    def set_navigation_mode(self, mode: str) -> str:
        self.navigation_mode = (
            "RETURN_TO_BASE" if mode == "RETURN_TO_BASE" else "EXPLORE_FORWARD"
        )
        return self.navigation_mode

    def set_goal_world(self, x_m: float, y_m: float) -> tuple[float, float]:
        self.target_goal_xy = (float(x_m), float(y_m))
        return self.target_goal_xy

    def set_goal_normalized(self, norm_x: float, norm_y: float) -> tuple[float, float]:
        nx = max(0.0, min(1.0, float(norm_x)))
        ny = max(0.0, min(1.0, float(norm_y)))
        rows, cols = self.grid_builder.grid_shape
        col = int(round(nx * (cols - 1)))
        row = int(round((1.0 - ny) * (rows - 1)))
        wx, wy = self.grid_builder.cell_to_world(row, col)
        self.target_goal_xy = (wx, wy)
        return self.target_goal_xy

    def clear_goal(self) -> None:
        self.target_goal_xy = None

    def reset_global_map(self) -> None:
        self.grid_builder.reset()
        self.visual_odometry.reset()
        self.target_goal_xy = None

    def advance_pose_manually(
        self,
        dx_local_m: float = 0.0,
        dy_local_m: float = 0.0,
        dyaw_deg: float = 0.0,
    ) -> Pose2D:
        return self.visual_odometry.apply_manual_step(
            dx_local_m=dx_local_m,
            dy_local_m=dy_local_m,
            dyaw_deg=dyaw_deg,
        )

    def process_frame(
        self,
        frame: FrameBundle,
    ) -> tuple[PipelineOutput, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()

        filtered_depth_m, valid_mask = preprocess_depth(frame.depth_m, self.config)

        terrain = self.ground_analyzer.analyze(
            filtered_depth_m,
            valid_mask,
            frame.intrinsics,
            camera_pitch_deg=frame.imu_pitch,
        )
        self._last_terrain = terrain

        pose = self.visual_odometry.update(
            frame.color_bgr,
            terrain.points_3d,
            terrain.valid_mask,
        )

        frame_obstacles, boulder_mask, crevice_mask = self.obstacle_detector.detect(
            terrain
        )

        occupancy_grid, cost_grid, safe_corridor = self.grid_builder.build(
            terrain,
            frame_obstacles,
            boulder_mask,
            crevice_mask,
            pose=pose,
        )

        # Usar la lista persistente de obstaculos globales acumulados a lo largo del recorrido
        active_obstacles = (
            list(self.grid_builder.global_obstacles)
            if self.config.persistent_mapping
            else frame_obstacles
        )

        navigation = self.path_planner.plan(
            occupancy_grid,
            cost_grid,
            safe_corridor,
            algorithm=self.planner_algorithm,
            pose=pose,
            navigation_mode=self.navigation_mode,
            goal_world_xy=self.target_goal_xy,
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
            obstacles=active_obstacles,
            navigation=navigation,
            edge_fps=self._ema_fps,
            rover_pose_xy_yaw=(pose.x_m, pose.y_m, pose.yaw_deg),
            traversed_distance_m=pose.total_distance_m,
            frame_obstacles=frame_obstacles,
        )
        return output, boulder_mask, crevice_mask
