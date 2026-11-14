from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import (
    GRID_BOULDER,
    GRID_CREVICE,
    GRID_FREE,
    GRID_INFLATED,
    GRID_STEEP,
    GRID_UNKNOWN,
    Obstacle,
    TerrainAnalysis,
)
from src.odometry.visual_odometry import Pose2D


class OccupancyGridBuilder:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        rows, cols = self.grid_shape
        self._log_odds = np.zeros((rows, cols), dtype=np.float32)
        self._observed = np.zeros((rows, cols), dtype=bool)
        self._obstacle_type_layer = np.zeros((rows, cols), dtype=np.uint8)
        self.trajectory_cells: list[tuple[int, int]] = []
        self.global_obstacles: list[Obstacle] = []
        self._next_global_id: int = 1
        self._frame_counter: int = 0

    @property
    def grid_shape(self) -> tuple[int, int]:
        rows = max(10, int(round(self.config.grid_depth_m / self.config.grid_resolution_m)))
        cols = max(10, int(round(self.config.grid_width_m / self.config.grid_resolution_m)))
        return rows, cols

    @property
    def y_min_m(self) -> float:
        if self.config.nominal_camera_pitch_deg <= -65.0:
            return -self.config.grid_depth_m / 2.0
        return 0.0

    def reset(self) -> None:
        rows, cols = self.grid_shape
        self._log_odds.fill(0.0)
        self._observed.fill(False)
        self._obstacle_type_layer.fill(0)
        self.trajectory_cells.clear()
        self.global_obstacles.clear()
        self._next_global_id = 1
        self._frame_counter = 0

    def _ensure_capacity_for_pose(self, pose: Pose2D) -> None:
        rows, cols = self._log_odds.shape
        res = self.config.grid_resolution_m
        needed_y_m = (pose.y_m - self.y_min_m) + 3.0
        needed_rows = int(math.ceil(needed_y_m / res))
        if needed_rows > rows:
            extra_rows = max(needed_rows - rows, 20)
            self._log_odds = np.pad(
                self._log_odds, ((0, extra_rows), (0, 0)), mode="constant"
            )
            self._observed = np.pad(
                self._observed, ((0, extra_rows), (0, 0)), mode="constant"
            )
            self._obstacle_type_layer = np.pad(
                self._obstacle_type_layer, ((0, extra_rows), (0, 0)), mode="constant"
            )
            self.config.grid_depth_m = self._log_odds.shape[0] * res

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        rows, cols = self._log_odds.shape
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0

        row = int(np.clip(math.floor((y_m - self.y_min_m) / res), 0, rows - 1))
        col = int(np.clip(math.floor((x_m + half_w) / res), 0, cols - 1))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0
        x_m = (col + 0.5) * res - half_w
        y_m = (row + 0.5) * res + self.y_min_m
        return float(x_m), float(y_m)

    def _local_to_world_xy(
        self,
        x_local: np.ndarray | float,
        y_local: np.ndarray | float,
        pose: Pose2D,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        cos_y = math.cos(pose.yaw_rad)
        sin_y = math.sin(pose.yaw_rad)
        xw = pose.x_m + x_local * cos_y + y_local * sin_y
        yw = pose.y_m - x_local * sin_y + y_local * cos_y
        return xw, yw

    def _merge_obstacles_into_global_registry(
        self,
        frame_obstacles: list[Obstacle],
        pose: Pose2D,
    ) -> list[Obstacle]:
        for obs in frame_obstacles:
            lx, ly, lz = obs.centroid_3d
            wx, wy = self._local_to_world_xy(lx, ly, pose)
            wx_f, wy_f = float(wx), float(wy)

            matched: Optional[Obstacle] = None
            best_d = self.config.obstacle_merge_radius_m

            for existing in self.global_obstacles:
                if existing.type != obs.type:
                    continue
                ex, ey = existing.world_xy
                d = math.hypot(wx_f - ex, wy_f - ey)
                if d < best_d:
                    best_d = d
                    matched = existing

            if matched is not None:
                n = matched.observations
                new_wx = (matched.world_xy[0] * n + wx_f) / (n + 1)
                new_wy = (matched.world_xy[1] * n + wy_f) / (n + 1)
                new_mag = max(matched.magnitude_m, obs.magnitude_m)
                matched.world_xy = (new_wx, new_wy)
                matched.magnitude_m = new_mag
                matched.bbox_2d = obs.bbox_2d
                matched.observations = n + 1
            else:
                new_obs = Obstacle(
                    id=self._next_global_id,
                    type=obs.type,
                    distance_m=obs.distance_m,
                    bearing_deg=obs.bearing_deg,
                    magnitude_m=obs.magnitude_m,
                    bbox_2d=obs.bbox_2d,
                    centroid_3d=(wx_f, wy_f, lz),
                    pixel_count=obs.pixel_count,
                    world_xy=(wx_f, wy_f),
                    observations=1,
                )
                self._next_global_id += 1
                self.global_obstacles.append(new_obs)

        # Actualizar distancia y bearing relativos a la posicion actual del Rover/Dron
        cos_y = math.cos(pose.yaw_rad)
        sin_y = math.sin(pose.yaw_rad)
        active_list: list[Obstacle] = []

        for g_obs in self.global_obstacles:
            dx_w = g_obs.world_xy[0] - pose.x_m
            dy_w = g_obs.world_xy[1] - pose.y_m
            # Rotacion inversa al marco local actual
            lx_rel = dx_w * cos_y - dy_w * sin_y
            ly_rel = dx_w * sin_y + dy_w * cos_y
            dist_rel = math.hypot(lx_rel, ly_rel)
            bearing_rel = math.degrees(math.atan2(lx_rel, max(ly_rel, 1e-3)))

            g_obs.distance_m = dist_rel
            g_obs.bearing_deg = bearing_rel
            active_list.append(g_obs)

        active_list.sort(key=lambda o: o.distance_m)
        return active_list

    def build(
        self,
        terrain: TerrainAnalysis,
        obstacles: list[Obstacle],
        boulder_mask: np.ndarray,
        crevice_mask: np.ndarray,
        pose: Optional[Pose2D] = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        current_pose = pose or Pose2D()

        self._frame_counter += 1
        if not self.config.persistent_mapping:
            self.reset()

        self._ensure_capacity_for_pose(current_pose)
        rows, cols = self._log_odds.shape
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0

        cur_r, cur_c = self.world_to_cell(current_pose.x_m, current_pose.y_m)
        if not self.trajectory_cells or self.trajectory_cells[-1] != (cur_r, cur_c):
            self.trajectory_cells.append((cur_r, cur_c))

        s = max(1, int(self.config.pointcloud_stride))
        pts = terrain.points_3d[::s, ::s]
        valid = terrain.valid_mask[::s, ::s]
        h_sub = terrain.height_map_m[::s, ::s]
        b_sub = boulder_mask[::s, ::s]
        c_sub = crevice_mask[::s, ::s]

        X_local = pts[:, :, 0]
        Y_local = pts[:, :, 1]
        horiz_r = np.hypot(X_local, Y_local)
        valid_ratio = float(np.mean(valid))
        if valid_ratio < 0.25:
            max_footprint = max(0.40, float(terrain.altitude_m) * 1.6)
        else:
            max_footprint = max(2.2, float(terrain.altitude_m) * 2.8)
        X_world, Y_world = self._local_to_world_xy(X_local, Y_local, current_pose)

        y_min = self.y_min_m
        in_bounds = (
            valid
            & (np.abs(h_sub) <= self.config.max_anomaly_magnitude_m)
            & (horiz_r <= max_footprint)
            & (X_world >= -half_w)
            & (X_world < half_w)
            & (Y_world >= y_min)
            & (Y_world < y_min + rows * res)
        )

        # Suelo libre observado en este frame -> reduce log_odds (confirma libre)
        free_pts = in_bounds & (~b_sub) & (~c_sub)
        if np.any(free_pts):
            fr = np.clip(((Y_world[free_pts] - y_min) / res).astype(np.int32), 0, rows - 1)
            fc = np.clip(((X_world[free_pts] + half_w) / res).astype(np.int32), 0, cols - 1)
            self._observed[fr, fc] = True
            self._log_odds[fr, fc] = np.clip(
                self._log_odds[fr, fc] + self.config.log_odds_free,
                self.config.log_odds_min,
                self.config.log_odds_max,
            )

        if self.config.nominal_camera_pitch_deg <= -65.0:
            r0 = max(0, cur_r - 8)
            r1 = min(rows, cur_r + 9)
            c0 = max(0, cur_c - 8)
            c1 = min(cols, cur_c + 9)
            self._observed[r0:r1, c0:c1] = True
        else:
            for local_r_step in range(min(rows - cur_r, 45)):
                r = cur_r + local_r_step
                half_span = max(3, int((local_r_step + 2) * 0.55))
                c0 = max(0, cur_c - half_span)
                c1 = min(cols, cur_c + half_span + 1)
                self._observed[r, c0:c1] = True

        # Obstaculos BOULDER observados en este frame -> incrementa log_odds
        b_in = in_bounds & b_sub
        if np.any(b_in):
            br = np.clip(((Y_world[b_in] - y_min) / res).astype(np.int32), 0, rows - 1)
            bc = np.clip(((X_world[b_in] + half_w) / res).astype(np.int32), 0, cols - 1)
            self._observed[br, bc] = True
            self._log_odds[br, bc] = np.clip(
                self._log_odds[br, bc] + self.config.log_odds_hit,
                self.config.log_odds_min,
                self.config.log_odds_max,
            )
            self._obstacle_type_layer[br, bc] = GRID_BOULDER

        # Obstaculos CREVICE observados en este frame -> incrementa log_odds
        c_in = in_bounds & c_sub
        if np.any(c_in):
            cr = np.clip(((Y_world[c_in] - y_min) / res).astype(np.int32), 0, rows - 1)
            cc = np.clip(((X_world[c_in] + half_w) / res).astype(np.int32), 0, cols - 1)
            self._observed[cr, cc] = True
            self._log_odds[cr, cc] = np.clip(
                self._log_odds[cr, cc] + self.config.log_odds_hit,
                self.config.log_odds_min,
                self.config.log_odds_max,
            )
            self._obstacle_type_layer[cr, cc] = GRID_CREVICE

        # Actualizar registro persistente de obstaculos globales y estamparlos en la malla
        merged_obstacles = self._merge_obstacles_into_global_registry(obstacles, current_pose)
        if self._frame_counter > 3 and len(self.global_obstacles) > 12:
            self.global_obstacles = [
                o for o in self.global_obstacles if o.observations >= 2
            ][:20]
            merged_obstacles = self.global_obstacles

        min_obs_confirm = 1 if self._frame_counter <= 2 else 2
        for g_obs in merged_obstacles:
            if g_obs.observations < min_obs_confirm:
                continue
            ox, oy = g_obs.world_xy
            if -half_w <= ox < half_w and y_min <= oy < y_min + rows * res:
                orow, ocol = self.world_to_cell(ox, oy)
                code = GRID_BOULDER if g_obs.type == "BOULDER" else GRID_CREVICE
                cv2.circle(self._obstacle_type_layer, (ocol, orow), 2, int(code), thickness=-1)
                cv2.circle(
                    self._log_odds,
                    (ocol, orow),
                    2,
                    float(self.config.log_odds_occupied_threshold + 0.6),
                    thickness=-1,
                )
                self._observed[
                    max(0, orow - 2) : min(rows, orow + 3),
                    max(0, ocol - 2) : min(cols, ocol + 3),
                ] = True

        # Sintetizar la malla de ocupacion global acumulada a partir del estado persistente
        grid = np.full((rows, cols), GRID_UNKNOWN, dtype=np.uint8)
        grid[self._observed] = GRID_FREE

        occupied_cells = self._log_odds >= self.config.log_odds_occupied_threshold
        boulder_cells = occupied_cells & (self._obstacle_type_layer == GRID_BOULDER)
        crevice_cells = occupied_cells & (self._obstacle_type_layer == GRID_CREVICE)
        grid[boulder_cells] = GRID_BOULDER
        grid[crevice_cells] = GRID_CREVICE

        hard_obstacle_mask = (
            (grid == GRID_BOULDER)
            | (grid == GRID_CREVICE)
            | (grid == GRID_STEEP)
        )
        inf_cells = max(1, int(math.ceil(self.config.robot_radius_m / res)))
        k_size = 2 * inf_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        inflated_u8 = cv2.dilate(hard_obstacle_mask.astype(np.uint8) * 255, kernel)
        inflated_zone = (inflated_u8 > 0) & (~hard_obstacle_mask)
        grid[inflated_zone] = GRID_INFLATED

        free_for_dt = (~hard_obstacle_mask).astype(np.uint8)
        dist_cells = cv2.distanceTransform(free_for_dt, cv2.DIST_L2, 3)
        dist_m = dist_cells * res

        cost_grid = (
            1.0
            + self.config.proximity_penalty_weight * np.exp(-dist_m / 0.25)
        ).astype(np.float32)
        cost_grid[grid == GRID_INFLATED] += 8.0
        cost_grid[grid == GRID_UNKNOWN] += 2.5
        cost_grid[hard_obstacle_mask] = np.inf

        # Evaluar safe_corridor en el cono delante de la posicion actual (cur_r, cur_c)
        c_span = max(1, int(math.ceil(self.config.corridor_half_width_m / res)))
        r_check = max(2, int(math.ceil(self.config.corridor_check_distance_m / res)))
        c_min = max(0, cur_c - c_span)
        c_max = min(cols, cur_c + c_span + 1)
        r_min = min(rows - 1, cur_r + 3)
        r_max = min(rows, cur_r + r_check)

        corridor_slice = grid[r_min:r_max, c_min:c_max]
        corridor_blocked = bool(
            np.any(
                (corridor_slice == GRID_BOULDER)
                | (corridor_slice == GRID_CREVICE)
                | (corridor_slice == GRID_STEEP)
            )
        )
        incline_ok = terrain.max_incline_deg <= self.config.max_safe_incline_deg
        safe_corridor = (not corridor_blocked) and incline_ok

        return grid, cost_grid, safe_corridor
