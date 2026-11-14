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


class OccupancyGridBuilder:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    @property
    def grid_shape(self) -> tuple[int, int]:
        rows = max(10, int(round(self.config.grid_depth_m / self.config.grid_resolution_m)))
        cols = max(10, int(round(self.config.grid_width_m / self.config.grid_resolution_m)))
        return rows, cols

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        rows, cols = self.grid_shape
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0

        row = int(np.clip(math.floor(y_m / res), 0, rows - 1))
        col = int(np.clip(math.floor((x_m + half_w) / res), 0, cols - 1))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0
        x_m = (col + 0.5) * res - half_w
        y_m = (row + 0.5) * res
        return float(x_m), float(y_m)

    def build(
        self,
        terrain: TerrainAnalysis,
        obstacles: list[Obstacle],
        boulder_mask: np.ndarray,
        crevice_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        rows, cols = self.grid_shape
        res = self.config.grid_resolution_m
        half_w = self.config.grid_width_m / 2.0

        grid = np.full((rows, cols), GRID_UNKNOWN, dtype=np.uint8)

        s = max(1, int(self.config.pointcloud_stride))
        pts = terrain.points_3d[::s, ::s]
        valid = terrain.valid_mask[::s, ::s]
        b_sub = boulder_mask[::s, ::s]
        c_sub = crevice_mask[::s, ::s]

        X = pts[:, :, 0]
        Y = pts[:, :, 1]

        in_bounds = (
            valid
            & (X >= -half_w)
            & (X < half_w)
            & (Y >= 0.0)
            & (Y < self.config.grid_depth_m)
        )

        r_idx = np.clip((Y[in_bounds] / res).astype(np.int32), 0, rows - 1)
        c_idx = np.clip(((X[in_bounds] + half_w) / res).astype(np.int32), 0, cols - 1)

        grid[r_idx, c_idx] = GRID_FREE

        start_c = cols // 2
        for r in range(rows):
            half_span = max(3, int((r + 2) * 0.65))
            c0 = max(0, start_c - half_span)
            c1 = min(cols, start_c + half_span + 1)
            unknown_slice = grid[r, c0:c1] == GRID_UNKNOWN
            grid[r, c0:c1][unknown_slice] = GRID_FREE

        b_in = in_bounds & b_sub
        if np.any(b_in):
            br = np.clip((Y[b_in] / res).astype(np.int32), 0, rows - 1)
            bc = np.clip(((X[b_in] + half_w) / res).astype(np.int32), 0, cols - 1)
            grid[br, bc] = GRID_BOULDER

        c_in = in_bounds & c_sub
        if np.any(c_in):
            cr = np.clip((Y[c_in] / res).astype(np.int32), 0, rows - 1)
            cc = np.clip(((X[c_in] + half_w) / res).astype(np.int32), 0, cols - 1)
            grid[cr, cc] = GRID_CREVICE

        for obs in obstacles:
            ox, oy, _ = obs.centroid_3d
            if -half_w <= ox < half_w and 0.0 <= oy < self.config.grid_depth_m:
                orow, ocol = self.world_to_cell(ox, oy)
                code = GRID_BOULDER if obs.type == "BOULDER" else GRID_CREVICE
                cv2.circle(grid, (ocol, orow), 2, int(code), thickness=-1)

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
        cost_grid[hard_obstacle_mask] = np.inf

        c_min = max(0, int(math.floor((half_w - self.config.corridor_half_width_m) / res)))
        c_max = min(cols, int(math.ceil((half_w + self.config.corridor_half_width_m) / res)))
        r_min = min(rows - 1, max(1, int(math.floor(0.25 / res))))
        r_max = min(rows, int(math.ceil(self.config.corridor_check_distance_m / res)))

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
