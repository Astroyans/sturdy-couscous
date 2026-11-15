from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.models import GRID_UNKNOWN, PipelineOutput

QUADRANT_WIDTH_M = 1.5
QUADRANT_HEIGHT_M = 1.0


@dataclass(frozen=True)
class QuadrantBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center_xy_m(self) -> tuple[float, float]:
        return (
            round(0.5 * (self.x_min + self.x_max), 2),
            round(0.5 * (self.y_min + self.y_max), 2),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": round(float(self.x_min), 2),
            "x_max": round(float(self.x_max), 2),
            "y_min": round(float(self.y_min), 2),
            "y_max": round(float(self.y_max), 2),
        }

    def overlaps_point(self, x_m: float, y_m: float) -> bool:
        return self.x_min <= x_m < self.x_max and self.y_min <= y_m < self.y_max


@dataclass
class TrajectoryPrediction:
    timestamp_ms: int
    current_bounds: QuadrantBounds
    next_bounds: QuadrantBounds
    predicted_pose_3s: tuple[float, float]
    remaining_quadrants: list[dict[str, float]] = field(default_factory=list)
    heading_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "current_quadrant": self.current_bounds.to_dict(),
            "next_quadrant": self.next_bounds.to_dict(),
            "predicted_pose_3s": [
                round(float(self.predicted_pose_3s[0]), 2),
                round(float(self.predicted_pose_3s[1]), 2),
            ],
            "heading_deg": round(float(self.heading_deg), 1),
            "remaining_quadrants": list(self.remaining_quadrants),
        }


def snap_point_to_quadrant_bounds(
    x_m: float,
    y_m: float,
    q_width: float = QUADRANT_WIDTH_M,
    q_height: float = QUADRANT_HEIGHT_M,
) -> QuadrantBounds:
    qx = math.floor(x_m / q_width)
    qy = math.floor(y_m / q_height)
    x_min = qx * q_width
    y_min = qy * q_height
    return QuadrantBounds(
        x_min=x_min,
        x_max=x_min + q_width,
        y_min=y_min,
        y_max=y_min + q_height,
    )


def predict_trajectory(
    output: PipelineOutput,
    lookahead_s: float = 3.0,
    nominal_speed_mps: float = 0.35,
) -> TrajectoryPrediction:
    rx, ry, _ryaw = output.rover_pose_xy_yaw
    heading_deg = float(output.navigation.recommended_heading_deg)
    heading_rad = math.radians(heading_deg)

    current_bounds = snap_point_to_quadrant_bounds(rx, ry)

    pred_x = rx + math.sin(heading_rad) * nominal_speed_mps * lookahead_s
    pred_y = ry + math.cos(heading_rad) * nominal_speed_mps * lookahead_s

    astar_wps = output.navigation.astar_path_world_xy
    remaining: list[dict[str, float]] = []
    seen_keys: set[tuple[float, float]] = {
        (round(current_bounds.x_min, 2), round(current_bounds.y_min, 2))
    }

    next_bounds: Optional[QuadrantBounds] = None
    for wx, wy in astar_wps:
        qb = snap_point_to_quadrant_bounds(wx, wy)
        key = (round(qb.x_min, 2), round(qb.y_min, 2))
        if key not in seen_keys:
            seen_keys.add(key)
            if next_bounds is None:
                next_bounds = qb
            remaining.append(qb.to_dict())

    if next_bounds is None:
        step_x = rx + math.sin(heading_rad) * QUADRANT_WIDTH_M
        step_y = ry + math.cos(heading_rad) * QUADRANT_HEIGHT_M
        next_bounds = snap_point_to_quadrant_bounds(step_x, step_y)
        if (round(next_bounds.x_min, 2), round(next_bounds.y_min, 2)) == (
            round(current_bounds.x_min, 2),
            round(current_bounds.y_min, 2),
        ):
            next_bounds = QuadrantBounds(
                x_min=current_bounds.x_min,
                x_max=current_bounds.x_max,
                y_min=current_bounds.y_min + QUADRANT_HEIGHT_M,
                y_max=current_bounds.y_max + QUADRANT_HEIGHT_M,
            )
        remaining.append(next_bounds.to_dict())

    return TrajectoryPrediction(
        timestamp_ms=int(time.time() * 1000),
        current_bounds=current_bounds,
        next_bounds=next_bounds,
        predicted_pose_3s=(pred_x, pred_y),
        remaining_quadrants=remaining[:6],
        heading_deg=heading_deg,
    )


def extract_quadrant_patch(
    occupancy_grid: np.ndarray,
    cost_grid: np.ndarray,
    bounds: QuadrantBounds,
    resolution_m: float,
    y_min_m: float,
    grid_width_m: float,
) -> Optional[dict[str, Any]]:
    rows, cols = occupancy_grid.shape
    half_w = grid_width_m / 2.0
    c0 = max(0, int(math.floor((bounds.x_min + half_w) / resolution_m)))
    c1 = min(cols, int(math.ceil((bounds.x_max + half_w) / resolution_m)))
    r0 = max(0, int(math.floor((bounds.y_min - y_min_m) / resolution_m)))
    r1 = min(rows, int(math.ceil((bounds.y_max - y_min_m) / resolution_m)))
    if r1 <= r0 or c1 <= c0:
        return None

    sub_occ = occupancy_grid[r0:r1, c0:c1]
    sub_cost = cost_grid[r0:r1, c0:c1]
    known_ratio = float(np.count_nonzero(sub_occ != GRID_UNKNOWN)) / float(sub_occ.size)
    if known_ratio < 0.15:
        return None

    clean_cost = np.where(np.isfinite(sub_cost), np.round(sub_cost, 1), 999.0)
    return {
        "id": int(time.time() * 1000),
        "bounds": bounds.to_dict(),
        "explored_ratio": round(known_ratio, 3),
        "resolution_m": round(float(resolution_m), 3),
        "occupancy_subgrid": sub_occ.astype(int).tolist(),
        "cost_subgrid": clean_cost.tolist(),
    }


def format_esp32_ruta_txt(
    mode: str,
    planner: str,
    safe_corridor: bool,
    altitude_m: float,
    heading_deg: float,
    timestamp_ms: int,
    waypoints_xy: list[tuple[float, float]],
    scan_alert_id: Optional[int] = None,
) -> str:
    if len(waypoints_xy) > 6:
        idxs = np.linspace(0, len(waypoints_xy) - 1, 6).astype(int)
        sampled_wp = [waypoints_xy[i] for i in idxs]
    else:
        sampled_wp = list(waypoints_xy)
    wp_str = (
        "->".join(f"({wx:+.2f},{wy:+.2f})" for wx, wy in sampled_wp)
        if sampled_wp
        else "IDLE"
    )
    scan_str = str(scan_alert_id) if scan_alert_id is not None else "0"
    line = (
        f"RUTA_V1|ts={int(timestamp_ms)}|mode={mode}|plan={planner}"
        f"|safe={1 if safe_corridor else 0}|alt={altitude_m:.2f}"
        f"|hdg={heading_deg:+.1f}|scan={scan_str}|wp={wp_str}\n"
    )
    if len(line) > 500:
        line = line[:499] + "\n"
    return line
