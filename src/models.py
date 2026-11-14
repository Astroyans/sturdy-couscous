from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ObstacleType = Literal["BOULDER", "CREVICE"]

GRID_FREE: int = 0
GRID_INFLATED: int = 1
GRID_BOULDER: int = 2
GRID_CREVICE: int = 3
GRID_STEEP: int = 4
GRID_UNKNOWN: int = 255


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001

    @classmethod
    def default_d415(cls, width: int = 640, height: int = 480) -> CameraIntrinsics:
        fx = width * 0.92
        fy = height * 1.22
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            ppx=width / 2.0,
            ppy=height / 2.0,
            depth_scale=0.001,
        )


@dataclass
class FrameBundle:
    timestamp: float
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    imu_yaw: float = 0.0
    imu_pitch: float = -15.0
    imu_roll: float = 0.0


@dataclass
class Obstacle:
    id: int
    type: ObstacleType
    distance_m: float
    bearing_deg: float
    magnitude_m: float
    bbox_2d: tuple[int, int, int, int]
    centroid_3d: tuple[float, float, float]
    pixel_count: int = 0
    world_xy: tuple[float, float] = (0.0, 0.0)
    observations: int = 1

    def to_contract_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": int(self.id),
            "type": self.type,
            "distance_m": round(float(self.distance_m), 2),
            "bearing_deg": round(float(self.bearing_deg), 1),
        }
        if self.type == "BOULDER":
            base["height_m"] = round(float(self.magnitude_m), 2)
        else:
            base["depth_m"] = round(float(self.magnitude_m), 2)
        return base


@dataclass
class TerrainAnalysis:
    plane_normal: np.ndarray
    plane_d: float
    altitude_m: float
    pitch_deg: float
    roll_deg: float
    max_incline_deg: float
    height_map_m: np.ndarray
    slope_map_deg: np.ndarray
    points_3d: np.ndarray
    valid_mask: np.ndarray


@dataclass
class NavigationResult:
    occupancy_grid: np.ndarray
    cost_grid: np.ndarray
    path_cells: list[tuple[int, int]] = field(default_factory=list)
    path_world_xy: list[tuple[float, float]] = field(default_factory=list)
    safe_corridor: bool = True
    recommended_heading_deg: float = 0.0
    planner_used: str = "astar"
    explored_nodes: int = 0
    start_cell: tuple[int, int] = (0, 0)
    current_cell: tuple[int, int] = (0, 0)
    goal_cell: tuple[int, int] = (0, 0)
    trajectory_cells: list[tuple[int, int]] = field(default_factory=list)
    explored_area_m2: float = 0.0
    total_Persistent_obstacles: int = 0
    navigation_mode: str = "EXPLORE_FORWARD"


@dataclass
class PipelineOutput:
    frame: FrameBundle
    preprocessed_depth_m: np.ndarray
    terrain: TerrainAnalysis
    obstacles: list[Obstacle]
    navigation: NavigationResult
    edge_fps: float
    rover_pose_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    traversed_distance_m: float = 0.0

    def to_drone_cv_payload(self, node_name: str = "drone_cv") -> dict[str, Any]:
        ts = int(self.frame.timestamp) if self.frame.timestamp > 0 else int(time.time())
        return {
            "node": node_name,
            "timestamp": ts,
            "data": {
                "altitude_m": round(float(self.terrain.altitude_m), 2),
                "imu": {
                    "yaw": round(float(self.frame.imu_yaw + self.rover_pose_xy_yaw[2]), 1),
                    "pitch": round(float(self.terrain.pitch_deg), 1),
                    "roll": round(float(self.terrain.roll_deg), 1),
                },
                "max_incline_deg": round(float(self.terrain.max_incline_deg), 1),
                "safe_corridor": bool(self.navigation.safe_corridor),
                "obstacles": [obs.to_contract_dict() for obs in self.obstacles],
                "edge_fps": round(float(self.edge_fps), 1),
            },
        }
