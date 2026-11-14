"""Modelos de datos tipados y serializadores para el contrato oficial `drone_cv`."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ObstacleType = Literal["BOULDER", "CREVICE"]

# Códigos de celda para la malla de ocupación 2D
GRID_FREE: int = 0
GRID_INFLATED: int = 1
GRID_BOULDER: int = 2
GRID_CREVICE: int = 3
GRID_STEEP: int = 4
GRID_UNKNOWN: int = 255


@dataclass(frozen=True)
class CameraIntrinsics:
    """Parámetros intrínsecos de la cámara de profundidad (modelo Pinhole)."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001

    @classmethod
    def default_d415(cls, width: int = 640, height: int = 480) -> CameraIntrinsics:
        """Intrínsecos nominales de una Intel RealSense D415 (FOV ~65° x 40°)."""
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
    """Par de frames RGB-D alineados provenientes de la D415 o del simulador."""

    timestamp: float
    color_bgr: np.ndarray  # (H, W, 3) uint8
    depth_m: np.ndarray  # (H, W) float32 en metros
    intrinsics: CameraIntrinsics
    imu_yaw: float = 0.0
    imu_pitch: float = -15.0
    imu_roll: float = 0.0


@dataclass
class Obstacle:
    """Obstáculo topográfico detectado sobre o bajo el plano del suelo."""

    id: int
    type: ObstacleType
    distance_m: float
    bearing_deg: float
    magnitude_m: float  # height_m si BOULDER, depth_m si CREVICE
    bbox_2d: tuple[int, int, int, int]  # (x, y, w, h) en imagen
    centroid_3d: tuple[float, float, float]  # (X_right, Y_forward, Z_up) en metros
    pixel_count: int = 0

    def to_contract_dict(self) -> dict[str, Any]:
        """Serializa el obstáculo al esquema JSON v1.0 definido en Obsidian."""
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
    """Resultado de la Fase 2: reconstrucción 3D, plano del suelo e inclinación."""

    plane_normal: np.ndarray  # (3,) vector normal unitario [nx, ny, nz]
    plane_d: float  # término independiente del plano nx*X + ny*Y + nz*Z + d = 0
    altitude_m: float  # distancia vertical estimada de la cámara al plano del suelo
    pitch_deg: float  # cabeceo relativo estimado del plano
    roll_deg: float  # alabeo relativo estimado del plano
    max_incline_deg: float  # pendiente máxima relevante del terreno (percentil 95)
    height_map_m: np.ndarray  # (H, W) float32: elevación firmada Delta_h respecto al suelo
    slope_map_deg: np.ndarray  # (H, W) float32: pendiente local en grados
    points_3d: np.ndarray  # (H, W, 3) float32: coordenadas (X_right, Y_forward, Z_up)
    valid_mask: np.ndarray  # (H, W) bool


@dataclass
class NavigationResult:
    """Resultado de la Fase 4: malla de ocupación 2D y ruta mínima A* / Dijkstra."""

    occupancy_grid: np.ndarray  # (rows, cols) uint8
    cost_grid: np.ndarray  # (rows, cols) float32
    path_cells: list[tuple[int, int]] = field(default_factory=list)  # [(row, col), ...]
    path_world_xy: list[tuple[float, float]] = field(default_factory=list)  # [(X_m, Y_m), ...]
    safe_corridor: bool = True
    recommended_heading_deg: float = 0.0
    planner_used: str = "astar"
    explored_nodes: int = 0
    start_cell: tuple[int, int] = (0, 0)
    goal_cell: tuple[int, int] = (0, 0)


@dataclass
class PipelineOutput:
    """Salida consolidada de un ciclo completo de las 5 fases del pipeline."""

    frame: FrameBundle
    preprocessed_depth_m: np.ndarray
    terrain: TerrainAnalysis
    obstacles: list[Obstacle]
    navigation: NavigationResult
    edge_fps: float

    def to_drone_cv_payload(self, node_name: str = "drone_cv") -> dict[str, Any]:
        """Genera el diccionario exacto del contrato JSON `drone_cv` (Especificación v1.0)."""
        ts = int(self.frame.timestamp) if self.frame.timestamp > 0 else int(time.time())
        return {
            "node": node_name,
            "timestamp": ts,
            "data": {
                "altitude_m": round(float(self.terrain.altitude_m), 2),
                "imu": {
                    "yaw": round(float(self.frame.imu_yaw), 1),
                    "pitch": round(float(self.terrain.pitch_deg), 1),
                    "roll": round(float(self.terrain.roll_deg), 1),
                },
                "max_incline_deg": round(float(self.terrain.max_incline_deg), 1),
                "safe_corridor": bool(self.navigation.safe_corridor),
                "obstacles": [obs.to_contract_dict() for obs in self.obstacles],
                "edge_fps": round(float(self.edge_fps), 1),
            },
        }
