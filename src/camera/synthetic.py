"""Simulador matemático 3D de terreno marciano para pruebas deterministas y modo demo."""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import CameraIntrinsics, FrameBundle


class SyntheticTerrainCamera:
    """Genera frames RGB-D sintéticos proyectando un plano con rocas (`BOULDER`) y grietas (`CREVICE`)."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        terrain_incline_deg: float = 6.0,
        add_noise: bool = False,
        animate: bool = False,
    ) -> None:
        self.config = config or PipelineConfig()
        self.terrain_incline_deg = terrain_incline_deg
        self.add_noise = add_noise
        self.animate = animate
        self.intrinsics = CameraIntrinsics.default_d415(
            self.config.width, self.config.height
        )
        self.device_name = "Synthetic Martian Terrain D415"
        self._frame_idx = 0

    def start(self) -> None:
        self._frame_idx = 0

    def stop(self) -> None:
        pass

    def read_frame(self) -> FrameBundle:
        """Sintetiza un mapa de profundidad `depth_m` e imagen `color_bgr` físicamente consistentes."""
        w, h = self.intrinsics.width, self.intrinsics.height
        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        # Rayos unitarios en el marco óptico de la cámara:
        # x_cam (derecha), y_cam (abajo), z_cam (adelante)
        rx = (u - self.intrinsics.ppx) / self.intrinsics.fx
        ry = (v - self.intrinsics.ppy) / self.intrinsics.fy
        rz = np.ones_like(rx)

        # Transformación al marco nivelado del Rover (X_right, Y_forward, Z_up)
        # Cuando pitch = 0°, la cámara mira hacia +Y_forward (z_cam -> Y, -y_cam -> Z).
        # Con pitch negativo (ej. -15°), la cámara se inclina hacia el suelo (-Z).
        pitch_rad = math.radians(self.config.nominal_camera_pitch_deg)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        # Vector dirección del rayo en marco Rover (X, Y, Z)
        dx = rx
        dy = rz * cos_p - (-ry) * sin_p
        dz = rz * sin_p + (-ry) * cos_p

        # Plano del terreno con una pendiente longitudinal `terrain_incline_deg`
        # Ecuación del suelo: Z_ground(X, Y) = -cam_h + tan(incline) * (Y - 1.0)
        slope_tan = math.tan(math.radians(self.terrain_incline_deg))
        cam_h = self.config.nominal_camera_height_m

        # Intersección rayo-plano:
        # t * dz = -cam_h + slope_tan * (t * dy - 1.0)
        # t * (dz - slope_tan * dy) = -cam_h - slope_tan
        denom = dz - slope_tan * dy
        valid_ray = denom < -1e-3
        t = np.where(valid_ray, (-cam_h - slope_tan) / np.where(valid_ray, denom, -1.0), 3.8)
        t = np.clip(t, self.config.min_depth_m, self.config.max_depth_m)

        # Coordenadas 3D sobre el plano base
        X = t * dx
        Y = t * dy

        # Desplazamiento suave opcional si animate=True
        phase = 0.08 * math.sin(self._frame_idx * 0.1) if self.animate else 0.0

        # 1. Obstáculo BOULDER (Roca elevada a la izquierda frontal: X=-0.28m, Y=1.25m, altura=0.14m)
        b_x, b_y, b_r, b_h = -0.28 + phase, 1.25, 0.16, 0.14
        dist_b2 = (X - b_x) ** 2 + (Y - b_y) ** 2
        boulder_profile = np.where(
            dist_b2 < b_r**2,
            b_h * np.sqrt(np.clip(1.0 - dist_b2 / (b_r**2), 0.0, 1.0)),
            0.0,
        )

        # 2. Obstáculo CREVICE (Grieta/depresión a la derecha frontal: X=+0.30m, Y=1.45m, prof=0.13m)
        c_x, c_y, c_rx, c_ry, c_d = 0.30 - phase, 1.45, 0.20, 0.12, 0.13
        dist_c2 = ((X - c_x) / c_rx) ** 2 + ((Y - c_y) / c_ry) ** 2
        crevice_profile = np.where(
            dist_c2 < 1.0,
            -c_d * (1.0 - dist_c2),
            0.0,
        )

        delta_h = boulder_profile + crevice_profile

        # Ajustar profundidad Z_cam (`t`) según la elevación local delta_h:
        # Un punto más alto (+delta_h) está más cerca de la cámara (menor profundidad `t`)
        t_adjusted = np.where(
            valid_ray,
            (-cam_h - slope_tan + delta_h) / np.where(valid_ray, denom, -1.0),
            self.config.max_depth_m,
        )
        depth_m = np.clip(
            t_adjusted, self.config.min_depth_m, self.config.max_depth_m
        ).astype(np.float32)

        if self.add_noise:
            rng = np.random.default_rng(42 + self._frame_idx)
            noise = rng.normal(0.0, 0.003, size=depth_m.shape).astype(np.float32)
            depth_m = np.clip(depth_m + noise, self.config.min_depth_m, self.config.max_depth_m)

        # Generar textura RGB marciana (tonos ocre/rojizos + sombreado de roca y grieta)
        norm_d = np.clip((depth_m - 0.3) / 3.0, 0.0, 1.0)
        base_r = (195 - 70 * norm_d).astype(np.uint8)
        base_g = (115 - 45 * norm_d).astype(np.uint8)
        base_b = (75 - 30 * norm_d).astype(np.uint8)
        color_bgr = np.stack([base_b, base_g, base_r], axis=-1)

        # Colorear visualmente la roca y la grieta en el canal RGB
        boulder_px = boulder_profile > 0.02
        crevice_px = crevice_profile < -0.02
        color_bgr[boulder_px] = np.array([90, 130, 225], dtype=np.uint8)
        color_bgr[crevice_px] = np.array([35, 30, 50], dtype=np.uint8)
        color_bgr = cv2.GaussianBlur(color_bgr, (3, 3), 0.8)

        self._frame_idx += 1
        return FrameBundle(
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            imu_yaw=92.0,
            imu_pitch=self.config.nominal_camera_pitch_deg,
            imu_roll=0.0,
        )

    def __enter__(self) -> SyntheticTerrainCamera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
