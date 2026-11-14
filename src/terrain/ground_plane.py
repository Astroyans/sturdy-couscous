from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import CameraIntrinsics, TerrainAnalysis


class GroundPlaneAnalyzer:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._locked_normal: Optional[np.ndarray] = None
        self._locked_d: Optional[float] = None
        self._ema_altitude_m: Optional[float] = None
        self._last_valid_plane: Optional[tuple[np.ndarray, float]] = None
        self._grid_cache: dict[tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]] = {}

    def lock_plane(self, normal: np.ndarray, d: float) -> None:
        self._locked_normal = normal.astype(np.float32).copy()
        self._locked_d = float(d)

    def unlock_plane(self) -> None:
        self._locked_normal = None
        self._locked_d = None

    @property
    def is_plane_locked(self) -> bool:
        return self._locked_normal is not None and self._locked_d is not None

    def _get_ray_factors(self, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
        key = (
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )
        if key not in self._grid_cache:
            u, v = np.meshgrid(
                np.arange(intrinsics.width, dtype=np.float32),
                np.arange(intrinsics.height, dtype=np.float32),
            )
            rx = (u - intrinsics.ppx) / max(intrinsics.fx, 1e-5)
            ry = (v - intrinsics.ppy) / max(intrinsics.fy, 1e-5)
            self._grid_cache[key] = (rx, ry)
        return self._grid_cache[key]

    def deproject_to_rover_frame(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pitch_deg: Optional[float] = None,
    ) -> np.ndarray:
        rx, ry = self._get_ray_factors(intrinsics)
        h, w = depth_m.shape
        if rx.shape != (h, w):
            rx = rx[:h, :w]
            ry = ry[:h, :w]

        pitch_deg = (
            self.config.nominal_camera_pitch_deg
            if camera_pitch_deg is None
            else camera_pitch_deg
        )
        pitch_rad = math.radians(pitch_deg)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        x_cam = depth_m * rx
        y_up_cam = -depth_m * ry
        z_cam = depth_m

        X = x_cam
        Y = z_cam * cos_p - y_up_cam * sin_p
        Z = z_cam * sin_p + y_up_cam * cos_p

        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    def _fit_plane_ransac(
        self,
        points_3d: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self._locked_normal is not None and self._locked_d is not None:
            return self._locked_normal, self._locked_d

        s = max(1, int(self.config.pointcloud_stride))
        sub_pts = points_3d[::s, ::s]
        sub_mask = valid_mask[::s, ::s]

        h_sub = sub_mask.shape[0]
        ground_roi_mask = sub_mask.copy()
        ground_roi_mask[: h_sub // 5, :] = False

        pts = sub_pts[ground_roi_mask]
        if pts.shape[0] < 50:
            pts = sub_pts[sub_mask]

        if pts.shape[0] < 15:
            default_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return default_normal, float(self.config.nominal_camera_height_m)

        rng = np.random.default_rng(12345)
        if pts.shape[0] > 1200:
            idx = rng.choice(pts.shape[0], size=1200, replace=False)
            sample_pts = pts[idx]
        else:
            sample_pts = pts

        n_pts = sample_pts.shape[0]
        iters = max(20, int(self.config.ransac_iterations))
        triplets = rng.integers(0, n_pts, size=(iters, 3))

        p1 = sample_pts[triplets[:, 0]]
        p2 = sample_pts[triplets[:, 1]]
        p3 = sample_pts[triplets[:, 2]]

        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_n = norms[:, 0] > 1e-6

        normals = np.where(valid_n[:, None], normals / np.maximum(norms, 1e-6), 0.0)
        flip = normals[:, 2] < 0
        normals[flip] *= -1.0

        plausible = valid_n & (normals[:, 2] > 0.707)
        if not np.any(plausible):
            default_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            d_est = -float(np.median(sample_pts[:, 2]))
            return default_normal, d_est

        cand_normals = normals[plausible]
        cand_p1 = p1[plausible]
        cand_d = -np.sum(cand_normals * cand_p1, axis=1)

        dists = np.abs(cand_normals @ sample_pts.T + cand_d[:, None])
        inliers = dists < self.config.ransac_inlier_threshold_m
        inlier_counts = np.sum(inliers, axis=1)

        best_idx = int(np.argmax(inlier_counts))
        best_inlier_mask = inliers[best_idx]
        inlier_pts = sample_pts[best_inlier_mask]

        if inlier_pts.shape[0] >= 10:
            centroid = np.mean(inlier_pts, axis=0)
            centered = inlier_pts - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            refined_n = vh[-1, :].astype(np.float32)
            if refined_n[2] < 0:
                refined_n = -refined_n
            if refined_n[2] > 0.707:
                refined_d = -float(np.dot(refined_n, centroid))
                return refined_n, refined_d

        best_n = cand_normals[best_idx].astype(np.float32)
        best_d = float(cand_d[best_idx])
        return best_n, best_d

    def analyze(
        self,
        depth_m: np.ndarray,
        valid_mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pitch_deg: Optional[float] = None,
    ) -> TerrainAnalysis:
        points_3d = self.deproject_to_rover_frame(
            depth_m, intrinsics, camera_pitch_deg=camera_pitch_deg
        )
        normal, plane_d = self._fit_plane_ransac(points_3d, valid_mask)
        if normal[2] <= 0.707:
            if self._last_valid_plane is not None:
                normal, plane_d = self._last_valid_plane
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                plane_d = float(self.config.nominal_camera_height_m)
        else:
            self._last_valid_plane = (normal.copy(), float(plane_d))

        height_map_m = (
            points_3d[:, :, 0] * normal[0]
            + points_3d[:, :, 1] * normal[1]
            + points_3d[:, :, 2] * normal[2]
            + plane_d
        ).astype(np.float32)
        height_map_m = np.where(valid_mask, height_map_m, 0.0)

        nz_clipped = float(np.clip(normal[2], -1.0, 1.0))
        plane_tilt_deg = math.degrees(math.acos(nz_clipped))
        pitch_deg = math.degrees(math.atan2(-float(normal[1]), max(float(normal[2]), 1e-5)))
        roll_deg = math.degrees(math.atan2(float(normal[0]), max(float(normal[2]), 1e-5)))

        valid_f = valid_mask.astype(np.float32)
        h_smooth_num = cv2.GaussianBlur(
            np.where(valid_mask, height_map_m, 0.0), (7, 7), 1.5
        )
        h_smooth_den = cv2.GaussianBlur(valid_f, (7, 7), 1.5)
        h_smooth = np.where(
            valid_mask & (h_smooth_den > 0.1),
            h_smooth_num / np.maximum(h_smooth_den, 1e-4),
            0.0,
        )

        dh_dy_px, dh_dx_px = np.gradient(h_smooth)
        dx_world = np.gradient(points_3d[:, :, 0], axis=1)
        dy_world_x = np.gradient(points_3d[:, :, 1], axis=1)
        dx_world_y = np.gradient(points_3d[:, :, 0], axis=0)
        dy_world = np.gradient(points_3d[:, :, 1], axis=0)
        step_x_m = np.maximum(np.hypot(dx_world, dy_world_x), 0.005)
        step_y_m = np.maximum(np.hypot(dx_world_y, dy_world), 0.005)
        grad_mag = np.sqrt((dh_dx_px / step_x_m) ** 2 + (dh_dy_px / step_y_m) ** 2)

        dd_dy, dd_dx = np.gradient(np.where(valid_mask, depth_m, 0.0))
        jump_edge = (np.abs(dd_dx) > 0.12) | (np.abs(dd_dy) > 0.12)
        eroded_valid = cv2.erode(valid_f.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        slope_valid = eroded_valid & (~jump_edge)

        extra_slope_deg = np.degrees(np.arctan(grad_mag)) * 0.25
        slope_map_deg = np.where(
            slope_valid,
            np.clip(plane_tilt_deg + extra_slope_deg, 0.0, 85.0),
            np.where(valid_mask, plane_tilt_deg, 0.0),
        ).astype(np.float32)

        raw_altitude_m = max(0.05, abs(float(plane_d)))
        if self._ema_altitude_m is None:
            self._ema_altitude_m = raw_altitude_m
        else:
            self._ema_altitude_m = 0.75 * self._ema_altitude_m + 0.25 * raw_altitude_m
        altitude_m = float(self._ema_altitude_m)

        ground_inliers = slope_valid & (np.abs(height_map_m) <= self.config.boulder_min_height_m)
        if np.any(ground_inliers):
            max_incline_deg = float(np.percentile(slope_map_deg[ground_inliers], 90))
        else:
            max_incline_deg = float(plane_tilt_deg)

        return TerrainAnalysis(
            plane_normal=normal,
            plane_d=plane_d,
            altitude_m=altitude_m,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            max_incline_deg=max_incline_deg,
            height_map_m=height_map_m,
            slope_map_deg=slope_map_deg,
            points_3d=points_3d,
            valid_mask=valid_mask,
        )
