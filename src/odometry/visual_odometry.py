from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig


@dataclass
class Pose2D:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    total_distance_m: float = 0.0

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)


class RGBDVisualOdometry:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.pose = Pose2D()
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts_3d: Optional[np.ndarray] = None
        self._prev_valid: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.pose = Pose2D()
        self._prev_gray = None
        self._prev_pts_3d = None
        self._prev_valid = None

    def apply_manual_step(
        self,
        dx_local_m: float = 0.0,
        dy_local_m: float = 0.0,
        dyaw_deg: float = 0.0,
    ) -> Pose2D:
        self.pose.yaw_rad += math.radians(dyaw_deg)
        cos_y = math.cos(self.pose.yaw_rad)
        sin_y = math.sin(self.pose.yaw_rad)

        dx_w = dx_local_m * cos_y + dy_local_m * sin_y
        dy_w = -dx_local_m * sin_y + dy_local_m * cos_y

        step_dist = math.hypot(dx_w, dy_w)
        self.pose.x_m += dx_w
        self.pose.y_m += dy_w
        self.pose.total_distance_m += step_dist
        return self.pose

    def update(
        self,
        color_bgr: np.ndarray,
        points_3d: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Pose2D:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

        if (
            self._prev_gray is None
            or self._prev_pts_3d is None
            or self._prev_valid is None
            or self._prev_gray.shape != gray.shape
        ):
            self._prev_gray = gray
            self._prev_pts_3d = points_3d.copy()
            self._prev_valid = valid_mask.copy()
            return self.pose

        p0 = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=180,
            qualityLevel=0.02,
            minDistance=10,
            blockSize=7,
        )

        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                p0,
                None,
                winSize=(15, 15),
                maxLevel=2,
            )
            if p1 is not None and st is not None:
                good_prev = p0[st.ravel() == 1].reshape(-1, 2)
                good_curr = p1[st.ravel() == 1].reshape(-1, 2)

                h, w = gray.shape
                u0 = np.clip(np.round(good_prev[:, 0]).astype(np.int32), 0, w - 1)
                v0 = np.clip(np.round(good_prev[:, 1]).astype(np.int32), 0, h - 1)
                u1 = np.clip(np.round(good_curr[:, 0]).astype(np.int32), 0, w - 1)
                v1 = np.clip(np.round(good_curr[:, 1]).astype(np.int32), 0, h - 1)

                both_valid = self._prev_valid[v0, u0] & valid_mask[v1, u1]
                if int(np.sum(both_valid)) >= 10:
                    pts0 = self._prev_pts_3d[v0[both_valid], u0[both_valid], :2]
                    pts1 = points_3d[v1[both_valid], u1[both_valid], :2]

                    # Si la camara avanza +dY, los objetos estaticos parecen moverse -dY en el marco camara
                    deltas = pts0 - pts1
                    norms = np.linalg.norm(deltas, axis=1)
                    inlier_flow = norms < 0.35

                    if int(np.sum(inlier_flow)) >= 8:
                        med_delta = np.median(deltas[inlier_flow], axis=0)
                        dx_local = float(med_delta[0])
                        dy_local = float(med_delta[1])
                        step_norm = math.hypot(dx_local, dy_local)

                        # Zona muerta anti-vibracion cuando la camara esta quieta en la mesa
                        if step_norm >= self.config.vo_deadband_m:
                            self.apply_manual_step(
                                dx_local_m=dx_local,
                                dy_local_m=dy_local,
                                dyaw_deg=0.0,
                            )

        self._prev_gray = gray
        self._prev_pts_3d = points_3d.copy()
        self._prev_valid = valid_mask.copy()
        return self.pose
