from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from src.config import PipelineConfig
from src.models import Obstacle, ObstacleType, TerrainAnalysis
from src.preprocessing.filters import clean_binary_mask


class ObstacleDetector:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def _extract_clusters_from_mask(
        self,
        clean_mask: np.ndarray,
        terrain: TerrainAnalysis,
        obs_type: ObstacleType,
    ) -> list[dict]:
        if not np.any(clean_mask):
            return []

        mask_u8 = clean_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )

        raw_clusters: list[dict] = []

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < self.config.min_cluster_pixels:
                continue

            x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
            y0 = int(stats[lbl, cv2.CC_STAT_TOP])
            w0 = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h0 = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            roi_labels = labels[y0 : y0 + h0, x0 : x0 + w0]
            comp_ys_local, comp_xs_local = np.nonzero(roi_labels == lbl)
            comp_ys = comp_ys_local + y0
            comp_xs = comp_xs_local + x0

            pts_3d = terrain.points_3d[comp_ys, comp_xs]
            heights = terrain.height_map_m[comp_ys, comp_xs]

            if self.config.use_dbscan and pts_3d.shape[0] >= self.config.dbscan_min_samples * 2:
                n_pts = pts_3d.shape[0]
                max_pts = max(200, int(self.config.dbscan_max_points))
                if n_pts > max_pts:
                    step = max(1, n_pts // max_pts)
                    sub_idx = np.arange(0, n_pts, step)
                else:
                    sub_idx = np.arange(n_pts)

                sub_pts = pts_3d[sub_idx]
                clustering = DBSCAN(
                    eps=self.config.dbscan_eps_m,
                    min_samples=self.config.dbscan_min_samples,
                ).fit(sub_pts)
                db_labels = clustering.labels_
                unique_db = [c for c in np.unique(db_labels) if c >= 0]

                if len(unique_db) >= 1:
                    for db_c in unique_db:
                        c_mask = db_labels == db_c
                        if int(np.sum(c_mask)) < self.config.dbscan_min_samples:
                            continue
                        sel_idx = sub_idx[c_mask]
                        c_xs = comp_xs[sel_idx]
                        c_ys = comp_ys[sel_idx]
                        c_pts = pts_3d[sel_idx]
                        c_h = heights[sel_idx]

                        bx0, bx1 = int(np.min(c_xs)), int(np.max(c_xs))
                        by0, by1 = int(np.min(c_ys)), int(np.max(c_ys))
                        raw_clusters.append(
                            {
                                "type": obs_type,
                                "pts_3d": c_pts,
                                "heights": c_h,
                                "bbox_2d": (bx0, by0, max(1, bx1 - bx0 + 1), max(1, by1 - by0 + 1)),
                                "pixel_count": int(len(sel_idx)),
                            }
                        )
                    continue

            raw_clusters.append(
                {
                    "type": obs_type,
                    "pts_3d": pts_3d,
                    "heights": heights,
                    "bbox_2d": (x0, y0, w0, h0),
                    "pixel_count": area,
                }
            )

        return raw_clusters

    @staticmethod
    def _merge_nearby_obstacles(
        obstacles: list[Obstacle],
        px_gap: int = 28,
        dist_3d_max: float = 0.45,
    ) -> list[Obstacle]:
        merged: list[Obstacle] = []
        for obs in obstacles:
            ox, oy, ow, oh = obs.bbox_2d
            ox1, oy1 = ox + ow, oy + oh
            found = False
            for idx, m in enumerate(merged):
                if m.type != obs.type:
                    continue
                mx, my, mw, mh = m.bbox_2d
                mx1, my1 = mx + mw, my + mh
                overlap_x = (ox - px_gap <= mx1) and (mx - px_gap <= ox1)
                overlap_y = (oy - px_gap <= my1) and (my - px_gap <= oy1)
                d3 = math.hypot(
                    obs.centroid_3d[0] - m.centroid_3d[0],
                    obs.centroid_3d[1] - m.centroid_3d[1],
                )
                if overlap_x and overlap_y and d3 <= dist_3d_max:
                    nx0, ny0 = min(ox, mx), min(oy, my)
                    nx1, ny1 = max(ox1, mx1), max(oy1, my1)
                    tot_px = m.pixel_count + obs.pixel_count
                    w_m = m.pixel_count / max(tot_px, 1)
                    w_o = obs.pixel_count / max(tot_px, 1)
                    cx = m.centroid_3d[0] * w_m + obs.centroid_3d[0] * w_o
                    cy = m.centroid_3d[1] * w_m + obs.centroid_3d[1] * w_o
                    cz = m.centroid_3d[2] * w_m + obs.centroid_3d[2] * w_o
                    dist_m = math.hypot(cx, cy)
                    bearing_deg = math.degrees(math.atan2(cx, max(cy, 1e-4)))
                    merged[idx] = Obstacle(
                        id=0,
                        type=m.type,
                        distance_m=dist_m,
                        bearing_deg=bearing_deg,
                        magnitude_m=max(m.magnitude_m, obs.magnitude_m),
                        bbox_2d=(nx0, ny0, nx1 - nx0, ny1 - ny0),
                        centroid_3d=(cx, cy, cz),
                        pixel_count=tot_px,
                    )
                    found = True
                    break
            if not found:
                merged.append(obs)
        return merged

    def detect(
        self,
        terrain: TerrainAnalysis,
    ) -> tuple[list[Obstacle], np.ndarray, np.ndarray]:
        h, w = terrain.valid_mask.shape
        interior = np.ones((h, w), dtype=bool)
        margin = 16 if (h >= 120 and w >= 160) else 4
        interior[:margin, :] = False
        interior[-margin:, :] = False
        interior[:, :margin] = False
        interior[:, -margin:] = False

        eroded_valid = (
            cv2.erode(
                terrain.valid_mask.astype(np.uint8),
                np.ones((7, 7), dtype=np.uint8),
            )
            > 0
        )
        num_v, v_labels, v_stats, _ = cv2.connectedComponentsWithStats(
            eroded_valid.astype(np.uint8), connectivity=8
        )
        min_valid_island = max(150, int(h * w * 0.012))
        contiguous_valid = np.zeros((h, w), dtype=bool)
        for v_lbl in range(1, num_v):
            if int(v_stats[v_lbl, cv2.CC_STAT_AREA]) >= min_valid_island:
                contiguous_valid |= v_labels == v_lbl

        depth_f32 = terrain.height_map_m.astype(np.float32)
        k3 = np.ones((3, 3), dtype=np.uint8)
        local_jump_m = cv2.dilate(depth_f32, k3) - cv2.erode(depth_f32, k3)
        smooth_depth = local_jump_m <= 0.18

        base_valid = terrain.valid_mask & interior
        valid_ratio = float(np.mean(terrain.valid_mask))
        max_crevice_depth = min(
            self.config.max_anomaly_magnitude_m,
            max(0.35, float(terrain.altitude_m) * 0.85),
        )

        max_footprint = max(2.2, float(terrain.altitude_m) * 2.8)

        raw_boulder = (
            base_valid
            & contiguous_valid
            & smooth_depth
            & (valid_ratio >= 0.25)
            & (terrain.height_map_m >= self.config.boulder_min_height_m)
            & (terrain.height_map_m <= self.config.max_anomaly_magnitude_m)
        )
        raw_crevice = (
            base_valid
            & contiguous_valid
            & smooth_depth
            & (valid_ratio >= 0.25)
            & (terrain.height_map_m <= -self.config.crevice_min_depth_m)
            & (terrain.height_map_m >= -max_crevice_depth)
        )

        boulder_mask = clean_binary_mask(
            raw_boulder,
            kernel_size=self.config.morph_kernel_size,
            open_iterations=self.config.morph_open_iterations,
            close_iterations=self.config.morph_close_iterations,
        )
        crevice_mask = clean_binary_mask(
            raw_crevice,
            kernel_size=self.config.morph_kernel_size,
            open_iterations=self.config.morph_open_iterations,
            close_iterations=self.config.morph_close_iterations,
        )

        candidates = self._extract_clusters_from_mask(
            boulder_mask, terrain, "BOULDER"
        ) + self._extract_clusters_from_mask(crevice_mask, terrain, "CREVICE")

        unsorted_obstacles: list[Obstacle] = []
        for cand in candidates:
            pts = cand["pts_3d"]
            heights = cand["heights"]
            obs_type: ObstacleType = cand["type"]

            if pts.shape[0] < 3:
                continue
            extent_xy = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
            if extent_xy < self.config.min_obstacle_extent_m:
                continue

            q75, q25 = np.percentile(heights, [75, 25])
            if float(q75 - q25) > 0.35:
                continue

            centroid = np.median(pts, axis=0)
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])

            distance_m = math.sqrt(cx * cx + cy * cy)
            if distance_m < 0.10 or distance_m > min(self.config.max_depth_m, max_footprint):
                continue

            bearing_deg = math.degrees(math.atan2(cx, max(cy, 1e-4)))

            if obs_type == "BOULDER":
                magnitude_m = float(np.percentile(heights, 95))
            else:
                magnitude_m = float(np.percentile(np.abs(heights), 95))

            unsorted_obstacles.append(
                Obstacle(
                    id=0,
                    type=obs_type,
                    distance_m=distance_m,
                    bearing_deg=bearing_deg,
                    magnitude_m=magnitude_m,
                    bbox_2d=cand["bbox_2d"],
                    centroid_3d=(cx, cy, cz),
                    pixel_count=cand["pixel_count"],
                )
            )

        unsorted_obstacles = self._merge_nearby_obstacles(unsorted_obstacles)
        unsorted_obstacles.sort(key=lambda o: o.distance_m)
        obstacles: list[Obstacle] = []
        final_boulder_mask = np.zeros_like(boulder_mask, dtype=bool)
        final_crevice_mask = np.zeros_like(crevice_mask, dtype=bool)

        for idx, obs in enumerate(unsorted_obstacles, start=1):
            obstacles.append(
                Obstacle(
                    id=idx,
                    type=obs.type,
                    distance_m=obs.distance_m,
                    bearing_deg=obs.bearing_deg,
                    magnitude_m=obs.magnitude_m,
                    bbox_2d=obs.bbox_2d,
                    centroid_3d=obs.centroid_3d,
                    pixel_count=obs.pixel_count,
                )
            )
            bx, by, bw, bh = obs.bbox_2d
            x0 = max(0, bx)
            y0 = max(0, by)
            x1 = min(w, bx + bw)
            y1 = min(h, by + bh)
            if obs.type == "BOULDER":
                final_boulder_mask[y0:y1, x0:x1] |= boulder_mask[y0:y1, x0:x1]
            else:
                final_crevice_mask[y0:y1, x0:x1] |= crevice_mask[y0:y1, x0:x1]

        return obstacles, final_boulder_mask, final_crevice_mask
