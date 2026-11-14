"""Segmentación morfológica y clustering espacial 3D con DBSCAN para `BOULDER` y `CREVICE`."""

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
    """Ejecuta la Fase 3: detección de rocas (`BOULDER`) y grietas (`CREVICE`) en 3D."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def _extract_clusters_from_mask(
        self,
        clean_mask: np.ndarray,
        terrain: TerrainAnalysis,
        obs_type: ObstacleType,
    ) -> list[dict]:
        """Agrupa píxeles anómalos mediante componentes conexos 2D + refinamiento 3D DBSCAN."""
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
                # Submuestrear si el componente tiene demasiados píxeles para mantener <5 ms
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

    def detect(
        self,
        terrain: TerrainAnalysis,
    ) -> tuple[list[Obstacle], np.ndarray, np.ndarray]:
        """Detecta y clasifica todos los obstáculos `BOULDER` y `CREVICE` del frame actual.

        Returns:
            obstacles: Lista ordenada por cercanía (`distance_m`).
            boulder_mask: Máscara booleana 2D de píxeles clasificados como `BOULDER`.
            crevice_mask: Máscara booleana 2D de píxeles clasificados como `CREVICE`.
        """
        raw_boulder = (
            terrain.valid_mask
            & (terrain.height_map_m >= self.config.boulder_min_height_m)
            & (terrain.height_map_m <= self.config.max_anomaly_magnitude_m)
        )
        raw_crevice = (
            terrain.valid_mask
            & (terrain.height_map_m <= -self.config.crevice_min_depth_m)
            & (terrain.height_map_m >= -self.config.max_anomaly_magnitude_m)
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

            centroid = np.median(pts, axis=0)
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])

            distance_m = math.sqrt(cx * cx + cy * cy)
            if distance_m < 0.10 or distance_m > self.config.max_depth_m:
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

        # Ordenar de más cercano a más lejano y asignar IDs 1..N
        unsorted_obstacles.sort(key=lambda o: o.distance_m)
        obstacles: list[Obstacle] = []
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

        return obstacles, boulder_mask, crevice_mask
