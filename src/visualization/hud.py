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
    PipelineOutput,
)


class HUDRenderer:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.canvas_w = 1280
        self.canvas_h = 720

    def _project_world_to_pixel(
        self,
        wx_m: float,
        wy_m: float,
        z_m: float,
        output: PipelineOutput,
    ) -> Optional[tuple[int, int]]:
        rx, ry, ryaw_deg = output.rover_pose_xy_yaw
        yaw_rad = math.radians(ryaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        dx_w = wx_m - rx
        dy_w = wy_m - ry
        x_local = dx_w * cos_y - dy_w * sin_y
        y_local = dx_w * sin_y + dy_w * cos_y

        pitch_rad = math.radians(output.frame.imu_pitch)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        z_cam = y_local * cos_p + z_m * sin_p
        y_up_cam = -y_local * sin_p + z_m * cos_p
        if z_cam <= 0.15:
            return None

        intr = output.frame.intrinsics
        u = int(round((x_local / z_cam) * intr.fx + intr.ppx))
        v = int(round((-y_up_cam / z_cam) * intr.fy + intr.ppy))
        if 0 <= u < intr.width and 0 <= v < intr.height:
            return (u, v)
        return None

    def _render_rgb_panel(self, output: PipelineOutput) -> np.ndarray:
        rgb = output.frame.color_bgr.copy()
        h, w = rgb.shape[:2]

        ground_z = -output.terrain.altitude_m
        projected_pts: list[tuple[int, int]] = []
        for wx, wy in output.navigation.path_world_xy:
            uv = self._project_world_to_pixel(wx, wy, ground_z, output)
            if uv is not None:
                projected_pts.append(uv)

        if len(projected_pts) >= 2:
            pts_arr = np.array(projected_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts_arr], False, (0, 255, 120), 3, cv2.LINE_AA)

        live_obstacles = (
            output.frame_obstacles
            if output.frame_obstacles is not None
            else output.obstacles
        )
        for obs in live_obstacles:
            x, y, bw, bh = obs.bbox_2d
            if obs.type == "BOULDER":
                color = (30, 110, 255)
                tag = "OBJETO"
                metric_str = f"H:{obs.magnitude_m:.2f}m"
            else:
                color = (255, 210, 0)
                tag = "DESNIVEL"
                metric_str = f"D:{obs.magnitude_m:.2f}m"

            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), color, 2)
            label = f"#{obs.id} {tag} {obs.distance_m:.2f}m {metric_str}"
            ty = max(18, y - 6)
            cv2.rectangle(
                rgb,
                (x, ty - 15),
                (min(w - 2, x + len(label) * 7 + 6), ty + 4),
                (15, 15, 20),
                -1,
            )
            cv2.putText(
                rgb,
                label,
                (x + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        panel = cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(panel, (0, 0), (480, 28), (18, 22, 30), -1)
        cv2.putText(
            panel,
            "1. RGB TACTICAL FEED + 3D OBSTACLE & PATH OVERLAY",
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 230, 245),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _render_depth_elevation_panel(
        self,
        output: PipelineOutput,
        boulder_mask: np.ndarray,
        crevice_mask: np.ndarray,
    ) -> np.ndarray:
        depth = output.preprocessed_depth_m
        valid = output.terrain.valid_mask

        norm = np.clip(
            (depth - self.config.min_depth_m)
            / max(self.config.max_depth_m - self.config.min_depth_m, 0.1),
            0.0,
            1.0,
        )
        depth_u8 = ((1.0 - norm) * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        heatmap[~valid] = (18, 18, 24)

        b_cnts, _ = cv2.findContours(
            boulder_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c_cnts, _ = cv2.findContours(
            crevice_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(heatmap, b_cnts, -1, (255, 255, 255), 2)
        cv2.drawContours(heatmap, c_cnts, -1, (255, 255, 0), 2)

        panel = cv2.resize(heatmap, (480, 360), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (480, 28), (18, 22, 30), -1)
        cv2.putText(
            panel,
            "2. PREPROCESSED DEPTH & MORPHOLOGICAL SEGMENTATION",
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 230, 245),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _render_occupancy_radar_panel(self, output: PipelineOutput) -> np.ndarray:
        grid = output.navigation.occupancy_grid
        rows, cols = grid.shape

        radar_img = np.full((rows, cols, 3), (26, 29, 36), dtype=np.uint8)
        radar_img[grid == GRID_FREE] = (44, 62, 50)
        radar_img[grid == GRID_INFLATED] = (38, 88, 118)
        radar_img[grid == GRID_BOULDER] = (35, 95, 245)
        radar_img[grid == GRID_CREVICE] = (240, 200, 25)
        radar_img[grid == GRID_STEEP] = (140, 40, 200)

        radar_flipped = np.flipud(radar_img)

        map_w, map_h = 420, 620
        scaled = cv2.resize(radar_flipped, (map_w, map_h), interpolation=cv2.INTER_NEAREST)

        sx = map_w / float(cols)
        sy = map_h / float(rows)
        res = self.config.grid_resolution_m
        max_y_m = rows * res
        y_min_m = -max_y_m / 2.0 if self.config.nominal_camera_pitch_deg <= -65.0 else 0.0

        step_m = 1.0 if max_y_m > 4.0 else 0.5
        d_m = step_m
        while d_m < max_y_m:
            r = int(round(d_m / res))
            if 0 < r < rows:
                py = int(round((rows - 1 - r) * sy))
                label_m = d_m + y_min_m
                cv2.line(scaled, (0, py), (map_w, py), (60, 75, 65), 1)
                cv2.putText(
                    scaled,
                    f"{label_m:+.1f}m" if y_min_m < 0 else f"{label_m:.1f}m",
                    (6, max(12, py - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (150, 175, 160),
                    1,
                    cv2.LINE_AA,
                )
            d_m += step_m

        cx_px = int(round((cols / 2.0) * sx))
        cv2.line(scaled, (cx_px, 0), (cx_px, map_h), (65, 80, 70), 1)

        if len(output.navigation.trajectory_cells) >= 2:
            t_pts = []
            for r, c in output.navigation.trajectory_cells:
                px = int(round((c + 0.5) * sx))
                py = int(round((rows - 1 - r + 0.5) * sy))
                t_pts.append([px, py])
            t_arr = np.array(t_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(scaled, [t_arr], False, (255, 180, 60), 2, cv2.LINE_AA)

        # Dibujar ruta Dijkstra (azul/cian) y ruta A* (verde brillante) simultaneamente
        if len(output.navigation.dijkstra_path_cells) >= 2:
            d_pts = [
                [int(round((c + 0.5) * sx)), int(round((rows - 1 - r + 0.5) * sy))]
                for r, c in output.navigation.dijkstra_path_cells
            ]
            d_arr = np.array(d_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(scaled, [d_arr], False, (255, 190, 50), 5, cv2.LINE_AA)

        if len(output.navigation.astar_path_cells) >= 2:
            a_pts = [
                [int(round((c + 0.5) * sx)), int(round((rows - 1 - r + 0.5) * sy))]
                for r, c in output.navigation.astar_path_cells
            ]
            a_arr = np.array(a_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(scaled, [a_arr], False, (40, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(scaled, tuple(a_pts[-1]), 6, (0, 255, 255), -1)

        sr, sc = output.navigation.start_cell
        base_px = int(round((sc + 0.5) * sx))
        base_py = int(round((rows - 1 - sr) * sy))
        cv2.rectangle(
            scaled,
            (base_px - 6, base_py - 6),
            (base_px + 6, base_py + 6),
            (220, 220, 220),
            -1,
        )

        cr, cc = output.navigation.current_cell
        rover_px = int(round((cc + 0.5) * sx))
        rover_py = int(round((rows - 1 - cr) * sy))
        cv2.circle(scaled, (rover_px, rover_py), 8, (255, 255, 255), -1)
        cv2.circle(scaled, (rover_px, rover_py), 5, (0, 180, 255), -1)

        panel = np.full((720, 440, 3), (18, 22, 30), dtype=np.uint8)
        panel[40 : 40 + map_h, 10 : 10 + map_w] = scaled

        cv2.putText(
            panel,
            "3. PERSISTENT GLOBAL MAP (A* & DIJKSTRA)",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (220, 230, 245),
            1,
            cv2.LINE_AA,
        )

        legend_y = 682
        items = [
            ("SUELO", (44, 62, 50)),
            ("OBJETO", (35, 95, 245)),
            ("DESNIVEL", (240, 200, 25)),
            ("A*", (40, 255, 100)),
            ("DIJKSTRA", (255, 190, 50)),
        ]
        lx = 12
        for name, col in items:
            cv2.rectangle(panel, (lx, legend_y - 10), (lx + 13, legend_y + 4), col, -1)
            cv2.putText(
                panel,
                name,
                (lx + 16, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 210, 220),
                1,
                cv2.LINE_AA,
            )
            lx += 28 + len(name) * 7

        return panel

    def _render_telemetry_sidebar(
        self,
        output: PipelineOutput,
        plane_locked: bool,
        source_name: str,
    ) -> np.ndarray:
        side = np.full((720, 360, 3), (14, 17, 24), dtype=np.uint8)
        cv2.line(side, (0, 0), (0, 720), (45, 55, 75), 2)

        cv2.putText(
            side,
            "ROVER D415 TELEMETRY",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (90, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            side,
            f"Source: {source_name[:34]}",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (160, 175, 195),
            1,
            cv2.LINE_AA,
        )

        corridor_ok = output.navigation.safe_corridor
        status_text = "SAFE CORRIDOR: TRUE" if corridor_ok else "SAFE CORRIDOR: FALSE"
        status_bg = (25, 110, 50) if corridor_ok else (30, 45, 180)
        cv2.rectangle(side, (16, 60), (344, 94), status_bg, -1)
        cv2.putText(
            side,
            status_text,
            (28, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        rx, ry, _ = output.rover_pose_xy_yaw
        metrics = [
            ("Edge Processing FPS", f"{output.edge_fps:.1f} FPS"),
            ("Camera / Drone Altitude", f"{output.terrain.altitude_m:.2f} m"),
            ("Max Terrain Incline", f"{output.terrain.max_incline_deg:.1f} deg"),
            ("Global Pose (X, Y)", f"({rx:+.2f}, {ry:+.2f}) m"),
            ("Traversed Distance", f"{output.traversed_distance_m:.2f} m"),
            ("Explored Map Area", f"{output.navigation.explored_area_m2:.2f} m2"),
            (
                "A* vs Dijkstra Nodes",
                f"{output.navigation.astar_explored_nodes} vs {output.navigation.dijkstra_explored_nodes}",
            ),
            ("Mission Nav Mode", output.navigation.navigation_mode),
            (
                "Recommended Heading",
                f"{output.navigation.recommended_heading_deg:+.1f} deg",
            ),
            ("Persistent Obstacles", str(len(output.obstacles))),
        ]

        y = 118
        for k, v in metrics:
            cv2.putText(
                side,
                k,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (155, 170, 190),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                side,
                v,
                (198, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (235, 245, 255),
                1,
                cv2.LINE_AA,
            )
            y += 23

        live_obs_list = (
            output.frame_obstacles
            if output.frame_obstacles is not None
            else output.obstacles
        )
        cv2.line(side, (16, y - 8), (344, y - 8), (40, 50, 65), 1)
        cv2.putText(
            side,
            "LIVE OBJECTS & DESNIVELES (drone_cv)",
            (16, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (90, 200, 255),
            1,
            cv2.LINE_AA,
        )
        y += 30

        if not live_obs_list:
            cv2.putText(
                side,
                "Flat terrain - no objects or drops in view",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (120, 140, 130),
                1,
                cv2.LINE_AA,
            )
            y += 22
        else:
            for obs in live_obs_list[:5]:
                col = (50, 130, 255) if obs.type == "BOULDER" else (255, 210, 40)
                tag = "OBJETO" if obs.type == "BOULDER" else "DESNIVEL"
                dim_key = "H" if obs.type == "BOULDER" else "D"
                line = (
                    f"#{obs.id} {tag:<8} d={obs.distance_m:.2f}m "
                    f"ang={obs.bearing_deg:+.1f}deg {dim_key}={obs.magnitude_m:.2f}m"
                )
                cv2.putText(
                    side,
                    line,
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                y += 21

        ctrl_y = 562
        cv2.line(side, (16, ctrl_y - 14), (344, ctrl_y - 14), (40, 50, 65), 1)
        cv2.putText(
            side,
            "AUTOMATED PIPELINE (OPTIONAL KEYS)",
            (16, ctrl_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (90, 200, 255),
            1,
            cv2.LINE_AA,
        )
        controls = [
            "Auto-runs: Filters -> RANSAC -> DBSCAN -> VO -> A*&Dijkstra",
            "[W/A/S/D] Manual Pose Override (Optional)",
            "[M] Mode (Explore <-> Return to Base)",
            "[R] Reset Map  [P] Primary Planner",
            "[C] Lock Ground  [S] Snapshot  [Q] Quit",
        ]
        cy = ctrl_y + 22
        for c_txt in controls:
            cv2.putText(
                side,
                c_txt,
                (16, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 195, 210),
                1,
                cv2.LINE_AA,
            )
            cy += 21

        return side

    def render(
        self,
        output: PipelineOutput,
        boulder_mask: np.ndarray,
        crevice_mask: np.ndarray,
        plane_locked: bool = False,
        source_name: str = "Intel RealSense D415",
    ) -> np.ndarray:
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        rgb_panel = self._render_rgb_panel(output)
        depth_panel = self._render_depth_elevation_panel(
            output, boulder_mask, crevice_mask
        )
        radar_panel = self._render_occupancy_radar_panel(output)
        sidebar = self._render_telemetry_sidebar(output, plane_locked, source_name)

        canvas[0:360, 0:480] = rgb_panel
        canvas[360:720, 0:480] = depth_panel
        canvas[0:720, 480:920] = radar_panel
        canvas[0:720, 920:1280] = sidebar

        return canvas
