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
        x_m: float,
        y_m: float,
        z_m: float,
        output: PipelineOutput,
    ) -> Optional[tuple[int, int]]:
        pitch_rad = math.radians(output.frame.imu_pitch)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        z_cam = y_m * cos_p + z_m * sin_p
        y_up_cam = -y_m * sin_p + z_m * cos_p
        if z_cam <= 0.15:
            return None

        intr = output.frame.intrinsics
        u = int(round((x_m / z_cam) * intr.fx + intr.ppx))
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
            if wy < 0.25:
                continue
            uv = self._project_world_to_pixel(wx, wy, ground_z, output)
            if uv is not None:
                projected_pts.append(uv)

        if len(projected_pts) >= 2:
            pts_arr = np.array(projected_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts_arr], False, (0, 255, 120), 3, cv2.LINE_AA)

        for obs in output.obstacles:
            x, y, bw, bh = obs.bbox_2d
            if obs.type == "BOULDER":
                color = (30, 110, 255)
                metric_str = f"H:{obs.magnitude_m:.2f}m"
            else:
                color = (255, 210, 0)
                metric_str = f"D:{obs.magnitude_m:.2f}m"

            cv2.rectangle(rgb, (x, y), (x + bw, y + bh), color, 2)
            label = f"#{obs.id} {obs.type} {obs.distance_m:.2f}m {obs.bearing_deg:+.0f}deg {metric_str}"
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

        for dist_m in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            r = int(round(dist_m / res))
            if 0 < r < rows:
                py = int(round((rows - 1 - r) * sy))
                cv2.line(scaled, (0, py), (map_w, py), (60, 75, 65), 1)
                cv2.putText(
                    scaled,
                    f"{dist_m:.1f}m",
                    (6, max(12, py - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (150, 175, 160),
                    1,
                    cv2.LINE_AA,
                )

        cx_px = int(round((cols / 2.0) * sx))
        cv2.line(scaled, (cx_px, 0), (cx_px, map_h), (65, 80, 70), 1)

        if len(output.navigation.path_cells) >= 2:
            pts = []
            for r, c in output.navigation.path_cells:
                px = int(round((c + 0.5) * sx))
                py = int(round((rows - 1 - r + 0.5) * sy))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(scaled, [pts_arr], False, (40, 255, 100), 3, cv2.LINE_AA)
            cv2.circle(scaled, tuple(pts[-1]), 6, (0, 255, 255), -1)

        sr, sc = output.navigation.start_cell
        rover_px = int(round((sc + 0.5) * sx))
        rover_py = int(round((rows - 1 - sr) * sy))
        cv2.circle(scaled, (rover_px, rover_py), 8, (255, 255, 255), -1)
        cv2.circle(scaled, (rover_px, rover_py), 5, (0, 180, 255), -1)

        panel = np.full((720, 440, 3), (18, 22, 30), dtype=np.uint8)
        panel[40 : 40 + map_h, 10 : 10 + map_w] = scaled

        cv2.putText(
            panel,
            f"3. 2D OCCUPANCY GRID & {output.navigation.planner_used.upper()} PATH",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 230, 245),
            1,
            cv2.LINE_AA,
        )

        legend_y = 682
        items = [
            ("FREE", (44, 62, 50)),
            ("INFLATED", (38, 88, 118)),
            ("BOULDER", (35, 95, 245)),
            ("CREVICE", (240, 200, 25)),
            ("PATH", (40, 255, 100)),
        ]
        lx = 12
        for name, col in items:
            cv2.rectangle(panel, (lx, legend_y - 10), (lx + 14, legend_y + 4), col, -1)
            cv2.putText(
                panel,
                name,
                (lx + 18, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 210, 220),
                1,
                cv2.LINE_AA,
            )
            lx += 84

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
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (90, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            side,
            f"Source: {source_name[:34]}",
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 175, 195),
            1,
            cv2.LINE_AA,
        )

        corridor_ok = output.navigation.safe_corridor
        status_text = "SAFE CORRIDOR: TRUE" if corridor_ok else "SAFE CORRIDOR: FALSE"
        status_bg = (25, 110, 50) if corridor_ok else (30, 45, 180)
        cv2.rectangle(side, (16, 66), (344, 104), status_bg, -1)
        cv2.putText(
            side,
            status_text,
            (28, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        metrics = [
            ("Edge Processing FPS", f"{output.edge_fps:.1f} FPS"),
            ("Camera / Rover Altitude", f"{output.terrain.altitude_m:.2f} m"),
            ("Max Terrain Incline", f"{output.terrain.max_incline_deg:.1f} deg"),
            (
                "Ground Pitch / Roll",
                f"{output.terrain.pitch_deg:+.1f} / {output.terrain.roll_deg:+.1f} deg",
            ),
            (
                "Ground Plane Mode",
                "LOCKED (Manual Ref)" if plane_locked else "RANSAC Dynamic",
            ),
            ("Active Path Planner", output.navigation.planner_used.upper()),
            ("Explored Graph Nodes", str(output.navigation.explored_nodes)),
            (
                "Recommended Heading",
                f"{output.navigation.recommended_heading_deg:+.1f} deg",
            ),
            ("Detected Obstacles", str(len(output.obstacles))),
        ]

        y = 132
        for k, v in metrics:
            cv2.putText(
                side,
                k,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (155, 170, 190),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                side,
                v,
                (205, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (235, 245, 255),
                1,
                cv2.LINE_AA,
            )
            y += 28

        cv2.line(side, (16, y - 10), (344, y - 10), (40, 50, 65), 1)
        cv2.putText(
            side,
            "DETECTED OBSTACLES (drone_cv)",
            (16, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (90, 200, 255),
            1,
            cv2.LINE_AA,
        )
        y += 36

        if not output.obstacles:
            cv2.putText(
                side,
                "No topographical anomalies detected",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (120, 140, 130),
                1,
                cv2.LINE_AA,
            )
            y += 24
        else:
            for obs in output.obstacles[:6]:
                col = (50, 130, 255) if obs.type == "BOULDER" else (255, 210, 40)
                dim_key = "H" if obs.type == "BOULDER" else "D"
                line = (
                    f"#{obs.id} {obs.type:<7} d={obs.distance_m:.2f}m "
                    f"ang={obs.bearing_deg:+.1f}deg {dim_key}={obs.magnitude_m:.2f}m"
                )
                cv2.putText(
                    side,
                    line,
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                y += 24

        ctrl_y = 585
        cv2.line(side, (16, ctrl_y - 16), (344, ctrl_y - 16), (40, 50, 65), 1)
        cv2.putText(
            side,
            "KEYBOARD CONTROLS",
            (16, ctrl_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 200, 255),
            1,
            cv2.LINE_AA,
        )
        controls = [
            "[P] Toggle Planner (A* <-> Dijkstra)",
            "[C] Lock / Unlock Ground Plane Ref",
            "[S] Save HUD Image & JSON Snapshot",
            "[Q / ESC] Exit Application",
        ]
        cy = ctrl_y + 24
        for c_txt in controls:
            cv2.putText(
                side,
                c_txt,
                (16, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (180, 195, 210),
                1,
                cv2.LINE_AA,
            )
            cy += 22

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
