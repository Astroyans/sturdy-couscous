from __future__ import annotations

import base64
from typing import Any, Optional

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


def encode_jpeg_base64(
    image_bgr: np.ndarray,
    width: int = 640,
    height: int = 480,
    quality: int = 80,
    as_data_uri: bool = True,
) -> str:
    h, w = image_bgr.shape[:2]
    if (w, h) != (width, height):
        resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr

    ok, buf = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 30, 95))],
    )
    if not ok:
        raise RuntimeError("No se pudo codificar imagen JPEG")

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if as_data_uri else b64


def render_rgb_annotated(output: PipelineOutput) -> np.ndarray:
    rgb = output.frame.color_bgr.copy()
    h, w = rgb.shape[:2]

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
    return rgb


def render_depth_map(
    output: PipelineOutput,
    config: Optional[PipelineConfig] = None,
    boulder_mask: Optional[np.ndarray] = None,
    crevice_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    cfg = config or PipelineConfig()
    depth = output.preprocessed_depth_m
    valid = output.terrain.valid_mask

    norm = np.clip(
        (depth - cfg.min_depth_m) / max(cfg.max_depth_m - cfg.min_depth_m, 0.1),
        0.0,
        1.0,
    )
    depth_u8 = ((1.0 - norm) * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    heatmap[~valid] = (18, 18, 24)

    if boulder_mask is not None:
        b_cnts, _ = cv2.findContours(
            boulder_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(heatmap, b_cnts, -1, (255, 255, 255), 2)
    if crevice_mask is not None:
        c_cnts, _ = cv2.findContours(
            crevice_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(heatmap, c_cnts, -1, (255, 255, 0), 2)

    return heatmap


def render_route_map(
    output: PipelineOutput,
    config: Optional[PipelineConfig] = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    cfg = config or PipelineConfig()
    grid = output.navigation.occupancy_grid
    rows, cols = grid.shape

    radar_img = np.full((rows, cols, 3), (26, 29, 36), dtype=np.uint8)
    radar_img[grid == GRID_FREE] = (44, 62, 50)
    radar_img[grid == GRID_INFLATED] = (38, 88, 118)
    radar_img[grid == GRID_BOULDER] = (35, 95, 245)
    radar_img[grid == GRID_CREVICE] = (240, 200, 25)
    radar_img[grid == GRID_STEEP] = (140, 40, 200)

    radar_flipped = np.flipud(radar_img)
    map_h = height - 34
    scaled = cv2.resize(radar_flipped, (width, map_h), interpolation=cv2.INTER_NEAREST)

    sx = width / float(cols)
    sy = map_h / float(rows)
    res = cfg.grid_resolution_m
    max_y_m = rows * res
    y_min_m = -max_y_m / 2.0 if cfg.nominal_camera_pitch_deg <= -65.0 else 0.0

    step_m = 1.0 if max_y_m > 4.0 else 0.5
    d_m = step_m
    while d_m < max_y_m:
        r = int(round(d_m / res))
        if 0 < r < rows:
            py = int(round((rows - 1 - r) * sy))
            label_m = d_m + y_min_m
            cv2.line(scaled, (0, py), (width, py), (60, 75, 65), 1)
            cv2.putText(
                scaled,
                f"{label_m:+.1f}m" if y_min_m < 0 else f"{label_m:.1f}m",
                (8, max(14, py - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (160, 185, 170),
                1,
                cv2.LINE_AA,
            )
        d_m += step_m

    cx_px = int(round((cols / 2.0) * sx))
    cv2.line(scaled, (cx_px, 0), (cx_px, map_h), (65, 80, 70), 1)

    if len(output.navigation.trajectory_cells) >= 2:
        t_pts = [
            [int(round((c + 0.5) * sx)), int(round((rows - 1 - r + 0.5) * sy))]
            for r, c in output.navigation.trajectory_cells
        ]
        t_arr = np.array(t_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(scaled, [t_arr], False, (255, 180, 60), 2, cv2.LINE_AA)

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

    canvas = np.full((height, width, 3), (18, 22, 30), dtype=np.uint8)
    canvas[0:map_h, 0:width] = scaled

    legend_y = height - 12
    items = [
        ("SUELO", (44, 62, 50)),
        ("OBJETO", (35, 95, 245)),
        ("DESNIVEL", (240, 200, 25)),
        ("A*", (40, 255, 100)),
        ("DIJKSTRA", (255, 190, 50)),
    ]
    lx = 12
    for name, col in items:
        cv2.rectangle(canvas, (lx, legend_y - 10), (lx + 14, legend_y + 3), col, -1)
        cv2.putText(
            canvas,
            name,
            (lx + 18, legend_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 220, 230),
            1,
            cv2.LINE_AA,
        )
        lx += 36 + len(name) * 8

    return canvas


def build_expo_payload(
    output: PipelineOutput,
    config: Optional[PipelineConfig] = None,
    boulder_mask: Optional[np.ndarray] = None,
    crevice_mask: Optional[np.ndarray] = None,
    image_width: int = 640,
    image_height: int = 480,
    jpeg_quality: int = 80,
    annotate_rgb: bool = True,
) -> dict[str, Any]:
    payload = output.to_drone_cv_payload()
    rgb_img = render_rgb_annotated(output) if annotate_rgb else output.frame.color_bgr
    depth_img = render_depth_map(output, config, boulder_mask, crevice_mask)
    route_img = render_route_map(output, config, width=image_width, height=image_height)

    payload["images"] = {
        "format": "jpeg_base64",
        "width": image_width,
        "height": image_height,
        "rgb": encode_jpeg_base64(
            rgb_img, width=image_width, height=image_height, quality=jpeg_quality
        ),
        "depth": encode_jpeg_base64(
            depth_img, width=image_width, height=image_height, quality=jpeg_quality
        ),
        "route_map": encode_jpeg_base64(
            route_img, width=image_width, height=image_height, quality=jpeg_quality
        ),
    }
    return payload
