"""Configuración central del pipeline de análisis de profundidad D415 y navegación."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Parámetros físicos y algorítmicos para percepción 3D y navegación del Rover."""

    # --- Fase 1: Cámara Intel RealSense D415 y Rango Operativo ---
    width: int = 640
    height: int = 480
    fps: int = 30
    min_depth_m: float = 0.25
    max_depth_m: float = 4.00
    nominal_camera_height_m: float = 0.45
    nominal_camera_pitch_deg: float = -15.0

    # --- Fase 1: Preprocesamiento (Filtro Gaussiano + Morfología Matemática) ---
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.2
    morph_kernel_size: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2

    # --- Fase 2: Estimación del Plano del Suelo (RANSAC) e Inclinación ---
    pointcloud_stride: int = 2
    ransac_iterations: int = 60
    ransac_inlier_threshold_m: float = 0.025
    max_safe_incline_deg: float = 20.0
    slope_window_size: int = 5

    # --- Fase 3: Segmentación de Obstáculos (BOULDER / CREVICE) + DBSCAN ---
    boulder_min_height_m: float = 0.045
    crevice_min_depth_m: float = 0.045
    max_anomaly_magnitude_m: float = 1.50
    min_cluster_pixels: int = 100
    use_dbscan: bool = True
    dbscan_eps_m: float = 0.08
    dbscan_min_samples: int = 10
    dbscan_max_points: int = 1500

    # --- Fase 4: Malla de Ocupación 2D (Bird's-Eye View) y Ruta (A* / Dijkstra) ---
    grid_width_m: float = 3.00
    grid_depth_m: float = 3.50
    grid_resolution_m: float = 0.05
    robot_radius_m: float = 0.15
    corridor_half_width_m: float = 0.25
    corridor_check_distance_m: float = 1.80
    default_planner: str = "astar"  # "astar" | "dijkstra"
    proximity_penalty_weight: float = 3.5
