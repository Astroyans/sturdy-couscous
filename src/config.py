from dataclasses import dataclass


@dataclass
class PipelineConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    min_depth_m: float = 0.25
    max_depth_m: float = 6.50
    nominal_camera_height_m: float = 0.45
    nominal_camera_pitch_deg: float = -15.0

    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.2
    morph_kernel_size: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2

    pointcloud_stride: int = 2
    ransac_iterations: int = 60
    ransac_inlier_threshold_m: float = 0.025
    max_safe_incline_deg: float = 20.0
    slope_window_size: int = 5

    boulder_min_height_m: float = 0.08
    crevice_min_depth_m: float = 0.08
    max_anomaly_magnitude_m: float = 1.20
    min_cluster_pixels: int = 320
    min_obstacle_extent_m: float = 0.09
    use_dbscan: bool = True
    dbscan_eps_m: float = 0.22
    dbscan_min_samples: int = 6
    dbscan_max_points: int = 320

    grid_width_m: float = 4.00
    grid_depth_m: float = 6.00
    grid_resolution_m: float = 0.05
    robot_radius_m: float = 0.15
    corridor_half_width_m: float = 0.25
    corridor_check_distance_m: float = 1.80
    default_planner: str = "astar"
    auto_frontier_route: bool = False
    proximity_penalty_weight: float = 3.5

    persistent_mapping: bool = True
    vo_deadband_m: float = 0.035
    log_odds_hit: float = 0.85
    log_odds_free: float = -0.55
    log_odds_min: float = -4.0
    log_odds_max: float = 5.0
    log_odds_occupied_threshold: float = 0.60
    obstacle_merge_radius_m: float = 0.40
