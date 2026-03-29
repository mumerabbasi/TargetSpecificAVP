"""Configuration for CNN-based pursuit inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Runtime configuration for pursuit inference."""

    checkpoint_path: str = ""

    carla_host: str = "localhost"
    carla_port: int = 2150

    image_width: int = 768
    image_height: int = 768
    fov: float = 90.0

    town: str = "Town02"
    num_target_vehicles: int = 3
    initial_target_distance: float = 6.0

    mpc_horizon: int = 30
    mpc_dt: float = 0.1
    wheelbase: float = 2.87

    desired_distance: float = 4.0
    desired_lateral_offset: float = 0.0

    max_throttle: float = 0.8
    max_brake: float = 0.8
    max_steer: float = 0.7
    max_steer_rad: float = 0.5
    max_accel: float = 3.0
    max_decel: float = -6.0
    max_speed: float = 30.0

    w_dist: float = 8.0
    w_lat: float = 15.0
    w_yaw: float = 3.0
    w_vel: float = 1.0
    w_accel: float = 1.5
    w_steer: float = 10.0
    w_daccel: float = 10.0
    w_dsteer: float = 200.0
    steer_filter_alpha: float = 0.9

    collision_distance: float = 0.0
    slowdown_distance: float = 0.0

    sync_mode: bool = True
    fixed_delta_seconds: float = 0.1
    num_frames: int = 1000

    show_debug_info: bool = True
    save_video: bool = False
    video_output_dir: str = "inference_output"
