"""Configuration for the simplified in-process pursuit evaluation pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PursuitEvalConfig:
    """Configuration shared by the CARLA runtime and in-process perception stack."""

    pose_source: str = "gt"  # one of: gt, detector
    output_dir: str = os.path.join(_REPO_ROOT, "pursuit_eval_output")
    run_name: str = ""

    # CARLA runtime
    carla_host: str = "localhost"
    carla_port: int = 2150
    tm_port: int = 8000
    client_timeout_s: float = 60.0
    town: str = "Town02"
    random_seed: int = 7
    sync_mode: bool = True
    fixed_delta_seconds: float = 0.1
    num_frames: int = 300
    warmup_ticks: int = 25
    clear_existing_vehicles: bool = True

    # Scenario setup
    num_background_vehicles: int = 20
    initial_target_distance_m: float = 12.0
    ego_initial_speed_mps: float = 0.0
    target_speed_difference_pct: float = 80.0
    background_speed_difference_pct: float = 15.0
    traffic_follow_distance_m: float = 2.5
    spawn_attempts: int = 30
    background_spawn_exclusion_radius_m: float = 15.0
    target_spawn_z_offset_m: float = 0.3
    require_follow_friendly_spawn: bool = True
    follow_spawn_lookahead_m: float = 45.0
    follow_spawn_step_m: float = 5.0
    follow_spawn_max_yaw_delta_deg: float = 12.0

    # Ego camera / lidar
    image_width: int = 1024
    image_height: int = 1024
    fov: float = 90.0
    camera_x_m: float = 1.5
    camera_y_m: float = 0.0
    camera_z_m: float = 1.6
    lidar_x_m: float = 0.0
    lidar_y_m: float = 0.0
    lidar_z_m: float = 1.73
    lidar_range_m: float = 80.0
    lidar_rotation_frequency_hz: float = 20.0
    lidar_points_per_second: int = 600000
    lidar_channels: int = 64
    lidar_upper_fov_deg: float = 2.0
    lidar_lower_fov_deg: float = -24.8

    # MPC
    desired_distance_m: float = 8.0
    collision_distance_m: float = 4.0
    slowdown_distance_m: float = 7.0
    wheelbase_m: float = 2.87
    mpc_horizon: int = 25
    mpc_dt: float = 0.1
    max_throttle: float = 0.8
    max_brake: float = 0.8
    max_steer: float = 0.7
    max_steer_rad: float = 0.5
    max_accel: float = 3.0
    max_decel: float = -6.0
    launch_throttle_floor: float = 0.22
    launch_speed_threshold_mps: float = 2.0
    steer_filter_alpha: float = 0.9
    w_dist: float = 8.0
    w_lat: float = 18.0
    w_yaw: float = 3.0
    w_vel: float = 1.0
    w_accel: float = 1.5
    w_steer: float = 10.0
    w_daccel: float = 10.0
    w_dsteer: float = 200.0

    # Pursuit evaluation thresholds
    follow_band_distance_abs_m: float = 2.0
    follow_band_lateral_abs_m: float = 1.5
    follow_band_yaw_abs_deg: float = 20.0
    follow_guard_lateral_abs_m: float = 4.0
    follow_guard_yaw_abs_deg: float = 25.0
    follow_guard_min_dx_m: float = 2.0
    follow_guard_breach_frames: int = 3
    stop_on_follow_guard_breach: bool = False
    target_out_of_view_breach_frames: int = 20
    ego_offroad_breach_frames: int = 15
    max_pose_hold_frames: int = 12

    # In-process detector / tracker
    bootstrap_with_gt_bbox: bool = True
    enable_bbox_reseed: bool = True
    prompt_bbox_pad_px: int = 24
    sam3_repo_path: str = "/my_workspace/4DHHOI/sam3"
    sam3_checkpoint_path: str = ""
    sam3_confidence_threshold: float = 0.35
    sam3_duplicate_iou_thr: float = 0.75
    sam3_device: str = "cuda:0"
    detector_name: str = "centerpoint"
    detector_config: str = ""
    detector_checkpoint: str = ""
    detector_score_thr: float = 0.10
    detector_device: str = "cuda:0"
    detector_projection_iou_thr: float = 0.02
    detector_projection_score_weight: float = 0.15

    # Output / debugging
    enable_spectator_camera: bool = True
    spectator_width: int = 1024
    spectator_height: int = 1024
    spectator_fov: float = 110.0
    spectator_x_m: float = 0.0
    spectator_y_m: float = 0.0
    spectator_z_m: float = 24.0
    spectator_pitch_deg: float = -90.0
    spectator_yaw_deg: float = 0.0
    spectator_roll_deg: float = 0.0
    save_debug_images: bool = False
    save_tracking_masks: bool = False

    def __post_init__(self) -> None:
        if not self.detector_config:
            self.detector_config = os.path.join(
                _REPO_ROOT,
                "mmdet3d_models",
                "configs",
                "centerpoint",
                "centerpoint_voxel0075_second_secfpn_head-dcn-"
                "circlenms_8xb4-cyclic-20e_nus-3d.py",
            )
        if not self.detector_checkpoint:
            self.detector_checkpoint = os.path.join(
                _REPO_ROOT,
                "mmdet3d_models",
                "centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_"
                "cyclic_20e_nus_20220810_025930-657f67e0.pth",
            )

    @property
    def run_output_dir(self) -> str:
        if self.run_name:
            return os.path.join(self.output_dir, self.run_name)
        return self.output_dir

    @property
    def frame_log_path(self) -> str:
        return os.path.join(self.run_output_dir, "frames.jsonl")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.run_output_dir, "metrics.json")

    @property
    def debug_dir(self) -> str:
        return os.path.join(self.run_output_dir, "debug")

    @property
    def tracker_masks_dir(self) -> str:
        return os.path.join(self.debug_dir, "tracker_masks")

    @property
    def spectator_frames_dir(self) -> str:
        return os.path.join(self.run_output_dir, "spectator_frames")

    @property
    def spectator_video_path(self) -> str:
        return os.path.join(self.run_output_dir, "spectator.mp4")

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self) -> str:
        os.makedirs(self.run_output_dir, exist_ok=True)
        path = os.path.join(self.run_output_dir, "config.json")
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path
