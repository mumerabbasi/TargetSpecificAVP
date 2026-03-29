"""Configuration for the per-target CARLA dataset pipeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass
class Config:
    """Configuration shared by raw capture, dataset build, and benchmarking.

    The pipeline is intentionally split into two stages:

    1. Raw capture from CARLA in a CARLA-compatible Python environment.
    2. Offline dataset building with SAM3 and the 3D detector in a modern env.
    """

    # ------------------------------------------------------------------
    # Output layout
    # ------------------------------------------------------------------
    output_dir: str = "carla_dataset"
    fresh_start: bool = False

    raw_subdir: str = "raw_capture"
    raw_rgb_subdir: str = "rgb"
    raw_lidar_subdir: str = "lidar"
    raw_instance_subdir: str = "instance"
    raw_metadata_subdir: str = "metadata"

    final_rgb_subdir: str = "rgb"
    final_masks_subdir: str = "masks"
    gt_csv_name: str = "gt_poses.csv"
    pred_csv_name: str = "pred_poses.csv"
    benchmark_subdir: str = "benchmarks"

    # ------------------------------------------------------------------
    # CARLA connection
    # ------------------------------------------------------------------
    carla_host: str = "localhost"
    carla_port: int = 2150
    tm_port: int = 8000
    client_timeout_s: float = 120.0

    # ------------------------------------------------------------------
    # Capture settings
    # ------------------------------------------------------------------
    towns: Tuple[str, ...] = ("Town01", "Town02", "Town03", "Town04", "Town05")
    target_samples_per_town: int = 3000
    max_frames_per_town: int = 12000
    max_episodes_per_town: int = 4
    episode_frame_budget: int = 3000
    warmup_ticks: int = 80

    num_traffic_vehicles: int = 80
    traffic_mode: str = "traffic_manager"
    follow_only: bool = False
    min_follow_actors_per_frame: int = 1
    max_follow_actors_per_frame: int = 0
    follow_lateral_limit_m: float = 12.0
    follow_yaw_limit_deg: float = 120.0
    background_speed_difference_pct: float = 20.0
    ego_speed_difference_pct: float = 5.0
    traffic_follow_distance_m: float = 2.5
    constant_velocity_ego_speed_mps: float = 8.0
    constant_velocity_background_min_speed_mps: float = 5.0
    constant_velocity_background_max_speed_mps: float = 10.0
    nearby_vehicle_radius_m: float = 90.0

    sync_mode: bool = True
    fixed_delta_seconds: float = 0.05

    # ------------------------------------------------------------------
    # Sensor settings
    # ------------------------------------------------------------------
    image_width: int = 1024
    image_height: int = 1024
    fov: float = 90.0
    lidar_z_offset: float = 1.73

    # ------------------------------------------------------------------
    # Capture-time visibility filters
    # ------------------------------------------------------------------
    vehicle_semantic_tag: int = 14
    min_visible_vehicle_pixels: int = 100
    min_visible_bbox_width: int = 20
    min_visible_bbox_height: int = 20
    edge_margin_px: int = 6
    instance_bbox_dilation_px: int = 4

    # Used to keep long-range samples from being drowned out by near traffic.
    distance_bins_m: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    lateral_bins_m: Tuple[float, ...] = (0.0, 1.5, 3.5, 1000.0)
    yaw_bins_deg: Tuple[float, ...] = (0.0, 10.0, 30.0, 60.0, 120.0, 180.0)

    # ------------------------------------------------------------------
    # SAM3 mask generation
    # ------------------------------------------------------------------
    sam3_repo_path: str = "/my_workspace/4DHHOI/sam3"
    sam3_checkpoint_path: str = ""
    sam3_prompt: str = "car"
    sam3_fallback_prompt: str = "vehicle"
    sam3_confidence_threshold: float = 0.35
    sam3_duplicate_iou_thr: float = 0.75
    sam3_device: str = "cuda:0"

    # ------------------------------------------------------------------
    # Offline sample acceptance
    # ------------------------------------------------------------------
    min_mask_area_px: int = 120
    sam3_actor_iou_thr: float = 0.15

    # ------------------------------------------------------------------
    # 3D detector
    # ------------------------------------------------------------------
    model_dir: str = "mmdet3d_models"
    detector_name: str = "centerpoint"
    detector_config: str = field(default="")
    detector_checkpoint: str = field(default="")
    detector_score_thr: float = 0.15
    detector_match_dist_m: float = 4.0
    detector_device: str = "cuda:0"

    # ------------------------------------------------------------------
    # Benchmarking / reporting
    # ------------------------------------------------------------------
    save_reports: bool = True

    def __post_init__(self) -> None:
        if not self.detector_config:
            self.detector_config = os.path.join(
                self.model_dir,
                "configs/centerpoint/"
                "centerpoint_voxel0075_second_secfpn_head-dcn-circlenms"
                "_8xb4-cyclic-20e_nus-3d.py",
            )

        if not self.detector_checkpoint:
            self.detector_checkpoint = os.path.join(
                self.model_dir,
                "centerpoint_0075voxel_second_secfpn_dcn_circlenms"
                "_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth",
            )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    @property
    def raw_capture_dir(self) -> str:
        return os.path.join(self.output_dir, self.raw_subdir)

    @property
    def raw_rgb_dir(self) -> str:
        return os.path.join(self.raw_capture_dir, self.raw_rgb_subdir)

    @property
    def raw_lidar_dir(self) -> str:
        return os.path.join(self.raw_capture_dir, self.raw_lidar_subdir)

    @property
    def raw_instance_dir(self) -> str:
        return os.path.join(self.raw_capture_dir, self.raw_instance_subdir)

    @property
    def raw_metadata_dir(self) -> str:
        return os.path.join(self.raw_capture_dir, self.raw_metadata_subdir)

    @property
    def final_rgb_dir(self) -> str:
        return os.path.join(self.output_dir, self.final_rgb_subdir)

    @property
    def final_masks_dir(self) -> str:
        return os.path.join(self.output_dir, self.final_masks_subdir)

    @property
    def gt_csv_path(self) -> str:
        return os.path.join(self.output_dir, self.gt_csv_name)

    @property
    def pred_csv_path(self) -> str:
        return os.path.join(self.output_dir, self.pred_csv_name)

    @property
    def benchmark_dir(self) -> str:
        return os.path.join(self.output_dir, self.benchmark_subdir)

    @property
    def capture_dirs(self) -> Sequence[str]:
        return (
            self.raw_capture_dir,
            self.raw_rgb_dir,
            self.raw_lidar_dir,
            self.raw_instance_dir,
            self.raw_metadata_dir,
        )

    @property
    def final_dirs(self) -> Sequence[str]:
        return (
            self.final_rgb_dir,
            self.final_masks_dir,
            self.benchmark_dir,
        )

    @property
    def per_distance_bin_target(self) -> int:
        num_bins = max(len(self.distance_bins_m) - 1, 1)
        return max(1, math.ceil(self.target_samples_per_town / num_bins))
