"""Configuration for CARLA data collection."""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Configuration for CARLA 3D detection dataset collection."""

    # ---------------------------------------------------------------------------
    # Model paths
    # ---------------------------------------------------------------------------
    model_dir: str = "mmdet3d_models"

    # CenterPoint config
    centerpoint_config: str = field(default="")
    centerpoint_checkpoint: str = field(default="")

    # YOLO model
    yolo_model: str = "/usr/prakt/s0050/ravp/sam2/yolo11x.pt"

    # SAM2 model
    sam2_path: str = "/usr/prakt/s0050/ravp/sam2"
    sam2_checkpoint: str = ""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # ---------------------------------------------------------------------------
    # Detection thresholds
    # ---------------------------------------------------------------------------
    score_thr: float = 0.15  # CenterPoint score threshold
    yolo_conf: float = 0.5   # YOLO confidence threshold

    # ---------------------------------------------------------------------------
    # Outlier filtering thresholds
    # ---------------------------------------------------------------------------
    max_err_dx: float = 0.5   # meters
    max_err_dy: float = 0.5   # meters
    max_err_yaw: float = 5.0  # degrees

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    output_dir: str = "carla_dataset"
    fresh_start: bool = False  # If True, overwrite existing data instead of resuming

    # ---------------------------------------------------------------------------
    # CARLA connection
    # ---------------------------------------------------------------------------
    carla_host: str = "localhost"
    carla_port: int = 2150

    # ---------------------------------------------------------------------------
    # Dataset collection parameters
    # ---------------------------------------------------------------------------
    # List of CARLA towns to collect from
    towns: tuple = ("Town01", "Town02", "Town03", "Town04", "Town05")
    frames_per_town: int = 100  # Number of frames to collect per town
    min_targets: int = 1
    max_targets: int = 5

    # Derived parameters (computed automatically)
    num_waypoints: int = 0  # Will be computed from frames_per_town
    frames_per_waypoint: int = 1  # Keep at 1 for diversity

    # ---------------------------------------------------------------------------
    # Target spawn Gaussian distribution parameters
    # ---------------------------------------------------------------------------
    # dx: longitudinal distance (ahead of ego)
    target_dx_mean: float = 10.0
    target_dx_std: float = 3.0
    target_dx_min: float = 4.0
    target_dx_max: float = 20.0

    # dy: lateral offset (0 = same lane)
    target_dy_mean: float = 0.0
    target_dy_std: float = 2.0
    target_dy_min: float = -5.0
    target_dy_max: float = 5.0

    # dyaw: relative yaw (0 = same heading as ego)
    target_dyaw_mean: float = 0.0
    target_dyaw_std: float = 20.0
    target_dyaw_min: float = -70.0
    target_dyaw_max: float = 70.0

    # ---------------------------------------------------------------------------
    # Ego spawn Gaussian yaw offset (for scene diversity)
    # ---------------------------------------------------------------------------
    ego_dyaw_mean: float = 0.0
    ego_dyaw_std: float = 5.0
    ego_dyaw_min: float = -15.0
    ego_dyaw_max: float = 15.0

    # ---------------------------------------------------------------------------
    # Camera settings
    # ---------------------------------------------------------------------------
    image_width: int = 1024
    image_height: int = 1024
    fov: float = 90.0

    # LiDAR height offset
    lidar_z_offset: float = 1.73

    def __post_init__(self) -> None:
        """Set derived paths after initialization."""
        # Compute num_waypoints from frames_per_town
        # Each waypoint gives 1 frame (frames_per_waypoint=1)
        if self.num_waypoints == 0:
            self.num_waypoints = self.frames_per_town

        # CenterPoint paths
        if not self.centerpoint_config:
            cfg = "centerpoint_voxel0075_second_secfpn_head-dcn-circlenms"
            cfg += "_8xb4-cyclic-20e_nus-3d.py"
            self.centerpoint_config = os.path.join(
                self.model_dir, "configs/centerpoint", cfg
            )

        if not self.centerpoint_checkpoint:
            ckpt = "centerpoint_0075voxel_second_secfpn_dcn_circlenms"
            ckpt += "_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth"
            self.centerpoint_checkpoint = os.path.join(self.model_dir, ckpt)

        # SAM2 checkpoint
        if not self.sam2_checkpoint:
            self.sam2_checkpoint = os.path.join(
                self.sam2_path, "checkpoints/sam2.1_hiera_small.pt"
            )

    @property
    def csv_output(self) -> str:
        """Path to output CSV file."""
        return os.path.join(self.output_dir, "poses.csv")

    @property
    def dx_range(self) -> Tuple[float, float]:
        """Range for dx clipping."""
        return (self.target_dx_min, self.target_dx_max)

    @property
    def dy_range(self) -> Tuple[float, float]:
        """Range for dy clipping."""
        return (self.target_dy_min, self.target_dy_max)

    @property
    def dyaw_range(self) -> Tuple[float, float]:
        """Range for dyaw clipping."""
        return (self.target_dyaw_min, self.target_dyaw_max)

    @property
    def ego_dyaw_range(self) -> Tuple[float, float]:
        """Range for ego dyaw clipping."""
        return (self.ego_dyaw_min, self.ego_dyaw_max)
