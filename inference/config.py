"""Configuration for inference and vehicle pursuit."""

from dataclasses import dataclass
from typing import Tuple
import json
import os


@dataclass
class InferenceConfig:
    """Configuration for pose estimation inference and MPC pursuit.

    Attributes:
        Paths and model settings for inference.
        CARLA connection and camera settings.
        MPC controller parameters.
    """

    # -------------------------------------------------------------------------
    # Model paths
    # -------------------------------------------------------------------------
    checkpoint_path: str = (
        "/usr/prakt/s0050/ravp/pose_estimation_runs/"
        "pose_estimation_20251129_220737/best_model.pth"
    )
    config_path: str = (
        "/usr/prakt/s0050/ravp/pose_estimation_runs/"
        "pose_estimation_20251129_220737/config.json"
    )

    # -------------------------------------------------------------------------
    # CARLA connection
    # -------------------------------------------------------------------------
    carla_host: str = "localhost"
    carla_port: int = 2150

    # -------------------------------------------------------------------------
    # Camera settings (must match training)
    # -------------------------------------------------------------------------
    image_width: int = 1024
    image_height: int = 1024
    fov: float = 90.0

    # -------------------------------------------------------------------------
    # Model input settings (loaded from config.json)
    # -------------------------------------------------------------------------
    model_image_size: Tuple[int, int] = (224, 224)
    backbone: str = "resnet50"
    bbox_mode: str = "mask"

    # Pose normalization (loaded from config.json)
    pose_mean: Tuple[float, float, float, float] = (8.0, 0.0, -0.95, 0.0)
    pose_std: Tuple[float, float, float, float] = (2.5, 1.7, 0.1, 25.0)

    # Which outputs are predicted
    predict_dx: bool = True
    predict_dy: bool = True
    predict_dz: bool = False
    predict_yaw: bool = True

    # -------------------------------------------------------------------------
    # Scenario settings
    # -------------------------------------------------------------------------
    town: str = "Town04"
    num_target_vehicles: int = 3
    initial_target_distance: float = 6.0  # meters ahead

    # -------------------------------------------------------------------------
    # MPC controller parameters
    # -------------------------------------------------------------------------
    # Prediction horizon
    mpc_horizon: int = 30  # Number of steps to predict
    mpc_dt: float = 0.1  # Time step for MPC prediction (seconds)

    # Vehicle parameters
    wheelbase: float = 2.87  # Distance between axles (meters)

    # Target following parameters
    desired_distance: float = 4.0  # meters behind target
    desired_lateral_offset: float = 0.0  # meters (0 = same lane)

    # -------------------------------------------------------------------------
    # MPC constraints (physical limits)
    # -------------------------------------------------------------------------
    max_throttle: float = 0.8
    max_brake: float = 0.8
    max_steer: float = 0.7  # CARLA steering limit [-1, 1]
    max_steer_rad: float = 0.5  # Max steering angle in radians (~28 deg)
    max_accel: float = 3.0  # m/s^2
    max_decel: float = -6.0  # m/s^2
    max_speed: float = 30.0  # m/s (~108 km/h)

    # -------------------------------------------------------------------------
    # MPC cost weights (tune these for behavior)
    # -------------------------------------------------------------------------
    # Tracking weights
    w_dist: float = 8.0  # Longitudinal distance error
    w_lat: float = 15.0  # Lateral offset error
    w_yaw: float = 3.0  # Heading error
    w_vel: float = 1.0  # Velocity matching error

    # Control effort weights
    w_accel: float = 1.5  # Penalize acceleration magnitude
    w_steer: float = 10.0  # Penalize steering magnitude

    # Smoothness weights (penalize control changes)
    w_daccel: float = 10.0  # Smooth acceleration changes
    w_dsteer: float = 200.0  # Smooth steering changes (HIGH for stability)

    # Low-pass filter for steering (0 = no filter, 1 = full filter)
    steer_filter_alpha: float = 0.9

    # -------------------------------------------------------------------------
    # Safety parameters
    # -------------------------------------------------------------------------
    collision_distance: float = 0  # Emergency brake if closer
    slowdown_distance: float = 0  # Start slowing if closer

    # -------------------------------------------------------------------------
    # Simulation settings
    # -------------------------------------------------------------------------
    sync_mode: bool = True
    fixed_delta_seconds: float = 0.1  # 10 FPS
    num_frames: int = 1000  # Number of frames to run

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    show_debug_info: bool = True
    save_video: bool = False
    video_output_dir: str = "inference_output"

    def __post_init__(self) -> None:
        """Load model config from checkpoint directory."""
        if os.path.exists(self.config_path):
            self.load_model_config()

    def load_model_config(self) -> None:
        """Load pose estimation config from training checkpoint."""
        with open(self.config_path, "r") as f:
            model_config = json.load(f)

        # Update relevant fields
        self.backbone = model_config.get("backbone", self.backbone)
        self.bbox_mode = model_config.get("bbox_mode", self.bbox_mode)
        self.model_image_size = tuple(
            model_config.get("image_size", list(self.model_image_size))
        )
        self.pose_mean = tuple(
            model_config.get("pose_mean", list(self.pose_mean))
        )
        self.pose_std = tuple(
            model_config.get("pose_std", list(self.pose_std))
        )
        self.predict_dx = model_config.get("predict_dx", self.predict_dx)
        self.predict_dy = model_config.get("predict_dy", self.predict_dy)
        self.predict_dz = model_config.get("predict_dz", self.predict_dz)
        self.predict_yaw = model_config.get("predict_yaw", self.predict_yaw)

    @property
    def num_outputs(self) -> int:
        """Get the number of model output dimensions."""
        return sum([
            self.predict_dx,
            self.predict_dy,
            self.predict_dz,
            self.predict_yaw,
        ])

    @property
    def output_names(self) -> list:
        """Get names of output dimensions."""
        names = []
        if self.predict_dx:
            names.append("dx")
        if self.predict_dy:
            names.append("dy")
        if self.predict_dz:
            names.append("dz")
        if self.predict_yaw:
            names.append("yaw")
        return names
