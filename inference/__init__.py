"""Inference module for vehicle pursuit using pose estimation and MPC control.

This module provides:
- PoseEstimator: CNN-based pose estimation from RGB + mask
- MPCController: Model Predictive Control for vehicle following
- CARLA utilities: World setup, sensors, target mask extraction
- VehiclePursuit: Main pursuit loop integrating all components

Usage:
    cd robust-autonomous-vehicle-pursuit
    conda activate 3d_detector
    python -m inference.run_pursuit --town Town04 --duration 60 --save-images

Or programmatically:
    from inference.config import InferenceConfig
    from inference.pose_estimator import PoseEstimator
    from inference.mpc_controller import MPCController

    config = InferenceConfig()
    estimator = PoseEstimator(config)
    controller = MPCController(config)
"""

from .config import InferenceConfig
from .pose_estimator import PoseEstimator
from .mpc_controller import (
    MPCController,
    TargetPose,
    VehicleState,
    ControlCommand,
)

__all__ = [
    "InferenceConfig",
    "PoseEstimator",
    "MPCController",
    "TargetPose",
    "VehicleState",
    "ControlCommand",
]
