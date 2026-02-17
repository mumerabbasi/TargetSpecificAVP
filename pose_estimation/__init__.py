"""Pose estimation module for predicting vehicle poses from RGB images."""

from .config import PoseEstimationConfig
from .dataset import (
    PoseEstimationDataset,
    compute_pose_statistics,
    create_data_splits,
)
from .model import (
    PoseEstimationCNN,
    PoseEstimationLoss,
    PoseEstimationMetrics,
    create_model,
)

__all__ = [
    "PoseEstimationConfig",
    "PoseEstimationDataset",
    "PoseEstimationCNN",
    "PoseEstimationLoss",
    "PoseEstimationMetrics",
    "compute_pose_statistics",
    "create_data_splits",
    "create_model",
]
