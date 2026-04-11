"""Training package for target pose regression from RGB and mask inputs."""

from .config import TargetPoseTrainingConfig
from .dataset import TargetPoseDataset, TranslationStats
from .model import (
    TargetPoseLoss,
    TargetPoseRegressor,
    compute_pose_metrics,
    decode_predictions,
)

__all__ = [
    "TargetPoseDataset",
    "TargetPoseLoss",
    "TargetPoseRegressor",
    "TargetPoseTrainingConfig",
    "TranslationStats",
    "compute_pose_metrics",
    "decode_predictions",
]
