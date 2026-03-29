"""Mask-conditioned target pose learning."""

from .config import PoseEstimationConfig
from .dataset import (
    PoseEstimationDataset,
    TranslationStats,
    build_frame_splits,
    denormalize_translation,
    load_pose_rows,
    normalize_translation,
    split_summary,
)
from .model import (
    PoseEstimationCNN,
    PoseEstimationLoss,
    compute_pose_metrics,
    decode_predictions,
)

__all__ = [
    "PoseEstimationConfig",
    "PoseEstimationDataset",
    "TranslationStats",
    "build_frame_splits",
    "denormalize_translation",
    "load_pose_rows",
    "normalize_translation",
    "split_summary",
    "PoseEstimationCNN",
    "PoseEstimationLoss",
    "compute_pose_metrics",
    "decode_predictions",
]
