"""Load a trained pose model and run RGB-plus-mask inference."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from target_pose_regression.config import TargetPoseTrainingConfig
from target_pose_regression.dataset import TranslationStats
from target_pose_regression.model import TargetPoseRegressor, decode_predictions
from target_pose_regression.preprocessing import build_model_inputs, mask_bbox

from .config import InferenceConfig


class PoseEstimator:
    """Inference wrapper for the mask-conditioned target pose CNN."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model_config = TargetPoseTrainingConfig.from_dict(checkpoint["config"])
        self.model_config = model_config
        self.translation_stats = TranslationStats.from_dict(
            checkpoint["translation_stats"]
        )

        self.model = TargetPoseRegressor(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def estimate_pose(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Dict[str, float]:
        full_input, crop_input, geometry = self.preprocess(rgb_image, target_mask)
        outputs = self.model(full_input, crop_input, geometry)
        decoded = decode_predictions(outputs, self.translation_stats)
        translation = decoded["translation"]
        yaw_follow_deg = decoded["yaw_follow_deg"]

        return {
            "dx": float(translation[0, 0].item()),
            "dy": float(translation[0, 1].item()),
            "dyaw": float(yaw_follow_deg[0].item()),
            "yaw_follow_deg": float(yaw_follow_deg[0].item()),
        }

    def preprocess(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must be HxWx3")
        if target_mask.ndim != 2:
            raise ValueError("target_mask must be HxW")

        full_input, crop_input, geometry = build_model_inputs(
            rgb_array=rgb_image,
            mask_array=(target_mask > 0).astype(np.uint8),
            image_size=self.model_config.image_size,
            crop_size=self.model_config.crop_size,
            crop_context_scale=self.model_config.crop_context_scale,
        )
        return (
            full_input.unsqueeze(0).to(self.device),
            crop_input.unsqueeze(0).to(self.device),
            geometry.unsqueeze(0).to(self.device),
        )

    def get_bbox_from_mask(
        self,
        mask: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        return mask_bbox((mask > 0).astype(np.uint8))
