"""Load a trained pose model and run RGB-plus-mask inference."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from pose_estimation.config import PoseEstimationConfig
from pose_estimation.dataset import TranslationStats, denormalize_translation
from pose_estimation.model import PoseEstimationCNN

from .config import InferenceConfig


class PoseEstimator:
    """Inference wrapper for the mask-conditioned target pose CNN."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=self.device,
            weights_only=False)
        model_config = PoseEstimationConfig.from_dict(checkpoint["config"])
        self.model_config = model_config
        self.translation_stats = TranslationStats.from_dict(
            checkpoint["translation_stats"])

        self.model = PoseEstimationCNN(model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def estimate_pose(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Dict[str, float]:
        input_tensor, mask_tensor, geometry_tensor = self.preprocess(
            rgb_image, target_mask)
        outputs = self.model(input_tensor, mask_tensor, geometry_tensor)

        translation = denormalize_translation(
            outputs["translation"], self.translation_stats)
        yaw_rad = torch.atan2(
            outputs["yaw_vector"][:, 0], outputs["yaw_vector"][:, 1])
        yaw_deg = torch.rad2deg(yaw_rad)

        return {
            "dx": float(translation[0, 0].item()),
            "dy": float(translation[0, 1].item()),
            "dyaw": float(yaw_deg[0].item()),
            "yaw_follow_deg": float(yaw_deg[0].item()),
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

        original_height, original_width = target_mask.shape
        target_width, target_height = self.model_config.image_size

        rgb_pil = Image.fromarray(rgb_image).resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )
        mask_pil = Image.fromarray(
            (target_mask > 0).astype(
                np.uint8) *
            255).resize(
            (target_width,
             target_height),
            Image.Resampling.NEAREST,
        )

        rgb_tensor = TF.to_tensor(rgb_pil)
        rgb_tensor = TF.normalize(rgb_tensor, self.rgb_mean, self.rgb_std)

        mask_array = (
            np.asarray(
                mask_pil,
                dtype=np.float32) > 127).astype(
            np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        geometry_tensor = self._build_geometry_tensor(
            target_mask, original_width, original_height)
        input_tensor = torch.cat(
            [rgb_tensor, mask_tensor], dim=0).unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
        geometry_tensor = geometry_tensor.unsqueeze(0).to(self.device)

        return input_tensor, mask_tensor, geometry_tensor

    def _build_geometry_tensor(
        self,
        target_mask: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> torch.Tensor:
        ys, xs = np.where(target_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return torch.zeros(5, dtype=torch.float32)

        x1 = float(xs.min())
        x2 = float(xs.max() + 1)
        y1 = float(ys.min())
        y2 = float(ys.max() + 1)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        mask_area = float((target_mask > 0).sum())

        return torch.tensor(
            [
                center_x / max(image_width, 1),
                center_y / max(image_height, 1),
                width / max(image_width, 1),
                height / max(image_height, 1),
                mask_area / max(image_width * image_height, 1),
            ],
            dtype=torch.float32,
        )

    def get_bbox_from_mask(
        self,
        mask: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(
            xs.max() + 1), int(ys.max() + 1)
