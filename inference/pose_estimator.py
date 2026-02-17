"""Pose estimator for vehicle pose estimation using trained CNN model."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from pose_estimation.config import PoseEstimationConfig
from pose_estimation.model import PoseEstimationCNN

from .config import InferenceConfig


class PoseEstimator:
    """Pose estimator that uses trained CNN model.

    Loads a trained checkpoint and provides inference functionality
    for estimating target vehicle pose (dx, dy, dyaw) from RGB image
    and binary mask.

    Attributes:
        config: Inference configuration.
        model: Loaded pose estimation model.
        device: Torch device for inference.
        transform: Image preprocessing transforms.
        pose_mean: Mean values for pose denormalization.
        pose_std: Std values for pose denormalization.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize the pose estimator.

        Args:
            config: Inference configuration with model paths and settings.
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[PoseEstimator] Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Setup transforms
        self.transform = self._create_transform()

        # Setup pose normalization tensors
        self._setup_normalization()

    def _load_model(self) -> nn.Module:
        """Load the trained pose estimation model.

        Returns:
            Loaded model moved to device.
        """
        # Create model config from inference config
        model_config = PoseEstimationConfig(
            backbone=self.config.backbone,
            bbox_mode=self.config.bbox_mode,
            image_size=self.config.model_image_size,
            predict_dx=self.config.predict_dx,
            predict_dy=self.config.predict_dy,
            predict_dz=self.config.predict_dz,
            predict_yaw=self.config.predict_yaw,
            pose_mean=self.config.pose_mean,
            pose_std=self.config.pose_std,
            pretrained=False,  # We'll load weights from checkpoint
        )

        # Create model
        model = PoseEstimationCNN(model_config)

        # Load checkpoint
        print(f"[PoseEstimator] Loading checkpoint: {self.config.checkpoint_path}")
        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        print("[PoseEstimator] Model loaded successfully")

        return model

    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transforms.

        Returns:
            Composed transforms for inference.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _setup_normalization(self) -> None:
        """Setup pose normalization tensors."""
        # Get indices for predicted components
        indices = []
        if self.config.predict_dx:
            indices.append(0)
        if self.config.predict_dy:
            indices.append(1)
        if self.config.predict_dz:
            indices.append(2)
        if self.config.predict_yaw:
            indices.append(3)

        # Extract mean and std for predicted components
        full_mean = torch.tensor(self.config.pose_mean, dtype=torch.float32)
        full_std = torch.tensor(self.config.pose_std, dtype=torch.float32)

        self.pose_mean = full_mean[indices].to(self.device)
        self.pose_std = full_std[indices].to(self.device)

        print(f"[PoseEstimator] Pose mean: {self.pose_mean}")
        print(f"[PoseEstimator] Pose std: {self.pose_std}")

    def preprocess(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> torch.Tensor:
        """Preprocess RGB image and mask for model input.

        Args:
            rgb_image: RGB image as HxWx3 numpy array (uint8, 0-255).
            target_mask: Binary mask as HxW numpy array (0 or 1).

        Returns:
            Preprocessed input tensor [1, 4, H, W].
        """
        target_h, target_w = self.config.model_image_size

        # Resize RGB image
        pil_image = Image.fromarray(rgb_image)
        pil_image = pil_image.resize(
            (target_w, target_h),
            Image.Resampling.BILINEAR,
        )

        # Apply transforms to RGB
        rgb_tensor = self.transform(pil_image)

        # Resize and process mask
        if target_mask is not None and target_mask.sum() > 0:
            # Resize mask
            mask_pil = Image.fromarray(
                (target_mask * 255).astype(np.uint8)
            )
            mask_pil = mask_pil.resize(
                (target_w, target_h),
                Image.Resampling.NEAREST,
            )
            mask_tensor = torch.tensor(
                np.array(mask_pil) / 255.0,
                dtype=torch.float32,
            ).unsqueeze(0)
        else:
            # Empty mask if no target detected
            mask_tensor = torch.zeros(
                1, target_h, target_w,
                dtype=torch.float32,
            )

        # Concatenate RGB and mask as 4-channel input
        input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor.to(self.device)

    def denormalize_pose(self, normalized_pose: torch.Tensor) -> torch.Tensor:
        """Denormalize pose predictions to actual metric values.

        Args:
            normalized_pose: Normalized pose tensor [B, num_outputs].

        Returns:
            Denormalized pose tensor [B, num_outputs].
        """
        return normalized_pose * self.pose_std + self.pose_mean

    @torch.no_grad()
    def estimate_pose(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Estimate target vehicle pose from RGB image and mask.

        Args:
            rgb_image: RGB image as HxWx3 numpy array.
            target_mask: Binary mask of target vehicle as HxW array.

        Returns:
            Dictionary with pose estimates:
                - dx: Longitudinal distance in meters (+x forward)
                - dy: Lateral offset in meters (+y right)
                - dyaw: Relative yaw in degrees
        """
        # Preprocess input
        input_tensor = self.preprocess(rgb_image, target_mask)

        # Run inference
        normalized_pred = self.model(input_tensor)

        # Denormalize to actual values
        pose_pred = self.denormalize_pose(normalized_pred)

        # Extract values
        pose_np = pose_pred.cpu().numpy()[0]

        result = {}
        idx = 0

        if self.config.predict_dx:
            result["dx"] = float(pose_np[idx])
            idx += 1

        if self.config.predict_dy:
            result["dy"] = float(pose_np[idx])
            idx += 1

        if self.config.predict_dz:
            result["dz"] = float(pose_np[idx])
            idx += 1

        if self.config.predict_yaw:
            result["dyaw"] = float(pose_np[idx])
            idx += 1

        return result

    def get_bbox_from_mask(
        self,
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Extract bounding box from binary mask.

        Args:
            mask: Binary mask as HxW numpy array.

        Returns:
            Tuple (x1, y1, x2, y2) or None if mask is empty.
        """
        if mask.sum() == 0:
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(x1), int(y1), int(x2), int(y2))
