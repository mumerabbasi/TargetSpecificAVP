"""PyTorch Dataset for pose estimation from RGB images and bounding boxes."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import PoseEstimationConfig


class PoseEstimationDataset(Dataset):
    """Dataset for pose estimation from RGB images and bounding boxes.

    This dataset loads CSV entries and finds corresponding RGB images.
    Each sample contains an RGB image, bounding box information, and
    ground truth or predicted pose labels.

    Attributes:
        config: Configuration object with dataset parameters.
        df: DataFrame containing pose annotations.
        transform: Image transformations to apply.
        indices: List of valid indices after filtering.
    """

    def __init__(
        self,
        config: PoseEstimationConfig,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True,
    ):
        """Initialize the dataset.

        Args:
            config: Configuration object.
            df: DataFrame with pose annotations (already filtered for split).
            transform: Optional image transformations.
            is_training: Whether this is for training (enables augmentation).
        """
        self.config = config
        self.df = df.reset_index(drop=True)
        self.is_training = is_training

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

        # Skip image validation - too slow on remote storage
        # Images are assumed to exist since they were generated with the CSV

        # Compute normalization statistics if needed
        self._setup_normalization()

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations.

        Returns:
            Compose object with default transforms.
        """
        transform_list = []

        if self.is_training:
            transform_list.extend([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        return transforms.Compose(transform_list)

    def _get_image_path(self, frame_id: int) -> str:
        """Get the path to an RGB image.

        Args:
            frame_id: Frame ID from the CSV.

        Returns:
            Full path to the image file.
        """
        return os.path.join(
            self.config.images_dir,
            f"rgb_{frame_id:05d}.png"
        )

    def _setup_normalization(self) -> None:
        """Set up pose normalization statistics."""
        # Use config values or compute from data
        self.pose_mean = torch.tensor(
            self.config.pose_mean, dtype=torch.float32
        )
        self.pose_std = torch.tensor(
            self.config.pose_std, dtype=torch.float32
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary containing:
                - image: Processed image tensor [C, H, W]
                - bbox: Bounding box coordinates [4] (x1, y1, x2, y2)
                - bbox_mask: Binary mask of bbox (if bbox_mode includes mask)
                - pose: Target pose values [num_outputs]
                - pose_raw: Un-normalized pose values
                - frame_id: Frame ID for reference
        """
        row = self.df.iloc[idx]

        # Load image
        frame_id = int(row["frame_id"])
        img_path = self._get_image_path(frame_id)
        image = Image.open(img_path).convert("RGB")

        # Get bounding box (in original image coordinates)
        bbox = np.array([
            row["bbox_x1"],
            row["bbox_y1"],
            row["bbox_x2"],
            row["bbox_y2"]
        ], dtype=np.float32)

        # Get pose (GT or predicted based on config)
        if self.config.use_gt_poses:
            pose_raw = self._get_gt_pose(row)
        else:
            pose_raw = self._get_pred_pose(row)

        # Process image based on bbox_mode
        processed_data = self._process_image_and_bbox(image, bbox)

        # Normalize pose
        pose_normalized = self._normalize_pose(pose_raw)

        return {
            "image": processed_data["image"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "bbox_normalized": processed_data.get(
                "bbox_normalized",
                torch.zeros(4, dtype=torch.float32)
            ),
            "bbox_mask": processed_data.get(
                "bbox_mask",
                torch.zeros(1, dtype=torch.float32)
            ),
            "pose": pose_normalized,
            "pose_raw": pose_raw,
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }

    def _get_gt_pose(self, row: pd.Series) -> torch.Tensor:
        """Extract ground truth pose from row.

        Args:
            row: DataFrame row.

        Returns:
            Tensor with pose values based on config.
        """
        pose = []
        if self.config.predict_dx:
            pose.append(row["gt_dx_m"])
        if self.config.predict_dy:
            pose.append(row["gt_dy_m"])
        if self.config.predict_dz:
            pose.append(row["gt_dz_m"])
        if self.config.predict_yaw:
            pose.append(row["gt_yaw_deg"])

        return torch.tensor(pose, dtype=torch.float32)

    def _get_pred_pose(self, row: pd.Series) -> torch.Tensor:
        """Extract predicted pose from row (from 3D detector).

        Args:
            row: DataFrame row.

        Returns:
            Tensor with pose values based on config.
        """
        pose = []
        if self.config.predict_dx:
            pose.append(row["pred_dx_m"])
        if self.config.predict_dy:
            pose.append(row["pred_dy_m"])
        if self.config.predict_dz:
            pose.append(row["pred_dz_m"])
        if self.config.predict_yaw:
            pose.append(row["pred_yaw_deg"])

        return torch.tensor(pose, dtype=torch.float32)

    def _normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Normalize pose values.

        Args:
            pose: Raw pose tensor.

        Returns:
            Normalized pose tensor.
        """
        # Select only the dimensions we're predicting
        indices = []
        if self.config.predict_dx:
            indices.append(0)
        if self.config.predict_dy:
            indices.append(1)
        if self.config.predict_dz:
            indices.append(2)
        if self.config.predict_yaw:
            indices.append(3)

        mean = self.pose_mean[indices]
        std = self.pose_std[indices]

        return (pose - mean) / std

    def denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Denormalize pose values.

        Args:
            pose: Normalized pose tensor.

        Returns:
            Raw pose tensor.
        """
        indices = []
        if self.config.predict_dx:
            indices.append(0)
        if self.config.predict_dy:
            indices.append(1)
        if self.config.predict_dz:
            indices.append(2)
        if self.config.predict_yaw:
            indices.append(3)

        mean = self.pose_mean[indices].to(pose.device)
        std = self.pose_std[indices].to(pose.device)

        return pose * std + mean

    def _process_image_and_bbox(
        self,
        image: Image.Image,
        bbox: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Process image and bounding box based on bbox_mode.

        Args:
            image: PIL Image.
            bbox: Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Dictionary with processed tensors.
        """
        orig_w, orig_h = image.size

        if self.config.bbox_mode == "crop":
            return self._process_crop_mode(image, bbox)
        elif self.config.bbox_mode == "mask":
            return self._process_mask_mode(image, bbox, orig_w, orig_h)
        elif self.config.bbox_mode == "numeric":
            return self._process_numeric_mode(image, bbox, orig_w, orig_h)
        elif self.config.bbox_mode == "both":
            return self._process_both_mode(image, bbox, orig_w, orig_h)
        else:
            raise ValueError(f"Unknown bbox_mode: {self.config.bbox_mode}")

    def _process_crop_mode(
        self,
        image: Image.Image,
        bbox: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Crop image to bounding box region.

        Args:
            image: PIL Image.
            bbox: Bounding box coordinates.

        Returns:
            Dictionary with cropped and resized image.
        """
        x1, y1, x2, y2 = bbox.astype(int)

        # Add some padding around the bbox
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.width, x2 + pad)
        y2 = min(image.height, y2 + pad)

        # Crop and resize
        cropped = image.crop((x1, y1, x2, y2))
        resized = cropped.resize(
            self.config.image_size,
            Image.Resampling.BILINEAR
        )

        # Apply transforms
        img_tensor = self.transform(resized)

        return {"image": img_tensor}

    def _process_mask_mode(
        self,
        image: Image.Image,
        bbox: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> Dict[str, torch.Tensor]:
        """Create binary mask and resize full image.

        Args:
            image: PIL Image.
            bbox: Bounding box coordinates.
            orig_w: Original image width.
            orig_h: Original image height.

        Returns:
            Dictionary with resized image and mask.
        """
        target_h, target_w = self.config.image_size

        # Resize image
        resized = image.resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        img_tensor = self.transform(resized)

        # Scale bbox to new size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        x1 = int(bbox[0] * scale_x)
        y1 = int(bbox[1] * scale_y)
        x2 = int(bbox[2] * scale_x)
        y2 = int(bbox[3] * scale_y)

        # Create binary mask
        mask = torch.zeros(1, target_h, target_w, dtype=torch.float32)
        mask[0, y1:y2, x1:x2] = 1.0

        # Concatenate mask as 4th channel
        img_with_mask = torch.cat([img_tensor, mask], dim=0)

        return {"image": img_with_mask, "bbox_mask": mask}

    def _process_numeric_mode(
        self,
        image: Image.Image,
        bbox: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> Dict[str, torch.Tensor]:
        """Resize image and provide normalized bbox coordinates.

        Args:
            image: PIL Image.
            bbox: Bounding box coordinates.
            orig_w: Original image width.
            orig_h: Original image height.

        Returns:
            Dictionary with resized image and normalized bbox.
        """
        target_h, target_w = self.config.image_size

        # Resize image
        resized = image.resize(
            (target_w, target_h), Image.Resampling.BILINEAR
        )
        img_tensor = self.transform(resized)

        # Normalize bbox to [0, 1]
        bbox_normalized = torch.tensor([
            bbox[0] / orig_w,
            bbox[1] / orig_h,
            bbox[2] / orig_w,
            bbox[3] / orig_h,
        ], dtype=torch.float32)

        return {"image": img_tensor, "bbox_normalized": bbox_normalized}

    def _process_both_mode(
        self,
        image: Image.Image,
        bbox: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> Dict[str, torch.Tensor]:
        """Combine mask and numeric modes.

        Args:
            image: PIL Image.
            bbox: Bounding box coordinates.
            orig_w: Original image width.
            orig_h: Original image height.

        Returns:
            Dictionary with image+mask and normalized bbox.
        """
        mask_result = self._process_mask_mode(image, bbox, orig_w, orig_h)
        numeric_result = self._process_numeric_mode(image, bbox, orig_w, orig_h)

        return {
            "image": mask_result["image"],
            "bbox_mask": mask_result["bbox_mask"],
            "bbox_normalized": numeric_result["bbox_normalized"],
        }


def compute_pose_statistics(
    df: pd.DataFrame,
    use_gt: bool = True
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Compute mean and std of pose values from the dataset.

    Args:
        df: DataFrame with pose annotations.
        use_gt: Whether to use GT poses or predicted poses.

    Returns:
        Tuple of (mean, std) for (dx, dy, dz, yaw).
    """
    prefix = "gt" if use_gt else "pred"

    mean = (
        df[f"{prefix}_dx_m"].mean(),
        df[f"{prefix}_dy_m"].mean(),
        df[f"{prefix}_dz_m"].mean(),
        df[f"{prefix}_yaw_deg"].mean(),
    )

    std = (
        df[f"{prefix}_dx_m"].std(),
        df[f"{prefix}_dy_m"].std(),
        df[f"{prefix}_dz_m"].std(),
        df[f"{prefix}_yaw_deg"].std(),
    )

    return mean, std


def create_data_splits(
    df: pd.DataFrame,
    config: PoseEstimationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.

    Splits are stratified by town to ensure balanced representation.

    Args:
        df: Full DataFrame.
        config: Configuration with split ratios.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    np.random.seed(config.random_seed)

    # Get unique frame IDs (some frames may have multiple detections)
    unique_frames = df["frame_id"].unique()
    np.random.shuffle(unique_frames)

    n_total = len(unique_frames)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)

    train_frames = set(unique_frames[:n_train])
    val_frames = set(unique_frames[n_train:n_train + n_val])
    test_frames = set(unique_frames[n_train + n_val:])

    train_df = df[df["frame_id"].isin(train_frames)].copy()
    val_df = df[df["frame_id"].isin(val_frames)].copy()
    test_df = df[df["frame_id"].isin(test_frames)].copy()

    print("Data splits:")
    print(f"  Train: {len(train_df)} samples ({len(train_frames)} frames)")
    print(f"  Val:   {len(val_df)} samples ({len(val_frames)} frames)")
    print(f"  Test:  {len(test_df)} samples ({len(test_frames)} frames)")

    return train_df, val_df, test_df
