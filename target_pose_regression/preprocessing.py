"""Shared preprocessing utilities for target pose regression."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225]


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an exclusive-end bounding box for a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expand_square_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale: float,
) -> Tuple[int, int, int, int]:
    """Expand a bbox into a square context crop clipped to the image."""
    x1, y1, x2, y2 = bbox
    box_width = max(float(x2 - x1), 1.0)
    box_height = max(float(y2 - y1), 1.0)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(box_width, box_height) * float(scale)
    side = max(side, 2.0)

    crop_x1 = int(round(center_x - side * 0.5))
    crop_y1 = int(round(center_y - side * 0.5))
    crop_x2 = int(round(center_x + side * 0.5))
    crop_y2 = int(round(center_y + side * 0.5))

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > image_width:
        shift = crop_x2 - image_width
        crop_x1 = max(0, crop_x1 - shift)
        crop_x2 = image_width
    if crop_y2 > image_height:
        shift = crop_y2 - image_height
        crop_y1 = max(0, crop_y1 - shift)
        crop_y2 = image_height

    if crop_x2 <= crop_x1:
        crop_x2 = min(image_width, crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(image_height, crop_y1 + 1)

    return crop_x1, crop_y1, crop_x2, crop_y2


def build_geometry_features(mask: np.ndarray) -> torch.Tensor:
    """Encode coarse normalized geometry from a binary mask."""
    image_height, image_width = mask.shape
    bbox = mask_bbox(mask)
    if bbox is None:
        return torch.zeros(5, dtype=torch.float32)

    x1, y1, x2, y2 = bbox
    box_width = max(float(x2 - x1), 1.0)
    box_height = max(float(y2 - y1), 1.0)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    area_ratio = float(mask.sum()) / max(float(image_width * image_height), 1.0)

    return torch.tensor(
        [
            center_x / max(float(image_width), 1.0),
            center_y / max(float(image_height), 1.0),
            box_width / max(float(image_width), 1.0),
            box_height / max(float(image_height), 1.0),
            area_ratio,
        ],
        dtype=torch.float32,
    )


def build_model_inputs(
    rgb_array: np.ndarray,
    mask_array: np.ndarray,
    image_size: Tuple[int, int],
    crop_size: Tuple[int, int],
    crop_context_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build full-frame input, context crop input, and geometry features."""
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb_array must be HxWx3")
    if mask_array.ndim != 2:
        raise ValueError("mask_array must be HxW")

    binary_mask = (mask_array > 0).astype(np.uint8)
    geometry = build_geometry_features(binary_mask)

    full_input = _build_four_channel_input(rgb_array, binary_mask, image_size)

    bbox = mask_bbox(binary_mask)
    if bbox is None:
        crop_rgb = rgb_array
        crop_mask = binary_mask
    else:
        image_height, image_width = binary_mask.shape
        crop_bbox = expand_square_bbox(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            scale=crop_context_scale,
        )
        crop_rgb, crop_mask = _crop_rgb_and_mask(rgb_array, binary_mask, crop_bbox)

    crop_input = _build_four_channel_input(crop_rgb, crop_mask, crop_size)
    return full_input, crop_input, geometry


def _crop_rgb_and_mask(
    rgb_array: np.ndarray,
    mask_array: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    return rgb_array[y1:y2, x1:x2], mask_array[y1:y2, x1:x2]


def _build_four_channel_input(
    rgb_array: np.ndarray,
    mask_array: np.ndarray,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    width, height = int(image_size[1]), int(image_size[0])

    rgb_image = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    mask_image = Image.fromarray((mask_array > 0).astype(np.uint8) * 255, mode="L")

    rgb_image = rgb_image.resize((width, height), Image.Resampling.BILINEAR)
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)

    rgb_tensor = TF.to_tensor(rgb_image)
    rgb_tensor = TF.normalize(rgb_tensor, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD)

    mask_tensor = torch.from_numpy(
        (np.asarray(mask_image, dtype=np.float32) > 127).astype(np.float32)
    ).unsqueeze(0)

    return torch.cat([rgb_tensor, mask_tensor], dim=0)
