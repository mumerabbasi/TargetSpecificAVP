"""Geometry helpers for target-specific pursuit inference."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to the [-180, 180] range."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def canonicalize_follow_yaw_deg(yaw_deg: float) -> float:
    """Fold a relative yaw into the forward-follow branch [-90, 90]."""
    yaw_deg = wrap_angle_deg(yaw_deg)
    if yaw_deg > 90.0:
        yaw_deg -= 180.0
    elif yaw_deg < -90.0:
        yaw_deg += 180.0
    return float(yaw_deg)


def get_camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Return a simple pinhole intrinsic matrix for the CARLA RGB camera."""
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_world_points_to_image(
    world_points_xyz: np.ndarray,
    camera_world_matrix: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points into image coordinates."""
    camera_inv = np.linalg.inv(camera_world_matrix)
    points_h = np.hstack(
        [world_points_xyz[:, :3], np.ones((world_points_xyz.shape[0], 1))]
    )
    points_cam = (camera_inv @ points_h.T).T

    x_cam = points_cam[:, 1]
    y_cam = -points_cam[:, 2]
    z_cam = points_cam[:, 0]
    valid_depth = z_cam > 0.1

    u = intrinsic[0, 0] * x_cam / np.maximum(z_cam, 1e-6) + intrinsic[0, 2]
    v = intrinsic[1, 1] * y_cam / np.maximum(z_cam, 1e-6) + intrinsic[1, 2]
    valid_bounds = (
        (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )
    return np.stack([u, v], axis=1), valid_depth & valid_bounds


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an xyxy bounding box from a binary mask."""
    if mask.size == 0 or not np.any(mask):
        return None
    ys, xs = np.where(mask.astype(bool))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    pad_px: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Pad and clamp an xyxy bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, int(x1) - int(pad_px)),
        max(0, int(y1) - int(pad_px)),
        min(int(image_width) - 1, int(x2) + int(pad_px)),
        min(int(image_height) - 1, int(y2) + int(pad_px)),
    )


def _bbox_area(bbox_xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def bbox_iou(
    bbox_a: Optional[Tuple[int, int, int, int]],
    bbox_b: Optional[Tuple[int, int, int, int]],
) -> float:
    """Compute IoU between two xyxy boxes."""
    if bbox_a is None or bbox_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0

    inter = _bbox_area((ix1, iy1, ix2, iy2))
    union = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)
