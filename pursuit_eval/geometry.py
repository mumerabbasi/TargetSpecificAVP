"""Geometry helpers shared by the new pursuit evaluation runtime and worker."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap degrees to [-180, 180]."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap radians to [-pi, pi]."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def canonicalize_follow_yaw_deg(yaw_deg: float) -> float:
    """Fold a pursuit-style relative yaw into the forward branch [-90, 90]."""
    yaw_deg = wrap_angle_deg(yaw_deg)
    if yaw_deg > 90.0:
        yaw_deg -= 180.0
    elif yaw_deg < -90.0:
        yaw_deg += 180.0
    return float(yaw_deg)


def get_camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Return a pinhole intrinsic matrix for the CARLA RGB camera."""
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_lidar_points_to_image(
    points_xyz: np.ndarray,
    lidar_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR-frame 3D points into image pixel coordinates."""
    if points_xyz.size == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)

    points_h = np.hstack([points_xyz[:, :3], np.ones((points_xyz.shape[0], 1))])
    points_cam = (lidar_to_camera @ points_h.T).T

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
    valid = valid_depth & valid_bounds
    return np.stack([u, v], axis=1), valid, z_cam


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return xyxy bbox from a binary mask or None when empty."""
    if mask.size == 0 or not np.any(mask):
        return None
    ys, xs = np.where(mask.astype(bool))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_area(bbox_xyxy: Tuple[int, int, int, int]) -> int:
    """Return inclusive bbox area in pixels."""
    x1, y1, x2, y2 = bbox_xyxy
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    pad_px: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Pad and clamp an xyxy bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, int(x1) - pad_px),
        max(0, int(y1) - pad_px),
        min(image_width - 1, int(x2) + pad_px),
        min(image_height - 1, int(y2) + pad_px),
    )


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

    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - inter
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


def build_oriented_box_corners(
    center_xyz: np.ndarray,
    dims_xyz: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """Return 8 LiDAR-frame corners for a centered 3D box."""
    dx, dy, dz = [float(v) for v in dims_xyz[:3]]
    hx = dx * 0.5
    hy = dy * 0.5
    hz = dz * 0.5
    corners = np.array(
        [
            [hx, hy, hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [hx, hy, -hz],
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
        ],
        dtype=np.float64,
    )

    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    rot = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return corners @ rot.T + center_xyz.reshape(1, 3)


def project_detection_box_to_image(
    center_xyz: np.ndarray,
    dims_xyz: np.ndarray,
    yaw_rad: float,
    lidar_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Project an oriented 3D box into the image as bbox and raster mask."""
    corners = build_oriented_box_corners(center_xyz, dims_xyz, yaw_rad)
    uv, valid, depth = project_lidar_points_to_image(
        corners,
        lidar_to_camera,
        intrinsic,
        image_width,
        image_height,
    )
    if valid.sum() < 4:
        return None, None

    uv_valid = uv[valid]
    x1 = int(np.clip(np.floor(uv_valid[:, 0].min()), 0, image_width - 1))
    y1 = int(np.clip(np.floor(uv_valid[:, 1].min()), 0, image_height - 1))
    x2 = int(np.clip(np.ceil(uv_valid[:, 0].max()), 0, image_width - 1))
    y2 = int(np.clip(np.ceil(uv_valid[:, 1].max()), 0, image_height - 1))
    bbox = (x1, y1, x2, y2)

    hull = cv2.convexHull(uv_valid.astype(np.float32)).astype(np.int32)
    raster = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillConvexPoly(raster, hull, 1)
    return bbox, raster.astype(bool)
