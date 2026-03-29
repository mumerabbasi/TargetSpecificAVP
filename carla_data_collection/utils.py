"""Utility helpers for geometry, masks, and file handling."""

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def get_camera_intrinsic(width: int, height: int, fov: float) -> np.ndarray:
    """Compute camera intrinsic matrix from horizontal FOV."""
    focal = width / (2.0 * np.tan(np.radians(fov / 2.0)))
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
        dtype=np.float64,
    )


def project_lidar_to_camera(
    points_lidar: np.ndarray,
    lidar_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int,
    img_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points into image space."""
    n_points = points_lidar.shape[0]
    points_hom = np.hstack([points_lidar[:, :3], np.ones((n_points, 1))])
    points_cam = (lidar_to_camera @ points_hom.T).T

    x_cam = points_cam[:, 1]
    y_cam = -points_cam[:, 2]
    z_cam = points_cam[:, 0]

    valid_depth = z_cam > 0.1
    u = (intrinsic[0, 0] * x_cam / z_cam) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * y_cam / z_cam) + intrinsic[1, 2]

    valid_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask = valid_depth & valid_bounds
    uv = np.stack([u, v], axis=1)

    return uv, valid_mask, z_cam


def project_world_points_to_image(
    world_points: np.ndarray,
    camera_world_matrix: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int,
    img_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D world points into image space."""
    camera_inv = np.linalg.inv(camera_world_matrix)
    n_points = world_points.shape[0]
    points_hom = np.hstack([world_points[:, :3], np.ones((n_points, 1))])
    points_cam = (camera_inv @ points_hom.T).T

    x_cam = points_cam[:, 1]
    y_cam = -points_cam[:, 2]
    z_cam = points_cam[:, 0]

    valid_depth = z_cam > 0.1
    u = (intrinsic[0, 0] * x_cam / z_cam) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * y_cam / z_cam) + intrinsic[1, 2]

    valid_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask = valid_depth & valid_bounds
    uv = np.stack([u, v], axis=1)

    return uv, valid_mask, z_cam


def filter_points_by_mask(
    points_lidar: np.ndarray,
    uv: np.ndarray,
    valid_mask: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Filter LiDAR points that project inside a binary mask."""
    filtered_indices = []
    for i in range(len(points_lidar)):
        if not valid_mask[i]:
            continue

        u, v = int(uv[i, 0]), int(uv[i, 1])
        if u < 0 or u >= binary_mask.shape[1]:
            continue
        if v < 0 or v >= binary_mask.shape[0]:
            continue
        if binary_mask[v, u]:
            filtered_indices.append(i)

    if not filtered_indices:
        return np.empty((0, 4), dtype=np.float32)

    return points_lidar[filtered_indices]


def carla_to_nuscenes_points(points_carla: np.ndarray) -> np.ndarray:
    """Convert CARLA LiDAR points to the nuScenes frame expected by detectors."""
    n_points = points_carla.shape[0]
    if n_points == 0:
        return np.zeros((0, 5), dtype=np.float32)

    points_nus = np.zeros((n_points, 5), dtype=np.float32)
    points_nus[:, 0] = points_carla[:, 0]
    points_nus[:, 1] = -points_carla[:, 1]
    points_nus[:, 2] = points_carla[:, 2]
    points_nus[:, 3] = np.clip(points_carla[:, 3], 0.0, 1.0)
    points_nus[:, 4] = 0.0
    return points_nus


def nuscenes_to_carla_box(box_nus: np.ndarray) -> Dict[str, Any]:
    """Convert a nuScenes-format 3D box to CARLA LiDAR coordinates."""
    cx_carla = box_nus[0]
    cy_carla = -box_nus[1]
    # MMDetection3D LiDAR boxes use bottom-center z; the downstream pose target
    # uses the 3D box center in ego LiDAR coordinates.
    cz_carla = box_nus[2] + (box_nus[5] * 0.5)
    yaw_carla = -box_nus[6]

    return {
        "center": np.array([cx_carla, cy_carla, cz_carla], dtype=np.float32),
        "dims": np.array([box_nus[3], box_nus[4], box_nus[5]], dtype=np.float32),
        "yaw": float(wrap_angle_rad(yaw_carla)),
        "yaw_deg": float(math.degrees(wrap_angle_rad(yaw_carla))),
    }


def compute_lidar_to_camera_transform(
    lidar_matrix: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Compute a LiDAR-to-camera rigid transform."""
    camera_inv = np.linalg.inv(camera_matrix)
    return camera_inv @ lidar_matrix


def binary_mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) for a binary mask or None if empty."""
    if mask.size == 0 or not np.any(mask):
        return None

    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_touches_edge(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    margin_px: int,
) -> bool:
    """Check whether a bbox is too close to the image boundary."""
    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin_px
        or y1 <= margin_px
        or x2 >= (width - 1 - margin_px)
        or y2 >= (height - 1 - margin_px)
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def deduplicate_mask_candidates(
    candidates: Iterable[Dict[str, Any]],
    iou_thr: float,
) -> list[Dict[str, Any]]:
    """Greedily suppress duplicate segmentation masks by score and IoU."""
    ordered = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    kept: list[Dict[str, Any]] = []
    for candidate in ordered:
        if all(mask_iou(candidate["mask"], prev["mask"]) < iou_thr for prev in kept):
            kept.append(candidate)
    return kept


def extract_instance_ids(instance_image: np.ndarray) -> np.ndarray:
    """Extract packed instance ids from a CARLA BGRA instance image."""
    return (
        instance_image[:, :, 0].astype(np.uint32)
        + instance_image[:, :, 1].astype(np.uint32) * 256
    )


def extract_semantic_tags(instance_image: np.ndarray) -> np.ndarray:
    """Extract semantic tags from a CARLA BGRA instance image."""
    return instance_image[:, :, 2]


def vehicle_instance_mask_from_array(
    instance_image: np.ndarray,
    instance_id: int,
    vehicle_semantic_tag: int,
) -> np.ndarray:
    """Return a binary mask for one vehicle instance in a saved raw frame."""
    semantic_tags = extract_semantic_tags(instance_image)
    instance_ids = extract_instance_ids(instance_image)
    return (semantic_tags == vehicle_semantic_tag) & (instance_ids == instance_id)


def ensure_dir(path: str) -> None:
    """Create a directory if needed."""
    os.makedirs(path, exist_ok=True)


def link_or_copy(src: str, dst: str) -> None:
    """Hard-link a file when possible, otherwise fall back to copy."""
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_binary_mask(mask: np.ndarray, path: str) -> None:
    """Save a binary mask as an 8-bit PNG."""
    ensure_dir(os.path.dirname(path))
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(path, mask_u8)


def relative_path(path: str, root: str) -> str:
    """Compute a normalized relative path for CSV outputs."""
    return os.path.relpath(path, root).replace(os.sep, "/")
