"""Utility functions for coordinate transforms and data processing."""

import math
from typing import Any, Dict, Tuple

import numpy as np


def wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def get_camera_intrinsic(width: int, height: int, fov: float) -> np.ndarray:
    """
    Compute camera intrinsic matrix from FOV.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Horizontal field of view in degrees.

    Returns:
        3x3 camera intrinsic matrix.
    """
    focal = width / (2.0 * np.tan(np.radians(fov / 2.0)))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def project_lidar_to_camera(
    points_lidar: np.ndarray,
    lidar_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int,
    img_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points to camera image coordinates.

    Args:
        points_lidar: Nx4 LiDAR points (x, y, z, intensity).
        lidar_to_camera: 4x4 transform from LiDAR to camera frame.
        intrinsic: 3x3 camera intrinsic matrix.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        uv: Nx2 pixel coordinates.
        valid_mask: N boolean mask for valid projections.
        depths: N depth values.
    """
    n_points = points_lidar.shape[0]
    points_hom = np.hstack([points_lidar[:, :3], np.ones((n_points, 1))])
    points_cam = (lidar_to_camera @ points_hom.T).T

    x_cam = points_cam[:, 1]
    y_cam = -points_cam[:, 2]
    z_cam = points_cam[:, 0]

    valid_depth = z_cam > 0.1

    u = (intrinsic[0, 0] * x_cam / z_cam) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * y_cam / z_cam) + intrinsic[1, 2]

    valid_bounds = (
        (u >= 0) & (u < img_width) &
        (v >= 0) & (v < img_height)
    )

    valid_mask = valid_depth & valid_bounds
    uv = np.stack([u, v], axis=1)

    return uv, valid_mask, z_cam


def filter_points_by_mask(
    points_lidar: np.ndarray,
    uv: np.ndarray,
    valid_mask: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Filter LiDAR points that project to a binary mask.

    Args:
        points_lidar: Nx4 LiDAR points.
        uv: Nx2 pixel coordinates.
        valid_mask: N boolean mask for valid projections.
        binary_mask: HxW binary mask.

    Returns:
        Filtered Mx4 LiDAR points.
    """
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

    if len(filtered_indices) == 0:
        return np.array([]).reshape(0, 4)

    return points_lidar[filtered_indices]


def carla_to_nuscenes_points(points_carla: np.ndarray) -> np.ndarray:
    """
    Convert CARLA LiDAR points to nuScenes format.

    CARLA uses x-forward, y-right, z-up.
    nuScenes uses x-right, y-forward, z-up.

    Args:
        points_carla: Nx4 CARLA LiDAR points (x, y, z, intensity).

    Returns:
        Nx5 nuScenes format points (x, y, z, intensity, ring_index).
    """
    n_points = points_carla.shape[0]
    if n_points == 0:
        return np.zeros((0, 5), dtype=np.float32)

    points_nus = np.zeros((n_points, 5), dtype=np.float32)
    points_nus[:, 0] = points_carla[:, 0]
    points_nus[:, 1] = -points_carla[:, 1]
    points_nus[:, 2] = points_carla[:, 2]
    points_nus[:, 3] = np.clip(points_carla[:, 3], 0.0, 1.0)
    points_nus[:, 4] = 0.0  # Ring index (not used)

    return points_nus


def nuscenes_to_carla_box(box_nus: np.ndarray) -> Dict[str, Any]:
    """
    Convert nuScenes 3D box to CARLA LiDAR frame.

    Args:
        box_nus: nuScenes box [cx, cy, cz, l, w, h, yaw].

    Returns:
        Dictionary with center, dims, yaw in CARLA frame.
    """
    cx_carla = box_nus[0]
    cy_carla = -box_nus[1]
    cz_carla = box_nus[2]
    yaw_carla = -box_nus[6]

    return {
        "center": np.array([cx_carla, cy_carla, cz_carla], dtype=np.float32),
        "dims": np.array([box_nus[3], box_nus[4], box_nus[5]], dtype=np.float32),
        "yaw": float(wrap_angle_rad(yaw_carla)),
    }


def compute_lidar_to_camera_transform(
    lidar_matrix: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute 4x4 transform from LiDAR frame to camera frame.

    Args:
        lidar_matrix: 4x4 LiDAR pose matrix.
        camera_matrix: 4x4 camera pose matrix.

    Returns:
        4x4 transform from LiDAR to camera.
    """
    camera_inv = np.linalg.inv(camera_matrix)
    return camera_inv @ lidar_matrix
