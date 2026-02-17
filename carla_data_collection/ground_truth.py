"""Ground truth computation and detection matching."""

import math
from typing import Dict, List, Optional

import carla

from .utils import wrap_angle_rad


def compute_gt_in_lidar_frame(
    target: carla.Vehicle,
    ego: carla.Vehicle,
    lidar_z_offset: float = 1.73,
) -> Dict[str, float]:
    """
    Compute ground truth pose of target in ego's LiDAR frame.

    Args:
        target: Target vehicle.
        ego: Ego vehicle.
        lidar_z_offset: Height of LiDAR sensor above ego origin.

    Returns:
        Dictionary with gt_dx, gt_dy, gt_dz, gt_yaw (in LiDAR frame).
    """
    target_tf = target.get_transform()
    ego_tf = ego.get_transform()
    target_bb = target.bounding_box

    # Target world position (center of bounding box)
    target_loc = target_tf.location
    target_yaw_world = math.radians(target_tf.rotation.yaw)

    # Ego world position and yaw
    ego_loc = ego_tf.location
    ego_yaw_world = math.radians(ego_tf.rotation.yaw)

    # Vector from ego to target in world frame
    dx_world = target_loc.x - ego_loc.x
    dy_world = target_loc.y - ego_loc.y
    dz_world = target_loc.z - ego_loc.z

    # Rotate into ego frame (CARLA: x-forward, y-right)
    cos_e = math.cos(-ego_yaw_world)
    sin_e = math.sin(-ego_yaw_world)

    dx_ego = cos_e * dx_world - sin_e * dy_world
    dy_ego = sin_e * dx_world + cos_e * dy_world

    # LiDAR frame has z-offset from ego origin
    dz_lidar = dz_world - lidar_z_offset + target_bb.extent.z

    # Relative yaw
    rel_yaw = wrap_angle_rad(target_yaw_world - ego_yaw_world)

    return {
        "gt_dx": dx_ego,
        "gt_dy": dy_ego,
        "gt_dz": dz_lidar,
        "gt_yaw": rel_yaw,
        "gt_length": target_bb.extent.x * 2,
        "gt_width": target_bb.extent.y * 2,
        "gt_height": target_bb.extent.z * 2,
    }


def match_detection_to_target(
    pred_dx: float,
    pred_dy: float,
    targets: List[carla.Vehicle],
    ego: carla.Vehicle,
    lidar_z_offset: float = 1.73,
    max_match_dist: float = 5.0,
) -> Optional[Dict[str, float]]:
    """
    Match a predicted detection to the closest target vehicle.

    Args:
        pred_dx, pred_dy: Predicted position in LiDAR frame.
        targets: List of target vehicles.
        ego: Ego vehicle.
        lidar_z_offset: Height of LiDAR sensor.
        max_match_dist: Maximum distance for a valid match.

    Returns:
        Ground truth dict if matched, None otherwise.
    """
    best_gt = None
    best_dist = float('inf')

    for target in targets:
        if not target.is_alive:
            continue

        gt = compute_gt_in_lidar_frame(target, ego, lidar_z_offset)
        dist = math.sqrt((pred_dx - gt["gt_dx"])**2 + (pred_dy - gt["gt_dy"])**2)

        if dist < best_dist and dist < max_match_dist:
            best_dist = dist
            best_gt = gt

    return best_gt
