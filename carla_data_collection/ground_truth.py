"""Ground-truth helpers shared by capture, build, and benchmarking."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .utils import wrap_angle_deg, wrap_angle_rad


def canonicalize_follow_yaw_deg(yaw_deg: float) -> float:
    """Map a relative yaw to the forward-facing pursuit branch in [-90, 90]."""
    yaw = wrap_angle_deg(float(yaw_deg))
    if yaw > 90.0:
        yaw -= 180.0
    elif yaw < -90.0:
        yaw += 180.0
    return float(yaw)


def actor_is_follow_valid(actor: Mapping[str, Any], config: Any) -> bool:
    """Return whether a target actor fits the pursuit-style follow regime."""
    dx_m = float(actor["dx_m"])
    dy_m = float(actor["dy_m"])
    yaw_deg = wrap_angle_deg(float(actor["yaw_deg"]))
    return (
        dx_m > 0.0
        and abs(dy_m) <= float(config.follow_lateral_limit_m)
        and abs(yaw_deg) <= float(config.follow_yaw_limit_deg)
    )


def compute_relative_pose_from_transforms(
    ego_location: Mapping[str, float],
    ego_yaw_deg: float,
    target_location: Mapping[str, float],
    target_yaw_deg: float,
    target_half_height_m: float,
    lidar_z_offset: float,
) -> Dict[str, float]:
    """Compute target pose in ego LiDAR coordinates."""
    dx_world = target_location["x"] - ego_location["x"]
    dy_world = target_location["y"] - ego_location["y"]
    dz_world = target_location["z"] - ego_location["z"]

    ego_yaw = math.radians(ego_yaw_deg)
    target_yaw = math.radians(target_yaw_deg)

    cos_e = math.cos(-ego_yaw)
    sin_e = math.sin(-ego_yaw)

    dx_ego = cos_e * dx_world - sin_e * dy_world
    dy_ego = sin_e * dx_world + cos_e * dy_world
    dz_lidar = dz_world - lidar_z_offset + target_half_height_m
    rel_yaw = wrap_angle_rad(target_yaw - ego_yaw)

    return {
        "dx_m": float(dx_ego),
        "dy_m": float(dy_ego),
        "dz_m": float(dz_lidar),
        "yaw_deg": float(math.degrees(rel_yaw)),
    }


def distance_bin_index(dx_m: float, bin_edges: Iterable[float]) -> int:
    """Get the configured distance bin index for a longitudinal distance."""
    bins = list(bin_edges)
    if len(bins) < 2:
        return 0

    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]
        if lower <= dx_m < upper:
            return idx

    return len(bins) - 2


def match_detections_to_actor_records(
    detections: List[Mapping[str, object]],
    actor_records: List[Mapping[str, object]],
    max_match_dist_m: float,
) -> Dict[int, Mapping[str, object]]:
    """Match detector outputs to visible GT actors using center distance."""
    pairs: List[tuple[float, int, int]] = []

    for det_idx, det in enumerate(detections):
        det_center = det.get("center", np.zeros(3, dtype=np.float32))
        if isinstance(det_center, list):
            det_center = np.asarray(det_center, dtype=np.float32)

        for actor_idx, actor in enumerate(actor_records):
            dx = float(det_center[0]) - float(actor["dx_m"])
            dy = float(det_center[1]) - float(actor["dy_m"])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= max_match_dist_m:
                pairs.append((dist, det_idx, actor_idx))

    pairs.sort(key=lambda item: item[0])

    matched_dets = set()
    matched_actors = set()
    matches: Dict[int, Mapping[str, object]] = {}

    for _, det_idx, actor_idx in pairs:
        if det_idx in matched_dets or actor_idx in matched_actors:
            continue
        actor_id = int(actor_records[actor_idx]["actor_id"])
        matches[actor_id] = detections[det_idx]
        matched_dets.add(det_idx)
        matched_actors.add(actor_idx)

    return matches


def compute_pose_errors(
    actor_records: List[Mapping[str, object]],
    matches: Mapping[int, Mapping[str, object]],
) -> Dict[str, float]:
    """Compute aggregate detection errors for benchmark reporting."""
    errors_dx: List[float] = []
    errors_dy: List[float] = []
    errors_dz: List[float] = []
    errors_yaw: List[float] = []
    errors_yaw_mod_180: List[float] = []

    actor_lookup = {int(actor["actor_id"]): actor for actor in actor_records}
    for actor_id, det in matches.items():
        actor = actor_lookup.get(int(actor_id))
        if actor is None:
            continue

        center = det["center"]
        if isinstance(center, list):
            center = np.asarray(center, dtype=np.float32)

        errors_dx.append(float(center[0]) - float(actor["dx_m"]))
        errors_dy.append(float(center[1]) - float(actor["dy_m"]))
        errors_dz.append(float(center[2]) - float(actor["dz_m"]))
        yaw_error = float(
            wrap_angle_deg(float(det["yaw_deg"]) - float(actor["yaw_deg"]))
        )
        errors_yaw.append(yaw_error)
        errors_yaw_mod_180.append(
            float(
                min(
                    abs(yaw_error),
                    abs(wrap_angle_deg(yaw_error - 180.0)),
                    abs(wrap_angle_deg(yaw_error + 180.0)),
                )
            )
        )

    if not errors_dx:
        return {
            "matched_samples": 0.0,
            "mae_dx_m": 0.0,
            "mae_dy_m": 0.0,
            "mae_dz_m": 0.0,
            "mae_yaw_deg": 0.0,
            "mae_yaw_deg_mod_180": 0.0,
        }

    return {
        "matched_samples": float(len(errors_dx)),
        "mae_dx_m": float(np.mean(np.abs(errors_dx))),
        "mae_dy_m": float(np.mean(np.abs(errors_dy))),
        "mae_dz_m": float(np.mean(np.abs(errors_dz))),
        "mae_yaw_deg": float(np.mean(np.abs(errors_yaw))),
        "mae_yaw_deg_mod_180": float(np.mean(np.abs(errors_yaw_mod_180))),
    }
