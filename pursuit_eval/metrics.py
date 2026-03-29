"""Metrics for fresh MPC pursuit evaluation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .config import PursuitEvalConfig
from .geometry import canonicalize_follow_yaw_deg, wrap_angle_deg


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _yaw_mod_180_abs(pred_yaw_deg: float, gt_yaw_deg: float) -> float:
    yaw_error = abs(float(wrap_angle_deg(pred_yaw_deg - gt_yaw_deg)))
    return float(
        min(
            yaw_error,
            abs(wrap_angle_deg(yaw_error - 180.0)),
            abs(wrap_angle_deg(yaw_error + 180.0)),
        )
    )


class PursuitMetrics:
    """Accumulate per-frame pose and pursuit metrics and write a JSON report."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        self.config = config
        self.rows: List[Dict[str, object]] = []

    def add_row(self, row: Dict[str, object]) -> None:
        self.rows.append(dict(row))

    def _pose_summary(
            self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        pose_rows = [
            row for row in rows if bool(
                row.get(
                    "pose_available",
                    False))]
        fresh_rows = [
            row for row in pose_rows if not bool(
                row.get(
                    "pose_stale",
                    False))]
        mask_rows = [
            row for row in rows if bool(
                row.get(
                    "mask_available",
                    False))]
        detector_rows = [
            row for row in rows if bool(
                row.get(
                    "detector_pose_available",
                    False))]
        reseed_rows = [
            row for row in rows if bool(
                row.get(
                    "bbox_reseed_used",
                    False))]

        def summarize_pose(
                subset: List[Dict[str, object]]) -> Dict[str, object]:
            dx_abs = [abs(float(row["pose_dx_error_m"])) for row in subset]
            dy_abs = [abs(float(row["pose_dy_error_m"])) for row in subset]
            yaw_abs = [abs(float(row["pose_yaw_error_deg"])) for row in subset]
            yaw180_abs = [float(row["pose_yaw_mod_180_error_deg"])
                          for row in subset]
            yaw_follow_abs = [float(row["pose_follow_yaw_error_deg"])
                              for row in subset]
            mask_ious = [
                float(row["mask_iou"])
                for row in subset
                if row.get("mask_iou") is not None
            ]
            latencies = [
                float(row["pose_latency_ms"])
                for row in subset
                if row.get("pose_latency_ms") is not None
            ]
            return {
                "frames": len(subset),
                "dx_abs_m": _stats(dx_abs),
                "dy_abs_m": _stats(dy_abs),
                "yaw_abs_deg": _stats(yaw_abs),
                "yaw_mod_180_abs_deg": _stats(yaw180_abs),
                "follow_yaw_abs_deg": _stats(yaw_follow_abs),
                "mask_iou": _stats(mask_ious),
                "latency_ms": _stats(latencies),
            }

        return {
            "availability_ratio": float(len(pose_rows) / max(len(rows), 1)),
            "fresh_ratio": float(len(fresh_rows) / max(len(rows), 1)),
            "mask_available_ratio": float(len(mask_rows) / max(len(rows), 1)),
            "detector_match_ratio": float(len(detector_rows) / max(len(rows), 1)),
            "stale_ratio": float(
                sum(bool(row.get("pose_stale", False)) for row in pose_rows)
                / max(len(rows), 1)
            ),
            "bbox_bootstrap_frames": int(
                sum(bool(row.get("bbox_bootstrap_used", False)) for row in rows)
            ),
            "bbox_reseed_requested_frames": int(
                sum(bool(row.get("bbox_reseed_requested", False)) for row in rows)
            ),
            "bbox_reseed_used_frames": int(len(reseed_rows)),
            "bbox_reseed_ratio": float(len(reseed_rows) / max(len(rows), 1)),
            "all_pose_frames": summarize_pose(pose_rows),
            "fresh_pose_frames": summarize_pose(fresh_rows),
        }

    def _pursuit_summary(
            self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        distance_abs = [abs(float(row["distance_error_m"])) for row in rows]
        distance_signed = [float(row["distance_error_m"]) for row in rows]
        lateral_abs = [abs(float(row["gt_dy_m"])) for row in rows]
        follow_yaw_abs = [abs(float(row["gt_follow_yaw_deg"])) for row in rows]
        within_band_ratio = float(
            sum(bool(row["within_follow_band"]) for row in rows) / max(len(rows), 1)
        )
        offroad_ratio = float(
            sum(bool(row["offroad"]) for row in rows) / max(len(rows), 1)
        )

        throttle_delta = []
        steer_delta = []
        brake_delta = []
        for prev, cur in zip(rows[:-1], rows[1:]):
            throttle_delta.append(
                abs(float(cur["throttle"]) - float(prev["throttle"])))
            steer_delta.append(abs(float(cur["steer"]) - float(prev["steer"])))
            brake_delta.append(abs(float(cur["brake"]) - float(prev["brake"])))

        first_capture_frame = None
        for row in rows:
            if bool(row["within_follow_band"]):
                first_capture_frame = int(row["frame"])
                break

        return {
            "distance_error_abs_m": _stats(distance_abs),
            "distance_error_signed_m": _stats(distance_signed),
            "lateral_error_abs_m": _stats(lateral_abs),
            "follow_yaw_abs_deg": _stats(follow_yaw_abs),
            "within_follow_band_ratio": within_band_ratio,
            "first_capture_frame": (
                int(first_capture_frame)
                if first_capture_frame is not None
                else -1
            ),
            "offroad_ratio": offroad_ratio,
            "collision_count": int(
                sum(int(row["collision_events"]) for row in rows)
            ),
            "control_delta_throttle": _stats(throttle_delta),
            "control_delta_steer": _stats(steer_delta),
            "control_delta_brake": _stats(brake_delta),
            "ego_speed_mps": _stats(
                [float(row["ego_speed_mps"]) for row in rows]
            ),
            "target_speed_mps": _stats(
                [float(row["target_speed_mps"]) for row in rows]
            ),
        }

    def summarize(self, completion_reason: str) -> Dict[str, object]:
        rows = list(self.rows)
        return {
            "config": self.config.to_dict(),
            "frames": len(rows),
            "completion_reason": completion_reason,
            "pose_source": self.config.pose_source,
            "pose_source_metrics": self._pose_summary(rows),
            "pursuit_quality": self._pursuit_summary(rows),
        }

    def write(self, completion_reason: str) -> str:
        summary = self.summarize(completion_reason)
        with open(self.config.summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(self.config.frame_log_path, "w") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        return self.config.summary_path


def build_frame_metrics_row(
    config: PursuitEvalConfig,
    frame_idx: int,
    tick: int,
    pose_source: str,
    gt_pose: Dict[str, float],
    ego_speed_mps: float,
    target_speed_mps: float,
    throttle: float,
    steer: float,
    brake: float,
    offroad: bool,
    collision_events: int,
    pose_available: bool,
    pose_stale: bool,
    pose_latency_ms: Optional[float],
    used_pose: Optional[Dict[str, float]],
    mask_iou: Optional[float],
    mask_available: bool,
    detector_pose_available: bool,
    bbox_bootstrap_used: bool,
    bbox_reseed_requested: bool,
    bbox_reseed_used: bool,
    bbox_reseed_reason: str,
    tracker_logit_max: Optional[float],
    tracker_threshold: Optional[float],
) -> Dict[str, object]:
    used_pose = used_pose or {
        "dx_m": gt_pose["dx_m"],
        "dy_m": gt_pose["dy_m"],
        "yaw_deg": gt_pose["yaw_deg"],
    }
    pose_yaw_error = wrap_angle_deg(
        float(
            used_pose["yaw_deg"]) -
        float(
            gt_pose["yaw_deg"]))

    return {
        "frame": int(frame_idx),
        "tick": int(tick),
        "pose_source": pose_source,
        "pose_available": bool(pose_available),
        "pose_stale": bool(pose_stale),
        "pose_latency_ms": None if pose_latency_ms is None else float(pose_latency_ms),
        "mask_available": bool(mask_available),
        "detector_pose_available": bool(detector_pose_available),
        "bbox_bootstrap_used": bool(bbox_bootstrap_used),
        "bbox_reseed_requested": bool(bbox_reseed_requested),
        "bbox_reseed_used": bool(bbox_reseed_used),
        "bbox_reseed_reason": str(bbox_reseed_reason),
        "tracker_logit_max": (
            None if tracker_logit_max is None else float(tracker_logit_max)
        ),
        "tracker_threshold": (
            None if tracker_threshold is None else float(tracker_threshold)
        ),
        "gt_dx_m": float(gt_pose["dx_m"]),
        "gt_dy_m": float(gt_pose["dy_m"]),
        "gt_yaw_deg": float(gt_pose["yaw_deg"]),
        "gt_follow_yaw_deg": float(canonicalize_follow_yaw_deg(gt_pose["yaw_deg"])),
        "used_dx_m": float(used_pose["dx_m"]),
        "used_dy_m": float(used_pose["dy_m"]),
        "used_yaw_deg": float(used_pose["yaw_deg"]),
        "pose_dx_error_m": float(used_pose["dx_m"]) - float(gt_pose["dx_m"]),
        "pose_dy_error_m": float(used_pose["dy_m"]) - float(gt_pose["dy_m"]),
        "pose_yaw_error_deg": float(pose_yaw_error),
        "pose_yaw_mod_180_error_deg": float(
            _yaw_mod_180_abs(float(used_pose["yaw_deg"]), float(gt_pose["yaw_deg"]))
        ),
        "pose_follow_yaw_error_deg": float(
            abs(
                canonicalize_follow_yaw_deg(float(used_pose["yaw_deg"]))
                - canonicalize_follow_yaw_deg(float(gt_pose["yaw_deg"]))
            )
        ),
        "distance_error_m": float(gt_pose["dx_m"]) - float(config.desired_distance_m),
        "ego_speed_mps": float(ego_speed_mps),
        "target_speed_mps": float(target_speed_mps),
        "throttle": float(throttle),
        "steer": float(steer),
        "brake": float(brake),
        "offroad": int(bool(offroad)),
        "collision_events": int(collision_events),
        "within_follow_band": int(
            abs(float(gt_pose["dx_m"]) - float(config.desired_distance_m))
            <= float(config.follow_band_distance_abs_m)
            and abs(float(gt_pose["dy_m"])) <= float(config.follow_band_lateral_abs_m)
            and abs(canonicalize_follow_yaw_deg(float(gt_pose["yaw_deg"])))
            <= float(config.follow_band_yaw_abs_deg)
        ),
        "mask_iou": None if mask_iou is None else float(mask_iou),
    }
