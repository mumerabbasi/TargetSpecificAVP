"""Metrics for target-specific pursuit inference."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import InferenceConfig
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


def _compute_infraction(invasion_num: int, collision_num: int) -> float:
    return float((0.8 ** int(invasion_num)) * (0.5 ** int(collision_num)))


def _find_closest_point(point: np.ndarray, pos_array: np.ndarray) -> int:
    diff = pos_array[:, :3] - point[:3][None, :]
    dis_array = np.linalg.norm(diff, axis=1, keepdims=False)
    return int(np.argmin(dis_array))


def _point_distance_matrix(
    ref_pos: np.ndarray,
    ego_pos: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    ego = ego_pos[start_idx:, :]
    if ego.size == 0 or ref_pos.size == 0:
        return np.zeros((0, 0), dtype=np.float64)

    rows = []
    for point in ego:
        diff = ref_pos[:, :3] - point[:3][None, :]
        rows.append(np.linalg.norm(diff, axis=1, keepdims=False))
    return np.stack(rows, axis=0)


def _compute_trans_error(ego_pos: np.ndarray, ref_pos: np.ndarray) -> float:
    if ego_pos.size == 0 or ref_pos.size == 0:
        return 0.0

    rel_translation = []
    for ref_state, ego_state in zip(ref_pos, ego_pos):
        x1, y1, theta1, _ = ref_state
        x2, y2, theta2, _ = ego_state
        g1 = np.array(
            [
                [np.cos(theta1), -np.sin(theta1), x1],
                [np.sin(theta1), np.cos(theta1), y1],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        g2 = np.array(
            [
                [np.cos(theta2), -np.sin(theta2), x2],
                [np.sin(theta2), np.cos(theta2), y2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        g12 = np.linalg.inv(g2).dot(g1)
        rel_translation.append(g12[:2, 2])

    rel_translation_arr = np.asarray(rel_translation, dtype=np.float64)
    return float(
        np.sum(np.linalg.norm(rel_translation_arr, axis=1)) / max(len(rel_translation_arr), 1)
    )


def _control_difference(
    target_controls: np.ndarray,
    ego_controls: np.ndarray,
    ref_idx: np.ndarray,
    ego_idx: np.ndarray,
    start_idx: int,
) -> float:
    if target_controls.size == 0 or ego_controls.size == 0 or len(ref_idx) == 0:
        return 0.0
    differences = [
        float(
            np.linalg.norm(
                target_controls[int(ref_i)] - ego_controls[int(ego_i) + int(start_idx)]
            )
        )
        for ref_i, ego_i in zip(ref_idx, ego_idx)
    ]
    return float(np.mean(differences)) if differences else 0.0


class InferenceMetrics:
    """Accumulate per-frame pursuit metrics and write a JSON report."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.rows: List[Dict[str, object]] = []

    def add_row(self, row: Dict[str, object]) -> None:
        self.rows.append(dict(row))

    def _pose_summary(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        pose_rows = [row for row in rows if bool(row.get("pose_available", False))]
        fresh_rows = [row for row in pose_rows if not bool(row.get("pose_stale", False))]

        def summarize(subset: List[Dict[str, object]]) -> Dict[str, object]:
            dx_abs = [abs(float(row["pose_dx_error_m"])) for row in subset]
            dy_abs = [abs(float(row["pose_dy_error_m"])) for row in subset]
            yaw_abs = [abs(float(row["pose_follow_yaw_error_deg"])) for row in subset]
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
                "follow_yaw_abs_deg": _stats(yaw_abs),
                "mask_iou": _stats(mask_ious),
                "latency_ms": _stats(latencies),
            }

        return {
            "availability_ratio": float(len(pose_rows) / max(len(rows), 1)),
            "fresh_ratio": float(len(fresh_rows) / max(len(rows), 1)),
            "stale_ratio": float(
                sum(bool(row.get("pose_stale", False)) for row in pose_rows)
                / max(len(rows), 1)
            ),
            "all_pose_frames": summarize(pose_rows),
            "fresh_pose_frames": summarize(fresh_rows),
        }

    def _pursuit_summary(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
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

        command_throttle_delta = []
        command_steer_delta = []
        command_brake_delta = []
        for prev, cur in zip(rows[:-1], rows[1:]):
            command_throttle_delta.append(
                abs(float(cur["command_throttle"]) - float(prev["command_throttle"]))
            )
            command_steer_delta.append(
                abs(float(cur["command_steer"]) - float(prev["command_steer"]))
            )
            command_brake_delta.append(
                abs(float(cur["command_brake"]) - float(prev["command_brake"]))
            )

        return {
            "distance_error_abs_m": _stats(distance_abs),
            "distance_error_signed_m": _stats(distance_signed),
            "lateral_error_abs_m": _stats(lateral_abs),
            "follow_yaw_abs_deg": _stats(follow_yaw_abs),
            "within_follow_band_ratio": within_band_ratio,
            "offroad_ratio": offroad_ratio,
            "ego_speed_mps": _stats([float(row["ego_speed_mps"]) for row in rows]),
            "target_speed_mps": _stats(
                [float(row["target_speed_mps"]) for row in rows]
            ),
            "command_delta_throttle": _stats(command_throttle_delta),
            "command_delta_steer": _stats(command_steer_delta),
            "command_delta_brake": _stats(command_brake_delta),
        }

    def _closed_loop_summary(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        if not rows:
            return {
                "completion_ratio": 0.0,
                "completion_percent": 0.0,
                "target_lane_invasions": 0,
                "ego_lane_invasions": 0,
                "target_collisions": 0,
                "ego_collisions": 0,
                "relative_infraction": 1.0,
                "absolute_infraction": 1.0,
                "control_difference": 0.0,
                "ate": 0.0,
                "matched_frames": 0,
            }

        last_row = rows[-1]
        target_lane_invasions = int(last_row["target_lane_invasions_total"])
        ego_lane_invasions = int(last_row["ego_lane_invasions_total"])
        target_collisions = int(last_row["target_collisions_total"])
        ego_collisions = int(last_row["ego_collisions_total"])

        relative_invasion_num = max(ego_lane_invasions - target_lane_invasions, 0)
        relative_collision_num = max(ego_collisions - target_collisions, 0)

        target_states = np.asarray(
            [
                [
                    float(row["target_x_m"]),
                    float(row["target_y_m"]),
                    float(row["target_yaw_rad"]),
                    float(row["target_speed_mps"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        ego_states = np.asarray(
            [
                [
                    float(row["ego_x_m"]),
                    float(row["ego_y_m"]),
                    float(row["ego_yaw_rad"]),
                    float(row["ego_speed_mps"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        target_controls = np.asarray(
            [
                [
                    float(row["target_control_throttle"]),
                    float(row["target_control_steer"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        ego_controls = np.asarray(
            [
                [
                    float(row["ego_control_throttle"]),
                    float(row["ego_control_steer"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        start_idx = _find_closest_point(target_states[0], ego_states)
        dis_matrix = _point_distance_matrix(target_states, ego_states, start_idx)
        matched_frames = 0
        ate = 0.0
        control_difference = 0.0
        if dis_matrix.size > 0:
            ego_idx, ref_idx = linear_sum_assignment(dis_matrix)
            matched_ego = ego_states[ego_idx + start_idx]
            matched_ref = target_states[ref_idx]
            matched_frames = int(len(ref_idx))
            ate = _compute_trans_error(matched_ego, matched_ref)
            control_difference = _control_difference(
                target_controls,
                ego_controls,
                ref_idx,
                ego_idx,
                start_idx,
            )

        completion_ratio = float(len(rows) / max(int(self.config.num_frames), 1))
        return {
            "completion_ratio": completion_ratio,
            "completion_percent": 100.0 * completion_ratio,
            "target_lane_invasions": target_lane_invasions,
            "ego_lane_invasions": ego_lane_invasions,
            "target_collisions": target_collisions,
            "ego_collisions": ego_collisions,
            "relative_infraction": _compute_infraction(
                relative_invasion_num,
                relative_collision_num,
            ),
            "absolute_infraction": _compute_infraction(
                ego_lane_invasions,
                ego_collisions,
            ),
            "control_difference": control_difference,
            "ate": ate,
            "matched_frames": matched_frames,
        }

    def summarize(self, completion_reason: str) -> Dict[str, object]:
        rows = list(self.rows)
        completion_ratio = float(len(rows) / max(int(self.config.num_frames), 1))
        return {
            "config": self.config.to_dict(),
            "frames": len(rows),
            "completion_reason": completion_reason,
            "completion_ratio": completion_ratio,
            "completion_percent": 100.0 * completion_ratio,
            "pose_metrics": self._pose_summary(rows),
            "pursuit_metrics": self._pursuit_summary(rows) if rows else {},
            "closed_loop_metrics": self._closed_loop_summary(rows),
        }

    def _write_closed_loop_report(
        self,
        closed_loop_metrics: Dict[str, object],
        completion_reason: str,
    ) -> None:
        lines = [
            f"Town: {self.config.town}",
            f"Completion_Reason: {completion_reason}",
            f"Completion_Percent: "
            f"{float(closed_loop_metrics['completion_percent']):.2f}",
            f"Target_Lane_Invasions: "
            f"{int(closed_loop_metrics['target_lane_invasions'])}",
            f"Ego_Lane_Invasions: "
            f"{int(closed_loop_metrics['ego_lane_invasions'])}",
            f"Target_Collisions: "
            f"{int(closed_loop_metrics['target_collisions'])}",
            f"Ego_Collisions: "
            f"{int(closed_loop_metrics['ego_collisions'])}",
            f"Relative_Infraction: "
            f"{float(closed_loop_metrics['relative_infraction']):.4f}",
            f"Absolute_Infraction: "
            f"{float(closed_loop_metrics['absolute_infraction']):.4f}",
            f"Control_Difference: "
            f"{float(closed_loop_metrics['control_difference']):.4f}",
            f"ATE: {float(closed_loop_metrics['ate']):.4f}",
            f"Matched_Frames: {int(closed_loop_metrics['matched_frames'])}",
        ]
        with open(self.config.closed_loop_report_path, "w") as handle:
            handle.write("\n".join(lines) + "\n")

    def write(self, completion_reason: str) -> str:
        summary = self.summarize(completion_reason)
        with open(self.config.summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(self.config.frame_log_path, "w") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        self._write_closed_loop_report(
            summary["closed_loop_metrics"],
            completion_reason,
        )
        return self.config.summary_path


def build_frame_metrics_row(
    *,
    config: InferenceConfig,
    frame_idx: int,
    tick: int,
    gt_pose: Dict[str, float],
    used_pose: Optional[Dict[str, float]],
    pose_available: bool,
    pose_stale: bool,
    pose_latency_ms: Optional[float],
    mask_iou: Optional[float],
    mask_available: bool,
    bbox_bootstrap_used: bool,
    bbox_reseed_requested: bool,
    bbox_reseed_used: bool,
    tracker_logit_max: Optional[float],
    tracker_threshold: Optional[float],
    ego_state: Dict[str, float],
    target_state: Dict[str, float],
    ego_world_state: Dict[str, float],
    command_throttle: float,
    command_steer: float,
    command_brake: float,
    offroad: bool,
    ego_collision_events: int,
    target_collision_events: int,
    ego_lane_invasions_total: int,
    ego_collisions_total: int,
    target_lane_invasions_total: int,
    target_collisions_total: int,
    absolute_distance_m: float,
) -> Dict[str, object]:
    pred_pose = used_pose or {}
    gt_follow_yaw_deg = canonicalize_follow_yaw_deg(float(gt_pose["yaw_deg"]))
    pred_follow_yaw_deg = (
        None
        if "yaw_follow_deg" not in pred_pose
        else float(pred_pose["yaw_follow_deg"])
    )
    pose_follow_yaw_error_deg = None
    if pred_follow_yaw_deg is not None:
        pose_follow_yaw_error_deg = abs(
            wrap_angle_deg(pred_follow_yaw_deg - gt_follow_yaw_deg)
        )

    return {
        "frame": int(frame_idx),
        "tick": int(tick),
        "pose_available": bool(pose_available),
        "pose_stale": bool(pose_stale),
        "pose_latency_ms": None if pose_latency_ms is None else float(pose_latency_ms),
        "mask_available": bool(mask_available),
        "bbox_bootstrap_used": bool(bbox_bootstrap_used),
        "bbox_reseed_requested": bool(bbox_reseed_requested),
        "bbox_reseed_used": bool(bbox_reseed_used),
        "tracker_logit_max": (
            None if tracker_logit_max is None else float(tracker_logit_max)
        ),
        "tracker_threshold": (
            None if tracker_threshold is None else float(tracker_threshold)
        ),
        "gt_dx_m": float(gt_pose["dx_m"]),
        "gt_dy_m": float(gt_pose["dy_m"]),
        "gt_yaw_deg": float(gt_pose["yaw_deg"]),
        "gt_follow_yaw_deg": float(gt_follow_yaw_deg),
        "used_dx_m": None if "dx_m" not in pred_pose else float(pred_pose["dx_m"]),
        "used_dy_m": None if "dy_m" not in pred_pose else float(pred_pose["dy_m"]),
        "used_yaw_follow_deg": pred_follow_yaw_deg,
        "pose_dx_error_m": (
            None
            if "dx_m" not in pred_pose
            else float(pred_pose["dx_m"]) - float(gt_pose["dx_m"])
        ),
        "pose_dy_error_m": (
            None
            if "dy_m" not in pred_pose
            else float(pred_pose["dy_m"]) - float(gt_pose["dy_m"])
        ),
        "pose_follow_yaw_error_deg": pose_follow_yaw_error_deg,
        "distance_error_m": float(gt_pose["dx_m"]) - float(config.desired_distance_m),
        "mask_iou": None if mask_iou is None else float(mask_iou),
        "ego_speed_mps": float(ego_state["speed_mps"]),
        "target_speed_mps": float(target_state["speed_mps"]),
        "command_throttle": float(command_throttle),
        "command_steer": float(command_steer),
        "command_brake": float(command_brake),
        "ego_control_throttle": float(ego_state["throttle"]),
        "ego_control_steer": float(ego_state["steer"]),
        "ego_control_brake": float(ego_state["brake"]),
        "target_control_throttle": float(target_state["throttle"]),
        "target_control_steer": float(target_state["steer"]),
        "target_control_brake": float(target_state["brake"]),
        "ego_x_m": float(ego_world_state["x_m"]),
        "ego_y_m": float(ego_world_state["y_m"]),
        "ego_yaw_rad": float(ego_world_state["yaw_rad"]),
        "target_x_m": float(target_state["x_m"]),
        "target_y_m": float(target_state["y_m"]),
        "target_yaw_rad": float(target_state["yaw_rad"]),
        "offroad": int(bool(offroad)),
        "ego_collision_events": int(ego_collision_events),
        "target_collision_events": int(target_collision_events),
        "ego_lane_invasions_total": int(ego_lane_invasions_total),
        "ego_collisions_total": int(ego_collisions_total),
        "target_lane_invasions_total": int(target_lane_invasions_total),
        "target_collisions_total": int(target_collisions_total),
        "absolute_distance_m": float(absolute_distance_m),
        "within_follow_band": int(
            abs(float(gt_pose["dx_m"]) - float(config.desired_distance_m))
            <= float(config.follow_band_distance_abs_m)
            and abs(float(gt_pose["dy_m"])) <= float(config.follow_band_lateral_abs_m)
            and abs(float(gt_follow_yaw_deg)) <= float(config.follow_band_yaw_abs_deg)
        ),
    }
