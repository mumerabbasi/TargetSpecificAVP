"""Run multiple fresh pursuit-evaluation sequences and aggregate their results."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from .config import PursuitEvalConfig
from .run import run_pursuit


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r") as handle:
        return json.load(handle)


def _sequence_run_name(town: str, pose_source: str, seed: int, attempt_idx: int) -> str:
    return "{}_{}_seed{:03d}_try{:02d}".format(town, pose_source, seed, attempt_idx + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fresh multi-sequence pursuit suite.")
    parser.add_argument("--towns", nargs="+", default=["Town02", "Town10HD", "Town10HD_Opt"])
    parser.add_argument("--pose-sources", nargs="+", default=["gt", "detector"])
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2150)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=500)
    parser.add_argument("--attempts-per-sequence", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=101)
    parser.add_argument("--num-background-vehicles", type=int, default=35)
    parser.add_argument("--initial-target-distance", type=float, default=12.0)
    parser.add_argument("--ego-initial-speed", type=float, default=0.0)
    parser.add_argument("--target-speed-difference", type=float, default=80.0)
    parser.add_argument("--sam3-worker-env", default="ravp")
    parser.add_argument("--detector-worker-env", default="ravp-det")
    return parser.parse_args()


def _sequence_summary(summary: Dict[str, object], summary_path: str) -> Dict[str, object]:
    pose_metrics = dict(summary["pose_source_metrics"])
    pursuit_quality = dict(summary["pursuit_quality"])
    artifacts = dict(summary.get("artifacts", {}))
    return {
        "summary_path": summary_path,
        "frames": int(summary["frames"]),
        "completion_reason": str(summary["completion_reason"]),
        "pose_source": str(summary["pose_source"]),
        "pose_metrics": {
            "availability_ratio": float(pose_metrics["availability_ratio"]),
            "fresh_ratio": float(pose_metrics["fresh_ratio"]),
            "stale_ratio": float(pose_metrics["stale_ratio"]),
            "dx_abs_mean_m": float(pose_metrics["all_pose_frames"]["dx_abs_m"]["mean"]),
            "dy_abs_mean_m": float(pose_metrics["all_pose_frames"]["dy_abs_m"]["mean"]),
            "yaw_abs_mean_deg": float(pose_metrics["all_pose_frames"]["yaw_abs_deg"]["mean"]),
            "yaw_mod_180_abs_mean_deg": float(
                pose_metrics["all_pose_frames"]["yaw_mod_180_abs_deg"]["mean"]
            ),
            "follow_yaw_abs_mean_deg": float(
                pose_metrics["all_pose_frames"]["follow_yaw_abs_deg"]["mean"]
            ),
            "latency_mean_ms": float(pose_metrics["all_pose_frames"]["latency_ms"]["mean"]),
            "mask_iou_mean": float(pose_metrics["all_pose_frames"]["mask_iou"]["mean"]),
        },
        "pursuit_quality": {
            "distance_error_abs_mean_m": float(pursuit_quality["distance_error_abs_m"]["mean"]),
            "lateral_error_abs_mean_m": float(pursuit_quality["lateral_error_abs_m"]["mean"]),
            "follow_yaw_abs_mean_deg": float(pursuit_quality["follow_yaw_abs_deg"]["mean"]),
            "within_follow_band_ratio": float(pursuit_quality["within_follow_band_ratio"]),
            "first_capture_frame": int(pursuit_quality["first_capture_frame"]),
            "offroad_ratio": float(pursuit_quality["offroad_ratio"]),
            "collision_count": int(pursuit_quality["collision_count"]),
            "ego_speed_mean_mps": float(pursuit_quality["ego_speed_mps"]["mean"]),
            "target_speed_mean_mps": float(pursuit_quality["target_speed_mps"]["mean"]),
        },
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    suite_results: List[Dict[str, object]] = []

    for town in args.towns:
        for pose_source in args.pose_sources:
            last_summary = None
            last_summary_path = None
            for attempt_idx in range(int(args.attempts_per_sequence)):
                seed = int(args.base_seed) + int(attempt_idx)
                run_name = _sequence_run_name(town, pose_source, seed, attempt_idx)
                config = PursuitEvalConfig(
                    pose_source=pose_source,
                    town=town,
                    carla_host=args.carla_host,
                    carla_port=int(args.carla_port),
                    random_seed=seed,
                    num_frames=int(args.frames_per_sequence),
                    output_dir=args.output_dir,
                    run_name=run_name,
                    num_background_vehicles=int(args.num_background_vehicles),
                    initial_target_distance_m=float(args.initial_target_distance),
                    ego_initial_speed_mps=float(args.ego_initial_speed),
                    target_speed_difference_pct=float(args.target_speed_difference),
                    stop_on_follow_guard_breach=False,
                    target_out_of_view_breach_frames=20,
                    ego_offroad_breach_frames=15,
                    sam3_worker_env=args.sam3_worker_env,
                    detector_worker_env=args.detector_worker_env,
                )
                summary_path = run_pursuit(config)
                summary = _read_json(summary_path)
                last_summary = summary
                last_summary_path = summary_path
                if (
                    int(summary["frames"]) >= int(args.frames_per_sequence)
                    and str(summary["completion_reason"]) == "max_frames"
                ):
                    break

            if last_summary is None or last_summary_path is None:
                raise RuntimeError("No pursuit run completed for {} / {}.".format(town, pose_source))
            if int(last_summary["frames"]) < int(args.frames_per_sequence):
                raise RuntimeError(
                    "Failed to obtain {} frames for {} / {}. Last summary: {}".format(
                        args.frames_per_sequence,
                        town,
                        pose_source,
                        last_summary_path,
                    )
                )

            sequence_record = {
                "town": town,
                "pose_source": pose_source,
                "run_name": str(last_summary["config"]["run_name"]),
            }
            sequence_record.update(_sequence_summary(last_summary, last_summary_path))
            suite_results.append(sequence_record)

    suite_summary = {
        "output_dir": args.output_dir,
        "towns": list(args.towns),
        "pose_sources": list(args.pose_sources),
        "frames_per_sequence": int(args.frames_per_sequence),
        "results": suite_results,
    }
    summary_path = os.path.join(args.output_dir, "suite_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(suite_summary, handle, indent=2)
    print(summary_path)


if __name__ == "__main__":
    main()
