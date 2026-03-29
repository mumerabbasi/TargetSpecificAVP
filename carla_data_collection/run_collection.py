#!/usr/bin/env python3
"""CLI for the per-target CARLA dataset pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from carla_data_collection import Config  # noqa: E402


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        default="carla_dataset",
        help="Root output directory",
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="CARLA server host",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2150,
        help="CARLA server port",
    )
    parser.add_argument(
        "--towns",
        type=str,
        nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
        help="CARLA towns to process",
    )
    parser.add_argument(
        "--follow-only",
        action="store_true",
        help="Keep only pursuit-like target cars in front of ego during build and capture selection",
    )
    parser.add_argument(
        "--min-follow-actors-per-frame",
        type=int,
        default=1,
        help="Minimum number of follow-valid actors required to keep a frame when --follow-only is used",
    )
    parser.add_argument(
        "--max-follow-actors-per-frame",
        type=int,
        default=0,
        help="Optional maximum number of follow-valid actors allowed in a kept frame when --follow-only is used; 0 disables the cap",
    )
    parser.add_argument(
        "--follow-lateral-limit-m",
        type=float,
        default=12.0,
        help="Maximum |dy| for a target to count as follow-valid",
    )
    parser.add_argument(
        "--follow-yaw-limit-deg",
        type=float,
        default=120.0,
        help="Maximum |dyaw| for a target to count as follow-valid",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-target CARLA dataset pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture-raw",
        help="Stage A: capture reusable raw CARLA frames",
    )
    _add_common_arguments(capture)
    capture.add_argument("--fresh", action="store_true", help="Delete old raw capture output")
    capture.add_argument(
        "--target-samples-per-town",
        type=int,
        default=3000,
        help="Stop after this many target samples per town",
    )
    capture.add_argument(
        "--max-frames-per-town",
        type=int,
        default=12000,
        help="Upper bound on accepted raw frames per town",
    )
    capture.add_argument(
        "--num-traffic-vehicles",
        type=int,
        default=80,
        help="Background traffic vehicle count",
    )
    capture.add_argument(
        "--traffic-mode",
        type=str,
        choices=("traffic_manager", "constant_velocity"),
        default="traffic_manager",
        help="How to move the ego and background traffic during capture",
    )
    capture.add_argument(
        "--max-episodes-per-town",
        type=int,
        default=4,
        help="Maximum restart attempts per town",
    )
    capture.add_argument(
        "--episode-frame-budget",
        type=int,
        default=3000,
        help="Maximum world ticks to spend in one traffic episode before restarting it",
    )

    build_gt = subparsers.add_parser(
        "build-gt-dataset",
        help="Build shared RGB/masks plus GT pose CSV only",
    )
    _add_common_arguments(build_gt)
    build_gt.add_argument(
        "--sam3-repo-path",
        type=str,
        default="/my_workspace/4DHHOI/sam3",
        help="Path to the SAM3 repository",
    )
    build_gt.add_argument(
        "--sam3-checkpoint-path",
        type=str,
        default="",
        help="Optional SAM3 checkpoint path; defaults to HF/cache loading",
    )
    build_gt.add_argument(
        "--sam3-device",
        type=str,
        default="cuda:0",
        help="Device for SAM3 inference",
    )

    attach_pred = subparsers.add_parser(
        "attach-predictions",
        help="Attach detector-derived pose rows to an existing GT dataset",
    )
    _add_common_arguments(attach_pred)
    attach_pred.add_argument(
        "--detector-name",
        type=str,
        default="centerpoint",
        help="Human-readable detector name",
    )
    attach_pred.add_argument(
        "--detector-config",
        type=str,
        default="",
        help="MMDet3D config path",
    )
    attach_pred.add_argument(
        "--detector-checkpoint",
        type=str,
        default="",
        help="MMDet3D checkpoint path",
    )
    attach_pred.add_argument(
        "--detector-device",
        type=str,
        default="cuda:0",
        help="Device for detector inference",
    )
    attach_pred.add_argument(
        "--detector-score-thr",
        type=float,
        default=0.15,
        help="Detector score threshold",
    )

    benchmark = subparsers.add_parser(
        "benchmark-detectors",
        help="Benchmark detector candidates on GT target samples",
    )
    _add_common_arguments(benchmark)
    benchmark.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Repeated detector candidate spec: name=config_path::checkpoint_path",
    )
    benchmark.add_argument(
        "--detector-device",
        type=str,
        default="cuda:0",
        help="Device for benchmarked detectors",
    )
    benchmark.add_argument(
        "--detector-score-thr",
        type=float,
        default=0.15,
        help="Detector score threshold",
    )

    report_metrics = subparsers.add_parser(
        "report-metrics",
        help="Write detailed dataset and detector-vs-GT metrics for an existing dataset",
    )
    _add_common_arguments(report_metrics)

    return parser.parse_args()


def _namespace_to_config(args: argparse.Namespace) -> Config:
    kwargs = {
        "output_dir": args.output_dir,
        "carla_host": args.carla_host,
        "carla_port": args.carla_port,
        "towns": tuple(args.towns),
    }

    if hasattr(args, "fresh"):
        kwargs["fresh_start"] = args.fresh
    if hasattr(args, "target_samples_per_town"):
        kwargs["target_samples_per_town"] = args.target_samples_per_town
    if hasattr(args, "max_frames_per_town"):
        kwargs["max_frames_per_town"] = args.max_frames_per_town
    if hasattr(args, "num_traffic_vehicles"):
        kwargs["num_traffic_vehicles"] = args.num_traffic_vehicles
    if hasattr(args, "traffic_mode"):
        kwargs["traffic_mode"] = args.traffic_mode
    if hasattr(args, "max_episodes_per_town"):
        kwargs["max_episodes_per_town"] = args.max_episodes_per_town
    if hasattr(args, "episode_frame_budget"):
        kwargs["episode_frame_budget"] = args.episode_frame_budget
    if hasattr(args, "follow_only"):
        kwargs["follow_only"] = args.follow_only
    if hasattr(args, "min_follow_actors_per_frame"):
        kwargs["min_follow_actors_per_frame"] = args.min_follow_actors_per_frame
    if hasattr(args, "max_follow_actors_per_frame"):
        kwargs["max_follow_actors_per_frame"] = args.max_follow_actors_per_frame
    if hasattr(args, "follow_lateral_limit_m"):
        kwargs["follow_lateral_limit_m"] = args.follow_lateral_limit_m
    if hasattr(args, "follow_yaw_limit_deg"):
        kwargs["follow_yaw_limit_deg"] = args.follow_yaw_limit_deg

    if hasattr(args, "sam3_repo_path"):
        kwargs["sam3_repo_path"] = args.sam3_repo_path
    if hasattr(args, "sam3_checkpoint_path"):
        kwargs["sam3_checkpoint_path"] = args.sam3_checkpoint_path
    if hasattr(args, "sam3_device"):
        kwargs["sam3_device"] = args.sam3_device
    if hasattr(args, "detector_name"):
        kwargs["detector_name"] = args.detector_name
    if hasattr(args, "detector_config") and args.detector_config:
        kwargs["detector_config"] = args.detector_config
    if hasattr(args, "detector_checkpoint") and args.detector_checkpoint:
        kwargs["detector_checkpoint"] = args.detector_checkpoint
    if hasattr(args, "detector_device"):
        kwargs["detector_device"] = args.detector_device
    if hasattr(args, "detector_score_thr"):
        kwargs["detector_score_thr"] = args.detector_score_thr

    return Config(**kwargs)


def main() -> None:
    args = parse_args()
    config = _namespace_to_config(args)

    if args.command == "capture-raw":
        from carla_data_collection.raw_capture import capture_raw_data

        capture_raw_data(config)
        return

    if args.command == "build-gt-dataset":
        from carla_data_collection.dataset_builder import build_gt_dataset

        build_gt_dataset(config)
        return

    if args.command == "attach-predictions":
        from carla_data_collection.dataset_builder import attach_predicted_poses

        attach_predicted_poses(config)
        return

    if args.command == "benchmark-detectors":
        from carla_data_collection.benchmark_detectors import (
            benchmark_detectors,
            parse_candidate_spec,
        )
        from carla_data_collection.detector_3d import DetectorSpec

        candidates: List[Any] = []
        if args.candidate:
            candidates = [
                parse_candidate_spec(
                    spec,
                    default_score_thr=args.detector_score_thr,
                    default_device=args.detector_device,
                )
                for spec in args.candidate
            ]
        else:
            candidates = [
                DetectorSpec(
                    name=config.detector_name,
                    config_path=config.detector_config,
                    checkpoint_path=config.detector_checkpoint,
                    score_thr=config.detector_score_thr,
                    device=config.detector_device,
                )
            ]

        benchmark_detectors(config, candidates)
        return

    if args.command == "report-metrics":
        from carla_data_collection.report_metrics import write_detailed_metrics_report

        out_path = write_detailed_metrics_report(config)
        print(f"[report-metrics] wrote {out_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
