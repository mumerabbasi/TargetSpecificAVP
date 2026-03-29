#!/usr/bin/env python3
"""CLI for the compact single-pass RAVP dataset pipeline."""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from carla_data_collection import Config  # noqa: E402


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=str, default="carla_dataset")
    parser.add_argument("--carla-host", type=str, default="localhost")
    parser.add_argument("--carla-port", type=int, default=2150)
    parser.add_argument(
        "--towns",
        type=str,
        nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
    )
    parser.add_argument("--follow-only", action="store_true")
    parser.add_argument("--min-follow-actors-per-frame", type=int, default=1)
    parser.add_argument("--max-follow-actors-per-frame", type=int, default=0)
    parser.add_argument("--follow-lateral-limit-m", type=float, default=12.0)
    parser.add_argument("--follow-yaw-limit-deg", type=float, default=120.0)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--rgb-jpeg-quality", type=int, default=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact RAVP dataset pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect-dataset",
        help="Collect the final per-target dataset in one pass",
    )
    _add_common_arguments(collect)
    collect.add_argument("--fresh", action="store_true")
    collect.add_argument("--target-samples-per-town", type=int, default=3000)
    collect.add_argument("--max-frames-per-town", type=int, default=12000)
    collect.add_argument("--num-traffic-vehicles", type=int, default=80)
    collect.add_argument(
        "--traffic-mode",
        type=str,
        choices=("traffic_manager", "constant_velocity"),
        default="traffic_manager",
    )
    collect.add_argument("--max-episodes-per-town", type=int, default=4)
    collect.add_argument("--episode-frame-budget", type=int, default=3000)
    collect.add_argument(
        "--sam3-repo-path",
        type=str,
        default="/my_workspace/4DHHOI/sam3")
    collect.add_argument("--sam3-checkpoint-path", type=str, default="")
    collect.add_argument("--sam3-device", type=str, default="cuda:0")
    collect.add_argument("--detector-name", type=str, default="centerpoint")
    collect.add_argument("--detector-config", type=str, default="")
    collect.add_argument("--detector-checkpoint", type=str, default="")
    collect.add_argument("--detector-device", type=str, default="cuda:0")
    collect.add_argument("--detector-score-thr", type=float, default=0.15)

    report = subparsers.add_parser(
        "report-metrics",
        help="Write a detailed detector-vs-GT report for an existing dataset",
    )
    _add_common_arguments(report)

    return parser.parse_args()


def _namespace_to_config(args: argparse.Namespace) -> Config:
    kwargs = {
        "output_dir": args.output_dir,
        "carla_host": args.carla_host,
        "carla_port": args.carla_port,
        "towns": tuple(args.towns),
        "follow_only": bool(args.follow_only),
        "min_follow_actors_per_frame": int(args.min_follow_actors_per_frame),
        "max_follow_actors_per_frame": int(args.max_follow_actors_per_frame),
        "follow_lateral_limit_m": float(args.follow_lateral_limit_m),
        "follow_yaw_limit_deg": float(args.follow_yaw_limit_deg),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "rgb_jpeg_quality": int(args.rgb_jpeg_quality),
    }

    if args.command == "collect-dataset":
        kwargs.update(
            {
                "fresh_start": bool(args.fresh),
                "target_samples_per_town": int(args.target_samples_per_town),
                "max_frames_per_town": int(args.max_frames_per_town),
                "num_traffic_vehicles": int(args.num_traffic_vehicles),
                "traffic_mode": str(args.traffic_mode),
                "max_episodes_per_town": int(args.max_episodes_per_town),
                "episode_frame_budget": int(args.episode_frame_budget),
                "sam3_repo_path": str(args.sam3_repo_path),
                "sam3_checkpoint_path": str(args.sam3_checkpoint_path),
                "sam3_device": str(args.sam3_device),
                "detector_name": str(args.detector_name),
                "detector_device": str(args.detector_device),
                "detector_score_thr": float(args.detector_score_thr),
            }
        )
        if args.detector_config:
            kwargs["detector_config"] = str(args.detector_config)
        if args.detector_checkpoint:
            kwargs["detector_checkpoint"] = str(args.detector_checkpoint)

    return Config(**kwargs)


def main() -> None:
    args = parse_args()
    config = _namespace_to_config(args)

    if args.command == "collect-dataset":
        from carla_data_collection.collector import collect_dataset

        collect_dataset(config)
        return

    if args.command == "report-metrics":
        from carla_data_collection.report_metrics import write_detailed_metrics_report

        out_path = write_detailed_metrics_report(config)
        print(f"[report-metrics] wrote {out_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
