#!/usr/bin/env python3
"""
CARLA 3D Detection Dataset Collection

Usage:
    python carla_data_collection/run_collection.py --towns Town01 Town02 --frames_per_town 50

See Config class for all available options.
"""

import argparse
import os
import sys
import warnings

# Add parent directory to path so we can import carla_data_collection
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

warnings.filterwarnings("ignore")

from carla_data_collection import Config, run_collection  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect 3D detection dataset from CARLA"
    )

    # CARLA connection
    parser.add_argument(
        "--carla_host", type=str, default="localhost",
        help="CARLA server host"
    )
    parser.add_argument(
        "--carla_port", type=int, default=2150,
        help="CARLA server port"
    )

    # Towns and frames
    parser.add_argument(
        "--towns", type=str, nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
        help="List of CARLA towns to collect from"
    )
    parser.add_argument(
        "--frames_per_town", type=int, default=10000,
        help="Number of frames to collect per town"
    )

    # Target spawning
    parser.add_argument(
        "--min_targets", type=int, default=1,
        help="Minimum targets per waypoint"
    )
    parser.add_argument(
        "--max_targets", type=int, default=5,
        help="Maximum targets per waypoint"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="/storage/remote/atcremers45/s0050/carla_dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh (overwrite existing data instead of resuming)"
    )

    # Gaussian distribution parameters
    parser.add_argument(
        "--dx_mean", type=float, default=8.0,
        help="Target dx mean (m)"
    )
    parser.add_argument(
        "--dx_std", type=float, default=3.0,
        help="Target dx std (m)"
    )
    parser.add_argument(
        "--dy_std", type=float, default=2.0,
        help="Target dy std (m)"
    )
    parser.add_argument(
        "--dyaw_std", type=float, default=20.0,
        help="Target dyaw std (deg)"
    )

    # Outlier thresholds
    parser.add_argument(
        "--max_err_dx", type=float, default=0.5,
        help="Max dx error (m)"
    )
    parser.add_argument(
        "--max_err_dy", type=float, default=0.5,
        help="Max dy error (m)"
    )
    parser.add_argument(
        "--max_err_yaw", type=float, default=5.0,
        help="Max yaw error (deg)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    config = Config(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        towns=tuple(args.towns),
        frames_per_town=args.frames_per_town,
        min_targets=args.min_targets,
        max_targets=args.max_targets,
        output_dir=args.output_dir,
        fresh_start=args.fresh,
        target_dx_mean=args.dx_mean,
        target_dx_std=args.dx_std,
        target_dy_std=args.dy_std,
        target_dyaw_std=args.dyaw_std,
        max_err_dx=args.max_err_dx,
        max_err_dy=args.max_err_dy,
        max_err_yaw=args.max_err_yaw,
    )

    print("Configuration:")
    print(f"  Towns: {config.towns}")
    print(f"  Frames per town: {config.frames_per_town}")
    print(f"  Total expected frames: {len(config.towns) * config.frames_per_town}")
    print(f"  Output: {config.output_dir}")

    run_collection(config)


if __name__ == "__main__":
    main()
