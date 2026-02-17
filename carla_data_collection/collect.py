"""Main data collection loop."""

import csv
import math
import os
import queue
import random
import time
from collections import defaultdict
from typing import List, Any, Dict, Tuple, Optional

import carla
import numpy as np

from .config import Config
from .utils import (
    get_camera_intrinsic,
    project_lidar_to_camera,
    filter_points_by_mask,
    compute_lidar_to_camera_transform,
    wrap_angle_rad,
)
from .carla_utils import (
    setup_world,
    spawn_ego_vehicle,
    setup_sensors,
    spawn_target_vehicles,
    destroy_actors,
    parse_lidar_measurement,
    parse_rgb_image,
    save_rgb_image,
)
from .vision_detector import VisionDetector
from .detector_3d import load_centerpoint_model, run_centerpoint_detection
from .ground_truth import match_detection_to_target


def get_resume_info(csv_path: str) -> Tuple[Optional[str], Dict[str, int], int, int]:
    """
    Read existing CSV to determine resume point.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        (last_town, town_frame_counts, next_global_frame_id, total_detections)
        - last_town: Name of the last town with data
        - town_frame_counts: Dict mapping town name -> number of FRAMES collected
        - next_global_frame_id: The next frame_id to use
        - total_detections: Total number of detection rows in CSV
        Returns (None, {}, 0, 0) if file doesn't exist or is empty.
    """
    if not os.path.isfile(csv_path):
        return None, {}, 0, 0

    try:
        # Track unique frame_ids per town (not detection count!)
        town_frames: Dict[str, set] = defaultdict(set)
        last_town = None
        max_frame_id = -1
        total_rows = 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                town = row.get("town", "")
                frame_id = int(row.get("frame_id", 0))
                town_frames[town].add(frame_id)
                if frame_id > max_frame_id:
                    max_frame_id = frame_id
                    last_town = town
                total_rows += 1

        if total_rows == 0:
            return None, {}, 0, 0

        # Convert sets to counts
        town_frame_counts = {town: len(frames) for town, frames in town_frames.items()}

        # next_global_frame_id should be max_frame_id + 1
        next_global_frame_id = max_frame_id + 1

        print(f"\n{'='*60}")
        print("Resume info from existing CSV:")
        print(f"  Total detections (rows): {total_rows}")
        print(f"  Last frame ID: {max_frame_id}")
        print(f"  Last town: {last_town}")
        print("  Frames collected per town:")
        for town in sorted(town_frame_counts.keys()):
            count = town_frame_counts[town]
            print(f"    {town}: {count} frames")
        print("=" * 60)

        return last_town, town_frame_counts, next_global_frame_id, total_rows

    except Exception as e:
        print(f"[WARNING] Could not read existing CSV: {e}")
        return None, {}, 0, 0


def collect_from_town(
    client: carla.Client,
    town_name: str,
    config: Config,
    model: Any,
    vision_detector: VisionDetector,
    writer: csv.writer,
    csvfile: Any,
    global_frame_id: int,
    total_detections: int,
    start_frame_idx: int = 0,
) -> tuple:
    """
    Collect data from a single town.

    Args:
        start_frame_idx: Frame index to start/resume from (0-based).

    Returns:
        (global_frame_id, total_detections, completed_frame_idx) after collection.
    """
    print(f"\n{'#'*60}")
    print(f"Loading town: {town_name}")
    print("#" * 60)

    # Load the town (no retry here - let outer loop handle it)
    world = client.load_world(town_name)
    world = setup_world(client)

    print("Cleaning up existing vehicles...")
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            actor.destroy()
        except RuntimeError:
            pass
    for _ in range(10):
        world.tick()

    # Get all available spawn points
    all_spawn_points = world.get_map().get_spawn_points()
    print(f"Found {len(all_spawn_points)} spawn points in {town_name}")

    remaining_frames = config.frames_per_town - start_frame_idx
    if start_frame_idx > 0:
        print(f"Resuming from frame {start_frame_idx + 1}/{config.frames_per_town}")
        print(f"Will collect {remaining_frames} more frames")
    else:
        print(f"Will collect {config.frames_per_town} frames "
              f"(uniformly sampling spawn points)")

    # Track completed frames for potential resume
    completed_frame_idx = start_frame_idx

    # Collect frames by uniformly sampling spawn points
    for frame_idx in range(start_frame_idx, config.frames_per_town):
        spawn_point = random.choice(all_spawn_points)

        print(f"\n{'='*60}")
        print(f"[{town_name}] Frame {frame_idx + 1}/{config.frames_per_town}")
        print(f"Location: ({spawn_point.location.x:.1f}, "
              f"{spawn_point.location.y:.1f}, "
              f"{spawn_point.location.z:.1f})")
        print("=" * 60)

        waypoint_actors: List[carla.Actor] = []

        try:
            ego = spawn_ego_vehicle(world, waypoint_actors, spawn_point, config)

            (
                lidar_actor, rgb_camera,
                lidar_queue, rgb_queue,
                lidar_transform_local, camera_transform_local,
            ) = setup_sensors(world, ego, waypoint_actors, config)

            # Warm up sensors
            for _ in range(10):
                world.tick()
                try:
                    lidar_queue.get(timeout=1.0)
                    rgb_queue.get(timeout=1.0)
                except queue.Empty:
                    pass

            # Spawn targets
            num_targets = random.randint(
                config.min_targets, config.max_targets
            )
            targets = spawn_target_vehicles(
                world, ego, waypoint_actors, num_targets, config
            )
            print(f"Spawned {len(targets)} target vehicles")

            # Wait for vehicles to settle
            for _ in range(15):
                world.tick()
                for q in [lidar_queue, rgb_queue]:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass

            # Compute transforms
            lidar_matrix = np.array(lidar_transform_local.get_matrix())
            camera_matrix = np.array(camera_transform_local.get_matrix())
            lidar_to_camera = compute_lidar_to_camera_transform(
                lidar_matrix, camera_matrix
            )
            intrinsic = get_camera_intrinsic(
                config.image_width, config.image_height, config.fov
            )

            # Collect frames
            for _ in range(config.frames_per_waypoint):
                # Clear queues
                for q in [lidar_queue, rgb_queue]:
                    try:
                        while True:
                            q.get_nowait()
                    except queue.Empty:
                        pass

                world.tick()

                try:
                    lidar_meas = lidar_queue.get(timeout=2.0)
                    rgb_image_raw = rgb_queue.get(timeout=2.0)
                except queue.Empty:
                    print("Missing sensor data, skipping")
                    continue

                points_lidar = parse_lidar_measurement(lidar_meas)
                rgb_image = parse_rgb_image(rgb_image_raw)

                # Project LiDAR to camera
                uv, valid_mask, depths = project_lidar_to_camera(
                    points_lidar,
                    lidar_to_camera,
                    intrinsic,
                    config.image_width,
                    config.image_height,
                )

                # Run YOLO + SAM2 detection
                vision_detections = vision_detector.detect_and_segment(
                    rgb_image,
                    conf_threshold=config.yolo_conf,
                )

                print(f"\n[{town_name}] Frame {frame_idx + 1}: "
                      f"{len(points_lidar)} LiDAR points, "
                      f"{len(vision_detections)} cars detected")

                # Track detections for this frame
                frame_detections = 0

                # Process each detected car
                for det_id, vis_det in enumerate(vision_detections):
                    mask = vis_det["mask"]
                    bbox = vis_det["bbox"]
                    yolo_conf = vis_det["conf"]
                    sam2_score = vis_det["mask_score"]

                    # Filter LiDAR points by SAM2 mask
                    filtered_points = filter_points_by_mask(
                        points_lidar, uv, valid_mask, mask
                    )

                    num_points = len(filtered_points)
                    print(f"  Detection {det_id}: {num_points} pts, "
                          f"YOLO={yolo_conf:.2f}, SAM2={sam2_score:.2f}")

                    if num_points < 10:
                        print("    Too few points, skipping")
                        continue

                    # Run 3D detection with error handling for CUDA issues
                    try:
                        detections_3d = run_centerpoint_detection(
                            model,
                            filtered_points,
                            score_thr=config.score_thr,
                        )
                    except RuntimeError as e:
                        if "cuda" in str(e).lower():
                            print(f"    [WARNING] CUDA error: {e}")
                            print("    Clearing CUDA cache and skipping...")
                            import torch
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                    if not detections_3d:
                        print("    No 3D detection, skipping")
                        continue

                    best_det = max(
                        detections_3d, key=lambda d: d["score"]
                    )

                    pred_dx = float(best_det["center"][0])
                    pred_dy = float(best_det["center"][1])
                    pred_dz = float(best_det["center"][2])
                    pred_yaw = float(best_det["yaw"])
                    score = float(best_det["score"])
                    pred_yaw_deg = math.degrees(pred_yaw)

                    # Match to ground truth target
                    gt = match_detection_to_target(
                        pred_dx, pred_dy,
                        targets, ego,
                        lidar_z_offset=config.lidar_z_offset,
                        max_match_dist=10.0,
                    )

                    if gt is None:
                        print("    No GT match, skipping")
                        continue

                    gt_dx = gt["gt_dx"]
                    gt_dy = gt["gt_dy"]
                    gt_dz = gt["gt_dz"]
                    gt_yaw = gt["gt_yaw"]
                    gt_yaw_deg = math.degrees(gt_yaw)

                    # Filter out targets facing ego
                    if gt_yaw_deg > 120 or gt_yaw_deg < -120:
                        print("    Target facing ego, skipping")
                        continue

                    # Compute errors
                    err_dx = pred_dx - gt_dx
                    err_dy = pred_dy - gt_dy
                    err_dz = pred_dz - gt_dz
                    err_yaw = wrap_angle_rad(pred_yaw - gt_yaw)
                    err_yaw_deg = math.degrees(err_yaw)

                    # Filter outliers
                    if abs(err_dx) > config.max_err_dx:
                        print(f"    Outlier: |err_dx|="
                              f"{abs(err_dx):.2f}m > {config.max_err_dx}m")
                        continue
                    if abs(err_dy) > config.max_err_dy:
                        print(f"    Outlier: |err_dy|="
                              f"{abs(err_dy):.2f}m > {config.max_err_dy}m")
                        continue
                    if abs(err_yaw_deg) > config.max_err_yaw:
                        print(f"    Outlier: |err_yaw|="
                              f"{abs(err_yaw_deg):.1f}° > "
                              f"{config.max_err_yaw}°")
                        continue

                    print(f"    Pred=({pred_dx:.1f}, {pred_dy:.1f}, "
                          f"{pred_yaw_deg:.1f}°), score={score:.2f}")
                    print(f"    GT  =({gt_dx:.1f}, {gt_dy:.1f}, "
                          f"{gt_yaw_deg:.1f}°)")
                    print(f"    Err =({err_dx:.2f}, {err_dy:.2f}, "
                          f"{err_yaw_deg:.1f}°)")

                    writer.writerow([
                        global_frame_id,
                        town_name,
                        pred_dx, pred_dy, pred_dz, pred_yaw_deg,
                        gt_dx, gt_dy, gt_dz, gt_yaw_deg,
                        err_dx, err_dy, err_dz, err_yaw_deg,
                        bbox[0], bbox[1], bbox[2], bbox[3],
                    ])
                    frame_detections += 1
                    total_detections += 1

                # Only save image and increment frame_id if we have detections
                if frame_detections > 0:
                    rgb_path = os.path.join(
                        config.output_dir, f"rgb_{global_frame_id:05d}.png"
                    )
                    save_rgb_image(rgb_image_raw, rgb_path)
                    global_frame_id += 1
                    csvfile.flush()
                    print(f"  -> Saved frame {global_frame_id - 1} "
                          f"with {frame_detections} detection(s)")
                else:
                    print("  -> No valid detections, frame not saved")

                # Track progress for resume capability
                completed_frame_idx = frame_idx + 1

        finally:
            destroy_actors(waypoint_actors)
            for _ in range(5):
                world.tick()

    return global_frame_id, total_detections, completed_frame_idx


def run_collection(config: Config) -> None:
    """
    Run dataset collection across multiple towns.
    Simple logic: on any failure, read CSV and resume from there.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.isfile(config.centerpoint_config):
        raise FileNotFoundError(
            f"CenterPoint config not found: {config.centerpoint_config}"
        )
    if not os.path.isfile(config.centerpoint_checkpoint):
        raise FileNotFoundError(
            f"CenterPoint checkpoint not found: {config.centerpoint_checkpoint}"
        )

    # Load models (once)
    model = load_centerpoint_model(
        config.centerpoint_config,
        config.centerpoint_checkpoint,
    )

    print("Initializing YOLO + SAM2...")
    vision_detector = VisionDetector(
        yolo_path=config.yolo_model,
        sam2_checkpoint=config.sam2_checkpoint,
        sam2_config=config.sam2_config,
        sam2_path=config.sam2_path,
    )

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(30.0)

    # Main loop - keep trying until all towns complete
    while True:
        try:
            # Read CSV to get current state
            if config.fresh_start and not os.path.exists(config.csv_output):
                # First run with fresh start
                last_town, town_frame_counts = None, {}
                global_frame_id, total_detections = 0, 0
            else:
                last_town, town_frame_counts, global_frame_id, total_detections = \
                    get_resume_info(config.csv_output)

            # Find which towns still need collection
            towns_to_collect = []
            for town in config.towns:
                frames_collected = town_frame_counts.get(town, 0)
                if frames_collected < config.frames_per_town:
                    towns_to_collect.append((town, frames_collected))

            if not towns_to_collect:
                print("\n" + "=" * 60)
                print("All towns complete!")
                print(f"Total frames: {global_frame_id}")
                print(f"Total detections: {total_detections}")
                print("=" * 60)
                break

            print(f"\n{'#'*60}")
            print("Towns remaining:")
            for town, frames in towns_to_collect:
                print(f"  {town}: {frames}/{config.frames_per_town} frames")
            print("#" * 60)

            # Open CSV in append mode (or write mode if fresh)
            file_mode = "a" if last_town else "w"
            with open(config.csv_output, file_mode, newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header if new file
                if not last_town:
                    writer.writerow([
                        "frame_id", "town",
                        "pred_dx_m", "pred_dy_m", "pred_dz_m", "pred_yaw_deg",
                        "gt_dx_m", "gt_dy_m", "gt_dz_m", "gt_yaw_deg",
                        "err_dx_m", "err_dy_m", "err_dz_m", "err_yaw_deg",
                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    ])

                # Collect from each remaining town
                for town_name, start_frame_idx in towns_to_collect:
                    result = collect_from_town(
                        client=client,
                        town_name=town_name,
                        config=config,
                        model=model,
                        vision_detector=vision_detector,
                        writer=writer,
                        csvfile=csvfile,
                        global_frame_id=global_frame_id,
                        total_detections=total_detections,
                        start_frame_idx=start_frame_idx,
                    )
                    global_frame_id = result[0]
                    total_detections = result[1]

        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Waiting 30 seconds, then will reconnect and resume...")
            time.sleep(30)
            # Reconnect to CARLA (in case it restarted)
            try:
                client = carla.Client(config.carla_host, config.carla_port)
                client.set_timeout(60.0)
                print("Reconnected to CARLA")
            except Exception as conn_err:
                print(f"Reconnection failed: {conn_err}, will retry...")
            # Loop continues - will re-read CSV and resume
