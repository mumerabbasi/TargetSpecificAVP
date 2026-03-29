"""Stage A: capture reusable raw CARLA frames for per-target dataset building."""

from __future__ import annotations

import json
import os
import queue
import shutil
from collections import defaultdict
from typing import Dict, List

import carla
import numpy as np

from .carla_utils import (
    SensorRig,
    collect_visible_vehicle_records,
    configure_traffic_manager,
    destroy_actors,
    ego_on_driving_lane,
    save_rgb_image,
    setup_world,
    spawn_background_traffic,
    spawn_ego_vehicle,
)
from .config import Config
from .ground_truth import actor_is_follow_valid
from .utils import ensure_dir, relative_path


def _discover_next_frame_id(metadata_dir: str) -> int:
    if not os.path.isdir(metadata_dir):
        return 0

    max_id = -1
    for name in os.listdir(metadata_dir):
        if not name.startswith("frame_") or not name.endswith(".json"):
            continue
        try:
            frame_id = int(name[len("frame_") : -len(".json")])
        except ValueError:
            continue
        max_id = max(max_id, frame_id)
    return max_id + 1


def _load_existing_counts(config: Config) -> Dict[str, Dict[int, int]]:
    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    if not os.path.isdir(config.raw_metadata_dir):
        return counts

    for name in sorted(os.listdir(config.raw_metadata_dir)):
        if not name.endswith(".json"):
            continue
        meta_path = os.path.join(config.raw_metadata_dir, name)
        with open(meta_path, "r") as f:
            payload = json.load(f)
        town = payload["town"]
        for actor in payload.get("visible_actors", []):
            counts[town][int(actor["distance_bin"])] += 1
    return counts


def _wipe_capture_output(config: Config) -> None:
    for path in (config.raw_capture_dir,):
        if os.path.isdir(path):
            shutil.rmtree(path)


def _save_raw_frame(
    config: Config,
    frame_id: int,
    episode_id: int,
    town: str,
    snapshot,
    ego: carla.Vehicle,
    visible_actors: List[Dict[str, object]],
) -> None:
    rgb_path = os.path.join(config.raw_rgb_dir, f"frame_{frame_id:06d}.png")
    lidar_path = os.path.join(config.raw_lidar_dir, f"frame_{frame_id:06d}.npy")
    instance_path = os.path.join(
        config.raw_instance_dir, f"frame_{frame_id:06d}.npy"
    )
    meta_path = os.path.join(config.raw_metadata_dir, f"frame_{frame_id:06d}.json")

    save_rgb_image(snapshot.rgb_raw, rgb_path)
    np.save(lidar_path, snapshot.lidar)
    np.save(instance_path, snapshot.instance)

    ego_tf = ego.get_transform()
    payload = {
        "frame_id": frame_id,
        "episode_id": episode_id,
        "town": town,
        "tick": snapshot.tick,
        "rgb_path": relative_path(rgb_path, config.output_dir),
        "lidar_path": relative_path(lidar_path, config.output_dir),
        "instance_path": relative_path(instance_path, config.output_dir),
        "ego": {
            "location": {
                "x": float(ego_tf.location.x),
                "y": float(ego_tf.location.y),
                "z": float(ego_tf.location.z),
            },
            "rotation_yaw_deg": float(ego_tf.rotation.yaw),
            "velocity": {
                "x": float(ego.get_velocity().x),
                "y": float(ego.get_velocity().y),
                "z": float(ego.get_velocity().z),
            },
        },
        "camera": {
            "image_width": config.image_width,
            "image_height": config.image_height,
            "fov": config.fov,
        },
        "visible_actors": visible_actors,
    }

    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)


def _prepare_capture_dirs(config: Config) -> None:
    if config.fresh_start:
        _wipe_capture_output(config)
    for path in config.capture_dirs:
        ensure_dir(path)


def _town_total(counts: Dict[int, int]) -> int:
    return int(sum(counts.values()))


def capture_raw_data(config: Config) -> None:
    """Run continuous-traffic CARLA episodes and save accepted raw frames."""
    _prepare_capture_dirs(config)
    counts = _load_existing_counts(config)
    next_frame_id = _discover_next_frame_id(config.raw_metadata_dir)

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(config.client_timeout_s)

    for town in config.towns:
        town_counts = counts[town]
        town_total = _town_total(town_counts)
        if town_total >= config.target_samples_per_town:
            print(f"[capture] {town} already complete with {town_total} target samples")
            continue

        print(f"\n[capture] Loading {town}")
        world = setup_world(client, town, config)
        traffic_manager = None
        if config.traffic_mode == "traffic_manager":
            traffic_manager = configure_traffic_manager(client, world, config)

        saved_frames = 0
        for episode_idx in range(config.max_episodes_per_town):
            if town_total >= config.target_samples_per_town:
                break
            if saved_frames >= config.max_frames_per_town:
                break

            print(f"[capture] {town}: starting episode {episode_idx}")
            actors_to_cleanup: List[object] = []
            rig = None
            try:
                ego = spawn_ego_vehicle(
                    world,
                    config,
                    traffic_manager=traffic_manager,
                )
                actors_to_cleanup.append(ego)

                traffic_ids = spawn_background_traffic(
                    client,
                    world,
                    ego,
                    config.num_traffic_vehicles,
                    config,
                    traffic_manager=traffic_manager,
                )
                actors_to_cleanup.extend(traffic_ids)

                rig = SensorRig(world, ego, config)
                rig.warmup(config.warmup_ticks)

                episode_frames = 0
                offroad_ticks = 0
                while (
                    episode_frames < config.episode_frame_budget
                    and saved_frames < config.max_frames_per_town
                    and town_total < config.target_samples_per_town
                ):
                    world.tick()
                    episode_frames += 1

                    if not ego_on_driving_lane(world, ego):
                        offroad_ticks += 1
                        if offroad_ticks >= 5:
                            print(
                                f"[capture] {town}: restarting episode after ego left "
                                "the driving lane"
                            )
                            break
                        continue
                    offroad_ticks = 0

                    try:
                        snapshot = rig.get_snapshot(timeout=2.0)
                    except queue.Empty:
                        continue

                    visible_actors = collect_visible_vehicle_records(
                        world,
                        ego,
                        snapshot,
                        config,
                    )
                    if not visible_actors:
                        continue

                    target_actors = visible_actors
                    if config.follow_only:
                        target_actors = [
                            actor
                            for actor in visible_actors
                            if actor_is_follow_valid(actor, config)
                        ]
                        if (
                            len(target_actors)
                            < config.min_follow_actors_per_frame
                        ):
                            continue
                        if (
                            config.max_follow_actors_per_frame > 0
                            and len(target_actors)
                            > config.max_follow_actors_per_frame
                        ):
                            continue

                    useful_actors = [
                        actor
                        for actor in target_actors
                        if town_counts[int(actor["distance_bin"])] < config.per_distance_bin_target
                    ]
                    if not useful_actors and town_total < config.target_samples_per_town:
                        continue

                    _save_raw_frame(
                        config,
                        next_frame_id,
                        episode_idx,
                        town,
                        snapshot,
                        ego,
                        visible_actors,
                    )

                    next_frame_id += 1
                    saved_frames += 1
                    for actor in target_actors:
                        town_counts[int(actor["distance_bin"])] += 1
                    town_total = _town_total(town_counts)

                    if saved_frames % 25 == 0:
                        print(
                            f"[capture] {town}: saved {saved_frames} frames, "
                            f"{town_total}/{config.target_samples_per_town} target samples"
                        )

            finally:
                if rig is not None:
                    rig.destroy()
                destroy_actors(world, actors_to_cleanup)
                for _ in range(5):
                    world.tick()

        print(
            f"[capture] {town}: finished with {_town_total(town_counts)} target "
            f"samples across {saved_frames} saved frames"
        )
