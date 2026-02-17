"""CARLA world setup and actor management."""

import math
import queue
import random
from typing import Any, List, Optional, Tuple

import carla
import cv2
import numpy as np

from .config import Config


def setup_world(client: carla.Client) -> carla.World:
    """
    Enable synchronous mode for CARLA world.

    Args:
        client: CARLA client.

    Returns:
        Configured world.
    """
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return world


def spawn_ego_vehicle(
    world: carla.World,
    actor_list: List[carla.Actor],
    spawn_point: Optional[carla.Transform] = None,
    config: Optional[Config] = None,
) -> carla.Vehicle:
    """
    Spawn ego vehicle at specified or random spawn point.

    Applies a small Gaussian yaw offset for scene diversity.

    Args:
        world: CARLA world.
        actor_list: List to append spawned actor.
        spawn_point: Optional spawn location.
        config: Optional config for ego yaw parameters.

    Returns:
        Spawned ego vehicle.
    """
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    ego_bp.set_attribute("role_name", "ego")

    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

    # Apply Gaussian yaw offset for scene diversity
    if config is not None:
        dyaw = sample_gaussian_clipped(
            config.ego_dyaw_mean,
            config.ego_dyaw_std,
            config.ego_dyaw_min,
            config.ego_dyaw_max,
        )
        spawn_point = carla.Transform(
            location=spawn_point.location,
            rotation=carla.Rotation(
                pitch=spawn_point.rotation.pitch,
                yaw=spawn_point.rotation.yaw + dyaw,
                roll=spawn_point.rotation.roll,
            ),
        )

    ego = world.try_spawn_actor(ego_bp, spawn_point)
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")

    actor_list.append(ego)
    return ego


def setup_sensors(
    world: carla.World,
    ego: carla.Vehicle,
    actor_list: List[carla.Actor],
    config: Config,
) -> Tuple[Any, Any, queue.Queue, queue.Queue, carla.Transform, carla.Transform]:
    """
    Attach RGB camera and LiDAR to ego.

    Args:
        world: CARLA world.
        ego: Ego vehicle.
        actor_list: List to append spawned actors.
        config: Configuration.

    Returns:
        lidar_actor, rgb_camera, lidar_queue, rgb_queue,
        lidar_transform, camera_transform
    """
    bp_lib = world.get_blueprint_library()

    camera_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=1.6),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    # RGB Camera
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(config.image_width))
    rgb_bp.set_attribute("image_size_y", str(config.image_height))
    rgb_bp.set_attribute("fov", str(config.fov))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=ego)
    actor_list.append(rgb_camera)

    # LiDAR
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "80.0")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("points_per_second", "600000")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("upper_fov", "2.0")
    lidar_bp.set_attribute("lower_fov", "-24.8")
    lidar_bp.set_attribute("dropoff_general_rate", "0.0")
    lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
    lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

    lidar_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=config.lidar_z_offset),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )
    lidar_actor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego)
    actor_list.append(lidar_actor)

    # Queues
    lidar_queue: queue.Queue = queue.Queue()
    rgb_queue: queue.Queue = queue.Queue()

    lidar_actor.listen(lidar_queue.put)
    rgb_camera.listen(rgb_queue.put)

    return (
        lidar_actor, rgb_camera,
        lidar_queue, rgb_queue,
        lidar_transform, camera_transform,
    )


def sample_gaussian_clipped(
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Sample from Gaussian distribution, clipped to [min_val, max_val].

    Args:
        mean: Gaussian mean.
        std: Gaussian standard deviation.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clipped sample.
    """
    value = random.gauss(mean, std)
    return max(min_val, min(max_val, value))


def ego_to_world_offset(
    ego_transform: carla.Transform,
    dx_ego: float,
    dy_ego: float,
    dz_ego: float = 0.0,
) -> carla.Location:
    """
    Convert ego-frame offset to world location.

    Args:
        ego_transform: Ego vehicle transform.
        dx_ego: Forward offset in ego frame.
        dy_ego: Right offset in ego frame.
        dz_ego: Up offset in ego frame.

    Returns:
        World location.
    """
    x_e = ego_transform.location.x
    y_e = ego_transform.location.y
    yaw_e = math.radians(ego_transform.rotation.yaw)

    cos_e = math.cos(yaw_e)
    sin_e = math.sin(yaw_e)

    dx_world = cos_e * dx_ego - sin_e * dy_ego
    dy_world = sin_e * dx_ego + cos_e * dy_ego

    return carla.Location(
        x=x_e + dx_world,
        y=y_e + dy_world,
        z=ego_transform.location.z + dz_ego,
    )


def spawn_target_vehicles(
    world: carla.World,
    ego: carla.Vehicle,
    actor_list: List[carla.Actor],
    num_targets: int,
    config: Config,
) -> List[carla.Vehicle]:
    """
    Spawn target vehicles around ego using Gaussian distributions.

    Uses Gaussian sampling for more realistic scenarios:
    - Most targets are at typical following distances
    - Most targets are roughly in front (small lateral offset)
    - Most targets face similar direction (small yaw difference)
    - Extreme cases (far, side, perpendicular) are rarer

    Args:
        world: CARLA world.
        ego: Ego vehicle.
        actor_list: List to append spawned actors.
        num_targets: Number of targets to spawn.
        config: Configuration with Gaussian parameters.

    Returns:
        List of spawned target vehicles.
    """
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    ego_tf = ego.get_transform()

    vehicle_bps = bp_lib.filter("vehicle.*")
    preferred = [
        bp for bp in vehicle_bps
        if any(v in bp.id.lower() for v in
               ["mini", "a2", "tt", "prius", "cooper", "c3"])
    ]
    if not preferred:
        preferred = list(vehicle_bps)

    targets = []
    used_waypoints: List[carla.Waypoint] = []
    min_waypoint_separation = 4.0

    for i in range(num_targets):
        for _ in range(30):
            # Sample from Gaussian distributions
            dx = sample_gaussian_clipped(
                config.target_dx_mean, config.target_dx_std,
                config.target_dx_min, config.target_dx_max,
            )
            dy = sample_gaussian_clipped(
                config.target_dy_mean, config.target_dy_std,
                config.target_dy_min, config.target_dy_max,
            )
            delta_yaw = sample_gaussian_clipped(
                config.target_dyaw_mean, config.target_dyaw_std,
                config.target_dyaw_min, config.target_dyaw_max,
            )

            target_loc = ego_to_world_offset(ego_tf, dx, dy, 0.0)

            waypoint = carla_map.get_waypoint(
                target_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if waypoint is None:
                continue

            too_close = False
            for used_wp in used_waypoints:
                dist = waypoint.transform.location.distance(
                    used_wp.transform.location
                )
                if dist < min_waypoint_separation:
                    too_close = True
                    break

            if too_close:
                continue

            wp_loc = waypoint.transform.location
            ego_loc = ego_tf.location

            dx_world = wp_loc.x - ego_loc.x
            dy_world = wp_loc.y - ego_loc.y
            yaw_e = math.radians(ego_tf.rotation.yaw)
            cos_e, sin_e = math.cos(yaw_e), math.sin(yaw_e)

            wp_dx_ego = cos_e * dx_world + sin_e * dy_world
            wp_dy_ego = -sin_e * dx_world + cos_e * dy_world

            if wp_dx_ego < config.target_dx_min or wp_dx_ego > config.target_dx_max:
                continue
            if wp_dy_ego < config.target_dy_min or wp_dy_ego > config.target_dy_max:
                continue

            break
        else:
            continue

        wp_yaw = waypoint.transform.rotation.yaw
        target_yaw = wp_yaw + delta_yaw

        spawn_loc = waypoint.transform.location
        spawn_loc.z += 0.3

        target_tf = carla.Transform(
            location=spawn_loc,
            rotation=carla.Rotation(pitch=0.0, yaw=target_yaw, roll=0.0),
        )

        target_bp = random.choice(preferred)
        target_bp.set_attribute("role_name", f"target_{i}")

        target = world.try_spawn_actor(target_bp, target_tf)
        if target is not None:
            actor_list.append(target)
            targets.append(target)
            used_waypoints.append(waypoint)

    return targets


def destroy_actors(actors: List[carla.Actor]) -> None:
    """
    Safely destroy a list of actors.

    Args:
        actors: List of CARLA actors to destroy.
    """
    for actor in actors:
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


def get_diverse_spawn_points(
    world: carla.World,
    num_points: int,
    min_distance: float = 50.0,
) -> List[carla.Transform]:
    """
    Get diverse spawn points spread across the map.

    Args:
        world: CARLA world.
        num_points: Number of spawn points needed.
        min_distance: Minimum distance between spawn points.

    Returns:
        List of spawn point transforms.
    """
    all_spawn_points = world.get_map().get_spawn_points()

    if len(all_spawn_points) <= num_points:
        return all_spawn_points

    random.shuffle(all_spawn_points)
    selected: List[carla.Transform] = []

    for sp in all_spawn_points:
        if len(selected) >= num_points:
            break

        is_diverse = True
        for existing in selected:
            dist = sp.location.distance(existing.location)
            if dist < min_distance:
                is_diverse = False
                break

        if is_diverse:
            selected.append(sp)

    if len(selected) < num_points:
        remaining = [sp for sp in all_spawn_points if sp not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:num_points - len(selected)])

    return selected


def parse_lidar_measurement(lidar_meas: carla.LidarMeasurement) -> np.ndarray:
    """
    Convert CARLA LiDAR measurement to Nx4 array.

    Args:
        lidar_meas: CARLA LiDAR measurement.

    Returns:
        Nx4 numpy array (x, y, z, intensity).
    """
    data = np.frombuffer(lidar_meas.raw_data, dtype=np.float32)
    points = np.reshape(data, (-1, 4))
    return points


def parse_rgb_image(image: carla.Image) -> np.ndarray:
    """
    Parse CARLA RGB image to numpy array (RGB format).

    Args:
        image: CARLA image.

    Returns:
        HxWx3 RGB numpy array.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # BGRA -> RGB
    rgb = array[:, :, :3][:, :, ::-1].copy()
    return rgb


def save_rgb_image(image: carla.Image, path: str) -> None:
    """
    Save CARLA RGB image to disk.

    Args:
        image: CARLA image.
        path: Output file path.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3]
    cv2.imwrite(path, bgr)
