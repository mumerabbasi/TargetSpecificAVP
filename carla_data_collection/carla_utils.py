"""CARLA helpers for the single-pass collection pipeline."""

from __future__ import annotations

import math
import queue
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import carla
import numpy as np

from .config import Config
from .ground_truth import compute_relative_pose_from_transforms, distance_bin_index
from .utils import (
    bbox_touches_edge,
    binary_mask_to_bbox,
    get_camera_intrinsic,
    project_world_points_to_image,
)


def setup_world(
    client: carla.Client,
    town: str,
    config: Config,
) -> carla.World:
    """Load a town and enable synchronous stepping."""
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = config.sync_mode
    settings.fixed_delta_seconds = config.fixed_delta_seconds
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    return world


def configure_traffic_manager(
    client: carla.Client,
    world: carla.World,
    config: Config,
) -> carla.TrafficManager:
    """Configure Traffic Manager for long-running natural traffic episodes."""
    traffic_manager = client.get_trafficmanager(config.tm_port)
    traffic_manager.set_synchronous_mode(config.sync_mode)
    traffic_manager.set_global_distance_to_leading_vehicle(
        config.traffic_follow_distance_m
    )
    traffic_manager.global_percentage_speed_difference(
        config.background_speed_difference_pct
    )
    traffic_manager.set_respawn_dormant_vehicles(True)
    return traffic_manager


def _choose_vehicle_blueprints(
        world: carla.World) -> list[carla.ActorBlueprint]:
    vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
    preferred = [
        bp
        for bp in vehicle_bps
        if any(
            token in bp.id.lower()
            for token in ["mini", "a2", "tt", "prius", "cooper", "c3", "model3"]
        )
    ]
    return preferred if preferred else list(vehicle_bps)


def spawn_ego_vehicle(
    world: carla.World,
    config: Config,
    traffic_manager: Optional[carla.TrafficManager] = None,
    spawn_point: Optional[carla.Transform] = None,
) -> carla.Vehicle:
    """Spawn the ego vehicle and enable autopilot."""
    spawn_points = world.get_map().get_spawn_points()
    if spawn_point is None:
        spawn_point = random.choice(spawn_points)

    ego_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    ego_bp.set_attribute("role_name", "ego")
    ego = world.try_spawn_actor(ego_bp, spawn_point)
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")

    if traffic_manager is not None:
        ego.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(
            ego, config.ego_speed_difference_pct
        )
    else:
        ego.enable_constant_velocity(
            carla.Vector3D(config.constant_velocity_ego_speed_mps, 0.0, 0.0)
        )
    return ego


def spawn_background_traffic(
    client: carla.Client,
    world: carla.World,
    ego: carla.Vehicle,
    num_vehicles: int,
    config: Config,
    traffic_manager: Optional[carla.TrafficManager] = None,
) -> List[int]:
    """Spawn background traffic vehicles on autopilot."""
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    ego_loc = ego.get_location()
    spawn_points = [
        sp for sp in spawn_points if sp.location.distance(ego_loc) > 8.0
    ]

    blueprints = _choose_vehicle_blueprints(world)
    if traffic_manager is not None:
        batch = []
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        for spawn_point in spawn_points[:num_vehicles]:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(
                SpawnActor(blueprint, spawn_point).then(
                    SetAutopilot(FutureActor, True, traffic_manager.get_port())
                )
            )
        actor_ids: List[int] = []
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                continue
            actor_ids.append(response.actor_id)
        return actor_ids

    actor_ids = []
    for spawn_point in spawn_points[:num_vehicles]:
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(
                blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        blueprint.set_attribute("role_name", "constant_velocity")
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is None:
            continue
        vehicle.enable_constant_velocity(
            carla.Vector3D(
                random.uniform(
                    config.constant_velocity_background_min_speed_mps,
                    config.constant_velocity_background_max_speed_mps,
                ),
                0.0,
                0.0,
            )
        )
        actor_ids.append(vehicle.id)

    return actor_ids


@dataclass
class SensorSnapshot:
    """Parsed synchronized sensor frame."""

    tick: int
    rgb_raw: carla.Image
    rgb: np.ndarray
    instance: np.ndarray
    lidar: np.ndarray
    camera_world_matrix: np.ndarray
    intrinsic: np.ndarray


class SensorRig:
    """Synchronized RGB, instance, and LiDAR sensors attached to ego."""

    def __init__(
            self,
            world: carla.World,
            ego: carla.Vehicle,
            config: Config) -> None:
        self.world = world
        self.ego = ego
        self.config = config

        self.rgb_queue: queue.Queue = queue.Queue()
        self.instance_queue: queue.Queue = queue.Queue()
        self.lidar_queue: queue.Queue = queue.Queue()

        bp_lib = world.get_blueprint_library()
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.6),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        self.camera_transform_local = camera_transform

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(config.image_width))
        rgb_bp.set_attribute("image_size_y", str(config.image_height))
        rgb_bp.set_attribute("fov", str(config.fov))

        instance_bp = bp_lib.find("sensor.camera.instance_segmentation")
        instance_bp.set_attribute("image_size_x", str(config.image_width))
        instance_bp.set_attribute("image_size_y", str(config.image_height))
        instance_bp.set_attribute("fov", str(config.fov))

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

        self.rgb_camera = world.spawn_actor(
            rgb_bp, camera_transform, attach_to=ego)
        self.instance_camera = world.spawn_actor(
            instance_bp, camera_transform, attach_to=ego
        )

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=config.lidar_z_offset),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        self.lidar_transform_local = lidar_transform
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=ego)

        self.rgb_camera.listen(self.rgb_queue.put)
        self.instance_camera.listen(self.instance_queue.put)
        self.lidar_sensor.listen(self.lidar_queue.put)

    def warmup(self, ticks: int) -> None:
        for _ in range(ticks):
            self.world.tick()
            self.clear_queues()

    def clear_queues(self) -> None:
        for q in (self.rgb_queue, self.instance_queue, self.lidar_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def get_snapshot(self, timeout: float = 2.0) -> SensorSnapshot:
        packets = {
            "rgb": self.rgb_queue.get(timeout=timeout),
            "instance": self.instance_queue.get(timeout=timeout),
            "lidar": self.lidar_queue.get(timeout=timeout),
        }

        while True:
            frames = {name: int(packet.frame)
                      for name, packet in packets.items()}
            if len(set(frames.values())) == 1:
                break

            target_frame = max(frames.values())
            if frames["rgb"] < target_frame:
                packets["rgb"] = self.rgb_queue.get(timeout=timeout)
            if frames["instance"] < target_frame:
                packets["instance"] = self.instance_queue.get(timeout=timeout)
            if frames["lidar"] < target_frame:
                packets["lidar"] = self.lidar_queue.get(timeout=timeout)

        rgb_raw = packets["rgb"]
        instance_raw = packets["instance"]
        lidar_raw = packets["lidar"]

        rgb = parse_rgb_image(rgb_raw)
        instance = parse_instance_image(instance_raw)
        lidar = parse_lidar_measurement(lidar_raw)
        camera_world_matrix = np.array(
            self.rgb_camera.get_transform().get_matrix())
        intrinsic = get_camera_intrinsic(
            self.config.image_width,
            self.config.image_height,
            self.config.fov,
        )
        return SensorSnapshot(
            tick=int(rgb_raw.frame),
            rgb_raw=rgb_raw,
            rgb=rgb,
            instance=instance,
            lidar=lidar,
            camera_world_matrix=camera_world_matrix,
            intrinsic=intrinsic,
        )

    def destroy(self) -> None:
        for sensor in (
                self.rgb_camera,
                self.instance_camera,
                self.lidar_sensor):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()


def destroy_actors(
        world: carla.World,
        actors_or_ids: Sequence[object]) -> None:
    """Safely destroy actors represented either as ids or actor objects."""
    for item in actors_or_ids:
        actor = item
        if isinstance(item, int):
            actor = world.get_actor(item)
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


def ego_on_driving_lane(world: carla.World, ego: carla.Vehicle) -> bool:
    """Return whether the ego currently sits on a drivable lane."""
    waypoint = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    return waypoint is not None and waypoint.lane_type == carla.LaneType.Driving


def parse_lidar_measurement(lidar_meas: carla.LidarMeasurement) -> np.ndarray:
    """Convert CARLA LiDAR measurement to Nx4 numpy array."""
    data = np.frombuffer(lidar_meas.raw_data, dtype=np.float32)
    return np.reshape(data, (-1, 4))


def parse_rgb_image(image: carla.Image) -> np.ndarray:
    """Parse CARLA RGB image into RGB uint8 format."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def parse_instance_image(image: carla.Image) -> np.ndarray:
    """Parse CARLA instance segmentation image to BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array.copy()


def get_instance_ids(instance_image: np.ndarray) -> np.ndarray:
    """Extract packed instance ids from a CARLA instance image."""
    return (
        instance_image[:, :, 0].astype(np.uint32)
        + instance_image[:, :, 1].astype(np.uint32) * 256
    )


def get_semantic_tags(instance_image: np.ndarray) -> np.ndarray:
    """Extract semantic tags from a CARLA instance image."""
    return instance_image[:, :, 2]


def vehicle_instance_mask(
    instance_image: np.ndarray,
    instance_id: int,
    vehicle_semantic_tag: int,
) -> np.ndarray:
    """Return a binary mask for one vehicle instance."""
    semantic_tags = get_semantic_tags(instance_image)
    instance_ids = get_instance_ids(instance_image)
    return (
        semantic_tags == vehicle_semantic_tag) & (
        instance_ids == instance_id)


def _location_to_dict(location: carla.Location) -> Dict[str, float]:
    return {
        "x": float(
            location.x), "y": float(
            location.y), "z": float(
                location.z)}


def _velocity_to_dict(velocity: carla.Vector3D) -> Dict[str, float]:
    return {
        "x": float(
            velocity.x), "y": float(
            velocity.y), "z": float(
                velocity.z)}


def speed_mps(actor: carla.Vehicle) -> float:
    velocity = actor.get_velocity()
    return float(
        math.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2 +
            velocity.z ** 2))


def project_actor_bbox(
    actor: carla.Vehicle,
    camera_world_matrix: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Project an actor 3D bounding box into image space."""
    world_vertices = actor.bounding_box.get_world_vertices(
        actor.get_transform())
    world_points = np.asarray(
        [[vertex.x, vertex.y, vertex.z] for vertex in world_vertices],
        dtype=np.float64,
    )
    uv, valid_mask, _ = project_world_points_to_image(
        world_points,
        camera_world_matrix,
        intrinsic,
        img_width,
        img_height,
    )
    if not np.any(valid_mask):
        return None

    uv = uv[valid_mask]
    x1 = max(0, int(np.floor(uv[:, 0].min())))
    y1 = max(0, int(np.floor(uv[:, 1].min())))
    x2 = min(img_width - 1, int(np.ceil(uv[:, 0].max())))
    y2 = min(img_height - 1, int(np.ceil(uv[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def infer_actor_instance_id(
    actor: carla.Vehicle,
    projected_bbox: Tuple[int, int, int, int],
    instance_image: np.ndarray,
    config: Config,
    used_instance_ids: set[int],
) -> Optional[Tuple[int, np.ndarray, Tuple[int, int, int, int]]]:
    """Infer which CARLA instance id belongs to a projected actor."""
    x1, y1, x2, y2 = projected_bbox
    x1 = max(0, x1 - config.instance_bbox_dilation_px)
    y1 = max(0, y1 - config.instance_bbox_dilation_px)
    x2 = min(config.image_width - 1, x2 + config.instance_bbox_dilation_px)
    y2 = min(config.image_height - 1, y2 + config.instance_bbox_dilation_px)

    semantic_tags = get_semantic_tags(instance_image)
    instance_ids = get_instance_ids(instance_image)

    crop_semantic = semantic_tags[y1: y2 + 1, x1: x2 + 1]
    crop_instances = instance_ids[y1: y2 + 1, x1: x2 + 1]
    vehicle_pixels = crop_instances[crop_semantic ==
                                    config.vehicle_semantic_tag]

    if vehicle_pixels.size == 0:
        return None

    unique_ids, counts = np.unique(vehicle_pixels, return_counts=True)
    ranked = sorted(
        zip(unique_ids.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    for instance_id, _ in ranked:
        if instance_id in used_instance_ids:
            continue

        mask = vehicle_instance_mask(
            instance_image, instance_id, config.vehicle_semantic_tag
        )
        bbox = binary_mask_to_bbox(mask)
        if bbox is None:
            continue

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        pixel_count = int(mask.sum())
        if pixel_count < config.min_visible_vehicle_pixels:
            continue
        if bbox_width < config.min_visible_bbox_width:
            continue
        if bbox_height < config.min_visible_bbox_height:
            continue
        if bbox_touches_edge(
            bbox,
            config.image_width,
            config.image_height,
            config.edge_margin_px,
        ):
            continue

        return int(instance_id), mask, bbox

    return None


def collect_visible_vehicle_records(
    world: carla.World,
    ego: carla.Vehicle,
    snapshot: SensorSnapshot,
    config: Config,
) -> List[Dict[str, object]]:
    """Build per-vehicle visibility records for one frame."""
    records: List[Dict[str, object]] = []
    used_instance_ids: set[int] = set()

    ego_tf = ego.get_transform()
    ego_location = _location_to_dict(ego_tf.location)
    ego_yaw_deg = float(ego_tf.rotation.yaw)

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id or not actor.is_alive:
            continue
        if actor.get_location().distance(
                ego_tf.location) > config.nearby_vehicle_radius_m:
            continue

        projected_bbox = project_actor_bbox(
            actor,
            snapshot.camera_world_matrix,
            snapshot.intrinsic,
            config.image_width,
            config.image_height,
        )
        if projected_bbox is None:
            continue

        inferred = infer_actor_instance_id(
            actor,
            projected_bbox,
            snapshot.instance,
            config,
            used_instance_ids,
        )
        if inferred is None:
            continue

        instance_id, mask, bbox = inferred
        used_instance_ids.add(instance_id)

        actor_tf = actor.get_transform()
        pose = compute_relative_pose_from_transforms(
            ego_location=ego_location,
            ego_yaw_deg=ego_yaw_deg,
            target_location=_location_to_dict(actor_tf.location),
            target_yaw_deg=float(actor_tf.rotation.yaw),
            target_half_height_m=float(actor.bounding_box.extent.z),
            lidar_z_offset=config.lidar_z_offset,
        )

        record = {
            "actor_id": int(actor.id),
            "instance_id": int(instance_id),
            "type_id": actor.type_id,
            "role_name": actor.attributes.get("role_name", ""),
            "pixel_area": int(mask.sum()),
            "bbox_x1": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x2": int(bbox[2]),
            "bbox_y2": int(bbox[3]),
            "distance_bin": int(
                distance_bin_index(pose["dx_m"], config.distance_bins_m)
            ),
            "speed_mps": speed_mps(actor),
            "location": _location_to_dict(actor_tf.location),
            "rotation_yaw_deg": float(actor_tf.rotation.yaw),
            "velocity": _velocity_to_dict(actor.get_velocity()),
            **pose,
        }
        records.append(record)

    return records
