"""Clean CARLA scenario runtime for pursuit evaluation."""

from __future__ import annotations

import math
import queue
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import carla
import numpy as np

from .config import PursuitEvalConfig
from .geometry import (
    bbox_from_mask,
    bbox_iou,
    get_camera_intrinsic,
    wrap_angle_deg,
)


def destroy_actors(actors: Sequence[object]) -> None:
    """Destroy any still-alive CARLA actors."""
    for actor in actors:
        if actor is None:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            continue


def _parse_rgb(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def _parse_instance(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4)).copy()


def _parse_lidar(meas: carla.LidarMeasurement) -> np.ndarray:
    data = np.frombuffer(meas.raw_data, dtype=np.float32)
    points = data.reshape((-1, 4))
    return points.copy()


def _preferred_vehicle_blueprints(
        bp_lib: carla.BlueprintLibrary) -> List[carla.ActorBlueprint]:
    vehicles = [
        bp
        for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) == 4
    ]
    preferred = [
        bp
        for bp in vehicles
        if any(
            token in bp.id.lower()
            for token in (
                "model3",
                "mini",
                "a2",
                "tt",
                "prius",
                "cooper",
                "c3",
                "mustang",
            )
        )
    ]
    return preferred or vehicles


def _random_blueprint(
    blueprints: Sequence[carla.ActorBlueprint],
    role_name: str,
) -> carla.ActorBlueprint:
    blueprint = random.choice(list(blueprints))
    blueprint = blueprint.clone() if hasattr(blueprint, "clone") else blueprint
    blueprint.set_attribute("role_name", role_name)
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", random.choice(colors))
    return blueprint


def compute_relative_pose(
        ego: carla.Vehicle, target: carla.Vehicle) -> Dict[str, float]:
    """Compute target pose in ego coordinates."""
    ego_tf = ego.get_transform()
    target_tf = target.get_transform()
    dx_world = target_tf.location.x - ego_tf.location.x
    dy_world = target_tf.location.y - ego_tf.location.y
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    dx_ego = cos_yaw * dx_world + sin_yaw * dy_world
    dy_ego = -sin_yaw * dx_world + cos_yaw * dy_world
    dyaw = wrap_angle_deg(target_tf.rotation.yaw - ego_tf.rotation.yaw)
    return {
        "dx_m": float(dx_ego),
        "dy_m": float(dy_ego),
        "yaw_deg": float(dyaw),
    }


def get_vehicle_speed(vehicle: carla.Vehicle) -> float:
    """Return Euclidean speed in m/s."""
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def ego_on_driving_lane(world: carla.World, ego: carla.Vehicle) -> bool:
    """Check whether ego is still on a driving lane."""
    waypoint = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    return waypoint is not None


def follow_friendly_waypoint_chain(
    carla_map: carla.Map,
    start_location: carla.Location,
    *,
    lookahead_m: float,
    step_m: float,
    max_yaw_delta_deg: float,
) -> bool:
    """Require a mostly straight, non-junction lane ahead of the spawn."""
    waypoint = carla_map.get_waypoint(
        start_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None or waypoint.is_junction:
        return False

    start_yaw = float(waypoint.transform.rotation.yaw)
    travelled_m = 0.0
    current = waypoint
    while travelled_m < float(lookahead_m):
        next_candidates = current.next(float(step_m))
        if len(next_candidates) != 1:
            return False
        current = next_candidates[0]
        travelled_m += float(step_m)
        if current.is_junction:
            return False
        yaw_delta = abs(
            float(
                wrap_angle_deg(
                    current.transform.rotation.yaw -
                    start_yaw)))
        if yaw_delta > float(max_yaw_delta_deg):
            return False
    return True


def project_world_points_to_image(
    world_points_xyz: np.ndarray,
    camera_world_matrix: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points into image coordinates."""
    camera_inv = np.linalg.inv(camera_world_matrix)
    points_h = np.hstack(
        [world_points_xyz[:, :3], np.ones((world_points_xyz.shape[0], 1))])
    points_cam = (camera_inv @ points_h.T).T
    x_cam = points_cam[:, 1]
    y_cam = -points_cam[:, 2]
    z_cam = points_cam[:, 0]
    valid_depth = z_cam > 0.1
    u = intrinsic[0, 0] * x_cam / np.maximum(z_cam, 1e-6) + intrinsic[0, 2]
    v = intrinsic[1, 1] * y_cam / np.maximum(z_cam, 1e-6) + intrinsic[1, 2]
    valid_bounds = (
        (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )
    return np.stack([u, v], axis=1), valid_depth & valid_bounds


@dataclass
class SensorPacket:
    """Synchronized sensor data for one CARLA tick."""

    tick: int
    rgb_image: np.ndarray
    instance_image: np.ndarray
    lidar_points: np.ndarray
    spectator_image: Optional[np.ndarray] = None


class EgoSensorSuite:
    """RGB, instance, lidar, and collision sensors attached to the ego car."""

    vehicle_semantic_tag = 14

    def __init__(
            self,
            world: carla.World,
            ego: carla.Vehicle,
            config: PursuitEvalConfig) -> None:
        self.world = world
        self.ego = ego
        self.config = config
        self.rgb_queue = queue.Queue()
        self.instance_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.collision_events: List[carla.CollisionEvent] = []

        bp_lib = world.get_blueprint_library()
        self.camera_transform = carla.Transform(
            carla.Location(
                x=config.camera_x_m,
                y=config.camera_y_m,
                z=config.camera_z_m),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0),
        )
        self.lidar_transform = carla.Transform(
            carla.Location(
                x=config.lidar_x_m,
                y=config.lidar_y_m,
                z=config.lidar_z_m),
            carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0),
        )

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(config.image_width))
        rgb_bp.set_attribute("image_size_y", str(config.image_height))
        rgb_bp.set_attribute("fov", str(config.fov))

        instance_bp = bp_lib.find("sensor.camera.instance_segmentation")
        instance_bp.set_attribute("image_size_x", str(config.image_width))
        instance_bp.set_attribute("image_size_y", str(config.image_height))
        instance_bp.set_attribute("fov", str(config.fov))

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(config.lidar_range_m))
        lidar_bp.set_attribute(
            "rotation_frequency", str(
                config.lidar_rotation_frequency_hz))
        lidar_bp.set_attribute(
            "points_per_second", str(
                config.lidar_points_per_second))
        lidar_bp.set_attribute("channels", str(config.lidar_channels))
        lidar_bp.set_attribute("upper_fov", str(config.lidar_upper_fov_deg))
        lidar_bp.set_attribute("lower_fov", str(config.lidar_lower_fov_deg))
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

        collision_bp = bp_lib.find("sensor.other.collision")

        self.rgb_camera = world.spawn_actor(
            rgb_bp, self.camera_transform, attach_to=ego)
        self.instance_camera = world.spawn_actor(
            instance_bp, self.camera_transform, attach_to=ego)
        self.lidar_sensor = world.spawn_actor(
            lidar_bp, self.lidar_transform, attach_to=ego)
        self.collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego)

        self.rgb_camera.listen(self.rgb_queue.put)
        self.instance_camera.listen(self.instance_queue.put)
        self.lidar_sensor.listen(self.lidar_queue.put)
        self.collision_sensor.listen(self.collision_events.append)

        self.intrinsic = get_camera_intrinsic(
            config.image_width, config.image_height, config.fov)

    def destroy(self) -> None:
        sensors = [
            self.rgb_camera,
            self.instance_camera,
            self.lidar_sensor,
            self.collision_sensor,
        ]
        for sensor in sensors:
            if sensor is None:
                continue
            try:
                sensor.stop()
            except RuntimeError:
                pass
        destroy_actors(sensors)

    def clear_queues(self) -> None:
        for q in (self.rgb_queue, self.instance_queue, self.lidar_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def warmup(self, ticks: int) -> None:
        for _ in range(int(ticks)):
            self.world.tick()
            self.clear_queues()
            self.collision_events[:] = []

    def get_packet(self, timeout: float = 2.0) -> SensorPacket:
        packets = {
            "rgb": self.rgb_queue.get(timeout=timeout),
            "instance": self.instance_queue.get(timeout=timeout),
            "lidar": self.lidar_queue.get(timeout=timeout),
        }

        while True:
            frame_ids = {name: int(packet.frame)
                         for name, packet in packets.items()}
            if len(set(frame_ids.values())) == 1:
                break

            target_frame = max(frame_ids.values())
            for name, packet in list(packets.items()):
                if int(packet.frame) < target_frame:
                    packets[name] = getattr(
                        self,
                        name +
                        "_queue").get(
                        timeout=timeout)

        return SensorPacket(
            tick=int(packets["rgb"].frame),
            rgb_image=_parse_rgb(packets["rgb"]),
            instance_image=_parse_instance(packets["instance"]),
            lidar_points=_parse_lidar(packets["lidar"]),
        )

    def consume_collision_events(self) -> int:
        count = len(self.collision_events)
        self.collision_events[:] = []
        return count

    def camera_world_matrix(self) -> np.ndarray:
        return np.asarray(
            self.rgb_camera.get_transform().get_matrix(),
            dtype=np.float64)

    def lidar_world_matrix(self) -> np.ndarray:
        return np.asarray(
            self.lidar_sensor.get_transform().get_matrix(),
            dtype=np.float64)

    def lidar_to_camera_matrix(self) -> np.ndarray:
        camera_inv = np.linalg.inv(self.camera_world_matrix())
        return camera_inv @ self.lidar_world_matrix()

    def project_actor_bbox(
            self, actor: carla.Vehicle) -> Optional[Tuple[int, int, int, int]]:
        bbox = actor.bounding_box
        vertices = bbox.get_world_vertices(actor.get_transform())
        world_points = np.array([[v.x, v.y, v.z]
                                for v in vertices], dtype=np.float64)
        uv, valid = project_world_points_to_image(
            world_points,
            self.camera_world_matrix(),
            self.intrinsic,
            self.config.image_width,
            self.config.image_height,
        )
        if valid.sum() < 4:
            return None
        uv_valid = uv[valid]
        x1 = int(
            np.clip(
                np.floor(uv_valid[:, 0].min()),
                0, self.config.image_width - 1))
        y1 = int(
            np.clip(
                np.floor(uv_valid[:, 1].min()),
                0, self.config.image_height - 1))
        x2 = int(
            np.clip(
                np.ceil(uv_valid[:, 0].max()),
                0, self.config.image_width - 1))
        y2 = int(
            np.clip(
                np.ceil(uv_valid[:, 1].max()),
                0, self.config.image_height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def target_instance_mask(
        self,
        instance_image: np.ndarray,
        target_bbox_xyxy: Optional[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        """Approximate the target GT mask from instance segmentation."""
        if target_bbox_xyxy is None:
            return None

        semantic_tags = instance_image[:, :, 2]
        instance_ids = (
            instance_image[:, :, 0].astype(np.uint32)
            + instance_image[:, :, 1].astype(np.uint32) * 256
        )
        vehicle_mask = semantic_tags == self.vehicle_semantic_tag
        if not np.any(vehicle_mask):
            return None

        best_mask = None
        best_iou = 0.0
        unique_ids = np.unique(instance_ids[vehicle_mask])
        for instance_id in unique_ids:
            mask = (instance_ids == instance_id) & vehicle_mask
            bbox = bbox_from_mask(mask)
            if bbox is None:
                continue
            score = bbox_iou(target_bbox_xyxy, bbox)
            if score > best_iou:
                best_iou = score
                best_mask = mask

        return best_mask


class TargetSpectatorCamera:
    """Overhead RGB camera attached to the target for qualitative evaluation."""

    def __init__(
            self,
            world: carla.World,
            target: carla.Vehicle,
            config: PursuitEvalConfig) -> None:
        self.world = world
        self.target = target
        self.config = config
        self.queue: "queue.Queue[carla.Image]" = queue.Queue()

        bp_lib = world.get_blueprint_library()
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(config.spectator_width))
        rgb_bp.set_attribute("image_size_y", str(config.spectator_height))
        rgb_bp.set_attribute("fov", str(config.spectator_fov))
        transform = carla.Transform(
            carla.Location(
                x=float(config.spectator_x_m),
                y=float(config.spectator_y_m),
                z=float(config.spectator_z_m),
            ),
            carla.Rotation(
                pitch=float(config.spectator_pitch_deg),
                yaw=float(config.spectator_yaw_deg),
                roll=float(config.spectator_roll_deg),
            ),
        )
        self.camera = world.spawn_actor(rgb_bp, transform, attach_to=target)
        self.camera.listen(self.queue.put)

    def destroy(self) -> None:
        if self.camera is None:
            return
        try:
            self.camera.stop()
        except RuntimeError:
            pass
        destroy_actors([self.camera])

    def clear_queue(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def get_image(self, target_frame: int, timeout: float = 2.0) -> np.ndarray:
        packet = self.queue.get(timeout=timeout)
        while int(packet.frame) < int(target_frame):
            packet = self.queue.get(timeout=timeout)
        return _parse_rgb(packet)


class PursuitScenario:
    """CARLA pursuit scenario containing ego, target, traffic, and sensors."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.ego: Optional[carla.Vehicle] = None
        self.target: Optional[carla.Vehicle] = None
        self.background_vehicles: List[carla.Vehicle] = []
        self.sensors: Optional[EgoSensorSuite] = None
        self.spectator_camera: Optional[TargetSpectatorCamera] = None
        self._spawned_actors: List[object] = []

    def setup(self) -> None:
        random.seed(int(self.config.random_seed))
        np.random.seed(int(self.config.random_seed))
        self.client = carla.Client(
            self.config.carla_host,
            self.config.carla_port)
        self.client.set_timeout(float(self.config.client_timeout_s))
        self.world = self.client.load_world(self.config.town)
        for _ in range(15):
            self.world.tick()

        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.config.sync_mode)
        settings.fixed_delta_seconds = float(self.config.fixed_delta_seconds)
        self.world.apply_settings(settings)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.traffic_manager = self.client.get_trafficmanager(
            int(self.config.tm_port))
        try:
            self.traffic_manager.set_random_device_seed(
                int(self.config.random_seed))
        except RuntimeError:
            pass
        self.traffic_manager.set_synchronous_mode(bool(self.config.sync_mode))
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            float(self.config.traffic_follow_distance_m)
        )
        self.traffic_manager.global_percentage_speed_difference(
            float(self.config.background_speed_difference_pct)
        )
        self.traffic_manager.set_respawn_dormant_vehicles(True)

        if self.config.clear_existing_vehicles:
            destroy_actors(self.world.get_actors().filter("vehicle.*"))
            for _ in range(5):
                self.world.tick()

        self._spawn_ego_target_and_traffic()
        self.sensors = EgoSensorSuite(self.world, self.ego, self.config)
        if bool(self.config.enable_spectator_camera):
            self.spectator_camera = TargetSpectatorCamera(
                self.world, self.target, self.config)
        self._spawned_actors.extend([self.ego, self.target] +
                                    list(self.background_vehicles) +
                                    [self.sensors, self.spectator_camera])
        self._warmup_sensors(self.config.warmup_ticks)
        self._start_target_autopilot()
        self.background_vehicles = self._spawn_background_traffic()
        if float(self.config.ego_initial_speed_mps) > 0.0:
            self._set_ego_forward_speed(
                float(self.config.ego_initial_speed_mps))
        self.world.tick()

    def cleanup(self) -> None:
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)

        if self.sensors is not None:
            self.sensors.destroy()
            self.sensors = None
        if self.spectator_camera is not None:
            self.spectator_camera.destroy()
            self.spectator_camera = None
        actors = [self.ego, self.target] + list(self.background_vehicles)
        actor_ids = []
        for actor in actors:
            if actor is None:
                continue
            try:
                actor_ids.append(actor.id)
            except RuntimeError:
                continue
        if self.client is not None and actor_ids:
            try:
                self.client.apply_batch(
                    [carla.command.DestroyActor(actor_id)
                     for actor_id in actor_ids])
            except RuntimeError:
                pass
        else:
            destroy_actors(actors)
        self.background_vehicles = []
        self.ego = None
        self.target = None

    def _spawn_ego_target_and_traffic(self) -> None:
        assert self.world is not None
        assert self.traffic_manager is not None
        bp_lib = self.world.get_blueprint_library()
        blueprints = _preferred_vehicle_blueprints(bp_lib)
        carla_map = self.world.get_map()
        spawn_points = list(carla_map.get_spawn_points())
        random.shuffle(spawn_points)

        for spawn_point in spawn_points[: int(self.config.spawn_attempts)]:
            if bool(self.config.require_follow_friendly_spawn):
                if not follow_friendly_waypoint_chain(
                    carla_map,
                    spawn_point.location,
                    lookahead_m=float(
                        self.config.follow_spawn_lookahead_m,
                    ),
                    step_m=float(self.config.follow_spawn_step_m),
                    max_yaw_delta_deg=float(
                        self.config.follow_spawn_max_yaw_delta_deg,
                    ),
                ):
                    continue

            ego_bp = _random_blueprint(blueprints, "pursuit_ego")
            ego = self.world.try_spawn_actor(ego_bp, spawn_point)
            if ego is None:
                continue
            ego.set_autopilot(False)
            ego.set_simulate_physics(True)
            ego.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False,
                )
            )

            ego_waypoint = carla_map.get_waypoint(
                spawn_point.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if ego_waypoint is None:
                destroy_actors([ego])
                continue

            ahead_candidates = ego_waypoint.next(
                float(self.config.initial_target_distance_m))
            if not ahead_candidates:
                destroy_actors([ego])
                continue

            target_waypoint = ahead_candidates[0]
            target_transform = carla.Transform(
                carla.Location(
                    x=target_waypoint.transform.location.x,
                    y=target_waypoint.transform.location.y,
                    z=target_waypoint.transform.location.z +
                    float(
                        self.config.target_spawn_z_offset_m),
                ),
                target_waypoint.transform.rotation,
            )
            target_bp = _random_blueprint(blueprints, "pursuit_target")
            target = self.world.try_spawn_actor(target_bp, target_transform)
            if target is None:
                destroy_actors([ego])
                continue

            self.ego = ego
            self.target = target
            return

        raise RuntimeError(
            "Failed to spawn a valid ego-target pair for pursuit evaluation.")

    def _set_ego_forward_speed(self, speed_mps: float) -> None:
        assert self.ego is not None
        forward = self.ego.get_transform().get_forward_vector()
        self.ego.set_target_velocity(
            carla.Vector3D(
                x=float(forward.x) * float(speed_mps),
                y=float(forward.y) * float(speed_mps),
                z=0.0,
            )
        )
        self.ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )

    def _start_target_autopilot(self) -> None:
        assert self.target is not None
        assert self.traffic_manager is not None
        self.target.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.auto_lane_change(self.target, False)
        self.traffic_manager.vehicle_percentage_speed_difference(
            self.target,
            float(self.config.target_speed_difference_pct),
        )
        self.traffic_manager.distance_to_leading_vehicle(
            self.target,
            float(self.config.traffic_follow_distance_m),
        )

    def _spawn_background_traffic(self) -> List[carla.Vehicle]:
        assert self.world is not None
        assert self.ego is not None
        assert self.target is not None
        assert self.traffic_manager is not None
        bp_lib = self.world.get_blueprint_library()
        blueprints = _preferred_vehicle_blueprints(bp_lib)
        vehicles: List[carla.Vehicle] = []

        spawn_points = list(self.world.get_map().get_spawn_points())
        random.shuffle(spawn_points)
        ego_loc = self.ego.get_location()
        target_loc = self.target.get_location()

        for spawn_point in spawn_points:
            if len(vehicles) >= int(self.config.num_background_vehicles):
                break

            dist_ego = spawn_point.location.distance(ego_loc)
            dist_target = spawn_point.location.distance(target_loc)
            exclusion_radius = float(
                self.config.background_spawn_exclusion_radius_m,
            )
            if (
                dist_ego < exclusion_radius
                or dist_target < exclusion_radius
            ):
                continue

            blueprint = _random_blueprint(blueprints, "autopilot")
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.auto_lane_change(vehicle, True)
            self.traffic_manager.distance_to_leading_vehicle(
                vehicle,
                float(self.config.traffic_follow_distance_m),
            )
            vehicles.append(vehicle)

        return vehicles

    def _warmup_sensors(self, ticks: int) -> None:
        assert self.world is not None
        assert self.sensors is not None
        for _ in range(int(ticks)):
            self.world.tick()
            self.sensors.clear_queues()
            self.sensors.collision_events[:] = []
            if self.spectator_camera is not None:
                self.spectator_camera.clear_queue()

    def tick(self) -> SensorPacket:
        assert self.world is not None
        assert self.sensors is not None
        self.world.tick()
        packet = self.sensors.get_packet()
        if self.spectator_camera is not None:
            packet.spectator_image = self.spectator_camera.get_image(
                packet.tick)
        return packet

    @staticmethod
    def _actor_alive(actor: Optional[carla.Actor]) -> bool:
        if actor is None:
            return False
        try:
            return bool(actor.is_alive)
        except RuntimeError:
            return False

    def ego_alive(self) -> bool:
        return self._actor_alive(self.ego)

    def target_alive(self) -> bool:
        return self._actor_alive(self.target)

    def ego_vehicle_state(self) -> Dict[str, float]:
        assert self.ego is not None
        velocity = self.ego.get_velocity()
        control = self.ego.get_control()
        return {
            "speed_mps": math.sqrt(
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2),
            "throttle": float(
                control.throttle),
            "steer": float(
                control.steer),
            "brake": float(
                control.brake),
        }

    def apply_control(
            self,
            throttle: float,
            steer: float,
            brake: float) -> None:
        assert self.ego is not None
        self.ego.apply_control(
            carla.VehicleControl(
                throttle=float(np.clip(throttle, 0.0, 1.0)),
                steer=float(np.clip(steer, -1.0, 1.0)),
                brake=float(np.clip(brake, 0.0, 1.0)),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )

    def ground_truth_pose(self) -> Dict[str, float]:
        assert self.ego is not None
        assert self.target is not None
        return compute_relative_pose(self.ego, self.target)

    def target_speed(self) -> float:
        assert self.target is not None
        return get_vehicle_speed(self.target)

    def ego_offroad(self) -> bool:
        assert self.world is not None
        assert self.ego is not None
        return not ego_on_driving_lane(self.world, self.ego)
