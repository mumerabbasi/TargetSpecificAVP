"""CARLA utilities for inference and vehicle pursuit."""

import math
import queue
import random
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

from .config import InferenceConfig


def connect_to_carla(config: InferenceConfig) -> carla.Client:
    """Connect to CARLA server.

    Args:
        config: Inference configuration.

    Returns:
        Connected CARLA client.
    """
    print(f"[CARLA] Connecting to {config.carla_host}:{config.carla_port}")
    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(30.0)
    print(f"[CARLA] Connected to CARLA {client.get_server_version()}")
    return client


def setup_world(
    client: carla.Client,
    config: InferenceConfig,
) -> carla.World:
    """Setup CARLA world with synchronous mode.

    Args:
        client: CARLA client.
        config: Inference configuration.

    Returns:
        Configured CARLA world.
    """
    # Load town
    print(f"[CARLA] Loading {config.town}")
    world = client.load_world(config.town)

    # Wait for town to load
    for _ in range(20):
        world.tick()

    # Configure settings
    settings = world.get_settings()
    settings.synchronous_mode = config.sync_mode
    settings.fixed_delta_seconds = config.fixed_delta_seconds
    world.apply_settings(settings)

    # Set weather to clear
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)

    print(f"[CARLA] World configured (sync={config.sync_mode})")
    return world


def spawn_ego_vehicle(
    world: carla.World,
    spawn_point: Optional[carla.Transform] = None,
) -> carla.Vehicle:
    """Spawn the ego vehicle.

    Args:
        world: CARLA world.
        spawn_point: Optional spawn location.

    Returns:
        Spawned ego vehicle.
    """
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    ego_bp.set_attribute("role_name", "ego")

    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

    ego = world.try_spawn_actor(ego_bp, spawn_point)
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle")

    print(f"[CARLA] Spawned ego vehicle at "
          f"({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

    return ego


def spawn_target_vehicles(
    world: carla.World,
    ego: carla.Vehicle,
    config: InferenceConfig,
) -> List[carla.Vehicle]:
    """Spawn target vehicles in specific positions relative to ego.

    Spawns exactly 3 vehicles:
    - One directly ahead in same lane at initial_target_distance
    - One in the left lane (if available)
    - One in the right lane (if available)

    All vehicles are snapped to waypoints for proper road positioning.

    Args:
        world: CARLA world.
        ego: Ego vehicle.
        config: Inference configuration.

    Returns:
        List of spawned target vehicles.
    """
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    ego_tf = ego.get_transform()

    # Get diverse vehicle blueprints - prefer smaller vehicles
    vehicle_bps = bp_lib.filter("vehicle.*")
    preferred = [
        bp for bp in vehicle_bps
        if any(v in bp.id.lower() for v in
               ["mini", "a2", "tt", "prius", "cooper", "c3", "mustang"])
    ]
    if not preferred:
        preferred = list(vehicle_bps)

    targets = []

    # Get waypoint at ego's current position
    ego_waypoint = carla_map.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if ego_waypoint is None:
        print("[CARLA] Warning: Could not find waypoint for ego")
        return targets

    # 1. Spawn vehicle AHEAD in same lane
    ahead_waypoints = ego_waypoint.next(config.initial_target_distance)
    if ahead_waypoints:
        ahead_wp = ahead_waypoints[0]
        spawn_loc = ahead_wp.transform.location
        spawn_loc.z += 0.3

        spawn_tf = carla.Transform(
            location=spawn_loc,
            rotation=ahead_wp.transform.rotation,
        )

        target_bp = random.choice(preferred)
        target_bp.set_attribute("role_name", "target_ahead")

        target = world.try_spawn_actor(target_bp, spawn_tf)
        if target is not None:
            targets.append(target)
            print(f"[CARLA] Spawned target_ahead at distance "
                  f"~{config.initial_target_distance:.1f}m")

    # 2. Spawn vehicle in LEFT lane
    left_lane = ego_waypoint.get_left_lane()
    if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
        left_ahead = left_lane.next(config.initial_target_distance)
        if left_ahead:
            left_wp = left_ahead[0]
            spawn_loc = left_wp.transform.location
            spawn_loc.z += 0.3

            spawn_tf = carla.Transform(
                location=spawn_loc,
                rotation=left_wp.transform.rotation,
            )

            target_bp = random.choice(preferred)
            target_bp.set_attribute("role_name", "target_left")

            target = world.try_spawn_actor(target_bp, spawn_tf)
            if target is not None:
                targets.append(target)
                print("[CARLA] Spawned target_left in left lane")

    # 3. Spawn vehicle in RIGHT lane
    right_lane = ego_waypoint.get_right_lane()
    if right_lane is not None and right_lane.lane_type == carla.LaneType.Driving:
        right_ahead = right_lane.next(config.initial_target_distance)
        if right_ahead:
            right_wp = right_ahead[0]
            spawn_loc = right_wp.transform.location
            spawn_loc.z += 0.3

            spawn_tf = carla.Transform(
                location=spawn_loc,
                rotation=right_wp.transform.rotation,
            )

            target_bp = random.choice(preferred)
            target_bp.set_attribute("role_name", "target_right")

            target = world.try_spawn_actor(target_bp, spawn_tf)
            if target is not None:
                targets.append(target)
                print("[CARLA] Spawned target_right in right lane")

    print(f"[CARLA] Total targets spawned: {len(targets)}")
    return targets


def ego_to_world_offset(
    ego_transform: carla.Transform,
    dx_ego: float,
    dy_ego: float,
    dz_ego: float = 0.0,
) -> carla.Location:
    """Convert ego-frame offset to world location.

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


class SensorManager:
    """Manager for CARLA sensors attached to ego vehicle.

    Handles RGB camera and instance segmentation camera.
    Finds and tracks vehicles directly from the instance segmentation image.

    CARLA Instance Segmentation Format (raw_data is BGRA):
    - When reshaped to (H, W, 4), the channels are [B, G, R, A]
    - R channel (index 2): Semantic tag (10 = Vehicle)
    - G channel (index 1): Instance ID high byte
    - B channel (index 0): Instance ID low byte
    - Instance ID = B + G * 256

    Attributes:
        world: CARLA world.
        ego: Ego vehicle.
        config: Inference configuration.
        rgb_camera: RGB camera sensor.
        instance_camera: Instance segmentation camera.
        rgb_queue: Queue for RGB images.
        instance_queue: Queue for instance segmentation images.
        tracked_instance_id: Currently tracked vehicle instance ID.
    """

    # Semantic tag for vehicles in CARLA 0.9.15
    VEHICLE_SEMANTIC_TAG = 14

    def __init__(
        self,
        world: carla.World,
        ego: carla.Vehicle,
        config: InferenceConfig,
    ):
        """Initialize sensor manager.

        Args:
            world: CARLA world.
            ego: Ego vehicle.
            config: Inference configuration.
        """
        self.world = world
        self.ego = ego
        self.config = config

        self.rgb_camera: Optional[carla.Sensor] = None
        self.instance_camera: Optional[carla.Sensor] = None
        self.rgb_queue: queue.Queue = queue.Queue()
        self.instance_queue: queue.Queue = queue.Queue()

        # Track a specific vehicle instance ID across frames
        self.tracked_instance_id: Optional[int] = None

        self._setup_sensors()

    def _setup_sensors(self) -> None:
        """Setup RGB and instance segmentation cameras."""
        bp_lib = self.world.get_blueprint_library()

        # Camera transform (front-mounted)
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.6),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        # RGB Camera
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.config.image_width))
        rgb_bp.set_attribute("image_size_y", str(self.config.image_height))
        rgb_bp.set_attribute("fov", str(self.config.fov))

        self.rgb_camera = self.world.spawn_actor(
            rgb_bp, camera_transform, attach_to=self.ego
        )
        self.rgb_camera.listen(self.rgb_queue.put)

        # Instance Segmentation Camera
        instance_bp = bp_lib.find("sensor.camera.instance_segmentation")
        instance_bp.set_attribute("image_size_x", str(self.config.image_width))
        instance_bp.set_attribute("image_size_y", str(self.config.image_height))
        instance_bp.set_attribute("fov", str(self.config.fov))

        self.instance_camera = self.world.spawn_actor(
            instance_bp, camera_transform, attach_to=self.ego
        )
        self.instance_camera.listen(self.instance_queue.put)

        print("[CARLA] Sensors attached (RGB + Instance Segmentation)")

    def get_sensor_data(
        self,
        timeout: float = 2.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get synchronized RGB and instance segmentation images.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Tuple of (rgb_image, instance_image) as numpy arrays.
            Returns (None, None) if timeout.
        """
        try:
            rgb_raw = self.rgb_queue.get(timeout=timeout)
            instance_raw = self.instance_queue.get(timeout=timeout)

            rgb_image = self._parse_rgb_image(rgb_raw)
            instance_image = self._parse_instance_image(instance_raw)

            return rgb_image, instance_image

        except queue.Empty:
            return None, None

    def _parse_rgb_image(self, image: carla.Image) -> np.ndarray:
        """Parse CARLA RGB image to numpy array.

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

    def _parse_instance_image(self, image: carla.Image) -> np.ndarray:
        """Parse CARLA instance segmentation image.

        CARLA instance segmentation encoding (BGRA format):
        - B (index 0): Unique instance ID (low byte)
        - G (index 1): Unique instance ID (high byte)
        - R (index 2): Semantic tag (e.g., 10 = Vehicle)
        - A (index 3): Always 255

        Args:
            image: CARLA instance segmentation image.

        Returns:
            HxWx4 numpy array with raw BGRA data.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array.copy()

    def _get_instance_ids(self, instance_image: np.ndarray) -> np.ndarray:
        """Extract instance IDs from parsed instance image.

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.

        Returns:
            HxW array of instance IDs (B + G * 256).
        """
        instance_ids = (instance_image[:, :, 0].astype(np.uint32) +
                        instance_image[:, :, 1].astype(np.uint32) * 256)
        return instance_ids

    def _get_semantic_tags(self, instance_image: np.ndarray) -> np.ndarray:
        """Extract semantic tags from parsed instance image.

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.

        Returns:
            HxW array of semantic tags (R channel).
        """
        return instance_image[:, :, 2]

    def find_vehicle_instances(
        self,
        instance_image: np.ndarray,
        min_pixels: int = 100,
        min_height: int = 20,
        min_width: int = 20,
        debug: bool = False,
    ) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
        """Find all vehicle instances in the image.

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.
            min_pixels: Minimum pixels for a valid vehicle detection.
            min_height: Minimum bounding box height in pixels.
            min_width: Minimum bounding box width in pixels.
            debug: Print debug information.

        Returns:
            List of (instance_id, pixel_count, bbox) tuples.
            bbox is (x_min, y_min, x_max, y_max).
        """
        semantic_tags = self._get_semantic_tags(instance_image)
        instance_ids = self._get_instance_ids(instance_image)

        # Find all vehicle pixels (semantic tag 10)
        vehicle_mask = (semantic_tags == self.VEHICLE_SEMANTIC_TAG)

        if debug:
            unique_tags, tag_counts = np.unique(semantic_tags, return_counts=True)
            print(f"[Debug] Semantic tags in image: {dict(zip(unique_tags, tag_counts))}")
            print(f"[Debug] Vehicle pixels (tag=14): {vehicle_mask.sum()}")

        if not np.any(vehicle_mask):
            return []

        # Get unique vehicle instance IDs
        vehicle_instance_ids = instance_ids[vehicle_mask]
        unique_ids, counts = np.unique(vehicle_instance_ids, return_counts=True)

        vehicles = []
        for inst_id, count in zip(unique_ids, counts):
            if count >= min_pixels:
                # Get bounding box for this instance
                inst_mask = (instance_ids == inst_id) & vehicle_mask
                ys, xs = np.where(inst_mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                
                # Filter by bounding box size - vehicles should be reasonably sized
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                
                if bbox_width >= min_width and bbox_height >= min_height:
                    vehicles.append((int(inst_id), int(count), bbox))
                elif debug:
                    print(f"[Debug] Filtered out instance {inst_id}: bbox too small ({bbox_width}x{bbox_height})")

        if debug and vehicles:
            print(f"[Debug] Found {len(vehicles)} valid vehicles:")
            for inst_id, count, bbox in vehicles:
                print(f"  - Instance {inst_id}: {count} pixels, bbox={bbox}")

        return vehicles

    def select_target_vehicle(
        self,
        instance_image: np.ndarray,
        min_pixels: int = 100,
        debug: bool = False,
    ) -> Optional[int]:
        """Select the best vehicle to track.

        This method is only used as a fallback. The primary tracking approach
        is to use initialize_tracking() in the first frame.

        Prioritizes:
        1. Currently tracked vehicle if still visible
        2. Vehicle closest to center of frame (most likely the chase target)

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.
            min_pixels: Minimum pixels for valid detection.
            debug: Print debug information.

        Returns:
            Instance ID of selected vehicle, or None if no vehicle found.
        """
        vehicles = self.find_vehicle_instances(instance_image, min_pixels, debug=debug)

        if not vehicles:
            return None

        # Check if currently tracked vehicle is still visible
        if self.tracked_instance_id is not None:
            for inst_id, count, bbox in vehicles:
                if inst_id == self.tracked_instance_id:
                    return inst_id
            # Tracked vehicle not visible - don't switch, return None
            print(f"[Tracking] Lost track of instance {self.tracked_instance_id}")
            return None

        # No tracking initialized - select center vehicle as fallback
        height, width = instance_image.shape[:2]
        center_x = width // 2

        best_vehicle = None
        best_dist = float("inf")

        for inst_id, count, bbox in vehicles:
            bbox_center_x = (bbox[0] + bbox[2]) // 2
            dist_to_center = abs(bbox_center_x - center_x)

            if dist_to_center < best_dist:
                best_dist = dist_to_center
                best_vehicle = (inst_id, count, bbox)

        if best_vehicle is not None:
            self.tracked_instance_id = best_vehicle[0]
            print(
                f"[Tracking] Fallback: selected center vehicle {self.tracked_instance_id} "
                f"({best_vehicle[1]} pixels)"
            )
            return self.tracked_instance_id

        return None

    def set_tracked_instance(self, instance_id: int) -> None:
        """Set the instance ID to track.

        Args:
            instance_id: Instance ID to track.
        """
        self.tracked_instance_id = instance_id
        print(f"[Tracking] Now tracking instance {instance_id}")

    def get_tracked_mask(
        self,
        instance_image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Get bbox mask for the currently tracked instance.

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.

        Returns:
            Tuple of (bbox_mask, instance_id) or (None, None) if not visible.
        """
        if self.tracked_instance_id is None:
            return None, None

        height, width = instance_image.shape[:2]
        semantic_tags = self._get_semantic_tags(instance_image)
        instance_ids = self._get_instance_ids(instance_image)

        # Check if tracked instance is visible
        target_mask = ((instance_ids == self.tracked_instance_id) &
                       (semantic_tags == self.VEHICLE_SEMANTIC_TAG))

        if not np.any(target_mask):
            return None, None

        # Get bounding box
        ys, xs = np.where(target_mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Create filled bounding box mask
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

        return bbox_mask, self.tracked_instance_id

    def get_target_bbox_mask(
        self,
        instance_image: np.ndarray,
        min_pixels: int = 100,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Get binary bounding box mask for the target vehicle.

        This method:
        1. Finds/tracks a vehicle in the instance segmentation image
        2. Gets the bounding box of that vehicle
        3. Returns a filled bounding box mask (like training data)

        Args:
            instance_image: HxWx4 BGRA instance segmentation image.
            min_pixels: Minimum pixels for valid detection.

        Returns:
            Tuple of (bbox_mask, instance_id) or (None, None) if no target.
            bbox_mask is HxW binary mask with the bounding box filled.
        """
        height, width = instance_image.shape[:2]

        # Debug on first call (when not tracking yet)
        debug = (self.tracked_instance_id is None)

        # Select target vehicle
        target_id = self.select_target_vehicle(instance_image, min_pixels, debug=debug)

        if target_id is None:
            if debug:
                print("[Warning] No vehicles found in instance segmentation")
            return None, None

        # Get semantic tags and instance IDs
        semantic_tags = self._get_semantic_tags(instance_image)
        instance_ids = self._get_instance_ids(instance_image)

        # Get mask for target vehicle
        target_mask = ((instance_ids == target_id) &
                       (semantic_tags == self.VEHICLE_SEMANTIC_TAG))

        if not np.any(target_mask):
            self.tracked_instance_id = None
            return None, None

        # Get bounding box
        ys, xs = np.where(target_mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Create filled bounding box mask (like training bbox_mode="mask")
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        bbox_mask[y_min:y_max+1, x_min:x_max+1] = 1

        return bbox_mask, target_id

    def clear_queues(self) -> None:
        """Clear sensor data queues."""
        while not self.rgb_queue.empty():
            try:
                self.rgb_queue.get_nowait()
            except queue.Empty:
                break

        while not self.instance_queue.empty():
            try:
                self.instance_queue.get_nowait()
            except queue.Empty:
                break

    def destroy(self) -> None:
        """Destroy all sensors."""
        if self.rgb_camera is not None:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()

        if self.instance_camera is not None:
            self.instance_camera.stop()
            self.instance_camera.destroy()

        print("[CARLA] Sensors destroyed")


class SpectatorManager:
    """Manager for spectator camera that follows the pursuit.

    Positions camera behind target vehicle, facing toward ego vehicle,
    so both vehicles are visible in the frame.
    """

    def __init__(
        self,
        world: carla.World,
        height: float = 30.0,
        distance: float = 20.0,
    ):
        """Initialize spectator manager.

        Args:
            world: CARLA world.
            height: Height above vehicles.
            distance: Distance behind target (toward ego).
        """
        self.world = world
        self.spectator = world.get_spectator()
        self.height = height
        self.distance = distance

        # Spectator camera sensor for saving images
        self.camera: Optional[carla.Sensor] = None
        self.image_queue: queue.Queue = queue.Queue()

        # Target and ego vehicles for positioning
        self.target_vehicle: Optional[carla.Vehicle] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None

    def setup_camera(
        self,
        target: carla.Vehicle,
        ego: Optional[carla.Vehicle] = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """Setup spectator camera.

        Camera is positioned behind target, facing toward ego.

        Args:
            target: Target vehicle to follow.
            ego: Ego vehicle (for orientation).
            width: Image width.
            height: Image height.
        """
        self.target_vehicle = target
        self.ego_vehicle = ego

        bp_lib = self.world.get_blueprint_library()

        # Camera in front of target (between target and ego), facing back
        # This means positive x offset and yaw=180 to look backward
        camera_transform = carla.Transform(
            carla.Location(x=self.distance, y=0.0, z=self.height),
            carla.Rotation(pitch=-30.0, yaw=180.0, roll=0.0),
        )

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", "90")

        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=target
        )
        self.camera.listen(self.image_queue.put)

        print(f"[CARLA] Spectator camera attached to target {target.id}, facing ego")

    def update(
        self,
        target: Optional[carla.Vehicle] = None,
        ego: Optional[carla.Vehicle] = None,
    ) -> None:
        """Update spectator view.

        Positions spectator behind target, looking toward ego.

        Args:
            target: Target vehicle. If None, uses self.target_vehicle.
            ego: Ego vehicle (unused, kept for API compatibility).
        """
        target_v = target if target is not None else self.target_vehicle

        if target_v is None:
            return

        target_tf = target_v.get_transform()
        target_loc = target_tf.location
        target_yaw = math.radians(target_tf.rotation.yaw)

        # Position in front of target (between target and ego)
        # Since ego is behind target, we go in the opposite direction of target's heading
        dx = self.distance * math.cos(target_yaw)
        dy = self.distance * math.sin(target_yaw)

        spectator_loc = carla.Location(
            x=target_loc.x + dx,
            y=target_loc.y + dy,
            z=target_loc.z + self.height,
        )

        # Look back toward target (and ego behind it)
        # Rotate 180 degrees from target heading
        look_yaw = target_tf.rotation.yaw + 180.0

        spectator_rot = carla.Rotation(
            pitch=-30.0,
            yaw=look_yaw,
            roll=0.0,
        )

        self.spectator.set_transform(
            carla.Transform(spectator_loc, spectator_rot)
        )

    def get_image(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get spectator camera image.

        Args:
            timeout: Timeout in seconds.

        Returns:
            RGB image or None if timeout.
        """
        if self.camera is None:
            return None

        try:
            image_raw = self.image_queue.get(timeout=timeout)
            array = np.frombuffer(image_raw.raw_data, dtype=np.uint8)
            array = array.reshape((image_raw.height, image_raw.width, 4))
            rgb = array[:, :, :3][:, :, ::-1].copy()
            return rgb
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Clear image queue."""
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break

    def destroy(self) -> None:
        """Destroy spectator camera."""
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            print("[CARLA] Spectator camera destroyed")


def select_target(
    targets: List[carla.Vehicle],
    ego: carla.Vehicle,
) -> Optional[Tuple[carla.Vehicle, int]]:
    """Select the closest target vehicle ahead of ego.

    Args:
        targets: List of target vehicles.
        ego: Ego vehicle.

    Returns:
        Tuple of (target_vehicle, actor_id) or None if no valid target.
    """
    if not targets:
        return None

    ego_tf = ego.get_transform()
    ego_fwd = ego_tf.get_forward_vector()

    best_target = None
    best_dist = float("inf")

    for target in targets:
        target_loc = target.get_location()
        ego_loc = ego_tf.location

        # Vector from ego to target
        dx = target_loc.x - ego_loc.x
        dy = target_loc.y - ego_loc.y

        # Distance
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Check if target is ahead (dot product with forward vector > 0)
        dot = dx * ego_fwd.x + dy * ego_fwd.y

        if dot > 0 and dist < best_dist:
            best_dist = dist
            best_target = target

    if best_target is not None:
        return (best_target, best_target.id)

    return None


def get_vehicle_state(ego: carla.Vehicle) -> Dict[str, float]:
    """Get current state of the ego vehicle.

    Args:
        ego: Ego vehicle.

    Returns:
        Dictionary with speed, throttle, steer, brake values.
    """
    velocity = ego.get_velocity()
    speed = math.sqrt(
        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
    )

    control = ego.get_control()

    return {
        "speed": speed,
        "throttle": control.throttle,
        "steer": control.steer,
        "brake": control.brake,
    }


def get_vehicle_speed(vehicle: carla.Vehicle) -> float:
    """Get vehicle speed in m/s.

    Args:
        vehicle: CARLA vehicle actor.

    Returns:
        Speed in m/s.
    """
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def apply_control(
    ego: carla.Vehicle,
    throttle: float,
    steer: float,
    brake: float,
) -> None:
    """Apply control command to ego vehicle.

    Args:
        ego: Ego vehicle.
        throttle: Throttle value [0, 1].
        steer: Steering value [-1, 1].
        brake: Brake value [0, 1].
    """
    control = carla.VehicleControl(
        throttle=float(np.clip(throttle, 0.0, 1.0)),
        steer=float(np.clip(steer, -1.0, 1.0)),
        brake=float(np.clip(brake, 0.0, 1.0)),
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )
    ego.apply_control(control)


def set_autopilot(vehicle: carla.Vehicle, enabled: bool = True) -> None:
    """Enable/disable autopilot for a vehicle.

    Args:
        vehicle: Vehicle actor.
        enabled: Whether to enable autopilot.
    """
    vehicle.set_autopilot(enabled)


def cleanup_actors(actors: List[carla.Actor]) -> None:
    """Safely destroy a list of actors.

    Args:
        actors: List of actors to destroy.
    """
    for actor in actors:
        try:
            if actor is not None and actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


def compute_ground_truth_pose(
    ego: carla.Vehicle,
    target: carla.Vehicle,
) -> Dict[str, float]:
    """Compute ground truth relative pose of target from ego.

    For debugging and comparison with estimated pose.

    Args:
        ego: Ego vehicle.
        target: Target vehicle.

    Returns:
        Dictionary with dx, dy, dyaw (ground truth).
    """
    ego_tf = ego.get_transform()
    target_tf = target.get_transform()

    # Vector from ego to target in world frame
    dx_world = target_tf.location.x - ego_tf.location.x
    dy_world = target_tf.location.y - ego_tf.location.y

    # Transform to ego frame
    yaw_e = math.radians(ego_tf.rotation.yaw)
    cos_e = math.cos(yaw_e)
    sin_e = math.sin(yaw_e)

    # Rotate to ego frame (ego +x forward, +y right)
    dx_ego = cos_e * dx_world + sin_e * dy_world
    dy_ego = -sin_e * dx_world + cos_e * dy_world

    # Relative yaw
    yaw_target = target_tf.rotation.yaw
    dyaw = yaw_target - ego_tf.rotation.yaw

    # Wrap to [-180, 180]
    while dyaw > 180:
        dyaw -= 360
    while dyaw < -180:
        dyaw += 360

    return {
        "dx": dx_ego,
        "dy": dy_ego,
        "dyaw": dyaw,
    }


def match_instance_to_actor(
    instance_poses: Dict[int, Dict[str, float]],
    targets: List[carla.Vehicle],
    ego: carla.Vehicle,
    max_match_dist: float = 5.0,
) -> Dict[int, carla.Vehicle]:
    """Match instance IDs to actor IDs based on pose similarity.

    Instance IDs from instance segmentation are not related to CARLA actor IDs.
    This function matches them by comparing the predicted pose (from pose estimator)
    with the ground truth pose (from actor transforms).

    Args:
        instance_poses: Dict mapping instance_id -> predicted pose dict (dx, dy, dyaw).
        targets: List of target vehicle actors.
        ego: Ego vehicle for computing ground truth poses.
        max_match_dist: Maximum distance for a valid match.

    Returns:
        Dict mapping instance_id -> matched carla.Vehicle (or None if no match).
    """
    if not instance_poses or not targets:
        return {}

    # Compute ground truth poses for all targets
    target_gt_poses = []
    for target in targets:
        if target.is_alive:
            gt = compute_ground_truth_pose(ego, target)
            target_gt_poses.append((target, gt))

    # Match each instance to closest target
    matches: Dict[int, carla.Vehicle] = {}

    for inst_id, pred_pose in instance_poses.items():
        best_target = None
        best_dist = float("inf")

        pred_dx = pred_pose.get("dx", 0.0)
        pred_dy = pred_pose.get("dy", 0.0)

        for target, gt in target_gt_poses:
            gt_dx = gt["dx"]
            gt_dy = gt["dy"]

            # Euclidean distance between predicted and GT pose
            dist = math.sqrt((pred_dx - gt_dx) ** 2 + (pred_dy - gt_dy) ** 2)

            if dist < best_dist and dist < max_match_dist:
                best_dist = dist
                best_target = target

        if best_target is not None:
            matches[inst_id] = best_target

    return matches


def get_all_vehicle_masks(
    sensor_manager: "SensorManager",
    instance_image: np.ndarray,
    min_pixels: int = 100,
) -> Dict[int, np.ndarray]:
    """Get binary bbox masks for all visible vehicles.

    Args:
        sensor_manager: SensorManager instance.
        instance_image: HxWx4 BGRA instance segmentation image.
        min_pixels: Minimum pixels for valid detection.

    Returns:
        Dict mapping instance_id -> binary bbox mask (HxW).
    """
    height, width = instance_image.shape[:2]
    vehicles = sensor_manager.find_vehicle_instances(instance_image, min_pixels)

    masks = {}
    semantic_tags = sensor_manager._get_semantic_tags(instance_image)
    instance_ids = sensor_manager._get_instance_ids(instance_image)

    for inst_id, count, bbox in vehicles:
        # Get mask for this instance
        inst_mask = ((instance_ids == inst_id) &
                     (semantic_tags == sensor_manager.VEHICLE_SEMANTIC_TAG))

        if not np.any(inst_mask):
            continue

        # Get bounding box
        ys, xs = np.where(inst_mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Create filled bounding box mask
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

        masks[inst_id] = bbox_mask

    return masks
