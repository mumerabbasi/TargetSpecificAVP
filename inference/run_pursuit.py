"""Main script for vehicle pursuit using pose estimation and MPC control."""

import argparse
import os
import time
from typing import Dict, Optional

import carla
import cv2
import numpy as np

from .carla_utils import (
    apply_control,
    cleanup_actors,
    compute_ground_truth_pose,
    connect_to_carla,
    get_all_vehicle_masks,
    get_vehicle_speed,
    get_vehicle_state,
    match_instance_to_actor,
    SensorManager,
    SpectatorManager,
    set_autopilot,
    setup_world,
    spawn_ego_vehicle,
    spawn_target_vehicles,
)
from .config import InferenceConfig
from .mpc_controller import (
    MPCController,
    TargetPose,
    VehicleState,
)
from .pose_estimator import PoseEstimator


class VehiclePursuit:
    """Main class for vehicle pursuit demonstration.

    Integrates pose estimation, MPC control, and CARLA simulation
    to pursue a target vehicle.

    Attributes:
        config: Inference configuration.
        client: CARLA client.
        world: CARLA world.
        ego: Ego vehicle.
        targets: List of target vehicles (for autopilot).
        sensor_manager: Sensor manager for cameras.
        spectator_manager: Spectator camera manager.
        pose_estimator: CNN-based pose estimator.
        controller: MPC controller.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize vehicle pursuit.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.ego: Optional[carla.Vehicle] = None
        self.targets: list = []  # For autopilot targets
        self.tracked_target: Optional[carla.Vehicle] = None  # The target we're pursuing
        self.instance_to_actor: dict = {}  # Maps instance_id -> carla.Vehicle
        self.sensor_manager: Optional[SensorManager] = None
        self.spectator_manager: Optional[SpectatorManager] = None
        self.pose_estimator: Optional[PoseEstimator] = None
        self.controller: Optional[MPCController] = None

        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
        self.pose_errors = []

        # Output directories
        self.output_dir = config.video_output_dir
        self.ego_output_dir = os.path.join(self.output_dir, "ego")
        self.spectator_output_dir = os.path.join(self.output_dir, "spectator")
        os.makedirs(self.ego_output_dir, exist_ok=True)
        os.makedirs(self.spectator_output_dir, exist_ok=True)

    def setup(self) -> None:
        """Setup CARLA world, vehicles, sensors, and models."""
        print("\n" + "=" * 60)
        print("VEHICLE PURSUIT SETUP")
        print("=" * 60)

        # Connect to CARLA
        self.client = connect_to_carla(self.config)
        self.world = setup_world(self.client, self.config)

        # Clear existing vehicles
        print("[Setup] Cleaning up existing vehicles...")
        for actor in self.world.get_actors().filter("vehicle.*"):
            try:
                actor.destroy()
            except RuntimeError:
                pass

        for _ in range(10):
            self.world.tick()

        # Spawn ego vehicle
        spawn_pts = self.world.get_map().get_spawn_points()
        self.ego = spawn_ego_vehicle(self.world, spawn_point=spawn_pts[100])

        # Spawn target vehicles
        self.targets = spawn_target_vehicles(
            self.world, self.ego, self.config
        )

        # Let vehicles settle
        for _ in range(20):
            self.world.tick()

        # Enable autopilot for targets (so they drive)
        tm = self.client.get_trafficmanager()
        for target in self.targets:
            set_autopilot(target, enabled=True)
            tm.vehicle_percentage_speed_difference(target, 60)

        # Select the first target (ahead) as the one to track
        if self.targets:
            self.tracked_target = self.targets[0]
            print(f"[Setup] Primary target: {self.tracked_target.id}")

        print(f"[Setup] {len(self.targets)} target vehicles with autopilot")

        # Setup sensors
        self.sensor_manager = SensorManager(
            self.world, self.ego, self.config
        )

        # Setup spectator camera - behind target, facing ego
        self.spectator_manager = SpectatorManager(
            self.world, height=25.0, distance=25.0
        )
        # Attach to tracked target, with ego as reference
        if self.tracked_target:
            self.spectator_manager.setup_camera(self.tracked_target, self.ego)
        else:
            self.spectator_manager.setup_camera(self.ego)

        # Warm up sensors
        print("[Setup] Warming up sensors...")
        for _ in range(10):
            self.world.tick()
            self.sensor_manager.clear_queues()
            self.spectator_manager.clear_queue()

        # Initialize pose estimator
        print("[Setup] Loading pose estimation model...")
        self.pose_estimator = PoseEstimator(self.config)

        # Initialize controller
        print("[Setup] Initializing MPC controller...")
        self.controller = MPCController(self.config)

        print("\n[Setup] Complete!")
        print("=" * 60 + "\n")

    def _initialize_tracking(
        self,
        rgb_image: np.ndarray,
        instance_image: np.ndarray,
    ) -> bool:
        """Initialize tracking by mapping instance IDs to actor IDs.

        In the first frame:
        1. Find all vehicle instance IDs
        2. Get bbox mask for each and run pose estimation
        3. Get ground truth poses for all target actors
        4. Match instance IDs to actor IDs by pose similarity
        5. Select which instance to track (the one matching target_ahead)

        Args:
            rgb_image: RGB image.
            instance_image: Instance segmentation image.

        Returns:
            True if tracking initialized successfully.
        """
        print("\n[Init] Initializing instance-to-actor mapping...")

        # Get all vehicle masks
        all_masks = get_all_vehicle_masks(
            self.sensor_manager, instance_image, min_pixels=100
        )

        if not all_masks:
            print("[Init] No vehicles found in first frame")
            return False

        print(f"[Init] Found {len(all_masks)} vehicle instances: {list(all_masks.keys())}")

        # Run pose estimation on each instance
        instance_poses: Dict[int, dict] = {}
        for inst_id, mask in all_masks.items():
            pose = self.pose_estimator.estimate_pose(rgb_image, mask)
            instance_poses[inst_id] = pose
            print(f"[Init]   Instance {inst_id}: dx={pose['dx']:.2f}m, "
                  f"dy={pose['dy']:.2f}m, dyaw={pose.get('dyaw', 0):.1f}°")

        # Match instances to actors using ground truth poses
        self.instance_to_actor = match_instance_to_actor(
            instance_poses, self.targets, self.ego, max_match_dist=8.0
        )

        print(f"[Init] Matched {len(self.instance_to_actor)} instances to actors:")
        for inst_id, actor in self.instance_to_actor.items():
            role = actor.attributes.get("role_name", "unknown") if actor else "None"
            actor_id = actor.id if actor else "N/A"
            print(f"[Init]   Instance {inst_id} -> Actor {actor_id} (role={role})")

        # Select instance to track: prefer the one matching target_ahead
        selected_instance = None

        # Find instance matching tracked_target (target_ahead)
        if self.tracked_target:
            for inst_id, actor in self.instance_to_actor.items():
                if actor is not None and actor.id == self.tracked_target.id:
                    selected_instance = inst_id
                    print(f"[Init] Selected instance {inst_id} (matched target_ahead)")
                    break

        # Fallback: select largest vehicle closest to center
        if selected_instance is None and all_masks:
            height, width = instance_image.shape[:2]
            center_x = width // 2

            best_inst = None
            best_dist = float("inf")

            vehicles = self.sensor_manager.find_vehicle_instances(instance_image, 100)
            for inst_id, count, bbox in vehicles:
                if inst_id in all_masks:
                    bbox_center = (bbox[0] + bbox[2]) // 2
                    dist = abs(bbox_center - center_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_inst = inst_id

            if best_inst is not None:
                selected_instance = best_inst
                print(f"[Init] Fallback: selected instance {selected_instance} (center)")

        if selected_instance is not None:
            self.sensor_manager.set_tracked_instance(selected_instance)
            return True

        print("[Init] Failed to select a vehicle to track")
        return False

    def run(self) -> None:
        """Run the pursuit loop."""
        print("\n" + "=" * 60)
        print("STARTING PURSUIT")
        print("=" * 60)
        print(f"Number of frames: {self.config.num_frames}")
        print(f"Desired following distance: {self.config.desired_distance}m")
        print("Press Ctrl+C to stop\n")

        start_time = time.time()
        last_print_time = start_time
        tracking_initialized = False

        try:
            while self.frame_count < self.config.num_frames:

                # Step simulation
                self.world.tick()

                # Update spectator view (behind target, facing ego)
                self.spectator_manager.update(self.tracked_target, self.ego)

                # Get sensor data
                rgb_image, instance_image = self.sensor_manager.get_sensor_data(
                    timeout=1.0
                )

                if rgb_image is None or instance_image is None:
                    print("[Warning] Missing sensor data, skipping frame")
                    continue

                # First frame: initialize tracking with instance-to-actor mapping
                if not tracking_initialized:
                    if not self._initialize_tracking(rgb_image, instance_image):
                        print("[Warning] Tracking init failed, retrying...")
                        continue
                    tracking_initialized = True

                # Get mask for tracked instance
                target_mask, tracked_id = self.sensor_manager.get_tracked_mask(
                    instance_image
                )

                if target_mask is None:
                    print("[Warning] Lost track of instance, skipping frame")
                    continue

                # Estimate pose for the tracked vehicle
                frame_start = time.time()
                estimated_pose = self.pose_estimator.estimate_pose(
                    rgb_image, target_mask
                )
                inference_time = time.time() - frame_start

                # Get ground truth from matched actor
                gt_pose = None
                matched_actor = self.instance_to_actor.get(tracked_id)
                if matched_actor is not None and matched_actor.is_alive:
                    gt_pose = compute_ground_truth_pose(self.ego, matched_actor)

                # Store pose for statistics
                self.pose_errors.append({
                    "dx": estimated_pose["dx"],
                    "dy": estimated_pose["dy"],
                    "dyaw": estimated_pose.get("dyaw", 0.0),
                    "gt_dx": gt_pose["dx"] if gt_pose else None,
                    "gt_dy": gt_pose["dy"] if gt_pose else None,
                    "gt_dyaw": gt_pose["dyaw"] if gt_pose else None,
                })

                # Get vehicle state
                vehicle_state_dict = get_vehicle_state(self.ego)
                vehicle_state = VehicleState(
                    speed=vehicle_state_dict["speed"],
                    throttle=vehicle_state_dict["throttle"],
                    steer=vehicle_state_dict["steer"],
                    brake=vehicle_state_dict["brake"],
                )

                # Get target velocity from matched actor (if available)
                target_speed = 0.0
                if matched_actor is not None and matched_actor.is_alive:
                    target_speed = get_vehicle_speed(matched_actor)

                # Create target pose object with velocity
                target_pose = TargetPose(
                    dx=estimated_pose["dx"],
                    dy=estimated_pose["dy"],
                    dyaw=estimated_pose.get("dyaw", 0.0),
                    target_speed=target_speed,
                )

                # Compute control
                control_cmd = self.controller.compute_control(
                    target_pose, vehicle_state
                )

                # Apply control
                apply_control(
                    self.ego,
                    control_cmd.throttle,
                    control_cmd.steer,
                    control_cmd.brake,
                )

                # Update statistics
                self.frame_count += 1
                self.total_time += inference_time

                # Print debug info periodically
                current_time = time.time()
                elapsed = current_time - start_time
                if current_time - last_print_time >= 1.0:
                    self._print_debug_info(
                        elapsed,
                        estimated_pose,
                        vehicle_state,
                        control_cmd,
                        inference_time,
                        tracked_id,
                        gt_pose,
                    )
                    last_print_time = current_time

                # Save images
                if self.config.save_video:
                    self._save_images(
                        rgb_image,
                        target_mask,
                        estimated_pose,
                        control_cmd,
                        gt_pose,
                    )

        except KeyboardInterrupt:
            print("\n[Pursuit] Interrupted by user")

        finally:
            self._print_final_stats()

    def _print_debug_info(
        self,
        elapsed: float,
        estimated_pose: dict,
        vehicle_state: VehicleState,
        control_cmd,
        inference_time: float,
        tracked_id: Optional[int] = None,
        gt_pose: Optional[dict] = None,
    ) -> None:
        """Print debug information."""
        progress = self.frame_count / self.config.num_frames * 100
        print(f"\n[t={elapsed:.1f}s] Frame {self.frame_count}/{self.config.num_frames} "
              f"({progress:.1f}%)")
        if tracked_id is not None:
            actor_id = self.instance_to_actor.get(tracked_id)
            actor_str = f", actor={actor_id.id}" if actor_id else ""
            print(f"  Tracking: instance_id={tracked_id}{actor_str}")
        print(f"  Pred: dx={estimated_pose['dx']:.2f}m, "
              f"dy={estimated_pose['dy']:.2f}m, "
              f"dyaw={estimated_pose.get('dyaw', 0.0):.1f}°")
        if gt_pose is not None:
            print(f"  GT:   dx={gt_pose['dx']:.2f}m, "
                  f"dy={gt_pose['dy']:.2f}m, "
                  f"dyaw={gt_pose['dyaw']:.1f}°")
            err_dx = estimated_pose['dx'] - gt_pose['dx']
            err_dy = estimated_pose['dy'] - gt_pose['dy']
            err_dyaw = estimated_pose.get('dyaw', 0.0) - gt_pose['dyaw']
            print(f"  Err:  dx={err_dx:.2f}m, dy={err_dy:.2f}m, dyaw={err_dyaw:.1f}°")
        print(f"  Speed: {vehicle_state.speed * 3.6:.1f} km/h")
        print(f"  Control: throttle={control_cmd.throttle:.2f}, "
              f"steer={control_cmd.steer:.2f}, "
              f"brake={control_cmd.brake:.2f}")
        print(f"  Inference: {inference_time * 1000:.1f}ms")

    def _save_images(
        self,
        rgb_image: np.ndarray,
        target_mask: np.ndarray,
        estimated_pose: dict,
        control_cmd,
        gt_pose: Optional[dict] = None,
    ) -> None:
        """Save ego camera and spectator camera images.

        Args:
            rgb_image: Ego camera RGB image.
            target_mask: Binary mask of target.
            estimated_pose: Estimated pose.
            control_cmd: Control command.
            gt_pose: Ground truth pose (if available).
        """
        # Create ego camera visualization
        ego_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Overlay mask with transparency
        mask_overlay = np.zeros_like(ego_frame)
        mask_overlay[:, :, 1] = target_mask * 255  # Green channel
        ego_frame = cv2.addWeighted(ego_frame, 1.0, mask_overlay, 0.3, 0)

        # Draw bounding box around target
        if target_mask.sum() > 0:
            bbox = self.pose_estimator.get_bbox_from_mask(target_mask)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(ego_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        # Predicted pose (green)
        pred_dyaw = estimated_pose.get('dyaw', 0.0)
        text = f"Pred: dx={estimated_pose['dx']:.1f}m, dy={estimated_pose['dy']:.1f}m, yaw={pred_dyaw:.1f}"
        cv2.putText(ego_frame, text, (10, y_offset), font, 0.6, (0, 255, 0), 2)
        y_offset += 25

        # Ground truth pose (cyan)
        if gt_pose is not None:
            text = f"GT:   dx={gt_pose['dx']:.1f}m, dy={gt_pose['dy']:.1f}m, yaw={gt_pose['dyaw']:.1f}"
            cv2.putText(ego_frame, text, (10, y_offset), font, 0.6, (255, 255, 0), 2)
            y_offset += 25

            # Error (red/magenta)
            err_dx = estimated_pose['dx'] - gt_pose['dx']
            err_dy = estimated_pose['dy'] - gt_pose['dy']
            err_dyaw = pred_dyaw - gt_pose['dyaw']
            text = f"Err:  dx={err_dx:.2f}m, dy={err_dy:.2f}m, yaw={err_dyaw:.1f}"
            cv2.putText(ego_frame, text, (10, y_offset), font, 0.6, (255, 0, 255), 2)
            y_offset += 25

        # Control (white)
        text = f"Ctrl: T={control_cmd.throttle:.2f}, S={control_cmd.steer:.2f}, B={control_cmd.brake:.2f}"
        cv2.putText(ego_frame, text, (10, y_offset), font, 0.6, (255, 255, 255), 2)
        y_offset += 25

        # Frame number
        text = f"Frame: {self.frame_count}"
        cv2.putText(ego_frame, text, (10, y_offset), font, 0.6, (255, 255, 255), 2)

        # Save ego camera image
        ego_path = os.path.join(
            self.ego_output_dir, f"ego_{self.frame_count:05d}.png"
        )
        cv2.imwrite(ego_path, ego_frame)

        # Get and save spectator camera image
        spectator_image = self.spectator_manager.get_image(timeout=0.1)
        if spectator_image is not None:
            spectator_frame = cv2.cvtColor(spectator_image, cv2.COLOR_RGB2BGR)

            # Add frame number overlay
            cv2.putText(
                spectator_frame,
                f"Frame: {self.frame_count}",
                (10, 30),
                font,
                1.0,
                (255, 255, 255),
                2,
            )

            spectator_path = os.path.join(
                self.spectator_output_dir, f"spectator_{self.frame_count:05d}.png"
            )
            cv2.imwrite(spectator_path, spectator_frame)

    def _print_final_stats(self) -> None:
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("PURSUIT STATISTICS")
        print("=" * 60)

        if self.frame_count > 0:
            avg_inference = (self.total_time / self.frame_count) * 1000
            fps = self.frame_count / self.total_time if self.total_time > 0 else 0

            print(f"Total frames: {self.frame_count}")
            print(f"Average inference time: {avg_inference:.1f}ms")
            print(f"Inference FPS: {fps:.1f}")

            if self.pose_errors:
                dx_vals = [e["dx"] for e in self.pose_errors]
                dy_vals = [e["dy"] for e in self.pose_errors]
                dyaw_vals = [e["dyaw"] for e in self.pose_errors]

                print("\nPose predictions:")
                print(f"  Mean dx: {np.mean(dx_vals):.3f}m")
                print(f"  Mean dy: {np.mean(dy_vals):.3f}m")
                print(f"  Mean dyaw: {np.mean(dyaw_vals):.2f}°")

                # Compute errors vs GT if available
                gt_available = [e for e in self.pose_errors if e.get("gt_dx") is not None]
                if gt_available:
                    err_dx = [e["dx"] - e["gt_dx"] for e in gt_available]
                    err_dy = [e["dy"] - e["gt_dy"] for e in gt_available]
                    err_dyaw = [e["dyaw"] - e["gt_dyaw"] for e in gt_available]

                    print(f"\nPose errors vs ground truth ({len(gt_available)} frames):")
                    print(f"  MAE dx: {np.mean(np.abs(err_dx)):.3f}m")
                    print(f"  MAE dy: {np.mean(np.abs(err_dy)):.3f}m")
                    print(f"  MAE dyaw: {np.mean(np.abs(err_dyaw)):.2f}°")
                    print(f"  RMSE dx: {np.sqrt(np.mean(np.square(err_dx))):.3f}m")
                    print(f"  RMSE dy: {np.sqrt(np.mean(np.square(err_dy))):.3f}m")

            if self.config.save_video:
                print(f"\nEgo images saved to: {self.ego_output_dir}")
                print(f"Spectator images saved to: {self.spectator_output_dir}")

        print("=" * 60)

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("\n[Cleanup] Cleaning up...")

        # Destroy sensors
        if self.sensor_manager is not None:
            self.sensor_manager.destroy()

        # Destroy spectator camera
        if self.spectator_manager is not None:
            self.spectator_manager.destroy()

        # Destroy vehicles
        actors_to_destroy = []
        if self.ego is not None:
            actors_to_destroy.append(self.ego)
        actors_to_destroy.extend(self.targets)

        cleanup_actors(actors_to_destroy)
        print("[Cleanup] Vehicles destroyed")

        # Reset world settings
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print("[Cleanup] Complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Vehicle pursuit using pose estimation and MPC control"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/usr/prakt/s0050/ravp/pose_estimation_runs/"
            "pose_estimation_20251129_220737/best_model.pth"
        ),
        help="Path to model checkpoint",
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
        "--town",
        type=str,
        default="Town04",
        help="CARLA town to use",
    )

    parser.add_argument(
        "--num-targets",
        type=int,
        default=3,
        help="Number of target vehicles to spawn",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=2000,
        help="Number of frames to run the pursuit",
    )

    parser.add_argument(
        "--desired-distance",
        type=float,
        default=3.0,
        help="Desired following distance in meters",
    )

    parser.add_argument(
        "--not-save-images",
        action="store_true",
        help="Save images from ego and spectator cameras",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/usr/prakt/s0050/ravp/inference_output",
        help="Directory to save output images",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create config
    config = InferenceConfig()

    # Override from command line
    config.checkpoint_path = args.checkpoint
    config.carla_host = args.carla_host
    config.carla_port = args.carla_port
    config.town = args.town
    config.num_target_vehicles = args.num_targets
    config.num_frames = args.num_frames
    config.desired_distance = args.desired_distance
    config.save_video = not args.not_save_images
    config.video_output_dir = args.output_dir

    # Load model config from checkpoint directory
    config_dir = os.path.dirname(config.checkpoint_path)
    config.config_path = os.path.join(config_dir, "config.json")
    config.load_model_config()

    # Create pursuit instance
    pursuit = VehiclePursuit(config)

    try:
        pursuit.setup()
        pursuit.run()
    finally:
        pursuit.cleanup()


if __name__ == "__main__":
    main()
