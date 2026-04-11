"""CLI entry point for CNN-based target-specific pursuit inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import InferenceConfig
from .geometry import canonicalize_follow_yaw_deg, expand_bbox, mask_iou
from .metrics import InferenceMetrics, build_frame_metrics_row
from .mpc_controller import ControlCommand, MPCController, TargetPose, VehicleState
from .pose_estimator import PoseEstimator
from .scenario import PursuitScenario
from .tracker import OnlineSam3Tracker


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_run_name(config: InferenceConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{config.town}"


def _save_spectator_frame(
    config: InferenceConfig,
    frame_idx: int,
    image: Optional[np.ndarray],
) -> Optional[str]:
    if image is None:
        return None
    _ensure_dir(config.spectator_frames_dir)
    frame_path = os.path.join(config.spectator_frames_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return frame_path


def _save_ego_frame(
    config: InferenceConfig,
    frame_idx: int,
    image: np.ndarray,
) -> str:
    _ensure_dir(config.ego_frames_dir)
    frame_path = os.path.join(config.ego_frames_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return frame_path


def _save_ego_debug_frame(
    config: InferenceConfig,
    frame_idx: int,
    rgb_image: np.ndarray,
    target_mask: Optional[np.ndarray],
    used_pose: Optional[dict],
    gt_pose: dict,
    control: ControlCommand,
) -> Optional[str]:
    if not bool(config.save_debug_images):
        return None

    debug_frames_dir = os.path.join(config.debug_dir, "ego_debug_frames")
    _ensure_dir(debug_frames_dir)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if target_mask is not None and np.any(target_mask):
        mask_overlay = np.zeros_like(frame)
        mask_overlay[:, :, 1] = target_mask.astype(np.uint8) * 255
        frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0.0)

        ys, xs = np.where(target_mask.astype(bool))
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    def add_text(text: str, color: Tuple[int, int, int]) -> None:
        nonlocal y
        cv2.putText(frame, text, (10, y), font, 0.6, color, 2)
        y += 24

    add_text(f"Frame: {frame_idx}", (255, 255, 255))
    add_text(
        "GT: "
        f"dx={gt_pose['dx_m']:.2f}m "
        f"dy={gt_pose['dy_m']:.2f}m "
        f"yaw_f={canonicalize_follow_yaw_deg(gt_pose['yaw_deg']):.1f}deg",
        (255, 255, 0),
    )
    if used_pose is not None:
        add_text(
            "Pred: "
            f"dx={used_pose['dx_m']:.2f}m "
            f"dy={used_pose['dy_m']:.2f}m "
            f"yaw_f={used_pose['yaw_follow_deg']:.1f}deg",
            (0, 255, 0),
        )
    else:
        add_text("Pred: unavailable", (0, 0, 255))
    add_text(
        f"Ctrl: T={control.throttle:.2f} S={control.steer:.2f} B={control.brake:.2f}",
        (255, 255, 255),
    )

    path = os.path.join(debug_frames_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(path, frame)
    return path


def _build_video(frames_dir: str, output_path: str, fps: float) -> Optional[str]:
    if not os.path.isdir(frames_dir):
        return None
    png_files = [name for name in os.listdir(frames_dir) if name.endswith(".png")]
    if not png_files:
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.4f}",
        "-i",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _write_artifact_summary(
    summary_path: str,
    *,
    ego_video_path: Optional[str],
    spectator_video_path: Optional[str],
) -> None:
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    summary["artifacts"] = {
        "ego_video_path": ego_video_path,
        "spectator_video_path": spectator_video_path,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)


def run_pursuit(config: InferenceConfig) -> str:
    """Run the full SAM3 + CNN + MPC pursuit pipeline."""
    if not config.run_name:
        config.run_name = _next_run_name(config)
    if os.path.isdir(config.run_output_dir):
        shutil.rmtree(config.run_output_dir)
    _ensure_dir(config.run_output_dir)
    config.write()

    scenario = PursuitScenario(config)
    metrics = InferenceMetrics(config)
    tracker = OnlineSam3Tracker(config)
    pose_estimator = PoseEstimator(config)
    controller = MPCController(config)

    completion_reason = "max_frames"
    last_pose: Optional[dict] = None
    last_prompt_bbox: Optional[Tuple[int, int, int, int]] = None
    stale_frames = 0
    invisible_streak = 0
    offroad_streak = 0
    lost_streak = 0

    try:
        scenario.setup()

        for frame_idx in range(int(config.num_frames)):
            packet = scenario.tick()
            ego_collision_events = scenario.ego_events.consume_collision_events()
            target_collision_events = scenario.target_events.consume_collision_events()

            if not scenario.target_alive():
                completion_reason = "target_destroyed"
                break
            if not scenario.ego_alive():
                completion_reason = "ego_destroyed"
                break

            gt_pose = scenario.ground_truth_pose()
            ego_state = scenario.ego_vehicle_state()
            ego_world_state = scenario.ego_world_state()
            target_state = scenario.target_vehicle_state()
            offroad = bool(scenario.ego_offroad())
            offroad_streak = offroad_streak + 1 if offroad else 0
            absolute_distance_m = float(scenario.absolute_distance())

            gt_prompt_bbox = scenario.sensors.project_actor_bbox(scenario.target)
            gt_target_mask = scenario.sensors.target_instance_mask(
                packet.instance_image,
                gt_prompt_bbox,
            )

            bootstrap_bbox = None
            if frame_idx == 0:
                if config.bootstrap_bbox_xyxy is not None:
                    bootstrap_bbox = tuple(int(v) for v in config.bootstrap_bbox_xyxy)
                elif (
                    bool(config.bootstrap_with_projected_bbox)
                    and gt_prompt_bbox is not None
                ):
                    bootstrap_bbox = expand_bbox(
                        gt_prompt_bbox,
                        int(config.prompt_bbox_pad_px),
                        int(config.image_width),
                        int(config.image_height),
                    )

            used_pose = None
            pose_available = False
            pose_stale = False
            pose_latency_ms = 0.0
            mask_iou_value = None
            mask_available = False
            bbox_bootstrap_used = False
            bbox_reseed_requested = False
            bbox_reseed_used = False
            tracker_logit_max = None
            tracker_threshold = None
            pred_mask = None

            if frame_idx == 0 and bootstrap_bbox is None:
                invisible_streak += 1
                stale_frames += 1
            else:
                tracker_response = tracker.track(
                    frame_idx,
                    packet.rgb_image,
                    bootstrap_bbox_xyxy=bootstrap_bbox,
                )
                pose_latency_ms += float(tracker_response.get("latency_ms", 0.0) or 0.0)
                bbox_bootstrap_used = bool(
                    tracker_response.get("bbox_bootstrap_used", False)
                )
                mask_available = bool(tracker_response.get("mask_available", False))
                tracker_logit_max = tracker_response.get("tracker_logit_max")
                tracker_threshold = tracker_response.get("tracker_threshold")
                if mask_available:
                    pred_mask = tracker_response["mask"]
                    last_prompt_bbox = tuple(
                        int(v) for v in tracker_response["mask_bbox_xyxy"]
                    )
                    invisible_streak = 0
                else:
                    invisible_streak += 1

                if (
                    not mask_available
                    and bool(config.enable_bbox_reseed)
                    and frame_idx > 0
                    and last_prompt_bbox is not None
                ):
                    bbox_reseed_requested = True
                    reseed_bbox = expand_bbox(
                        last_prompt_bbox,
                        max(int(config.prompt_bbox_pad_px) * 3, 64),
                        int(config.image_width),
                        int(config.image_height),
                    )
                    tracker_response = tracker.track(
                        frame_idx,
                        packet.rgb_image,
                        reseed_bbox_xyxy=reseed_bbox,
                    )
                    pose_latency_ms += float(
                        tracker_response.get("latency_ms", 0.0) or 0.0
                    )
                    bbox_reseed_used = bool(
                        tracker_response.get("bbox_reseed_used", False)
                    )
                    mask_available = bool(tracker_response.get("mask_available", False))
                    tracker_logit_max = tracker_response.get("tracker_logit_max")
                    tracker_threshold = tracker_response.get("tracker_threshold")
                    if mask_available:
                        pred_mask = tracker_response["mask"]
                        last_prompt_bbox = tuple(
                            int(v) for v in tracker_response["mask_bbox_xyxy"]
                        )
                        invisible_streak = 0

                if mask_available and pred_mask is not None:
                    if gt_target_mask is not None:
                        mask_iou_value = float(mask_iou(gt_target_mask, pred_mask))
                    pose_start = time.time()
                    pose_prediction = pose_estimator.estimate_pose(
                        packet.rgb_image,
                        pred_mask.astype(np.uint8),
                    )
                    pose_latency_ms += (time.time() - pose_start) * 1000.0
                    used_pose = {
                        "dx_m": float(pose_prediction["dx_m"]),
                        "dy_m": float(pose_prediction["dy_m"]),
                        "yaw_follow_deg": float(pose_prediction["yaw_follow_deg"]),
                    }
                    pose_available = True
                    pose_stale = False
                    stale_frames = 0
                    last_pose = dict(used_pose)
                else:
                    stale_frames += 1

            if not pose_available and last_pose is not None:
                if stale_frames <= int(config.max_pose_hold_frames):
                    used_pose = dict(last_pose)
                    pose_available = True
                    pose_stale = True
                else:
                    used_pose = None

            if used_pose is None:
                control = ControlCommand(throttle=0.0, steer=0.0, brake=0.4)
            else:
                control = controller.compute_control(
                    TargetPose(
                        dx_m=float(used_pose["dx_m"]),
                        dy_m=float(used_pose["dy_m"]),
                        yaw_follow_deg=float(used_pose["yaw_follow_deg"]),
                        target_speed_mps=float(target_state["speed_mps"]),
                    ),
                    VehicleState(
                        speed_mps=float(ego_state["speed_mps"]),
                        throttle=float(ego_state["throttle"]),
                        steer=float(ego_state["steer"]),
                        brake=float(ego_state["brake"]),
                    ),
                )

            metrics.add_row(
                build_frame_metrics_row(
                    config=config,
                    frame_idx=frame_idx,
                    tick=packet.tick,
                    gt_pose=gt_pose,
                    used_pose=used_pose,
                    pose_available=pose_available,
                    pose_stale=pose_stale,
                    pose_latency_ms=pose_latency_ms,
                    mask_iou=mask_iou_value,
                    mask_available=mask_available,
                    bbox_bootstrap_used=bbox_bootstrap_used,
                    bbox_reseed_requested=bbox_reseed_requested,
                    bbox_reseed_used=bbox_reseed_used,
                    tracker_logit_max=tracker_logit_max,
                    tracker_threshold=tracker_threshold,
                    ego_state=ego_state,
                    target_state=target_state,
                    ego_world_state=ego_world_state,
                    command_throttle=control.throttle,
                    command_steer=control.steer,
                    command_brake=control.brake,
                    offroad=offroad,
                    ego_collision_events=ego_collision_events,
                    target_collision_events=target_collision_events,
                    ego_lane_invasions_total=scenario.ego_events.total_lane_invasions,
                    ego_collisions_total=scenario.ego_events.total_collisions,
                    target_lane_invasions_total=scenario.target_events.total_lane_invasions,
                    target_collisions_total=scenario.target_events.total_collisions,
                    absolute_distance_m=absolute_distance_m,
                )
            )

            scenario.apply_control(
                control.throttle,
                control.steer,
                control.brake,
            )
            _save_ego_frame(config, frame_idx, packet.rgb_image)
            _save_spectator_frame(config, frame_idx, packet.spectator_image)
            _save_ego_debug_frame(
                config,
                frame_idx,
                packet.rgb_image,
                pred_mask,
                used_pose,
                gt_pose,
                control,
            )

            lost_condition = (
                absolute_distance_m >= float(config.lost_distance_m)
                or (
                    float(ego_state["speed_mps"])
                    < float(config.lost_ego_stationary_speed_mps)
                    and float(target_state["speed_mps"])
                    > float(config.lost_target_speed_mps)
                )
            )
            lost_streak = lost_streak + 1 if lost_condition else 0

            if ego_collision_events > 0:
                completion_reason = "ego_collision"
                break
            if target_collision_events > 0:
                completion_reason = "target_collision"
                break
            if offroad_streak >= int(config.ego_offroad_breach_frames):
                completion_reason = "ego_left_driving_lane"
                break
            if invisible_streak >= int(config.target_out_of_view_breach_frames):
                completion_reason = "target_out_of_view"
                break
            if lost_streak >= int(config.lost_patience_frames):
                completion_reason = "ego_lost_target"
                break
    finally:
        scenario.cleanup()

    summary_path = metrics.write(completion_reason)
    fps = 10.0
    if float(config.fixed_delta_seconds) > 1e-6:
        fps = 1.0 / float(config.fixed_delta_seconds)
    ego_video_path = _build_video(config.ego_frames_dir, config.ego_video_path, fps)
    spectator_video_path = None
    if bool(config.enable_spectator_camera):
        spectator_video_path = _build_video(
            config.spectator_frames_dir,
            config.spectator_video_path,
            fps,
        )
    _write_artifact_summary(
        summary_path,
        ego_video_path=ego_video_path,
        spectator_video_path=spectator_video_path,
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 + CNN + MPC target-specific pursuit inference",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2150)
    parser.add_argument("--town", default="Town02")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "inference_output",
    ))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--desired-distance", type=float, default=8.0)
    parser.add_argument("--num-background-vehicles", type=int, default=20)
    parser.add_argument("--initial-target-distance", type=float, default=12.0)
    parser.add_argument("--ego-initial-speed", type=float, default=0.0)
    parser.add_argument("--target-speed-difference", type=float, default=80.0)
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument("--pose-device", default="cuda:0")
    parser.add_argument(
        "--bootstrap-bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Optional first-frame target bbox in xyxy format.",
    )
    parser.add_argument(
        "--no-projected-bootstrap",
        action="store_true",
        help="Disable the simulator-projected frame-0 bootstrap box.",
    )
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--save-tracking-masks", action="store_true")
    parser.add_argument("--no-spectator-camera", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        checkpoint_path=args.checkpoint_path,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        town=args.town,
        random_seed=args.random_seed,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        run_name=args.run_name,
        desired_distance_m=args.desired_distance,
        num_background_vehicles=args.num_background_vehicles,
        initial_target_distance_m=args.initial_target_distance,
        ego_initial_speed_mps=args.ego_initial_speed,
        target_speed_difference_pct=args.target_speed_difference,
        sam3_device=args.sam3_device,
        pose_device=args.pose_device,
        bootstrap_bbox_xyxy=(
            tuple(args.bootstrap_bbox) if args.bootstrap_bbox is not None else None
        ),
        bootstrap_with_projected_bbox=not args.no_projected_bootstrap,
        save_debug_images=args.save_debug_images,
        save_tracking_masks=args.save_tracking_masks,
        enable_spectator_camera=not args.no_spectator_camera,
    )
    summary_path = run_pursuit(config)
    print(summary_path)


if __name__ == "__main__":
    main()
