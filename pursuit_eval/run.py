"""CLI entry point for GT and detector-driven pursuit evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .config import PursuitEvalConfig
from .controller import ControlCommand, MPCFollower, RelativeTargetPose, VehicleState
from .geometry import canonicalize_follow_yaw_deg, expand_bbox, mask_iou
from .metrics import PursuitMetrics, build_frame_metrics_row
from .perception import MMDet3DPoseSource, OnlineSam3Tracker
from .scenario import PursuitScenario


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_spectator_frame(
    config: PursuitEvalConfig,
    frame_idx: int,
    image: Optional[np.ndarray],
) -> Optional[str]:
    if image is None:
        return None
    _ensure_dir(config.spectator_frames_dir)
    frame_path = os.path.join(
        config.spectator_frames_dir, f"frame_{
            frame_idx:06d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return frame_path


def _build_spectator_video(config: PursuitEvalConfig) -> Optional[str]:
    if not bool(config.enable_spectator_camera):
        return None
    if not os.path.isdir(config.spectator_frames_dir):
        return None
    png_files = [name for name in os.listdir(
        config.spectator_frames_dir) if name.endswith(".png")]
    if not png_files:
        return None

    fps = 10.0
    if float(config.fixed_delta_seconds) > 1e-6:
        fps = 1.0 / float(config.fixed_delta_seconds)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.4f}",
        "-i",
        os.path.join(config.spectator_frames_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        config.spectator_video_path,
    ]
    subprocess.run(cmd, check=True)
    return config.spectator_video_path


def _write_artifact_summary(
    summary_path: str,
    *,
    spectator_video_path: Optional[str],
    spectator_frames_dir: Optional[str],
) -> None:
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    summary["artifacts"] = {
        "spectator_video_path": spectator_video_path,
        "spectator_frames_dir": spectator_frames_dir,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)


def _next_run_name(config: PursuitEvalConfig) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{config.pose_source}_{config.town}"


def run_pursuit(config: PursuitEvalConfig) -> str:
    if not config.run_name:
        config.run_name = _next_run_name(config)
    if os.path.isdir(config.run_output_dir):
        shutil.rmtree(config.run_output_dir)
    _ensure_dir(config.run_output_dir)
    config.write()

    scenario = PursuitScenario(config)
    metrics = PursuitMetrics(config)
    controller = MPCFollower(config)
    tracker = OnlineSam3Tracker(
        config) if config.pose_source == "detector" else None
    detector = MMDet3DPoseSource(
        config) if config.pose_source == "detector" else None

    completion_reason = "max_frames"
    last_pose = None
    last_prompt_bbox = None
    stale_frames = 0
    invisible_streak = 0
    offroad_streak = 0
    follow_guard_streak = 0
    trace_path = os.path.join(config.run_output_dir, "trace.log")

    def trace(message: str) -> None:
        with open(trace_path, "a") as handle:
            handle.write(message + "\n")

    try:
        trace("setup:start")
        scenario.setup()
        trace("setup:done")

        for frame_idx in range(int(config.num_frames)):
            trace(f"frame:{frame_idx}:start")
            packet = scenario.tick()
            collision_events = scenario.sensors.consume_collision_events()

            if not scenario.target_alive():
                completion_reason = "target_destroyed"
                break
            if not scenario.ego_alive():
                completion_reason = "ego_destroyed"
                break

            gt_pose = scenario.ground_truth_pose()
            ego_state_dict = scenario.ego_vehicle_state()
            ego_speed = float(ego_state_dict["speed_mps"])
            target_speed = float(scenario.target_speed())
            offroad = bool(scenario.ego_offroad())
            offroad_streak = offroad_streak + 1 if offroad else 0
            follow_guard_active = (
                float(
                    gt_pose["dx_m"]) < float(
                    config.follow_guard_min_dx_m) or abs(
                    float(
                        gt_pose["dy_m"])) > float(
                        config.follow_guard_lateral_abs_m) or abs(
                            float(
                                gt_pose["yaw_deg"])) > float(
                                    config.follow_guard_yaw_abs_deg))
            follow_guard_streak = follow_guard_streak + 1 if follow_guard_active else 0

            used_pose = None
            pose_available = False
            pose_stale = False
            pose_latency_ms = 0.0
            mask_iou_value = None
            sam3_mask_available = config.pose_source == "gt"
            detector_pose_available = config.pose_source == "gt"
            bbox_bootstrap_used = False
            bbox_reseed_requested = False
            bbox_reseed_used = False
            bbox_reseed_reason = ""
            tracker_logit_max = None
            tracker_threshold = None

            gt_prompt_bbox = scenario.sensors.project_actor_bbox(
                scenario.target)
            target_mask = scenario.sensors.target_instance_mask(
                packet.instance_image, gt_prompt_bbox)

            if config.pose_source == "gt":
                used_pose = dict(gt_pose)
                pose_available = True
            else:
                bootstrap_bbox = None
                if frame_idx == 0 and bool(
                        config.bootstrap_with_gt_bbox) and gt_prompt_bbox is not None:
                    bootstrap_bbox = expand_bbox(
                        gt_prompt_bbox,
                        int(config.prompt_bbox_pad_px),
                        int(config.image_width),
                        int(config.image_height),
                    )

                if frame_idx == 0 and bootstrap_bbox is None:
                    invisible_streak += 1
                    stale_frames += 1
                    if last_pose is not None and stale_frames <= int(
                            config.max_pose_hold_frames):
                        used_pose = dict(last_pose)
                        pose_available = True
                        pose_stale = True
                    else:
                        used_pose = None
                        pose_available = False
                else:
                    sam3_response = tracker.track(
                        frame_idx,
                        packet.rgb_image,
                        bootstrap_bbox_xyxy=bootstrap_bbox,
                    )
                    if sam3_response.get("latency_ms") is not None:
                        pose_latency_ms += float(sam3_response["latency_ms"])
                    bbox_bootstrap_used = bool(
                        sam3_response.get(
                            "bbox_bootstrap_used", False))
                    sam3_mask_available = bool(
                        sam3_response.get("mask_available", False))
                    tracker_logit_max = sam3_response.get("tracker_logit_max")
                    tracker_threshold = sam3_response.get("tracker_threshold")
                    if not sam3_mask_available and sam3_response.get("error"):
                        trace(
                            f"frame:{frame_idx}:sam3_error:{
                                sam3_response.get('error')}")

                    pred_mask = sam3_response.get("mask")
                    if target_mask is not None and pred_mask is not None:
                        mask_iou_value = float(
                            mask_iou(target_mask, pred_mask))

                    if sam3_mask_available and pred_mask is not None:
                        candidate_mask_bbox = tuple(
                            int(v) for v in sam3_response["mask_bbox_xyxy"])
                        detector_response = detector.estimate_pose(
                            lidar_points=packet.lidar_points,
                            target_mask=pred_mask,
                            mask_bbox_xyxy=candidate_mask_bbox,
                            lidar_to_camera=scenario.sensors.lidar_to_camera_matrix(),
                        )
                        if detector_response.get("latency_ms") is not None:
                            pose_latency_ms += float(
                                detector_response["latency_ms"])
                        detector_pose_available = bool(
                            detector_response.get("pose_available", False))
                        if detector_pose_available:
                            used_pose = {
                                "dx_m": float(
                                    detector_response["dx_m"]), "dy_m": float(
                                    detector_response["dy_m"]), "yaw_deg": float(
                                    canonicalize_follow_yaw_deg(
                                        float(
                                            detector_response["yaw_deg"]))), }
                            pose_available = True
                            pose_stale = False
                            stale_frames = 0
                            last_pose = dict(used_pose)
                            last_prompt_bbox = tuple(
                                int(v)
                                for v in detector_response
                                ["projection_bbox_xyxy"])

                    if (
                        not detector_pose_available
                        and bool(config.enable_bbox_reseed)
                        and frame_idx > 0
                        and last_prompt_bbox is not None
                    ):
                        bbox_reseed_requested = True
                        bbox_reseed_reason = (
                            "sam3_mask_missing"
                            if not sam3_mask_available
                            else "detector_no_match"
                        )
                        trace(
                            "frame:"
                            f"{frame_idx}:bbox_reseed_requested:"
                            f"{bbox_reseed_reason}"
                        )
                        reseed_bbox = expand_bbox(
                            last_prompt_bbox,
                            max(int(config.prompt_bbox_pad_px) * 3, 64),
                            int(config.image_width),
                            int(config.image_height),
                        )
                        sam3_response = tracker.track(
                            frame_idx,
                            packet.rgb_image,
                            reseed_bbox_xyxy=reseed_bbox,
                        )
                        if sam3_response.get("latency_ms") is not None:
                            pose_latency_ms += float(
                                sam3_response["latency_ms"])
                        bbox_reseed_used = bool(
                            sam3_response.get(
                                "bbox_reseed_used", False))
                        sam3_mask_available = bool(
                            sam3_response.get("mask_available", False))
                        tracker_logit_max = sam3_response.get(
                            "tracker_logit_max")
                        tracker_threshold = sam3_response.get(
                            "tracker_threshold")
                        if not sam3_mask_available and sam3_response.get(
                                "error"):
                            trace(
                                f"frame:{frame_idx}:sam3_reseed_error:{
                                    sam3_response.get('error')}")

                        pred_mask = sam3_response.get("mask")
                        if target_mask is not None and pred_mask is not None:
                            mask_iou_value = float(
                                mask_iou(target_mask, pred_mask))
                        if sam3_mask_available and pred_mask is not None:
                            reseed_mask_bbox = tuple(
                                int(v) for v in sam3_response["mask_bbox_xyxy"])
                            detector_response = detector.estimate_pose(
                                lidar_points=packet.lidar_points,
                                target_mask=pred_mask,
                                mask_bbox_xyxy=reseed_mask_bbox,
                                lidar_to_camera=(
                                    scenario.sensors.lidar_to_camera_matrix()
                                ),
                            )
                            if detector_response.get("latency_ms") is not None:
                                pose_latency_ms += float(
                                    detector_response["latency_ms"])
                            detector_pose_available = bool(
                                detector_response.get("pose_available", False))
                            if detector_pose_available:
                                used_pose = {
                                    "dx_m": float(
                                        detector_response["dx_m"]), "dy_m": float(
                                        detector_response["dy_m"]), "yaw_deg": float(
                                        canonicalize_follow_yaw_deg(
                                            float(
                                                detector_response["yaw_deg"]))), }
                                pose_available = True
                                pose_stale = False
                                stale_frames = 0
                                last_pose = dict(used_pose)
                                last_prompt_bbox = tuple(
                                    int(v)
                                    for v in detector_response
                                    ["projection_bbox_xyxy"])
                                trace(f"frame:{frame_idx}:bbox_reseed_success")

                    if not detector_pose_available:
                        stale_frames += 1
                        if last_pose is not None and stale_frames <= int(
                                config.max_pose_hold_frames):
                            used_pose = dict(last_pose)
                            pose_available = True
                            pose_stale = True
                        else:
                            used_pose = None
                            pose_available = False

            if gt_prompt_bbox is not None:
                invisible_streak = 0

            if used_pose is None:
                control = ControlCommand(throttle=0.0, steer=0.0, brake=0.4)
            else:
                control = controller.compute_control(
                    RelativeTargetPose(
                        dx_m=float(used_pose["dx_m"]),
                        dy_m=float(used_pose["dy_m"]),
                        yaw_deg=float(used_pose["yaw_deg"]),
                        target_speed_mps=target_speed,
                    ),
                    VehicleState(
                        speed_mps=ego_speed,
                        throttle=float(ego_state_dict["throttle"]),
                        steer=float(ego_state_dict["steer"]),
                        brake=float(ego_state_dict["brake"]),
                    ),
                )

            scenario.apply_control(
                control.throttle,
                control.steer,
                control.brake)
            metrics.add_row(
                build_frame_metrics_row(
                    config=config,
                    frame_idx=frame_idx,
                    tick=packet.tick,
                    pose_source=config.pose_source,
                    gt_pose=gt_pose,
                    ego_speed_mps=ego_speed,
                    target_speed_mps=target_speed,
                    throttle=control.throttle,
                    steer=control.steer,
                    brake=control.brake,
                    offroad=offroad,
                    collision_events=collision_events,
                    pose_available=pose_available,
                    pose_stale=pose_stale,
                    pose_latency_ms=pose_latency_ms,
                    used_pose=used_pose,
                    mask_iou=mask_iou_value,
                    mask_available=sam3_mask_available,
                    detector_pose_available=detector_pose_available,
                    bbox_bootstrap_used=bbox_bootstrap_used,
                    bbox_reseed_requested=bbox_reseed_requested,
                    bbox_reseed_used=bbox_reseed_used,
                    bbox_reseed_reason=bbox_reseed_reason,
                    tracker_logit_max=tracker_logit_max,
                    tracker_threshold=tracker_threshold,
                )
            )
            _save_spectator_frame(config, frame_idx, packet.spectator_image)
            trace(f"frame:{frame_idx}:done")

            if collision_events > 0:
                completion_reason = "collision"
                break
            if offroad_streak >= int(config.ego_offroad_breach_frames):
                completion_reason = "ego_left_driving_lane"
                break
            if invisible_streak >= int(
                    config.target_out_of_view_breach_frames):
                completion_reason = "target_out_of_view"
                break
            if (
                bool(config.stop_on_follow_guard_breach)
                and follow_guard_streak >= int(config.follow_guard_breach_frames)
            ):
                completion_reason = "target_left_follow_regime"
                break
            if not scenario.target_alive():
                completion_reason = "target_destroyed"
                break
            if not scenario.ego_alive():
                completion_reason = "ego_destroyed"
                break
    finally:
        trace("cleanup:start")
        scenario.cleanup()
        trace("cleanup:done")

    trace("metrics:write:start")
    summary_path = metrics.write(completion_reason)
    spectator_video_path = _build_spectator_video(config)
    _write_artifact_summary(
        summary_path,
        spectator_video_path=spectator_video_path,
        spectator_frames_dir=(
            config.spectator_frames_dir if os.path.isdir(
                config.spectator_frames_dir) else None),
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh MPC pursuit evaluation")
    parser.add_argument(
        "--pose-source",
        default="gt",
        choices=(
            "gt",
            "detector"))
    parser.add_argument("--town", default="Town02")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2150)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pursuit_eval_output",
        ),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--desired-distance", type=float, default=8.0)
    parser.add_argument("--num-background-vehicles", type=int, default=60)
    parser.add_argument("--initial-target-distance", type=float, default=12.0)
    parser.add_argument("--ego-initial-speed", type=float, default=0.0)
    parser.add_argument("--target-speed-difference", type=float, default=80.0)
    parser.add_argument("--stop-on-follow-guard-breach", action="store_true")
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--save-tracking-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PursuitEvalConfig(
        pose_source=args.pose_source,
        town=args.town,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        random_seed=args.random_seed,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        run_name=args.run_name,
        desired_distance_m=args.desired_distance,
        num_background_vehicles=args.num_background_vehicles,
        initial_target_distance_m=args.initial_target_distance,
        ego_initial_speed_mps=args.ego_initial_speed,
        target_speed_difference_pct=args.target_speed_difference,
        stop_on_follow_guard_breach=args.stop_on_follow_guard_breach,
        sam3_device=args.sam3_device,
        detector_device=args.detector_device,
        save_debug_images=args.save_debug_images,
        save_tracking_masks=args.save_tracking_masks,
    )
    report_path = run_pursuit(config)
    print(report_path)


if __name__ == "__main__":
    main()
