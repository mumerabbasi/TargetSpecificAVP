"""CLI entry point for fresh GT and detector-driven pursuit evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import PursuitEvalConfig
from .controller import ControlCommand, MPCFollower, RelativeTargetPose, VehicleState
from .geometry import canonicalize_follow_yaw_deg, expand_bbox, mask_iou
from .metrics import PursuitMetrics, build_frame_metrics_row
from .scenario import PursuitScenario


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_json_write(path: str, payload: Dict[str, object]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _save_spectator_frame(config: PursuitEvalConfig, frame_idx: int, image: Optional[np.ndarray]) -> Optional[str]:
    if image is None:
        return None
    _ensure_dir(config.spectator_frames_dir)
    frame_path = os.path.join(
        config.spectator_frames_dir,
        "frame_{:06d}.png".format(frame_idx),
    )
    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return frame_path


def _build_spectator_video(config: PursuitEvalConfig) -> Optional[str]:
    if not bool(config.enable_spectator_camera):
        return None
    if not os.path.isdir(config.spectator_frames_dir):
        return None
    png_files = [name for name in os.listdir(config.spectator_frames_dir) if name.endswith(".png")]
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
        "{:.4f}".format(fps),
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


class FileWorkerClient:
    """Minimal file-backed subprocess worker client."""

    def __init__(
        self,
        config: PursuitEvalConfig,
        *,
        worker_dir: str,
        requests_dir: str,
        responses_dir: str,
        assets_dir: str,
        env_name: str,
        module_name: str,
    ) -> None:
        self.config = config
        self.worker_dir = worker_dir
        self.requests_dir = requests_dir
        self.responses_dir = responses_dir
        self.assets_dir = assets_dir
        self.env_name = env_name
        self.module_name = module_name
        self.process = None

    def start(self) -> None:
        if os.path.isdir(self.worker_dir):
            shutil.rmtree(self.worker_dir)
        _ensure_dir(self.requests_dir)
        _ensure_dir(self.responses_dir)
        _ensure_dir(self.assets_dir)

        cmd = [
            "conda",
            "run",
            "-n",
            self.env_name,
            "python",
            "-m",
            self.module_name,
            "--config",
            os.path.join(self.config.run_output_dir, "config.json"),
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        ready_path = os.path.join(self.worker_dir, "ready.json")
        deadline = time.time() + float(self.config.worker_start_timeout_s)
        while time.time() < deadline:
            if os.path.exists(ready_path):
                return
            if self.process.poll() is not None:
                raise RuntimeError("{} exited during startup.".format(self.module_name))
            time.sleep(float(self.config.worker_poll_interval_s))
        raise TimeoutError("Timed out waiting for {} to become ready.".format(self.module_name))

    def stop(self) -> None:
        if self.process is None:
            return

        stop_path = os.path.join(self.requests_dir, "__stop__.json")
        _atomic_json_write(stop_path, {"type": "stop"})
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _wait_for_response(self, frame_idx: int) -> Dict[str, object]:
        response_path = os.path.join(
            self.responses_dir,
            "frame_{:06d}.json".format(frame_idx),
        )
        deadline = time.time() + float(self.config.worker_step_timeout_s)
        while time.time() < deadline:
            if os.path.exists(response_path):
                try:
                    with open(response_path, "r") as handle:
                        response = json.load(handle)
                    os.remove(response_path)
                    return response
                except json.JSONDecodeError:
                    time.sleep(float(self.config.worker_poll_interval_s))
                    continue
            if self.process.poll() is not None:
                raise RuntimeError("{} exited during inference.".format(self.module_name))
            time.sleep(float(self.config.worker_poll_interval_s))
        raise TimeoutError("Timed out waiting for response at frame {}".format(frame_idx))


class Sam3WorkerClient(FileWorkerClient):
    """Client for the SAM3 localization worker."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        super(Sam3WorkerClient, self).__init__(
            config,
            worker_dir=config.sam3_worker_dir,
            requests_dir=config.sam3_requests_dir,
            responses_dir=config.sam3_responses_dir,
            assets_dir=config.sam3_assets_dir,
            env_name=config.sam3_worker_env,
            module_name="pursuit_eval.sam3_worker",
        )
        _ensure_dir(config.sam3_masks_dir)

    def infer(
        self,
        frame_idx: int,
        rgb_image: np.ndarray,
        bootstrap_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
        reseed_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
        reseed_reason: str = "",
    ) -> Dict[str, object]:
        if self.process is None:
            raise RuntimeError("SAM3 worker is not running.")

        rgb_path = os.path.join(
            self.assets_dir,
            "frame_{:06d}_rgb.png".format(frame_idx),
        )
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        request = {
            "frame": int(frame_idx),
            "rgb_path": rgb_path,
            "bootstrap_bbox_xyxy": None
            if bootstrap_bbox_xyxy is None
            else [int(v) for v in bootstrap_bbox_xyxy],
            "reseed_bbox_xyxy": None
            if reseed_bbox_xyxy is None
            else [int(v) for v in reseed_bbox_xyxy],
            "reseed_reason": str(reseed_reason),
        }
        request_path = os.path.join(
            self.requests_dir,
            "frame_{:06d}.json".format(frame_idx),
        )
        _atomic_json_write(request_path, request)
        return self._wait_for_response(frame_idx)


class DetectorWorkerClient(FileWorkerClient):
    """Client for the 3D detector worker."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        super(DetectorWorkerClient, self).__init__(
            config,
            worker_dir=config.detector_worker_dir,
            requests_dir=config.detector_requests_dir,
            responses_dir=config.detector_responses_dir,
            assets_dir=config.detector_assets_dir,
            env_name=config.detector_worker_env,
            module_name="pursuit_eval.detector_worker",
        )

    def infer(
        self,
        frame_idx: int,
        lidar_points: np.ndarray,
        mask: np.ndarray,
        mask_bbox_xyxy: Tuple[int, int, int, int],
        lidar_to_camera: np.ndarray,
    ) -> Dict[str, object]:
        if self.process is None:
            raise RuntimeError("Detector worker is not running.")

        lidar_path = os.path.join(
            self.assets_dir,
            "frame_{:06d}_lidar.npy".format(frame_idx),
        )
        mask_path = os.path.join(
            self.assets_dir,
            "frame_{:06d}_mask.npy".format(frame_idx),
        )
        np.save(lidar_path, lidar_points)
        np.save(mask_path, mask.astype(np.uint8))

        request = {
            "frame": int(frame_idx),
            "lidar_path": lidar_path,
            "mask_path": mask_path,
            "mask_bbox_xyxy": [int(v) for v in mask_bbox_xyxy],
            "lidar_to_camera": np.asarray(lidar_to_camera, dtype=np.float64).tolist(),
        }
        request_path = os.path.join(
            self.requests_dir,
            "frame_{:06d}.json".format(frame_idx),
        )
        _atomic_json_write(request_path, request)
        return self._wait_for_response(frame_idx)


def _load_binary_mask(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _next_run_name(config: PursuitEvalConfig) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return "{}_{}_{}".format(stamp, config.pose_source, config.town)


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
    sam3_worker = Sam3WorkerClient(config) if config.pose_source == "detector" else None
    detector_worker = DetectorWorkerClient(config) if config.pose_source == "detector" else None

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
        if sam3_worker is not None:
            trace("sam3_worker:start")
            sam3_worker.start()
            trace("sam3_worker:done")
        if detector_worker is not None:
            trace("detector_worker:start")
            detector_worker.start()
            trace("detector_worker:done")

        for frame_idx in range(int(config.num_frames)):
            trace("frame:{}:start".format(frame_idx))
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
                float(gt_pose["dx_m"]) < float(config.follow_guard_min_dx_m)
                or abs(float(gt_pose["dy_m"])) > float(config.follow_guard_lateral_abs_m)
                or abs(float(gt_pose["yaw_deg"])) > float(config.follow_guard_yaw_abs_deg)
            )
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

            gt_prompt_bbox = scenario.sensors.project_actor_bbox(scenario.target)
            target_mask = scenario.sensors.target_instance_mask(packet.instance_image, gt_prompt_bbox)
            if config.pose_source == "gt":
                used_pose = dict(gt_pose)
                pose_available = True
            else:
                bootstrap_bbox = None
                if frame_idx == 0 and bool(config.bootstrap_with_gt_bbox) and gt_prompt_bbox is not None:
                    bootstrap_bbox = expand_bbox(
                        gt_prompt_bbox,
                        int(config.prompt_bbox_pad_px),
                        int(config.image_width),
                        int(config.image_height),
                    )

                if frame_idx == 0 and bootstrap_bbox is None:
                    invisible_streak += 1
                    stale_frames += 1
                    if last_pose is not None and stale_frames <= int(config.max_pose_hold_frames):
                        used_pose = dict(last_pose)
                        pose_available = True
                        pose_stale = True
                    else:
                        used_pose = None
                        pose_available = False
                else:
                    sam3_response = sam3_worker.infer(
                        frame_idx,
                        packet.rgb_image,
                        bootstrap_bbox_xyxy=bootstrap_bbox,
                    )
                    if sam3_response.get("latency_ms") is not None:
                        pose_latency_ms = float(pose_latency_ms) + float(sam3_response["latency_ms"])
                    bbox_bootstrap_used = bool(sam3_response.get("bbox_bootstrap_used", False))
                    sam3_mask_available = bool(sam3_response.get("mask_available", False))
                    tracker_logit_max = sam3_response.get("tracker_logit_max")
                    tracker_threshold = sam3_response.get("tracker_threshold")
                    if not sam3_mask_available and sam3_response.get("error"):
                        trace(
                            "frame:{}:sam3_error:{}".format(
                                frame_idx,
                                sam3_response.get("error"),
                            )
                        )
                    pred_mask = _load_binary_mask(sam3_response.get("mask_path"))
                    if target_mask is not None and pred_mask is not None:
                        mask_iou_value = float(mask_iou(target_mask, pred_mask))

                    if sam3_mask_available and pred_mask is not None:
                        candidate_mask_bbox = tuple(int(v) for v in sam3_response["mask_bbox_xyxy"])
                        detector_response = detector_worker.infer(
                            frame_idx,
                            packet.lidar_points,
                            pred_mask,
                            candidate_mask_bbox,
                            scenario.sensors.lidar_to_camera_matrix(),
                        )
                        if detector_response.get("latency_ms") is not None:
                            pose_latency_ms = float(pose_latency_ms) + float(detector_response["latency_ms"])
                        detector_pose_available = bool(detector_response.get("pose_available", False))
                        if detector_pose_available:
                            used_pose = {
                                "dx_m": float(detector_response["dx_m"]),
                                "dy_m": float(detector_response["dy_m"]),
                                "yaw_deg": float(
                                    canonicalize_follow_yaw_deg(float(detector_response["yaw_deg"]))
                                ),
                            }
                            pose_available = True
                            pose_stale = False
                            stale_frames = 0
                            last_pose = dict(used_pose)
                            last_prompt_bbox = tuple(int(v) for v in detector_response["projection_bbox_xyxy"])

                    if (
                        not detector_pose_available
                        and bool(config.enable_bbox_reseed)
                        and frame_idx > 0
                        and last_prompt_bbox is not None
                    ):
                        bbox_reseed_requested = True
                        bbox_reseed_reason = (
                            "sam3_mask_missing" if not sam3_mask_available else "detector_no_match"
                        )
                        trace(
                            "frame:{}:bbox_reseed_requested:{}".format(
                                frame_idx,
                                bbox_reseed_reason,
                            )
                        )
                        reseed_bbox = expand_bbox(
                            last_prompt_bbox,
                            max(int(config.prompt_bbox_pad_px) * 3, 64),
                            int(config.image_width),
                            int(config.image_height),
                        )
                        sam3_response = sam3_worker.infer(
                            frame_idx,
                            packet.rgb_image,
                            reseed_bbox_xyxy=reseed_bbox,
                            reseed_reason=bbox_reseed_reason,
                        )
                        if sam3_response.get("latency_ms") is not None:
                            pose_latency_ms = float(pose_latency_ms) + float(sam3_response["latency_ms"])
                        bbox_reseed_used = bool(sam3_response.get("bbox_reseed_used", False))
                        sam3_mask_available = bool(sam3_response.get("mask_available", False))
                        tracker_logit_max = sam3_response.get("tracker_logit_max")
                        tracker_threshold = sam3_response.get("tracker_threshold")
                        if not sam3_mask_available and sam3_response.get("error"):
                            trace(
                                "frame:{}:sam3_reseed_error:{}".format(
                                    frame_idx,
                                    sam3_response.get("error"),
                                )
                            )
                        pred_mask = _load_binary_mask(sam3_response.get("mask_path"))
                        if target_mask is not None and pred_mask is not None:
                            mask_iou_value = float(mask_iou(target_mask, pred_mask))
                        if sam3_mask_available and pred_mask is not None:
                            reseed_mask_bbox = tuple(int(v) for v in sam3_response["mask_bbox_xyxy"])
                            detector_response = detector_worker.infer(
                                frame_idx,
                                packet.lidar_points,
                                pred_mask,
                                reseed_mask_bbox,
                                scenario.sensors.lidar_to_camera_matrix(),
                            )
                            if detector_response.get("latency_ms") is not None:
                                pose_latency_ms = float(pose_latency_ms) + float(detector_response["latency_ms"])
                            detector_pose_available = bool(detector_response.get("pose_available", False))
                            if detector_pose_available:
                                used_pose = {
                                    "dx_m": float(detector_response["dx_m"]),
                                    "dy_m": float(detector_response["dy_m"]),
                                    "yaw_deg": float(
                                        canonicalize_follow_yaw_deg(float(detector_response["yaw_deg"]))
                                    ),
                                }
                                pose_available = True
                                pose_stale = False
                                stale_frames = 0
                                last_pose = dict(used_pose)
                                last_prompt_bbox = tuple(int(v) for v in detector_response["projection_bbox_xyxy"])
                                trace("frame:{}:bbox_reseed_success".format(frame_idx))

                    if not detector_pose_available:
                        stale_frames += 1
                        if last_pose is not None and stale_frames <= int(config.max_pose_hold_frames):
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

            scenario.apply_control(control.throttle, control.steer, control.brake)
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
            trace("frame:{}:done".format(frame_idx))

            if collision_events > 0:
                completion_reason = "collision"
                break
            if offroad_streak >= int(config.ego_offroad_breach_frames):
                completion_reason = "ego_left_driving_lane"
                break
            if invisible_streak >= int(config.target_out_of_view_breach_frames):
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
        if detector_worker is not None:
            detector_worker.stop()
        if sam3_worker is not None:
            sam3_worker.stop()
        scenario.cleanup()
        trace("cleanup:done")

    trace("metrics:write:start")
    summary_path = metrics.write(completion_reason)
    spectator_video_path = _build_spectator_video(config)
    _write_artifact_summary(
        summary_path,
        spectator_video_path=spectator_video_path,
        spectator_frames_dir=config.spectator_frames_dir if os.path.isdir(config.spectator_frames_dir) else None,
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh MPC pursuit evaluation")
    parser.add_argument("--pose-source", default="gt", choices=("gt", "detector"))
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
    parser.add_argument("--sam3-worker-env", default="ravp")
    parser.add_argument("--detector-worker-env", default="ravp-det")
    parser.add_argument("--sam3-device", default="cuda:1")
    parser.add_argument("--detector-device", default="cuda:2")
    parser.add_argument("--save-debug-images", action="store_true")
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
        sam3_worker_env=args.sam3_worker_env,
        detector_worker_env=args.detector_worker_env,
        sam3_device=args.sam3_device,
        detector_device=args.detector_device,
        save_debug_images=args.save_debug_images,
    )
    report_path = run_pursuit(config)
    print(report_path)


if __name__ == "__main__":
    main()
