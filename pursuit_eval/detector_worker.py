"""Persistent 3D detector worker for target pose extraction from LiDAR."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .config import PursuitEvalConfig
from .geometry import (
    bbox_iou,
    get_camera_intrinsic,
    mask_iou,
    project_detection_box_to_image,
    wrap_angle_rad,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _carla_to_nuscenes_points(points_carla: np.ndarray) -> np.ndarray:
    if points_carla.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    points_nus = np.zeros((points_carla.shape[0], 5), dtype=np.float32)
    points_nus[:, 0] = points_carla[:, 0]
    points_nus[:, 1] = -points_carla[:, 1]
    points_nus[:, 2] = points_carla[:, 2]
    points_nus[:, 3] = np.clip(points_carla[:, 3], 0.0, 1.0)
    points_nus[:, 4] = 0.0
    return points_nus


def _nuscenes_to_carla_box(box_nus: np.ndarray) -> Dict[str, object]:
    center = np.array(
        [box_nus[0], -box_nus[1], box_nus[2] + (box_nus[5] * 0.5)],
        dtype=np.float32,
    )
    yaw = float(wrap_angle_rad(-float(box_nus[6])))
    return {
        "center": center,
        "dims": np.array([box_nus[3], box_nus[4], box_nus[5]], dtype=np.float32),
        "yaw_rad": yaw,
        "yaw_deg": float(np.degrees(yaw)),
    }


def _extract_data_sample(det_output):
    if hasattr(det_output, "pred_instances_3d"):
        return det_output
    if isinstance(det_output, (list, tuple)):
        for item in det_output:
            try:
                return _extract_data_sample(item)
            except RuntimeError:
                continue
    raise RuntimeError("Could not find Det3DDataSample in detector output.")


class MMDet3DPoseSource:
    """MMDetection3D detector wrapper for full-frame LiDAR inference."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        from mmdet3d.apis import inference_detector, init_model

        device = config.detector_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))

        self.inference_detector = inference_detector
        self.model = init_model(config.detector_config, config.detector_checkpoint, device=device)
        self.score_thr = float(config.detector_score_thr)

    def detect(self, lidar_points_carla: np.ndarray) -> List[Dict[str, object]]:
        if lidar_points_carla.size == 0 or lidar_points_carla.shape[0] < 10:
            return []

        pred = _extract_data_sample(
            self.inference_detector(self.model, _carla_to_nuscenes_points(lidar_points_carla))
        ).pred_instances_3d
        bboxes_3d = pred.bboxes_3d.tensor.detach().cpu().numpy()
        scores_3d = pred.scores_3d.detach().cpu().numpy()
        labels_3d = pred.labels_3d.detach().cpu().numpy()

        class_names = getattr(self.model, "dataset_meta", {}).get("classes", None)
        if class_names is not None:
            car_ids = [idx for idx, name in enumerate(class_names) if name.lower() == "car"]
            if car_ids:
                cls_mask = np.isin(labels_3d, car_ids)
                bboxes_3d = bboxes_3d[cls_mask]
                scores_3d = scores_3d[cls_mask]

        detections: List[Dict[str, object]] = []
        for box_nus, score in zip(bboxes_3d, scores_3d):
            if float(score) < self.score_thr:
                continue
            det = _nuscenes_to_carla_box(box_nus)
            det["score"] = float(score)
            detections.append(det)
        return detections


class DetectorWorker:
    """Long-lived file-backed detector worker."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        self.config = config
        self.detector = MMDet3DPoseSource(config)
        self.intrinsic = get_camera_intrinsic(config.image_width, config.image_height, config.fov)

    def run(self) -> None:
        for path in (
            self.config.detector_worker_dir,
            self.config.detector_requests_dir,
            self.config.detector_responses_dir,
            self.config.detector_assets_dir,
        ):
            _ensure_dir(path)
        ready_path = os.path.join(self.config.detector_worker_dir, "ready.json")
        with open(ready_path, "w") as handle:
            json.dump({"ready": True, "pid": os.getpid()}, handle)

        while True:
            request_files = sorted(
                name for name in os.listdir(self.config.detector_requests_dir) if name.endswith(".json")
            )
            if not request_files:
                time.sleep(float(self.config.worker_poll_interval_s))
                continue
            for name in request_files:
                request_path = os.path.join(self.config.detector_requests_dir, name)
                try:
                    with open(request_path, "r") as handle:
                        request = json.load(handle)
                except json.JSONDecodeError:
                    time.sleep(float(self.config.worker_poll_interval_s))
                    continue
                if request.get("type") == "stop":
                    os.remove(request_path)
                    return
                response = self._handle_request(request)
                response_path = os.path.join(
                    self.config.detector_responses_dir,
                    "frame_{:06d}.json".format(int(request["frame"])),
                )
                tmp_response_path = response_path + ".tmp"
                with open(tmp_response_path, "w") as handle:
                    json.dump(response, handle, indent=2)
                os.replace(tmp_response_path, response_path)
                os.remove(request_path)

    def _handle_request(self, request: Dict[str, object]) -> Dict[str, object]:
        frame_idx = int(request["frame"])
        lidar_path = str(request["lidar_path"])
        mask_path = str(request["mask_path"])
        lidar_to_camera = np.asarray(request["lidar_to_camera"], dtype=np.float64)
        try:
            lidar_points = np.load(lidar_path)
            target_mask = np.load(mask_path).astype(bool)
            start = time.time()
            detections = self.detector.detect(lidar_points)
            latency_ms = (time.time() - start) * 1000.0

            best_det = None
            best_score = 0.0
            for det in detections:
                proj_bbox, proj_mask = project_detection_box_to_image(
                    np.asarray(det["center"], dtype=np.float64),
                    np.asarray(det["dims"], dtype=np.float64),
                    float(det["yaw_rad"]),
                    lidar_to_camera,
                    self.intrinsic,
                    self.config.image_width,
                    self.config.image_height,
                )
                if proj_bbox is None or proj_mask is None:
                    continue
                overlap = max(mask_iou(target_mask, proj_mask), bbox_iou(tuple(request["mask_bbox_xyxy"]), proj_bbox))
                score = overlap + float(self.config.detector_projection_score_weight) * float(det["score"])
                if overlap >= float(self.config.detector_projection_iou_thr) and score > best_score:
                    best_score = score
                    best_det = {
                        "pose": det,
                        "projection_bbox": proj_bbox,
                        "projection_overlap": overlap,
                    }

            response = {
                "frame": frame_idx,
                "pose_available": best_det is not None,
                "latency_ms": latency_ms,
            }
            if best_det is not None:
                response.update(
                    {
                        "dx_m": float(best_det["pose"]["center"][0]),
                        "dy_m": float(best_det["pose"]["center"][1]),
                        "dz_m": float(best_det["pose"]["center"][2]),
                        "yaw_deg": float(best_det["pose"]["yaw_deg"]),
                        "pose_score": float(best_det["pose"]["score"]),
                        "projection_bbox_xyxy": [int(v) for v in best_det["projection_bbox"]],
                        "projection_overlap": float(best_det["projection_overlap"]),
                    }
                )
            else:
                response["error"] = "no detector box matched the target mask"
            return response
        except Exception as exc:
            return {
                "frame": frame_idx,
                "pose_available": False,
                "latency_ms": None,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        finally:
            if not self.config.keep_worker_frame_assets:
                for path in (lidar_path, mask_path):
                    if os.path.exists(path):
                        os.remove(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector worker")
    parser.add_argument("--config", required=True, help="Path to pursuit_eval config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as handle:
        payload = json.load(handle)
    config = PursuitEvalConfig(**payload)
    DetectorWorker(config).run()


if __name__ == "__main__":
    main()
