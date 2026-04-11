"""MMDetection3D inference wrapper for per-frame pose labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from .utils import carla_to_nuscenes_points, nuscenes_to_carla_box


@dataclass
class DetectorSpec:
    """Configuration for an MMDetection3D LiDAR detector."""

    name: str
    config_path: str
    checkpoint_path: str
    score_thr: float = 0.15
    device: str = "cuda:0"


def extract_data_sample(det_output: Any) -> Any:
    """Extract a Det3DDataSample from varying MMDet3D output shapes."""
    if hasattr(det_output, "pred_instances_3d"):
        return det_output
    if isinstance(det_output, list):
        for item in det_output:
            if hasattr(item, "pred_instances_3d"):
                return item
    if isinstance(det_output, tuple):
        for item in det_output:
            if isinstance(item, list):
                for sub_item in item:
                    if hasattr(sub_item, "pred_instances_3d"):
                        return sub_item
            if hasattr(item, "pred_instances_3d"):
                return item
    raise RuntimeError("Could not find Det3DDataSample in inference output.")


class MMDet3DDetector:
    """Run full-frame LiDAR detection with a configurable MMDet3D model."""

    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec
        try:
            from mmdet3d.apis import init_model
        except ImportError as exc:
            raise ImportError(
                "mmdet3d is required for dataset collection and reporting."
            ) from exc

        self.device = spec.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[detector] CUDA unavailable, falling back to CPU")
            self.device = "cpu"

        print(f"Initializing {spec.name} from {spec.config_path}")
        self._inference_detector = self._load_inference_detector()
        self.model = init_model(
            spec.config_path,
            spec.checkpoint_path,
            device=self.device,
        )

    def _load_inference_detector(self):
        from mmdet3d.apis import inference_detector

        return inference_detector

    def detect(self, points_carla: np.ndarray) -> List[Dict[str, Any]]:
        """Run the configured detector on the full frame point cloud."""
        if len(points_carla) < 10:
            return []

        points_nus = carla_to_nuscenes_points(points_carla)
        try:
            det_output = self._inference_detector(self.model, points_nus)
        except RuntimeError as exc:
            message = str(exc)
            if "not implemented on CPU" in message:
                raise RuntimeError(
                    "The configured 3D detector requires CUDA-visible GPUs for "
                    "inference. Make sure the detector env can see NVIDIA "
                    "devices."
                ) from exc
            raise
        det_sample = extract_data_sample(det_output)

        pred = det_sample.pred_instances_3d
        bboxes_3d = pred.bboxes_3d.tensor.cpu().numpy()
        scores_3d = pred.scores_3d.cpu().numpy()
        labels_3d = pred.labels_3d.cpu().numpy()

        class_names = getattr(
            self.model,
            "dataset_meta",
            {}).get(
            "classes",
            None)
        if class_names is not None:
            car_ids = [idx for idx, name in enumerate(
                class_names) if name.lower() == "car"]
            if car_ids:
                mask_cls = np.isin(labels_3d, car_ids)
                bboxes_3d = bboxes_3d[mask_cls]
                scores_3d = scores_3d[mask_cls]

        mask_score = scores_3d >= self.spec.score_thr
        bboxes_3d = bboxes_3d[mask_score]
        scores_3d = scores_3d[mask_score]

        detections: List[Dict[str, Any]] = []
        for box, score in zip(bboxes_3d, scores_3d):
            box_carla = nuscenes_to_carla_box(box)
            box_carla["score"] = float(score)
            detections.append(box_carla)

        return detections
