"""3D detection with CenterPoint."""

from typing import Any, Dict, List

import numpy as np
from mmdet3d.apis import inference_detector, init_model

from .utils import carla_to_nuscenes_points, nuscenes_to_carla_box


def load_centerpoint_model(
    config_file: str,
    checkpoint_file: str,
    device: str = "cuda:0",
) -> Any:
    """
    Load CenterPoint model.

    Args:
        config_file: Path to model config.
        checkpoint_file: Path to model checkpoint.
        device: Device for inference.

    Returns:
        Initialized model.
    """
    print("Initializing CenterPoint model...")
    return init_model(config_file, checkpoint_file, device=device)


def extract_data_sample(det_output: Any) -> Any:
    """Extract Det3DDataSample from inference output."""
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


def run_centerpoint_detection(
    model: Any,
    points_carla: np.ndarray,
    score_thr: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Run CenterPoint on filtered CARLA LiDAR points.

    Args:
        model: CenterPoint model.
        points_carla: Nx4 CARLA LiDAR points.
        score_thr: Score threshold for detections.

    Returns:
        List of detections with center, dims, yaw, score.
    """
    if len(points_carla) < 10:
        return []

    points_nus = carla_to_nuscenes_points(points_carla)
    det_output = inference_detector(model, points_nus)
    det_sample = extract_data_sample(det_output)

    pred = det_sample.pred_instances_3d
    bboxes_3d = pred.bboxes_3d.tensor.cpu().numpy()
    scores_3d = pred.scores_3d.cpu().numpy()
    labels_3d = pred.labels_3d.cpu().numpy()

    # Filter by class (car)
    class_names = getattr(model, "dataset_meta", {}).get("classes", None)
    if class_names is not None:
        car_ids = [i for i, name in enumerate(class_names) if name.lower() == "car"]
        if car_ids:
            mask_cls = np.isin(labels_3d, car_ids)
            bboxes_3d = bboxes_3d[mask_cls]
            scores_3d = scores_3d[mask_cls]
            labels_3d = labels_3d[mask_cls]

    # Filter by score
    mask_score = scores_3d >= score_thr
    bboxes_3d = bboxes_3d[mask_score]
    scores_3d = scores_3d[mask_score]

    detections = []
    for box, score in zip(bboxes_3d, scores_3d):
        box_carla = nuscenes_to_carla_box(box)
        box_carla["score"] = float(score)
        detections.append(box_carla)

    return detections
