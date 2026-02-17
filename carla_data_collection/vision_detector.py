"""Vision models: YOLO for detection, SAM2 for segmentation."""

import sys
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO


def _ensure_sam2_path(sam2_path: str) -> None:
    """Add SAM2 to Python path if not already present."""
    if sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)


class VisionDetector:
    """YOLO + SAM2 based car detection and segmentation."""

    def __init__(
        self,
        yolo_path: str,
        sam2_checkpoint: str,
        sam2_config: str,
        sam2_path: str = "/usr/prakt/s0050/ravp/sam2",
        device: str = "cuda",
    ) -> None:
        """
        Initialize YOLO and SAM2 models.

        Args:
            yolo_path: Path to YOLO weights.
            sam2_checkpoint: Path to SAM2 checkpoint.
            sam2_config: SAM2 config name.
            sam2_path: Path to SAM2 repository.
            device: Device for inference.
        """
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_path)

        # Import SAM2 (requires path modification)
        _ensure_sam2_path(sam2_path)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        print("Loading SAM2 model...")
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2)

        # COCO class IDs for vehicles
        self.vehicle_classes = [2, 5, 7]  # car, bus, truck

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        conf_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect cars with YOLO and segment each with SAM2.

        Args:
            rgb_image: HxWx3 RGB image.
            conf_threshold: YOLO confidence threshold.

        Returns:
            List of detections, each with:
                - bbox: [x1, y1, x2, y2] bounding box
                - mask: HxW binary mask
                - conf: detection confidence
                - mask_score: SAM2 mask score
                - class_id: COCO class ID
        """
        # Run YOLO detection
        results = self.yolo(rgb_image, conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        detections = []

        # Filter for vehicle classes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            if class_id not in self.vehicle_classes:
                continue

            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].item())

            detections.append({
                "bbox": bbox,
                "conf": conf,
                "class_id": class_id,
            })

        if not detections:
            return []

        # Run SAM2 segmentation for each detected car
        self.sam2_predictor.set_image(rgb_image)

        for det in detections:
            bbox = det["bbox"]

            # Use bounding box as prompt for SAM2
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],  # SAM2 expects [1, 4]
                multimask_output=False,
            )

            # Take the best mask
            det["mask"] = masks[0].astype(bool)  # HxW boolean mask
            det["mask_score"] = float(scores[0])

        return detections
