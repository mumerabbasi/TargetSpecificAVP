"""SAM3-based vehicle mask generation for offline dataset building."""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .utils import binary_mask_to_bbox, deduplicate_mask_candidates


def _ensure_repo_on_path(repo_path: str) -> None:
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


class VisionDetector:
    """Generate per-vehicle masks from an RGB frame using SAM3."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str = "",
        prompt: str = "car",
        fallback_prompt: str = "vehicle",
        confidence_threshold: float = 0.35,
        duplicate_iou_thr: float = 0.75,
        device: str = "cuda:0",
    ) -> None:
        _ensure_repo_on_path(repo_path)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError(
                "sam3 must be importable in the offline build environment."
            ) from exc

        print("Loading SAM3 image model...")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path or None,
            load_from_HF=not bool(checkpoint_path),
            device=device,
        )
        model = model.to(device)
        model.eval()
        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.prompt = prompt
        self.fallback_prompt = fallback_prompt
        self.duplicate_iou_thr = duplicate_iou_thr

    def set_image(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Cache backbone features for an RGB frame."""
        # Sam3Processor expects PIL images or CHW tensors; passing an HWC numpy
        # array makes it misread the width as the channel count.
        pil_image = Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")
        return self.processor.set_image(pil_image)

    def _results_to_candidates(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a SAM3 processor state into deduplicated mask candidates."""
        if len(results["scores"]) == 0:
            return []

        masks = results["masks"].detach().cpu().numpy()
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()

        candidates: List[Dict[str, Any]] = []
        for mask, box, score in zip(masks, boxes, scores):
            binary_mask = mask.squeeze(0).astype(bool)
            bbox = binary_mask_to_bbox(binary_mask)
            if bbox is None:
                continue
            candidates.append(
                {
                    "mask": binary_mask,
                    "bbox": np.asarray(box, dtype=np.float32),
                    "bbox_from_mask": bbox,
                    "score": float(score),
                }
            )

        return deduplicate_mask_candidates(candidates, self.duplicate_iou_thr)

    def detect_and_segment(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Return SAM3 masks, bboxes, and confidence scores for vehicles."""
        state = self.set_image(rgb_image)
        results = self.processor.set_text_prompt(self.prompt, state)

        if len(results["scores"]) == 0 and self.fallback_prompt:
            self.processor.reset_all_prompts(state)
            results = self.processor.set_text_prompt(self.fallback_prompt, state)

        return self._results_to_candidates(results)

    def segment_from_box(
        self,
        state: Dict[str, Any],
        bbox_xyxy: tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """Return SAM3 masks conditioned on a target-specific 2D box prompt."""
        img_w = float(state["original_width"])
        img_h = float(state["original_height"])
        x1, y1, x2, y2 = bbox_xyxy
        cx = ((x1 + x2) * 0.5) / img_w
        cy = ((y1 + y2) * 0.5) / img_h
        width = max(float(x2 - x1), 1.0) / img_w
        height = max(float(y2 - y1), 1.0) / img_h

        self.processor.reset_all_prompts(state)
        if self.prompt:
            results = self.processor.set_text_prompt(self.prompt, state)
            if len(results["scores"]) == 0 and self.fallback_prompt:
                self.processor.reset_all_prompts(state)
                self.processor.set_text_prompt(self.fallback_prompt, state)
        results = self.processor.add_geometric_prompt(
            [cx, cy, width, height],
            True,
            state,
        )
        return self._results_to_candidates(results)
