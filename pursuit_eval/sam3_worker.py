"""Persistent SAM3 online tracker worker with bbox reseed recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import PursuitEvalConfig
from .geometry import bbox_from_mask, bbox_iou


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


class OnlineSam3Tracker:
    """Simple online SAM3 tracker: bootstrap, track, reseed."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        if config.sam3_repo_path and config.sam3_repo_path not in sys.path:
            sys.path.insert(0, config.sam3_repo_path)

        from sam3.model.utils.sam1_utils import SAM2Transforms
        from sam3.model_builder import build_tracker, download_ckpt_from_hf

        self.config = config
        self.device = _resolve_device(config.sam3_device)
        self.object_id = 1
        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))

        model = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
        checkpoint_path = config.sam3_checkpoint_path or download_ckpt_from_hf()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        tracker_state = {
            key.replace("tracker.", "", 1): value
            for key, value in checkpoint.items()
            if key.startswith("tracker.")
        }
        tracker_state.update(
            {
                key.replace("detector.backbone.", "backbone.", 1): value
                for key, value in checkpoint.items()
                if key.startswith("detector.backbone.")
            }
        )
        missing_keys, unexpected_keys = model.load_state_dict(tracker_state, strict=False)
        if missing_keys or unexpected_keys:
            print(
                "sam3 tracker load diagnostics: missing_keys={}, unexpected_keys={}".format(
                    len(missing_keys),
                    len(unexpected_keys),
                )
            )

        self.model = model.to(self.device).eval()
        self.transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.inference_state = None
        self.propagation_iterator = None
        self.last_output_frame_idx = -1

    def _ensure_state(self, frame_idx: int) -> None:
        if self.inference_state is not None:
            return
        num_frames = max(int(self.config.num_frames), int(frame_idx) + 2)
        self.inference_state = self.model.init_state(
            video_height=int(self.config.image_height),
            video_width=int(self.config.image_width),
            num_frames=num_frames,
            cached_features={},
            offload_state_to_cpu=True,
        )

    def _cache_frame_features(self, frame_idx: int, rgb_image: np.ndarray) -> None:
        input_image = self.transforms(rgb_image)[None, ...].to(self.device)
        backbone_out = self.model.forward_image(input_image)
        # The online tracker only needs the current frame's backbone features.
        # Keeping every frame here quickly blows up GPU memory.
        self.inference_state["cached_features"] = {
            int(frame_idx): (input_image, backbone_out),
        }

    def _restart_stream(self, start_frame_idx: int) -> None:
        if self.propagation_iterator is not None:
            self.propagation_iterator.close()
        max_frame_num_to_track = max(int(self.config.num_frames) - int(start_frame_idx), 0)
        self.propagation_iterator = self.model.propagate_in_video(
            self.inference_state,
            start_frame_idx=int(start_frame_idx),
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=True,
        )

    def _next_output(self, expected_frame_idx: int):
        if self.propagation_iterator is None:
            raise RuntimeError("SAM3 tracker stream is not initialized.")
        frame_idx, obj_ids, _, video_res_masks, obj_scores = next(self.propagation_iterator)
        if int(frame_idx) != int(expected_frame_idx):
            raise RuntimeError(
                "SAM3 tracker yielded frame {} while frame {} was requested".format(
                    frame_idx,
                    expected_frame_idx,
                )
            )
        self.last_output_frame_idx = int(frame_idx)
        return obj_ids, video_res_masks, obj_scores

    @staticmethod
    def _largest_component(mask: np.ndarray, hint_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask
        best_label = 0
        best_score = -1.0
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = float(stats[label, cv2.CC_STAT_AREA])
            bbox = (x, y, x + w - 1, y + h - 1)
            score = area
            if hint_bbox is not None:
                score += 5000.0 * bbox_iou(bbox, hint_bbox)
            if score > best_score:
                best_score = score
                best_label = label
        if best_label <= 0:
            return mask
        return labels == best_label

    def _decode_mask(
        self,
        obj_ids,
        video_res_masks,
        hint_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        if obj_ids is None or video_res_masks is None:
            return None, None, None
        obj_ids = [int(value) for value in obj_ids]
        try:
            obj_index = obj_ids.index(self.object_id)
        except ValueError:
            return None, None, None

        mask_logits = video_res_masks[obj_index]
        if hasattr(mask_logits, "detach"):
            mask_logits = mask_logits.detach().float().cpu().numpy()
        mask_logits = np.asarray(mask_logits)
        while mask_logits.ndim > 2:
            mask_logits = mask_logits[0]
        if mask_logits.size == 0 or not np.isfinite(mask_logits).any():
            return None, None, None

        logit_max = float(np.max(mask_logits))
        threshold = 0.0
        if np.any(mask_logits > threshold):
            mask = mask_logits > threshold
        else:
            if logit_max < -6.0:
                return None, logit_max, threshold
            threshold = logit_max - 0.5
            mask = mask_logits > threshold
        if not np.any(mask):
            return None, logit_max, threshold
        mask = self._largest_component(mask.astype(bool), hint_bbox).astype(bool)
        if not np.any(mask):
            return None, logit_max, threshold
        return mask, logit_max, threshold

    def track(
        self,
        frame_idx: int,
        rgb_image: np.ndarray,
        bootstrap_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
        reseed_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, object]:
        self._ensure_state(frame_idx)
        self._cache_frame_features(frame_idx, rgb_image)

        bootstrap_used = bool(frame_idx == 0 and bootstrap_bbox_xyxy is not None)
        reseed_used = bool(reseed_bbox_xyxy is not None)
        hint_bbox = reseed_bbox_xyxy if reseed_bbox_xyxy is not None else bootstrap_bbox_xyxy

        start = time.time()
        if frame_idx == 0:
            if bootstrap_bbox_xyxy is None:
                raise ValueError("Frame 0 requires bootstrap_bbox_xyxy.")
            self.model.add_new_points_or_box(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=self.object_id,
                box=np.asarray(bootstrap_bbox_xyxy, dtype=np.float32),
                clear_old_points=True,
                rel_coordinates=False,
            )
            self._restart_stream(start_frame_idx=int(frame_idx))
        elif reseed_bbox_xyxy is not None:
            self.model.add_new_points_or_box(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=self.object_id,
                box=np.asarray(reseed_bbox_xyxy, dtype=np.float32),
                clear_old_points=True,
                rel_coordinates=False,
            )
            self._restart_stream(start_frame_idx=int(frame_idx))
        else:
            if self.propagation_iterator is None:
                raise RuntimeError("SAM3 tracker was not bootstrapped.")
            if int(frame_idx) != self.last_output_frame_idx + 1:
                raise RuntimeError(
                    "SAM3 tracker expected frame {} but got {}".format(
                        self.last_output_frame_idx + 1,
                        frame_idx,
                    )
                )

        obj_ids, video_res_masks, obj_scores = self._next_output(frame_idx)
        mask, tracker_logit_max, threshold_used = self._decode_mask(obj_ids, video_res_masks, hint_bbox)
        latency_ms = (time.time() - start) * 1000.0
        if mask is None:
            return {
                "frame": int(frame_idx),
                "mask_available": False,
                "bbox_bootstrap_used": bootstrap_used,
                "bbox_reseed_used": reseed_used,
                "latency_ms": latency_ms,
                "tracker_logit_max": tracker_logit_max,
                "tracker_threshold": threshold_used,
                "error": "sam3 tracker returned no usable target mask",
            }

        bbox = bbox_from_mask(mask)
        if bbox is None:
            return {
                "frame": int(frame_idx),
                "mask_available": False,
                "bbox_bootstrap_used": bootstrap_used,
                "bbox_reseed_used": reseed_used,
                "latency_ms": latency_ms,
                "tracker_logit_max": tracker_logit_max,
                "tracker_threshold": threshold_used,
                "error": "sam3 tracker produced no valid bbox",
            }

        score_value = None
        if obj_scores is not None:
            try:
                obj_ids = [int(value) for value in obj_ids]
                obj_index = obj_ids.index(self.object_id)
                score_tensor = obj_scores[obj_index]
                if hasattr(score_tensor, "detach"):
                    score_tensor = score_tensor.detach().float().cpu().numpy()
                score_value = float(np.asarray(score_tensor).reshape(-1)[0])
            except Exception:
                score_value = None

        mask_path = os.path.join(
            self.config.sam3_masks_dir,
            "frame_{:06d}.png".format(int(frame_idx)),
        )
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
        return {
            "frame": int(frame_idx),
            "mask_available": True,
            "mask_bbox_xyxy": [int(v) for v in bbox],
            "mask_area_px": int(mask.sum()),
            "mask_score": score_value,
            "mask_path": mask_path,
            "bbox_bootstrap_used": bootstrap_used,
            "bbox_reseed_used": reseed_used,
            "latency_ms": latency_ms,
            "tracker_logit_max": tracker_logit_max,
            "tracker_threshold": threshold_used,
        }


class Sam3Worker:
    """Long-lived file-backed SAM3 worker."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        self.config = config
        self.tracker = OnlineSam3Tracker(config)

    def run(self) -> None:
        for path in (
            self.config.sam3_worker_dir,
            self.config.sam3_requests_dir,
            self.config.sam3_responses_dir,
            self.config.sam3_assets_dir,
            self.config.sam3_masks_dir,
        ):
            _ensure_dir(path)
        ready_path = os.path.join(self.config.sam3_worker_dir, "ready.json")
        with open(ready_path, "w") as handle:
            json.dump({"ready": True, "pid": os.getpid()}, handle)

        while True:
            request_files = sorted(
                name for name in os.listdir(self.config.sam3_requests_dir) if name.endswith(".json")
            )
            if not request_files:
                time.sleep(float(self.config.worker_poll_interval_s))
                continue

            for name in request_files:
                request_path = os.path.join(self.config.sam3_requests_dir, name)
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
                    self.config.sam3_responses_dir,
                    "frame_{:06d}.json".format(int(request["frame"])),
                )
                tmp_response_path = response_path + ".tmp"
                with open(tmp_response_path, "w") as handle:
                    json.dump(response, handle, indent=2)
                os.replace(tmp_response_path, response_path)
                os.remove(request_path)

    def _handle_request(self, request: Dict[str, object]) -> Dict[str, object]:
        frame_idx = int(request["frame"])
        rgb_path = str(request["rgb_path"])
        bootstrap_bbox = request.get("bootstrap_bbox_xyxy")
        reseed_bbox = request.get("reseed_bbox_xyxy")
        reseed_reason = str(request.get("reseed_reason", ""))
        try:
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError("Could not read RGB frame {}".format(rgb_path))
            rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            response = self.tracker.track(
                frame_idx=frame_idx,
                rgb_image=rgb_image,
                bootstrap_bbox_xyxy=None
                if bootstrap_bbox is None
                else tuple(int(v) for v in bootstrap_bbox),
                reseed_bbox_xyxy=None
                if reseed_bbox is None
                else tuple(int(v) for v in reseed_bbox),
            )
            response["bbox_reseed_reason"] = reseed_reason
            return response
        except Exception as exc:
            return {
                "frame": frame_idx,
                "mask_available": False,
                "bbox_bootstrap_used": bool(frame_idx == 0 and bootstrap_bbox is not None),
                "bbox_reseed_used": bool(reseed_bbox is not None),
                "bbox_reseed_reason": reseed_reason,
                "latency_ms": None,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        finally:
            if not self.config.keep_worker_frame_assets and os.path.exists(rgb_path):
                os.remove(rgb_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 worker")
    parser.add_argument("--config", required=True, help="Path to pursuit_eval config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as handle:
        payload = json.load(handle)
    config = PursuitEvalConfig(**payload)
    Sam3Worker(config).run()


if __name__ == "__main__":
    main()
