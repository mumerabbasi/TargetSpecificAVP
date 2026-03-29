"""Stage B: build per-target GT and predicted datasets from raw captures."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .ground_truth import (
    actor_is_follow_valid,
    canonicalize_follow_yaw_deg,
    match_detections_to_actor_records,
)
from .utils import (
    bbox_touches_edge,
    binary_mask_to_bbox,
    ensure_dir,
    extract_instance_ids,
    extract_semantic_tags,
    link_or_copy,
    mask_iou,
    relative_path,
    save_binary_mask,
    vehicle_instance_mask_from_array,
)
from .vision_detector import VisionDetector


CSV_FIELDNAMES = [
    "sample_id",
    "frame_id",
    "episode_id",
    "town",
    "tick",
    "actor_id",
    "rgb_path",
    "mask_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "mask_area_px",
    "dx_m",
    "dy_m",
    "dz_m",
    "yaw_deg",
    "yaw_follow_deg",
    "follow_valid",
    "pose_score",
]


def _absolute_value_bin(value: float, bin_edges: Iterable[float]) -> int:
    bins = list(bin_edges)
    value = abs(float(value))
    if len(bins) < 2:
        return 0
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]
        upper_cmp = value <= upper if idx == len(bins) - 2 else value < upper
        if lower <= value and upper_cmp:
            return idx
    return len(bins) - 2


def _sorted_metadata_files(metadata_dir: str) -> List[str]:
    return sorted(
        os.path.join(metadata_dir, name)
        for name in os.listdir(metadata_dir)
        if name.endswith(".json")
    )


def _wipe_final_outputs(config: Config) -> None:
    for path in (config.final_rgb_dir, config.final_masks_dir, config.benchmark_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
    for path in (config.gt_csv_path, config.pred_csv_path):
        if os.path.exists(path):
            os.remove(path)


def _prepare_final_outputs(config: Config) -> None:
    _wipe_final_outputs(config)
    for path in config.final_dirs:
        ensure_dir(path)


def _prepare_pred_output(config: Config) -> None:
    ensure_dir(config.benchmark_dir)
    if os.path.exists(config.pred_csv_path):
        os.remove(config.pred_csv_path)


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _valid_mask_candidate(mask: np.ndarray, config: Config) -> Optional[Tuple[int, int, int, int]]:
    bbox = binary_mask_to_bbox(mask)
    if bbox is None:
        return None
    if int(mask.sum()) < config.min_mask_area_px:
        return None
    if bbox_touches_edge(
        bbox,
        config.image_width,
        config.image_height,
        config.edge_margin_px,
    ):
        return None
    return bbox


def _actor_masks_from_frame(
    actor_records: Iterable[Mapping[str, object]],
    instance_image: np.ndarray,
    config: Config,
) -> Tuple[Dict[int, Mapping[str, object]], Dict[int, np.ndarray]]:
    records_by_id: Dict[int, Mapping[str, object]] = {}
    masks_by_id: Dict[int, np.ndarray] = {}
    semantic_tags = extract_semantic_tags(instance_image)
    instance_ids = extract_instance_ids(instance_image)

    for actor in actor_records:
        actor_id = int(actor["actor_id"])
        instance_id = int(actor["instance_id"])
        mask = (semantic_tags == config.vehicle_semantic_tag) & (
            instance_ids == instance_id
        )
        if not np.any(mask):
            continue
        records_by_id[actor_id] = actor
        masks_by_id[actor_id] = mask

    return records_by_id, masks_by_id


def _match_masks_to_actors(
    candidates: List[Mapping[str, object]],
    actor_masks: Mapping[int, np.ndarray],
    config: Config,
) -> Dict[int, Mapping[str, object]]:
    pairs: List[tuple[float, float, int, int]] = []

    for cand_idx, candidate in enumerate(candidates):
        for actor_id, actor_mask in actor_masks.items():
            iou = mask_iou(candidate["mask"], actor_mask)
            if iou >= config.sam3_actor_iou_thr:
                pairs.append((iou, float(candidate["score"]), cand_idx, actor_id))

    pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)

    matched_candidates = set()
    matched_actors = set()
    matches: Dict[int, Mapping[str, object]] = {}
    for _, _, cand_idx, actor_id in pairs:
        if cand_idx in matched_candidates or actor_id in matched_actors:
            continue
        matches[actor_id] = candidates[cand_idx]
        matched_candidates.add(cand_idx)
        matched_actors.add(actor_id)

    return matches


def _prompt_actor_masks(
    mask_generator: VisionDetector,
    rgb: np.ndarray,
    actor_records: Mapping[int, Mapping[str, object]],
    actor_masks: Mapping[int, np.ndarray],
    config: Config,
) -> Dict[int, Mapping[str, object]]:
    state = mask_generator.set_image(rgb)
    prompted_masks: Dict[int, Mapping[str, object]] = {}

    for actor_id, actor in actor_records.items():
        prompt_bbox = (
            int(actor["bbox_x1"]),
            int(actor["bbox_y1"]),
            int(actor["bbox_x2"]),
            int(actor["bbox_y2"]),
        )
        best_candidate: Optional[Dict[str, object]] = None
        best_iou = 0.0

        for candidate in mask_generator.segment_from_box(state, prompt_bbox):
            bbox = _valid_mask_candidate(candidate["mask"], config)
            if bbox is None:
                continue
            iou = mask_iou(candidate["mask"], actor_masks[actor_id])
            if iou > best_iou:
                best_iou = iou
                best_candidate = {
                    "mask": candidate["mask"],
                    "bbox": bbox,
                    "score": float(candidate["score"]),
                }

        if best_candidate is not None and best_iou >= config.sam3_actor_iou_thr:
            prompted_masks[actor_id] = best_candidate

    return prompted_masks


def _copy_rgb_if_needed(
    raw_rgb_path: str,
    frame_id: int,
    config: Config,
    copied_frames: set[int],
) -> str:
    dst = os.path.join(config.final_rgb_dir, f"frame_{frame_id:06d}.png")
    if frame_id not in copied_frames:
        link_or_copy(raw_rgb_path, dst)
        copied_frames.add(frame_id)
    return dst


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_groups(csv_path: str) -> Dict[int, List[Dict[str, str]]]:
    groups: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    if not os.path.exists(csv_path):
        return groups
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups[int(row["frame_id"])].append(dict(row))
    return groups


def _build_row(
    *,
    sample_id: str,
    meta: Mapping[str, object],
    actor_id: int,
    rgb_path: str,
    mask_path: str,
    bbox: Tuple[int, int, int, int],
    mask_area_px: int,
    pose: Mapping[str, float],
    yaw_follow_deg: float,
    follow_valid: bool,
    pose_score: float,
    output_root: str,
) -> Dict[str, object]:
    return {
        "sample_id": sample_id,
        "frame_id": int(meta["frame_id"]),
        "episode_id": int(meta["episode_id"]),
        "town": str(meta["town"]),
        "tick": int(meta["tick"]),
        "actor_id": actor_id,
        "rgb_path": relative_path(rgb_path, output_root),
        "mask_path": relative_path(mask_path, output_root),
        "bbox_x1": int(bbox[0]),
        "bbox_y1": int(bbox[1]),
        "bbox_x2": int(bbox[2]),
        "bbox_y2": int(bbox[3]),
        "mask_area_px": int(mask_area_px),
        "dx_m": float(pose["dx_m"]),
        "dy_m": float(pose["dy_m"]),
        "dz_m": float(pose["dz_m"]),
        "yaw_deg": float(pose["yaw_deg"]),
        "yaw_follow_deg": float(yaw_follow_deg),
        "follow_valid": int(bool(follow_valid)),
        "pose_score": float(pose_score),
    }


def build_gt_dataset(config: Config) -> None:
    """Build shared RGB/masks plus GT pose rows from reusable raw captures."""
    _prepare_final_outputs(config)
    metadata_files = _sorted_metadata_files(config.raw_metadata_dir)

    if not metadata_files:
        raise FileNotFoundError(
            f"No raw metadata found in {config.raw_metadata_dir}. Run capture first."
        )

    mask_generator = VisionDetector(
        repo_path=config.sam3_repo_path,
        checkpoint_path=config.sam3_checkpoint_path,
        prompt=config.sam3_prompt,
        fallback_prompt=config.sam3_fallback_prompt,
        confidence_threshold=config.sam3_confidence_threshold,
        duplicate_iou_thr=config.sam3_duplicate_iou_thr,
        device=config.sam3_device,
    )

    gt_rows: List[Dict[str, object]] = []
    copied_frames: set[int] = set()
    coverage = defaultdict(int)
    lateral_coverage = defaultdict(int)
    yaw_coverage = defaultdict(int)
    samples_per_frame = defaultdict(int)
    processed_frames = 0

    for meta_path in metadata_files:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        raw_rgb_path = os.path.join(config.output_dir, meta["rgb_path"])
        raw_instance_path = os.path.join(config.output_dir, meta["instance_path"])

        rgb = _load_rgb(raw_rgb_path)
        instance_image = np.load(raw_instance_path)

        visible_actors = list(meta.get("visible_actors", []))
        if config.follow_only:
            visible_actors = [
                actor for actor in visible_actors if actor_is_follow_valid(actor, config)
            ]
            if len(visible_actors) < config.min_follow_actors_per_frame:
                continue
            if (
                config.max_follow_actors_per_frame > 0
                and len(visible_actors) > config.max_follow_actors_per_frame
            ):
                continue

        actor_records, actor_masks = _actor_masks_from_frame(
            visible_actors,
            instance_image,
            config,
        )
        if not actor_records:
            continue

        matched_masks = _prompt_actor_masks(
            mask_generator,
            rgb,
            actor_records,
            actor_masks,
            config,
        )
        if not matched_masks:
            continue

        frame_id = int(meta["frame_id"])
        rgb_out_path = _copy_rgb_if_needed(raw_rgb_path, frame_id, config, copied_frames)

        for actor_id, candidate in matched_masks.items():
            actor = actor_records[actor_id]
            sample_id = f"{frame_id:06d}_{actor_id}"
            mask_out_path = os.path.join(
                config.final_masks_dir,
                f"frame_{frame_id:06d}_actor_{actor_id}.png",
            )
            save_binary_mask(candidate["mask"], mask_out_path)

            gt_row = _build_row(
                sample_id=sample_id,
                meta=meta,
                actor_id=actor_id,
                rgb_path=rgb_out_path,
                mask_path=mask_out_path,
                bbox=candidate["bbox"],
                mask_area_px=int(candidate["mask"].sum()),
                pose={
                    "dx_m": float(actor["dx_m"]),
                    "dy_m": float(actor["dy_m"]),
                    "dz_m": float(actor["dz_m"]),
                    "yaw_deg": float(actor["yaw_deg"]),
                },
                yaw_follow_deg=canonicalize_follow_yaw_deg(float(actor["yaw_deg"])),
                follow_valid=actor_is_follow_valid(actor, config),
                pose_score=1.0,
                output_root=config.output_dir,
            )
            gt_rows.append(gt_row)
            coverage[(str(meta["town"]), int(actor["distance_bin"]))] += 1
            lateral_bin = _absolute_value_bin(
                float(actor["dy_m"]),
                config.lateral_bins_m,
            )
            yaw_bin = _absolute_value_bin(
                float(actor["yaw_deg"]),
                config.yaw_bins_deg,
            )
            lateral_coverage[(str(meta["town"]), lateral_bin)] += 1
            yaw_coverage[(str(meta["town"]), yaw_bin)] += 1
            samples_per_frame[frame_id] += 1

        processed_frames += 1
        if processed_frames % 25 == 0:
            print(
                f"[build] processed {processed_frames} frames, "
                f"{len(gt_rows)} GT samples"
            )

    _write_csv(config.gt_csv_path, gt_rows)

    if config.save_reports:
        report = {
            "processed_frames": processed_frames,
            "gt_samples": len(gt_rows),
            "pred_samples": 0,
            "max_targets_in_frame": int(max(samples_per_frame.values(), default=0)),
            "avg_targets_per_frame": float(
                len(gt_rows) / max(len(samples_per_frame), 1)
            ),
            "coverage_by_town_and_distance_bin": {
                f"{town}|bin_{bin_idx}": count
                for (town, bin_idx), count in sorted(coverage.items())
            },
            "coverage_by_town_and_lateral_bin": {
                f"{town}|bin_{bin_idx}": count
                for (town, bin_idx), count in sorted(lateral_coverage.items())
            },
            "coverage_by_town_and_yaw_bin": {
                f"{town}|bin_{bin_idx}": count
                for (town, bin_idx), count in sorted(yaw_coverage.items())
            },
        }
        with open(os.path.join(config.benchmark_dir, "build_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    print(
        f"[build-gt] done: {processed_frames} frames, "
        f"{len(gt_rows)} GT rows"
    )


def attach_predicted_poses(config: Config) -> None:
    """Attach detector-derived pose rows to an existing GT dataset."""
    from .detector_3d import DetectorSpec, MMDet3DDetector

    if not os.path.exists(config.gt_csv_path):
        raise FileNotFoundError(
            f"{config.gt_csv_path} not found. Build the GT dataset first."
        )

    _prepare_pred_output(config)
    gt_groups = _load_csv_groups(config.gt_csv_path)
    if not gt_groups:
        raise RuntimeError("GT dataset is empty. No target samples to label.")

    detector = MMDet3DDetector(
        DetectorSpec(
            name=config.detector_name,
            config_path=config.detector_config,
            checkpoint_path=config.detector_checkpoint,
            score_thr=config.detector_score_thr,
            device=config.detector_device,
        )
    )

    pred_rows: List[Dict[str, object]] = []
    processed_frames = 0

    for frame_id, gt_rows in sorted(gt_groups.items()):
        meta_path = os.path.join(
            config.raw_metadata_dir,
            f"frame_{frame_id:06d}.json",
        )
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        actor_ids = {int(row["actor_id"]) for row in gt_rows}
        actor_records = [
            actor
            for actor in meta.get("visible_actors", [])
            if int(actor["actor_id"]) in actor_ids
        ]
        if not actor_records:
            continue

        raw_lidar_path = os.path.join(config.output_dir, meta["lidar_path"])
        lidar = np.load(raw_lidar_path)
        detector_matches = match_detections_to_actor_records(
            detector.detect(lidar),
            actor_records,
            config.detector_match_dist_m,
        )

        for gt_row in gt_rows:
            actor_id = int(gt_row["actor_id"])
            det = detector_matches.get(actor_id)
            if det is None:
                continue

            pred_row = dict(gt_row)
            pred_row["dx_m"] = float(det["center"][0])
            pred_row["dy_m"] = float(det["center"][1])
            pred_row["dz_m"] = float(det["center"][2])
            pred_row["yaw_deg"] = float(det["yaw_deg"])
            pred_row["yaw_follow_deg"] = canonicalize_follow_yaw_deg(
                float(det["yaw_deg"])
            )
            pred_row["pose_score"] = float(det["score"])
            pred_rows.append(pred_row)

        processed_frames += 1
        if processed_frames % 25 == 0:
            print(
                f"[attach-pred] processed {processed_frames} frames, "
                f"{len(pred_rows)} predicted rows"
            )

    _write_csv(config.pred_csv_path, pred_rows)

    if config.save_reports:
        report_path = os.path.join(config.benchmark_dir, "build_report.json")
        report: Dict[str, object] = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
        report["pred_samples"] = len(pred_rows)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    print(
        f"[attach-pred] done: {processed_frames} frames, "
        f"{len(pred_rows)} predicted rows"
    )
