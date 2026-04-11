"""Single-pass CARLA dataset collection for the RAVP training format."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import carla
import numpy as np

from .carla_utils import (
    SensorRig,
    collect_visible_vehicle_records,
    configure_traffic_manager,
    destroy_actors,
    ego_on_driving_lane,
    setup_world,
    spawn_background_traffic,
    spawn_ego_vehicle,
    vehicle_instance_mask,
)
from .config import Config
from .detector_3d import DetectorSpec, MMDet3DDetector
from .ground_truth import (
    actor_is_follow_valid,
    canonicalize_follow_yaw_deg,
    distance_bin_index,
    match_detections_to_actor_records,
)
from .utils import (
    bbox_touches_edge,
    binary_mask_to_bbox,
    ensure_dir,
    mask_iou,
    relative_path,
    save_binary_mask,
    save_rgb_jpeg,
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


@dataclass
class TargetSample:
    """One per-target training sample anchored to a shared RGB frame."""

    actor_record: Mapping[str, object]
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    pred_pose: Optional[Mapping[str, object]] = None

    @property
    def actor_id(self) -> int:
        return int(self.actor_record["actor_id"])

    @property
    def mask_area_px(self) -> int:
        return int(self.mask.astype(bool).sum())

    @property
    def follow_valid(self) -> bool:
        return bool(self.actor_record.get("follow_valid", False))


def _sample_id(frame_id: int, actor_id: int) -> str:
    return f"frame_{frame_id:06d}_actor_{int(actor_id)}"


def _mask_path_for_frame(config: Config, frame_id: int, actor_id: int) -> str:
    return os.path.join(
        config.masks_dir,
        f"frame_{frame_id:06d}_actor_{int(actor_id)}.png",
    )


def _rgb_path_for_frame(config: Config, frame_id: int) -> str:
    return os.path.join(config.rgb_dir, f"frame_{frame_id:06d}.jpg")


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writerows(rows)


def _prune_untracked_files(
    directory: str,
    valid_relative_paths: Iterable[str],
    root_dir: str,
) -> None:
    if not os.path.isdir(directory):
        return
    valid_paths = {
        os.path.normpath(os.path.join(root_dir, rel_path))
        for rel_path in valid_relative_paths
    }
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            continue
        if os.path.normpath(path) not in valid_paths:
            os.remove(path)


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


class DatasetWriter:
    """Append-only dataset writer with manifest-based resume."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if bool(config.fresh_start) and os.path.isdir(config.output_dir):
            shutil.rmtree(config.output_dir)
        for path in config.dataset_dirs:
            ensure_dir(path)

        self.manifest_records = _read_jsonl(config.frames_manifest_path)
        valid_sample_ids = {
            _sample_id(int(record["frame_id"]), int(actor_id))
            for record in self.manifest_records
            for actor_id in record.get("accepted_actor_ids", [])
        }

        gt_rows = self._reconcile_csv(config.gt_csv_path, valid_sample_ids)
        pred_rows = self._reconcile_csv(config.pred_csv_path, valid_sample_ids)

        valid_rgb_paths = [str(record["rgb_path"]) for record in self.manifest_records]
        valid_mask_paths = [str(row["mask_path"]) for row in gt_rows]
        _prune_untracked_files(config.rgb_dir, valid_rgb_paths, config.output_dir)
        _prune_untracked_files(config.masks_dir, valid_mask_paths, config.output_dir)

        self.next_frame_id = (
            max(
                (int(record["frame_id"]) for record in self.manifest_records),
                default=-1,
            )
            + 1
        )
        self.frames_per_town: Counter[str] = Counter(
            str(record["town"]) for record in self.manifest_records
        )
        self.gt_samples_per_town: Counter[str] = Counter()
        self.pred_samples_per_town: Counter[str] = Counter(
            str(row["town"]) for row in pred_rows
        )
        self.distance_bins_per_town: Dict[str, Counter[int]] = defaultdict(Counter)
        self.max_episode_id_per_town: Dict[str, int] = defaultdict(lambda: -1)
        self._town_order: List[str] = []
        for record in self.manifest_records:
            town = str(record["town"])
            self._remember_town(town)
            self.max_episode_id_per_town[town] = max(
                int(self.max_episode_id_per_town[town]),
                int(record["episode_id"]),
            )
        for row in gt_rows:
            town = str(row["town"])
            self._remember_town(town)
            self.gt_samples_per_town[town] += 1
            bin_idx = distance_bin_index(
                float(row["dx_m"]), self.config.distance_bins_m)
            self.distance_bins_per_town[town][bin_idx] += 1
        for row in pred_rows:
            self._remember_town(str(row["town"]))

        self.bytes_rgb = _directory_size(self.config.rgb_dir)
        self.bytes_masks = _directory_size(self.config.masks_dir)

        self._ensure_csv_exists(config.gt_csv_path)
        self._ensure_csv_exists(config.pred_csv_path)
        self.write_summary()

    def _remember_town(self, town: str) -> None:
        if town not in self._town_order:
            self._town_order.append(town)

    def _summary_towns(self) -> List[str]:
        return list(self._town_order)

    def _ensure_csv_exists(self, path: str) -> None:
        if not os.path.exists(path):
            _write_csv_rows(path, [])

    def _reconcile_csv(
            self, path: str, valid_sample_ids: set[str]) -> List[Dict[str, str]]:
        rows = _read_csv_rows(path)
        if not rows:
            _write_csv_rows(path, [])
            return []

        kept_rows: List[Dict[str, str]] = []
        seen_sample_ids: set[str] = set()
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id not in valid_sample_ids:
                continue
            if sample_id in seen_sample_ids:
                continue
            kept_rows.append(row)
            seen_sample_ids.add(sample_id)

        _write_csv_rows(path, kept_rows)
        return kept_rows

    def town_complete(self, town: str) -> bool:
        return (
            int(self.gt_samples_per_town[town])
            >= int(self.config.target_samples_per_town)
            or int(self.frames_per_town[town]) >= int(self.config.max_frames_per_town)
        )

    def next_episode_id(self, town: str) -> int:
        self.max_episode_id_per_town[town] = int(
            self.max_episode_id_per_town[town]) + 1
        return int(self.max_episode_id_per_town[town])

    def distance_bin_count(self, town: str, bin_idx: int) -> int:
        return int(self.distance_bins_per_town[town][int(bin_idx)])

    def append_frame(
        self,
        *,
        town: str,
        episode_id: int,
        tick: int,
        rgb_image: np.ndarray,
        samples: Sequence[TargetSample],
    ) -> int:
        if not samples:
            raise ValueError(
                "append_frame requires at least one accepted sample")

        frame_id = int(self.next_frame_id)
        self.next_frame_id += 1

        rgb_abs_path = _rgb_path_for_frame(self.config, frame_id)
        save_rgb_jpeg(
            rgb_image, rgb_abs_path, quality=int(
                self.config.rgb_jpeg_quality))
        self.bytes_rgb += os.path.getsize(rgb_abs_path)

        gt_rows: List[Dict[str, object]] = []
        pred_rows: List[Dict[str, object]] = []
        actor_ids: List[int] = []
        for sample in samples:
            actor_ids.append(sample.actor_id)
            mask_abs_path = _mask_path_for_frame(
                self.config, frame_id, sample.actor_id)
            save_binary_mask(sample.mask.astype(bool), mask_abs_path)
            self.bytes_masks += os.path.getsize(mask_abs_path)

            gt_rows.append(
                self._build_row(
                    sample_id=_sample_id(frame_id, sample.actor_id),
                    frame_id=frame_id,
                    episode_id=episode_id,
                    tick=tick,
                    town=town,
                    actor_id=sample.actor_id,
                    rgb_abs_path=rgb_abs_path,
                    mask_abs_path=mask_abs_path,
                    bbox_xyxy=sample.bbox_xyxy,
                    mask_area_px=sample.mask_area_px,
                    pose=sample.actor_record,
                    pose_score=1.0,
                )
            )
            if sample.pred_pose is not None:
                pred_rows.append(
                    self._build_row(
                        sample_id=_sample_id(frame_id, sample.actor_id),
                        frame_id=frame_id,
                        episode_id=episode_id,
                        tick=tick,
                        town=town,
                        actor_id=sample.actor_id,
                        rgb_abs_path=rgb_abs_path,
                        mask_abs_path=mask_abs_path,
                        bbox_xyxy=sample.bbox_xyxy,
                        mask_area_px=sample.mask_area_px,
                        pose=sample.pred_pose,
                        pose_score=float(sample.pred_pose.get("score", 0.0)),
                    )
                )

        _append_csv_rows(self.config.gt_csv_path, gt_rows)
        _append_csv_rows(self.config.pred_csv_path, pred_rows)

        manifest_record = {
            "frame_id": frame_id,
            "episode_id": int(episode_id),
            "town": str(town),
            "tick": int(tick),
            "rgb_path": relative_path(rgb_abs_path, self.config.output_dir),
            "accepted_target_count": len(gt_rows),
            "predicted_target_count": len(pred_rows),
            "accepted_actor_ids": [int(actor_id) for actor_id in actor_ids],
        }
        with open(self.config.frames_manifest_path, "a") as handle:
            handle.write(
                json.dumps(
                    manifest_record,
                    default=_json_ready) +
                "\n")

        self.manifest_records.append(manifest_record)
        self._remember_town(town)
        self.frames_per_town[town] += 1
        self.gt_samples_per_town[town] += len(gt_rows)
        self.pred_samples_per_town[town] += len(pred_rows)
        for row in gt_rows:
            bin_idx = distance_bin_index(
                float(row["dx_m"]), self.config.distance_bins_m)
            self.distance_bins_per_town[town][bin_idx] += 1

        self.write_summary()
        return frame_id

    def _build_row(
        self,
        *,
        sample_id: str,
        frame_id: int,
        episode_id: int,
        tick: int,
        town: str,
        actor_id: int,
        rgb_abs_path: str,
        mask_abs_path: str,
        bbox_xyxy: Tuple[int, int, int, int],
        mask_area_px: int,
        pose: Mapping[str, object],
        pose_score: float,
    ) -> Dict[str, object]:
        yaw_deg = float(pose["yaw_deg"])
        yaw_follow_deg = float(
            pose.get("yaw_follow_deg", canonicalize_follow_yaw_deg(yaw_deg))
        )
        if "follow_valid" in pose:
            follow_valid = bool(pose["follow_valid"])
        else:
            follow_valid = actor_is_follow_valid(pose, self.config)
        return {
            "sample_id": sample_id,
            "frame_id": int(frame_id),
            "episode_id": int(episode_id),
            "town": str(town),
            "tick": int(tick),
            "actor_id": int(actor_id),
            "rgb_path": relative_path(rgb_abs_path, self.config.output_dir),
            "mask_path": relative_path(mask_abs_path, self.config.output_dir),
            "bbox_x1": int(bbox_xyxy[0]),
            "bbox_y1": int(bbox_xyxy[1]),
            "bbox_x2": int(bbox_xyxy[2]),
            "bbox_y2": int(bbox_xyxy[3]),
            "mask_area_px": int(mask_area_px),
            "dx_m": float(pose["dx_m"]),
            "dy_m": float(pose["dy_m"]),
            "dz_m": float(pose["dz_m"]),
            "yaw_deg": yaw_deg,
            "yaw_follow_deg": yaw_follow_deg,
            "follow_valid": int(follow_valid),
            "pose_score": float(pose_score),
        }

    def write_summary(self) -> str:
        towns = self._summary_towns()
        total_frames = int(sum(self.frames_per_town.values()))
        total_gt = int(sum(self.gt_samples_per_town.values()))
        total_pred = int(sum(self.pred_samples_per_town.values()))
        bytes_other = 0
        for path in (
            self.config.gt_csv_path,
            self.config.pred_csv_path,
            self.config.frames_manifest_path,
        ):
            if os.path.exists(path):
                bytes_other += os.path.getsize(path)
        summary = {
            "output_dir": self.config.output_dir,
            "configured_towns": list(self.config.towns),
            "towns": towns,
            "total_frames": total_frames,
            "total_gt_samples": total_gt,
            "total_pred_samples": total_pred,
            "avg_gt_targets_per_frame": float(total_gt / max(total_frames, 1)),
            "avg_pred_targets_per_frame": float(total_pred / max(total_frames, 1)),
            "bytes_total": int(self.bytes_rgb + self.bytes_masks + bytes_other),
            "bytes_rgb": int(self.bytes_rgb),
            "bytes_masks": int(self.bytes_masks),
            "town_progress": {
                town: {
                    "frames": int(self.frames_per_town[town]),
                    "gt_samples": int(self.gt_samples_per_town[town]),
                    "pred_samples": int(self.pred_samples_per_town[town]),
                    "distance_bin_counts": {
                        str(bin_idx): int(count)
                        for bin_idx, count in sorted(
                            self.distance_bins_per_town[town].items()
                        )
                    },
                }
                for town in towns
            },
        }
        with open(self.config.collection_summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        return self.config.collection_summary_path


def _directory_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return total
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return int(total)


def _valid_mask_candidate(
    mask: np.ndarray,
    config: Config,
) -> Optional[Tuple[int, int, int, int]]:
    bbox = binary_mask_to_bbox(mask)
    if bbox is None:
        return None
    if int(mask.astype(bool).sum()) < int(config.min_mask_area_px):
        return None
    if bbox_touches_edge(
        bbox,
        int(config.image_width),
        int(config.image_height),
        int(config.edge_margin_px),
    ):
        return None
    return bbox


def _actor_gt_masks(
    actor_records: Sequence[Mapping[str, object]],
    instance_image: np.ndarray,
    config: Config,
) -> Dict[int, np.ndarray]:
    masks: Dict[int, np.ndarray] = {}
    for actor in actor_records:
        actor_id = int(actor["actor_id"])
        instance_id = int(actor["instance_id"])
        mask = vehicle_instance_mask(
            instance_image,
            instance_id,
            config.vehicle_semantic_tag)
        if np.any(mask):
            masks[actor_id] = mask
    return masks


def _sort_actor_priority(
    actor: Mapping[str, object],
    writer: DatasetWriter,
    town: str,
) -> Tuple[int, int, float]:
    return (
        writer.distance_bin_count(town, int(actor["distance_bin"])),
        -int(actor["pixel_area"]),
        float(actor["dx_m"]),
    )


def _select_target_actors(
    visible_actors: Sequence[Mapping[str, object]],
    *,
    writer: DatasetWriter,
    town: str,
    config: Config,
) -> List[Mapping[str, object]]:
    candidates: List[Mapping[str, object]] = []
    for actor in visible_actors:
        follow_valid = actor_is_follow_valid(actor, config)
        actor_dict = dict(actor)
        actor_dict["follow_valid"] = int(follow_valid)
        if bool(config.follow_only) and not follow_valid:
            continue
        candidates.append(actor_dict)

    if bool(config.follow_only) and len(candidates) < int(
            config.min_follow_actors_per_frame):
        return []

    candidates.sort(key=lambda item: _sort_actor_priority(item, writer, town))

    max_targets = int(config.max_follow_actors_per_frame)
    if bool(config.follow_only) and max_targets > 0:
        candidates = candidates[:max_targets]

    if bool(config.follow_only) and len(candidates) < int(
            config.min_follow_actors_per_frame):
        return []
    return candidates


def _collect_sam3_samples(
    *,
    mask_generator: VisionDetector,
    rgb_image: np.ndarray,
    instance_image: np.ndarray,
    actor_records: Sequence[Mapping[str, object]],
    config: Config,
) -> List[TargetSample]:
    if not actor_records:
        return []

    gt_masks = _actor_gt_masks(actor_records, instance_image, config)
    if not gt_masks:
        return []

    state = mask_generator.set_image(rgb_image)
    samples: List[TargetSample] = []
    for actor in actor_records:
        actor_id = int(actor["actor_id"])
        gt_mask = gt_masks.get(actor_id)
        if gt_mask is None:
            continue

        prompt_bbox = (
            int(actor["bbox_x1"]),
            int(actor["bbox_y1"]),
            int(actor["bbox_x2"]),
            int(actor["bbox_y2"]),
        )
        best_mask = None
        best_bbox = None
        best_iou = 0.0
        for candidate in mask_generator.segment_from_box(state, prompt_bbox):
            candidate_mask = candidate["mask"].astype(bool)
            bbox = _valid_mask_candidate(candidate_mask, config)
            if bbox is None:
                continue
            overlap = mask_iou(candidate_mask, gt_mask)
            if overlap > best_iou:
                best_iou = overlap
                best_mask = candidate_mask
                best_bbox = bbox

        if best_mask is None or best_bbox is None:
            continue
        if best_iou < float(config.sam3_actor_iou_thr):
            continue

        samples.append(
            TargetSample(
                actor_record=actor,
                mask=best_mask,
                bbox_xyxy=best_bbox,
            )
        )
    return samples


def _attach_detector_predictions(
    samples: Sequence[TargetSample],
    detections: Sequence[Mapping[str, object]],
    *,
    config: Config,
) -> List[TargetSample]:
    if not samples:
        return []

    actor_records = [sample.actor_record for sample in samples]
    matches = match_detections_to_actor_records(
        list(detections),
        actor_records,
        max_match_dist_m=float(config.detector_match_dist_m),
    )

    enriched: List[TargetSample] = []
    for sample in samples:
        actor_id = sample.actor_id
        pred_match = matches.get(actor_id)
        pred_pose = None
        if pred_match is not None:
            center = pred_match.get("center", np.zeros(3, dtype=np.float32))
            if isinstance(center, list):
                center = np.asarray(center, dtype=np.float32)
            pred_pose = {
                "dx_m": float(center[0]),
                "dy_m": float(center[1]),
                "dz_m": float(center[2]),
                "yaw_deg": float(pred_match["yaw_deg"]),
                "yaw_follow_deg": float(
                    canonicalize_follow_yaw_deg(float(pred_match["yaw_deg"]))
                ),
                "follow_valid": int(
                    sample.actor_record.get("follow_valid", False)
                ),
                "score": float(pred_match.get("score", 0.0)),
            }
        enriched.append(
            TargetSample(
                actor_record=sample.actor_record,
                mask=sample.mask,
                bbox_xyxy=sample.bbox_xyxy,
                pred_pose=pred_pose,
            )
        )
    return enriched


def collect_dataset(config: Config) -> None:
    """Run single-pass data collection and write the final dataset."""
    writer = DatasetWriter(config)
    mask_generator = VisionDetector(
        repo_path=config.sam3_repo_path,
        checkpoint_path=config.sam3_checkpoint_path,
        prompt=config.sam3_prompt,
        fallback_prompt=config.sam3_fallback_prompt,
        confidence_threshold=config.sam3_confidence_threshold,
        duplicate_iou_thr=config.sam3_duplicate_iou_thr,
        device=config.sam3_device,
    )
    detector = MMDet3DDetector(
        DetectorSpec(
            name=config.detector_name,
            config_path=config.detector_config,
            checkpoint_path=config.detector_checkpoint,
            score_thr=config.detector_score_thr,
            device=config.detector_device,
        )
    )

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(float(config.client_timeout_s))

    for town in config.towns:
        if writer.town_complete(town):
            print(
                f"[collect-dataset] {town} already complete with "
                f"{writer.gt_samples_per_town[town]} GT samples across "
                f"{writer.frames_per_town[town]} frames"
            )
            continue

        print(f"[collect-dataset] Loading {town}")
        world = setup_world(client, town, config)
        traffic_manager = None
        if config.traffic_mode == "traffic_manager":
            traffic_manager = configure_traffic_manager(client, world, config)

        for _ in range(int(config.max_episodes_per_town)):
            if writer.town_complete(town):
                break

            episode_id = writer.next_episode_id(town)
            print(
                f"[collect-dataset] {town}: starting episode {episode_id} "
                f"from frame {writer.frames_per_town[town]}"
            )

            actors_to_cleanup: List[object] = []
            rig: Optional[SensorRig] = None
            try:
                ego = spawn_ego_vehicle(
                    world, config, traffic_manager=traffic_manager)
                actors_to_cleanup.append(ego)
                traffic_ids = spawn_background_traffic(
                    client,
                    world,
                    ego,
                    int(config.num_traffic_vehicles),
                    config,
                    traffic_manager=traffic_manager,
                )
                actors_to_cleanup.extend(traffic_ids)

                rig = SensorRig(world, ego, config)
                rig.warmup(int(config.warmup_ticks))

                episode_ticks = 0
                offroad_ticks = 0
                while (
                    episode_ticks < int(config.episode_frame_budget)
                    and not writer.town_complete(town)
                ):
                    world.tick()
                    episode_ticks += 1

                    if not ego_on_driving_lane(world, ego):
                        offroad_ticks += 1
                        if offroad_ticks >= 5:
                            print(
                                "[collect-dataset] "
                                f"{town}: restarting episode {episode_id} "
                                "after ego left the driving lane"
                            )
                            break
                        continue
                    offroad_ticks = 0

                    snapshot = rig.get_snapshot(timeout=2.0)
                    visible_actors = collect_visible_vehicle_records(
                        world, ego, snapshot, config)
                    if not visible_actors:
                        continue

                    target_actors = _select_target_actors(
                        visible_actors,
                        writer=writer,
                        town=town,
                        config=config,
                    )
                    if not target_actors:
                        continue

                    samples = _collect_sam3_samples(
                        mask_generator=mask_generator,
                        rgb_image=snapshot.rgb,
                        instance_image=snapshot.instance,
                        actor_records=target_actors,
                        config=config,
                    )
                    if not samples:
                        continue

                    detections = detector.detect(snapshot.lidar)
                    samples = _attach_detector_predictions(
                        samples,
                        detections,
                        config=config,
                    )

                    frame_id = writer.append_frame(
                        town=town,
                        episode_id=episode_id,
                        tick=int(snapshot.tick),
                        rgb_image=snapshot.rgb,
                        samples=samples,
                    )

                    if frame_id % 25 == 0:
                        print(
                            f"[collect-dataset] {town}: "
                            f"{writer.frames_per_town[town]} frames, "
                            f"{writer.gt_samples_per_town[town]} GT samples, "
                            f"{writer.pred_samples_per_town[town]} predicted samples"
                        )
            finally:
                if rig is not None:
                    rig.destroy()
                destroy_actors(world, actors_to_cleanup)
                for _ in range(5):
                    world.tick()

        print(
            f"[collect-dataset] {town}: finished with "
            f"{writer.gt_samples_per_town[town]} GT samples across "
            f"{writer.frames_per_town[town]} frames"
        )

    summary_path = writer.write_summary()
    print(f"[collect-dataset] wrote {summary_path}")
