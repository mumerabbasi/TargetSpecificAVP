"""Dataset utilities for mask-conditioned target pose learning."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .config import PoseEstimationConfig


@dataclass(frozen=True)
class TranslationStats:
    """Mean and standard deviation for translation targets."""

    mean: Tuple[float, float]
    std: Tuple[float, float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_rows(
            cls, rows: Sequence[Mapping[str, str]]) -> "TranslationStats":
        if not rows:
            raise ValueError(
                "Cannot compute translation statistics from an empty split")
        values = np.array(
            [
                [float(row["dx_m"]), float(row["dy_m"])]
                for row in rows
            ],
            dtype=np.float32,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.clip(std, a_min=1e-6, a_max=None)
        return cls(
            mean=tuple(float(v) for v in mean),
            std=tuple(float(v) for v in std),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str,
                  Sequence[float]]) -> "TranslationStats":
        return cls(
            mean=tuple(float(v) for v in payload["mean"]),
            std=tuple(float(v) for v in payload["std"]),
        )


def load_pose_rows(config: PoseEstimationConfig) -> List[Dict[str, str]]:
    """Load the selected pose CSV and filter rows for training."""

    dataset_root = Path(config.dataset_root)
    csv_path = dataset_root / config.csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing pose CSV: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if config.require_follow_valid and row.get(
                    "follow_valid", "1") != "1":
                continue
            if int(float(row.get("mask_area_px", "0"))
                   ) < config.min_mask_area_px:
                continue

            rgb_path = dataset_root / row["rgb_path"]
            mask_path = dataset_root / row["mask_path"]
            if not rgb_path.exists() or not mask_path.exists():
                continue

            rows.append(dict(row))

    if not rows:
        raise ValueError(
            f"No training rows remained after filtering {csv_path}")
    return rows


def build_frame_splits(
    rows: Sequence[Mapping[str, str]],
    config: PoseEstimationConfig,
) -> Dict[str, List[Dict[str, str]]]:
    """Split rows by frame so one RGB frame belongs to exactly one split."""

    frames_by_town: MutableMapping[str,
                                   List[Tuple[str, str]]] = defaultdict(list)
    seen_keys = set()

    for row in rows:
        key = (row["town"], row["frame_id"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        frames_by_town[row["town"]].append(key)

    rng = random.Random(config.random_seed)
    frame_to_split: Dict[Tuple[str, str], str] = {}
    for town, frame_keys in frames_by_town.items():
        town_keys = list(frame_keys)
        rng.shuffle(town_keys)
        split_sizes = _resolve_split_sizes(
            len(town_keys),
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
        )
        start = 0
        for split_name, split_size in split_sizes.items():
            stop = start + split_size
            for key in town_keys[start:stop]:
                frame_to_split[key] = split_name
            start = stop

    split_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split = frame_to_split[(row["town"], row["frame_id"])]
        split_rows[split].append(dict(row))

    for split_name, limit in (
        ("train", config.max_train_samples),
        ("val", config.max_val_samples),
        ("test", config.max_test_samples),
    ):
        if limit > 0 and len(split_rows[split_name]) > limit:
            rng.shuffle(split_rows[split_name])
            split_rows[split_name] = split_rows[split_name][:limit]

    return split_rows


def _resolve_split_sizes(
    total_frames: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    if total_frames <= 0:
        return {"train": 0, "val": 0, "test": 0}

    train_frames = int(total_frames * train_ratio)
    val_frames = int(total_frames * val_ratio)
    test_frames = total_frames - train_frames - val_frames

    if total_frames >= 3:
        if train_frames == 0:
            train_frames = 1
        if val_frames == 0:
            val_frames = 1
        test_frames = total_frames - train_frames - val_frames
        if test_frames <= 0:
            if train_frames >= val_frames and train_frames > 1:
                train_frames -= 1
            elif val_frames > 1:
                val_frames -= 1
            test_frames = total_frames - train_frames - val_frames
    elif total_frames == 2:
        train_frames, val_frames, test_frames = 1, 1, 0
    else:
        train_frames, val_frames, test_frames = 1, 0, 0

    return {
        "train": train_frames,
        "val": val_frames,
        "test": test_frames,
    }


class PoseEstimationDataset(Dataset):
    """Dataset for mask-conditioned target pose prediction."""

    def __init__(
        self,
        config: PoseEstimationConfig,
        rows: Sequence[Mapping[str, str]],
        translation_stats: TranslationStats,
        training: bool,
    ) -> None:
        self.config = config
        self.rows = [dict(row) for row in rows]
        self.training = training
        self.dataset_root = Path(config.dataset_root)
        self.translation_stats = translation_stats
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.02,
        )
        self.rgb_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        rgb_image = Image.open(
            self.dataset_root /
            row["rgb_path"]).convert("RGB")
        mask_image = Image.open(
            self.dataset_root /
            row["mask_path"]).convert("L")

        original_width, original_height = rgb_image.size
        target_width, target_height = self.config.image_size

        if self.training:
            rgb_image = self.color_jitter(rgb_image)

        rgb_image = rgb_image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR)
        mask_image = mask_image.resize(
            (target_width, target_height), Image.Resampling.NEAREST)

        rgb_tensor = TF.to_tensor(rgb_image)
        rgb_tensor = TF.normalize(
            rgb_tensor,
            self.rgb_mean.tolist(),
            self.rgb_std.tolist())

        mask_array = np.asarray(mask_image, dtype=np.float32)
        mask_array = (mask_array > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        geometry = self._build_geometry_vector(
            row, original_width, original_height)
        translation_raw = torch.tensor(
            [
                float(row["dx_m"]),
                float(row["dy_m"]),
            ],
            dtype=torch.float32,
        )
        translation_target = normalize_translation(
            translation_raw, self.translation_stats)

        yaw_follow_deg = float(row["yaw_follow_deg"])
        yaw_follow_rad = math.radians(yaw_follow_deg)
        yaw_target = torch.tensor(
            [math.sin(yaw_follow_rad), math.cos(yaw_follow_rad)],
            dtype=torch.float32,
        )

        return {
            "input": torch.cat([rgb_tensor, mask_tensor], dim=0),
            "mask": mask_tensor,
            "geometry": geometry,
            "translation_target": translation_target,
            "translation_raw": translation_raw,
            "yaw_target": yaw_target,
            "yaw_follow_deg": torch.tensor(yaw_follow_deg, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "frame_id": int(row["frame_id"]),
            "town": row["town"],
        }

    def _build_geometry_vector(
        self,
        row: Mapping[str, str],
        image_width: int,
        image_height: int,
    ) -> torch.Tensor:
        x1 = float(row["bbox_x1"])
        y1 = float(row["bbox_y1"])
        x2 = float(row["bbox_x2"])
        y2 = float(row["bbox_y2"])

        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        mask_area = float(row["mask_area_px"])

        return torch.tensor(
            [
                center_x / max(image_width, 1),
                center_y / max(image_height, 1),
                box_w / max(image_width, 1),
                box_h / max(image_height, 1),
                mask_area / max(image_width * image_height, 1),
            ],
            dtype=torch.float32,
        )


def normalize_translation(
    translation: torch.Tensor,
    stats: TranslationStats,
) -> torch.Tensor:
    mean = torch.tensor(
        stats.mean,
        dtype=translation.dtype,
        device=translation.device)
    std = torch.tensor(
        stats.std,
        dtype=translation.dtype,
        device=translation.device)
    return (translation - mean) / std


def denormalize_translation(
    translation: torch.Tensor,
    stats: TranslationStats,
) -> torch.Tensor:
    mean = torch.tensor(
        stats.mean,
        dtype=translation.dtype,
        device=translation.device)
    std = torch.tensor(
        stats.std,
        dtype=translation.dtype,
        device=translation.device)
    return translation * std + mean


def split_summary(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
) -> Dict[str, Dict[str, int]]:
    """Build a compact summary of row and frame counts per split."""

    summary: Dict[str, Dict[str, int]] = {}
    for split_name, rows in rows_by_split.items():
        frames = {(row["town"], row["frame_id"]) for row in rows}
        summary[split_name] = {
            "rows": len(rows),
            "frames": len(frames),
        }
    return summary
