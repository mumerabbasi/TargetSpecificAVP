"""Dataset utilities for target pose regression."""

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

from .config import TargetPoseTrainingConfig
from .preprocessing import build_model_inputs


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
    def from_rows(cls, rows: Sequence[Mapping[str, str]]) -> "TranslationStats":
        if not rows:
            raise ValueError("Cannot compute translation statistics from an empty split")

        values = np.asarray(
            [[float(row["dx_m"]), float(row["dy_m"])] for row in rows],
            dtype=np.float32,
        )
        mean = values.mean(axis=0)
        std = np.clip(values.std(axis=0), a_min=1e-6, a_max=None)
        return cls(
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> "TranslationStats":
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
        )


def load_pose_rows(config: TargetPoseTrainingConfig) -> List[Dict[str, str]]:
    """Load the selected compact pose CSV and apply training filters."""
    dataset_root = Path(config.dataset_root)
    csv_path = dataset_root / config.csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing pose CSV: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if config.require_follow_valid and row.get("follow_valid", "1") != "1":
                continue
            if int(float(row.get("mask_area_px", "0"))) < config.min_mask_area_px:
                continue

            rgb_path = dataset_root / row["rgb_path"]
            mask_path = dataset_root / row["mask_path"]
            if not rgb_path.exists() or not mask_path.exists():
                continue

            rows.append(dict(row))

    if not rows:
        raise ValueError(f"No usable training rows remained in {csv_path}")
    return rows


def build_group_splits(
    rows: Sequence[Mapping[str, str]],
    config: TargetPoseTrainingConfig,
) -> Dict[str, List[Dict[str, str]]]:
    """Split rows by town-and-episode groups to reduce temporal leakage."""
    groups_by_town: MutableMapping[str, List[Tuple[str, str]]] = defaultdict(list)
    seen_groups = set()

    for row in rows:
        group_key = (str(row["town"]), str(row["episode_id"]))
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        groups_by_town[group_key[0]].append(group_key)

    rng = random.Random(config.random_seed)
    group_to_split: Dict[Tuple[str, str], str] = {}
    for town, group_keys in groups_by_town.items():
        town_groups = list(group_keys)
        rng.shuffle(town_groups)
        split_sizes = _resolve_split_sizes(
            total_groups=len(town_groups),
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )

        start = 0
        for split_name, split_size in split_sizes.items():
            stop = start + split_size
            for group_key in town_groups[start:stop]:
                group_to_split[group_key] = split_name
            start = stop

    rows_by_split = {"train": [], "val": [], "test": []}
    for row in rows:
        split_name = group_to_split[(str(row["town"]), str(row["episode_id"]))]
        rows_by_split[split_name].append(dict(row))

    for split_name, limit in (
        ("train", config.max_train_samples),
        ("val", config.max_val_samples),
        ("test", config.max_test_samples),
    ):
        if limit > 0 and len(rows_by_split[split_name]) > limit:
            rng.shuffle(rows_by_split[split_name])
            rows_by_split[split_name] = rows_by_split[split_name][:limit]

    return rows_by_split


def split_summary(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
) -> Dict[str, Dict[str, int]]:
    """Build a compact summary of row, frame, and group counts."""
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, rows in rows_by_split.items():
        frame_keys = {(row["town"], row["frame_id"]) for row in rows}
        group_keys = {(row["town"], row["episode_id"]) for row in rows}
        summary[split_name] = {
            "rows": len(rows),
            "frames": len(frame_keys),
            "groups": len(group_keys),
        }
    return summary


class TargetPoseDataset(Dataset):
    """Dataset for mask-conditioned target pose regression."""

    def __init__(
        self,
        config: TargetPoseTrainingConfig,
        rows: Sequence[Mapping[str, str]],
        translation_stats: TranslationStats,
        training: bool,
    ) -> None:
        self.config = config
        self.rows = [dict(row) for row in rows]
        self.translation_stats = translation_stats
        self.training = training
        self.dataset_root = Path(config.dataset_root)

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        row = self.rows[index]

        rgb_array = np.asarray(
            Image.open(self.dataset_root / row["rgb_path"]).convert("RGB"),
            dtype=np.uint8,
        )
        mask_array = (
            np.asarray(
                Image.open(self.dataset_root / row["mask_path"]).convert("L"),
                dtype=np.uint8,
            )
            > 127
        ).astype(np.uint8)

        dx_m = float(row["dx_m"])
        dy_m = float(row["dy_m"])
        yaw_follow_deg = float(row["yaw_follow_deg"])

        if self.training and random.random() < 0.5:
            rgb_array = np.ascontiguousarray(rgb_array[:, ::-1])
            mask_array = np.ascontiguousarray(mask_array[:, ::-1])
            dy_m = -dy_m
            yaw_follow_deg = -yaw_follow_deg

        if self.training:
            rgb_array = np.asarray(
                self.color_jitter(Image.fromarray(rgb_array, mode="RGB")),
                dtype=np.uint8,
            )

        full_input, crop_input, geometry = build_model_inputs(
            rgb_array=rgb_array,
            mask_array=mask_array,
            image_size=self.config.image_size,
            crop_size=self.config.crop_size,
            crop_context_scale=self.config.crop_context_scale,
        )

        translation_raw = torch.tensor([dx_m, dy_m], dtype=torch.float32)
        translation_target = normalize_translation(
            translation_raw,
            self.translation_stats,
        )

        yaw_rad = math.radians(yaw_follow_deg)
        yaw_target = torch.tensor(
            [math.sin(yaw_rad), math.cos(yaw_rad)],
            dtype=torch.float32,
        )

        return {
            "full_input": full_input,
            "crop_input": crop_input,
            "geometry": geometry,
            "translation_target": translation_target,
            "translation_raw": translation_raw,
            "yaw_target": yaw_target,
            "yaw_follow_deg": torch.tensor(yaw_follow_deg, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "frame_id": int(row["frame_id"]),
            "town": row["town"],
        }


def normalize_translation(
    translation: torch.Tensor,
    stats: TranslationStats,
) -> torch.Tensor:
    """Normalize dx and dy with training-split statistics."""
    mean = torch.tensor(stats.mean, dtype=translation.dtype, device=translation.device)
    std = torch.tensor(stats.std, dtype=translation.dtype, device=translation.device)
    return (translation - mean) / std


def denormalize_translation(
    translation: torch.Tensor,
    stats: TranslationStats,
) -> torch.Tensor:
    """Map normalized dx and dy back to metric units."""
    mean = torch.tensor(stats.mean, dtype=translation.dtype, device=translation.device)
    std = torch.tensor(stats.std, dtype=translation.dtype, device=translation.device)
    return translation * std + mean


def _resolve_split_sizes(
    total_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    if total_groups <= 0:
        return {"train": 0, "val": 0, "test": 0}

    train_groups = int(total_groups * train_ratio)
    val_groups = int(total_groups * val_ratio)
    test_groups = total_groups - train_groups - val_groups

    if total_groups >= 3:
        if train_groups == 0:
            train_groups = 1
        if val_groups == 0:
            val_groups = 1
        test_groups = total_groups - train_groups - val_groups
        if test_groups <= 0:
            if train_groups >= val_groups and train_groups > 1:
                train_groups -= 1
            elif val_groups > 1:
                val_groups -= 1
            test_groups = total_groups - train_groups - val_groups
    elif total_groups == 2:
        train_groups, val_groups, test_groups = 1, 1, 0
    else:
        train_groups, val_groups, test_groups = 1, 0, 0

    return {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }
