"""Detailed dataset and detector-vs-GT reporting for per-target CARLA datasets."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .config import Config
from .ground_truth import canonicalize_follow_yaw_deg
from .utils import wrap_angle_deg


def _load_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "frame_id": int(row["frame_id"]),
                    "episode_id": int(row["episode_id"]),
                    "town": row["town"],
                    "tick": int(row["tick"]),
                    "actor_id": int(row["actor_id"]),
                    "rgb_path": row["rgb_path"],
                    "mask_path": row["mask_path"],
                    "bbox_x1": int(row["bbox_x1"]),
                    "bbox_y1": int(row["bbox_y1"]),
                    "bbox_x2": int(row["bbox_x2"]),
                    "bbox_y2": int(row["bbox_y2"]),
                    "mask_area_px": int(row["mask_area_px"]),
                    "dx_m": float(row["dx_m"]),
                    "dy_m": float(row["dy_m"]),
                    "dz_m": float(row["dz_m"]),
                    "yaw_deg": float(row["yaw_deg"]),
                    "yaw_follow_deg": float(
                        row.get(
                            "yaw_follow_deg",
                            canonicalize_follow_yaw_deg(float(row["yaw_deg"])),
                        )
                    ),
                    "follow_valid": bool(
                        int(row["follow_valid"])
                        if "follow_valid" in row
                        else 1
                    ),
                    "pose_score": float(row["pose_score"]),
                }
            )
    return rows


def _targets_per_frame(rows: Iterable[Mapping[str, object]]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        counter[int(row["frame_id"])] += 1
    return counter


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _wrap_follow_error_deg(pred_yaw_deg: float, gt_yaw_deg: float) -> float:
    pred_follow = canonicalize_follow_yaw_deg(pred_yaw_deg)
    gt_follow = canonicalize_follow_yaw_deg(gt_yaw_deg)
    return abs(float(wrap_angle_deg(pred_follow - gt_follow)))


def _wrap_yaw_mod_180_deg(pred_yaw_deg: float, gt_yaw_deg: float) -> float:
    yaw_error = float(wrap_angle_deg(pred_yaw_deg - gt_yaw_deg))
    return float(
        min(
            abs(yaw_error),
            abs(wrap_angle_deg(yaw_error - 180.0)),
            abs(wrap_angle_deg(yaw_error + 180.0)),
        )
    )


def _label_value_bin(value: float, bin_edges: List[float]) -> str:
    if len(bin_edges) < 2:
        return "all"
    value = float(value)
    for idx in range(len(bin_edges) - 1):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        upper_cmp = value <= upper if idx == len(bin_edges) - 2 else value < upper
        if lower <= value and upper_cmp:
            upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"{lower:g}-{upper_label}"
    upper = bin_edges[-1]
    upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
    return f">={upper_label}"


def _summarize_matches(
    gt_rows: List[Mapping[str, object]],
    pred_rows: List[Mapping[str, object]],
    *,
    group_name: str,
    group_key_fn: Callable[[Mapping[str, object]], str],
) -> Dict[str, object]:
    pred_by_sample = {str(row["sample_id"]): row for row in pred_rows}
    grouped: Dict[str, Dict[str, List[float] | int]] = defaultdict(
        lambda: {
            "gt_samples": 0,
            "matched_samples": 0,
            "dx_abs": [],
            "dy_abs": [],
            "dz_abs": [],
            "yaw_abs": [],
            "yaw_mod_180_abs": [],
            "yaw_follow_abs": [],
            "scores": [],
        }
    )

    for gt_row in gt_rows:
        group_key = group_key_fn(gt_row)
        group = grouped[group_key]
        group["gt_samples"] += 1

        pred_row = pred_by_sample.get(str(gt_row["sample_id"]))
        if pred_row is None:
            continue

        group["matched_samples"] += 1
        group["dx_abs"].append(abs(float(pred_row["dx_m"]) - float(gt_row["dx_m"])))
        group["dy_abs"].append(abs(float(pred_row["dy_m"]) - float(gt_row["dy_m"])))
        group["dz_abs"].append(abs(float(pred_row["dz_m"]) - float(gt_row["dz_m"])))
        group["yaw_abs"].append(
            abs(float(wrap_angle_deg(float(pred_row["yaw_deg"]) - float(gt_row["yaw_deg"]))))
        )
        group["yaw_mod_180_abs"].append(
            _wrap_yaw_mod_180_deg(
                float(pred_row["yaw_deg"]),
                float(gt_row["yaw_deg"]),
            )
        )
        group["yaw_follow_abs"].append(
            _wrap_follow_error_deg(
                float(pred_row["yaw_deg"]),
                float(gt_row["yaw_deg"]),
            )
        )
        group["scores"].append(float(pred_row["pose_score"]))

    summary: Dict[str, object] = {}
    for key, values in sorted(grouped.items()):
        gt_samples = int(values["gt_samples"])
        matched_samples = int(values["matched_samples"])
        summary[key] = {
            "gt_samples": gt_samples,
            "matched_samples": matched_samples,
            "coverage_ratio": float(matched_samples / max(gt_samples, 1)),
            "dx_abs_m": _stats(list(values["dx_abs"])),
            "dy_abs_m": _stats(list(values["dy_abs"])),
            "dz_abs_m": _stats(list(values["dz_abs"])),
            "yaw_abs_deg": _stats(list(values["yaw_abs"])),
            "yaw_mod_180_abs_deg": _stats(list(values["yaw_mod_180_abs"])),
            "yaw_follow_abs_deg": _stats(list(values["yaw_follow_abs"])),
            "detector_score": _stats(list(values["scores"])),
        }

    return {
        "group_by": group_name,
        "groups": summary,
    }


def _dataset_summary(
    gt_rows: List[Mapping[str, object]],
    pred_rows: List[Mapping[str, object]],
) -> Dict[str, object]:
    gt_per_frame = _targets_per_frame(gt_rows)
    pred_per_frame = _targets_per_frame(pred_rows)
    gt_by_town = Counter(str(row["town"]) for row in gt_rows)
    pred_by_town = Counter(str(row["town"]) for row in pred_rows)

    return {
        "gt_frames": len(gt_per_frame),
        "pred_frames": len(pred_per_frame),
        "gt_samples": len(gt_rows),
        "pred_samples": len(pred_rows),
        "avg_gt_targets_per_frame": float(len(gt_rows) / max(len(gt_per_frame), 1)),
        "avg_pred_targets_per_frame": float(len(pred_rows) / max(len(pred_per_frame), 1)),
        "max_gt_targets_in_frame": int(max(gt_per_frame.values(), default=0)),
        "max_pred_targets_in_frame": int(max(pred_per_frame.values(), default=0)),
        "gt_targets_per_frame_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(gt_per_frame.values()).items())
        },
        "pred_targets_per_frame_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(pred_per_frame.values()).items())
        },
        "gt_samples_by_town": {town: int(count) for town, count in sorted(gt_by_town.items())},
        "pred_samples_by_town": {
            town: int(count) for town, count in sorted(pred_by_town.items())
        },
    }


def write_detailed_metrics_report(config: Config) -> str:
    if not os.path.exists(config.gt_csv_path):
        raise FileNotFoundError(
            f"{config.gt_csv_path} not found. Build the GT dataset first."
        )
    if not os.path.exists(config.pred_csv_path):
        raise FileNotFoundError(
            f"{config.pred_csv_path} not found. Attach predictions first."
        )

    gt_rows = _load_csv(config.gt_csv_path)
    pred_rows = _load_csv(config.pred_csv_path)

    distance_bins = list(config.distance_bins_m)
    area_bins = [0.0, 500.0, 1500.0, 4000.0, 8000.0, float("inf")]
    yaw_bins = [0.0, 10.0, 30.0, 60.0, 90.0]

    report = {
        "dataset_summary": _dataset_summary(gt_rows, pred_rows),
        "overall": _summarize_matches(
            gt_rows,
            pred_rows,
            group_name="overall",
            group_key_fn=lambda _row: "all",
        )["groups"]["all"],
        "by_town": _summarize_matches(
            gt_rows,
            pred_rows,
            group_name="town",
            group_key_fn=lambda row: str(row["town"]),
        ),
        "by_distance_bin": _summarize_matches(
            gt_rows,
            pred_rows,
            group_name="distance_bin_m",
            group_key_fn=lambda row: _label_value_bin(float(row["dx_m"]), distance_bins),
        ),
        "by_mask_area_bin": _summarize_matches(
            gt_rows,
            pred_rows,
            group_name="mask_area_px",
            group_key_fn=lambda row: _label_value_bin(float(row["mask_area_px"]), area_bins),
        ),
        "by_abs_yaw_follow_bin": _summarize_matches(
            gt_rows,
            pred_rows,
            group_name="abs_yaw_follow_deg",
            group_key_fn=lambda row: _label_value_bin(
                abs(float(row["yaw_follow_deg"])),
                yaw_bins,
            ),
        ),
    }

    os.makedirs(config.benchmark_dir, exist_ok=True)
    out_path = os.path.join(config.benchmark_dir, "detailed_metrics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
