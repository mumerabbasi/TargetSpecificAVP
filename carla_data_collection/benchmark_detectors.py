"""Benchmark 3D detectors at the per-target sample level."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from .config import Config
from .detector_3d import DetectorSpec, MMDet3DDetector
from .ground_truth import compute_pose_errors, match_detections_to_actor_records


def parse_candidate_spec(spec: str, default_score_thr: float, default_device: str) -> DetectorSpec:
    """Parse `name=config_path::checkpoint_path` CLI syntax."""
    if "=" not in spec or "::" not in spec:
        raise ValueError(
            "Candidate specs must use `name=config_path::checkpoint_path`."
        )
    name, payload = spec.split("=", 1)
    config_path, checkpoint_path = payload.split("::", 1)
    return DetectorSpec(
        name=name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        score_thr=default_score_thr,
        device=default_device,
    )


def _load_gt_groups(gt_csv_path: str) -> Dict[int, List[Dict[str, object]]]:
    groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    with open(gt_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            groups[frame_id].append(
                {
                    "actor_id": int(row["actor_id"]),
                    "sample_id": row["sample_id"],
                }
            )
    return groups


def benchmark_detectors(config: Config, candidate_specs: Iterable[DetectorSpec]) -> None:
    """Benchmark one or more detectors against accepted GT target samples."""
    if not os.path.exists(config.gt_csv_path):
        raise FileNotFoundError(
            f"{config.gt_csv_path} not found. Build the GT dataset first."
        )

    gt_groups = _load_gt_groups(config.gt_csv_path)
    if not gt_groups:
        raise RuntimeError("No GT samples available for benchmarking.")

    meta_by_frame = {
        int(name[len("frame_") : -len(".json")]): os.path.join(config.raw_metadata_dir, name)
        for name in os.listdir(config.raw_metadata_dir)
        if name.startswith("frame_") and name.endswith(".json")
    }

    results: Dict[str, Dict[str, float]] = {}
    for spec in candidate_specs:
        detector = MMDet3DDetector(spec)
        matched_total = 0
        actor_records_total = 0
        agg_errors = {
            "matched_samples": 0.0,
            "mae_dx_m": 0.0,
            "mae_dy_m": 0.0,
            "mae_dz_m": 0.0,
            "mae_yaw_deg": 0.0,
            "mae_yaw_deg_mod_180": 0.0,
        }
        per_town_counts: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "total_gt_samples": 0.0,
                "matched_pred_samples": 0.0,
                "matched_samples": 0.0,
                "mae_dx_m": 0.0,
                "mae_dy_m": 0.0,
                "mae_dz_m": 0.0,
                "mae_yaw_deg": 0.0,
                "mae_yaw_deg_mod_180": 0.0,
            }
        )

        for frame_id, gt_rows in gt_groups.items():
            meta_path = meta_by_frame.get(frame_id)
            if meta_path is None:
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

            lidar_path = os.path.join(config.output_dir, meta["lidar_path"])
            lidar = np.load(lidar_path)
            matches = match_detections_to_actor_records(
                detector.detect(lidar),
                actor_records,
                config.detector_match_dist_m,
            )
            matched_total += len(matches)
            actor_records_total += len(actor_records)
            town = str(meta["town"])
            per_town_counts[town]["total_gt_samples"] += float(len(actor_records))
            per_town_counts[town]["matched_pred_samples"] += float(len(matches))

            frame_errors = compute_pose_errors(actor_records, matches)
            if frame_errors["matched_samples"] > 0:
                agg_errors["matched_samples"] += frame_errors["matched_samples"]
                agg_errors["mae_dx_m"] += (
                    frame_errors["mae_dx_m"] * frame_errors["matched_samples"]
                )
                agg_errors["mae_dy_m"] += (
                    frame_errors["mae_dy_m"] * frame_errors["matched_samples"]
                )
                agg_errors["mae_dz_m"] += (
                    frame_errors["mae_dz_m"] * frame_errors["matched_samples"]
                )
                agg_errors["mae_yaw_deg"] += (
                    frame_errors["mae_yaw_deg"] * frame_errors["matched_samples"]
                )
                agg_errors["mae_yaw_deg_mod_180"] += (
                    frame_errors["mae_yaw_deg_mod_180"]
                    * frame_errors["matched_samples"]
                )
                per_town_counts[town]["matched_samples"] += frame_errors[
                    "matched_samples"
                ]
                per_town_counts[town]["mae_dx_m"] += (
                    frame_errors["mae_dx_m"] * frame_errors["matched_samples"]
                )
                per_town_counts[town]["mae_dy_m"] += (
                    frame_errors["mae_dy_m"] * frame_errors["matched_samples"]
                )
                per_town_counts[town]["mae_dz_m"] += (
                    frame_errors["mae_dz_m"] * frame_errors["matched_samples"]
                )
                per_town_counts[town]["mae_yaw_deg"] += (
                    frame_errors["mae_yaw_deg"] * frame_errors["matched_samples"]
                )
                per_town_counts[town]["mae_yaw_deg_mod_180"] += (
                    frame_errors["mae_yaw_deg_mod_180"]
                    * frame_errors["matched_samples"]
                )

        denom = max(agg_errors["matched_samples"], 1.0)
        results[spec.name] = {
            "total_gt_samples": float(actor_records_total),
            "matched_pred_samples": float(matched_total),
            "coverage_ratio": float(matched_total / max(actor_records_total, 1)),
            "mae_dx_m": float(agg_errors["mae_dx_m"] / denom),
            "mae_dy_m": float(agg_errors["mae_dy_m"] / denom),
            "mae_dz_m": float(agg_errors["mae_dz_m"] / denom),
            "mae_yaw_deg": float(agg_errors["mae_yaw_deg"] / denom),
            "mae_yaw_deg_mod_180": float(
                agg_errors["mae_yaw_deg_mod_180"] / denom
            ),
            "per_town": {
                town: {
                    "total_gt_samples": float(values["total_gt_samples"]),
                    "matched_pred_samples": float(values["matched_pred_samples"]),
                    "coverage_ratio": float(
                        values["matched_pred_samples"]
                        / max(values["total_gt_samples"], 1.0)
                    ),
                    "mae_dx_m": float(
                        values["mae_dx_m"] / max(values["matched_samples"], 1.0)
                    ),
                    "mae_dy_m": float(
                        values["mae_dy_m"] / max(values["matched_samples"], 1.0)
                    ),
                    "mae_dz_m": float(
                        values["mae_dz_m"] / max(values["matched_samples"], 1.0)
                    ),
                    "mae_yaw_deg": float(
                        values["mae_yaw_deg"] / max(values["matched_samples"], 1.0)
                    ),
                    "mae_yaw_deg_mod_180": float(
                        values["mae_yaw_deg_mod_180"]
                        / max(values["matched_samples"], 1.0)
                    ),
                }
                for town, values in sorted(per_town_counts.items())
            },
        }

    os.makedirs(config.benchmark_dir, exist_ok=True)
    out_path = os.path.join(config.benchmark_dir, "detector_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[benchmark] wrote {out_path}")
    for name, metrics in results.items():
        print(
            f"[benchmark] {name}: coverage={metrics['coverage_ratio']:.3f}, "
            f"mae_dx={metrics['mae_dx_m']:.3f}m, "
            f"mae_dy={metrics['mae_dy_m']:.3f}m, "
            f"mae_dz={metrics['mae_dz_m']:.3f}m, "
            f"mae_yaw={metrics['mae_yaw_deg']:.3f}deg"
        )
