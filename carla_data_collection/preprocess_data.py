"""Merge multiple per-target RAVP datasets into one dataset root.

Each source dataset is expected to follow the new shared-asset layout:

dataset/
├── rgb/
├── masks/
├── gt_poses.csv
└── pred_poses.csv

The merger:
1. Copies each RGB frame once with a new global frame id.
2. Copies each target mask once with a new frame-aware filename.
3. Rewrites both CSVs so the merged dataset keeps separate GT/pred labels
   while reusing the shared RGB and mask assets.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


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


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def _discover_dataset_roots(source_root: str, dest_root: str) -> List[str]:
    datasets: List[str] = []
    for name in sorted(os.listdir(source_root)):
        path = os.path.join(source_root, name)
        if not os.path.isdir(path):
            continue
        if os.path.abspath(path) == os.path.abspath(dest_root):
            continue
        if os.path.exists(
            os.path.join(
                path,
                "gt_poses.csv")) or os.path.exists(
            os.path.join(
                path,
                "pred_poses.csv")):
            datasets.append(path)
    return datasets


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_once(src: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collect_unique_frames(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
) -> Dict[str, List[Mapping[str, str]]]:
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for rows in rows_by_split.values():
        for row in rows:
            grouped[row["rgb_path"]].append(row)
    return grouped


def combine_datasets(source_root: str, dest_root: str) -> Tuple[int, int, int]:
    dataset_roots = _discover_dataset_roots(source_root, dest_root)
    if not dataset_roots:
        raise FileNotFoundError(
            f"No built per-target datasets found under {source_root}"
        )

    rgb_dest_dir = os.path.join(dest_root, "rgb")
    mask_dest_dir = os.path.join(dest_root, "masks")
    _ensure_dir(dest_root)
    _ensure_dir(rgb_dest_dir)
    _ensure_dir(mask_dest_dir)

    combined_gt_rows: List[Dict[str, str]] = []
    combined_pred_rows: List[Dict[str, str]] = []
    next_frame_id = 0
    copied_masks: Dict[Tuple[str, str], str] = {}

    for dataset_root in dataset_roots:
        gt_rows = _read_csv_rows(os.path.join(dataset_root, "gt_poses.csv"))
        pred_rows = _read_csv_rows(
            os.path.join(
                dataset_root,
                "pred_poses.csv"))
        grouped_frames = _collect_unique_frames(
            {"gt": gt_rows, "pred": pred_rows})
        rgb_mapping: Dict[str, Tuple[int, str]] = {}

        for source_rgb_rel in sorted(grouped_frames.keys()):
            src_rgb = os.path.join(dataset_root, source_rgb_rel)
            _, ext = os.path.splitext(source_rgb_rel)
            if not ext:
                ext = ".png"
            new_rgb_rel = os.path.join(
                "rgb", f"frame_{
                    next_frame_id:06d}{ext}")
            _copy_once(src_rgb, os.path.join(dest_root, new_rgb_rel))
            rgb_mapping[source_rgb_rel] = (
                next_frame_id,
                new_rgb_rel.replace(os.sep, "/"),
            )
            next_frame_id += 1

        for rows, target in (
            (gt_rows, combined_gt_rows),
            (pred_rows, combined_pred_rows),
        ):
            for row in rows:
                new_frame_id, new_rgb_rel = rgb_mapping[row["rgb_path"]]
                actor_id = int(row["actor_id"])
                mask_key = (dataset_root, row["mask_path"])
                if mask_key not in copied_masks:
                    src_mask = os.path.join(dataset_root, row["mask_path"])
                    new_mask_rel = os.path.join(
                        "masks",
                        f"frame_{new_frame_id:06d}_actor_{actor_id}.png",
                    ).replace(os.sep, "/")
                    _copy_once(src_mask, os.path.join(dest_root, new_mask_rel))
                    copied_masks[mask_key] = new_mask_rel

                merged_row = dict(row)
                merged_row["frame_id"] = str(new_frame_id)
                merged_row["sample_id"] = f"{new_frame_id:06d}_{actor_id}"
                merged_row["rgb_path"] = new_rgb_rel
                merged_row["mask_path"] = copied_masks[mask_key]
                target.append(merged_row)

    with open(os.path.join(dest_root, "gt_poses.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined_gt_rows)

    with open(os.path.join(dest_root, "pred_poses.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined_pred_rows)

    return next_frame_id, len(combined_gt_rows), len(combined_pred_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple built per-target RAVP datasets"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing one or more built dataset roots",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for the merged dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, gt_rows, pred_rows = combine_datasets(args.source, args.dest)
    print(
        f"Merged dataset written to {args.dest}: "
        f"{frames} frames, {gt_rows} GT rows, {pred_rows} predicted rows"
    )


if __name__ == "__main__":
    main()
