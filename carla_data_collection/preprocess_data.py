"""Preprocess and combine fragmented CARLA datasets into a unified dataset.

This script:
    1. Scans all dataset directories in the source folder
    2. Copies images with new sequential frame IDs
    3. Merges all CSV files with updated frame IDs
    4. Saves the combined dataset to the destination folder

Usage:
    python preprocess_data.py --source /path/to/fragments --dest /path/to/combined
"""

import argparse
import os
import shutil
from typing import Any

import pandas as pd
from tqdm import tqdm


def scan_datasets(source_base: str) -> list[dict[str, Any]]:
    """Scan source directory for dataset folders and count images.

    Args:
        source_base: Path to directory containing dataset folders.

    Returns:
        List of dicts with dataset info (name, path, num_images, has_csv).
    """
    dataset_dirs = sorted([
        d for d in os.listdir(source_base)
        if d.startswith("carla_dataset")
        and os.path.isdir(os.path.join(source_base, d))
    ])

    print(f"Found {len(dataset_dirs)} dataset directories:")

    dataset_info = []
    for d in dataset_dirs:
        dir_path = os.path.join(source_base, d)
        png_files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
        csv_path = os.path.join(dir_path, "poses.csv")
        has_csv = os.path.exists(csv_path)

        info = {
            "name": d,
            "path": dir_path,
            "num_images": len(png_files),
            "has_csv": has_csv,
        }
        dataset_info.append(info)
        print(f"  {d}: {len(png_files):,} images, CSV: {has_csv}")

    total_images = sum(d["num_images"] for d in dataset_info)
    print(f"\nTotal images to combine: {total_images:,}")

    return dataset_info


def combine_datasets(
    dataset_info: list[dict[str, Any]],
    dest_base: str,
) -> pd.DataFrame:
    """Combine all datasets by copying images and merging CSVs.

    Images are grouped by town so that all frames from the same town
    have consecutive frame IDs in the final dataset.

    Args:
        dataset_info: List of dataset info dicts from scan_datasets().
        dest_base: Destination directory for combined dataset.

    Returns:
        Combined DataFrame with all poses.
    """
    os.makedirs(dest_base, exist_ok=True)
    rgb_dir = os.path.join(dest_base, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    print(f"\nDestination directory: {dest_base}")
    print(f"RGB directory: {rgb_dir}")

    # =================================================================
    # PHASE 1: Load all CSVs to build frame-to-town mapping
    # =================================================================
    print("\nPhase 1: Loading CSVs to determine town for each frame...")

    # Collect all frames: list of (dataset_name, old_frame_id, town, img_path)
    # Only include frames that have CSV entries (poses)
    all_frames: list[tuple[str, int, str, str]] = []
    all_dfs: list[pd.DataFrame] = []

    for info in tqdm(dataset_info, desc="Loading CSVs"):
        if info["num_images"] == 0:
            continue

        dataset_path = info["path"]
        dataset_name = info["name"]
        csv_path = os.path.join(dataset_path, "poses.csv")

        if not os.path.exists(csv_path):
            print(f"  WARNING: No CSV for {dataset_name}, skipping")
            continue

        df = pd.read_csv(csv_path)
        df["_dataset_name"] = dataset_name  # Tag with source dataset
        all_dfs.append(df)

        # Build frame-to-town mapping from CSV
        # Each frame_id maps to exactly one town
        frame_to_town: dict[int, str] = {}
        for _, row in df.iterrows():
            frame_to_town[row["frame_id"]] = row["town"]

        # Get unique frame IDs from CSV (frames with poses)
        csv_frame_ids = set(df["frame_id"].unique())

        # Collect only image files that have corresponding CSV entries
        png_files = sorted([
            f for f in os.listdir(dataset_path) if f.endswith(".png")
        ])

        skipped_count = 0
        for png_file in png_files:
            old_frame_id = int(
                png_file.replace("rgb_", "").replace(".png", "")
            )
            # Skip frames without CSV entries (no poses)
            if old_frame_id not in csv_frame_ids:
                skipped_count += 1
                continue
            town = frame_to_town[old_frame_id]
            img_path = os.path.join(dataset_path, png_file)
            all_frames.append((dataset_name, old_frame_id, town, img_path))

        if skipped_count > 0:
            print(f"  {dataset_name}: skipped {skipped_count} frames without poses")

    print(f"  Total frames collected (with poses): {len(all_frames):,}")

    # =================================================================
    # PHASE 2: Group frames by town and sort
    # =================================================================
    print("\nPhase 2: Grouping frames by town...")

    # Group by town
    town_frames: dict[str, list[tuple[str, int, str]]] = {}
    for dataset_name, old_frame_id, town, img_path in all_frames:
        if town not in town_frames:
            town_frames[town] = []
        town_frames[town].append((dataset_name, old_frame_id, img_path))

    # Sort towns alphabetically for consistent ordering
    sorted_towns = sorted(town_frames.keys())

    print(f"  Found {len(sorted_towns)} towns:")
    for town in sorted_towns:
        print(f"    {town}: {len(town_frames[town]):,} frames")

    # =================================================================
    # PHASE 3: Assign new frame IDs and copy images (grouped by town)
    # =================================================================
    print("\nPhase 3: Copying images grouped by town...")

    global_frame_id = 0
    frame_id_mapping: dict[tuple[str, int], int] = {}
    town_ranges: dict[str, tuple[int, int]] = {}  # town -> (start_id, end_id)

    for town in tqdm(sorted_towns, desc="Processing towns"):
        town_start_id = global_frame_id
        frames = town_frames[town]

        for dataset_name, old_frame_id, img_path in tqdm(
            frames, desc=f"Copying {town}", leave=False
        ):
            # Map (dataset_name, old_frame_id) -> new global frame_id
            frame_id_mapping[(dataset_name, old_frame_id)] = global_frame_id

            # Copy image with new name
            dst_path = os.path.join(rgb_dir, f"rgb_{global_frame_id:05d}.png")
            shutil.copy2(img_path, dst_path)

            global_frame_id += 1

        town_end_id = global_frame_id - 1
        town_ranges[town] = (town_start_id, town_end_id)

    print(f"\nTotal frames copied: {global_frame_id:,}")
    print("\nTown frame ID ranges:")
    for town in sorted_towns:
        start_id, end_id = town_ranges[town]
        count = end_id - start_id + 1
        print(f"  {town}: {start_id:,} - {end_id:,} ({count:,} frames)")

    # =================================================================
    # PHASE 4: Update frame IDs in combined DataFrame
    # =================================================================
    print("\nPhase 4: Updating frame IDs in CSV...")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Update frame_ids using the mapping
        # Use -1 as sentinel for unmapped entries (should not happen)
        def map_frame_id(row: pd.Series) -> int:
            key = (row["_dataset_name"], row["frame_id"])
            if key in frame_id_mapping:
                return frame_id_mapping[key]
            else:
                # This should not happen if data is consistent
                print(
                    f"  WARNING: No mapping for {key}, "
                    "row will be dropped"
                )
                return -1

        combined_df["frame_id"] = combined_df.apply(map_frame_id, axis=1)

        # Drop rows that couldn't be mapped (shouldn't happen normally)
        unmapped_count = (combined_df["frame_id"] == -1).sum()
        if unmapped_count > 0:
            print(f"  WARNING: Dropping {unmapped_count} rows without mapping")
            combined_df = combined_df[combined_df["frame_id"] != -1]

        # Remove the temporary column
        combined_df = combined_df.drop(columns=["_dataset_name"])
        # Sort by frame_id so CSV is ordered by town groups
        combined_df = combined_df.sort_values("frame_id").reset_index(drop=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df


def save_combined_csv(combined_df: pd.DataFrame, dest_base: str) -> None:
    """Save the combined DataFrame to CSV.

    Args:
        combined_df: DataFrame with all combined poses.
        dest_base: Destination directory for combined dataset.
    """
    combined_csv_path = os.path.join(dest_base, "poses.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"\nCombined CSV saved to: {combined_csv_path}")
    print(f"Total detection rows: {len(combined_df):,}")
    print(f"Unique frame_ids: {combined_df['frame_id'].nunique():,}")
    if len(combined_df) > 0:
        frame_min = combined_df["frame_id"].min()
        frame_max = combined_df["frame_id"].max()
        print(f"Frame ID range: {frame_min} - {frame_max}")


def verify_combined_dataset(dest_base: str, combined_df: pd.DataFrame) -> None:
    """Verify the combined dataset integrity.

    Args:
        dest_base: Destination directory containing combined dataset.
        combined_df: DataFrame with all combined poses.
    """
    rgb_dir = os.path.join(dest_base, "rgb")
    dest_images = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
    dest_images_sorted = sorted(dest_images)

    print("\nVerification:")
    print(f"  Images in destination: {len(dest_images):,}")
    print(f"  CSV rows: {len(combined_df):,}")
    print(f"  Unique frames in CSV: {combined_df['frame_id'].nunique():,}")

    if dest_images_sorted:
        print(f"  First image: {dest_images_sorted[0]}")
        print(f"  Last image: {dest_images_sorted[-1]}")

    csv_frame_ids = set(combined_df["frame_id"].unique())
    image_frame_ids = set()
    for img in dest_images:
        frame_id = int(img.replace("rgb_", "").replace(".png", ""))
        image_frame_ids.add(frame_id)

    missing_in_images = csv_frame_ids - image_frame_ids
    missing_in_csv = image_frame_ids - csv_frame_ids

    if missing_in_images:
        print(
            f"  WARNING: {len(missing_in_images)} frame_ids in CSV "
            "but no image file"
        )
    if missing_in_csv:
        print(f"  WARNING: {len(missing_in_csv)} images without CSV entries")

    if not missing_in_images and not missing_in_csv:
        print("  ✓ All images have corresponding CSV entries")


def main() -> None:
    """Main entry point for the dataset combiner."""
    parser = argparse.ArgumentParser(
        description="Combine fragmented CARLA datasets into a unified dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/storage/remote/atcremers45/s0050/carla_data_3d_detector",
        help="Source directory containing fragmented dataset folders",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/storage/remote/atcremers45/s0050/carla_dataset",
        help="Destination directory for combined dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and report, don't copy files",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CARLA Dataset Combiner")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Destination: {args.dest}")
    print()

    dataset_info = scan_datasets(args.source)

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        return

    combined_df = combine_datasets(dataset_info, args.dest)
    save_combined_csv(combined_df, args.dest)
    verify_combined_dataset(args.dest, combined_df)

    print("\n" + "=" * 60)
    print("Dataset combination complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
