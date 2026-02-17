#!/usr/bin/env python3
"""
Assemble a sequence of images into a video.
"""

import argparse
from pathlib import Path
import cv2
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn a directory of images into a video."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        default="/usr/prakt/s0050/ravp/inference_output/spectator",
        help="Directory containing .jpg/.png frames"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=Path,
        default="/usr/prakt/s0050/ravp/inference_output/spectator.mp4",
        help="Output video file (e.g. out.mp4)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=10.0,
        help="Frame rate of the output video (default: 20)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Gather all .jpg/.png files, sorted by name
    img_paths = sorted(
        [*args.input_dir.glob("*.jpg"), *args.input_dir.glob("*.png")]
    )
    if not img_paths:
        print(f"No .jpg or .png images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Read first image to get size
    first_frame = cv2.imread(str(img_paths[0]))
    if first_frame is None:
        print(f"Could not read image: {img_paths[0]}", file=sys.stderr)
        sys.exit(1)

    height, width, channels = first_frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter
    # 'mp4v' for .mp4 output; change to 'XVID' for .avi if you prefer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(args.output_file),
        fourcc,
        args.fps,
        size
    )

    # Write each frame
    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: skipping unreadable image {img_path}", file=sys.stderr)
            continue
        # Ensure same size
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"Video written to {args.output_file}")


if __name__ == "__main__":
    main()
