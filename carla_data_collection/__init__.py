"""
CARLA 3D detection dataset collection package.

Uses YOLO + SAM2 for 2D detection/segmentation and CenterPoint for 3D detection.
"""

from .config import Config
from .collect import run_collection

__all__ = ["Config", "run_collection"]
