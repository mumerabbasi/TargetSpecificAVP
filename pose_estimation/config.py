"""Configuration for pose estimation CNN training."""

from dataclasses import dataclass
from typing import List, Tuple
import os


@dataclass
class PoseEstimationConfig:
    """Configuration for pose estimation model training.

    Note: Runtime settings like paths, batch_size, num_epochs, learning_rate,
    backbone, bbox_mode, use_gt_poses, and experiment_name are passed via
    command line arguments in train.py, not set here.
    """

    # -------------------------------------------------------------------------
    # Runtime settings (set via command line args in train.py)
    # -------------------------------------------------------------------------
    csv_path: str = ""
    images_dir: str = ""
    output_dir: str = ""
    backbone: str = "resnet50"
    bbox_mode: str = "mask"
    use_gt_poses: bool = True
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    experiment_name: str = "pose_estimation"

    # -------------------------------------------------------------------------
    # Data split
    # -------------------------------------------------------------------------
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # -------------------------------------------------------------------------
    # Image settings
    # -------------------------------------------------------------------------
    image_size: Tuple[int, int] = (224, 224)  # Resize for model input
    original_image_size: Tuple[int, int] = (1024, 1024)

    # -------------------------------------------------------------------------
    # Model settings
    # -------------------------------------------------------------------------
    pretrained: bool = True
    freeze_backbone: bool = False

    # -------------------------------------------------------------------------
    # Target settings
    # -------------------------------------------------------------------------
    # Which pose components to predict
    predict_dx: bool = True
    predict_dy: bool = True
    predict_dz: bool = False  # Not predicting dz
    predict_yaw: bool = True

    # -------------------------------------------------------------------------
    # Normalization statistics (computed from dataset at runtime)
    # -------------------------------------------------------------------------
    pose_mean: Tuple[float, float, float, float] = (8.0, 0.0, -0.95, 0.0)
    pose_std: Tuple[float, float, float, float] = (2.5, 1.7, 0.1, 25.0)

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    num_workers: int = 8
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # Options: 'step', 'cosine', 'plateau'
    lr_step_size: int = 15
    lr_gamma: float = 0.1

    # -------------------------------------------------------------------------
    # Loss weights for multi-task learning
    # -------------------------------------------------------------------------
    loss_weight_dx: float = 1.0
    loss_weight_dy: float = 1.0
    loss_weight_dz: float = 1.0
    loss_weight_yaw: float = 0.5  # Lower weight since yaw is in degrees

    # -------------------------------------------------------------------------
    # Logging and checkpointing
    # -------------------------------------------------------------------------
    log_interval: int = 50  # Log every N batches
    save_best_only: bool = True

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    @property
    def num_outputs(self) -> int:
        """Get the number of output dimensions based on config."""
        return sum([
            self.predict_dx,
            self.predict_dy,
            self.predict_dz,
            self.predict_yaw
        ])

    @property
    def output_names(self) -> List[str]:
        """Get the names of output dimensions."""
        names = []
        if self.predict_dx:
            names.append("dx")
        if self.predict_dy:
            names.append("dy")
        if self.predict_dz:
            names.append("dz")
        if self.predict_yaw:
            names.append("yaw")
        return names
