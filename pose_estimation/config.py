"""Configuration for mask-conditioned target pose learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class PoseEstimationConfig:
    """Runtime configuration for training a target pose model."""

    dataset_root: str = ""
    label_source: str = "gt"
    output_dir: str = "pose_estimation_runs"
    experiment_name: str = "mask_conditioned_pose"

    image_size: Tuple[int, int] = (256, 256)
    backbone: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.1

    batch_size: int = 64
    num_epochs: int = 40
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    num_workers: int = 8
    grad_clip_norm: float = 1.0
    log_every_steps: int = 25

    device: str = "cuda"
    use_amp: bool = True
    channels_last: bool = True

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    require_follow_valid: bool = True
    min_mask_area_px: int = 0

    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0
    max_test_batches: int = 0

    dx_loss_weight: float = 1.0
    dy_loss_weight: float = 1.0
    yaw_loss_weight: float = 1.0

    wandb_project: str = "ravp-pose"
    wandb_run_name: str = ""
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if self.label_source not in {"gt", "pred"}:
            raise ValueError("label_source must be one of: 'gt', 'pred'")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")

    @property
    def csv_name(self) -> str:
        return "gt_poses.csv" if self.label_source == "gt" else "pred_poses.csv"

    @property
    def target_names(self) -> Tuple[str, str, str]:
        return ("dx_m", "dy_m", "yaw_follow_deg")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PoseEstimationConfig":
        data = dict(payload)
        if "image_size" in data:
            data["image_size"] = tuple(data["image_size"])
        return cls(**data)
