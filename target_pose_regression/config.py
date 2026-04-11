"""Configuration for target pose regression training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class TargetPoseTrainingConfig:
    """Runtime configuration for target pose regression."""

    dataset_root: str = ""
    label_source: str = "pred"
    output_dir: str = "target_pose_runs"
    experiment_name: str = "target_pose_regressor"

    image_size: Tuple[int, int] = (384, 384)
    crop_size: Tuple[int, int] = (256, 256)
    crop_context_scale: float = 2.0

    backbone: str = "convnext_base"
    pretrained: bool = True
    dropout: float = 0.2

    batch_size: int = 16
    num_epochs: int = 40
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    num_workers: int = 8
    grad_clip_norm: float = 1.0
    log_every_steps: int = 25

    device: str = "cuda"
    use_amp: bool = True
    channels_last: bool = True

    train_ratio: float = 0.85
    val_ratio: float = 0.1
    test_ratio: float = 0.05
    random_seed: int = 42

    require_follow_valid: bool = True
    min_mask_area_px: int = 120

    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0
    max_test_batches: int = 0

    dx_loss_weight: float = 1.0
    dy_loss_weight: float = 1.0
    yaw_loss_weight: float = 2.0

    wandb_project: str = "ravp-target-pose"
    wandb_run_name: str = ""
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if self.label_source not in {"gt", "pred"}:
            raise ValueError("label_source must be one of: 'gt', 'pred'")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            )

        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError("crop_size must be positive")
        if self.crop_context_scale < 1.0:
            raise ValueError("crop_context_scale must be >= 1.0")

    @property
    def csv_name(self) -> str:
        return "gt_poses.csv" if self.label_source == "gt" else "pred_poses.csv"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = list(self.image_size)
        payload["crop_size"] = list(self.crop_size)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetPoseTrainingConfig":
        data = dict(payload)
        if "image_size" in data:
            data["image_size"] = tuple(data["image_size"])
        if "crop_size" in data:
            data["crop_size"] = tuple(data["crop_size"])
        return cls(**data)
