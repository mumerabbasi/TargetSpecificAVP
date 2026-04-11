"""Model and losses for target pose regression."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import TargetPoseTrainingConfig
from .dataset import TranslationStats, denormalize_translation


class TargetPoseRegressor(nn.Module):
    """Predict dx, dy, and yaw_follow from full-frame and local target views."""

    def __init__(self, config: TargetPoseTrainingConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone, feature_dim = _build_convnext_backbone(
            backbone_name=config.backbone,
            pretrained=config.pretrained,
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        fusion_dim = feature_dim * 3 + 128
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1536),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.translation_head = nn.Linear(512, 2)
        self.yaw_head = nn.Linear(512, 2)

    def forward(
        self,
        full_input: torch.Tensor,
        crop_input: torch.Tensor,
        geometry: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_features = self.backbone(full_input)
        crop_features = self.backbone(crop_input)
        geometry_features = self.geometry_encoder(geometry)

        fused = torch.cat(
            [
                full_features,
                crop_features,
                torch.abs(full_features - crop_features),
                geometry_features,
            ],
            dim=1,
        )
        fused = self.fusion_head(fused)

        translation = self.translation_head(fused)
        yaw_vector = F.normalize(self.yaw_head(fused), dim=1, eps=1e-6)
        return {
            "translation": translation,
            "yaw_vector": yaw_vector,
        }


class TargetPoseLoss(nn.Module):
    """Joint regression loss for translation and follow yaw."""

    def __init__(self, config: TargetPoseTrainingConfig) -> None:
        super().__init__()
        self.dx_loss_weight = config.dx_loss_weight
        self.dy_loss_weight = config.dy_loss_weight
        self.yaw_loss_weight = config.yaw_loss_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        translation_target: torch.Tensor,
        yaw_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        translation_loss = F.smooth_l1_loss(
            outputs["translation"],
            translation_target,
            reduction="none",
        )
        dx_loss = translation_loss[:, 0].mean()
        dy_loss = translation_loss[:, 1].mean()

        yaw_alignment = F.cosine_similarity(outputs["yaw_vector"], yaw_target, dim=1)
        yaw_loss = (1.0 - yaw_alignment).mean()

        total_loss = (
            self.dx_loss_weight * dx_loss
            + self.dy_loss_weight * dy_loss
            + self.yaw_loss_weight * yaw_loss
        )
        return total_loss, {
            "loss_total": float(total_loss.detach().item()),
            "loss_dx": float(dx_loss.detach().item()),
            "loss_dy": float(dy_loss.detach().item()),
            "loss_yaw": float(yaw_loss.detach().item()),
        }


def decode_predictions(
    outputs: Dict[str, torch.Tensor],
    translation_stats: TranslationStats,
) -> Dict[str, torch.Tensor]:
    """Decode normalized model outputs into metric-space predictions."""
    translation = denormalize_translation(outputs["translation"], translation_stats)
    yaw_rad = torch.atan2(outputs["yaw_vector"][:, 0], outputs["yaw_vector"][:, 1])
    yaw_follow_deg = torch.rad2deg(yaw_rad)
    return {
        "translation": translation,
        "yaw_follow_deg": yaw_follow_deg,
    }


def compute_pose_metrics(
    predicted_translation: torch.Tensor,
    predicted_yaw_deg: torch.Tensor,
    target_translation: torch.Tensor,
    target_yaw_deg: torch.Tensor,
) -> Dict[str, float]:
    """Compute human-readable regression metrics."""
    translation_error = (predicted_translation - target_translation).abs()
    yaw_error = wrap_angle_deg(predicted_yaw_deg - target_yaw_deg).abs()
    selection_score = (
        translation_error[:, 0].mean()
        + translation_error[:, 1].mean()
        + 0.05 * yaw_error.mean()
    )
    return {
        "mae_dx_m": float(translation_error[:, 0].mean().item()),
        "mae_dy_m": float(translation_error[:, 1].mean().item()),
        "mae_yaw_follow_deg": float(yaw_error.mean().item()),
        "selection_score": float(selection_score.item()),
    }


def wrap_angle_deg(angle_deg: torch.Tensor) -> torch.Tensor:
    """Wrap angles to the [-180, 180] range."""
    return torch.remainder(angle_deg + 180.0, 360.0) - 180.0


class _ConvNeXtFeatureExtractor(nn.Module):
    """Shared ConvNeXt feature extractor with a 4-channel input stem."""

    def __init__(self, model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.avgpool(features)
        return torch.flatten(pooled, start_dim=1)


def _build_convnext_backbone(
    backbone_name: str,
    pretrained: bool,
) -> Tuple[nn.Module, int]:
    if backbone_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_base(weights=weights)
    elif backbone_name == "convnext_large":
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_large(weights=weights)
    else:
        raise ValueError(
            "Unsupported backbone. Expected one of: convnext_base, convnext_large"
        )

    stem_conv = model.features[0][0]
    model.features[0][0] = _expand_input_conv(stem_conv, in_channels=4)
    feature_dim = int(model.classifier[2].in_features)
    return _ConvNeXtFeatureExtractor(model, feature_dim), feature_dim


def _expand_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapt a pretrained RGB convolution to additional input channels."""
    expanded = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        expanded.weight[:, :3] = conv.weight
        if in_channels > 3:
            mean_kernel = conv.weight.mean(dim=1, keepdim=True)
            expanded.weight[:, 3:in_channels] = mean_kernel.repeat(
                1, in_channels - 3, 1, 1
            )
        if conv.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(conv.bias)

    return expanded
