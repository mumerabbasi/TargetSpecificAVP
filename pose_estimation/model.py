"""Mask-conditioned CNN for target pose prediction."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import PoseEstimationConfig
from .dataset import TranslationStats, denormalize_translation


class PoseEstimationCNN(nn.Module):
    """Predict target translation and pursuit yaw from RGB plus mask."""

    def __init__(self, config: PoseEstimationConfig) -> None:
        super().__init__()
        self.config = config

        backbone, feature_dim = _build_resnet_backbone(
            backbone_name=config.backbone,
            pretrained=config.pretrained,
        )
        self.stem = backbone["stem"]
        self.layer1 = backbone["layer1"]
        self.layer2 = backbone["layer2"]
        self.layer3 = backbone["layer3"]
        self.layer4 = backbone["layer4"]

        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2 + 64, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
        )
        self.translation_head = nn.Linear(256, 2)
        self.yaw_head = nn.Linear(256, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        geometry: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)

        global_features = F.adaptive_avg_pool2d(
            features, output_size=1).flatten(1)
        target_features = _masked_average_pool(
            features, masks, global_features)
        geometry_features = self.geometry_encoder(geometry)

        fused = torch.cat(
            [global_features, target_features, geometry_features], dim=1)
        fused = self.fusion_head(fused)

        translation = self.translation_head(fused)
        yaw_vector = F.normalize(self.yaw_head(fused), dim=1)
        return {
            "translation": translation,
            "yaw_vector": yaw_vector,
        }


class PoseEstimationLoss(nn.Module):
    """Joint translation and pursuit-yaw loss."""

    def __init__(self, config: PoseEstimationConfig) -> None:
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
        yaw_alignment = F.cosine_similarity(
            outputs["yaw_vector"], yaw_target, dim=1)
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
    translation = denormalize_translation(
        outputs["translation"], translation_stats)
    yaw_rad = torch.atan2(outputs["yaw_vector"]
                          [:, 0], outputs["yaw_vector"][:, 1])
    yaw_deg = torch.rad2deg(yaw_rad)
    return {
        "translation": translation,
        "yaw_follow_deg": yaw_deg,
    }


def compute_pose_metrics(
    predicted_translation: torch.Tensor,
    predicted_yaw_deg: torch.Tensor,
    target_translation: torch.Tensor,
    target_yaw_deg: torch.Tensor,
) -> Dict[str, float]:
    translation_error = (predicted_translation - target_translation).abs()
    yaw_error = wrap_angle_deg(predicted_yaw_deg - target_yaw_deg).abs()
    yaw_axis_error = torch.minimum(yaw_error, (180.0 - yaw_error).abs())
    return {
        "mae_dx_m": float(translation_error[:, 0].mean().item()),
        "mae_dy_m": float(translation_error[:, 1].mean().item()),
        "mae_yaw_follow_deg": float(yaw_error.mean().item()),
        "mae_yaw_axis_deg": float(yaw_axis_error.mean().item()),
    }


def wrap_angle_deg(angle_deg: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle_deg + 180.0, 360.0) - 180.0


def _build_resnet_backbone(
    backbone_name: str,
    pretrained: bool,
) -> tuple[Dict[str, nn.Module], int]:
    weights_lookup = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    }
    if backbone_name not in weights_lookup:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    builder = getattr(models, backbone_name)
    weights = weights_lookup[backbone_name] if pretrained else None
    resnet = builder(weights=weights)

    original_conv = resnet.conv1
    resnet.conv1 = nn.Conv2d(
        4,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        resnet.conv1.weight[:, :3] = original_conv.weight
        resnet.conv1.weight[:, 3:4] = original_conv.weight.mean(
            dim=1, keepdim=True)

    feature_dim = resnet.fc.in_features
    return ({"stem": nn.Sequential(resnet.conv1,
                                   resnet.bn1,
                                   resnet.relu,
                                   resnet.maxpool),
             "layer1": resnet.layer1,
             "layer2": resnet.layer2,
             "layer3": resnet.layer3,
             "layer4": resnet.layer4,
             },
            feature_dim,
            )


def _masked_average_pool(
    features: torch.Tensor,
    masks: torch.Tensor,
    global_features: torch.Tensor,
) -> torch.Tensor:
    resized_masks = F.interpolate(
        masks,
        size=features.shape[-2:],
        mode="nearest",
    )
    weighted_sum = (features * resized_masks).sum(dim=(2, 3))
    mask_sum = resized_masks.sum(dim=(2, 3)).clamp_min(1.0)
    masked_features = weighted_sum / mask_sum

    empty_mask = resized_masks.sum(dim=(2, 3)).squeeze(1) <= 0
    if empty_mask.any():
        masked_features = masked_features.clone()
        masked_features[empty_mask] = global_features[empty_mask].to(
            masked_features.dtype)
    return masked_features
