"""CNN model for pose estimation from RGB images and bounding boxes."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from .config import PoseEstimationConfig


class PoseEstimationCNN(nn.Module):
    """CNN model for estimating vehicle pose from RGB images.

    This model uses a pretrained backbone (ResNet) and adds a pose
    regression head. It supports different bbox input modes:
    - crop: Image is cropped to bbox before feeding to network
    - mask: Bbox mask is concatenated as 4th channel
    - numeric: Bbox coordinates are concatenated to features
    - both: Combines mask and numeric modes

    Attributes:
        config: Configuration object.
        backbone: Feature extraction backbone.
        pose_head: Regression head for pose prediction.
    """

    def __init__(self, config: PoseEstimationConfig):
        """Initialize the model.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config

        # Determine input channels based on bbox_mode
        if config.bbox_mode in ["mask", "both"]:
            in_channels = 4  # RGB + mask
        else:
            in_channels = 3  # RGB only

        # Create backbone
        self.backbone, feature_dim = self._create_backbone(
            config.backbone,
            config.pretrained,
            in_channels
        )

        # Optionally freeze backbone
        if config.freeze_backbone:
            self._freeze_backbone()

        # Determine additional input features
        extra_features = 0
        if config.bbox_mode in ["numeric", "both"]:
            extra_features = 4  # bbox coordinates

        # Create pose regression head
        self.pose_head = self._create_pose_head(
            feature_dim + extra_features,
            config.num_outputs
        )

    def _create_backbone(
        self,
        backbone_name: str,
        pretrained: bool,
        in_channels: int
    ) -> Tuple[nn.Module, int]:
        """Create the feature extraction backbone.

        Args:
            backbone_name: Name of the backbone architecture.
            pretrained: Whether to use pretrained weights.
            in_channels: Number of input channels.

        Returns:
            Tuple of (backbone module, feature dimension).
        """
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Modify first conv layer if input channels != 3
        if in_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Copy weights for RGB channels
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = original_conv.weight
                # Initialize mask channel weights
                if in_channels > 3:
                    nn.init.kaiming_normal_(
                        backbone.conv1.weight[:, 3:],
                        mode="fan_out",
                        nonlinearity="relu"
                    )

        # Remove the final classification layer
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        return backbone, feature_dim

    def _create_pose_head(
        self,
        in_features: int,
        num_outputs: int
    ) -> nn.Module:
        """Create the pose regression head.

        Args:
            in_features: Number of input features.
            num_outputs: Number of output dimensions.

        Returns:
            Pose regression head module.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs),
        )

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        image: torch.Tensor,
        bbox_normalized: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: Input image tensor [B, C, H, W].
                   C=3 for crop/numeric mode, C=4 for mask/both mode.
            bbox_normalized: Normalized bbox coordinates [B, 4] for
                           numeric/both mode. Optional.

        Returns:
            Predicted pose tensor [B, num_outputs].
        """
        # Extract features from backbone
        features = self.backbone(image)
        features = features.view(features.size(0), -1)

        # Concatenate bbox coordinates if using numeric mode
        if self.config.bbox_mode in ["numeric", "both"]:
            if bbox_normalized is not None:
                features = torch.cat([features, bbox_normalized], dim=1)

        # Predict pose
        pose = self.pose_head(features)

        return pose


class PoseEstimationLoss(nn.Module):
    """Loss function for pose estimation.

    Computes weighted MSE loss for each pose component.

    Attributes:
        config: Configuration object.
        mse: MSE loss function.
    """

    def __init__(self, config: PoseEstimationConfig):
        """Initialize the loss function.

        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction="none")

        # Build loss weights based on which outputs are enabled
        weights = []
        if config.predict_dx:
            weights.append(config.loss_weight_dx)
        if config.predict_dy:
            weights.append(config.loss_weight_dy)
        if config.predict_dz:
            weights.append(config.loss_weight_dz)
        if config.predict_yaw:
            weights.append(config.loss_weight_yaw)

        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the loss.

        Args:
            predictions: Predicted pose tensor [B, num_outputs].
            targets: Target pose tensor [B, num_outputs].

        Returns:
            Tuple of (total_loss, loss_dict with individual losses).
        """
        # Compute MSE for each dimension
        mse_per_dim = self.mse(predictions, targets).mean(dim=0)

        # Apply weights
        weighted_losses = mse_per_dim * self.weights

        # Total loss
        total_loss = weighted_losses.sum()

        # Build loss dict for logging
        loss_dict = {"total": total_loss.item()}
        output_names = self.config.output_names

        for i, name in enumerate(output_names):
            loss_dict[f"mse_{name}"] = mse_per_dim[i].item()
            loss_dict[f"weighted_{name}"] = weighted_losses[i].item()

        return total_loss, loss_dict


class PoseEstimationMetrics:
    """Metrics for pose estimation evaluation.

    Computes MAE and RMSE for each pose component in original units.
    """

    def __init__(self, config: PoseEstimationConfig):
        """Initialize metrics.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> None:
        """Add predictions and targets to accumulator.

        Args:
            predictions: Predicted pose tensor [B, num_outputs].
            targets: Target pose tensor [B, num_outputs].
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        """Compute final metrics.

        Returns:
            Dictionary with MAE and RMSE for each component.
        """
        if not self.predictions:
            return {}

        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)

        # Compute errors
        errors = predictions - targets
        abs_errors = torch.abs(errors)
        squared_errors = errors ** 2

        metrics = {}
        output_names = self.config.output_names
        units = self._get_units()

        for i, name in enumerate(output_names):
            mae = abs_errors[:, i].mean().item()
            rmse = torch.sqrt(squared_errors[:, i].mean()).item()
            unit = units[name]

            metrics[f"mae_{name}"] = mae
            metrics[f"rmse_{name}"] = rmse
            metrics[f"mae_{name}_{unit}"] = mae
            metrics[f"rmse_{name}_{unit}"] = rmse

        # Overall metrics
        metrics["mae_total"] = abs_errors.mean().item()
        metrics["rmse_total"] = torch.sqrt(squared_errors.mean()).item()

        return metrics

    def _get_units(self) -> dict:
        """Get units for each output dimension.

        Returns:
            Dictionary mapping output names to units.
        """
        return {
            "dx": "m",
            "dy": "m",
            "dz": "m",
            "yaw": "deg",
        }


def create_model(config: PoseEstimationConfig) -> PoseEstimationCNN:
    """Create a pose estimation model.

    Args:
        config: Configuration object.

    Returns:
        Initialized model.
    """
    model = PoseEstimationCNN(config)
    return model
