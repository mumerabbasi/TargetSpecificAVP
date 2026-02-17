"""Training script for pose estimation CNN."""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PoseEstimationConfig
from .dataset import (
    PoseEstimationDataset,
    compute_pose_statistics,
    create_data_splits,
)
from .model import (
    PoseEstimationCNN,
    PoseEstimationLoss,
    PoseEstimationMetrics,
    create_model,
)

# Optimize CUDA performance
cudnn.benchmark = True  # Auto-tune convolution algorithms
cudnn.deterministic = False  # Allow non-deterministic for speed


class Trainer:
    """Trainer class for pose estimation model.

    Handles training loop, validation, logging, and checkpointing.

    Attributes:
        config: Configuration object.
        model: Pose estimation model.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
    """

    def __init__(
        self,
        config: PoseEstimationConfig,
        model: PoseEstimationCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        """Initialize the trainer.

        Args:
            config: Configuration object.
            model: Pose estimation model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Optional test data loader.
            use_wandb: Whether to use Weights & Biases logging.
        """
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss and metrics
        self.criterion = PoseEstimationLoss(config).to(self.device)
        self.metrics = PoseEstimationMetrics(config)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "learning_rate": [],
        }

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(
            config.output_dir,
            f"{config.experiment_name}_{timestamp}"
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save config
        self._save_config()

        # Initialize wandb
        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            return

        # Build config dict for wandb
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        for k, v in config_dict.items():
            if isinstance(v, tuple):
                config_dict[k] = list(v)

        # Add additional info
        config_dict["device"] = str(self.device)
        config_dict["num_train_samples"] = len(self.train_loader.dataset)
        config_dict["num_val_samples"] = len(self.val_loader.dataset)
        if self.test_loader is not None:
            config_dict["num_test_samples"] = len(self.test_loader.dataset)
        config_dict["num_parameters"] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Initialize wandb run
        wandb.init(
            project="pose-estimation",
            name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            dir=self.exp_dir,
            tags=[
                self.config.backbone,
                self.config.bbox_mode,
                "gt" if self.config.use_gt_poses else "pred",
            ],
        )

        # Watch model for gradient logging
        wandb.watch(self.model, log="all", log_freq=100)

    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        elif self.config.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_gamma,
                patience=5,
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {self.config.lr_scheduler}"
            )

    def _save_config(self) -> None:
        """Save configuration to experiment directory."""
        config_path = os.path.join(self.exp_dir, "config.json")
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        # Convert tuples to lists for JSON serialization
        for k, v in config_dict.items():
            if isinstance(v, tuple):
                config_dict[k] = list(v)

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            image = batch["image"].to(self.device)
            pose = batch["pose"].to(self.device)

            # Get bbox if needed
            bbox_normalized = None
            if self.config.bbox_mode in ["numeric", "both"]:
                bbox_normalized = batch["bbox_normalized"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(image, bbox_normalized)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, pose)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Log batch metrics to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_mse_dx": loss_dict.get("mse_dx", 0),
                        "train/batch_mse_dy": loss_dict.get("mse_dy", 0),
                        "train/batch_mse_dz": loss_dict.get("mse_dz", 0),
                        "train/batch_mse_yaw": loss_dict.get("mse_yaw", 0),
                        "global_step": self.global_step,
                    })

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, dict]:
        """Validate the model.

        Args:
            loader: Data loader for validation.

        Returns:
            Tuple of (average loss, metrics dictionary).
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = len(loader)

        for batch in tqdm(loader, desc="Validating"):
            # Move data to device
            image = batch["image"].to(self.device)
            pose = batch["pose"].to(self.device)
            pose_raw = batch["pose_raw"].to(self.device)

            # Get bbox if needed
            bbox_normalized = None
            if self.config.bbox_mode in ["numeric", "both"]:
                bbox_normalized = batch["bbox_normalized"].to(self.device)

            # Forward pass
            predictions = self.model(image, bbox_normalized)

            # Compute loss
            loss, _ = self.criterion(predictions, pose)
            total_loss += loss.item()

            # Denormalize predictions for metrics
            pred_raw = self.train_loader.dataset.denormalize_pose(predictions)
            self.metrics.update(pred_raw, pose_raw)

        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def train(self) -> None:
        """Run the full training loop."""
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Experiment directory: {self.exp_dir}")

        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time

            # Validate
            val_loss, val_metrics = self.validate(self.val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.config.lr_scheduler == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            self.history["learning_rate"].append(current_lr)

            # Print summary
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"LR:         {current_lr:.6f}")
            print(f"Time:       {train_time:.1f}s")

            print("\nValidation Metrics:")
            for name in self.config.output_names:
                mae = val_metrics.get(f"mae_{name}", 0)
                rmse = val_metrics.get(f"rmse_{name}", 0)
                unit = "deg" if name == "yaw" else "m"
                print(f"  {name}: MAE={mae:.4f}{unit}, RMSE={rmse:.4f}{unit}")

            # Log epoch metrics to wandb
            if self.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch_time_seconds": train_time,
                }
                # Add per-component metrics
                for name in self.config.output_names:
                    mae = val_metrics.get(f"mae_{name}", 0)
                    rmse = val_metrics.get(f"rmse_{name}", 0)
                    wandb_log[f"val/mae_{name}"] = mae
                    wandb_log[f"val/rmse_{name}"] = rmse
                # Add overall metrics
                wandb_log["val/mae_total"] = val_metrics.get("mae_total", 0)
                wandb_log["val/rmse_total"] = val_metrics.get("rmse_total", 0)
                wandb_log["best_val_loss"] = self.best_val_loss
                wandb.log(wandb_log)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print("\n*** New best model saved! ***")
            else:
                self.epochs_without_improvement += 1
                if not self.config.save_best_only:
                    self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if (self.epochs_without_improvement >=
                    self.config.early_stopping_patience):
                print(f"\nEarly stopping after {epoch + 1} epochs.")
                break

        # Save training history
        self._save_history()

        # Final evaluation on test set
        if self.test_loader is not None:
            print("\n" + "=" * 60)
            print("Final Evaluation on Test Set")
            print("=" * 60)
            self._load_best_model()
            test_loss, test_metrics = self.validate(self.test_loader)
            print(f"\nTest Loss: {test_loss:.4f}")
            print("\nTest Metrics:")
            for name in self.config.output_names:
                mae = test_metrics.get(f"mae_{name}", 0)
                rmse = test_metrics.get(f"rmse_{name}", 0)
                unit = "deg" if name == "yaw" else "m"
                print(f"  {name}: MAE={mae:.4f}{unit}, RMSE={rmse:.4f}{unit}")

            # Save test results
            test_results = {
                "test_loss": test_loss,
                "test_metrics": test_metrics
            }
            results_path = os.path.join(self.exp_dir, "test_results.json")
            with open(results_path, "w") as f:
                json.dump(test_results, f, indent=2)

            # Log test results to wandb
            if self.use_wandb:
                wandb_log = {"test/loss": test_loss}
                for name in self.config.output_names:
                    mae = test_metrics.get(f"mae_{name}", 0)
                    rmse = test_metrics.get(f"rmse_{name}", 0)
                    wandb_log[f"test/mae_{name}"] = mae
                    wandb_log[f"test/rmse_{name}"] = rmse
                wandb_log["test/mae_total"] = test_metrics.get("mae_total", 0)
                wandb_log["test/rmse_total"] = test_metrics.get("rmse_total", 0)
                wandb.log(wandb_log)

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if is_best:
            path = os.path.join(self.exp_dir, "best_model.pth")
        else:
            path = os.path.join(
                self.exp_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )

        torch.save(checkpoint, path)

        # Log model artifact to wandb
        '''if self.use_wandb and is_best:
            artifact = wandb.Artifact(
                name=f"model-{self.config.experiment_name}",
                type="model",
                description=f"Best model checkpoint at epoch {epoch + 1}",
                metadata={
                    "epoch": epoch,
                    "val_loss": self.best_val_loss,
                    "backbone": self.config.backbone,
                    "bbox_mode": self.config.bbox_mode,
                }
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)'''

    def _load_best_model(self) -> None:
        """Load the best model checkpoint."""
        path = os.path.join(self.exp_dir, "best_model.pth")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = os.path.join(self.exp_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def main(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    # Create config - args override defaults
    config = PoseEstimationConfig(
        # Paths (from args)
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        # Model settings (from args)
        backbone=args.backbone,
        bbox_mode=args.bbox_mode,
        # Training settings (from args)
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        # Pose source (from args)
        use_gt_poses=args.use_gt,
        # Experiment name (from args)
        experiment_name=args.experiment_name,
    )

    print("Configuration:")
    print(f"  CSV path:     {config.csv_path}")
    print(f"  Images dir:   {config.images_dir}")
    print(f"  Output dir:   {config.output_dir}")
    print(f"  Backbone:     {config.backbone}")
    print(f"  Bbox mode:    {config.bbox_mode}")
    print(f"  Use GT poses: {config.use_gt_poses}")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  Epochs:       {config.num_epochs}")
    print(f"  LR:           {config.learning_rate}")
    print(f"  Outputs:      {config.output_names}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(config.csv_path)
    print(f"Loaded {len(df)} samples")

    # Compute pose statistics
    pose_mean, pose_std = compute_pose_statistics(
        df, use_gt=config.use_gt_poses
    )
    config.pose_mean = pose_mean
    config.pose_std = pose_std
    print("\nPose statistics:")
    print(f"  Mean: {pose_mean}")
    print(f"  Std:  {pose_std}")

    # Create data splits
    train_df, val_df, test_df = create_data_splits(df, config)

    # Create datasets
    train_dataset = PoseEstimationDataset(
        config, train_df, is_training=True
    )
    val_dataset = PoseEstimationDataset(
        config, val_df, is_training=False
    )
    test_dataset = PoseEstimationDataset(
        config, test_df, is_training=False
    )

    # Create data loaders with optimized settings for faster I/O
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Initialize wandb with API key
    if args.wandb:
        os.environ["WANDB_API_KEY"] = "7c5329ecdd14418cc7e621e369822111fc474e08"
        print("\nWandB logging enabled")

    # Create trainer and train
    trainer = Trainer(
        config,
        model,
        train_loader,
        val_loader,
        test_loader,
        use_wandb=args.wandb,
    )
    trainer.train()

    print("\nTraining complete!")
    print(f"Results saved to: {trainer.exp_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train pose estimation CNN model"
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default=(
            "/storage/remote/atcremers45/s0050/carla_dataset/poses_filtered.csv"
        ),
        help="Path to CSV file with pose annotations",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/storage/remote/atcremers45/s0050/carla_dataset/rgb",
        help="Directory containing RGB images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/usr/prakt/s0050/ravp/pose_estimation_runs"
        ),
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--bbox-mode",
        type=str,
        default="mask",
        choices=["crop", "mask", "numeric", "both"],
        help="How to incorporate bounding box information",
    )
    parser.add_argument(
        "--use-gt",
        action="store_true",
        default=False,
        help="Use ground truth poses (default: False)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pose_estimation",
        help="Name for this experiment",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (default: False)",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
