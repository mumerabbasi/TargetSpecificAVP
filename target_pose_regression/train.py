"""Training entry point for target pose regression."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from .config import TargetPoseTrainingConfig
from .dataset import (
    TargetPoseDataset,
    TranslationStats,
    build_group_splits,
    load_pose_rows,
    split_summary,
)
from .model import (
    TargetPoseLoss,
    TargetPoseRegressor,
    compute_pose_metrics,
    decode_predictions,
)


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train the target pose regression CNN",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-source", choices=["gt", "pred"], default="pred")
    parser.add_argument("--output-dir", default="target_pose_runs")
    parser.add_argument("--experiment-name", default="target_pose_regressor")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-context-scale", type=float, default=2.0)
    parser.add_argument(
        "--backbone",
        choices=["convnext_base", "convnext_large"],
        default="convnext_base",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-mask-area-px", type=int, default=120)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--dx-loss-weight", type=float, default=1.0)
    parser.add_argument("--dy-loss-weight", type=float, default=1.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=2.0)
    parser.add_argument("--wandb-project", default="ravp-target-pose")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the full training workflow."""
    args = parse_args()
    config = TargetPoseTrainingConfig(
        dataset_root=args.dataset_root,
        label_source=args.label_source,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        image_size=(args.image_size, args.image_size),
        crop_size=(args.crop_size, args.crop_size),
        crop_context_scale=args.crop_context_scale,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        log_every_steps=args.log_every_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        min_mask_area_px=args.min_mask_area_px,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
        dx_loss_weight=args.dx_loss_weight,
        dy_loss_weight=args.dy_loss_weight,
        yaw_loss_weight=args.yaw_loss_weight,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_enabled=not args.no_wandb,
    )

    set_random_seed(config.random_seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    rows = load_pose_rows(config)
    rows_by_split = build_group_splits(rows, config)
    translation_stats = TranslationStats.from_rows(rows_by_split["train"])

    run_dir = create_run_dir(config)
    save_json(run_dir / "config.json", config.to_dict())
    save_json(run_dir / "translation_stats.json", translation_stats.to_dict())
    save_json(run_dir / "split_summary.json", split_summary(rows_by_split))

    train_dataset = TargetPoseDataset(
        config=config,
        rows=rows_by_split["train"],
        translation_stats=translation_stats,
        training=True,
    )
    val_dataset = TargetPoseDataset(
        config=config,
        rows=rows_by_split["val"],
        translation_stats=translation_stats,
        training=False,
    )
    test_dataset = TargetPoseDataset(
        config=config,
        rows=rows_by_split["test"],
        translation_stats=translation_stats,
        training=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        training=True,
    )
    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        training=False,
    )
    test_loader = make_loader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        training=False,
    )

    device = select_device(config.device)
    model = TargetPoseRegressor(config).to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = TargetPoseLoss(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(config.use_amp and device.type == "cuda"),
    )

    wandb_run = init_wandb(
        config=config,
        run_dir=run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
    )

    best_selection_score = float("inf")
    history = []
    for epoch in range(1, config.num_epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            translation_stats=translation_stats,
            training=True,
            use_amp=config.use_amp,
            max_batches=config.max_train_batches,
            channels_last=config.channels_last,
            grad_clip_norm=config.grad_clip_norm,
            progress_label=f"train {epoch}/{config.num_epochs}",
            log_every_steps=config.log_every_steps,
            wandb_run=wandb_run,
            global_step_base=(epoch - 1) * max(len(train_loader), 1),
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            translation_stats=translation_stats,
            training=False,
            use_amp=config.use_amp,
            max_batches=config.max_val_batches,
            channels_last=config.channels_last,
            grad_clip_norm=0.0,
            progress_label=f"val {epoch}/{config.num_epochs}",
            log_every_steps=config.log_every_steps,
            wandb_run=None,
            global_step_base=0,
        )

        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_stats,
            "val": val_stats,
        }
        history.append(epoch_record)
        save_json(run_dir / "history.json", history)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "lr": epoch_record["learning_rate"],
                    "train/loss_total": train_stats["loss_total"],
                    "train/loss_dx": train_stats["loss_dx"],
                    "train/loss_dy": train_stats["loss_dy"],
                    "train/loss_yaw": train_stats["loss_yaw"],
                    "val/loss_total": val_stats["loss_total"],
                    "val/loss_dx": val_stats["loss_dx"],
                    "val/loss_dy": val_stats["loss_dy"],
                    "val/loss_yaw": val_stats["loss_yaw"],
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": config.to_dict(),
            "translation_stats": translation_stats.to_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_stats["selection_score"] < best_selection_score:
            best_selection_score = val_stats["selection_score"]
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"val_loss={val_stats['loss_total']:.4f} "
            f"val_dx={val_stats['mae_dx_m']:.3f} "
            f"val_dy={val_stats['mae_dy_m']:.3f} "
            f"val_yaw={val_stats['mae_yaw_follow_deg']:.3f} "
            f"val_score={val_stats['selection_score']:.3f}"
        )

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])

    test_stats = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        device=device,
        translation_stats=translation_stats,
        training=False,
        use_amp=config.use_amp,
        max_batches=config.max_test_batches,
        channels_last=config.channels_last,
        grad_clip_norm=0.0,
        progress_label="test",
        log_every_steps=config.log_every_steps,
        wandb_run=None,
        global_step_base=0,
    )
    save_json(run_dir / "test_metrics.json", test_stats)

    if wandb_run is not None:
        wandb_run.log(
            {
                "test/loss_total": test_stats["loss_total"],
                "test/loss_dx": test_stats["loss_dx"],
                "test/loss_dy": test_stats["loss_dy"],
                "test/loss_yaw": test_stats["loss_yaw"],
            }
        )
        wandb_run.finish()

    print("\nTraining complete")
    print(f"Run directory: {run_dir}")
    print(json.dumps(test_stats, indent=2))


def run_epoch(
    model: TargetPoseRegressor,
    loader: DataLoader,
    criterion: TargetPoseLoss,
    optimizer: Optional[AdamW],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    translation_stats: TranslationStats,
    training: bool,
    use_amp: bool,
    max_batches: int,
    channels_last: bool,
    grad_clip_norm: float,
    progress_label: str,
    log_every_steps: int,
    wandb_run: Optional[object],
    global_step_base: int,
) -> Dict[str, float]:
    """Run one train or eval epoch."""
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dx_loss = 0.0
    total_dy_loss = 0.0
    total_yaw_loss = 0.0
    num_batches = 0

    predicted_translations = []
    predicted_yaws = []
    target_translations = []
    target_yaws = []

    iterator = tqdm(loader, desc=progress_label, leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches > 0 and batch_index >= max_batches:
            break

        full_input = batch["full_input"].to(device, non_blocking=True)
        crop_input = batch["crop_input"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        translation_target = batch["translation_target"].to(device, non_blocking=True)
        translation_raw = batch["translation_raw"].to(device, non_blocking=True)
        yaw_target = batch["yaw_target"].to(device, non_blocking=True)
        yaw_follow_deg = batch["yaw_follow_deg"].to(device, non_blocking=True)

        if channels_last:
            full_input = full_input.contiguous(memory_format=torch.channels_last)
            crop_input = crop_input.contiguous(memory_format=torch.channels_last)

        autocast_enabled = bool(use_amp and device.type == "cuda")
        autocast_context = (
            torch.cuda.amp.autocast(enabled=True)
            if autocast_enabled
            else nullcontext()
        )

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            outputs = model(full_input, crop_input, geometry)
            loss, loss_metrics = criterion(outputs, translation_target, yaw_target)

        if training and optimizer is not None:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        decoded = decode_predictions(outputs, translation_stats)
        predicted_translations.append(decoded["translation"].detach().cpu())
        predicted_yaws.append(decoded["yaw_follow_deg"].detach().cpu())
        target_translations.append(translation_raw.detach().cpu())
        target_yaws.append(yaw_follow_deg.detach().cpu())

        total_loss += loss_metrics["loss_total"]
        total_dx_loss += loss_metrics["loss_dx"]
        total_dy_loss += loss_metrics["loss_dy"]
        total_yaw_loss += loss_metrics["loss_yaw"]
        num_batches += 1

        iterator.set_postfix(
            loss=f"{loss_metrics['loss_total']:.4f}",
            dx=f"{loss_metrics['loss_dx']:.4f}",
            dy=f"{loss_metrics['loss_dy']:.4f}",
            yaw=f"{loss_metrics['loss_yaw']:.4f}",
        )

        if (
            training
            and wandb_run is not None
            and log_every_steps > 0
            and (batch_index + 1) % log_every_steps == 0
        ):
            wandb_run.log(
                {
                    "train_step/loss_total": loss_metrics["loss_total"],
                    "train_step/loss_dx": loss_metrics["loss_dx"],
                    "train_step/loss_dy": loss_metrics["loss_dy"],
                    "train_step/loss_yaw": loss_metrics["loss_yaw"],
                },
                step=global_step_base + batch_index + 1,
            )

    if num_batches == 0:
        raise ValueError(f"No batches were processed for {progress_label}")

    predicted_translation_tensor = torch.cat(predicted_translations, dim=0)
    predicted_yaw_tensor = torch.cat(predicted_yaws, dim=0)
    target_translation_tensor = torch.cat(target_translations, dim=0)
    target_yaw_tensor = torch.cat(target_yaws, dim=0)
    metrics = compute_pose_metrics(
        predicted_translation=predicted_translation_tensor,
        predicted_yaw_deg=predicted_yaw_tensor,
        target_translation=target_translation_tensor,
        target_yaw_deg=target_yaw_tensor,
    )
    metrics.update(
        {
            "loss_total": float(total_loss / num_batches),
            "loss_dx": float(total_dx_loss / num_batches),
            "loss_dy": float(total_dy_loss / num_batches),
            "loss_yaw": float(total_yaw_loss / num_batches),
        }
    )
    return metrics


def make_loader(
    dataset: TargetPoseDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    """Build a DataLoader with sensible defaults for image training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def create_run_dir(config: TargetPoseTrainingConfig) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{config.label_source}_{timestamp}"
    run_dir = Path(config.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def init_wandb(
    config: TargetPoseTrainingConfig,
    run_dir: Path,
    train_dataset: TargetPoseDataset,
    val_dataset: TargetPoseDataset,
    test_dataset: TargetPoseDataset,
    model: TargetPoseRegressor,
) -> Optional[object]:
    """Initialize a Weights & Biases run when requested."""
    if not config.wandb_enabled:
        return None
    if wandb is None:
        print("[wandb] not installed; continuing without external logging")
        return None

    run_name = config.wandb_run_name or run_dir.name
    return wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.to_dict(),
        dir=str(run_dir),
        notes=(
            f"train_rows={len(train_dataset)} "
            f"val_rows={len(val_dataset)} "
            f"test_rows={len(test_dataset)}"
        ),
    )


def save_json(path: Path, payload: object) -> None:
    """Write a JSON file with a stable pretty format."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def select_device(device_name: str) -> torch.device:
    """Return a usable torch device."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
