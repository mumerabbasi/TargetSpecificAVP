"""Training entry point for the mask-conditioned target pose CNN."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PoseEstimationConfig
from .dataset import (
    PoseEstimationDataset,
    TranslationStats,
    build_frame_splits,
    load_pose_rows,
    split_summary,
)
from .model import (
    PoseEstimationCNN,
    PoseEstimationLoss,
    compute_pose_metrics,
    decode_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mask-conditioned target pose CNN",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-source", choices=["gt", "pred"], default="gt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="mask_conditioned_pose")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet50",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-mask-area-px", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--dx-loss-weight", type=float, default=1.0)
    parser.add_argument("--dy-loss-weight", type=float, default=1.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=1.0)
    parser.add_argument("--wandb-project", default="ravp-pose")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PoseEstimationConfig(
        dataset_root=args.dataset_root,
        label_source=args.label_source,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        image_size=(args.image_size, args.image_size),
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
    torch.set_float32_matmul_precision("high")

    rows = load_pose_rows(config)
    rows_by_split = build_frame_splits(rows, config)
    translation_stats = TranslationStats.from_rows(rows_by_split["train"])

    run_dir = create_run_dir(config)
    save_json(run_dir / "config.json", config.to_dict())
    save_json(run_dir / "translation_stats.json", translation_stats.to_dict())
    save_json(run_dir / "split_summary.json", split_summary(rows_by_split))

    train_dataset = PoseEstimationDataset(
        config,
        rows_by_split["train"],
        translation_stats,
        training=True,
    )
    val_dataset = PoseEstimationDataset(
        config,
        rows_by_split["val"],
        translation_stats,
        training=False,
    )
    test_dataset = PoseEstimationDataset(
        config,
        rows_by_split["test"],
        translation_stats,
        training=False,
    )

    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        config.num_workers,
        training=True,
    )
    val_loader = make_loader(
        val_dataset,
        config.batch_size,
        config.num_workers,
        training=False,
    )
    test_loader = make_loader(
        test_dataset,
        config.batch_size,
        config.num_workers,
        training=False,
    )

    device = select_device(config.device)
    model = PoseEstimationCNN(config).to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = PoseEstimationLoss(config)
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
        config,
        run_dir,
        train_dataset,
        val_dataset,
        test_dataset,
        model,
    )

    best_val_loss = float("inf")
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
            "train": train_stats,
            "val": val_stats,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        save_json(run_dir / "history.json", history)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "lr": epoch_record["learning_rate"],
                    **prefix_metrics("train", train_stats),
                    **prefix_metrics("val", val_stats),
                }
            )

        checkpoint = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict()
            if scaler is not None else None, "config": config.to_dict(),
            "translation_stats": translation_stats.to_dict(),
            "train_stats": train_stats, "val_stats": val_stats, }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_stats["loss_total"] < best_val_loss:
            best_val_loss = val_stats["loss_total"]
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"val_loss={val_stats['loss_total']:.4f} "
            f"val_dx={val_stats['mae_dx_m']:.3f} "
            f"val_dy={val_stats['mae_dy_m']:.3f} "
            f"val_yaw_follow={val_stats['mae_yaw_follow_deg']:.3f}"
        )

    best_checkpoint = torch.load(
        run_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
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
        wandb_run.log(prefix_metrics("test", test_stats))
        wandb_run.finish()

    print("\nTraining complete")
    print(f"Run directory: {run_dir}")
    print(json.dumps(test_stats, indent=2))


def run_epoch(
    model: PoseEstimationCNN,
    loader: DataLoader,
    criterion: PoseEstimationLoss,
    optimizer: Optional[AdamW],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    translation_stats: TranslationStats,
    training: bool,
    use_amp: bool,
    max_batches: int,
    channels_last: bool,
    grad_clip_norm: float,
    progress_label: str,
    log_every_steps: int,
    wandb_run: Optional[wandb.sdk.wandb_run.Run],
    global_step_base: int,
) -> Dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dx = 0.0
    total_dy = 0.0
    total_yaw = 0.0
    num_batches = 0

    pred_translations = []
    pred_yaws = []
    gt_translations = []
    gt_yaws = []

    iterator = tqdm(loader, desc=progress_label, leave=False)
    for batch_index, batch in enumerate(iterator):
        if max_batches > 0 and batch_index >= max_batches:
            break

        inputs = batch["input"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        translation_target = batch["translation_target"].to(
            device, non_blocking=True)
        translation_raw = batch["translation_raw"].to(
            device, non_blocking=True)
        yaw_target = batch["yaw_target"].to(device, non_blocking=True)
        yaw_follow_deg = batch["yaw_follow_deg"].to(device, non_blocking=True)

        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=(use_amp and device.type == "cuda"),
        ):
            outputs = model(inputs, masks, geometry)
            loss, loss_dict = criterion(
                outputs, translation_target, yaw_target)

        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        decoded = decode_predictions(outputs, translation_stats)
        pred_translations.append(decoded["translation"].detach().cpu())
        pred_yaws.append(decoded["yaw_follow_deg"].detach().cpu())
        gt_translations.append(translation_raw.detach().cpu())
        gt_yaws.append(yaw_follow_deg.detach().cpu())

        total_loss += loss_dict["loss_total"]
        total_dx += loss_dict["loss_dx"]
        total_dy += loss_dict["loss_dy"]
        total_yaw += loss_dict["loss_yaw"]
        num_batches += 1

        if (
            training
            and wandb_run is not None
            and (batch_index + 1) % max(log_every_steps, 1) == 0
        ):
            wandb_run.log(
                {
                    "step": global_step_base + batch_index + 1,
                    "train_step/loss_total": loss_dict["loss_total"],
                    "train_step/loss_dx": loss_dict["loss_dx"],
                    "train_step/loss_dy": loss_dict["loss_dy"],
                    "train_step/loss_yaw": loss_dict["loss_yaw"],
                }
            )

    if num_batches == 0:
        raise ValueError(f"{progress_label} produced zero batches")

    predicted_translation = torch.cat(pred_translations, dim=0)
    predicted_yaw = torch.cat(pred_yaws, dim=0)
    target_translation = torch.cat(gt_translations, dim=0)
    target_yaw = torch.cat(gt_yaws, dim=0)
    metrics = compute_pose_metrics(
        predicted_translation,
        predicted_yaw,
        target_translation,
        target_yaw,
    )
    metrics.update(
        {
            "loss_total": total_loss / num_batches,
            "loss_dx": total_dx / num_batches,
            "loss_dy": total_dy / num_batches,
            "loss_yaw": total_yaw / num_batches,
        }
    )
    return metrics


def make_loader(
    dataset: PoseEstimationDataset,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=training,
    )


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(config: PoseEstimationConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def init_wandb(
    config: PoseEstimationConfig,
    run_dir: Path,
    train_dataset: PoseEstimationDataset,
    val_dataset: PoseEstimationDataset,
    test_dataset: PoseEstimationDataset,
    model: PoseEstimationCNN,
) -> Optional[wandb.sdk.wandb_run.Run]:
    if not config.wandb_enabled:
        return None

    run_name = config.wandb_run_name or run_dir.name
    payload = config.to_dict()
    payload.update(
        {
            "num_train_rows": len(train_dataset),
            "num_val_rows": len(val_dataset),
            "num_test_rows": len(test_dataset),
            "num_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
    )
    return wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=payload,
        dir=str(run_dir),
        save_code=False,
        settings=wandb.Settings(start_method="thread"),
    )


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    main()
