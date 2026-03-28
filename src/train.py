"""
Training script for aerial building segmentation.

Usage (local or Colab terminal):
    python src/train.py --model unet --seed 42 --epochs 20
    python src/train.py --model deeplabv3plus --seed 0
    python src/train.py --model segformer --seed 42 --resume

Google Drive checkpoints are written after every epoch when running in Colab.
Locally, checkpoints are written to ./checkpoints/.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Colab detection and Drive mounting
# ---------------------------------------------------------------------------

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_CKPT_BASE = "/content/drive/MyDrive/aerial-segmentation/checkpoints"
else:
    DRIVE_CKPT_BASE = str(Path(__file__).parent.parent / "checkpoints")

# Add project root to path so `src.*` imports work when running as a script
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import InriaDataset
from src.models  import get_model
from src.utils   import BCEDiceLoss, compute_iou, compute_dice, set_seed, save_checkpoint


# ---------------------------------------------------------------------------
# Train / validate helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: amp.GradScaler,
) -> dict:
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0
    n_batches  = len(loader)

    pbar = tqdm(loader, desc="  train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        iou  = compute_iou(logits.detach(), masks)
        dice = compute_dice(logits.detach(), masks)
        total_loss += loss.item()
        total_iou  += iou
        total_dice += dice

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    return {
        "loss": total_loss / n_batches,
        "iou":  total_iou  / n_batches,
        "dice": total_dice / n_batches,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0
    n_batches  = len(loader)

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with amp.autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_iou  += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)

    return {
        "loss": total_loss / n_batches,
        "iou":  total_iou  / n_batches,
        "dice": total_dice / n_batches,
    }


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_iou", "train_dice",
            "val_loss", "val_iou", "val_dice", "lr",
        ])
        writer.writeheader()


def append_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  model={args.model}  seed={args.seed}")

    # --- checkpoint directory (Drive when in Colab) ---
    ckpt_dir = Path(DRIVE_CKPT_BASE) / f"{args.model}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pth"
    last_ckpt_path = ckpt_dir / "last.pth"
    log_csv_path   = ckpt_dir / "metrics.csv"

    # --- data ---
    data_root = args.data_root or str(Path(__file__).parent.parent / "data" / "inria")
    train_ds = InriaDataset(data_root, split="train", tile_size=256, overlap=0.5)
    val_ds   = InriaDataset(data_root, split="val",   tile_size=256, overlap=0.5)
    print(f"[info] train tiles={len(train_ds)}  val tiles={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    # --- model ---
    model = get_model(args.model, num_classes=1).to(device)

    # torch.compile (PyTorch ≥ 2.0) fuses ops and generates optimised CUDA
    # kernels — typically 15–30 % faster per epoch with no code changes needed.
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[info] torch.compile enabled (reduce-overhead mode)")
        except Exception as exc:
            print(f"[info] torch.compile skipped: {exc}")

    # --- optimiser / scheduler / loss ---
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion  = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    scaler     = amp.GradScaler()

    # --- optional resume ---
    start_epoch = 1
    best_val_iou = 0.0
    if args.resume and last_ckpt_path.exists():
        print(f"[info] resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        print(f"[info] resuming at epoch {start_epoch}, best_val_iou={best_val_iou:.4f}")
    else:
        init_csv(str(log_csv_path))

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"  train  loss={train_metrics['loss']:.4f}  iou={train_metrics['iou']:.4f}  dice={train_metrics['dice']:.4f}\n"
            f"  val    loss={val_metrics['loss']:.4f}  iou={val_metrics['iou']:.4f}  dice={val_metrics['dice']:.4f}\n"
            f"  lr={lr_now:.2e}  elapsed={elapsed:.0f}s"
        )

        # CSV log
        append_csv(str(log_csv_path), {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_iou":  round(train_metrics["iou"],  6),
            "train_dice": round(train_metrics["dice"],  6),
            "val_loss":   round(val_metrics["loss"],   6),
            "val_iou":    round(val_metrics["iou"],    6),
            "val_dice":   round(val_metrics["dice"],   6),
            "lr":         round(lr_now, 8),
        })

        # Save last checkpoint every epoch (enables resume after interruption)
        last_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "val_iou":              val_metrics["iou"],
            "best_val_iou":         best_val_iou,
            "args":                 vars(args),
        }
        torch.save(last_state, str(last_ckpt_path))

        # Save best checkpoint
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(last_state, str(best_ckpt_path))
            print(f"  [*] new best val IoU: {best_val_iou:.4f} → saved to {best_ckpt_path}")

    print(f"\n[done] best val IoU = {best_val_iou:.4f}")
    print(f"       checkpoints: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train aerial segmentation model")

    parser.add_argument("--model",        type=str,   default="unet",
                        choices=["unet", "deeplabv3plus", "segformer"],
                        help="Model architecture")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--epochs",       type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="Batch size per GPU")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--workers",      type=int,   default=8,
                        help="DataLoader worker count")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from last checkpoint if available")
    parser.add_argument("--data_root",    type=str,   default=None,
                        help="Path to data/inria directory (overrides default)")

    args = parser.parse_args()
    main(args)
