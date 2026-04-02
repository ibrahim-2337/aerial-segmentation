"""
Evaluation and visualisation for aerial building segmentation.

Usage
-----
# Compute metrics for all checkpoints and write experiments/results.csv:
    python src/evaluate.py --mode metrics

# Generate side-by-side prediction visualisations (5 examples per model):
    python src/evaluate.py --mode visualize --model unet --seed 42

# Both at once:
    python src/evaluate.py --mode all
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe in Colab / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.amp as amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import InriaDataset
from src.models  import get_model
from src.utils   import compute_iou, compute_dice, load_checkpoint

# ---------------------------------------------------------------------------
# Checkpoint directory (local /content/ only — no Drive needed)
# ---------------------------------------------------------------------------

IN_COLAB = os.path.isdir("/content")
CKPT_BASE = "/content/checkpoints" if IN_COLAB else str(Path(PROJECT_ROOT) / "checkpoints")

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_available_checkpoints() -> list:
    """Return (model_name, seed) pairs that have a best.pth in CKPT_BASE."""
    base = Path(CKPT_BASE)
    available = []
    if not base.exists():
        return available
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "best.pth").exists() and "_seed" in d.name:
            model_name, seed_str = d.name.rsplit("_seed", 1)
            try:
                available.append((model_name, int(seed_str)))
            except ValueError:
                pass
    return available


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    seed: int,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Load best checkpoint for *model_name* / *seed* and compute val metrics."""
    ckpt_path = Path(CKPT_BASE) / f"{model_name}_seed{seed}" / "best.pth"
    if not ckpt_path.exists():
        print(f"[WARN] checkpoint not found: {ckpt_path}")
        return {"model": model_name, "seed": seed, "iou": float("nan"), "dice": float("nan")}

    model = get_model(model_name).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()

    all_iou  = []
    all_dice = []

    for images, masks in tqdm(val_loader, desc=f"  {model_name}/seed{seed}", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with amp.autocast("cuda"):
            logits = model(images)

        all_iou.append(compute_iou(logits, masks))
        all_dice.append(compute_dice(logits, masks))

    return {
        "model": model_name,
        "seed":  seed,
        "iou":   float(np.mean(all_iou)),
        "dice":  float(np.mean(all_dice)),
    }


def build_results_table(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed results into mean ± std per model."""
    df = pd.DataFrame(results).dropna(subset=["iou", "dice"])
    raw_path = Path(PROJECT_ROOT) / "experiments" / "results.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"[info] per-seed results saved to {raw_path}")

    agg = df.groupby("model")[["iou", "dice"]].agg(["mean", "std"]).round(4)
    agg.columns = ["iou_mean", "iou_std", "dice_mean", "dice_std"]
    agg = agg.reset_index()
    agg["IoU"]  = agg.apply(
        lambda r: f"{r.iou_mean:.4f}" + (f" ± {r.iou_std:.4f}" if not pd.isna(r.iou_std) else ""), axis=1)
    agg["Dice"] = agg.apply(
        lambda r: f"{r.dice_mean:.4f}" + (f" ± {r.dice_std:.4f}" if not pd.isna(r.dice_std) else ""), axis=1)
    return agg[["model", "IoU", "Dice"]]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3, H, W) tensor back to a uint8 HxWx3 array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def visualize_predictions(
    model_name: str,
    seed: int,
    val_dataset: InriaDataset,
    device: torch.device,
    n_samples: int = 5,
    out_dir: str = "experiments/figures",
) -> None:
    """Save side-by-side (input | ground truth | prediction) figures."""
    ckpt_path = Path(CKPT_BASE) / f"{model_name}_seed{seed}" / "best.pth"
    if not ckpt_path.exists():
        print(f"[WARN] checkpoint not found for {model_name}/seed{seed}, skipping viz.")
        return

    model = get_model(model_name).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Pick tiles that actually contain buildings in the ground truth
    building_indices = []
    for idx in range(len(val_dataset)):
        _, mask_t = val_dataset[idx]
        if mask_t.sum() > 50:          # at least 50 building pixels
            building_indices.append(idx)
        if len(building_indices) >= n_samples * 10:
            break                       # stop scanning early once we have enough candidates

    if len(building_indices) < n_samples:
        # fallback: evenly spaced
        building_indices = list(np.linspace(0, len(val_dataset) - 1, n_samples, dtype=int))

    np.random.seed(0)
    chosen = np.random.choice(building_indices, size=n_samples, replace=False).tolist()
    subset = Subset(val_dataset, chosen)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight="bold")

    for row_idx, (image_t, mask_t) in enumerate(subset):
        image_t = image_t.unsqueeze(0).to(device)  # (1, 3, H, W)
        mask_t  = mask_t.unsqueeze(0).to(device)   # (1, 1, H, W)

        with amp.autocast("cuda"):
            logits = model(image_t)

        pred_mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
        gt_mask   = mask_t.squeeze().cpu().numpy()
        rgb_image = denormalize(image_t.squeeze())

        iou  = compute_iou(logits, mask_t)
        dice = compute_dice(logits, mask_t)

        axes[row_idx, 0].imshow(rgb_image)
        axes[row_idx, 0].set_ylabel(f"sample {row_idx+1}", fontsize=10)
        axes[row_idx, 1].imshow(gt_mask,   cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_xlabel(
            f"IoU={iou:.3f}  Dice={dice:.3f}", fontsize=9, color="darkred"
        )

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{model_name.upper()} – seed {seed} – West Tyrol validation",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    save_path = Path(out_dir) / f"{model_name}_seed{seed}_predictions.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] saved figure → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or (
        "/content/inria" if IN_COLAB else str(Path(PROJECT_ROOT) / "data" / "inria")
    )

    available = find_available_checkpoints()
    if not available:
        print(f"[ERROR] No checkpoints found in {CKPT_BASE}. Run training first.")
        return

    print(f"[info] Found checkpoints: {[f'{m}/seed{s}' for m, s in available]}")

    # val = vienna (checkpoint selection); test = tyrol-w (final score, touch once)
    split = "test" if args.test else "val"
    split_label = "Test (tyrol-w)" if args.test else "Val (vienna)"

    if args.mode in ("metrics", "all"):
        ds     = InriaDataset(data_root, split=split, tile_size=256, overlap=0.0)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        results = []
        for model_name, seed in available:
            res = evaluate_checkpoint(model_name, seed, loader, device)
            results.append(res)
            print(f"  {model_name}/seed{seed} → IoU={res['iou']:.4f}  Dice={res['dice']:.4f}")

        table = build_results_table(results)
        print(f"\n=== Comparison Table — {split_label} ===")
        print(table.to_string(index=False))

    if args.mode in ("visualize", "all"):
        ds = InriaDataset(data_root, split=split, tile_size=256, overlap=0.0)
        viz_models = [(m, s) for m, s in available if s == args.seed] or available[:1]
        for model_name, seed in viz_models:
            visualize_predictions(
                model_name=model_name,
                seed=seed,
                val_dataset=ds,
                device=device,
                n_samples=5,
                out_dir=str(Path(PROJECT_ROOT) / "experiments" / "figures"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation models")
    parser.add_argument("--mode",      default="all", choices=["metrics", "visualize", "all"])
    parser.add_argument("--model",     default="unet", choices=["unet", "deeplabv3plus", "segformer"],
                        help="Model to visualise (only used when mode=visualize)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Seed to use for visualisation checkpoint")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data/inria directory (overrides default)")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on test split (tyrol-w) instead of val (vienna). "
                             "Only run this once training is complete.")
    args = parser.parse_args()
    main(args)
