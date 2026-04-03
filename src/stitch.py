"""
Full-image inference with sliding-window averaging and GeoTIFF export.

Runs a trained model over an entire 5000×5000 image using a sliding window,
averages overlapping predictions, and saves a binary mask GeoTIFF with the
same CRS and geotransform as the input — ready for QGIS / Google Earth.

Usage
-----
python src/stitch.py \\
    --model unet --seed 42 \\
    --image /content/inria/train/images/vienna1.tif \\
    --out   experiments/predictions/vienna1_unet.tif

# Disable TTA for speed:
python src/stitch.py --model unet --seed 42 --image ... --out ... --no-tta
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.amp as amp

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import get_model
from src.utils  import generate_tiles, load_checkpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

IN_COLAB  = os.path.isdir("/content")
CKPT_BASE = "/content/checkpoints" if IN_COLAB else str(Path(PROJECT_ROOT) / "checkpoints")

_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def _tta_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """8-fold TTA: 4 rotations × 2 horizontal flips, averaged."""
    preds = []
    for k in range(4):
        for flip in (False, True):
            aug = torch.rot90(x, k, dims=[2, 3])
            if flip:
                aug = torch.flip(aug, [3])
            with amp.autocast("cuda"):
                prob = torch.sigmoid(model(aug))
            if flip:
                prob = torch.flip(prob, [3])
            prob = torch.rot90(prob, -k, dims=[2, 3])
            preds.append(prob)
    return torch.stack(preds).mean(0)


# ---------------------------------------------------------------------------
# Core stitching logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def stitch_image(
    model_name: str,
    seed: int,
    image_path: str,
    out_path: str,
    tile_size: int = 256,
    overlap: float = 0.25,
    device: torch.device = None,
    use_tta: bool = True,
    batch_size: int = 16,
) -> np.ndarray:
    """Run sliding-window inference over a full image and save a GeoTIFF.

    Returns the float32 probability map (H, W) in [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path(CKPT_BASE) / f"{model_name}_seed{seed}" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = get_model(model_name).to(device)
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()

    with rasterio.open(image_path) as src:
        h, w    = src.height, src.width
        profile = src.profile.copy()
        image_data = src.read()           # (3, H, W) uint8

    acc   = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)

    # Process in mini-batches
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start : batch_start + batch_size]
        tensors = []
        coords  = []

        for row, col in batch_tiles:
            r2 = min(row + tile_size, h)
            c2 = min(col + tile_size, w)
            patch = image_data[:, row:r2, col:c2]

            # Zero-pad if edge tile
            pad = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            pad[:, :r2 - row, :c2 - col] = patch
            rgb = np.moveaxis(pad, 0, -1)    # (H, W, 3)

            dummy_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
            t = _transform(image=rgb, mask=dummy_mask)["image"]
            tensors.append(t)
            coords.append((row, col, r2, c2))

        x = torch.stack(tensors).to(device)   # (B, 3, 256, 256)

        if use_tta:
            probs = _tta_predict(model, x).squeeze(1).cpu().numpy()  # (B, 256, 256)
        else:
            with amp.autocast("cuda"):
                probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()

        for prob, (row, col, r2, c2) in zip(probs, coords):
            acc[row:r2, col:c2]   += prob[:r2 - row, :c2 - col]
            count[row:r2, col:c2] += 1.0

    prob_map = (acc / np.maximum(count, 1)).astype(np.float32)
    binary   = (prob_map > 0.5).astype(np.uint8) * 255

    # Save GeoTIFF with original CRS/transform
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype=rasterio.uint8, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary[np.newaxis])

    print(f"[info] saved prediction → {out_path}  ({h}×{w} px)")
    return prob_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch full-image prediction GeoTIFF")
    parser.add_argument("--model",      default="unet",
                        choices=["unet", "deeplabv3plus", "segformer"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--image",      required=True, help="Input GeoTIFF path")
    parser.add_argument("--out",        required=True, help="Output GeoTIFF path")
    parser.add_argument("--overlap",    type=float, default=0.25)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--no-tta",     action="store_true", help="Disable TTA (faster)")
    args = parser.parse_args()

    stitch_image(
        model_name=args.model,
        seed=args.seed,
        image_path=args.image,
        out_path=args.out,
        overlap=args.overlap,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_tta=not args.no_tta,
        batch_size=args.batch_size,
    )
