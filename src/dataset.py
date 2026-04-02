"""
Rasterio-based dataset for the INRIA Aerial Image Labeling benchmark.

Expected data layout
--------------------
data/inria/
    train/
        images/   *.tif   (RGB GeoTIFF, typically 5000×5000)
        gt/       *.tif   (single-band binary mask: 255 = building, 0 = background)

Train cities : austin, chicago, kitsap, vienna
Val city     : tyrol-w  (West Tyrol, held-out)
"""

import os
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import generate_tiles

# ---------------------------------------------------------------------------
# Per-worker file-handle cache
# ---------------------------------------------------------------------------
# rasterio.open() is expensive (~1–5 ms each call).  With 200 K tiles per
# epoch every __getitem__ would open and close two files — adding hours of
# pure I/O overhead.  thread.local() gives each DataLoader worker its own
# dict of open handles that persist across __getitem__ calls.

_tls = threading.local()


def _open_rasterio(path: str) -> rasterio.DatasetReader:
    """Return a cached rasterio handle for *path*, one per worker thread."""
    if not hasattr(_tls, "handles"):
        _tls.handles = {}
    if path not in _tls.handles:
        _tls.handles[path] = rasterio.open(path)
    return _tls.handles[path]

# ---------------------------------------------------------------------------
# City splits
# ---------------------------------------------------------------------------

TRAIN_CITIES: List[str] = ["austin", "chicago", "kitsap"]
VAL_CITIES:   List[str] = ["vienna"]
TEST_CITIES:  List[str] = ["tyrol-w"]


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_augmentation(split: str) -> A.Compose:
    """Return albumentations pipeline for *train* or *val* split."""
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
            ),
            A.RandomCrop(height=224, width=224, p=0.3),
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ---------------------------------------------------------------------------
# Tile index builder
# ---------------------------------------------------------------------------

TileRecord = Tuple[str, str, int, int]  # (image_path, mask_path, row, col)


def build_tile_index(
    data_root: str,
    cities: List[str],
    tile_size: int = 256,
    overlap: float = 0.5,
    max_tiles: int = 0,
) -> List[TileRecord]:
    """Scan *data_root*/train/{images,gt} for files belonging to *cities* and
    return the full list of (image_path, mask_path, row_start, col_start) tiles.

    Files are matched by checking whether the city name appears (case-insensitive)
    at the start of the filename stem, e.g. ``austin1.tif`` → city ``austin``.
    """
    images_dir = Path(data_root) / "train" / "images"
    gt_dir     = Path(data_root) / "train" / "gt"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Place the INRIA dataset under data/inria/ as described in README.md."
        )

    records: List[TileRecord] = []

    for img_path in sorted(images_dir.glob("*.tif")):
        stem = img_path.stem.lower()
        city = next((c for c in cities if stem.startswith(c)), None)
        if city is None:
            continue

        mask_path = gt_dir / img_path.name
        if not mask_path.exists():
            print(f"[WARNING] No mask found for {img_path.name}, skipping.")
            continue

        with rasterio.open(img_path) as src:
            h, w = src.height, src.width

        tiles = generate_tiles(h, w, tile_size=tile_size, overlap=overlap)
        for row, col in tiles:
            records.append((str(img_path), str(mask_path), row, col))

    if max_tiles > 0 and len(records) > max_tiles:
        import random as _rnd
        _rnd.shuffle(records)
        records = records[:max_tiles]

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InriaDataset(Dataset):
    """Tile-based dataset for INRIA aerial building segmentation.

    Each item is a (image_tensor, mask_tensor) pair extracted with a sliding
    window from the original GeoTIFF files.  Images are read on-the-fly with
    rasterio, preserving geospatial coordinate metadata (it stays in the file
    handle and is accessible via ``self.open_file()`` if needed downstream).

    Parameters
    ----------
    data_root : str
        Path to the ``data/inria/`` directory.
    split : str
        ``"train"`` or ``"val"``.
    tile_size : int
        Patch size in pixels (default 256).
    overlap : float
        Fractional overlap between adjacent tiles (default 0.5 → 50 %).
    transform : albumentations.Compose, optional
        If None, the default pipeline for *split* is used.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        tile_size: int = 256,
        overlap: float = 0.5,
        transform: Optional[A.Compose] = None,
        max_tiles: int = 0,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.split = split
        self.transform = transform if transform is not None else get_augmentation(split)

        if split == "train":
            cities = TRAIN_CITIES
        elif split == "val":
            cities = VAL_CITIES
        else:
            cities = TEST_CITIES
        self.tiles: List[TileRecord] = build_tile_index(
            data_root, cities, tile_size=tile_size, overlap=overlap, max_tiles=max_tiles
        )

        if len(self.tiles) == 0:
            raise RuntimeError(
                f"No tiles found for split='{split}' in {data_root}. "
                "Check that GeoTIFF files are present and city names match."
            )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path, row, col = self.tiles[idx]
        ts = self.tile_size
        window = rasterio.windows.Window(col, row, ts, ts)

        # --- read image via cached handle (no open/close overhead) ---
        image = _open_rasterio(img_path).read(window=window)   # (3, H, W) uint8

        # Pad if the window extends beyond the image boundary
        image = _pad_to_tile(image, ts)             # still (3, H, W)
        image = np.moveaxis(image, 0, -1)           # (H, W, 3)

        # --- read mask via cached handle ---
        mask = _open_rasterio(mask_path).read(1, window=window)  # (H, W) uint8

        mask = _pad_to_tile(mask[np.newaxis], ts)[0]  # (H, W)
        mask = (mask > 127).astype(np.float32)      # binary 0/1

        # --- augmentation ---
        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]                # (3, H, W)  float32 tensor
        mask_t  = augmented["mask"].unsqueeze(0)    # (1, H, W)  float32 tensor

        return image_t, mask_t

    def get_tile_info(self, idx: int) -> TileRecord:
        """Return raw tile metadata (useful for visualisation)."""
        return self.tiles[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_tile(arr: np.ndarray, tile_size: int) -> np.ndarray:
    """Zero-pad a (C, H, W) array so that H == W == tile_size."""
    c, h, w = arr.shape
    if h == tile_size and w == tile_size:
        return arr
    out = np.zeros((c, tile_size, tile_size), dtype=arr.dtype)
    out[:, :h, :w] = arr
    return out
