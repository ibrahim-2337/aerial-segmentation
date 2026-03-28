import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN pick the fastest convolution kernel for the
    # current hardware — significant throughput gain on A100.
    # deterministic=False is safe here because all Python/NumPy/Torch seeds
    # are fixed above, which is sufficient for run-to-run reproducibility.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ---------------------------------------------------------------------------
# Sliding-window tile generation
# ---------------------------------------------------------------------------

def generate_tiles(height: int, width: int, tile_size: int = 256, overlap: float = 0.5):
    """Return a list of (row_start, col_start) offsets for sliding-window tiling.

    The stride is tile_size * (1 - overlap).  A final row/column of tiles is
    appended when the image dimension is not exactly divisible, so every pixel
    is covered at least once.
    """
    stride = max(1, int(tile_size * (1 - overlap)))

    def axis_offsets(length):
        offsets = list(range(0, length - tile_size + 1, stride))
        if not offsets or offsets[-1] + tile_size < length:
            offsets.append(max(0, length - tile_size))
        return offsets

    rows = axis_offsets(height)
    cols = axis_offsets(width)
    return [(r, c) for r in rows for c in cols]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE and Dice losses."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Metrics (batch-level, numpy-safe)
# ---------------------------------------------------------------------------

def compute_iou(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred = pred.view(-1)
    targets = targets.view(-1).float()
    intersection = (pred * targets).sum()
    union = pred.sum() + targets.sum() - intersection
    return (intersection / (union + eps)).item()


def compute_dice(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred = pred.view(-1)
    targets = targets.view(-1).float()
    intersection = (pred * targets).sum()
    return (2.0 * intersection / (pred.sum() + targets.sum() + eps)).item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
