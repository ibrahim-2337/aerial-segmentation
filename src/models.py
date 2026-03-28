"""
Model definitions for aerial building segmentation.

Three architectures are supported:
  unet        – U-Net with ResNet-34 encoder  (segmentation-models-pytorch)
  deeplabv3+  – DeepLabV3+ with ResNet-50     (segmentation-models-pytorch)
  segformer   – SegFormer-B2                  (HuggingFace transformers)

All models return raw logits with shape (B, 1, H, W) for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# U-Net  (ResNet-34 encoder, ImageNet pretrained)
# ---------------------------------------------------------------------------

def get_unet(num_classes: int = 1) -> nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,        # raw logits
    )
    return model


# ---------------------------------------------------------------------------
# DeepLabV3+  (ResNet-50 encoder, ImageNet pretrained)
# ---------------------------------------------------------------------------

def get_deeplabv3plus(num_classes: int = 1) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


# ---------------------------------------------------------------------------
# SegFormer-B2  (HuggingFace transformers, ADE20K pretrained)
# ---------------------------------------------------------------------------

class SegFormerWrapper(nn.Module):
    """Thin wrapper around HuggingFace SegformerForSemanticSegmentation.

    The HuggingFace model outputs logits at 1/4 input resolution; this wrapper
    upsamples them back to the full input resolution so the interface matches
    the other two architectures.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained("nvidia/mit-b2")
        config.num_labels = num_classes
        config.id2label   = {i: str(i) for i in range(num_classes)}
        config.label2id   = {str(i): i for i in range(num_classes)}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            config=config,
            ignore_mismatched_sizes=True,   # replace ADE20K decode head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outputs = self.model(pixel_values=x)
        logits  = outputs.logits                    # (B, C, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def get_segformer(num_classes: int = 1) -> nn.Module:
    return SegFormerWrapper(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "unet":         get_unet,
    "deeplabv3plus": get_deeplabv3plus,
    "segformer":    get_segformer,
}


def get_model(name: str, num_classes: int = 1) -> nn.Module:
    """Return an initialised model by name.

    Parameters
    ----------
    name : str
        One of ``"unet"``, ``"deeplabv3plus"``, ``"segformer"``.
    num_classes : int
        Number of output channels (1 for binary segmentation).
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)
