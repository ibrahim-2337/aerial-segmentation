# Aerial Building Segmentation Benchmark

A benchmark comparing state-of-the-art architectures for building extraction from high-resolution aerial imagery. It evaluates **U-Net**, **DeepLabV3+**, and **SegFormer** on the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

## Summary
Building extraction is tough due to varying terrain, shadows, and roof types. This project provides a clean pipeline for training and evaluating models on large 5000×5000 GeoTIFFs using tiled sliding-window inference.

## Architectures
- **U-Net**: ResNet-34 encoder, ImageNet pretrained.
- **DeepLabV3+**: ResNet-50 encoder, ImageNet pretrained.
- **SegFormer-B2**: Mix Transformer B2, ADE20K pretrained (via HuggingFace).

## Quick Start
1.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset**: Download the INRIA training set and place the images in `data/inria/train/images/`.
3.  **Train**:
    ```bash
    python src/train.py --model unet --epochs 20 --seed 42
    ```
4.  **Evaluate**:
    ```bash
    # Compute IoU / Dice
    python src/evaluate.py --mode metrics
    # Generate visualization panels
    python src/evaluate.py --mode visualize
    ```

## Key Pipeline Features
- **Tiled Loading**: Uses Rasterio to handle large-scale GeoTIFFs with 50% tile overlap.
- **Optimization**: Uses `torch.compile` and mixed-precision (AMP) for faster training.
- **Logging**: Detailed CSV metrics and automatic checkpointing (including Drive support for Colab).

---
*Developed by Ibrahim Ahmad*
