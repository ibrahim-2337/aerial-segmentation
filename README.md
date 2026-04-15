# Aerial Building Segmentation Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

A comparative study of state-of-the-art semantic segmentation architectures for building extraction from high-resolution georeferenced aerial imagery. This project benchmarks **U-Net**, **DeepLabV3+**, and **SegFormer** on the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

---

## 🚀 Overview

Extracting building footprints from aerial imagery is a critical task for urban planning, disaster response, and infrastructure monitoring. This repository provides a robust pipeline for:
- **Tiled Data Loading**: Handling large-scale GeoTIFFs (5000×5000 px) using sliding-window inference and rasterio.
- **Architecture Benchmarking**: Directly comparing CNN-based (U-Net, DeepLabV3+) and Transformer-based (SegFormer) models.
- **Production-Ready Training**: Integrated with mixed-precision training (AMP), learning rate scheduling, and automated checkpointing.

---

## 📊 Performance Benchmark

Evaluated on the **West Tyrol** city split (zero-shot transfer evaluation).
Results are reported as mean ± standard deviation across multiple seeds.

| Architecture | Encoder | IoU (%) | Dice (%) |
|---|---|---|---|
| **U-Net** | ResNet-34 | — | — |
| **DeepLabV3+** | ResNet-50 | — | — |
| **SegFormer-B2** | Mix Transformer B2 | — | — |

*To reproduce these results, run the full training suite described in the [Reproducing Results](#-reproducing-results) section.*

---

## 🖼️ Sample Predictions

Visual comparisons from West Tyrol — **Input | Ground Truth | Prediction**:

| U-Net | DeepLabV3+ | SegFormer-B2 |
|---|---|---|
| ![](experiments/figures/unet_seed42_predictions.png) | ![](experiments/figures/deeplabv3plus_seed42_predictions.png) | ![](experiments/figures/segformer_seed42_predictions.png) |

---

## 📂 Project Structure

```text
aerial-segmentation/
├── src/
│   ├── dataset.py         # Rasterio-based sliding-window loader
│   ├── models.py          # U-Net, DeepLabV3+, SegFormer architectures
│   ├── train.py           # Optimized training loop with AMP and CSV logging
│   ├── evaluate.py        # Metrics computation and visualization
│   └── utils.py           # Loss functions, IoU/Dice metrics, and helper logic
├── experiments/           # Results, figures, and model logs
├── notebooks/             # Exploratory analysis and Colab support
├── LICENSE                # MIT License
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Environment
```bash
git clone https://github.com/ibrahim-2337/aerial-segmentation.git
cd aerial-segmentation
pip install -r requirements.txt
```

### 2. Dataset Preparation
1. Register at [INRIA Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/).
2. Download the **training set** archive (~15 GB).
3. Extract files into the following directory structure:
```
data/inria/train/images/austin1.tif ...
data/inria/train/gt/austin1.tif ...
```

---

## 🧪 Reproducing Results

### Train all models
```bash
python src/train.py --model unet --epochs 20 --seed 42
python src/train.py --model deeplabv3plus --epochs 20 --seed 42
python src/train.py --model segformer --epochs 20 --seed 42
```

### Evaluate Performance
```bash
# Compute comprehensive metrics (IoU/Dice)
python src/evaluate.py --mode metrics

# Generate side-by-side prediction visualizations
python src/evaluate.py --mode visualize
```

---

## 🛠️ Requirements
- `torch >= 2.0`
- `segmentation-models-pytorch >= 0.3.3`
- `transformers >= 4.35`
- `rasterio >= 1.3`
- `albumentations >= 1.3`
