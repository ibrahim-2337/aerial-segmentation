# Aerial Building Segmentation

Benchmarking U-Net, DeepLabV3+, and SegFormer for building extraction from georeferenced aerial imagery using the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

---

## Architecture Comparison

Evaluated on the held-out **West Tyrol** city (never seen during training).
Results are mean ± std across 2 independent seeds.

| Architecture | Encoder | IoU (mean ± std) | Dice (mean ± std) |
|---|---|---|---|
| U-Net | ResNet-34 | — | — |
| DeepLabV3+ | ResNet-50 | — | — |
| SegFormer-B2 | Mix Transformer B2 | — | — |

*Fill in after running `python src/evaluate.py --mode metrics`.*

---

## Sample Predictions

Five validation tiles from West Tyrol — **Input | Ground Truth | Prediction**:

| U-Net | DeepLabV3+ | SegFormer-B2 |
|---|---|---|
| ![](experiments/figures/unet_seed42_predictions.png) | ![](experiments/figures/deeplabv3plus_seed42_predictions.png) | ![](experiments/figures/segformer_seed42_predictions.png) |

---

## Project Structure

```
aerial-segmentation/
├── data/
│   └── inria/
│       └── train/
│           ├── images/    ← raw GeoTIFFs (RGB, 5000×5000)
│           └── gt/        ← binary building masks
├── src/
│   ├── dataset.py         ← rasterio dataloader with sliding-window tiling
│   ├── models.py          ← U-Net / DeepLabV3+ / SegFormer via smp + HF
│   ├── train.py           ← training loop with Drive checkpointing
│   ├── evaluate.py        ← metrics + visualisation
│   └── utils.py           ← seeds, tiling, losses, metrics
├── experiments/
│   ├── results.csv        ← per-seed IoU/Dice
│   └── figures/           ← prediction panels, bar charts, learning curves
├── notebooks/
│   └── visualization.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset Download

1. Register at [https://project.inria.fr/aerialimagelabeling/](https://project.inria.fr/aerialimagelabeling/) and accept the terms of use.
2. Download the **training set** archive (`NEW2-AerialImageDataset.zip`, ~15 GB).
3. Extract and place files so the layout matches `data/inria/train/{images,gt}/`.

```
data/inria/train/images/austin1.tif   austin2.tif   ...
data/inria/train/gt/austin1.tif       austin2.tif   ...
```

The expected city splits are:

| Split | Cities |
|---|---|
| Train | austin, chicago, kitsap, vienna |
| Validation | tyrol-w (West Tyrol) |

---

## Reproducing Results

### 1. Environment

```bash
git clone https://github.com/YOUR_USERNAME/aerial-segmentation.git
cd aerial-segmentation
pip install -r requirements.txt
```

### 2. Place the dataset

Follow the download instructions above and verify:

```bash
ls data/inria/train/images/ | head -5
# austin1.tif  austin2.tif  ...
```

### 3. Train all models

Run each model with both seeds (6 runs total, ~20 epochs each):

```bash
python src/train.py --model unet          --seed 42 --epochs 20
python src/train.py --model unet          --seed 0  --epochs 20
python src/train.py --model deeplabv3plus --seed 42 --epochs 20
python src/train.py --model deeplabv3plus --seed 0  --epochs 20
python src/train.py --model segformer     --seed 42 --epochs 20
python src/train.py --model segformer     --seed 0  --epochs 20
```

Add `--resume` to any command to continue from the last saved checkpoint.

On **Google Colab with A100**, open `notebooks/visualization.ipynb` — it mounts Google Drive and launches training automatically.

### 4. Evaluate and visualise

```bash
# Compute IoU / Dice for all checkpoints → experiments/results.csv
python src/evaluate.py --mode metrics

# Save side-by-side prediction panels → experiments/figures/
python src/evaluate.py --mode visualize

# Both at once
python src/evaluate.py --mode all
```

---

## Training Details

| Setting | Value |
|---|---|
| Input resolution | 256 × 256 px (tiled from 5000×5000 GeoTIFF) |
| Tile overlap | 50 % |
| Batch size | 16 |
| Epochs | 20 |
| Loss | BCE + Dice (0.5 / 0.5) |
| Optimiser | AdamW, lr=1e-4, wd=1e-4 |
| LR schedule | CosineAnnealingLR |
| Augmentation | H-flip, V-flip, 90° rot, ColorJitter, RandomCrop |
| Precision | AMP fp16 |
| Seeds | 42, 0 |

---

## Requirements

See `requirements.txt`. Key dependencies:

- `torch >= 2.0`
- `segmentation-models-pytorch >= 0.3.3`
- `transformers >= 4.35` (SegFormer)
- `rasterio >= 1.3` (GeoTIFF I/O with coordinate metadata)
- `albumentations >= 1.3` (augmentation)
