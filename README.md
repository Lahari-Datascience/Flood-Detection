# Flood Detection using Satellite Imagery — AISEHack 2026 (Theme 1, Phase 2)

A deep learning pipeline for **3-class flood segmentation** in West Bengal, India, built on multi-spectral and SAR satellite imagery. Developed for Phase 2 of the AISEHack hackathon, extending the Phase 1 binary flood-detection task to distinguish actual flood inundation from pre-existing water bodies.

## Overview

Phase 2 moves beyond simple water/no-water detection to a more realistic disaster-response setting. Every pixel is classified into one of three categories:

| Class | Label |
|---|---|
| `0` | No Flood |
| `1` | Flood |
| `2` | Water Body |

Separating **flood** from **permanent/seasonal water bodies** is critical for accurate disaster mapping — treating all water as "flood" (as in Phase 1) overestimates damage and misguides response efforts.

This project builds a semantic segmentation model that fuses SAR and optical satellite bands with engineered spectral indices to make this distinction.

## Data

- **Source:** Provided by AISEHack, following the same 512×512 patch structure as Phase 1, with enhanced pixel-level 3-class annotations.
- **Bands per image:** 6 channels — SAR (VV, VH) + Optical (Green, Red, NIR, SWIR).
- Related IBM/NASA foundation model: [Prithvi (IBM-NASA Geospatial)](https://huggingface.co/ibm-nasa-geospatial)
- Reference fine-tuning tooling: [TerraTorch](https://github.com/terrastackai/terratorch)
- Hackathon helper code: [AISEHack_Edition1_2026](https://github.com/AISEHack/AISEHack_Edition1_2026)

> **Note:** Raw dataset is not included in this repo (competition data). See the AISEHack competition page for access instructions.

## Approach

### 1. Feature Engineering
Beyond the 6 raw bands, 5 spectral indices are computed per patch to boost water/flood discrimination, giving **11 input channels** total:

- **NDWI** — Normalized Difference Water Index
- **MNDWI** — Modified NDWI (better performance in urban/built-up areas)
- **NDVI** — Normalized Difference Vegetation Index
- **NDMI** — Normalized Difference Moisture Index
- **SAR Ratio** — VV/VH ratio

All indices are clipped to `[-1, 1]` and each of the 11 channels is normalized (zero mean, unit variance) per sample.

### 2. Model Architecture
- **Backbone:** `maxvit_tiny_tf_512` (hybrid CNN-Transformer encoder) via `timm`, ImageNet-pretrained
- **Decoder:** U-Net++ with `scSE` (spatial + channel squeeze-excitation) attention, via [`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
- **Input channels:** 11 | **Output classes:** 3

### 3. Loss Function
A hybrid loss combining:
- **Dice Loss** (0.4 weight) — optimizes region overlap / IoU
- **Focal Loss** (0.6 weight) — focuses learning on hard-to-classify pixels

### 4. Training
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
- LR Schedule: OneCycleLR (`max_lr=3e-4`)
- Mixed-precision training (`torch.cuda.amp`)
- Epochs: 40 | Batch size: 4
- Augmentations (via Albumentations): random 90° rotation, horizontal/vertical flips, affine (scale/translate/rotate), and grid/optical distortion

### 5. Inference — Test-Time Augmentation (TTA)
4-way TTA averaging softmax predictions over:
- Original
- Horizontal flip
- Vertical flip
- Horizontal + Vertical flip

### 6. Output Format
Final predictions are encoded as **run-length encoded (RLE) masks** for the **Flood class (class 1) only**, written to `submission.csv` with columns `id, rle_mask`.

## Repository Structure

```
.
├── notebook.ipynb        # End-to-end pipeline: data loading, feature engineering,
│                          # model, training loop, TTA inference, submission generation
└── README.md
```

## Requirements

```bash
pip install segmentation_models_pytorch rasterio albumentations torch pandas numpy tqdm
```

- Python 3.12
- GPU recommended (developed/tested on NVIDIA Tesla T4)

## Usage

1. Set up the expected data directory structure:
   ```
   data/
   ├── image/       # training images (*_image.tif)
   ├── label/       # training labels (*_label.tif)
   ├── prediction/
   │   └── image/   # inference images
   └── split/
       ├── train.txt
       └── pred.txt
   ```
2. Update the `BASE` path in the notebook to point to your data directory.
3. Run all cells. Training runs for 40 epochs, followed by TTA inference.
4. Predictions are saved to `submission.csv`.

## Results

LeaderBoard Position-56 and Score-0.1102

## Acknowledgements

- [AISEHack](https://github.com/AISEHack/AISEHack_Edition1_2026) organizers for the dataset and starter tooling
- [IBM-NASA Geospatial](https://huggingface.co/ibm-nasa-geospatial) for the Prithvi foundation model
- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) for the U-Net++ implementation
- [TerraTorch](https://github.com/terrastackai/terratorch) for geospatial fine-tuning tooling


