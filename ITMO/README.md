# ITMO

## Purpose
Data preparation and model training pipeline for brightfield live-cell segmentation.

## Layout
- **01_BRIGHT_FIELD/**
  - `original_img/`, `original_msk/` — Source images and masks.
  - `augmented_img/`, `augmented_msk/` — To be generated via augmentation.
  - `857x21/` — `.npy` feature/array files (to be generated).
- **03_DataPrep/**
  - `augmentation_bulk_5.ipynb` — Bulk augmentation.
  - `dgen_3.ipynb` — `.npy`/data generation.
- **02_Bright_Field/**
  - `Unet_Resnet_Vgg16_Densenet.py` and training notebooks (`MODEL*_*.ipynb`).
- **008_OUTPUT/** — Trained model artifacts (output target).

## Minimal Procedure
1. Place raw data in `01_BRIGHT_FIELD/original_img/` and `original_msk/`.
2. Run `03_DataPrep/augmentation_bulk_5.ipynb` → populate `augmented_img/` and `augmented_msk/`.
3. Run `03_DataPrep/dgen_3.ipynb` → generate `.npy` files under `01_BRIGHT_FIELD/857x21/`.
4. Train using notebooks in `02_Bright_Field/`. Models are saved to `008_OUTPUT/`.

## Environment
- Designed for Google Colab with Drive mounted.
- Ensure consistent base path configuration before running notebooks.