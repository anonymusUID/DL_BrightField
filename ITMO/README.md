# ITMO

## Purpose
Data preparation and model training pipeline for brightfield live-cell segmentation.

## Layout
- **01_BRIGHT_FIELD/**
  - `original_img/`, `original_msk/` — Original images and masks.
  - `augmented_img/`, `augmented_msk/` — to contain augmented images and masks generated via `03_DataPrep/augmentation_bulk_5.ipynb`.
  - `857x21/` — to contain `".npy "` files generated via `03_DataPrep/dgen_3.ipynb`.
- **03_DataPrep/**
  - `augmentation_bulk_5.ipynb` — Bulk augmentation.
  - `dgen_3.ipynb` — `".npy"` data generation.
- **02_Bright_Field/**
  - training notebook
- **008_OUTPUT/** — Trained models and associated data
