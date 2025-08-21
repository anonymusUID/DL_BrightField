# Brightfield Microscopy Segmentation — LIVECell

## Overview
This repository provides data preparation, training scripts, and an ensemble-based inference pipeline for segmenting live-cell brightfield images. It includes a 10-image test set and evaluation on the full LIVECell dataset (~3,000 images).

---

## How to Use (Colab + Google Drive)
1. **Download/clone** this repository.
2. **Upload the three top-level folders** to your Google Drive (e.g., `MyDrive/brightfield_seg/`):
   - `ITMO/`
   - `ENSEMBLE_model/`
   - `Experimental_Results/`
3. In Google Colab:
   - Mount Drive and set your base path (e.g., `BASE_DIR = "/content/drive/MyDrive/brightfield_seg"`).
   - Follow **Model Training** or **Ensemble Inference** as needed.

---

## Workflows

### A) Model Training (optional)
- Use `ITMO/03_DataPrep/` to generate augmentations and `.npy` files.
- Source data layout:
  - `ITMO/01_BRIGHT_FIELD/original_img/` and `original_msk/`
  - Generate `augmented_img/`, `augmented_msk/`, and `857x21/` (`.npy`).
- Train using notebooks/scripts in `ITMO/02_Bright_Field/`.
- Trained models are saved to `ITMO/008_OUTPUT/`.

### B) Ensemble Inference (recommended for quick evaluation)
- Pretrained models are provided in `ENSEMBLE_model/models/`.
- Open `ENSEMBLE_model/RUNNER.ipynb` in Colab and run cells.
- Test data:
  - `ENSEMBLE_model/TEST_DATA/10_Lab_img/` and `10_Lab_msk/`
  - `ENSEMBLE_model/TEST_DATA/LIVECELL_img_msk.zip` → unzip to:
    - `LIVECELL_img/`, `LIVECELL_msk/`
- Inference outputs and overlays can be compared with `Experimental_Results/`.

---

## Directory Summary
- **ITMO/** — Data prep and training.
- **ENSEMBLE_model/** — Ensemble models, runner notebook, and test data package.
- **Experimental_Results/** — Organized outputs, overlays, comparisons, and CSV summaries.

See subdirectory READMEs for details.

---