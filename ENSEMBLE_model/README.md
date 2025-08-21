# ENSEMBLE_model

## Contents
- **models/** — 14 pretrained models used in the ensemble.
- **TEST_DATA/**
  - **10_Lab_img/** — 10 test images.
  - **10_Lab_msk/** — Corresponding masks.
  - **LIVECELL_img_msk.zip** — Unzip to obtain:
    - `LIVECELL_img/`
    - `LIVECELL_msk/`
- **RUNNER.ipynb** — Colab-friendly notebook for ensemble inference.
- **customfunc.py** — Utility functions used by the runner.

## Usage
1. Ensure this folder is in Google Drive (e.g., `MyDrive/brightfield_seg/ENSEMBLE_model`).
2. Unzip `TEST_DATA/LIVECELL_img_msk.zip` so that `LIVECELL_img/` and `LIVECELL_msk/` sit under `TEST_DATA/`.
3. Open `RUNNER.ipynb` in Colab, mount Drive, set the base path, and run all cells.
4. Outputs can be saved/compared against `Experimental_Results/`.