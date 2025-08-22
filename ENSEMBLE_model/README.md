# ENSEMBLE_model

## Purpose
This directory contains all resources needed to run the ensemble-based cell segmentation pipeline.

## Contents
- **models/** — 14 pretrained models used in the ensemble.
- **TEST_DATA/**
  - **10_Lab_img/** — 10 test images.
  - **10_Lab_msk/** — Corresponding masks.
  - **LIVECELL_img_msk.zip** — Unzip to obtain: 
    - `LIVECELL_img/`
    - `LIVECELL_msk/`
- **RUNNER.ipynb** — Colab-notebook for running ensemble model.
- **customfunc.py** — Utility functions used by the runner.
