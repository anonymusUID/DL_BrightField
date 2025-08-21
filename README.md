# Brightfield Microscopy Segmentation

## Overview
This repository provides scripts, models, and results for segmenting live-cell images captured using brightfield microscopy. It includes data preparation pipelines, model training scripts, and an ensemble-based inference framework evaluated on a test set and the full LIVECell dataset (~3,000 images).

---

## Repository Usage
1. **Clone or download** this repository.
2. Upload the following main directories to "MyDrive" in your Google Drive:
   - `ITMO/`
   - `ENSEMBLE_model/`
   - `Experimental_Results/`
3. Follow the respective sections below for training or inference.

---

## Directory Structure

### `Experimental_Results`
Contains segmentation outputs and overlays.

- **10_Lab** – Results for 10 test images.
- **LIVECELL** – Results for the full LIVECell dataset.
  - `LIVECELL_outputs_overlays.zip`  
    - Extracts into:  
      - `LIVECELL_outputs/` – Model predictions.  
      - `LIVECELL_overlays/` – Overlays of predictions on original images.  
  - `LIVECELL/` – Contains zipped dataset outputs and overlays.

---

### `ITMO`
Contains scripts and intermediate outputs for **data preparation and model training**.

- **008_OUTPUT/** – Trained model files.
- **03_DataPrep/** – Scripts for augmentation and `.npy` generation.
- **01_BRIGHT_FIELD/**  
  - `original_msk/` – Original masks.  
  - `original_img/` – Original images.  
  - `augmented_msk/` – To be generated.  
  - `augmented_img/` – To be generated.  
- **857x21/** – `.npy` files (to be generated).  
- **02_bright_field/** – Model training and evaluation scripts.

---

### `ENSEMBLE_model`
Contains resources for **ensemble inference**.

- **models/** – 14 pre-trained models used in the ensemble.
- **TEST_DATA/**  
  - Contains:
    - `10_test/` – 10 test images.  
    - `LIVECELL_dataset.zip` – Extracts into:
      - `LIVECELL_img/` – LIVECell images.  
      - `LIVECELL_msk/` – LIVECell masks.  
  - `RUNNER.ipynb` – Script to run ensemble inference.

---

## Workflow

### **1. Model Training**
- Prepare data using scripts in `ITMO/03_DataPrep/`.
- Generate augmented images and masks in `01_BRIGHT_FIELD/`.
- Generate `.npy` files in `857x21/`.
- Train models using scripts in `02_bright_field/`.
- Trained models are saved in `008_OUTPUT/`.

### **2. Ensemble Inference**
- Use pre-trained models from `ENSEMBLE_model/models/`.
- Run inference using `ENSEMBLE_model/TEST_DATA/RUNNER.ipynb`.
- Outputs and overlays are saved in `Experimental_Results/`.

---

## Notes
- Augmented datasets and `.npy` files must be generated before training.
- LIVECELL dataset and their respective outputs have been provided in zip format.
---