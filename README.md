# Brightfield Microscopy Segmentation

## Overview
This repository provides data preparation, training scripts, and an ensemble-model for segmenting live-cell brightfield images. It includes the 10-image test set (10_Lab) and evaluation on the full LIVECell dataset (~3,000 images).

---

## How to Use
1. **Download/clone** this repository.
2. **Upload the three top-level folders** to "MyDrive" Section of your Google Drive:
   - `ITMO/`
   - `ENSEMBLE_model/`
   - `Experimental_Results/`
---

## Directory Summary
**Notation:**  
- Folder names end with `/`  
- Descriptions follow a dash `-`

**Structure:**
- **ITMO/** — Data preparation and training.
- **ENSEMBLE_model/** — Ensemble model, runner notebook, and test data package.
- **Experimental_Results/** — Organized outputs, overlays, comparisons, CSVs, and scripts for generating results with other models (CellPose, StarDist,SSL).

See subdirectory READMEs for details.

---
