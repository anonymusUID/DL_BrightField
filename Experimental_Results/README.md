# Experimental_Results

## Structure

### 10_Lab
- **Inputs/** — 10 test images.
- **Ground_Truth/** — Corresponding masks.
- **Outputs/** — Model predictions for the 10 test images.
- **Overlays/** — Predicted masks overlaid on inputs.
- **Comparision_with_other_models/** — Visual comparisons.
- **10_Lab_results.csv** — Metrics for the 10-image set.

### LIVECELL
- **LIVECELL_Outputs_Overlay.zip** — Zip bundle containing:
  - `LIVECELL_outputs/` — Predictions for the full LIVECell dataset.
  - `LIVECELL_overlays/` — Overlays for the full LIVECell dataset.
- **LiveCell_results.csv** — Metrics for the full dataset.

## Notes
- The **10_Lab** section demonstrates per-image outputs for a small, interpretable subset.
- The **LIVECELL** section aggregates results for ~3,000 images. Unzip `LIVECELL_Outputs_Overlay.zip` to obtain `LIVECELL_outputs/` and `LIVECELL_overlays/`.