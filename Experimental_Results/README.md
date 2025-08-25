# Experimental_Results

## Structure

### 10_Lab
- **Inputs/** — 10 test images.
- **Ground_Truth/** — Corresponding masks.
- **Outputs/** — Outputs for the 10 test images by our Model.
- **Overlays/** — Predicted masks overlaid on inputs.
- **Comparision_with_other_models/** — Visual comparisons.
- **10_Lab_results.csv** — Metrics for the 10-image set.

### Comparision_with_other_models

- **10_Lab_img/** — Input images for comparison.  
- **10_Lab_msk/** — Ground truth masks for comparison.  
- **csv_results/** — Metrics for the model in csv.  
- **SSL_scripts/** — Supporting scripts for running SSL model.  
- **CellPose_Stardist.ipynb** — Runs CellPose-SAM, CellPose3 and StarDist.  
- **evalRes.ipynb** — Evaluates metrics across models.  
- **SSL.ipynb** — Runs SSL model and saves output.  

### LIVECELL

- **LIVECELL_Outputs_Overlay.zip** — Zip bundle containing:
  - `LIVECELL_outputs/` — Predicted masks for the full LIVECell dataset.
  - `LIVECELL_overlays/` — Overlays for the full LIVECell dataset.
- **LiveCell_results.csv** — Metrics for the full LIVECell dataset.

The original **LIVECell** images and ground truth masks can be obtained from the official **LIVECell** repository:  
[https://sartorius-research.github.io/LIVECell/](https://sartorius-research.github.io/LIVECell/)  
or their associated AWS storage links provided on that page.
