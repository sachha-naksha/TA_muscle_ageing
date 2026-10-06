#!/bin/bash
#SBATCH --job-name="metacell_fast2b_array"
#SBATCH --output="metacell_%A_%a.log"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -t 01:00:00
#SBATCH --array=0-5

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

SAMPLES=("M2_WT" "M4_KO" "M6_WT" "M11_WT" "M764_KO" "M784_KO")

# Get the sample ID for this specific array task
CURRENT_SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "Processing Array Task: $SLURM_ARRAY_TASK_ID"
echo "Target Sample ID: $CURRENT_SAMPLE"

# 2. Set Paths
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples"
INPUT_ADATA="${WORK_DIR}/adata_harmony_v2.h5ad"
BASE_RESULTS_DIR="${WORK_DIR}/analysis/meta_cell_fast2b_2x"


cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# ------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------

module load gcc/13.3.1-p20240614
module load anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate metasheller-py311

# ------------------------------------------------------------------
# RUN PYTHON
# ------------------------------------------------------------------

echo "Running metacell for $CURRENT_SAMPLE"

# We pass bash variables to python using sys.argv
python -c "
import sys
import os
import scanpy as sc
import metashells as ms
from metashells.seashells import SeaShells

target_sample = sys.argv[1]
base_out_dir = sys.argv[2]
adata_path = sys.argv[3]

print(f'Python script started for sample: {target_sample}')

# This prevents different jobs from overwriting the same files
results_dir = os.path.join(base_out_dir, target_sample)
os.makedirs(results_dir, exist_ok=True)

print(f'Loading data from {adata_path}...')
adata = sc.read_h5ad(adata_path)

if 'X_pca_harmony' in adata.obsm.keys():
    adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
else:
    print('Warning: X_pca_harmony not found, using existing X_pca')

print(f'Subsetting to sample_id == {target_sample}...')
adata_subset = adata[adata.obs['sample_id'] == target_sample].copy()
adata_subset = adata_subset[adata_subset.obs['C_scANVI'].isin(['Fast IIB', 'Fast IIX'])]
print(f'Subset shape: {adata_subset.shape}')

if adata_subset.n_obs == 0:
    print(f'Error: No cells found for sample {target_sample}. Check spelling.')
    sys.exit(1)

print('Initializing SeaShells...')
conch = SeaShells(
    adata_full = adata_subset,
    results_dir = results_dir,
    cells_per_metacell = 75,
    sample_col = 'sample_id' 
)

conch.run()
print('SeaShells run complete.')
" "$CURRENT_SAMPLE" "$BASE_RESULTS_DIR" "$INPUT_ADATA"

echo "Done with metacell for $CURRENT_SAMPLE"