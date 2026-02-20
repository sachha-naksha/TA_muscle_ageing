#!/bin/bash
#SBATCH --job-name="tf_ko_array"
#SBATCH --output="male_tf_ko_%A_%a.log"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -t 01:00:00
#SBATCH --array=0-14

# Load necessary modules and activate the environment
module load gcc/13.3.1-p20240614
module load anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate celloracle_env

# Define TF list
TFs=("Fli1" "Ets1" "Egr1" "Epas1" "Tcf7l1" "Atf3" "Klf8" "Klf2" "Nr4a3" "Nr2f2" "Ebf1" "Creb5" "Ppara" "Pbx3" "Npas2")

# Get TF for this array task
TF=${TFs[$SLURM_ARRAY_TASK_ID]}

echo "Running TF KO simulation for: $TF (Task ID: $SLURM_ARRAY_TASK_ID)"

# Run Python script with TF as argument
python - "$TF" << 'EOF'

import os
import shutil

# 1. Set environment variables BEFORE any imports that use SQLite/diskcache
os.environ['SQLITE_TMPDIR'] = '/tmp'
os.environ['TMPDIR'] = '/tmp'
os.environ['XDG_CACHE_HOME'] = '/tmp/cache_' + os.environ.get('USER', 'user')

# 2. Create the local cache directory
local_cache = os.environ['XDG_CACHE_HOME']
os.makedirs(local_cache, exist_ok=True)

# 3. Clean up old problematic cache locations
paths_to_remove = [
    os.path.expanduser("~/.cache/genomepy"),
    os.path.expanduser("~/.cache/diskcache"),
    os.path.expanduser("~/.local/share/genomepy"),
]

for path in paths_to_remove:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")

# 4. Disable genomepy caching entirely (optional but often helps)
os.environ['GENOMEPY_CACHE'] = 'false'

import sys
import celloracle as co
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

# Get TF from command line argument
goi = sys.argv[1]
print(f"Processing TF: {goi}", flush=True)

# Set up matplotlib for non-interactive use
plt.switch_backend('Agg')
plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams["savefig.dpi"] = 300

# Paths
wd = '/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples/grn_CO/female_type2'
out_path = os.path.join(wd, 'tf_ko_results')

# Create output directories
figures_path = os.path.join(out_path, 'figures')
metrics_path = os.path.join(out_path, 'metrics')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)

# Load oracle object
print("Loading oracle object...", flush=True)
oracle = co.load_hdf5(os.path.join(wd, "out_files/female_type2_fitted.celloracle.oracle"))

# Plot gene expression of TF in UMAP space
fig, ax = plt.subplots(figsize=[6, 6])
sc.pl.embedding(adata=oracle.adata, basis='X_umap', ax=ax, cmap="rainbow", color=[goi], s=50, show=False)
plt.savefig(os.path.join(figures_path, f"{goi}_expression_umap.png"), bbox_inches='tight')
plt.close()

# Simulate perturbation
print(f'Simulating perturbation for {goi}', flush=True)
oracle.simulate_shift(perturb_condition={goi: 0.0}, GRN_unit="cluster", n_propagation=2)

print(f'Estimating transition probability for {goi}', flush=True)
oracle.estimate_transition_prob(n_neighbors=750, knn_random=True, sampled_fraction=1)

print(f'Calculating embedding for {goi}', flush=True)
oracle.calculate_embedding_shift(sigma_corr=0.05)

# Calculate mass filter
min_mass = 30
fig, ax = plt.subplots()
oracle.calculate_mass_filter(min_mass=min_mass, plot=True)
plt.savefig(os.path.join(figures_path, f"{goi}_mass_filter.png"), bbox_inches='tight')
plt.close()

# Create combined visualization
scale_simulation = 15
fig, ax = plt.subplots(1, 3, figsize=[18, 6])

oracle.plot_cluster_whole(ax=ax[0], s=10)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0], show_background=False)

oracle.plot_cluster_whole(ax=ax[1], s=10)
ax[1].set_title(f"Randomized simulation shift vector: {goi} KO")
oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1], show_background=False)

ax[2].set_title("Clustered cell identity")
sc.pl.umap(oracle.adata, color=['condition'], ax=ax[2], show=False)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, f"{goi}_simulation_flow.png"), bbox_inches='tight')
plt.close()

# Quantify vector field
def quantify_vector_field(vector_field, return_magnitudes=False):
    if len(vector_field.shape) != 2 or vector_field.shape[1] != 2:
        raise ValueError("Input must be of shape (m, 2)")
    
    vx = vector_field[:, 0]
    vy = vector_field[:, 1]
    magnitudes = np.sqrt(vx**2 + vy**2)
    mag_flat = magnitudes.flatten()
    
    stats = {
        'mean': np.mean(mag_flat),
        'std': np.std(mag_flat),
        'median': np.median(mag_flat),
        'min': np.min(mag_flat),
        'max': np.max(mag_flat)
    }
    
    if return_magnitudes:
        stats['magnitudes'] = magnitudes
    
    return stats

# Calculate and save metrics
tf_ko_effect_size = quantify_vector_field(oracle.flow)

print("TF KO Effect Size (Vector Field Magnitude Statistics):")
for stat, value in tf_ko_effect_size.items():
    print(f"  {stat.capitalize()}: {value:.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'TF': [goi],
    'mean': [tf_ko_effect_size['mean']],
    'std': [tf_ko_effect_size['std']],
    'median': [tf_ko_effect_size['median']],
    'min': [tf_ko_effect_size['min']],
    'max': [tf_ko_effect_size['max']]
})
metrics_df.to_csv(os.path.join(metrics_path, f"{goi}_effect_size.csv"), index=False)

print(f"Completed processing for {goi}", flush=True)
EOF

echo "Finished TF KO for: $TF"