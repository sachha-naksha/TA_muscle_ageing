#!/bin/bash
#SBATCH --job-name="slide_male_pooled"
#SBATCH --output="slide_male_pooled.log"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -t 02:00:00

# Set working directory where the python script is located
WORK_DIR="/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples"
cd $WORK_DIR || { echo "Error: Could not change directory to $WORK_DIR"; exit 1; }

# Load necessary modules and activate the environment
# Load GCC first to provide newer libstdc++ (GLIBCXX_3.4.29)
module load gcc/13.3.1-p20240614

module load anaconda3/2022.10
eval "$(conda shell.bash hook)"  # Proper conda initialization
conda activate loveslide_dev

# Define file paths as bash variables
X_PATH="/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples/male_fast2b_2x_harmonyv2_X.csv"
Y_PATH="/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples/male_fast2b_2x_harmonyv2_Y.csv"
OUT_PATH="/ocean/projects/cis240075p/asachan/datasets/TA_muscle/ERCC1_KO_mice/integrated_samples/analysis/male/slide_outs/pooled"

# Run pyslide
echo "Running slide for batch male_pooled"

########################## Python script ##########################
python -c "
import loveslide
from loveslide import OptimizeSLIDE

# Configure the parameters for the slide analysis
input_params = {
    'x_path': '${X_PATH}',
    'y_path': '${Y_PATH}',
    'fdr': 0.1,
    'thresh_fdr': 0.1,
    'spec': 0.3,
    'y_factor': True,
    'niter': 500,
    'SLIDE_top_feats': 20,
    'rep_CV': 50,
    'pure_homo': True,
    'delta': [0.01],
    'lambda': [0.1],
    'out_path': '${OUT_PATH}'
}

slider = OptimizeSLIDE(input_params)
slider.run_pipeline(verbose=True, n_workers=1)
"

# Log completion of the batch
echo "Done with slide for batch male"