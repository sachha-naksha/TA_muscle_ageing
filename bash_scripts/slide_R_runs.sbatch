#!/bin/bash
#SBATCH --job-name=female_slide_cv
#SBATCH --output=female_cv.log
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G

set -euo pipefail

############################
# Use same R as RStudio
############################

R_BIN="/software/rhel9/manual/install/r/4.5.0/bin/Rscript"

# Optional sanity check (shows up in log)
$R_BIN --version
which $R_BIN

############################
# Run your script
############################

$R_BIN /ix/djishnu/Akanksha/analysis_code/snRNA_TA_muscle_analysis/TA_muscle_code/R_scripts/slide_runs.R

