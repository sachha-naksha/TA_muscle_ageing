# Install required packages if not already installed
if (!require("devtools")) install.packages("devtools")
if (!require("yaml")) install.packages("yaml")

# Load libraries
library(devtools)
library(yaml)

# Install SLIDE from GitHub
devtools::install_github("jishnu-lab/SLIDE")


yaml_path = "/ix/djishnu/Akanksha/analysis_code/snRNA_TA_muscle_analysis/TA_muscle_code/R_scripts/config.yaml"
input_params <- yaml::yaml.load_file(yaml_path)
SLIDE::checkDataParams(input_params)

SLIDE::optimizeSLIDE(input_params, sink_file = FALSE)

SLIDE::plotCorrelationNetworks(input_params)

# Run full CV after selecting delta and lambda

# SLIDE::SLIDEcv(yaml_path, nrep = 2000, k = 20)