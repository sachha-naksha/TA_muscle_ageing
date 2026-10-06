# Install required packages if not already installed
if (!require("devtools")) install.packages("devtools")
if (!require("yaml")) install.packages("yaml")

# Load libraries
library(devtools)
library(yaml)

# Install SLIDE from GitHub
devtools::install_github("jishnu-lab/SLIDE")


yaml_path = "/ix/djishnu/Akanksha/datasets/ERCC1_KO/metacell/slide_outs/optimize_slide/female_0.1_spec/0.01_0.1_out/yaml_params.yaml"
input_params <- yaml::yaml.load_file(yaml_path)
SLIDE::checkDataParams(input_params)

# SLIDE::optimizeSLIDE(input_params, sink_file = FALSE)

# SLIDE::plotCorrelationNetworks(input_params)

# Run full CV after selecting delta and lambda

SLIDE::SLIDEcv(yaml_path, nrep = 2000, k = 20)