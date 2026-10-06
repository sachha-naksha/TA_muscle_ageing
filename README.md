# TA muscle ageing: ERCC1 Skm-KO snRNA-seq analysis

Code for the single-nucleus analysis of tibialis anterior (TA) muscle from skeletal-muscle-specific
ERCC1 knockout (Skm-KO) and Control mice (3 replicates x genotype x sex, 12 samples), processed in
two batches and integrated, and compared with the public human skeletal-muscle ageing atlas
(Lai et al., *Nature* 2024). The repository covers preprocessing and
integration, cell-type annotation, gene-set activity, SLIDE latent-factor
analysis, and cross-species transfer.


## Related repositories

This analysis spans three code bases. All three are needed to reproduce the full manuscript.

| Component | Language | Where | Used for |
|---|---|---|---|
| This repository | Python, R, bash | https://github.com/sachha-naksha/TA_muscle_ageing | Everything except the two items below |
| SLIDE | **R** | https://github.com/jishnu-lab/SLIDE | Latent-factor discovery and cross-validation (`scripts/4_slide_analysis/slide_runs.R`) |
| maxtoki-perturb | Python | https://github.com/sachha-naksha/maxtoki-perturb | MaxToki in-silico gene perturbation (not run from this repo) |

Pinned versions used for the manuscript:

- SLIDE: `<commit/tag>` <!-- TODO: pin -->
- maxtoki-perturb: `<commit/tag>` <!-- TODO: pin -->
- This repository: `<commit/tag/Zenodo DOI>` <!-- TODO: pin at submission -->
- Data: [10.5281/zenodo.22651313](https://doi.org/10.5281/zenodo.22651313) v1.0.0

---

## 1. System requirements

### Software dependencies

No single lockfile is shipped yet; the analysis used separate environments per stage. Versions below are those recorded in the code; fill the
remaining ones from `pip freeze` / `sessionInfo()`. <!-- TODO: export environment files -->

| Stage | Environment | Key packages |
|---|---|---|
| Preprocessing, integration, annotation, gene-set scoring (`scripts/1_*` - `3_*`, `6_*`) | Python 3.12 | scanpy, anndata, scvi-tools 1.4.0, torch, scrublet, decoupler, gseapy, statsmodels, scikit-learn, seaborn, plotnine, `pygenelab` (local, see below) |
| Metacells (`scripts/slurm/metacell.sbatch`) | Python 3.11 (`metasheller-py311`) | scanpy, metashells |
| **SLIDE, R (primary run, `scripts/4_slide_analysis/slide_runs.R`)** | **R 4.5.0** | SLIDE (GitHub: `jishnu-lab/SLIDE`), devtools, yaml |
| Seurat analyses (`.Rmd` in `1_preproc/`, `2_single_nuc_inspection/`, `5_human_skm_multimodal/`) | R | Seurat, harmony, SingleR, SummarizedExperiment, ggplot2, dplyr, patchwork, openxlsx, EnhancedVolcano, MuDataSeurat, reticulate |

Exact package versions: <!-- TODO: fill from environment export -->

### Operating systems and tested versions

- Linux (RHEL 9), SLURM cluster. <!-- TODO: confirm OS/kernel versions -->
- Interactive notebooks also run on macOS (Apple silicon) for the lighter stages. <!-- TODO: confirm; list tested versions -->

### Non-standard hardware

- Preprocessing/scVI integration benefits from a GPU / 64+ cores; SLURM scripts request
  64 cores (`RM-shared`) or 64 GB RAM. <!-- TODO: state minimum RAM/GPU for the demo vs. full data -->
- The demo (section 3) needs no non-standard hardware. <!-- TODO: confirm once the demo dataset exists -->

---

## 2. Installation guide

```bash
git clone git@github.com:sachha-naksha/TA_muscle_ageing.git
cd TA_muscle_ageing
```

**Python environment (stages 1-3, 6)**

```bash
conda create -n ta_muscle python=3.12 -y
conda activate ta_muscle
pip install scanpy anndata scvi-tools==1.4.0 decoupler gseapy statsmodels scikit-learn seaborn plotnine
pip install -e scripts/pygenelab        # local helper library; needs a pyproject.toml, see TODO
```

<!-- TODO: scripts/pygenelab has no pyproject.toml/setup.py, so `pip install -e` will not work yet.
     Add one (or document `export PYTHONPATH=$PWD/scripts`). -->

**R and SLIDE (required for the latent-factor analysis)**

```r
install.packages(c("devtools", "yaml"))
devtools::install_github("jishnu-lab/SLIDE")   # R >= 4.5.0 was used
```

**MaxToki perturbation** is installed separately; see https://github.com/sachha-naksha/maxtoki-perturb.

Typical install time on a normal desktop computer: <!-- TODO: measure, e.g. "~X min for Python env, ~Y min for SLIDE" -->

---

## 3. Demo

### Data

Analysis-ready AnnData objects (CC-BY-4.0): **https://doi.org/10.5281/zenodo.22651313** (v1.0.0, requires `anndata >= 0.11`).

| File | Size | MD5 | Contents |
|---|---|---|---|
| `human_female_adata.h5ad` | 383.7 MB | `232ec56035f3397fc1375a43088cbf45` | 3,989 female type II myofiber nuclei, young (34 y) vs. older (80 y) donors; subpopulations, pseudotime, DNA-repair gene-set scores |
| `mice_adata.h5ad` | 3.2 GB | `cb604a03f73ccf40e954994bac924dfa` | 86,641 nuclei x 3,000 HVGs, TA muscle, 12 samples, 2 batches; annotations, normalized layers, embeddings |

The human nuclei derive from Lai et al., *Nature* 629:154-164 (2024); cite that atlas when using them.

```bash
mkdir -p data && cd data
curl -L -O https://zenodo.org/records/22651313/files/human_female_adata.h5ad
md5sum human_female_adata.h5ad    # macOS: md5 human_female_adata.h5ad
```

### Run the demo

The human female type II subset is the recommended demo (smaller, single sex). Notebooks to run on it:
`scripts/5_human_skm_multimodal/transfer_learning.ipynb` and `activity_score_trends.ipynb`.
<!-- TODO: confirm which notebooks/paths read human_female_adata.h5ad, and point their input cell at data/ -->

- **Expected output:** <!-- TODO: list figures/tables produced, with filenames -->
- **Expected run time on a normal desktop computer:** <!-- TODO: measure -->
- A smaller (~few-MB) subsample for a quick smoke test is still worth adding under `demo/`, since the journal asks for a *small* dataset. <!-- TODO optional -->

---

## 4. Instructions for use

### Repository layout

Scripts are organised by analysis stage; each numbered folder holds both its Python notebooks and R scripts.

```
TA_muscle_ageing/
├── README.md
├── LICENSE                                   Apache-2.0
└── scripts/
    ├── slurm/                                SLURM job scripts
    │   ├── metacell.sbatch                   metacells per sample (metashells)
    │   └── slide_R_runs.sbatch               SLIDE (R) cross-validation
    ├── pygenelab/                            helper library (Python)
    │   ├── data.py, utils.py, images.py, plotting.py
    │   ├── geneset_activity.py, deg_functional_enrichment.py, llm_categorize.py
    │   ├── transcriptional_noise.py, pseudotime_animation.py
    │   └── LF_viz.py, crossprediction.py
    ├── 1_preproc/                            QC, doublets, scVI/scANVI integration
    │   ├── ref_preproc.ipynb, query_preproc.ipynb, integration.ipynb
    │   ├── SKM_human_intercostal.ipynb
    │   ├── process_scRNA_QC_embed_seurat.Rmd                       [R]
    │   └── utils/public_atlas.py
    ├── 2_single_nuc_inspection/              annotation, composition, DEGs, heterogeneity
    │   ├── re_cluster.ipynb, snRNA_related.ipynb
    │   ├── Transcriptional_Heterogeneity.ipynb, DEGs_Volcano_Dotplot.ipynb
    │   ├── batch_effect_DEGs_analysis_seurat.Rmd                   [R]
    │   └── MF_DEGs_analysis_seurat.Rmd                             [R]
    ├── 3_geneset_scores/                     AUCell gene-set activity, DEG enrichment
    │   ├── geneset_activity.ipynb, DEG_Functional_Enrichment.ipynb
    │   └── msigdb metabolism enriched pathways mice/*.csv
    ├── 4_slide_analysis/                     SLIDE input prep (Python); SLIDE runs and plots (R)
    │   ├── meta_cell.ipynb, prep_data_slide.ipynb, LF_viz.ipynb
    │   ├── slide_runs.R                                            [R]  SLIDE CV
    │   ├── config_optimize_slide.yaml                              SLIDE parameters
    │   ├── slideCV_boxplot.Rmd                                     [R]
    │   └── slideLF_plot.Rmd                                        [R]
    ├── 5_human_skm_multimodal/               transfer to human skeletal muscle
    │   ├── rds_to_adata.Rmd                                        [R]
    │   ├── transfer_learning.ipynb, activity_score_trends.ipynb
    └── _archive/                             scratch, not used for results
```

### Running on your own data

Run the stages in order. Each notebook has a paths cell at the top; edit it for your data.

1. **Preprocess and integrate**: `scripts/1_preproc/` (Cell Ranger + CellBender output to a scVI/scANVI-integrated `.h5ad`).
2. **Annotate and inspect**: `scripts/2_single_nuc_inspection/`.
3. **Gene-set activity**: `scripts/3_geneset_scores/` (uses `pygenelab.geneset_activity`).
4. **Metacells and SLIDE input (Python, preprocessing only)**: `scripts/slurm/metacell.sbatch`, then `scripts/4_slide_analysis/prep_data_slide.ipynb`
   writes the `*_X.csv` (metacell x gene) and `*_Y.csv` (labels) files.
5. **SLIDE (R; all SLIDE runs are done in R)**: edit `scripts/4_slide_analysis/config_optimize_slide.yaml` (`x_path`, `y_path`, `out_path`, `delta`, `lambda`, `spec`, ...), then
   ```r
   library(SLIDE)
   input_params <- yaml::yaml.load_file("scripts/4_slide_analysis/config_optimize_slide.yaml")
   SLIDE::checkDataParams(input_params)
   SLIDE::optimizeSLIDE(input_params, sink_file = FALSE)   # grid over delta/lambda
   SLIDE::SLIDEcv("<out_path>/yaml_params.yaml", nrep = 2000, k = 20)   # final CV
   ```
   On SLURM: `sbatch scripts/slurm/slide_R_runs.sbatch` (edit the script path first).
   See the [SLIDE repository](https://github.com/jishnu-lab/SLIDE) for parameter documentation.
6. **MaxToki perturbation**: see [maxtoki-perturb](https://github.com/sachha-naksha/maxtoki-perturb).

Absolute paths in `scripts/slurm/` and some notebooks point to our cluster storage
(`/ocean/...`, `/ix/...`); replace them with your own.

### Reproduction instructions (optional)

| Manuscript item | Code |
|---|---|
| Fig. 1A, F: integration and UMAP | `scripts/1_preproc/integration.ipynb`, `2_single_nuc_inspection/re_cluster.ipynb` |
| Fig. 1B, C, E: composition, markers | `scripts/2_single_nuc_inspection/snRNA_related.ipynb` |
| Fig. 1D: transcriptional heterogeneity | `scripts/2_single_nuc_inspection/Transcriptional_Heterogeneity.ipynb`, `pygenelab/transcriptional_noise.py` |
| Fig. 1G: gene-set activity | `scripts/3_geneset_scores/geneset_activity.ipynb` |
| SLIDE latent factors | `scripts/4_slide_analysis/slide_runs.R`, `slideCV_boxplot.Rmd`, `slideLF_plot.Rmd`, `4_slide_analysis/LF_viz.ipynb` |

Processed data: [10.5281/zenodo.22651313](https://doi.org/10.5281/zenodo.22651313). Raw sequencing data: <!-- TODO: GEO accession -->. The only public dataset is the human skeletal-muscle ageing atlas (Lai et al., *Nature* 629:154-164, 2024).

---

## License

Apache License 2.0; see [LICENSE](LICENSE).

## Citation

<!-- TODO: add manuscript citation / Zenodo DOI -->
