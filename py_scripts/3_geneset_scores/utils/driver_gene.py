# =========================
# Imports
# =========================
import warnings

# Plotting helpers live in pygenelab.plotting (single source of truth).
from pygenelab.plotting import plot_heatmap, plot_violin_box_combo, display_plots_side_by_side
warnings.filterwarnings("ignore")

import io
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
from itertools import chain, repeat
from PIL import Image

from io import BytesIO
import base64
from IPython.display import HTML, display


# =========================
# GMT Parsing Functions
# =========================
def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """
    Parse a GMT file into a decoupler pathway dataframe.
    """

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )


def gmt_to_decoupler_multiple_pathways(
    gmt_paths,
    geneset_name=None,
    genesymbol_name=None,
):
    """
    Parse multiple GMT files and return a combined pathway dataframe.
    """

    all_records = []

    for pth in gmt_paths:
        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                all_records.extend(zip(repeat(name), genes))

    return pd.DataFrame.from_records(
        all_records,
        columns=[geneset_name, genesymbol_name],
    )


# =========================
# Gene Ranking
# =========================
def compute_gene_ranking(adata, geneset):
    """
    Compute Spearman correlation of each gene with pathway score
    and return ranked genes filtered by selected genes.
    """

    # Gene expression matrix
    X = adata.to_df()

    # AUCell pathway scores
    score_series = adata.obsm["score_aucell"].copy()

    # Compute correlations
    correlations = [
        spearmanr(X[gene], score_series)[0]
        for gene in X.columns
    ]

    # Create dataframe
    gene_importance = pd.DataFrame({
        "gene": X.columns,
        "spearman_corr": correlations
    })

    # Sort by correlation
    gene_importance = (
        gene_importance
        .sort_values("spearman_corr", ascending=False)
        .reset_index(drop=True)
    )

    # Add rank column
    gene_importance["rank"] = gene_importance.index + 1

    # Filter genes belonging to geneset
    ranked_gene_df = (
        gene_importance[gene_importance["gene"].isin(geneset)]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    return gene_importance, ranked_gene_df


# =========================
# Heatmap Plot
# =========================
# =========================
# Pairwise Significance
# =========================
def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    Calculate pairwise significance between all groups
    Returns a dictionary of p-values and significance levels
    """
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[data[x_var] == groups[i]][y_var]  # Changed from 'category' and 'senescence_score'
            group2 = data[data[x_var] == groups[j]][y_var]  # Changed from 'category' and 'senescence_score'
            
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Add significance stars
            if pvalue < 0.001:
                sig = '***'
            elif pvalue < 0.01:
                sig = '**'
            elif pvalue < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            results[(i, j)] = {'p-value': pvalue, 'significance': sig}
    
    return results
    

# =========================
# Violin + Box Plot Combo
# =========================
# =========================
# Combine Plots
# =========================
