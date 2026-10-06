# transcriptional_noise.py

"""
functions for transcriptional noise (low-cv intra-group variability) analysis
"""

# imports
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from .utils import get_dense_matrix, get_low_cv_genes


# _noise_pvalue
def _noise_pvalue(data1, data2, test="mannwhitneyu"):
    """
    compute two-sided p-value for two noise samples.
    test: "mannwhitneyu" (rank-based, non-parametric, default),
          "ttest" (Student's t-test, equal variance),
          "welch" (Welch's t-test, unequal variance).
    """
    if test == "mannwhitneyu":
        return mannwhitneyu(data1, data2, alternative="two-sided").pvalue
    if test == "ttest":
        return ttest_ind(data1, data2, equal_var=True).pvalue
    if test == "welch":
        return ttest_ind(data1, data2, equal_var=False).pvalue
    raise ValueError(
        f"unknown test {test!r}; expected 'mannwhitneyu', 'ttest', or 'welch'"
    )


# calculate_noise_one_celltype
def calculate_noise_one_celltype(
    adata_ct,
    condition_col="condition",
    group1="WT",
    group2="KO",
    layer=None,
    max_cells=300,
    min_cells=10,
    n_bins=10,
    bottom_frac=0.10,
    random_state=1,
    test="mannwhitneyu"
):
    """
    calculate transcriptional noise for one cell type
    """

    # calculate_noise_one_celltype
    # api:
    # noise_df, summary = calculate_noise_one_celltype(
    #     adata_ct=adata_ct,
    #     condition_col="condition",
    #     group1="WT",
    #     group2="KO",
    #     layer=None,
    # )

    rng = np.random.default_rng(random_state)

    # get cells from each condition
    group1_cells = adata_ct.obs_names[
        adata_ct.obs[condition_col] == group1
    ].to_numpy()

    group2_cells = adata_ct.obs_names[
        adata_ct.obs[condition_col] == group2
    ].to_numpy()

    # sample equal number of cells
    n_sample = min(len(group1_cells), len(group2_cells), max_cells)

    if n_sample < min_cells:
        summary = {
            "status": "skipped",
            "reason": "not enough cells",
            f"n_{group1}_available": len(group1_cells),
            f"n_{group2}_available": len(group2_cells)
        }

        return None, summary

    group1_sample = rng.choice(
        group1_cells,
        size=n_sample,
        replace=False
    )

    group2_sample = rng.choice(
        group2_cells,
        size=n_sample,
        replace=False
    )

    selected_cells = np.concatenate([group1_sample, group2_sample])
    adata_sub = adata_ct[selected_cells, :].copy()

    # get normalized expression matrix
    if layer is None:
        X = get_dense_matrix(adata_sub.X).astype(float)
    else:
        X = get_dense_matrix(adata_sub.layers[layer]).astype(float)

    # select low-cv genes
    low_cv_genes = get_low_cv_genes(
        X=X,
        gene_names=adata_sub.var_names,
        n_bins=n_bins,
        bottom_frac=bottom_frac
    )

    if len(low_cv_genes) == 0:
        summary = {
            "status": "skipped",
            "reason": "no low-cv genes found",
            f"n_{group1}_available": len(group1_cells),
            f"n_{group2}_available": len(group2_cells)
        }

        return None, summary

    # subset to low-cv genes
    gene_idx = adata_sub.var_names.get_indexer(low_cv_genes)
    X_lowcv = X[:, gene_idx]

    labels = adata_sub.obs[condition_col].to_numpy()
    cell_ids = adata_sub.obs_names.to_numpy()

    # split expression by condition
    X_group1 = X_lowcv[labels == group1]
    X_group2 = X_lowcv[labels == group2]

    ids_group1 = cell_ids[labels == group1]
    ids_group2 = cell_ids[labels == group2]

    # calculate average profile for each group
    mean_group1 = X_group1.mean(axis=0)
    mean_group2 = X_group2.mean(axis=0)

    # calculate euclidean distance from own group average
    noise_group1 = np.linalg.norm(X_group1 - mean_group1, axis=1)
    noise_group2 = np.linalg.norm(X_group2 - mean_group2, axis=1)

    # save per-cell results
    noise_df = pd.DataFrame({
        "cell_id": np.concatenate([ids_group1, ids_group2]),
        "condition": [group1] * len(noise_group1) + [group2] * len(noise_group2),
        "noise": np.concatenate([noise_group1, noise_group2])
    })

    # compare noise distributions
    pval = _noise_pvalue(noise_group1, noise_group2, test=test)

    mean_noise_group1 = noise_group1.mean()
    mean_noise_group2 = noise_group2.mean()

    log2_ratio = np.log2(
        (mean_noise_group2 + 1e-12) /
        (mean_noise_group1 + 1e-12)
    )

    summary = {
        "status": "done",
        f"n_{group1}": n_sample,
        f"n_{group2}": n_sample,
        f"mean_noise_{group1}": mean_noise_group1,
        f"mean_noise_{group2}": mean_noise_group2,
        f"log2_{group2}_over_{group1}": log2_ratio,
        "pval": pval,
        "test": test,
        "n_low_cv_genes": len(low_cv_genes)
    }

    return noise_df, summary
