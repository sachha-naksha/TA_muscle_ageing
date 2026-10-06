# utils.py 

"""
all utility functions
"""

# imports
from pathlib import Path
from itertools import chain, repeat

import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib.pyplot as plt
from scipy.stats import rankdata


# convert_gene_case
def convert_gene_case(gene_list, direction="human_to_mouse"):
    """
    convert gene symbols between human (UPPER) and mouse (Title) case.
    direction: "human_to_mouse" / "h2m" or "mouse_to_human" / "m2h".
    """
    d = direction.lower()
    if d in ("human_to_mouse", "h2m"):
        return [str(g).capitalize() for g in gene_list]
    if d in ("mouse_to_human", "m2h"):
        return [str(g).upper() for g in gene_list]
    raise ValueError(
        f"invalid direction: {direction}. "
        "use 'human_to_mouse'/'h2m' or 'mouse_to_human'/'m2h'"
    )


# convert_gmt_to_decoupler_format
def convert_gmt_to_decoupler_format(
    pth: Path,
    include_pathways=None,
    gene_origin="mice"
) -> pd.DataFrame:
    """
    convert .gmt file paths to decoupler input format
    """

    # convert_gmt_to_decoupler_format
    # api:
    # convert_gmt_to_decoupler_format(
    #     pth=gmt_path,
    #     include_pathways=["PATHWAY_1", "PATHWAY_2"],
    #     gene_origin="mice"
    # )

    # check gene origin
    if gene_origin not in ["mice", "human"]:
        raise ValueError("gene_origin must be either 'mice' or 'human'")

    # make pathway filter set
    if include_pathways is not None:
        include_pathways = set(include_pathways)

    # dictionary to store selected pathways
    pathways = {}

    # open .gmt path and get pathway: genes
    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")

            # skip pathways not in selected list
            if include_pathways is not None and name not in include_pathways:
                continue

            # format gene names
            if gene_origin == "mice":
                genes = [gene.capitalize() for gene in genes]

            elif gene_origin == "human":
                genes = [gene.upper() for gene in genes]

            pathways[name] = genes

    # decoupler accepts "source" for pathway and "target" for genes
    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["source", "target"],
    )


# calculate_pairwise_significance
def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    calculate pairwise mann-whitney significance between groups
    """

    results = {}

    # compare every pair of groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):

            # get values for each group
            group1 = data[data[x_var] == groups[i]][y_var].dropna()
            group2 = data[data[x_var] == groups[j]][y_var].dropna()

            # skip if one group is empty
            if len(group1) == 0 or len(group2) == 0:
                results[(i, j)] = {
                    "p-value": None,
                    "significance": "ns"
                }
                continue

            # run mann-whitney u test
            statistic, pvalue = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            # assign significance stars
            if pvalue < 0.001:
                sig = "***"
            elif pvalue < 0.01:
                sig = "**"
            elif pvalue < 0.05:
                sig = "*"
            else:
                sig = "ns"

            # store result using group positions
            results[(i, j)] = {
                "p-value": pvalue,
                "significance": sig
            }

    # return pairwise results
    return results


# top_n_intersecting_genes_from_degs
def top_n_intersecting_genes_from_degs(
    deg_dfs,
    n=10,
    gene_col="names",
    rank_by="logfoldchanges",
    ascending=False,
    df_names=None
):
    """
    find top n shared genes across multiple deg dataframes

    genes are ranked by the average value of rank_by across all dataframes
    """

    # top_n_intersecting_genes_from_degs
    # api:
    # top_n_intersecting_genes_from_degs(
    #     deg_dfs=[deg_df1, deg_df2, deg_df3],
    #     n=10,
    #     gene_col="names",
    #     rank_by="logfoldchanges",
    #     ascending=False,
    # )

    # check input
    if len(deg_dfs) < 2:
        raise ValueError("deg_dfs must contain at least two dataframes")

    # make default dataframe names
    if df_names is None:
        df_names = [f"df{i+1}" for i in range(len(deg_dfs))]

    # check names match dataframe count
    if len(df_names) != len(deg_dfs):
        raise ValueError("df_names must have the same length as deg_dfs")

    # check needed columns
    for df in deg_dfs:
        if gene_col not in df.columns:
            raise ValueError(f"{gene_col} was not found in one dataframe")

        if rank_by not in df.columns:
            raise ValueError(f"{rank_by} was not found in one dataframe")

    # find genes shared by all dataframes
    shared_genes = set(deg_dfs[0][gene_col])

    for df in deg_dfs[1:]:
        shared_genes = shared_genes.intersection(set(df[gene_col]))

    # create merged dataframe using shared genes
    merged_df = None

    for df, name in zip(deg_dfs, df_names):

        # keep only shared genes and ranking column
        temp_df = df[df[gene_col].isin(shared_genes)][[gene_col, rank_by]].copy()

        # rename rank column
        temp_df = temp_df.rename(columns={rank_by: f"{rank_by}_{name}"})

        # merge dataframes
        if merged_df is None:
            merged_df = temp_df
        else:
            merged_df = pd.merge(
                merged_df,
                temp_df,
                on=gene_col,
                how="inner"
            )

    # get score columns
    score_cols = [f"{rank_by}_{name}" for name in df_names]

    # calculate mean rank score
    merged_df["mean_rank_score"] = merged_df[score_cols].mean(axis=1)

    # sort and return top n
    merged_df = merged_df.sort_values(
        "mean_rank_score",
        ascending=ascending
    )

    # reset index
    merged_df = merged_df.reset_index(drop=True)

    # return top n genes
    return merged_df.head(n)


def cliffs_delta(x, y):
    """
    calculate cliff's delta between two groups
    """

    # cliffs_delta
    # api:
    # cliffs_delta(
    #     x=om6_scores,
    #     y=om9_scores,
    # )

    # remove missing values
    x = pd.Series(x).dropna().to_numpy()
    y = pd.Series(y).dropna().to_numpy()

    nx = len(x)
    ny = len(y)

    # return nan if either group is empty
    if nx == 0 or ny == 0:
        return float("nan")

    # rank all values together
    ranks = rankdata(list(x) + list(y))

    # get ranks for first group
    x_ranks = ranks[:nx]

    # mann-whitney u for first group
    u = x_ranks.sum() - nx * (nx + 1) / 2

    # convert u to cliff's delta
    delta = (2 * u / (nx * ny)) - 1

    return delta


def calculate_score_cliffs_delta(
    adata,
    score_col,
    group_col="sample",
    group1="OM6",
    group2="OM9",
    dropna=True
):
    """
    calculate cliff's delta for one obs score column between two groups
    """

    # calculate_obs_score_cliffs_delta
    # api:
    # result_df, group_score_df = calculate_obs_score_cliffs_delta(
    #     adata=adata,
    #     score_col="GOBP_PYRUVATE_METABOLIC_PROCESS",
    #     group_col="sample",
    #     group1="OM6",
    #     group2="OM9",
    # )

    # check required columns
    if group_col not in adata.obs.columns:
        raise ValueError(f"{group_col} not found in adata.obs")

    if score_col not in adata.obs.columns:
        raise ValueError(f"{score_col} not found in adata.obs")

    # prepare cell-level score dataframe
    group_score_df = adata.obs[[group_col, score_col]].copy()

    # remove missing values
    if dropna:
        group_score_df = group_score_df.dropna(subset=[group_col, score_col])

    # add pathway name
    group_score_df["pathway"] = score_col

    # get group scores
    group1_scores = group_score_df.loc[
        group_score_df[group_col] == group1,
        score_col
    ]

    group2_scores = group_score_df.loc[
        group_score_df[group_col] == group2,
        score_col
    ]

    # calculate cliff's delta
    delta = cliffs_delta(
        x=group1_scores,
        y=group2_scores
    )

    # decide direction
    if delta > 0:
        higher_group = group1
    elif delta < 0:
        higher_group = group2
    else:
        higher_group = "similar"

    # store result
    result = {
        "score_col": score_col,

        f"{group1}_count": group1_scores.dropna().shape[0],
        f"{group2}_count": group2_scores.dropna().shape[0],

        f"{group1}_mean": group1_scores.mean(),
        f"{group2}_mean": group2_scores.mean(),

        f"{group1}_median": group1_scores.median(),
        f"{group2}_median": group2_scores.median(),

        f"{group1}_std": group1_scores.std(),
        f"{group2}_std": group2_scores.std(),

        f"{group1}_min": group1_scores.min(),
        f"{group2}_min": group2_scores.min(),

        f"{group1}_max": group1_scores.max(),
        f"{group2}_max": group2_scores.max(),

        "mean_difference": group1_scores.mean() - group2_scores.mean(),
        "median_difference": group1_scores.median() - group2_scores.median(),

        "cliffs_delta": delta,
        "abs_cliffs_delta": abs(delta),
        "higher_group": higher_group
    }

    # convert result to dataframe
    result_df = pd.DataFrame([result])

    return result_df, group_score_df


def save_df_table_image(
    df,
    output_path=None,
    index=False,
    round_digits=4,
    font_size=10,
    row_height=0.5,
    min_col_width=0.08,
    max_col_width=0.45
):
    """
    create a dataframe table image and optionally save it
    """

    # save_df_table_image
    # api:
    # fig = save_df_table_image(
    #     df=cliffs_df,
    #     output_path="Output/cliffs_table.png",
    # )

    # copy df so original is not changed
    df_to_show = df.copy()

    # round numeric columns
    numeric_cols = df_to_show.select_dtypes(include="number").columns
    df_to_show[numeric_cols] = df_to_show[numeric_cols].round(round_digits)

    # include index if requested
    if index:
        df_to_show = df_to_show.reset_index()

    # convert everything to string for display
    df_display = df_to_show.astype(str)

    # get table shape
    nrows, ncols = df_display.shape

    # compute column widths from max text length
    col_lengths = []
    for col in df_display.columns:
        max_cell_len = df_display[col].map(len).max()
        header_len = len(str(col))
        col_lengths.append(max(max_cell_len, header_len))

    total_len = sum(col_lengths)

    # normalized widths
    col_widths = []
    for length in col_lengths:
        width = length / total_len
        width = max(min_col_width, width)
        width = min(max_col_width, width)
        col_widths.append(width)

    # renormalize so widths sum to about 1
    width_sum = sum(col_widths)
    col_widths = [w / width_sum for w in col_widths]

    # figure size
    fig_width = max(12, sum(col_lengths) * 0.18)
    fig_height = max(2.5, (nrows + 1) * row_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )

    # base style
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    # style cells
    for (row, col), cell in table.get_celld().items():
        # set per-column widths
        cell.set_width(col_widths[col])

        # header row
        if row == 0:
            cell.set_text_props(weight="bold")

        # left align first column for long pathway names
        if col == 0 and row > 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.01

    plt.tight_layout()

    # save only if path is provided
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


# get_dense_matrix
def get_dense_matrix(X):
    """
    convert sparse matrix to dense numpy array
    """

    # get_dense_matrix
    # api:
    # get_dense_matrix(
    #     X=adata.X,
    # )

    if sparse.issparse(X):
        return X.toarray()

    return np.asarray(X)


# get_low_cv_genes
def get_low_cv_genes(
    X,
    gene_names,
    n_bins=10,
    bottom_frac=0.10
):
    """
    select low-cv genes from expression matrix
    """

    # get_low_cv_genes
    # api:
    # get_low_cv_genes(
    #     X=X,
    #     gene_names=adata.var_names,
    #     n_bins=10,
    #     bottom_frac=0.10,
    # )

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)

    cv = stds / (means + 1e-12)

    valid = np.isfinite(cv) & np.isfinite(means) & (means > 0)
    valid_idx = np.where(valid)[0]

    if len(valid_idx) == 0:
        return []

    mean_series = pd.Series(means[valid_idx], index=valid_idx)

    bins = pd.qcut(
        mean_series,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    selected_idx = []

    for bin_id in sorted(bins.dropna().unique()):
        genes_in_bin = bins.index[bins == bin_id].to_numpy()

        if len(genes_in_bin) == 0:
            continue

        n_select = max(1, int(np.ceil(len(genes_in_bin) * bottom_frac)))

        bin_cv = cv[genes_in_bin]
        low_cv_idx = genes_in_bin[np.argsort(bin_cv)[:n_select]]

        selected_idx.extend(low_cv_idx)

    selected_idx = sorted(set(selected_idx))
    selected_genes = gene_names[selected_idx].tolist()

    return selected_genes