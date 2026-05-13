# utils.py 

"""
all utility functions
"""

# imports
from pathlib import Path
from itertools import chain, repeat

import pandas as pd
from scipy import stats


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
