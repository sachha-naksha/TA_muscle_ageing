# data.py

"""
functions relating data analysis
"""

# imports
import pandas as pd
from scipy.stats import spearmanr


# compute_gene_correlation_against_score
def compute_gene_correlation_against_score(adata, geneset, score="score_series"):
    """
    compute spearman correlation for genes in the geneset
    against a specified score stored in adata.obs

    returns ranked_gene_df only.
    """

    # keep only genes that are actually present in adata
    geneset_present = [gene for gene in geneset if gene in adata.var_names]

    if len(geneset_present) == 0:
        raise ValueError("none of the genes in geneset were found in adata.var_names")

    # expression matrix only for genes in the pathway
    X = adata[:, geneset_present].to_df()

    # get the score series
    score_series = adata.obs[score].copy()

    # if score_aucell is a dataframe with one column, convert to series
    if isinstance(score_series, pd.DataFrame):
        score_series = score_series.iloc[:, 0]

    # compute correlation only for pathway genes
    correlations = [
        spearmanr(X[gene], score_series)[0]
        for gene in X.columns
    ]

    # create ranked dataframe
    ranked_gene_df = pd.DataFrame({
        "gene": X.columns,
        "spearman_corr": correlations
    })

    # rank genes by strongest positive contribution
    ranked_gene_df = (
        ranked_gene_df
        .sort_values("spearman_corr", ascending=False)
        .reset_index(drop=True)
    )

    # start the ranking from 1 (index is 0)
    ranked_gene_df["rank"] = ranked_gene_df.index + 1

    # return ranked gene dataframe
    return ranked_gene_df


# preview_cell_gene_matrix
def preview_cell_gene_matrix(adata, layer=None, genes=None, cells=None, n_rows=5, n_cols=5):
    """
    preview cell_gene_matrix as a dataframe from anndata object
    """

    # check layer exists
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{layer} was not found in adata.layers")

    # subset cells first
    if cells is not None:
        if len(cells) == 0:
            raise ValueError("cells list is empty")

        if isinstance(cells[0], str):
            adata_subset = adata[cells, :].copy()
        else:
            adata_subset = adata[cells, :].copy()
    else:
        adata_subset = adata.copy()

    # subset genes first
    if genes is not None:
        genes_present = [gene for gene in genes if gene in adata_subset.var_names]

        if len(genes_present) == 0:
            raise ValueError("none of the genes were found in adata.var_names")

        adata_subset = adata_subset[:, genes_present].copy()

    # choose layer or main X matrix
    if layer is None:
        matrix = adata_subset.X
    else:
        matrix = adata_subset.layers[layer]

    # convert sparse matrix to dense
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # make dataframe
    df = pd.DataFrame(
        matrix,
        index=adata_subset.obs_names,
        columns=adata_subset.var_names
    )

    # show small preview if no specific cells/genes given
    if cells is None and genes is None:
        return df.iloc[:n_rows, :n_cols]

    # return cell x gene matrix
    return df


# check_genes_in_adata
def check_genes_in_adata(
    gene_df,
    adata,
    gene_col="target",
    source_col=None,
    source_name=None
):
    """
    check how many genes from the geneset are present in adata
    """

    df = gene_df.copy()

    # filter to one source if given
    if source_col is not None and source_name is not None:
        df = df[df[source_col] == source_name].copy()

    # check gene column exists
    if gene_col not in df.columns:
        raise ValueError(f"{gene_col} was not found in gene_df columns")

    # get unique genes from dataframe
    genes = (
        df[gene_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # get genes from adata
    adata_genes = set(adata.var_names.astype(str))

    # check which genes are present
    genes_in_adata = [gene for gene in genes if gene in adata_genes]
    genes_not_in_adata = [gene for gene in genes if gene not in adata_genes]

    # store summary result
    result = {
        "total_genes_checked": len(genes),
        "genes_found_in_adata": len(genes_in_adata),
        "genes_missing_from_adata": len(genes_not_in_adata),
        "matched_genes": genes_in_adata,
        "missing_genes": genes_not_in_adata,
    }

    # return gene matching result
    return result


# subset_adata_by_obs
def subset_adata_by_obs(adata, filters):
    """
    subset adata using values from adata.obs
    """

    # subset_adata_by_obs
    # api:
    # subset_adata_by_obs(
    #     adata,
    #     filters={"sex": "F", "condition": "KO", "C_scANVI": ["Fast IIX", "Fast IIB"]}
    # )

    # start with all cells
    mask = pd.Series(True, index=adata.obs_names)

    # apply each filter
    for col, value in filters.items():

        # check obs column exists
        if col not in adata.obs.columns:
            raise ValueError(f"{col} was not found in adata.obs columns")

        # filter multiple values
        if isinstance(value, list):
            mask = mask & adata.obs[col].isin(value)

        # filter one value
        else:
            mask = mask & (adata.obs[col] == value)

    # return filtered adata
    return adata[mask].copy()


# get_pathway_genes
def get_pathway_genes(
    geneset_df,
    pathway_name,
    source_col="source",
    target_col="target",
    dropna=True
):
    """
    get genes from one pathway as a list
    """

    # get_pathway_genes
    # api:
    # get_pathway_genes(
    #     geneset_df,
    #     pathway_name="SAUL_SEN_MAYO_UP_IN_SEN",
    #     source_col="source",
    #     target_col="target",
    #     dropna=True
    # )

    # check source column exists
    if source_col not in geneset_df.columns:
        raise ValueError(f"{source_col} was not found in geneset_df columns")

    # check target column exists
    if target_col not in geneset_df.columns:
        raise ValueError(f"{target_col} was not found in geneset_df columns")

    # subset to one pathway
    pathway_df = geneset_df[geneset_df[source_col] == pathway_name].copy()

    # check pathway exists
    if pathway_df.shape[0] == 0:
        raise ValueError(f"{pathway_name} was not found in {source_col}")

    # get pathway genes
    genes = pathway_df[target_col]

    # remove missing genes if needed
    if dropna:
        genes = genes.dropna()

    # convert genes to clean list
    genes = (
        genes
        .astype(str)
        .unique()
        .tolist()
    )

    # return pathway genes
    return genes


# get_cell_type_percentages
def get_cell_type_percentages(
    adata,
    *,
    cell_type_col="cell_type",
    condition_col="condition",
    conditions=("WT", "KO"),
    column_labels=None,
):
    """
    percentage of each cell type within each condition.
    returns a wide DataFrame: rows = cell types, columns = one per condition.
    """
    if column_labels is None:
        column_labels = {c: f"{c}%" for c in conditions}
    elif isinstance(column_labels, (list, tuple)):
        column_labels = dict(zip(conditions, column_labels))

    out = None
    for cond in conditions:
        pct = (
            adata[adata.obs[condition_col] == cond]
            .obs[cell_type_col]
            .value_counts(normalize=True) * 100
        ).rename(column_labels[cond])
        out = pct.to_frame() if out is None else out.join(pct, how="outer")

    return out.fillna(0).round(2).reset_index().rename(columns={"index": "Cell Type"})


# get_cell_type_percentages_by_sex
def get_cell_type_percentages_by_sex(
    adata,
    *,
    cell_type_col="cell_type",
    condition_col="condition",
    sex_col="sex",
    conditions=("WT", "KO"),
    sexes=("F", "M"),
):
    """
    percentage of each cell type within each (condition, sex) combination.
    returns a wide DataFrame with one column per (cond, sex) pair.
    """
    out = None
    for cond in conditions:
        for sx in sexes:
            mask = (adata.obs[condition_col] == cond) & (adata.obs[sex_col] == sx)
            pct = (
                adata[mask]
                .obs[cell_type_col]
                .value_counts(normalize=True) * 100
            ).rename(f"{cond} {sx}%")
            out = pct.to_frame() if out is None else out.join(pct, how="outer")

    return out.fillna(0).round(2).reset_index().rename(columns={"index": "Cell Type"})


# prepare_group_score_df
def prepare_group_score_df(
    adata,
    group_col,
    value_col,
    dropna=True
):
    """
    create a clean dataframe for plotting one score by one group
    """

    # prepare_plot_df
    # api:
    # prepare_plot_df(
    #     adata,
    #     group_col="age",
    #     value_col="SAUL_SEN_MAYO_UP_IN_SEN",
    #     dropna=True
    # )

    # check group column exists
    if group_col not in adata.obs.columns:
        raise ValueError(f"{group_col} was not found in adata.obs columns")

    # check value column exists
    if value_col not in adata.obs.columns:
        raise ValueError(f"{value_col} was not found in adata.obs columns")

    # create plotting dataframe
    plot_df = adata.obs[[group_col, value_col]].copy()

    # remove missing values if needed
    if dropna:
        plot_df = plot_df.dropna()

    # reset index
    plot_df = plot_df.reset_index(drop=True)

    # return plotting dataframe
    return plot_df


# summarize_score_by_group
def summarize_score_by_group(
    data,
    group_col,
    score_col,
    stats_to_include=None,
    dropna=True
):
    """
    summarize score values by group
    """

    # summarize_score_by_group
    # api:
    # summarize_score_by_group(
    #     data=group_score_df,
    #     group_col="age",
    #     score_col="SAUL_SEN_MAYO_UP_IN_SEN",
    #     stats_to_include=None,
    #     dropna=True
    # )

    # check group column exists
    if group_col not in data.columns:
        raise ValueError(f"{group_col} was not found in data columns")

    # check score column exists
    if score_col not in data.columns:
        raise ValueError(f"{score_col} was not found in data columns")

    # keep needed columns
    df = data[[group_col, score_col]].copy()

    # remove missing values if needed
    if dropna:
        df = df.dropna()

    # set default summary stats
    if stats_to_include is None:
        stats_to_include = ["count", "mean", "median", "std", "min", "max"]

    # summarize score by group
    summary_df = (
        df
        .groupby(group_col)[score_col]
        .agg(stats_to_include)
        .reset_index()
    )

    # return summary dataframe
    return summary_df
