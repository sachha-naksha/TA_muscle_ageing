# geneset_activity.py

"""
end-to-end pipeline for geneset / pathway activity scoring on AnnData objects.

steps:
    1. load adata (caller passes it in)
    2. build a geneset in decoupler format {source, target} from a custom
       gene list or one/more GMT files; report overlap with adata.var_names
    3. score every cell/metacell with decoupler.mt.aucell and copy scores
       to adata.obs for plotting / grouping
    4. plot violin-box combo per group (uses pygenelab.plotting)
    5. compute driver genes per group via spearman correlation between
       gene expression and pathway score, then heatmap the top genes
    6. cliff's delta table between two groups, optionally as a figure

the module re-uses building blocks already in pygenelab:
    - convert_gmt_to_decoupler_format, cliffs_delta, calculate_score_cliffs_delta,
      save_df_table_image  (utils)
    - compute_gene_correlation_against_score, check_genes_in_adata,
      prepare_group_score_df, subset_adata_by_obs  (data)
    - plot_violin_box_combo, plot_gene_contribution_heatmap  (plotting)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc

from .data import (
    check_genes_in_adata,
    compute_gene_correlation_against_score,
    prepare_group_score_df,
    subset_adata_by_obs,
)
from .plotting import (
    plot_violin_box_combo,
    plot_gene_contribution_heatmap,
)
from .utils import (
    calculate_score_cliffs_delta,
    convert_gmt_to_decoupler_format,
    save_df_table_image,
)


# ============================================================
# 1) adata loader (thin wrapper — caller can also use sc.read_h5ad directly)
# ============================================================
def load_adata(path, backed=None):
    """
    read an .h5ad file into AnnData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"adata not found: {path}")
    return sc.read_h5ad(path, backed=backed)


# ============================================================
# 2) geneset builders -> decoupler format DataFrame[source, target]
# ============================================================
def geneset_from_list(genes, name="custom_geneset", gene_origin=None):
    """
    wrap a python list of genes in a decoupler-style dataframe.

    gene_origin:
        None  -> use genes as-is
        "mice"   -> capitalize each gene
        "human"  -> uppercase each gene
    """
    if gene_origin == "mice":
        genes = [str(g).capitalize() for g in genes]
    elif gene_origin == "human":
        genes = [str(g).upper() for g in genes]
    elif gene_origin is not None:
        raise ValueError("gene_origin must be None, 'mice', or 'human'")

    return pd.DataFrame({
        "source": [name] * len(genes),
        "target": list(genes),
    })


def geneset_from_gmt(gmt_path, include_pathways=None, gene_origin="mice"):
    """
    parse a single GMT file into a decoupler dataframe.
    thin wrapper around pygenelab.utils.convert_gmt_to_decoupler_format
    """
    return convert_gmt_to_decoupler_format(
        pth=gmt_path,
        include_pathways=include_pathways,
        gene_origin=gene_origin,
    )


def geneset_from_gmts(gmt_paths, include_pathways=None, gene_origin="mice"):
    """
    parse multiple GMT files and stack them.
    accepts a list of paths or a dict {label: path}; labels are not used.
    """
    if isinstance(gmt_paths, dict):
        gmt_paths = list(gmt_paths.values())

    frames = [
        convert_gmt_to_decoupler_format(
            pth=p,
            include_pathways=include_pathways,
            gene_origin=gene_origin,
        )
        for p in gmt_paths
    ]
    if not frames:
        raise ValueError("gmt_paths is empty")
    return pd.concat(frames, ignore_index=True)


# ============================================================
# overlap check between geneset and adata
# ============================================================
def report_geneset_overlap(geneset_df, adata, pathway_name=None, verbose=True):
    """
    report overlap between geneset and adata.var_names.
    when pathway_name is given, only that pathway is checked; otherwise
    the whole geneset_df.
    """
    if pathway_name is not None:
        result = check_genes_in_adata(
            gene_df=geneset_df,
            adata=adata,
            gene_col="target",
            source_col="source",
            source_name=pathway_name,
        )
        header = pathway_name
    else:
        result = check_genes_in_adata(
            gene_df=geneset_df,
            adata=adata,
            gene_col="target",
        )
        header = "geneset"

    if verbose:
        print(
            f"[{header}] {result['genes_found_in_adata']}/{result['total_genes_checked']} "
            f"genes present in adata.var_names "
            f"({result['genes_missing_from_adata']} missing)"
        )
    return result


# ============================================================
# 3) AUCell scoring
# ============================================================
def score_geneset_aucell(
    adata,
    geneset_df,
    *,
    raw=False,
    tmin=None,
    verbose=True,
    copy_to_obs=True,
):
    """
    run decoupler.mt.aucell on adata using a {source, target} geneset.
    scores land in adata.obsm['score_aucell'] (a DataFrame indexed by cell).

    if copy_to_obs is True, each pathway column is also copied to adata.obs
    so downstream plotting / grouping helpers can read them directly.

    pathway names (the obs column names) are returned.
    """
    kwargs = dict(net=geneset_df, raw=raw, verbose=verbose)
    if tmin is not None:
        kwargs["tmin"] = tmin

    dc.mt.aucell(adata, **kwargs)

    score_df = adata.obsm["score_aucell"]
    pathways = list(score_df.columns)

    if copy_to_obs:
        for pw in pathways:
            adata.obs[pw] = score_df[pw].values

    return pathways


# ============================================================
# 3b) z-score-based signature scoring (alternative to AUCell)
# ============================================================
def score_signature_zscore(data, *, up_genes=None, down_genes=None, condition_col="condition"):
    """
    geneset signature score per sample / cell based on directional gene sets.

    each row gets a score = (sum of z-standardized up-gene expressions
    minus sum of z-standardized down-gene expressions) / sqrt(N), then
    z-normalized across samples.

    data: AnnData (cells x genes) or DataFrame (genes x cells).
    returns a DataFrame with one column 'geneset_score' (plus condition).
    """
    import scipy.sparse as sp

    if up_genes is None and down_genes is None:
        raise ValueError("at least one of up_genes or down_genes must be provided")

    condition_info = None
    if hasattr(data, "X"):
        if condition_col in data.obs.columns:
            condition_info = data.obs[condition_col].copy()
        X = data.X.toarray() if sp.issparse(data.X) else data.X
        data = pd.DataFrame(X.T, index=data.var_names, columns=data.obs_names)

    if "NAME" in data.columns:
        data = data.set_index("NAME")
    if "Description" in data.columns:
        data = data.drop("Description", axis=1)

    available = set(data.index)
    if up_genes is not None:
        up_genes = [g for g in set(up_genes) if g in available]
    if down_genes is not None:
        down_genes = [g for g in set(down_genes) if g in available]

    if not (up_genes or down_genes):
        raise ValueError("no genes from the gene sets were found in the expression data")

    expr = data.select_dtypes(include=[np.number])
    z = (expr - expr.mean(axis=1).values.reshape(-1, 1)) / expr.std(axis=1).values.reshape(-1, 1)

    scores = pd.Series(0.0, index=data.columns)
    total = (len(up_genes) if up_genes else 0) + (len(down_genes) if down_genes else 0)
    if up_genes:
        scores += z.loc[up_genes].sum()
    if down_genes:
        scores -= z.loc[down_genes].sum()
    scores = scores / np.sqrt(total)
    scores = (scores - scores.mean()) / scores.std()

    out = pd.DataFrame(scores, columns=["geneset_score"])
    if condition_info is not None:
        out[condition_col] = condition_info.values
    return out


# ============================================================
# 4) violin-box plot per group
# ============================================================
def plot_score_by_group(
    adata,
    score_col,
    group_col,
    *,
    palette=None,
    group_order=None,
    title=None,
    rotation=45,
    show_scatter=True,
    figsize=(5, 6),
    show_pvalue=True,
    delta_label=None,
    group_spacing=1.0,
    x_pad=0.5,
):
    """
    plot one pathway score across groups using the canonical violin-box combo.
    returns (plot_df, fig).
    """
    plot_df = prepare_group_score_df(
        adata=adata,
        group_col=group_col,
        value_col=score_col,
    )
    fig = plot_violin_box_combo(
        data=plot_df,
        x_var=group_col,
        y_var=score_col,
        title=title if title is not None else f"{score_col} by {group_col}",
        x_ticks=group_order,
        palette=palette,
        rotation=rotation,
        show_scatter=show_scatter,
        figsize=figsize,
        show_pvalue=show_pvalue,
        delta_label=delta_label,
        group_spacing=group_spacing,
        x_pad=x_pad,
    )
    return plot_df, fig


# ============================================================
# 5) driver genes per group + heatmap
# ============================================================
def compute_driver_genes_per_group(
    adata,
    geneset,
    score_col,
    group_col,
    groups,
):
    """
    for each group in `groups`, subset adata to that group and compute spearman
    correlation between every pathway gene and the pathway score.

    geneset:
        list of gene symbols OR DataFrame in decoupler format (uses 'target' col)
    returns:
        dict {group_label: ranked_gene_df}
    """
    if isinstance(geneset, pd.DataFrame):
        if "target" not in geneset.columns:
            raise ValueError("geneset DataFrame must contain a 'target' column")
        gene_list = geneset["target"].dropna().astype(str).unique().tolist()
    else:
        gene_list = list(geneset)

    ranked = {}
    for grp in groups:
        sub = subset_adata_by_obs(adata, {group_col: grp})
        ranked[grp] = compute_gene_correlation_against_score(
            adata=sub,
            geneset=gene_list,
            score=score_col,
        )
    return ranked


def plot_driver_heatmap(
    ranked_dfs,
    *,
    top_n=20,
    title="Driver Gene Contribution",
    cmap="coolwarm",
    sort_by="abs_max",
    figsize=(6, 8),
    vmin=None,
    vmax=None,
):
    """
    heatmap of top driver genes across groups.
    wraps pygenelab.plotting.plot_gene_contribution_heatmap.
    """
    return plot_gene_contribution_heatmap(
        ranked_dfs=ranked_dfs,
        top_n=top_n,
        gene_col="gene",
        corr_col="spearman_corr",
        sort_by=sort_by,
        figsize=figsize,
        cmap=cmap,
        title=title,
        annot=True,
        fmt=".2f",
        vmin=vmin,
        vmax=vmax,
    )


# ============================================================
# 6) cliff's delta table between two groups
# ============================================================
def cliffs_delta_table(
    adata,
    score_cols,
    group_col,
    group1,
    group2,
    *,
    sort_by="abs_cliffs_delta",
    ascending=False,
    output_csv=None,
    output_image=None,
):
    """
    compute cliff's delta for each score column between group1 and group2.
    returns the stacked result_df. optionally writes csv and/or a table image.
    """
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    rows = []
    for score_col in score_cols:
        result_df, _ = calculate_score_cliffs_delta(
            adata=adata,
            score_col=score_col,
            group_col=group_col,
            group1=group1,
            group2=group2,
        )
        rows.append(result_df)

    cliffs_df = pd.concat(rows, ignore_index=True)

    if sort_by in cliffs_df.columns:
        cliffs_df = cliffs_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        cliffs_df.to_csv(output_csv, index=False)

    if output_image is not None:
        Path(output_image).parent.mkdir(parents=True, exist_ok=True)
        save_df_table_image(cliffs_df, output_path=output_image)

    return cliffs_df


# ============================================================
# convenience: full pipeline driver
# ============================================================
def run_pipeline(
    adata,
    geneset_df,
    *,
    group_col,
    groups,
    palette=None,
    pathway_name=None,
    aucell_tmin=None,
    aucell_raw=False,
    output_dir=None,
    top_n_drivers=20,
    rotation=45,
    show_scatter=True,
):
    """
    run the 6-step pipeline end-to-end on one adata + one geneset_df.

    `groups` is a 2-tuple (group1, group2) used for the violin order and
    for cliff's delta. for the driver-gene step, both groups are correlated
    independently.

    returns a dict with all intermediate outputs:
        overlap, pathways, plot_df, violin_fig, ranked_dfs,
        heatmap_df, heatmap_fig, cliffs_df
    """
    if len(groups) != 2:
        raise ValueError("groups must be a 2-tuple (group1, group2)")
    group1, group2 = groups

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 2. overlap
    overlap = report_geneset_overlap(
        geneset_df=geneset_df,
        adata=adata,
        pathway_name=pathway_name,
    )

    # 3. AUCell scoring
    pathways = score_geneset_aucell(
        adata=adata,
        geneset_df=geneset_df,
        raw=aucell_raw,
        tmin=aucell_tmin,
    )

    score_col = pathway_name if pathway_name is not None else pathways[0]
    if score_col not in pathways:
        raise ValueError(
            f"pathway_name '{score_col}' not in computed scores: {pathways}"
        )

    # 4. violin
    plot_df, violin_fig = plot_score_by_group(
        adata=adata,
        score_col=score_col,
        group_col=group_col,
        palette=palette,
        group_order=list(groups),
        title=f"{score_col}",
        rotation=rotation,
        show_scatter=show_scatter,
    )
    if out_dir is not None:
        violin_fig.savefig(out_dir / f"{score_col}_violin.png", dpi=300, bbox_inches="tight")
        violin_fig.savefig(out_dir / f"{score_col}_violin.svg", bbox_inches="tight")

    # 5. driver genes + heatmap
    ranked_dfs = compute_driver_genes_per_group(
        adata=adata,
        geneset=geneset_df,
        score_col=score_col,
        group_col=group_col,
        groups=list(groups),
    )
    heatmap_df, heatmap_fig, _ = plot_driver_heatmap(
        ranked_dfs=ranked_dfs,
        top_n=top_n_drivers,
        title=f"Driver Genes — {score_col}",
    )
    if out_dir is not None:
        heatmap_fig.savefig(out_dir / f"{score_col}_drivers_heatmap.png", dpi=300, bbox_inches="tight")
        heatmap_fig.savefig(out_dir / f"{score_col}_drivers_heatmap.svg", bbox_inches="tight")

    # 6. cliff's delta
    cliffs_df = cliffs_delta_table(
        adata=adata,
        score_cols=score_col,
        group_col=group_col,
        group1=group1,
        group2=group2,
        output_csv=(out_dir / f"{score_col}_cliffs_delta.csv") if out_dir is not None else None,
        output_image=(out_dir / f"{score_col}_cliffs_delta.png") if out_dir is not None else None,
    )

    return {
        "overlap": overlap,
        "pathways": pathways,
        "score_col": score_col,
        "plot_df": plot_df,
        "violin_fig": violin_fig,
        "ranked_dfs": ranked_dfs,
        "heatmap_df": heatmap_df,
        "heatmap_fig": heatmap_fig,
        "cliffs_df": cliffs_df,
    }
