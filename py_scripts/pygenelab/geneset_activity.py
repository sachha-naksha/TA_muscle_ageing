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
import matplotlib.pyplot as plt

from .data import (
    check_genes_in_adata,
    compute_gene_correlation_against_score,
    prepare_group_score_df,
    subset_adata_by_obs,
)
from .plotting import (
    plot_violin_box_combo,
    plot_gene_contribution_heatmap,
    plot_expression_shift_heatmap,
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


def geneset_from_csv(csv_path, *, source_col="source", target_col="target",
                     include_pathways=None):
    """
    load a geneset that is already in decoupler format from a CSV with
    `source` (pathway) and `target` (gene) columns — e.g. the
    "metabolism_enriched_pathways_<sex>.csv" files written by the Step 9 cell
    of DEG_Functional_Enrichment.ipynb.

    no gene-case conversion is applied (the CSV is assumed to already match the
    adata's gene naming). `include_pathways` optionally restricts to a subset
    of `source` values. accepts a single path or a list of paths (stacked).
    """
    paths = [csv_path] if isinstance(csv_path, (str, Path)) else list(csv_path)
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"geneset CSV not found: {p}")
        df = pd.read_csv(p)
        missing = {source_col, target_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"{p} is missing column(s) {sorted(missing)}; "
                f"found {list(df.columns)}"
            )
        df = df[[source_col, target_col]].rename(
            columns={source_col: "source", target_col: "target"}
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    if include_pathways is not None:
        out = out[out["source"].isin(include_pathways)].reset_index(drop=True)
    return out


# default MSigDB collection prefixes stripped from a pathway label
_METAB_LABEL_DBS = ("HALLMARK", "WP", "REACTOME", "KEGG", "GOBP", "BIOCARTA", "PID")


def clean_metab_label(term, dbs=_METAB_LABEL_DBS):
    """
    drop the DB prefix (HALLMARK_/WP_/REACTOME_/...) and the 'metabolism'
    boilerplate from an MSigDB pathway term, then title-case for readability.
    Mirrors the `_clean_metab_label` helper used by Step 9 of
    DEG_Functional_Enrichment.ipynb so the AUCell scores and the heatmap share
    the same pathway labels.

    Examples:
      HALLMARK_FATTY_ACID_METABOLISM    -> 'Fatty Acid'
      WP_PURINE_METABOLISM              -> 'Purine'
      REACTOME_METABOLISM_OF_CARBOHYDRATES -> 'Carbohydrates'
    """
    s = str(term).upper()
    for db in dbs:
        if s.startswith(f"{db}_"):
            s = s[len(db) + 1:]
            break
    if s.startswith("METABOLISM_OF_"):
        s = s[len("METABOLISM_OF_"):]
    if s.endswith("_METABOLISM"):
        s = s[: -len("_METABOLISM")]
    elif s == "METABOLISM":
        s = "metabolism"
    return s.replace("_", " ").title()


def club_geneset_by_clean_label(geneset_df, dbs=_METAB_LABEL_DBS):
    """
    strip the DB prefix + 'metabolism' boilerplate from each `source` label
    (via `clean_metab_label`) and club the genes together per cleaned pathway:
    sources that simplify to the same label have their `target` genes unioned.

    Returns a decoupler-format (source, target) frame whose `source` values are
    the cleaned, heatmap-style pathway names. Use this on the geneset loaded
    from `metabolism_enriched_pathways_<sex>.csv` so the scored obs columns read
    'Fatty Acid', 'Purine', ... instead of 'HALLMARK_FATTY_ACID_METABOLISM'.
    """
    out = geneset_df.copy()
    out["source"] = out["source"].map(lambda t: clean_metab_label(t, dbs=dbs))
    out = out.drop_duplicates(["source", "target"]).reset_index(drop=True)
    return out


def metab_directions_from_csv(csv_path, *, source_col="source",
                              direction_col="direction", dbs=_METAB_LABEL_DBS):
    """
    read the heatmap-direction (Up / Down) per pathway from an enriched-metabolism
    CSV and key it by the *cleaned* pathway label, so the geneset-scoring notebook
    can keep only the pathways that were in the Step 9 heatmap's Up (or Down) panel.

    the CSV must carry a `direction` column — written by Step 9 of
    DEG_Functional_Enrichment.ipynb (the direction of the most-significant
    enrichment for that pathway in that sex). Raises a clear error if it is
    absent (re-run Step 9 to regenerate the CSVs).

    returns a dict {cleaned_label: 'Up'|'Down'}.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"enriched-pathway CSV not found: {p}")
    df = pd.read_csv(p)
    if direction_col not in df.columns:
        raise ValueError(
            f"{p} has no '{direction_col}' column (found {list(df.columns)}).\n"
            "Re-run Step 9 of DEG_Functional_Enrichment.ipynb to regenerate the "
            "enriched-metabolism CSVs with per-pathway direction."
        )
    out = {}
    for src, direction in df[[source_col, direction_col]].drop_duplicates().itertuples(index=False):
        out[clean_metab_label(src, dbs=dbs)] = str(direction)
    return out


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
    scatter_size=4,
    scatter_alpha=0.6,
    figsize=(5, 6),
    show_pvalue=True,
    delta_label=None,
    group_spacing=1.0,
    x_pad=0.5,
    ylim=None,
):
    """
    plot one pathway score across groups using the canonical violin-box combo.
    `scatter_size` / `scatter_alpha` control the per-cell dots when
    `show_scatter=True`. returns (plot_df, fig).
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
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
        figsize=figsize,
        show_pvalue=show_pvalue,
        delta_label=delta_label,
        group_spacing=group_spacing,
        x_pad=x_pad,
        ylim=ylim,
    )
    return plot_df, fig


def _render_celltype_score_panel(
    ax, adata, score_col, group_col, group1, group2, celltype_col, celltype,
    *, palette, title, rotation, show_scatter, show_pvalue, group_spacing,
    x_pad, annotate_delta,
):
    """
    render one (cell type) violin-box panel for `score_col` onto `ax`:
    the group1-vs-group2 contrast within that cell type, annotated with its
    own Cliff's delta. shared by `plot_score_by_celltype_panels` (one pathway,
    cell types side by side) and `plot_score_grid_by_celltype` (a grid of
    pathways x cell types). returns
        {"plot_df": DataFrame, "cliffs_row": Series|None, "ax": Axes}.
    """
    sub = subset_adata_by_obs(adata, {celltype_col: celltype})

    delta_label = None
    cliffs_row = None
    if sub.n_obs > 0:
        present = set(sub.obs[group_col].astype(str).unique())
        if annotate_delta and {str(group1), str(group2)} <= present:
            cliffs_df, _ = calculate_score_cliffs_delta(
                adata=sub,
                score_col=score_col,
                group_col=group_col,
                group1=group1,
                group2=group2,
            )
            cliffs_row = cliffs_df.iloc[0]
            delta_label = (
                f"Cliff's δ = {float(cliffs_row['abs_cliffs_delta']):.3f} "
                f"(higher in {cliffs_row['higher_group']})"
            )

    plot_df = prepare_group_score_df(
        adata=sub,
        group_col=group_col,
        value_col=score_col,
    )
    plot_violin_box_combo(
        data=plot_df,
        x_var=group_col,
        y_var=score_col,
        title=title,
        x_ticks=[group1, group2],
        palette=palette,
        rotation=rotation,
        show_scatter=show_scatter,
        show_pvalue=show_pvalue,
        delta_label=delta_label,
        group_spacing=group_spacing,
        x_pad=x_pad,
        ax=ax,
    )
    return {"plot_df": plot_df, "cliffs_row": cliffs_row, "ax": ax}


def plot_score_by_celltype_panels(
    adata,
    score_col,
    group_col,
    group1,
    group2,
    celltype_col,
    celltypes,
    *,
    palette=None,
    title=None,
    panel_size=(5, 6),
    rotation=45,
    show_scatter=False,
    show_pvalue=False,
    group_spacing=0.8,
    x_pad=0.5,
    share_y=True,
    annotate_delta=True,
):
    """
    plot one pathway score across two groups, with one violin-box panel per
    cell type laid out side by side in a single figure.

    intended for comparing the same group contrast (e.g. WT vs KO) across
    multiple cell types (e.g. Fast IIX then Fast IIB) on a shared y-axis so
    the panels are directly comparable. `adata` must already carry `score_col`
    in `.obs` (i.e. AUCell scoring was run on it).

    each panel is annotated with its own Cliff's delta between group1 and
    group2 (magnitude + which group is higher), computed within that cell type.

    parameters
    ----------
    celltypes : list[str]
        cell-type labels (values of `celltype_col`); rendered left -> right.
    share_y : bool, default True
        give every panel the same y-limits (the union across panels) so the
        score magnitudes line up across cell types.

    returns
    -------
    (fig, results) where results is a dict
        {celltype: {"plot_df": DataFrame, "cliffs_row": Series|None, "ax": Axes}}.
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n = len(celltypes)
    if n == 0:
        raise ValueError("celltypes is empty")

    fig, axes = plt.subplots(
        1, n,
        figsize=(panel_size[0] * n, panel_size[1]),
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.85, wspace=0.3)

    results = {}

    for ax, ct in zip(axes, celltypes):
        results[ct] = _render_celltype_score_panel(
            ax, adata, score_col, group_col, group1, group2, celltype_col, ct,
            palette=palette, title=str(ct), rotation=rotation,
            show_scatter=show_scatter, show_pvalue=show_pvalue,
            group_spacing=group_spacing, x_pad=x_pad, annotate_delta=annotate_delta,
        )

    # only the leftmost panel keeps a y-axis label / ticks for a clean strip
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel(score_col, fontweight="bold")

    if share_y and n > 1:
        lo = min(ax.get_ylim()[0] for ax in axes)
        hi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(lo, hi)

    if title:
        fig.suptitle(title, fontweight="bold")   # size inherits figure.titlesize

    # remove from pyplot's registry so the inline backend doesn't auto-render
    # this figure on top of the caller displaying the returned object (which
    # would show two identical copies in a notebook). The Figure object stays
    # valid for .savefig() and for display via its repr.
    plt.close(fig)
    return fig, results


def plot_score_grid_by_celltype(
    adata,
    score_cols,
    group_col,
    group1,
    group2,
    celltype_col,
    celltypes,
    *,
    palette=None,
    suptitle=None,
    panel_size=(4, 5),
    rotation=45,
    show_scatter=False,
    show_pvalue=False,
    group_spacing=0.8,
    x_pad=0.5,
    share_y_per_pathway=True,
    annotate_delta=True,
):
    """
    grid of violin-box panels: one ROW per pathway in `score_cols`, one COLUMN
    per cell type. each cell shows the group1-vs-group2 contrast for that
    (pathway, cell type), annotated with its own Cliff's delta + significance
    asterisks. cell-type names label the top row; pathway names label the
    left-most column.

    intended for the metabolism section: pass the list of cleaned metabolism
    pathways enriched in one direction (e.g. all "Up" pathways from the Step 9
    heatmap for one sex) so every pathway's Fast IIX | Fast IIB WT-vs-KO
    comparison appears as a row in a single figure. `adata` must already carry
    every `score_cols` entry in `.obs` (AUCell scoring was run on it).

    parameters
    ----------
    score_cols : list[str]
        obs columns to plot, one per row (order preserved top -> bottom).
    celltypes : list[str]
        cell-type labels (values of `celltype_col`); rendered left -> right.
    share_y_per_pathway : bool, default True
        within each pathway row, give the cell-type panels a common y-range
        (the union across that row) so Fast IIX / Fast IIB are comparable.
        y is NOT shared across pathways (AUCell magnitudes differ per geneset).

    returns
    -------
    (fig, results) where results is a nested dict
        {pathway: {celltype: {"plot_df": df, "cliffs_row": Series|None, "ax": Axes}}}.
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    score_cols = list(score_cols)
    missing = [c for c in score_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"score_cols not in adata.obs: {missing}")
    if not score_cols:
        raise ValueError("score_cols is empty")
    n_ct = len(celltypes)
    if n_ct == 0:
        raise ValueError("celltypes is empty")
    n_path = len(score_cols)

    fig, axes = plt.subplots(
        n_path, n_ct,
        figsize=(panel_size[0] * n_ct, panel_size[1] * n_path),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.08, top=0.92,
                        wspace=0.3, hspace=0.5)

    results = {}
    for r, score_col in enumerate(score_cols):
        row_results = {}
        for c, ct in enumerate(celltypes):
            ax = axes[r][c]
            # cell-type name heads only the top row; pathway name labels col 0
            row_results[ct] = _render_celltype_score_panel(
                ax, adata, score_col, group_col, group1, group2,
                celltype_col, ct,
                palette=palette,
                title=str(ct) if r == 0 else None,
                rotation=rotation,
                show_scatter=show_scatter,
                show_pvalue=show_pvalue,
                group_spacing=group_spacing,
                x_pad=x_pad,
                annotate_delta=annotate_delta,
            )
            ax.set_ylabel("")
        # left-most panel of the row carries the pathway label
        axes[r][0].set_ylabel(str(score_col), fontweight="bold")

        if share_y_per_pathway and n_ct > 1:
            row_axes = [axes[r][c] for c in range(n_ct)]
            lo = min(a.get_ylim()[0] for a in row_axes)
            hi = max(a.get_ylim()[1] for a in row_axes)
            for a in row_axes:
                a.set_ylim(lo, hi)

        results[score_col] = row_results

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")   # size inherits figure.titlesize

    # remove from pyplot's registry so the inline backend doesn't auto-render
    # this figure on top of the caller displaying the returned object (which
    # would show two identical copies in a notebook). The Figure object stays
    # valid for .savefig() and for display via its repr.
    plt.close(fig)
    return fig, results


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


def compute_group_expression_shift(
    adata,
    genes,
    group_col,
    groups,
    *,
    layer=None,
    standardize=True,
):
    """per-gene mean expression per group -- the "expression shift" view.

    with `standardize=True` (default) each gene is z-scored across ALL cells
    first (scanpy matrixplot style), so the returned value is the mean z-scored
    expression in each group: positive = above this gene's overall average in
    that group, negative = below. This shows how driver genes actually move
    between groups (e.g. up in old), as opposed to their within-group
    correlation with a score.

    genes: list of gene symbols (those absent from adata.var_names are dropped).
    returns a DataFrame indexed by gene (input order preserved), columns=groups.
    """
    present = [g for g in genes if g in adata.var_names]
    sub = adata[:, present]
    M = sub.layers[layer] if (layer and layer in adata.layers) else sub.X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    M = pd.DataFrame(M, columns=present, index=adata.obs_names)
    if standardize:
        M = (M - M.mean(axis=0)) / M.std(axis=0).replace(0, 1.0)

    grp = adata.obs[group_col].astype(str).values
    cols = [str(g) for g in groups]
    out = pd.DataFrame(
        {c: M.values[grp == c].mean(axis=0) for c in cols},
        index=present,
    )
    return out


def _rank_driver_genes(ranked_dfs, top_n, sort_by="abs_max",
                       gene_col="gene", corr_col="spearman_corr"):
    """select the top_n driver genes (by their score-correlation strength) from
    a `compute_driver_genes_per_group` result, returned in ranked order. Used to
    keep driver *selection/ordering* by correlation even when the heatmap
    *displays* expression shift."""
    cats = [df[[gene_col, corr_col]].rename(columns={corr_col: cat})
            for cat, df in ranked_dfs.items()]
    merged = cats[0]
    for d in cats[1:]:
        merged = pd.merge(merged, d, on=gene_col, how="outer")
    catcols = list(ranked_dfs.keys())
    merged[catcols] = merged[catcols].fillna(0)
    if sort_by == "max":
        s = merged[catcols].max(axis=1)
    elif sort_by == "mean":
        s = merged[catcols].mean(axis=1)
    elif sort_by == "abs_mean":
        s = merged[catcols].abs().mean(axis=1)
    else:  # "abs_max" (default)
        s = merged[catcols].abs().max(axis=1)
    merged = merged.assign(_s=s).sort_values("_s", ascending=False).head(top_n)
    return merged[gene_col].tolist()


def plot_driver_heatmap(
    ranked_dfs,
    *,
    top_n=20,
    title="Driver Gene Contribution",
    cmap=None,
    sort_by="abs_max",
    figsize=(6, 8),
    vmin=None,
    vmax=None,
    adata=None,
    group_col=None,
    groups=None,
    layer=None,
    standardize=True,
    value=None,
    annot=True,
):
    """heatmap of top driver genes across groups. Two display modes:

    * ``value="correlation"`` (legacy): each gene's Spearman correlation with the
      score, per group -- "which genes drive the score variance within each
      group". Wraps ``plot_gene_contribution_heatmap``.
    * ``value="expression"``: each gene's mean z-scored EXPRESSION per group --
      "how the driver genes actually shift between groups" (red = up in that
      group). Requires ``adata`` + ``group_col``; genes are still *selected and
      ordered* by their score-correlation in ``ranked_dfs`` (driver strength).

    Default (``value=None``): expression when ``adata`` is supplied, else
    correlation -- so existing correlation calls are unchanged. returns
    ``(df, fig, ax)``.
    """
    mode = value or ("expression" if adata is not None else "correlation")

    if mode == "correlation":
        return plot_gene_contribution_heatmap(
            ranked_dfs=ranked_dfs,
            top_n=top_n,
            gene_col="gene",
            corr_col="spearman_corr",
            sort_by=sort_by,
            figsize=figsize,
            cmap=cmap or "coolwarm",
            title=title,
            annot=annot,
            fmt=".2f",
            vmin=vmin,
            vmax=vmax,
        )

    if adata is None or group_col is None:
        raise ValueError("value='expression' needs adata and group_col")
    if groups is None:
        groups = list(ranked_dfs.keys())

    genes = _rank_driver_genes(ranked_dfs, top_n=top_n, sort_by=sort_by)
    shift = compute_group_expression_shift(
        adata, genes, group_col, groups, layer=layer, standardize=standardize)
    shift = shift.reindex([g for g in genes if g in shift.index])
    return plot_expression_shift_heatmap(
        shift, cmap=cmap or "RdBu_r", annot=annot,
        vmin=vmin, vmax=vmax, figsize=figsize, title=title)


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
