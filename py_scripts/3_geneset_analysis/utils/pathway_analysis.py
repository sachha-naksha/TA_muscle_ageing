"""
pathway_analysis.py

End-to-end pathway activity scoring pipeline. Wraps the six steps that were
duplicated across the notebooks in this folder:

  0. load adata             (mice single cell / mice seacells / human SKM,
                             optionally split by sex and/or cell type)
  1. load + intersect       gene set (from a GMT pathway, or a custom list)
                             against the genes captured in adata
  2. score                  pathway activity per cell with decoupler AUCell
  3. violin/box plot        of the score per group  (delegates to pygenelab
                             so the plot lives in exactly one place)
  4. driver genes           Spearman correlation between each pathway gene
                             and the per-cell score, plotted as a heatmap
  5. cliff's delta          non-parametric effect size between two groups

The notebooks become thin: pick a dataset, pick a pathway name, run
``run_pathway_analysis(...)``.
"""

# imports
import re
from pathlib import Path

import decoupler as dc
import pandas as pd
import scanpy as sc

import pygenelab as pgl


# =========================================================================
# Dataset registry
# =========================================================================

# Canonical paths used across the geneset analysis notebooks. Add new
# datasets here so every notebook picks them up by name.
DATASETS = {
    "mice_single_cells": {
        "path": (
            "/ocean/projects/cis240075p/asachan/datasets/TA_muscle/"
            "ERCC1_KO_mice/integrated_samples/objects/"
            "adata_harmony_celloracle_compatible_v2.h5ad"
        ),
        "gene_origin": "mice",
        "default_group_col": "condition",
        "default_groups": ("WT", "KO"),
    },
    "mice_seacells_male": {
        "path": (
            "/ocean/projects/cis240075p/asachan/datasets/TA_muscle/"
            "ERCC1_KO_mice/integrated_samples/objects/"
            "seacells_M_FastIIB_FastIIX_harmonyv2.h5ad"
        ),
        "gene_origin": "mice",
        "default_group_col": "condition",
        "default_groups": ("WT", "KO"),
        # seacells store the condition as numeric `y` -> normalize to WT/KO
        "post_load": "metacell_y_to_condition",
    },
    "mice_seacells_female": {
        "path": (
            "/ocean/projects/cis240075p/asachan/datasets/TA_muscle/"
            "ERCC1_KO_mice/integrated_samples/objects/"
            "seacells_F_FastIIB_FastIIX_harmonyv2.h5ad"
        ),
        "gene_origin": "mice",
        "default_group_col": "condition",
        "default_groups": ("WT", "KO"),
        "post_load": "metacell_y_to_condition",
    },
    "human_skm_female": {
        "path": (
            "/ocean/projects/cis240075p/asachan/datasets/TA_muscle/"
            "public_datasets/SKM_multimodal_ageing/objects/tmp/"
            "myofibers_female.h5ad"
        ),
        "gene_origin": "human",
        "default_group_col": "sample",
        "default_groups": ("OM6", "OM9"),
    },
}


# Default location for MSigDB GMT bundles. `mice` is the .Mm.symbols release;
# `human` is the .Hs.symbols release. Override per-call with ``gmt_path=``.
DEFAULT_GMT = {
    "mice": (
        "/ocean/projects/cis240075p/asachan/datasets/gene_sets/mouse/"
        "msigdb.v2024.1.Mm.symbols.gmt"
    ),
    "human": (
        "/ocean/projects/cis240075p/asachan/datasets/gene_sets/human/"
        "msigdb.v2024.1.Hs.symbols.gmt"
    ),
}


# Default condition palette (matches what the notebooks have been using).
DEFAULT_PALETTE = {
    "WT": "#B6D7A8",
    "KO": "#F08080",
    "OM6": "#B6D7A8",
    "OM9": "#F08080",
}


# =========================================================================
# Step 0 -- load adata
# =========================================================================

def load_adata(
    dataset,
    sex=None,
    cell_types=None,
    extra_filters=None,
):
    """
    Load one of the registered datasets and optionally subset it.

    Parameters
    ----------
    dataset : str
        Key into ``DATASETS`` (e.g. ``"mice_single_cells"``,
        ``"mice_seacells_female"``, ``"human_skm_female"``).
    sex : str or list of str, optional
        Restrict to ``"M"`` / ``"F"`` (single cell datasets only).
    cell_types : str or list of str, optional
        Restrict to one or more ``C_scANVI`` cell types.
    extra_filters : dict, optional
        Additional ``obs`` filters passed straight to
        ``pgl.subset_adata_by_obs`` (e.g. ``{"sample": ["OM6", "OM9"]}``).

    Returns
    -------
    AnnData
        A defensive copy ready for downstream scoring.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset '{dataset}'. options: {list(DATASETS)}"
        )

    cfg = DATASETS[dataset]
    adata = sc.read_h5ad(cfg["path"])

    # normalize metacell `y` (0/1) to a `condition` column so every
    # downstream step can use a single column name
    if cfg.get("post_load") == "metacell_y_to_condition":
        adata.obs["condition"] = adata.obs["y"].map({0: "WT", 1: "KO"}).copy()

    # build obs-level filters and apply via pgl
    filters = {}

    if sex is not None and "sex" in adata.obs.columns:
        filters["sex"] = sex if isinstance(sex, list) else [sex]

    if cell_types is not None and "C_scANVI" in adata.obs.columns:
        filters["C_scANVI"] = (
            cell_types if isinstance(cell_types, list) else [cell_types]
        )

    if extra_filters:
        filters.update(extra_filters)

    if filters:
        adata = pgl.subset_adata_by_obs(adata, filters=filters)

    return adata


# =========================================================================
# Step 1 -- load pathway genes + intersect with adata
# =========================================================================

def load_pathway_genes(
    pathway_name,
    gmt_path=None,
    custom_genes=None,
    gene_origin="mice",
):
    """
    Build a decoupler-format ``source/target`` dataframe for one pathway.

    Two modes:

    * ``custom_genes`` provided -- build the table from that list (used for
      the atrophy gene set from PMID 31325479 and other ad-hoc panels).
    * Otherwise -- parse ``gmt_path`` (defaults to MSigDB for
      ``gene_origin``) and keep only the row whose source matches
      ``pathway_name``.

    ``gene_origin`` controls casing: ``"mice"`` -> ``Title`` case,
    ``"human"`` -> ``UPPER`` case.
    """

    if custom_genes is not None:
        if gene_origin == "mice":
            targets = [g.capitalize() for g in custom_genes]
        elif gene_origin == "human":
            targets = [g.upper() for g in custom_genes]
        else:
            raise ValueError("gene_origin must be 'mice' or 'human'")

        return pd.DataFrame({
            "source": pathway_name,
            "target": list(dict.fromkeys(targets)),  # de-dup, keep order
        })

    if gmt_path is None:
        gmt_path = DEFAULT_GMT[gene_origin]

    return pgl.convert_gmt_to_decoupler_format(
        gmt_path,
        include_pathways=[pathway_name],
        gene_origin=gene_origin,
    )


def check_pathway_in_adata(pathway_df, adata, verbose=True):
    """
    Return the ``pgl.check_genes_in_adata`` summary and optionally print
    the matched/missing counts.
    """

    summary = pgl.check_genes_in_adata(pathway_df, adata)

    if verbose:
        print(
            f"  pathway genes: {summary['total_genes_checked']} | "
            f"found in adata: {summary['genes_found_in_adata']} | "
            f"missing: {summary['genes_missing_from_adata']}"
        )

    return summary


# =========================================================================
# Step 2 -- score with AUCell
# =========================================================================

def score_pathway(
    adata,
    pathway_df,
    pathway_name,
    tmin=None,
    verbose=True,
):
    """
    Run decoupler AUCell for ``pathway_df`` and copy the per-cell score
    into ``adata.obs[pathway_name]``.

    Set ``tmin`` only when the pathway has very few genes (decoupler's
    default minimum is 5 -- e.g. ``REACTOME_PYRUVATE_METABOLISM`` needs
    ``tmin=4``).
    """

    aucell_kwargs = dict(raw=False, verbose=verbose)
    if tmin is not None:
        aucell_kwargs["tmin"] = tmin

    dc.mt.aucell(adata, pathway_df, **aucell_kwargs)

    if pathway_name not in adata.obsm["score_aucell"].columns:
        raise RuntimeError(
            f"AUCell did not produce a score for '{pathway_name}'. "
            "Check that this pathway has enough genes in adata "
            "(consider tmin=)."
        )

    adata.obs[pathway_name] = adata.obsm["score_aucell"][pathway_name].copy()
    return adata


# =========================================================================
# Step 3 -- violin/box plot (delegates to pygenelab)
# =========================================================================

def plot_pathway_violin_box(
    adata,
    pathway_name,
    group_col,
    group_order=None,
    palette=None,
    title=None,
    show_scatter=False,
):
    """
    Thin wrapper around ``pgl.plot_violin_box_combo`` so the plot is
    edited in exactly one place (pygenelab.plotting).
    """

    group_score_df = pgl.prepare_group_score_df(
        adata,
        group_col=group_col,
        value_col=pathway_name,
        dropna=True,
    )

    if palette is None:
        palette = DEFAULT_PALETTE

    return pgl.plot_violin_box_combo(
        group_score_df,
        x_var=group_col,
        y_var=pathway_name,
        title=title or pathway_name,
        palette=palette,
        x_ticks=group_order,
        show_scatter=show_scatter,
    )


# =========================================================================
# Step 4 -- driver genes (Spearman) + heatmap
# =========================================================================

def compute_pathway_drivers(
    adata,
    pathway_df,
    pathway_name,
    group_col,
    groups,
):
    """
    For each value in ``groups``, subset adata to cells in that group and
    rank the pathway's genes by their Spearman correlation with the
    AUCell score.

    Returns a dict ``{group_value: ranked_genes_df}``.
    """

    genes = pgl.get_pathway_genes(pathway_df, pathway_name)
    ranked = {}

    for g in groups:
        adata_g = pgl.subset_adata_by_obs(adata, filters={group_col: g})
        ranked[g] = pgl.compute_gene_correlation_against_score(
            adata_g, genes, score=pathway_name
        )

    return ranked


def plot_pathway_drivers(ranked_genes_dict, title=None, top_n=20):
    """
    Heatmap of top driver genes per group. Delegates to
    ``pgl.plot_gene_contribution_heatmap``.

    Returns ``(heatmap_df, fig, ax)``.
    """

    return pgl.plot_gene_contribution_heatmap(
        ranked_genes_dict,
        title=title or "Gene Contribution",
        top_n=top_n,
    )


# =========================================================================
# Step 5 -- cliff's delta
# =========================================================================

def compute_cliffs_delta(adata, pathway_name, group_col, group1, group2):
    """
    Cliff's delta for one pathway. Returns a one-row dataframe with the
    same columns ``pgl.calculate_score_cliffs_delta`` produces.
    """

    result_df, _ = pgl.calculate_score_cliffs_delta(
        adata,
        score_col=pathway_name,
        group_col=group_col,
        group1=group1,
        group2=group2,
    )
    return result_df


def compute_cliffs_delta_multi(adata, pathway_names, group_col, group1, group2):
    """
    Run Cliff's delta for many pathways already scored on ``adata`` and
    return one combined dataframe sorted by ``abs_cliffs_delta``.
    """

    rows = []
    for p in pathway_names:
        if p not in adata.obs.columns:
            print(f"  skipping {p}: not in adata.obs (score it first)")
            continue
        rows.append(
            compute_cliffs_delta(adata, p, group_col, group1, group2)
        )

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values("abs_cliffs_delta", ascending=False).reset_index(
        drop=True
    )


def plot_cliffs_delta_table(cliffs_df, output_path=None):
    """
    Render the Cliff's delta summary as an image-table. The displayed
    columns mirror what the notebooks were previously slicing by hand.
    """

    keep_cols = [
        c
        for c in (
            "score_col",
            "mean_difference",
            "median_difference",
            "cliffs_delta",
            "higher_group",
        )
        if c in cliffs_df.columns
    ]
    return pgl.save_df_table_image(
        cliffs_df[keep_cols],
        output_path=output_path,
    )


# =========================================================================
# Orchestrator
# =========================================================================

def _safe_filename(text):
    """Make a path-safe stem from a pathway name (handles parens, colons)."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def run_pathway_analysis(
    adata,
    pathway_name,
    *,
    pathway_df=None,
    custom_genes=None,
    gmt_path=None,
    gene_origin="mice",
    group_col="condition",
    groups=("WT", "KO"),
    palette=None,
    title_prefix="",
    output_dir=None,
    file_id=None,
    tmin=None,
    top_n_drivers=20,
    verbose=True,
):
    """
    Run all six steps for one pathway and return every artifact.

    Pass either ``pathway_df`` (decoupler-format table you already built),
    ``custom_genes`` (list of gene symbols, e.g. atrophy panel), or
    neither -- then ``pathway_name`` is looked up in
    ``gmt_path`` / the default MSigDB GMT.

    Returns
    -------
    dict with keys: ``pathway_df``, ``intersection``, ``group_score_df``,
    ``violin_box_fig``, ``ranked_genes``, ``driver_heatmap_df``,
    ``driver_heatmap_fig``, ``cliffs_delta_df``, ``cliffs_table_fig``.
    Figures are also saved to ``output_dir`` (svg + png) when provided.
    """

    group1, group2 = groups[0], groups[1]

    # ---- step 1: pathway gene table -------------------------------------
    if pathway_df is None:
        pathway_df = load_pathway_genes(
            pathway_name,
            gmt_path=gmt_path,
            custom_genes=custom_genes,
            gene_origin=gene_origin,
        )

    intersection = check_pathway_in_adata(
        pathway_df, adata, verbose=verbose
    )

    # ---- step 2: score with AUCell --------------------------------------
    score_pathway(
        adata, pathway_df, pathway_name, tmin=tmin, verbose=verbose
    )

    group_score_df = pgl.prepare_group_score_df(
        adata,
        group_col=group_col,
        value_col=pathway_name,
        dropna=True,
    )

    if verbose:
        print(
            pgl.summarize_score_by_group(
                group_score_df,
                group_col=group_col,
                score_col=pathway_name,
            )
        )

    # ---- step 3: violin/box plot ----------------------------------------
    title = (
        f"{title_prefix}: {pathway_name} Scores".lstrip(": ").strip()
        if title_prefix
        else f"{pathway_name} Scores"
    )
    violin_box_fig = plot_pathway_violin_box(
        adata,
        pathway_name,
        group_col=group_col,
        group_order=list(groups),
        palette=palette,
        title=title,
    )

    # ---- step 4: driver genes -------------------------------------------
    ranked_genes = compute_pathway_drivers(
        adata, pathway_df, pathway_name, group_col, groups
    )
    driver_heatmap_df, driver_heatmap_fig, _ = plot_pathway_drivers(
        ranked_genes,
        title=f"{title_prefix} {pathway_name}: Gene Contribution".strip(),
        top_n=top_n_drivers,
    )

    # ---- step 5: cliff's delta ------------------------------------------
    cliffs_delta_df = compute_cliffs_delta(
        adata, pathway_name, group_col, group1, group2
    )
    cliffs_table_fig = plot_cliffs_delta_table(cliffs_delta_df)

    # ---- save figures ----------------------------------------------------
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = file_id or _safe_filename(pathway_name)

        for kind, fig in (
            ("Violin_Box", violin_box_fig),
            ("Gene_Contribution_Heatmap", driver_heatmap_fig),
            ("Cliffs_Delta_Table", cliffs_table_fig),
        ):
            pgl.save_fig_as_png(fig, out / f"{stem}_{kind}.png")
            pgl.save_editable_svg(fig, out / f"{stem}_{kind}.svg")

    return {
        "pathway_df": pathway_df,
        "intersection": intersection,
        "group_score_df": group_score_df,
        "violin_box_fig": violin_box_fig,
        "ranked_genes": ranked_genes,
        "driver_heatmap_df": driver_heatmap_df,
        "driver_heatmap_fig": driver_heatmap_fig,
        "cliffs_delta_df": cliffs_delta_df,
        "cliffs_table_fig": cliffs_table_fig,
    }


def run_cliffs_delta_sweep(
    adata,
    pathways,
    *,
    gmt_path=None,
    gene_origin="mice",
    custom_pathways=None,
    group_col="condition",
    groups=("WT", "KO"),
    output_dir=None,
    file_id=None,
    verbose=True,
):
    """
    Score many pathways on the same adata and return a combined
    Cliff's delta table sorted by effect size.

    ``custom_pathways`` is an optional ``{name: [genes]}`` mapping for
    panels that are not in the GMT (e.g. the atrophy gene list).
    """

    group1, group2 = groups[0], groups[1]
    custom_pathways = custom_pathways or {}

    for p in pathways:
        if p in custom_pathways:
            df_p = load_pathway_genes(
                p,
                custom_genes=custom_pathways[p],
                gene_origin=gene_origin,
            )
        else:
            df_p = load_pathway_genes(
                p, gmt_path=gmt_path, gene_origin=gene_origin
            )

        if verbose:
            print(f"scoring {p}")
        score_pathway(adata, df_p, p, verbose=False)

    cliffs_df = compute_cliffs_delta_multi(
        adata, pathways, group_col, group1, group2
    )
    fig = plot_cliffs_delta_table(cliffs_df)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = file_id or "Cliffs_Delta"
        pgl.save_fig_as_png(fig, out / f"{stem}.png")
        pgl.save_editable_svg(fig, out / f"{stem}.svg")

    return cliffs_df, fig
