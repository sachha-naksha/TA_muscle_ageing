# deg_functional_enrichment.py

"""
DEG -> functional (pathway) enrichment pipeline.

input:  per-(sex, cell_type) DEG CSVs (any source) and a GMT pathway file
output: pre-ranked GSEA results and a side-by-side Up/Down pathway heatmap

steps:
    1. load DEG CSVs (rename first column to a known gene column)
    2. build a pre-ranked gene list per DEG file (signed log2FC * -log10 padj)
    3. load pathways from a GMT and filter by size / keyword
    4. run decoupler.mt.gsea per ranked list
    5. annotate GSEA results (Up/Down group, simplified term, dedup)
    6. find pathways enriched in >=2 cell types and heatmap them

functions are gender / cell-type agnostic — pass any pair (or set) of DEG
files keyed by cell-type label.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import decoupler as dc


# ============================================================
# 0) DEG generation from AnnData (upstream of the CSV-driven pipeline)
# ============================================================
def run_deg_analysis(
    adata,
    *,
    condition_col="condition",
    reference="WT",
    comparison="KO",
    layer="log1p_norm_cb",
    sex_label="",
    log2fc_threshold=0.5,
    pval_threshold=0.05,
):
    """
    differential expression between two conditions using scanpy
    wilcoxon rank-sum test.

    returns (filtered_adata, result_df) where result_df has columns
    [names, scores, logfoldchanges, pvals, pvals_adj, log2FC, -log10(pval), significant].
    """
    work = adata.copy()
    conditions = work.obs[condition_col].unique()
    if reference not in conditions or comparison not in conditions:
        raise ValueError(
            f"{reference} or {comparison} not found in {condition_col}; got {list(conditions)}"
        )

    mask = work.obs[condition_col].isin([reference, comparison])
    sub = work[mask].copy()

    sc.tl.rank_genes_groups(
        sub,
        groupby=condition_col,
        groups=[comparison],
        reference=reference,
        method="wilcoxon",
        pts=True,
        tie_correct=True,
        use_raw=False,
        layer=layer,
    )

    result = sc.get.rank_genes_groups_df(sub, group=comparison)
    result["log2FC"] = result["logfoldchanges"]
    result["-log10(pval)"] = -np.log10(result["pvals"].replace(0, 1e-300))
    result["significant"] = (
        (result["pvals_adj"] < pval_threshold)
        & (result["logfoldchanges"].abs() > log2fc_threshold)
    )

    if sex_label:
        print(
            f"[{sex_label}] {comparison} vs {reference}: "
            f"{result['significant'].sum()} significant / {len(result)} total"
        )

    return sub, result


def save_deg_results(result_df, output_dir, sex_label="", basename=None):
    """
    write significant DEGs to two CSVs (up- and down-regulated), sorted by
    adjusted p-value. returns the significant subset.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = result_df[result_df["significant"]].copy().sort_values("pvals_adj")
    base = basename if basename is not None else (sex_label.lower() if sex_label else "deg")

    up = sig[sig["log2FC"] > 0]
    down = sig[sig["log2FC"] < 0]

    paths = {}
    if len(up) > 0:
        p = out_dir / f"DEG_results_{base}_up.csv"
        up.to_csv(p, index=False)
        paths["up"] = p
    if len(down) > 0:
        p = out_dir / f"DEG_results_{base}_down.csv"
        down.to_csv(p, index=False)
        paths["down"] = p

    return sig, paths


# ============================================================
# DEG dataframe manipulation
# ============================================================
def filter_degs(
    degs,
    *,
    log2fc_threshold=0.05,
    pval_threshold=0.01,
    log2fc_col="logfoldchanges",
    pval_col="pvals_adj",
):
    """
    keep DEGs with |log2FC| > threshold and pval_adj < threshold,
    sorted by |log2FC| descending.
    """
    out = degs[
        (degs[log2fc_col].abs() > log2fc_threshold)
        & (degs[pval_col] < pval_threshold)
    ]
    return out.sort_values(by=log2fc_col, key=lambda x: x.abs(), ascending=False)


def get_common_degs(
    degs_a,
    degs_b,
    *,
    gene_col="names",
    suffixes=("_a", "_b"),
):
    """
    inner-merge two DEG dataframes on the gene column. suffixes default to
    _a / _b — use ('_male', '_female') etc. for explicit labels.
    """
    return pd.merge(degs_a, degs_b, on=gene_col, how="inner", suffixes=suffixes)


def get_opposite_sign_degs(common_degs, *, log2fc_cols=("logfoldchanges_a", "logfoldchanges_b")):
    """
    keep rows where the two log2FC columns have opposite signs.
    """
    a, b = log2fc_cols
    return common_degs[common_degs[a] * common_degs[b] < 0]


def get_same_sign_degs(common_degs, *, log2fc_cols=("logfoldchanges_a", "logfoldchanges_b")):
    """
    keep rows where the two log2FC columns have the same sign.
    """
    a, b = log2fc_cols
    return common_degs[common_degs[a] * common_degs[b] > 0]


# ============================================================
# geneset gene extractor (from GSEA result + decoupler-format pathways)
# ============================================================
def get_geneset_genes_table(
    pathway_df,
    geneset_df,
    *,
    term_col="Term",
    group_col="Group",
    score_col="-log10(pval)",
    source_col="source",
    target_col="target",
    output_csv=None,
):
    """
    for each pathway term in `pathway_df` (a GSEA-result dataframe), look up
    the genes from `geneset_df` (decoupler format: source, target) and write
    a `pathway,group,significance,genes` table (genes semicolon-joined).
    """
    rows = []
    for _, row in pathway_df.iterrows():
        term = row[term_col]
        genes = geneset_df.loc[
            geneset_df[source_col] == term, target_col
        ].dropna().astype(str).unique()
        rows.append({
            "Pathway": term,
            "Group": row.get(group_col, ""),
            "Significance": row.get(score_col, None),
            "Genes": ";".join(genes),
        })

    out = pd.DataFrame(rows)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return out


# ============================================================
# 1) DEG CSV loaders
# ============================================================
def load_deg_csv(path, gene_col="gene_name", first_col_as_gene=True):
    """
    read a DEG CSV. by default the first column is renamed to `gene_col`
    so callers don't have to know whether genes live in an unnamed index
    column or a named column.
    """
    df = pd.read_csv(path)
    if first_col_as_gene:
        df = df.rename(columns={df.columns[0]: gene_col})
    return df


def load_deg_csvs(paths, gene_col="gene_name"):
    """
    load multiple DEG CSVs keyed by a user-provided label
    (e.g. {"Fast IIX": "...DEGs_KO_vs_WT_Fast_IIX.csv", ...}).
    """
    return {label: load_deg_csv(p, gene_col=gene_col) for label, p in paths.items()}


# ============================================================
# 2) ranked gene list (input to decoupler.mt.gsea)
# ============================================================
def create_ranked_genelist(
    deg_df,
    log2fc_col="logfoldchanges",
    pval_col="pvals_adj",
    gene_col="gene_name",
    min_pval=1e-300,
    top_n=None,
):
    """
    create a (1 x n_genes) ranked DataFrame for decoupler GSEA based on
    signed log2FC * -log10(padj).

    if top_n is given, keep only the top_n genes by |ranking_score|, which
    preserves the most extreme up- and down-regulated genes for signed GSEA.
    """
    df = deg_df.copy()
    df[pval_col] = df[pval_col].clip(lower=min_pval)
    df["neg_log10_pval"] = -np.log10(df[pval_col])
    df["ranking_score"] = df[log2fc_col] * df["neg_log10_pval"]
    if top_n is not None and len(df) > top_n:
        df = df.reindex(df["ranking_score"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values("ranking_score", ascending=False)
    return df.set_index(gene_col)[["ranking_score"]].T


# ============================================================
# 3) pathway filtering
# ============================================================
def filter_pathways(
    pathways_df,
    *,
    size_min=15,
    size_max=500,
    keywords=None,
    source_col="source",
):
    """
    filter a decoupler pathway dataframe by gene-set size and optional keywords.
    returns the filtered pathway dataframe (not just the names).
    """
    sizes = pathways_df.groupby(source_col).size()
    keep = sizes.index[(sizes >= size_min) & (sizes <= size_max)]

    if keywords:
        keep = [
            p for p in keep
            if any(k.lower() in p.lower() for k in keywords)
        ]

    return pathways_df[pathways_df[source_col].isin(keep)].copy()


# ============================================================
# 4) decoupler GSEA wrapper
# ============================================================
def run_gsea(ranked_df, pathways_df):
    """
    run decoupler.mt.gsea on one (1 x n_genes) ranked DataFrame and one
    pathways DataFrame. returns a long DataFrame
        columns: source, score, pval
    sorted by ascending pval.
    """
    estimate, pvals = dc.mt.gsea(data=ranked_df, net=pathways_df)
    out = pd.concat(
        {"score": estimate.iloc[0], "pval": pvals.iloc[0]},
        axis=1,
    )
    out = out.sort_values("pval").reset_index()
    out = out.rename(columns={out.columns[0]: "source"})
    return out


# ============================================================
# 5) annotate GSEA results -> plot-ready DataFrame
# ============================================================
def simplify_term(term):
    """
    simplify a pathway term for de-duplication: lowercase, strip prefixes
    (GOBP/REACTOME/KEGG/WP/...), drop stop-words, sort remaining words.
    """
    words = term.lower().replace("_", " ").split()
    stop_words = {
        "regulation", "of", "positive", "negative", "mediated", "dependent",
        "activity", "process", "gobp", "gomf", "wp", "reactome",
        "kegg", "hallmark", "gocc",
    }
    words = [w for w in words if w not in stop_words]
    return " ".join(sorted(words))


def prepare_gsea_plot_df(
    gsea_df,
    *,
    name_max_tokens=6,
    name_regex=r"(?i)(GOBP|REACTOME|KEGG|HALLMARK|GOMF|GOCC|GRAESSMANN|WP)_(.+)",
):
    """
    annotate a GSEA result DataFrame with:
        -log10(pval), Group (Up/Down), Gene_set, Name, Term
    and keep the most-significant pathway per `Name`.
    """
    df = gsea_df.copy()

    df["-log10(pval)"] = -np.log10(df["pval"].replace(0, 1e-300))
    df["Group"] = df["score"].apply(lambda x: "Up" if x > 0 else "Down")

    df = df.reset_index(drop=True)

    extracted = df["source"].str.extract(name_regex)
    df[["Gene_set", "Name"]] = extracted

    df["Name"] = (
        df["Name"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.split("_")
        .str[:name_max_tokens]
        .str.join("_")
    )

    df = df.loc[df.groupby("Name")["-log10(pval)"].idxmax()]
    df["Term"] = df["source"]

    if "index" in df.columns:
        df = df.drop(columns=["index"])

    return df


def dedup_by_simplified_term(plot_df, *, group_order=("Up", "Down")):
    """
    add a `simple_term` column, keep the most-significant pathway per
    simplified term, then sort Up first / Down second by significance.
    """
    df = plot_df.copy()
    df["simple_term"] = df["Term"].apply(simplify_term)
    df = df.loc[df.groupby("simple_term")["-log10(pval)"].idxmax()]

    if "Group" in df.columns:
        df["Group"] = pd.Categorical(df["Group"], categories=list(group_order), ordered=True)
        df = df.sort_values(["Group", "-log10(pval)"], ascending=[True, False])

    return df


# ============================================================
# 6) common pathways + heatmap
# ============================================================
def common_pathways(plot_dfs, *, term_col="Term"):
    """
    intersection of `term_col` across N plot_dfs (dict of label -> df).
    """
    if not plot_dfs:
        return []
    iterator = iter(plot_dfs.values())
    shared = set(next(iterator)[term_col])
    for df in iterator:
        shared &= set(df[term_col])
    return list(shared)


def plot_pathway_heatmap(
    plot_dfs,
    pathways=None,
    *,
    title="Pathway Enrichment",
    score_col="-log10(pval)",
    term_col="Term",
    group_col="Group",
    up_label="Up",
    down_label="Down",
    up_cmap="Reds",
    down_cmap="Blues",
    sort_by=None,
    figsize=None,
):
    """
    side-by-side heatmap of pathway enrichment across cell-type labels.

    plot_dfs:
        dict {label: df} where each df has columns
        [term_col, group_col, score_col] (from prepare_gsea_plot_df).
        the order of labels in the dict becomes the column order.
    pathways:
        list of pathway terms to show. if None, uses the intersection
        of `term_col` across all dfs (so each row has a value per column).
    sort_by:
        column label to sort rows by; defaults to the first label in plot_dfs.
    """
    if not plot_dfs:
        raise ValueError("plot_dfs is empty")

    labels = list(plot_dfs.keys())

    if pathways is None:
        pathways = common_pathways(plot_dfs, term_col=term_col)

    # restrict every df to the chosen pathways and copy
    dfs = {
        lbl: df[df[term_col].isin(pathways)].copy()
        for lbl, df in plot_dfs.items()
    }

    def _stack(group_value):
        # merge per-label scores for one direction (Up or Down)
        merged = None
        for lbl, df in dfs.items():
            sub = (
                df[df[group_col] == group_value][[term_col, score_col]]
                .rename(columns={score_col: lbl})
            )
            merged = sub if merged is None else pd.merge(merged, sub, on=term_col, how="outer")
        if merged is None or merged.empty:
            return pd.DataFrame(columns=labels)
        merged = merged.set_index(term_col)[labels]
        merged = merged.fillna(0)
        sort_key = sort_by if (sort_by in labels) else labels[0]
        merged = merged.sort_values(sort_key, ascending=False)
        return merged

    up = _stack(up_label)
    down = _stack(down_label)

    n_up, n_down = len(up), len(down)
    if figsize is None:
        figsize = (max(4, 1.5 * len(labels) + 2), (n_up + n_down) * 0.5 + 2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[20, 1],
        height_ratios=[max(n_up, 1), max(n_down, 1)],
        hspace=0.3,
    )

    def _panel(df, row, cmap, panel_title):
        if df.empty:
            return
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])

        im = ax.imshow(
            df.values,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=df.values.max() if df.values.size else 1,
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=10)

        ax.set_title(panel_title, fontsize=12, fontweight="bold")

        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(score_col)

    _panel(up, 0, up_cmap, "Upregulated Pathways")
    _panel(down, 1, down_cmap, "Downregulated Pathways")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig, {"up": up, "down": down}


# ============================================================
# convenience: full pipeline
# ============================================================
def run_pipeline(
    deg_paths,
    pathways_gmt,
    *,
    gene_origin="mice",
    pathway_size_min=15,
    pathway_size_max=500,
    pathway_keywords=None,
    log2fc_col="logfoldchanges",
    pval_col="pvals_adj",
    gene_col="gene_name",
    rank_top_n=None,
    title=None,
    output_dir=None,
    save_basename=None,
):
    """
    run the full DEG -> functional enrichment pipeline.

    deg_paths:
        dict {cell_type_label: path_to_csv}. labels become the heatmap columns.
        any sex / any number of cell types.
    pathways_gmt:
        path to a GMT file (uses pygenelab.convert_gmt_to_decoupler_format).
    pathway_keywords:
        optional list of substrings to filter pathway names.

    returns a dict with all intermediates:
        degs, ranked_lists, pathways, gsea_results, plot_dfs,
        common_terms, heatmap_fig, heatmap_panels
    """
    from .utils import convert_gmt_to_decoupler_format

    # 1. load DEGs
    degs = load_deg_csvs(deg_paths, gene_col=gene_col)

    # 2. ranked gene lists
    ranked_lists = {
        lbl: create_ranked_genelist(
            df,
            log2fc_col=log2fc_col,
            pval_col=pval_col,
            gene_col=gene_col,
            top_n=rank_top_n,
        )
        for lbl, df in degs.items()
    }

    # 3. pathways
    pathways = convert_gmt_to_decoupler_format(pathways_gmt, gene_origin=gene_origin)
    pathways = filter_pathways(
        pathways,
        size_min=pathway_size_min,
        size_max=pathway_size_max,
        keywords=pathway_keywords,
    )

    # 4. GSEA per ranked list
    gsea_results = {lbl: run_gsea(r, pathways) for lbl, r in ranked_lists.items()}

    # 5. prepare plot dfs + dedup by simplified term
    plot_dfs = {
        lbl: dedup_by_simplified_term(prepare_gsea_plot_df(res))
        for lbl, res in gsea_results.items()
    }

    # 6. heatmap
    common_terms = common_pathways(plot_dfs)
    heatmap_fig, panels = plot_pathway_heatmap(
        plot_dfs,
        pathways=common_terms,
        title=title or "Pathway Enrichment",
    )

    if output_dir is not None and save_basename is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_fig.savefig(out_dir / f"{save_basename}.png", dpi=300, bbox_inches="tight")
        heatmap_fig.savefig(out_dir / f"{save_basename}.svg", bbox_inches="tight")

    return {
        "degs": degs,
        "ranked_lists": ranked_lists,
        "pathways": pathways,
        "gsea_results": gsea_results,
        "plot_dfs": plot_dfs,
        "common_terms": common_terms,
        "heatmap_fig": heatmap_fig,
        "heatmap_panels": panels,
    }
