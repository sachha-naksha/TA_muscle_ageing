# LF_viz.py

"""
SLIDE latent-factor visualization utilities.

Functions for plotting SLIDE LF feature lists:
  * plot_lfs_dotplot_stacked  — multi-LF stacked AUC lollipop dotplot
  * plot_lf_aloading_pathway_bar — pathway-stacked A_loading bar plot

Both expect feature-list DataFrames (columns: 'names', 'A_loading', 'AUCs', ...)
together with a pathway_map (gene -> pathway) and a palette (pathway -> hex).
SVGs are saved with editable text (svg.fonttype = "none") into figures_dir.
"""

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# Nature-style + Illustrator-editable text. Applied at import so any notebook
# that uses these plotting utils gets consistent typography without boilerplate.
mpl.rcParams.update({
    "svg.fonttype":      "none",
    "ps.fonttype":       42,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
})


def plot_lfs_dotplot_stacked(
    lf_specs,
    figures_dir,
    sex=None,
    metric="AUCs",
    metric_label=None,
    max_genes=30,
    default_pathway="Non-coding/uncharacterized",
    figsize=None,
    row_height=0.22,
    marker_size=55,
    tick_fontsize=8,
    hspace=0.06,
    legend_title="Classifications",
    mark_common=True,
):
    """
    Stacked lollipop dotplot showing multiple SLIDE LFs in one figure
    with a shared x-axis (e.g. AUC). One panel per LF, top->bottom, each
    panel sized proportional to its kept gene count.

    Strategy
    --------
    1. Every gene that is UNIQUE to one LF is kept in that LF's panel.
    2. For COMMON genes (present in 2+ LFs), the gene is placed in the
       LF where its `metric` (e.g. AUC) is HIGHEST. The other copies are
       dropped - each gene appears at most once across the figure.
    3. If the resulting total > `max_genes`, COMMON genes with the lowest
       (assigned-LF) metric are dropped first; uniques are always kept.
    4. Within each panel, rows are sorted by `metric` descending.

    lf_specs: list of dicts with keys:
        df            (DataFrame with 'names' + metric column)
        pathway_map   (dict {gene: pathway})
        palette       (dict {pathway: hex})
        latent_factor (str, used as panel label + filename)
    """
    metric_label = metric_label if metric_label is not None else metric

    # ---- collect per-LF dataframes with pathway annotation ---------------
    lf_dfs = []
    for spec in lf_specs:
        if metric not in spec["df"].columns:
            raise KeyError(
                f"metric column '{metric}' missing for {spec['latent_factor']}"
            )
        d = spec["df"][["names", metric]].copy()
        d["pathway"] = (spec["df"]["names"]
                        .map(spec["pathway_map"]).fillna(default_pathway))
        lf_dfs.append(d)

    # ---- build {gene -> [(lf_index, metric_value)]} ---------------------
    gene_appearances = {}
    for idx, d in enumerate(lf_dfs):
        for _, r in d.iterrows():
            gene_appearances.setdefault(r["names"], []).append(
                (idx, float(r[metric]))
            )

    # ---- assign each gene to ONE LF (unique -> its LF; common -> max-metric LF)
    assignments = []   # list of (lf_index, gene_name, metric_value, is_common)
    for gene, hits in gene_appearances.items():
        is_common = len(hits) > 1
        best_idx, best_val = max(hits, key=lambda t: t[1])
        assignments.append((best_idx, gene, best_val, is_common))

    # ---- enforce max_genes by trimming commons first ---------------------
    uniques = [a for a in assignments if not a[3]]
    commons = sorted([a for a in assignments if a[3]],
                     key=lambda a: a[2], reverse=True)

    if len(uniques) > max_genes:
        print(f"⚠ {len(uniques)} unique genes exceed max_genes={max_genes}; "
              f"all uniques kept anyway, no commons shown.")
        commons_keep = []
    else:
        commons_keep = commons[: max_genes - len(uniques)]

    kept = uniques + commons_keep
    if not kept:
        raise ValueError("Nothing to plot.")

    # ---- build per-panel dataframes (preserve lf_specs order) ------------
    panels = []
    for idx, spec in enumerate(lf_specs):
        rows = []
        for (lf_idx, gene, val, is_common) in kept:
            if lf_idx != idx:
                continue
            src = lf_dfs[idx]
            pw = src.loc[src["names"] == gene, "pathway"].iloc[0]
            rows.append({"names": gene, metric: val,
                         "pathway": pw, "is_common": is_common})
        if rows:
            sub = (pd.DataFrame(rows)
                   .sort_values(metric, ascending=False)
                   .reset_index(drop=True))
            panels.append((spec, sub))

    if not panels:
        raise ValueError("No genes to plot.")

    # merged palette for legend (later LF wins on label collisions)
    merged_palette = {}
    for spec, _ in panels:
        merged_palette.update(spec["palette"])

    # ---- figure layout ---------------------------------------------------
    heights = [len(sub) for _, sub in panels]
    total_rows = sum(heights)
    if figsize is None:
        figsize = (7.5, max(3.0, row_height * total_rows + 0.9))

    fig, axes = plt.subplots(
        nrows=len(panels), ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": hspace},
        figsize=figsize,
    )
    if len(panels) == 1:
        axes = [axes]

    # shared x-range over all kept metric values
    all_vals = pd.concat([sub[metric] for _, sub in panels])
    vmax = float(all_vals.max())
    vmin_floor = min(0.0, float(all_vals.min()))
    x_pad = (vmax - vmin_floor) * 0.04
    line_start = 0.0 if vmin_floor >= 0 else vmin_floor

    # ---- per-panel lollipops --------------------------------------------
    for ax, (spec, sub) in zip(axes, panels):
        pal = spec["palette"]
        y_pos = list(range(len(sub)))
        for i, (_, row) in enumerate(sub.iterrows()):
            c = pal.get(row["pathway"], "#CFCFCF")
            ax.plot([line_start, row[metric]], [i, i],
                    linestyle="--", color="0.65", linewidth=0.8,
                    dash_capstyle="round", zorder=1)
            ax.scatter(row[metric], i,
                       color=c, s=marker_size, edgecolor="white",
                       linewidth=0.6, zorder=2, clip_on=False)
            if mark_common and row["is_common"]:
                ax.text(row[metric] + x_pad * 0.4, i, "•",
                        ha="left", va="center", color="0.4",
                        fontsize=tick_fontsize + 1, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["names"], fontstyle="italic",
                           fontsize=tick_fontsize)
        ax.invert_yaxis()
        ax.set_ylim(len(sub) - 0.5 + 0.25, -0.5 - 0.25)
        ax.set_ylabel(spec["latent_factor"], fontsize=11,
                      fontweight="bold", labelpad=8, rotation=0,
                      va="center", ha="right")

        ax.xaxis.grid(True, linestyle=":", color="0.85",
                      linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=3)
        ax.tick_params(axis="y", pad=2, length=0)

    # shared x range + label
    axes[-1].set_xlim(vmin_floor - x_pad * 0.5, vmax + x_pad * 3)
    axes[-1].set_xlabel(metric_label, labelpad=8)

    # ---- legend (right side, only pathways present) ---------------------
    present = set()
    for _, sub in panels:
        present.update(sub["pathway"].tolist())
    ordered = [p for p in merged_palette.keys() if p in present]
    handles = [Line2D([0], [0], marker="o", linestyle="",
                      color=merged_palette[p], label=p, markersize=8)
               for p in ordered]
    if mark_common:
        handles.append(Line2D([0], [0], marker="$•$", linestyle="",
                              color="0.4", label="in both LFs (assigned to higher AUC)",
                              markersize=9))
    leg = axes[0].legend(handles=handles, title=legend_title,
                         frameon=False, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0),
                         labelspacing=0.9, borderpad=0.0,
                         handletextpad=0.6,
                         fontsize=9, title_fontsize=10)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"

    # summary: per-panel counts (uniques + commons)
    summary = "   ".join(
        f"{spec['latent_factor']}: {len(sub)} "
        f"({int((~sub['is_common']).sum())}u + {int(sub['is_common'].sum())}c)"
        for spec, sub in panels
    )
    fig.suptitle(summary, fontsize=9, y=0.995)
    plt.subplots_adjust(left=0.16, right=0.74, top=0.96, bottom=0.08)

    # ---- save ------------------------------------------------------------
    safe_metric = str(metric).replace("/", "_").replace(" ", "_")
    lf_tag = "_".join(spec["latent_factor"] for spec, _ in panels)
    parts = [lf_tag]
    if sex is not None:
        parts.append(str(sex))
    parts += [safe_metric, "stacked_dotplot"]
    out_path = f"{figures_dir}/{'_'.join(parts)}.svg"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


def plot_lf_aloading_pathway_bar(
    df,
    pathway_map,
    palette,
    latent_factor,
    figures_dir,
    default_pathway="Non-coding/uncharacterized",
    figsize=(4.8, 6.2),
    legend_title="Classifications",
    sex=None,
    hero_threshold=0.9,
    show_legend=True,
):
    """
    Histogram-style stacked bar plot of A_loadings, one bar per pathway.

    - Bars touch (width=1.0), pathway order = ascending sum of A_loadings,
      so the hero pathway towers on the far RIGHT.
    - Each gene contributes a stacked segment colored by its pathway.
    - Genes with A_loading >= hero_threshold get their name written
      vertically inside the segment.
    - Y-axis is on the right to mirror the reference figure.
    """
    df = df.copy()
    df["pathway"] = df["names"].map(pathway_map).fillna(default_pathway)

    # Pathway order: total A_loading ascending -> hero pathway on the right
    pw_totals = (df.groupby("pathway")["A_loading"]
                   .sum().sort_values(ascending=True))
    pathway_order = pw_totals.index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    for xi, pw in enumerate(pathway_order):
        # Preserve original file order; iterate in reverse so the row that
        # appears first in the file is stacked LAST and sits on top.
        sub = df[df["pathway"] == pw]
        bottom = 0.0
        c = palette[pw]
        for _, row in sub.iloc[::-1].iterrows():
            h = row["A_loading"]
            ax.bar(xi, h, bottom=bottom,
                   color=c, edgecolor="white",
                   linewidth=0.6, width=1.0, zorder=2)

            if h >= hero_threshold:
                ax.text(xi, bottom + h / 2, row["names"],
                        ha="center", va="center",
                        fontsize=9, fontstyle="italic",
                        fontweight="bold", color="white",
                        rotation=90)
            bottom += h

    # --- Axes ---------------------------------------------------------
    ax.set_xticks(np.arange(len(pathway_order)))
    ax.set_xticklabels(pathway_order, rotation=55,
                       ha="right", fontsize=9)
    ax.set_xlim(-0.5, len(pathway_order) - 0.5)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Σ A loading", labelpad=8)

    ymax = pw_totals.max() * 1.08
    ax.set_ylim(0, ymax)

    ax.yaxis.grid(True, linestyle=":", color="0.85",
                  linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=3, pad=2)
    ax.tick_params(axis="y", length=3, pad=4,
                   left=False, right=True,
                   labelleft=False, labelright=True)

    # --- Legend (upper-left like the reference) -----------------------
    if show_legend:
        legend_pathways = pathway_order[::-1]
        handles = [Line2D([0], [0], marker="s", linestyle="",
                          color=palette[p], label=p,
                          markersize=9, markeredgewidth=0)
                   for p in legend_pathways]
        leg = ax.legend(handles=handles, title=legend_title,
                        frameon=False, loc="upper left",
                        bbox_to_anchor=(0.02, 0.98),
                        labelspacing=0.7, borderpad=0.0,
                        handletextpad=0.5, fontsize=8,
                        title_fontsize=9)
        leg.get_title().set_fontweight("bold")
        leg._legend_box.align = "left"

    plt.tight_layout()

    suffix = f"_{sex}" if sex else ""
    out_path = (f"{figures_dir}/{latent_factor}{suffix}"
                "_Aloading_pathway_bar.svg")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ---------------------------------------------------------------------------
# SLIDE-style correlation networks (port of R plotCorrelationNetworks /
# qgraph::qgraph). One network per latent factor: nodes = genes in the LF,
# edge weight = pairwise Spearman r between genes (filtered by |r| >=
# minimum). Node color encodes each gene's relationship to y; edge color
# encodes the sign of the gene-gene correlation.
# ---------------------------------------------------------------------------

_POS_EDGE_COLOR = "#40006D"  # qgraph posCol
_NEG_EDGE_COLOR = "#59A14F"  # qgraph negCol
_HIGH_Y_COLOR   = "salmon"   # gene up with high y / class 1
_LOW_Y_COLOR    = "skyblue"  # gene up with low  y / class 0
_NEUTRAL_COLOR  = "lightgray"


def _node_colors_from_y(
    x_gene,
    y,
    high_y_color=_HIGH_Y_COLOR,
    low_y_color=_LOW_Y_COLOR,
    neutral_color=_NEUTRAL_COLOR,
):
    """Color each gene by its association with y.

    Binary y -> AUC of the gene as a predictor (>0.5 high_y_color,
    <0.5 low_y_color). Continuous y -> Spearman r (>0 high_y_color,
    <0 low_y_color). Mirrors the R code's glmnet::auc / cor(method="spearman").

    high_y_color  -> gene UP with the higher Y class (e.g. KO / Y=1)
    low_y_color   -> gene UP with the lower  Y class (e.g. WT / Y=0)
    neutral_color -> at the threshold or insufficient data
    """
    y = np.asarray(y).ravel()
    uniq = np.unique(y[~pd.isna(y)])
    colors = []
    if uniq.size == 2:
        y_bin = (y == uniq.max()).astype(int)
        for g in x_gene.columns:
            xs = np.asarray(x_gene[g])
            mask = ~(np.isnan(xs) | np.isnan(y_bin.astype(float)))
            if mask.sum() < 2 or len(np.unique(y_bin[mask])) < 2:
                colors.append(neutral_color)
                continue
            a = roc_auc_score(y_bin[mask], xs[mask])
            colors.append(high_y_color if a > 0.5
                          else low_y_color if a < 0.5
                          else neutral_color)
    else:
        for g in x_gene.columns:
            xs = np.asarray(x_gene[g])
            mask = ~(np.isnan(xs) | np.isnan(y.astype(float)))
            if mask.sum() < 3:
                colors.append(neutral_color)
                continue
            r = spearmanr(y[mask], xs[mask]).statistic
            colors.append(high_y_color if r > 0
                          else low_y_color if r < 0
                          else neutral_color)
    return colors


def _edge_rgba(base_color, rs, min_alpha, max_alpha):
    """Per-edge RGBA colors with alpha scaled to |r|.

    Mirrors qgraph behavior where weak edges fade out and strong edges
    stay solid, so the visual edge weight tracks the correlation strength.
    """
    r0, g0, b0, _ = to_rgba(base_color)
    span = max_alpha - min_alpha
    return [(r0, g0, b0, min(max_alpha, min_alpha + span * abs(r))) for r in rs]


def plot_lf_correlation_network(
    x,
    feature_list,
    y,
    latent_factor,
    out_dir,
    minimum=0.25,
    repulsion=2.0,
    iterations=300,
    layout="spring",            # "spring" | "kamada_kawai" | "circular"
    figsize=(9, 7),
    node_size=1400,
    font_size=8,
    label_color="black",
    max_edge_width=4.5,
    min_edge_alpha=0.25,
    max_edge_alpha=1.0,
    high_y_color=_HIGH_Y_COLOR,
    low_y_color=_LOW_Y_COLOR,
    neutral_color=_NEUTRAL_COLOR,
    filetype="pdf",
    also_svg=True,
    seed=1,
    show=True,
):
    """Plot a qgraph-style correlation network for one SLIDE latent factor.

    Parameters
    ----------
    x : pd.DataFrame
        Sample x gene expression matrix (rows samples, columns genes).
    feature_list : pd.DataFrame
        Must contain a 'names' column listing genes in this LF (matches
        SLIDE's feature_list_Z*.txt / gene_list_Z*.txt).
    y : array-like
        Response vector (binary or continuous), aligned with rows of x.
    latent_factor : str
        Name used for the plot title and output filename (e.g. "Z12").
    out_dir : str | Path
        Directory the figure is written to (created if missing).
    minimum : float
        Drop edges with |Spearman r| below this threshold (qgraph minimum).
    repulsion : float
        Spring-layout repulsion. Spread is set by k = repulsion / sqrt(n_nodes);
        higher = more spread. Default 2.0 gives a layout that avoids overlap for
        ~10-30 node networks. The qgraph default of 0.1 produces clumped layouts
        in networkx (k ~ 0.02) and should not be used here.
    iterations : int
        Number of spring-layout iterations (more = better convergence).
    layout : {"spring", "kamada_kawai", "circular"}
        Layout algorithm. Spring is qgraph's default; kamada_kawai is
        typically the most overlap-free for small dense graphs.
    max_edge_width : float
        Edge width at |r| = 1; widths scale linearly with |r|.
    min_edge_alpha, max_edge_alpha : float
        Edge opacity at |r| = minimum vs |r| = 1. Edges at intermediate
        strength fade between these.
    high_y_color, low_y_color, neutral_color : str | tuple
        Node colors for genes UP with high-Y class, UP with low-Y class, and
        neutral. Pass SEX_CONDITION_PALETTE[(sex, 'KO')] and [(sex, 'WT')]
        to make node colors match the rest of the notebook's color scheme.
    also_svg : bool
        In addition to <out_dir>/<LF>.<filetype>, also save an editable-text
        SVG sibling (svg.fonttype="none") for Illustrator polish.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = [g for g in feature_list["names"].dropna().tolist()
             if g in x.columns]
    if len(genes) < 2:
        print(f"[{latent_factor}] <2 genes overlap with x; skipping.")
        return None

    x_gene = x.loc[:, genes].astype(float)
    node_colors = _node_colors_from_y(
        x_gene, y,
        high_y_color=high_y_color,
        low_y_color=low_y_color,
        neutral_color=neutral_color,
    )

    # Spearman correlation matrix across genes.
    corr = x_gene.corr(method="spearman").values
    np.fill_diagonal(corr, 0.0)

    G = nx.Graph()
    G.add_nodes_from(range(len(genes)))
    pos_edges, neg_edges = [], []
    pos_widths, neg_widths = [], []
    pos_rs, neg_rs = [], []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            r = corr[i, j]
            if np.isnan(r) or abs(r) < minimum:
                continue
            G.add_edge(i, j, weight=r)
            w = max_edge_width * abs(r)
            if r >= 0:
                pos_edges.append((i, j))
                pos_widths.append(w)
                pos_rs.append(r)
            else:
                neg_edges.append((i, j))
                neg_widths.append(w)
                neg_rs.append(r)

    # --- Layout -------------------------------------------------------
    n = len(genes)
    if layout == "kamada_kawai" and G.number_of_edges() > 0:
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        # spring (Fruchterman-Reingold). k controls ideal node spacing.
        # networkx default k = 1/sqrt(n) ~ 0.23 for n=19; we scale that up
        # by `repulsion` so the user has a single intuitive knob.
        k = repulsion / np.sqrt(max(n, 1))
        pos = nx.spring_layout(
            G, k=k, iterations=iterations, seed=seed, scale=1.5
        )

    # --- Draw ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    if neg_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=neg_edges, ax=ax,
            width=neg_widths,
            edge_color=_edge_rgba(
                _NEG_EDGE_COLOR, neg_rs, min_edge_alpha, max_edge_alpha
            ),
        )
    if pos_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=pos_edges, ax=ax,
            width=pos_widths,
            edge_color=_edge_rgba(
                _POS_EDGE_COLOR, pos_rs, min_edge_alpha, max_edge_alpha
            ),
        )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size,
        node_color=node_colors, node_shape="o",
        edgecolors="black", linewidths=0.6,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: g for i, g in enumerate(genes)},
        font_size=font_size, font_color=label_color,
    )

    ax.set_title(latent_factor, fontsize=12, fontweight="bold")
    ax.set_axis_off()
    ax.margins(0.12)        # breathing room so outer nodes aren't clipped
    plt.tight_layout()

    out_path = out_dir / f"{latent_factor}.{filetype}"
    plt.savefig(out_path, bbox_inches="tight")
    if also_svg and filetype != "svg":
        plt.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


_RUN_DIR_RE = re.compile(r"(\d+\.?\d*_\d+\.?\d*_out|_out)$")
_LF_FILE_RE = re.compile(r"(feature_list_Z\d+|gene_list_Z\d+)")
_LF_NUM_RE  = re.compile(r"Z\d+")


def plot_correlation_networks(input_params, minimum=0.25, **kwargs):
    """Python port of R `plotCorrelationNetworks`.

    Walks `input_params['out_path']` for SLIDE run directories matching
    `*_out`, then for each `feature_list_Z*` / `gene_list_Z*` file writes a
    correlation-network PDF into `<run_dir>/correlation_networks/`.

    `input_params` is a dict with keys:
        out_path : str   parent dir holding one or more *_out run dirs
        x_path   : str   CSV of expression (rows=samples, cols=genes;
                         first column = sample IDs)
        y_path   : str   CSV of response (rows=samples; first column = ID)

    Extra kwargs are forwarded to `plot_lf_correlation_network`.
    """
    out_root = input_params["out_path"]

    run_dirs = [os.path.join(out_root, d) for d in os.listdir(out_root)
                if os.path.isdir(os.path.join(out_root, d))
                and _RUN_DIR_RE.search(d)]
    if not run_dirs:
        run_dirs = [out_root]

    x = pd.read_csv(input_params["x_path"], index_col=0)
    x.columns = x.columns.str.replace(" ", "_", regex=False)
    y = pd.read_csv(input_params["y_path"], index_col=0).iloc[:, 0].values

    written = []
    for r in run_dirs:
        net_dir = os.path.join(r, "correlation_networks")
        feature_files = [os.path.join(r, f) for f in os.listdir(r)
                         if _LF_FILE_RE.search(f)]
        if not feature_files:
            print(f"[{r}] no feature lists found; run optimizeSLIDE first.")
            continue
        for f in feature_files:
            m = _LF_NUM_RE.search(os.path.basename(f))
            if not m:
                continue
            lf_num = m.group(0)
            fl = pd.read_csv(f, sep=r"\s+").dropna()
            path = plot_lf_correlation_network(
                x, fl, y,
                latent_factor=lf_num,
                out_dir=net_dir,
                minimum=minimum,
                **kwargs,
            )
            if path is not None:
                written.append(path)
    return written
