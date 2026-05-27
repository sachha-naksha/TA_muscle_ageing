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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
