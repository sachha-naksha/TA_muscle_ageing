# plotting.py 

"""
functions relating to developing plots
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import decoupler as dc

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, ttest_ind
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.path import Path
import matplotlib.patches as patches

from .utils import calculate_pairwise_significance, cliffs_delta


# plot_gene_contribution_heatmap
def plot_gene_contribution_heatmap(
    ranked_dfs,
    top_n=20,
    gene_col="gene",
    corr_col="spearman_corr",
    sort_by="max",
    fill_value=0,
    figsize=(6, 8),
    cmap="Reds",
    title="Gene Contribution",
    annot=True,
    fmt=".2f",
    vmin=0,
    vmax=None
):
    """
    plot heatmap of top gene contribution values across multiple categories
    input: dictionary of ranked dfs
    """

    if not isinstance(ranked_dfs, dict):
        raise ValueError("ranked_dfs must be a dictionary like {'KO': ko_df, 'WT': wt_df}")

    if len(ranked_dfs) < 2:
        raise ValueError("ranked_dfs must contain at least two categories")

    category_dfs = []

    # prepare each category dataframe
    for category, df in ranked_dfs.items():
        temp_df = df[[gene_col, corr_col]].copy()
        temp_df = temp_df.rename(columns={corr_col: category})
        category_dfs.append(temp_df)

    # merge all categories by gene
    heatmap_df = category_dfs[0]

    for temp_df in category_dfs[1:]:
        heatmap_df = pd.merge(
            heatmap_df,
            temp_df,
            on=gene_col,
            how="outer"
        )

    # get category columns
    category_cols = list(ranked_dfs.keys())

    # fill missing genes if needed
    heatmap_df[category_cols] = heatmap_df[category_cols].fillna(fill_value)

    # choose how to rank genes
    if sort_by == "max":
        heatmap_df["sort_value"] = heatmap_df[category_cols].max(axis=1)

    elif sort_by == "mean":
        heatmap_df["sort_value"] = heatmap_df[category_cols].mean(axis=1)

    elif sort_by == "abs_max":
        heatmap_df["sort_value"] = heatmap_df[category_cols].abs().max(axis=1)

    elif sort_by == "abs_mean":
        heatmap_df["sort_value"] = heatmap_df[category_cols].abs().mean(axis=1)

    elif sort_by in category_cols:
        heatmap_df["sort_value"] = heatmap_df[sort_by]

    else:
        raise ValueError(
            f"sort_by must be one of: max, mean, abs_max, abs_mean, or {category_cols}"
        )

    # select top genes
    heatmap_df = (
        heatmap_df
        .sort_values("sort_value", ascending=False)
        .head(top_n)
        .drop(columns="sort_value")
        .set_index(gene_col)
    )

    # create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("Gene")

    plt.tight_layout()
    plt.close(fig)

    # return dataframe and plot objects
    return heatmap_df, fig, ax


# plot_violin_box_combo
def _compress_group_spacing(ax, n_categories, spacing, x_pad=0.5):
    """move categorical artists from integer positions (0,1,2,...) to positions
    `spacing` apart, without resizing the violins/boxes/scatter. `spacing=1.0`
    leaves the plot unchanged; `spacing<1` reduces the gap between groups.
    """
    from matplotlib.collections import PathCollection
    if n_categories < 2:
        return
    center = (n_categories - 1) / 2.0
    shifts = {i: (i - center) * (spacing - 1.0) for i in range(n_categories)}

    def _idx(x):
        return max(0, min(n_categories - 1, int(round(x))))

    # Collections: PathCollection (scatter/strip) shifts via offsets;
    # PolyCollection (violins) shifts via path vertices. Mixing them caused
    # double-shifts because PolyCollection.get_offsets() returns [[0,0]]
    # by default rather than an empty array.
    seen_paths = set()
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            offsets = np.array(coll.get_offsets(), dtype=float)
            if len(offsets) > 0:
                for j in range(len(offsets)):
                    if not np.isnan(offsets[j, 0]):
                        offsets[j, 0] += shifts[_idx(offsets[j, 0])]
                coll.set_offsets(offsets)
        else:
            for path in coll.get_paths():
                if id(path) in seen_paths:
                    continue
                seen_paths.add(id(path))
                v = path.vertices
                if len(v):
                    v[:, 0] += shifts[_idx(v[:, 0].mean())]

    # Patches: the colored edge PathPatches share their path object with the
    # underlying box (created via `box.get_path()`), so dedup by id(path) to
    # avoid shifting the same box twice.
    for patch in ax.patches:
        path = patch.get_path()
        if id(path) in seen_paths:
            continue
        seen_paths.add(id(path))
        v = path.vertices
        if len(v):
            v[:, 0] += shifts[_idx(v[:, 0].mean())]

    # Lines (whiskers, medians, caps, significance bars)
    for line in ax.lines:
        xd = np.asarray(line.get_xdata(), dtype=float)
        if len(xd) == 0:
            continue
        line.set_xdata(xd + np.array([shifts[_idx(x)] for x in xd]))

    # Texts in data coords (significance labels span two categories)
    for txt in ax.texts:
        if txt.get_transform() is ax.transData:
            x, y = txt.get_position()
            i_lo = max(0, min(n_categories - 1, int(np.floor(x))))
            i_hi = max(0, min(n_categories - 1, int(np.ceil(x))))
            if i_lo == i_hi:
                shift = shifts[i_lo]
            else:
                frac = x - i_lo
                shift = shifts[i_lo] * (1 - frac) + shifts[i_hi] * frac
            txt.set_position((x + shift, y))

    ax.set_xticks([i + shifts[i] for i in range(n_categories)])
    half_extent = (n_categories - 1) * spacing / 2.0 + x_pad
    ax.set_xlim(center - half_extent, center + half_extent)


def plot_violin_box_combo(data, x_var, y_var, title=None, x_ticks=None,
                          palette=None, rotation=45, show_scatter=True,
                          figsize=(5, 6), scatter_size=4, scatter_alpha=0.6,
                          violin_width=0.7, box_width=0.35, jitter=0.15,
                          show_pvalue=True, delta_label=None,
                          group_spacing=1.0, x_pad=0.5, ylim=None):
    """
    Create a combined violin-box plot with optional scatter points.

    Parameters
    ----------
    show_scatter : bool, default True
        Overlay individual data points as a strip plot.
    figsize : tuple, default (5, 6)
        Figure dimensions in inches.
    scatter_size : float, default 4
        Marker size for scatter points.
    scatter_alpha : float, default 0.6
        Marker transparency (0–1).
    violin_width : float, default 0.7
        Width of the violin bodies.
    box_width : float, default 0.35
        Width of the box plots.
    jitter : float, default 0.15
        Horizontal jitter for scatter points.
    show_pvalue : bool, default True
        If True, display the numerical p-value alongside significance symbols.
        If False, show only the asterisk symbols (*, **, ***).
    delta_label : str or None, default None
        If given, annotate the plot with this string (e.g. "Cliff's δ = 0.42")
        in the top-left of the axes.
    group_spacing : float, default 1.0
        Distance between adjacent category centers, in data units. The default
        1.0 matches seaborn's built-in behavior. Set to a value <1 (e.g. 0.8)
        to reduce the gap between groups without shrinking the violins, boxes,
        or scatter points themselves. Must remain > violin_width to avoid
        overlap (default violin_width=0.7, so group_spacing>=0.75 is safe).
    x_pad : float, default 0.5
        Half-width of the empty margin on each side of the outermost category,
        in data units. Independent of group_spacing — only controls outer
        padding. Must be >= violin_width / 2 to avoid clipping edge violins.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.12, top=0.88)

    # ── y-axis limits ──────────────────────────────────────────────
    if ylim is not None:
        y_min_plot, y_max_plot = ylim
        y_range = y_max_plot - y_min_plot
    else:
        y_min, y_max = data[y_var].min(), data[y_var].max()
        y_range = y_max - y_min
        padding = y_range * 0.10
        y_min_plot = y_min - padding
        y_max_plot = y_max + padding

        if y_range > 1.0:
            y_min_plot = np.floor(y_min_plot)
            y_max_plot = np.ceil(y_max_plot)
        else:
            y_min_plot = max(0, y_min_plot)

    ax.set_ylim(y_min_plot, y_max_plot)

    # ── smart tick interval based on range ─────────────────────────
    if y_range < 0.1:
        tick_interval = 0.02
    elif y_range < 0.5:
        tick_interval = 0.05
    elif y_range < 2.0:
        tick_interval = 0.25
    elif y_range < 5.0:
        tick_interval = 0.5
    elif y_range < 15.0:
        tick_interval = 2.0
    elif y_range < 30.0:
        tick_interval = 5.0
    else:
        tick_interval = 10.0

    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # ── category order ─────────────────────────────────────────────
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(
            data[x_var].unique(),
            key=lambda x: float(x) if str(x).replace('.', '').isdigit() else x,
        )

    # ── violin plot ────────────────────────────────────────────────
    sns.violinplot(
        data=data, x=x_var, y=y_var,
        order=categories, palette=palette,
        inner=None, linewidth=0, saturation=1.0,
        alpha=0.45, width=violin_width, cut=0, ax=ax,
    )

    # ── box plot (skeleton only – we colour it below) ──────────────
    sns.boxplot(
        data=data, x=x_var, y=y_var,
        order=categories, width=box_width, linewidth=1.2,
        flierprops={'marker': ' '}, showmeans=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'none'},
        whiskerprops={'color': 'none'},
        medianprops={'color': 'none'},
        showcaps=False, ax=ax,
    )

    # ── colour boxes per category ──────────────────────────────────
    import matplotlib.patches as mpatches

    num_boxes = len(categories)
    lines_per_box = len(ax.lines) // num_boxes if num_boxes else 0

    for i, (name, box) in enumerate(zip(categories, ax.patches)):
        color = palette[name]

        box.set_facecolor(color)
        box.set_edgecolor('none')
        box.set_alpha(0.45)
        box.set_zorder(1)

        edges = mpatches.PathPatch(
            box.get_path(), facecolor='none',
            edgecolor=color, linewidth=1.2, alpha=1.0, zorder=2,
        )
        ax.add_patch(edges)

        for line in ax.lines[i * lines_per_box:(i + 1) * lines_per_box]:
            line.set_color(color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)
            line.set_zorder(2)

    # ── scatter overlay ────────────────────────────────────────────
    if show_scatter:
        sns.stripplot(
            data=data, x=x_var, y=y_var,
            order=categories, palette=palette,
            size=scatter_size, alpha=scatter_alpha,
            linewidth=0, jitter=jitter, zorder=3, ax=ax,
        )

    # ── significance bars ──────────────────────────────────────────
    significance_info = calculate_pairwise_significance(
        data, categories, x_var, y_var,
    )

    current_ymin, current_ymax = ax.get_ylim()
    y_range_plot = current_ymax - current_ymin
    bar_spacing = y_range_plot * 0.08
    bar_tips = y_range_plot * 0.02
    bar_height = current_ymax + bar_spacing * 0.5

    def add_significance_bar(start, end, height, p_value, sig_symbol):
        ax.plot(
            [start, start, end, end],
            [height, height + bar_tips, height + bar_tips, height],
            color='black', linewidth=0.8,
        )
        if not show_pvalue:
            text = sig_symbol
        elif p_value < 0.00005:
            text = f'p = {p_value:.2e} {sig_symbol}'
        else:
            text = f'p = {p_value:.4f} {sig_symbol}'
        ax.text(
            (start + end) * 0.5, height + bar_tips,
            text, ha='center', va='bottom', fontsize=8,
        )

    for (g1, g2), sig_data in significance_info.items():
        if sig_data['significance'] != 'ns':
            add_significance_bar(g1, g2, bar_height,
                                 sig_data['p-value'],
                                 sig_data['significance'])
            bar_height += bar_spacing

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(current_ymin, bar_height + bar_spacing * 0.5)

    # ── titles & labels ────────────────────────────────────────────
    if title:
        plt.title(title, pad=20)

    if delta_label:
        ax.text(
            0.02, 0.98, delta_label,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      boxstyle='round,pad=0.3', alpha=0.85),
            zorder=10,
        )

    if x_ticks is None:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=rotation, ha='right')
        ax.spines['bottom'].set_visible(True)

    # ── spine / tick cosmetics ─────────────────────────────────────
    ax.minorticks_off()
    ax.tick_params(axis='both', which='minor',
                   bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis='x', which='major', top=False)
    ax.tick_params(axis='y', which='major', right=False, width=0.8)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(width=0.8)

    plt.setp(ax.get_yticklabels(), weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.grid(False)

    sns.despine(offset=5, trim=True,
                bottom=(x_ticks is None), right=True)

    if x_ticks is not None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

    _compress_group_spacing(ax, len(categories), group_spacing, x_pad=x_pad)
    if x_ticks is not None:
        ax.set_xticklabels(x_ticks, rotation=rotation, ha='right')

    plt.close()
    return fig


# plot_gene_expression_by_group
def plot_gene_expression_by_group(
    adata,
    gene,
    group_col,
    layer=None,
    x_ticks=None,
    palette=None,
    title=None,
    rotation=45,
    show_scatter=True,
    dropna=True
):
    """
    plot expression of one gene across groups
    """

    # plot_gene_expression_by_group
    # api:
    # plot_gene_expression_by_group(
    #     adata,
    #     gene="Pdk4",
    #     group_col="age",
    #     layer=None,
    #     x_ticks=[17.0, 34.0, 80.0],
    #     palette=palette,
    #     title="Pdk4 expression by age",
    #     rotation=45,
    #     show_scatter=False,
    #     dropna=True
    # )

    # check gene exists
    if gene not in adata.var_names:
        raise ValueError(f"{gene} was not found in adata.var_names")

    # check group column exists
    if group_col not in adata.obs.columns:
        raise ValueError(f"{group_col} was not found in adata.obs columns")

    # check layer exists
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{layer} was not found in adata.layers")

    # get gene expression matrix
    if layer is None:
        matrix = adata[:, gene].X
    else:
        matrix = adata[:, gene].layers[layer]

    # convert sparse matrix to dense
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # flatten expression values
    expression = np.asarray(matrix).flatten()

    # create plotting dataframe
    plot_df = pd.DataFrame({
        group_col: adata.obs[group_col].values,
        gene: expression
    })

    # remove missing values if needed
    if dropna:
        plot_df = plot_df.dropna()

    # set default title
    if title is None:
        title = f"{gene} expression by {group_col}"

    # create violin-box plot
    fig = plot_violin_box_combo(
        data=plot_df,
        x_var=group_col,
        y_var=gene,
        title=title,
        x_ticks=x_ticks,
        palette=palette,
        rotation=rotation,
        show_scatter=show_scatter
    )

    # return dataframe and figure
    return plot_df, fig


# plot_multiple_gene_expression
def plot_multiple_gene_expression(
    adata,
    genes,
    group_col,
    layer=None,
    x_ticks=None,
    palette=None,
    n_cols=3,
    figsize_per_plot=(4, 4),
    title=None,
    rotation=45,
    show_scatter=False,
    dropna=True
):
    """
    plot expression of multiple genes across groups
    """

    # plot_multiple_gene_expression
    # api:
    # plot_multiple_gene_expression(
    #     adata,
    #     genes=["Pdk4", "Cdkn1a", "Cdkn2a"],
    #     group_col="age",
    #     layer=None,
    #     x_ticks=[17.0, 34.0, 80.0],
    #     palette=palette,
    #     n_cols=3,
    #     figsize_per_plot=(4, 4),
    #     title="gene expression by age",
    #     rotation=45,
    #     show_scatter=False,
    #     dropna=True
    # )

    # check group column exists
    if group_col not in adata.obs.columns:
        raise ValueError(f"{group_col} was not found in adata.obs columns")

    # check layer exists
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"{layer} was not found in adata.layers")

    # keep only genes present in adata
    genes_present = [gene for gene in genes if gene in adata.var_names]
    genes_missing = [gene for gene in genes if gene not in adata.var_names]

    if len(genes_present) == 0:
        raise ValueError("none of the genes were found in adata.var_names")

    # get expression matrix
    if layer is None:
        matrix = adata[:, genes_present].X
    else:
        matrix = adata[:, genes_present].layers[layer]

    # convert sparse matrix to dense
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # create expression dataframe
    expr_df = pd.DataFrame(
        matrix,
        index=adata.obs_names,
        columns=genes_present
    )

    # add group column
    expr_df[group_col] = adata.obs[group_col].values

    # convert to long format
    plot_df = expr_df.melt(
        id_vars=group_col,
        value_vars=genes_present,
        var_name="gene",
        value_name="expression"
    )

    # remove missing values if needed
    if dropna:
        plot_df = plot_df.dropna()

    # set category order
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(
            plot_df[group_col].unique(),
            key=lambda x: float(x) if str(x).replace(".", "").isdigit() else x
        )

    # set subplot layout
    n_genes = len(genes_present)
    n_rows = int(np.ceil(n_genes / n_cols))

    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    # plot each gene
    for i, gene in enumerate(genes_present):
        ax = axes[i]

        gene_df = plot_df[plot_df["gene"] == gene].copy()

        # create violin plot
        sns.violinplot(
            data=gene_df,
            x=group_col,
            y="expression",
            order=categories,
            palette=palette,
            inner=None,
            linewidth=0,
            saturation=1.0,
            alpha=0.3,
            width=0.5,
            cut=0,
            ax=ax
        )

        # create box plot
        sns.boxplot(
            data=gene_df,
            x=group_col,
            y="expression",
            order=categories,
            width=0.35,
            linewidth=1.0,
            flierprops={"marker": " "},
            showmeans=False,
            boxprops={
                "facecolor": "none",
                "edgecolor": "black"
            },
            whiskerprops={"color": "black"},
            medianprops={"color": "black"},
            showcaps=True,
            ax=ax
        )

        # add scatter points if needed
        if show_scatter:
            sns.stripplot(
                data=gene_df,
                x=group_col,
                y="expression",
                order=categories,
                palette=palette,
                size=3,
                alpha=0.7,
                linewidth=0,
                jitter=0.2,
                ax=ax
            )

        # format subplot
        ax.set_title(gene)
        ax.set_xlabel("")
        ax.set_ylabel("expression")
        ax.set_xticklabels(categories, rotation=rotation, ha="right")
        ax.grid(False)

    # hide empty subplots
    for j in range(n_genes, len(axes)):
        axes[j].axis("off")

    # add main title if given
    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    # return dataframe, figure, and axes
    return plot_df, fig, axes


# plot_top_intersecting_genes_dotplot
def plot_top_intersecting_genes_dotplot(
    intersecting_genes_df,
    labels=None,
    gene_col="names",
    score_col_prefix="logfoldchanges",
    value_cols=None,
    cluster_rows=False,
    cluster_cols=True,
    figsize=(12, 4),
    cmap="Reds",
    min_dot_size=30,
    max_dot_size=500,
    sort_genes=False,
    title="Top Intersecting DEGs",
    show=True
):
    """
    plot top intersecting deg genes from an already merged dataframe
    """

    # plot_top_intersecting_genes_dotplot
    # api:
    # plot_top_intersecting_genes_dotplot(
    #     intersecting_genes_df=male_top10_genes,
    #     labels=["Fast IIX", "Fast IIB", "Skeleton MuSc", "FAPs"],
    #     gene_col="names",
    #     score_col_prefix="logfoldchanges",
    # )

    # find value columns automatically
    if value_cols is None:
        value_cols = [
            col for col in intersecting_genes_df.columns
            if col.startswith(f"{score_col_prefix}_")
        ]

    # check gene column
    if gene_col not in intersecting_genes_df.columns:
        raise ValueError(f"{gene_col} was not found in intersecting_genes_df")

    # check value columns
    if len(value_cols) == 0:
        raise ValueError("No value columns were found")

    missing_cols = [
        col for col in value_cols
        if col not in intersecting_genes_df.columns
    ]

    if len(missing_cols) > 0:
        raise ValueError(f"missing value columns: {missing_cols}")

    # make default labels
    if labels is None:
        labels = value_cols

    # check labels
    if len(labels) != len(value_cols):
        raise ValueError("labels and value_cols must have the same length")

    # keep needed columns
    plot_input_df = intersecting_genes_df[[gene_col] + value_cols].copy()

    # rename columns to labels
    rename_map = {
        old_col: label
        for old_col, label in zip(value_cols, labels)
    }

    plot_input_df = plot_input_df.rename(columns=rename_map)

    # optional gene sorting
    if sort_genes:
        plot_input_df = plot_input_df.sort_values(gene_col)

    # build matrix: rows = labels, cols = genes
    plot_df = plot_input_df.set_index(gene_col)[labels].T

    # optional clustering of columns
    if cluster_cols and plot_df.shape[1] > 1:
        col_linkage = linkage(pdist(plot_df.T), method="average")
        col_order = leaves_list(col_linkage)
        plot_df = plot_df.iloc[:, col_order]

    # optional clustering of rows
    if cluster_rows and plot_df.shape[0] > 1:
        row_linkage = linkage(pdist(plot_df), method="average")
        row_order = leaves_list(row_linkage)
        plot_df = plot_df.iloc[row_order, :]

    # get rows and columns
    rows = plot_df.index.tolist()
    cols = plot_df.columns.tolist()

    # get score ranges
    all_scores = plot_df.values.flatten()
    abs_scores = np.abs(all_scores)

    abs_min = abs_scores.min()
    abs_max = abs_scores.max()

    # scale dot sizes
    def scale_size(val):
        aval = abs(val)

        if abs_max == abs_min:
            return (min_dot_size + max_dot_size) / 2

        return (
            min_dot_size
            + (aval - abs_min)
            * (max_dot_size - min_dot_size)
            / (abs_max - abs_min)
        )

    # build plotting coordinates
    x_coords = []
    y_coords = []
    color_values = []
    size_values = []

    for i, row_name in enumerate(rows):
        for j, col_name in enumerate(cols):

            val = plot_df.loc[row_name, col_name]

            x_coords.append(j)
            y_coords.append(i)
            color_values.append(val)
            size_values.append(scale_size(val))

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot dots
    sc = ax.scatter(
        x_coords,
        y_coords,
        s=size_values,
        c=color_values,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.5
    )

    # format axes
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)

    ax.invert_yaxis()
    ax.set_xlabel("Genes")
    ax.set_ylabel("Cell Types")
    ax.set_title(title)

    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(False)

    # colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(score_col_prefix.replace("_", " ").title())

    # size legend
    legend_vals = np.linspace(abs_min, abs_max, 4)
    legend_sizes = [scale_size(v) for v in legend_vals]

    handles = [
        plt.scatter(
            [],
            [],
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=0.5
        )
        for size in legend_sizes
    ]

    legend_labels = [f"{v:.2f}" for v in legend_vals]

    ax.legend(
        handles,
        legend_labels,
        title=score_col_prefix.replace("_", " ").title(),
        scatterpoints=1,
        frameon=False,
        bbox_to_anchor=(1.18, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # show plot if needed
    if show:
        plt.show()

    # return figure and plot dataframe
    return fig, plot_df


import numpy as np
import decoupler as dc
from matplotlib.lines import Line2D


def plot_deg_volcano(
    deg_df,
    gene_col="names",
    logfc_col="logfoldchanges",
    pval_col="pvals_adj",
    logfc_threshold=0.5,
    pval_threshold=0.05,
    positive_label="WT Higher",
    negative_label="KO Higher",
    neutral_label="Not Significant",
    top_n_labels=10,
    genes_to_label=None,
    title="Volcano Plot",
    figsize=(7, 6),
    dot_size=45,
    alpha=0.8,
    max_stat=None,
    max_sign=None,
    color_pos="firebrick",
    color_neg="royalblue",
    color_null="lightgray"
):
    """
    plot volcano plot for deg results using decoupler
    """

    # plot_deg_volcano
    # api:
    # plot_deg_volcano(
    #     deg_df,
    #     positive_label="WT Higher",
    #     negative_label="KO Higher",
    #     title="Volcano Plot",
    # )

    # check required columns
    required_cols = [gene_col, logfc_col, pval_col]
    missing_cols = [col for col in required_cols if col not in deg_df.columns]

    if len(missing_cols) > 0:
        raise ValueError(f"missing required columns: {missing_cols}")

    # copy needed columns
    plot_df = deg_df[required_cols].copy()

    # remove missing values
    plot_df = plot_df.dropna(subset=required_cols)

    # check p-values
    if (plot_df[pval_col] < 0).any():
        raise ValueError(f"{pval_col} contains negative values")

    # avoid log10(0)
    plot_df[pval_col] = plot_df[pval_col].replace(0, np.nextafter(0, 1))

    # set gene names as index for decoupler
    plot_df = plot_df.set_index(gene_col)

    # remove duplicate gene names if present
    plot_df = plot_df[~plot_df.index.duplicated(keep="first")]

    # choose labels
    if genes_to_label is not None:
        top = genes_to_label
    else:
        top = top_n_labels

    # plot using decoupler
    fig = dc.pl.volcano(
        data=plot_df,
        x=logfc_col,
        y=pval_col,
        top=top,
        thr_stat=logfc_threshold,
        thr_sign=pval_threshold,
        max_stat=max_stat,
        max_sign=max_sign,
        color_pos=color_pos,
        color_neg=color_neg,
        color_null=color_null,
        kw_scatter={
            "s": dot_size,
            "alpha": alpha,
            "edgecolor": "none"
        },
        figsize=figsize,
        return_fig=True
    )

    # get axis
    ax = fig.axes[0]

    # add title
    ax.set_title(title)

    # add custom legend
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color_null,
            markersize=8,
            label=neutral_label
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color_pos,
            markersize=8,
            label=positive_label
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color_neg,
            markersize=8,
            label=negative_label
        )
    ]

    ax.legend(handles=legend_handles, frameon=False)

    # return figure and prepared dataframe
    return fig, plot_df


# --- migrated from 3_geneset_scores/utils/geneset_analysis.py ---

def visualize_deg_results(result_df, sex_label, output_dir):
    """
    Create comprehensive visualizations for DEG results
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Differential Gene Expression Analysis - {sex_label} Cells (KO vs WT)', fontsize=16, fontweight='bold')
    
    # 1. Volcano plot
    ax1 = axes[0, 0]
    
    # Color points based on significance and fold change
    colors = []
    for _, row in result_df.iterrows():
        if row['pvals_adj'] < 0.05 and row['log2FC'] > 0.5:
            colors.append('red')  # Upregulated
        elif row['pvals_adj'] < 0.05 and row['log2FC'] < -0.5:
            colors.append('blue')  # Downregulated
        else:
            colors.append('gray')  # Not significant
    
    scatter = ax1.scatter(result_df['log2FC'], result_df['-log10(pval)'], 
                         c=colors, alpha=0.6, s=20)
    
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=-0.5, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Log2 Fold Change (KO vs WT)')
    ax1.set_ylabel('-Log10(p-value)')
    ax1.set_title('Volcano Plot')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Upregulated'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Downregulated'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Not significant')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. MA plot - Use scores for x-axis since we only have pct_nz_group
    ax2 = axes[0, 1]
    
    # Use scores as the mean expression measure
    mean_expr = result_df['scores']
    xlabel_text = 'Gene Scores'
    
    ax2.scatter(mean_expr, result_df['log2FC'], c=colors, alpha=0.6, s=20)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5)
    ax2.set_xlabel(xlabel_text)
    ax2.set_ylabel('Log2 Fold Change (KO vs WT)')
    ax2.set_title('MA Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top upregulated genes
    ax3 = axes[1, 0]
    top_up = result_df[(result_df['significant']) & (result_df['log2FC'] > 0)].head(10)
    if len(top_up) > 0:
        ax3.barh(range(len(top_up)), top_up['log2FC'], color='red', alpha=0.7)
        ax3.set_yticks(range(len(top_up)))
        ax3.set_yticklabels(top_up['names'], fontsize=10)
        ax3.set_xlabel('Log2 Fold Change')
        ax3.set_title('Top 10 Upregulated Genes')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No significantly\nupregulated genes', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top 10 Upregulated Genes')
    
    # 4. Top downregulated genes
    ax4 = axes[1, 1]
    top_down = result_df[(result_df['significant']) & (result_df['log2FC'] < 0)].head(10)
    if len(top_down) > 0:
        ax4.barh(range(len(top_down)), top_down['log2FC'], color='blue', alpha=0.7)
        ax4.set_yticks(range(len(top_down)))
        ax4.set_yticklabels(top_down['names'], fontsize=10)
        ax4.set_xlabel('Log2 Fold Change')
        ax4.set_title('Top 10 Downregulated Genes')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No significantly\ndownregulated genes', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Top 10 Downregulated Genes')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/DEG_analysis_{sex_label.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved at: {filename}")
    
    return fig


def create_volcano_plot(degs, cell_type, top_n=10, xlim=(-5, 5), title='', save=True):
    """
    Create volcano plot for DEGs
    
    Parameters:
    -----------
    degs : DataFrame
        DEG results from scanpy
    cell_type : str
        Name of cell type
    top_n : int
        Number of top genes to label
    title : str
        Title of the plot
    save : bool
        Whether to save the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate -log10(p-value)
    degs['-log10_padj'] = -np.log10(degs['pvals_adj'].replace(0, 1e-300))  # Handle p=0
    
    # Define significance thresholds
    pval_thresh = 0.05
    fc_thresh = 0.5
    
    # Classify genes
    degs['significant'] = 'Not Significant'
    degs.loc[
        (degs['pvals_adj'] < pval_thresh) & (degs['logfoldchanges'] > fc_thresh),
        'significant'
    ] = 'Upregulated'
    degs.loc[
        (degs['pvals_adj'] < pval_thresh) & (degs['logfoldchanges'] < -fc_thresh),
        'significant'
    ] = 'Downregulated'
    
    # Color map
    colors = {'Upregulated': '#d62728', 'Downregulated': '#1f77b4', 'Not Significant': '#7f7f7f'}
    
    # Plot points
    for category, color in colors.items():
        mask = degs['significant'] == category
        ax.scatter(
            degs.loc[mask, 'logfoldchanges'],
            degs.loc[mask, '-log10_padj'],
            c=color,
            label=category,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(pval_thresh), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-fc_thresh, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Label top genes
    sig_degs = degs[degs['significant'] != 'Not Significant'].copy()
    if xlim:
        sig_degs = sig_degs[(sig_degs['logfoldchanges'] >= xlim[0]) & 
                            (sig_degs['logfoldchanges'] <= xlim[1])]
    if len(sig_degs) > 0:
        # Top upregulated
        top_up = sig_degs[sig_degs['logfoldchanges'] > 0].nlargest(top_n, 'logfoldchanges')
        # Top downregulated
        top_down = sig_degs[sig_degs['logfoldchanges'] < 0].nsmallest(top_n, 'logfoldchanges')
        # Combine
        top_genes = pd.concat([top_up, top_down])
        
        for idx, row in top_genes.iterrows():
            ax.annotate(
                row['names'],
                xy=(row['logfoldchanges'], row['-log10_padj']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    # Count genes
    n_up = (degs['significant'] == 'Upregulated').sum()
    n_down = (degs['significant'] == 'Downregulated').sum()
    n_total = len(degs)
    
    # Labels and title
    ax.set_xlabel('Log2 Fold Change (KO / WT)', fontsize=12, fontweight='bold')
    ax.set_ylabel('-Log10(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax.set_title(f'{cell_type} + {title}\nKO vs WT', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis limits
    if xlim:
        ax.set_xlim(xlim)

    # Add counts to legend
    legend_labels = [
        f'Upregulated ({n_up})',
        f'Downregulated ({n_down})',
        f'Not Significant ({n_total - n_up - n_down})'
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    if save:
        filename = f"volcano_KO_vs_WT_{cell_type.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.show()
    plt.close()
    
    return fig, ax


def plot_pathway_dotplot(
    df_cell_level,
    score_cols,  # List of pathway score column names (y-axis)
    sample_col='sample',  # Column name for sample IDs (x-axis)
    annotation_col='Annotation',
    target_annotation=None,
    sample_order=None,  # Optional: list of samples in desired order
    figsize=(10, 6),
    min_dot_size=20,
    max_dot_size=500,
    dot_size_scale_factor=1.0,
    cmap_name="coolwarm",
    value_legend_title="Mean Score",
    size_legend_title="# Cells",
    ylabel="Pathways",
    xlabel="Samples"
):
    """
    Plots a dot plot where dot color is mean pathway score and dot size is number of cells.
    
    Parameters:
        df_cell_level (pd.DataFrame): Cell-level data with scores, sample, annotation.
        score_cols (list): List of score column names (pathways) - will be y-axis.
        sample_col (str): Column name for sample IDs - will be x-axis.
        annotation_col (str): Column name for annotations.
        target_annotation (str or None): If provided, subset to this annotation.
        sample_order (list or None): Order of samples for x-axis. If None, uses sorted order.
        figsize (tuple): Figure size.
        min_dot_size (int): Minimum size for dots.
        max_dot_size (int): Maximum size for dots.
        dot_size_scale_factor (float): Multiplier for raw cell counts before scaling to dot size.
        cmap_name (str): Colormap for the scores.
        value_legend_title (str): Title for the colorbar.
        size_legend_title (str): Title for the size legend.
        ylabel (str): Label for y-axis (pathways).
        xlabel (str): Label for x-axis (samples).
    """
    plot_df = df_cell_level.copy()

    # 1. Filter by annotation if specified
    if target_annotation is not None:
        if annotation_col not in plot_df.columns:
            print(f"Warning: Annotation column '{annotation_col}' not found. Cannot filter by '{target_annotation}'.")
            return
        plot_df = plot_df[plot_df[annotation_col] == target_annotation]
        if plot_df.empty:
            print(f"No cells found for annotation '{target_annotation}'.")
            return

    # Check required columns
    required_cols = [sample_col] + score_cols
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Required column '{col}' not found. Aborting.")
            return

    plot_df = plot_df.dropna(subset=[sample_col], how='any')
    if plot_df.empty:
        print("No data to plot after initial NaN filtering.")
        return

    # 2. Aggregate: mean scores and cell counts per sample
    grouped = plot_df.groupby(sample_col)
    mean_scores_df = grouped[score_cols].mean()
    cell_counts_series = grouped.size()

    # 3. Determine sample order
    if sample_order is None:
        ordered_samples = sorted(mean_scores_df.index.tolist())
    else:
        # Use provided order, but only include samples that exist in data
        ordered_samples = [s for s in sample_order if s in mean_scores_df.index]
        if not ordered_samples:
            print("None of the specified samples found in data.")
            return

    # 4. Prepare data for plotting (long format)
    plot_data_list = []
    for sample_id in ordered_samples:
        for pathway in score_cols:
            mean_score = mean_scores_df.loc[sample_id, pathway] if sample_id in mean_scores_df.index else np.nan
            cell_count = cell_counts_series.loc[sample_id] if sample_id in cell_counts_series.index else 0
            plot_data_list.append({
                'sample': sample_id,
                'pathway': pathway,
                'mean_score': mean_score,
                'cell_count': cell_count
            })
    
    plot_data_df = pd.DataFrame(plot_data_list)
    plot_data_df = plot_data_df.dropna(subset=['mean_score'])

    if plot_data_df.empty:
        print("No data to plot after aggregation.")
        return

    # 5. Scale cell counts for dot sizes
    min_count = plot_data_df['cell_count'].min()
    max_count = plot_data_df['cell_count'].max()
    
    if max_count == min_count:
        plot_data_df['dot_size'] = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
    else:
        scaled_counts = plot_data_df['cell_count'] * dot_size_scale_factor
        min_s_count = scaled_counts.min()
        max_s_count = scaled_counts.max()
        
        if max_s_count == min_s_count:
            plot_data_df['dot_size'] = min_dot_size if max_s_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            plot_data_df['dot_size'] = min_dot_size + \
                (scaled_counts - min_s_count) / (max_s_count - min_s_count) * (max_dot_size - min_dot_size)

    # 6. Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Create coordinate mappings
    pathway_y_coords = {name: i for i, name in enumerate(score_cols)}
    sample_x_coords = {name: i for i, name in enumerate(ordered_samples)}

    scatter = ax.scatter(
        x=plot_data_df['sample'].map(sample_x_coords),
        y=plot_data_df['pathway'].map(pathway_y_coords),
        s=plot_data_df['dot_size'],
        c=plot_data_df['mean_score'],
        cmap=cmap_name,
        edgecolors='gray',
        linewidths=0.5
    )

    # X-axis (Samples)
    ax.set_xticks(list(sample_x_coords.values()))
    ax.set_xticklabels(ordered_samples, rotation=45, ha="right")
    ax.set_xlabel(xlabel)

    # Y-axis (Pathways)
    ax.set_yticks(list(pathway_y_coords.values()))
    ax.set_yticklabels(score_cols)
    ax.set_ylabel(ylabel)

    # Colorbar for Mean Score
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.15)
    cbar.set_label(value_legend_title)

    # Legend for Dot Size (# Cells)
    if max_count > 0:
        legend_counts_raw = np.linspace(min_count, max_count, num=4, dtype=int)
        if min_count == 0 and 0 not in legend_counts_raw and len(legend_counts_raw) > 1:
            legend_counts_raw[0] = 0
        legend_counts_raw = np.unique(legend_counts_raw)
    else:
        legend_counts_raw = np.array([min_count]) if min_count > 0 else np.array([0])

    legend_dots = []
    for count_val in legend_counts_raw:
        if max_count == min_count:
            size_val = min_dot_size if max_count == 0 else (min_dot_size + max_dot_size) / 2
        else:
            scaled_c = count_val * dot_size_scale_factor
            size_val = min_dot_size + \
                (scaled_c - (min_count * dot_size_scale_factor)) / \
                ((max_count * dot_size_scale_factor) - (min_count * dot_size_scale_factor)) * \
                (max_dot_size - min_dot_size)
        size_val = max(min_dot_size, min(max_dot_size, size_val))
        legend_dots.append(plt.scatter([], [], s=size_val, c='gray', label=f"{int(count_val)}"))

    size_leg = ax.legend(
        handles=legend_dots,
        title=size_legend_title,
        bbox_to_anchor=(1.18, 0.4),
        loc='center left',
        labelspacing=1.5,
        borderpad=1,
        frameon=True,
        handletextpad=1.5,
        scatterpoints=1
    )

    # Layout adjustments
    fig_title = f'Pathway Activity Dot Plot (Annotation: {target_annotation})' if target_annotation else 'Pathway Activity Dot Plot'
    plt.suptitle(fig_title, fontsize=16, y=1.02)
    plt.subplots_adjust(bottom=0.15, right=0.8)
    plt.grid(True, linestyle='--', alpha=0.3, axis='both')
    ax.tick_params(axis='both', which='major', pad=7)

    plt.tight_layout()
    plt.show()

# --- migrated from 3_geneset_scores/utils/driver_gene.py ---

def plot_heatmap(
    male_ko_df,
    male_wt_df,
    female_ko_df,
    female_wt_df,
    title,
    show_top_genes=True,
    top_n=20,
    figsize=(6,6)
):

    # Combine correlations
    heatmap_df = pd.DataFrame({
        "Male KO": male_ko_df.set_index("gene")["spearman_corr"],
        "Male WT": male_wt_df.set_index("gene")["spearman_corr"],
        "Female KO": female_ko_df.set_index("gene")["spearman_corr"],
        "Female WT": female_wt_df.set_index("gene")["spearman_corr"]
    })

    if show_top_genes:

        heatmap_df["max_corr"] = heatmap_df.abs().max(axis=1)

        heatmap_df = (
            heatmap_df
            .sort_values("max_corr", ascending=False)
            .head(top_n)
            .drop(columns="max_corr")
        )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Gene")

    plt.tight_layout()
    plt.close()

    return fig


def display_plots_side_by_side(figures, width=400, save_path=None):
    """
    Display multiple matplotlib figures side by side and optionally
    save them as one combined image.

    Parameters
    ----------
    figures : list
        List of matplotlib figure objects
    width : int
        Width of each image in pixels (for display only)
    save_path : str or Path
        File path to save the combined image
    """

    html = '<div style="display:flex; gap:10px;">'
    images = []

    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)

        # convert to PIL image
        img = Image.open(buf)
        images.append(img)

        # for HTML display
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        html += f'<img src="data:image/png;base64,{img_base64}" width="{width}"/>'

    html += "</div>"
    display(HTML(html))

    # save combined image
    if save_path is not None:
        widths, heights = zip(*(img.size for img in images))

        total_width = sum(widths)
        max_height = max(heights)

        combined = Image.new("RGB", (total_width, max_height), (255,255,255))

        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        combined.save(save_path)


# _significance_label
def _significance_label(pval):
    """convert a p-value to a star/NS annotation string"""
    if pval is None:
        return "NS"
    try:
        p = float(pval)
    except (TypeError, ValueError):
        return "NS"
    if not np.isfinite(p):
        return "NS"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"


# plot_transcriptional_noise
def plot_transcriptional_noise(
    noise_all_df,
    celltype_col="celltype",
    condition_col="condition",
    noise_col="noise",
    celltype_order=None,
    group_order=("WT", "KO"),
    group_labels=None,
    colors=("#6F4E37", "#C19A6B"),
    figsize=(14, 5),
    title="Control vs Skm-KO Transcriptional Noise",
    bar_width=0.15,
    bar_step=0.18,
    face_alpha=0.85,
    whisker_alpha=0.4,
    show_significance=True,
    pvalues=None,
    deltas=None,
    sig_fontsize=10,
    test="mannwhitneyu"
):
    """
    plot transcriptional noise boxplot
    """

    # plot_transcriptional_noise
    # api:
    # fig, ax = plot_transcriptional_noise(
    #     noise_all_df=noise_all_df,
    #     celltype_col="celltype",
    #     condition_col="condition",
    #     noise_col="noise",
    # )

    if celltype_order is None:
        celltype_order = noise_all_df[celltype_col].dropna().unique().tolist()

    # default display labels: rename WT -> Control, KO -> Skm-KO (fallback to raw value)
    if group_labels is None:
        group_labels = {"WT": "Control", "KO": "Skm-KO"}
    display_labels = [group_labels.get(g, g) for g in group_order]

    plot_data = [
        noise_all_df[
            (noise_all_df[celltype_col] == celltype) &
            (noise_all_df[condition_col] == group)
        ][noise_col].dropna().values
        for celltype in celltype_order
        for group in group_order
    ]

    # uniform spacing: every adjacent bar is bar_step apart (no extra gap between celltype pairs)
    n_groups = len(group_order)
    positions = []

    pos = 1.0
    for celltype in celltype_order:
        positions.extend([pos + i * bar_step for i in range(n_groups)])
        pos += n_groups * bar_step

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=bar_width,
        patch_artist=True,
        showfliers=False
    )

    # box face translucent, stroke opaque
    for i, box in enumerate(bp["boxes"]):
        color = colors[i % len(colors)]
        box.set_facecolor(to_rgba(color, alpha=face_alpha))
        box.set_edgecolor("black")
        box.set_linewidth(1.2)

    # translucent error lines (whiskers + caps)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_alpha(whisker_alpha)
        whisker.set_linewidth(1.0)

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_alpha(whisker_alpha)
        cap.set_linewidth(1.0)

    # opaque median line
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    # set x-axis labels at the center of each celltype's group
    tick_positions = [
        np.mean(positions[i:i + n_groups])
        for i in range(0, len(positions), n_groups)
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(celltype_order, rotation=90)

    ax.set_ylabel("Transcriptional noise")
    ax.set_xlabel("Cell type")
    ax.set_title(title)

    # legend uses display labels (e.g. Control / Skm-KO) plus a glossary entry
    # explaining the triangle glyph used in the bracket annotations
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i % len(colors)],
            lw=8,
            label=display_labels[i]
        )
        for i in range(n_groups)
    ]

    if show_significance and n_groups == 2:
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                linestyle="None",
                marker="None",
                label="Cliff's δ = △"
            )
        )

    ax.legend(
        handles=legend_handles,
        title=condition_col,
        frameon=False
    )

    # significance brackets (only for 2-group comparisons)
    if show_significance and n_groups == 2:

        # ensure dicts exist so we can fill in missing entries below
        if pvalues is None:
            pvalues = {}
        if deltas is None:
            deltas = {}

        # fill in any missing pvalues or deltas per celltype
        for ct in celltype_order:
            need_pval = ct not in pvalues
            need_delta = ct not in deltas

            if not (need_pval or need_delta):
                continue

            data1 = noise_all_df[
                (noise_all_df[celltype_col] == ct) &
                (noise_all_df[condition_col] == group_order[0])
            ][noise_col].dropna().values

            data2 = noise_all_df[
                (noise_all_df[celltype_col] == ct) &
                (noise_all_df[condition_col] == group_order[1])
            ][noise_col].dropna().values

            if len(data1) == 0 or len(data2) == 0:
                if need_pval:
                    pvalues[ct] = np.nan
                if need_delta:
                    deltas[ct] = np.nan
                continue

            if need_pval:
                if test == "mannwhitneyu":
                    pvalues[ct] = mannwhitneyu(
                        data1, data2, alternative="two-sided"
                    ).pvalue
                elif test == "ttest":
                    pvalues[ct] = ttest_ind(
                        data1, data2, equal_var=True
                    ).pvalue
                elif test == "welch":
                    pvalues[ct] = ttest_ind(
                        data1, data2, equal_var=False
                    ).pvalue
                else:
                    raise ValueError(
                        f"unknown test {test!r}; expected "
                        "'mannwhitneyu', 'ttest', or 'welch'"
                    )

            if need_delta:
                # group2 vs group1 -> positive delta means group2 has larger values
                deltas[ct] = cliffs_delta(data2, data1)

        # size the bracket padding relative to y-range
        all_whisker_y = [v for w in bp["whiskers"] for v in w.get_ydata()]
        y_max = max(all_whisker_y) if all_whisker_y else 1.0
        y_min = min(all_whisker_y) if all_whisker_y else 0.0
        y_range = (y_max - y_min) if y_max > y_min else 1.0
        padding = y_range * 0.04
        tick_height = padding * 0.4

        for i, ct in enumerate(celltype_order):
            pair_whisker_start = i * n_groups * 2
            pair_whiskers = bp["whiskers"][pair_whisker_start:pair_whisker_start + n_groups * 2]

            if not pair_whiskers:
                continue

            pair_upper = max(max(w.get_ydata()) for w in pair_whiskers)
            bracket_y = pair_upper + padding

            x_left = positions[i * n_groups]
            x_right = positions[i * n_groups + (n_groups - 1)]

            ax.plot(
                [x_left, x_left, x_right, x_right],
                [bracket_y - tick_height, bracket_y, bracket_y, bracket_y - tick_height],
                color="black",
                linewidth=1.0
            )

            # always show cliff's delta (rendered as a triangle glyph; see legend)
            # (positive delta -> group2 (e.g. KO) higher; negative -> group1 (WT) higher)
            d = deltas.get(ct, np.nan)
            if d is not None and np.isfinite(d):
                label = f"△={d:.2f}"
            else:
                label = "NA"

            ax.text(
                (x_left + x_right) / 2,
                bracket_y + tick_height * 0.4,
                label,
                ha="center",
                va="bottom",
                fontsize=sig_fontsize
            )

        # extend y-axis so brackets/labels are not clipped
        cur_ylo, cur_yhi = ax.get_ylim()
        ax.set_ylim(cur_ylo, cur_yhi + padding * 3)

    plt.tight_layout()

    return fig, ax


# plot_sankey
def plot_sankey(
    data,
    left_col,
    right_col,
    left_order=None,
    right_order=None,
    left_colors=None,
    right_colors=None,
    title=None,
    figsize=(10, 7),
    bar_width=0.03,
    x_left=0.28,
    x_right=0.58,
    alpha=0.4,
    left_label_fontsize=10,
    right_label_fontsize=9,
    left_label_fontweight="bold",
    flow_color="left",
    default_color="#aaaaaa",
    show_counts=False,
    count_fontsize=8,
    left_gap=0.004,
    right_gap=0.004,
    left_group_gap=None,
    left_group_indices=None,
    right_group_gap=None,
    right_group_indices=None,
    flow_pad=0.0,
):
    """
    plot a two-column sankey diagram from a long-format dataframe

    Parameters
    ----------
    data : pd.DataFrame
        long-format dataframe with at least `left_col` and `right_col`
        (e.g. adata.obs). flow widths are computed from row counts.
    left_col, right_col : str
        column names mapped to the left and right stacked bars.
    left_order, right_order : list or None
        ordering of categories on each side. defaults to category order
        for pandas Categorical, otherwise unique() order.
    left_colors, right_colors : dict or None
        {category: color} for each side. missing categories fall back to
        `default_color`.
    title : str or None
        figure title.
    figsize : tuple
        figure dimensions.
    bar_width : float
        width of the stacked bars in axis units.
    x_left, x_right : float
        x-positions of the left and right bars in axis units (0-1).
    alpha : float
        flow ribbon transparency.
    flow_color : {"left", "right"}
        which side's color is used for the ribbons.
    default_color : str
        fallback color for categories not present in the color dict.
    show_counts : bool
        if True, draw the per-segment count between each side's bar and its
        category label so flow widths can be read off directly.
    count_fontsize : int
        font size for the per-segment count labels when ``show_counts`` is True.
    left_gap, right_gap : float
        baseline vertical gap (in axes units) between adjacent stacked bars on
        each side. Bar heights are shrunk to keep the total stack within the
        plotting band.
    left_group_gap, right_group_gap : float or None
        larger vertical gap used at sub-group boundaries. Falls back to
        ``left_gap`` / ``right_gap`` when None.
    left_group_indices, right_group_indices : iterable[int] or None
        indices ``i`` where the gap *after* category ``i`` should use the
        larger ``*_group_gap`` (e.g. ``[1]`` puts the big gap between the 2nd
        and 3rd left-bar categories). All other gaps use ``*_gap``.
    flow_pad : float
        horizontal padding (in axes units) between each bar's inner edge and
        where the ribbon curves start/end — adds visible whitespace between
        the bars and the flow.
    """

    # plot_sankey
    # api:
    # fig, ax = plot_sankey(
    #     data=adata.obs,
    #     left_col="sex",
    #     right_col="re_annot",
    #     left_order=["F", "M"],
    #     left_colors={"F": "#e8a0a0", "M": "#5dbdb2"},
    #     right_colors=cell_type_colors,
    #     title="Cell Proportion",
    # )

    # check columns exist
    for col in (left_col, right_col):
        if col not in data.columns:
            raise ValueError(f"{col} was not found in data columns")

    # check flow_color
    if flow_color not in ("left", "right"):
        raise ValueError("flow_color must be 'left' or 'right'")

    # build counts
    df = (
        data.groupby([left_col, right_col], observed=True)
        .size()
        .reset_index(name="count")
    )

    # default category orders
    def _default_order(col):
        series = data[col]
        if hasattr(series, "cat"):
            return series.cat.categories.tolist()
        return list(pd.unique(series))

    if left_order is None:
        left_order = _default_order(left_col)
    if right_order is None:
        right_order = _default_order(right_col)

    # default colors
    if left_colors is None:
        tab = plt.cm.tab20.colors
        left_colors = {g: tab[i % len(tab)] for i, g in enumerate(left_order)}
    if right_colors is None:
        tab = plt.cm.tab20.colors
        right_colors = {g: tab[i % len(tab)] for i, g in enumerate(right_order)}

    # figure / axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # per-side gap arrays of length N-1, with the larger group_gap inserted
    # at the boundaries listed in *_group_indices.
    def _build_gaps(order, gap, group_gap, group_indices):
        n = len(order)
        if n <= 1:
            return []
        big = gap if group_gap is None else group_gap
        idx_set = set(group_indices) if group_indices is not None else set()
        return [big if i in idx_set else gap for i in range(n - 1)]

    left_gaps = _build_gaps(left_order, left_gap, left_group_gap, left_group_indices)
    right_gaps = _build_gaps(right_order, right_gap, right_group_gap, right_group_indices)

    # stacked-bar positions; bars are shrunk to fit (0.9 - sum_of_gaps) so
    # larger gaps don't push the stack past the axis top.
    def _calc_positions(groups, counts_dict, gaps):
        total = sum(counts_dict.get(g, 0) for g in groups)
        y = 0.05
        if total == 0:
            return {g: (y, y) for g in groups}
        available = max(0.9 - sum(gaps), 0.1)
        positions = {}
        for i, g in enumerate(groups):
            h = (counts_dict.get(g, 0) / total) * available
            positions[g] = (y, y + h)
            y += h
            if i < len(groups) - 1:
                y += gaps[i]
        return positions

    left_totals = df.groupby(left_col, observed=True)["count"].sum().to_dict()
    right_totals = df.groupby(right_col, observed=True)["count"].sum().to_dict()

    left_pos = _calc_positions(left_order, left_totals, left_gaps)
    right_pos = _calc_positions(right_order, right_totals, right_gaps)

    # draw flows
    left_fill = {g: left_pos[g][0] for g in left_order}
    right_fill = {g: right_pos[g][0] for g in right_order}

    # iterate in stacked order so ribbons stack cleanly
    df_sorted = df.copy()
    df_sorted[left_col] = pd.Categorical(
        df_sorted[left_col], categories=left_order, ordered=True
    )
    df_sorted[right_col] = pd.Categorical(
        df_sorted[right_col], categories=right_order, ordered=True
    )
    df_sorted = df_sorted.sort_values([left_col, right_col])

    for _, row in df_sorted.iterrows():
        lg, rg, cnt = row[left_col], row[right_col], row["count"]

        l_total = left_totals.get(lg, 0)
        r_total = right_totals.get(rg, 0)
        if l_total == 0 or r_total == 0 or cnt == 0:
            continue

        l_span = left_pos[lg][1] - left_pos[lg][0]
        r_span = right_pos[rg][1] - right_pos[rg][0]

        lh = (cnt / l_total) * l_span
        rh = (cnt / r_total) * r_span

        ly0 = left_fill[lg]
        ry0 = right_fill[rg]
        ly1 = ly0 + lh
        ry1 = ry0 + rh

        left_fill[lg] += lh
        right_fill[rg] += rh

        flow_start = x_left + bar_width + flow_pad
        flow_end = x_right - flow_pad
        mid_x = (flow_start + flow_end) / 2
        verts = [
            (flow_start, ly0),
            (mid_x, ly0),
            (mid_x, ry0),
            (flow_end, ry0),
            (flow_end, ry1),
            (mid_x, ry1),
            (mid_x, ly1),
            (flow_start, ly1),
            (flow_start, ly0),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        path = Path(verts, codes)

        if flow_color == "left":
            face = left_colors.get(lg, default_color)
        else:
            face = right_colors.get(rg, default_color)

        patch = patches.PathPatch(
            path,
            facecolor=face,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    # draw bars
    # When show_counts is on, count sits flush against the bar and the
    # category label is pushed outward to leave room for it.
    left_count_pad = 0.005
    left_label_pad = 0.045 if show_counts else 0.01
    right_count_pad = 0.005
    right_label_pad = 0.045 if show_counts else 0.01

    for g in left_order:
        y0, y1 = left_pos[g]
        ax.add_patch(plt.Rectangle(
            (x_left, y0), bar_width, y1 - y0,
            color=left_colors.get(g, default_color),
            zorder=2, linewidth=0,
        ))
        ax.text(
            x_left - left_label_pad, (y0 + y1) / 2, str(g),
            ha="right", va="center",
            fontsize=left_label_fontsize,
            fontweight=left_label_fontweight,
        )
        if show_counts:
            ax.text(
                x_left - left_count_pad, (y0 + y1) / 2,
                f"{int(left_totals.get(g, 0))}",
                ha="right", va="center",
                fontsize=count_fontsize,
                color="#333333",
            )

    for g in right_order:
        y0, y1 = right_pos[g]
        ax.add_patch(plt.Rectangle(
            (x_right, y0), bar_width, y1 - y0,
            color=right_colors.get(g, default_color),
            zorder=2, linewidth=0,
        ))
        ax.text(
            x_right + bar_width + right_label_pad, (y0 + y1) / 2, str(g),
            ha="left", va="center",
            fontsize=right_label_fontsize,
        )
        if show_counts:
            ax.text(
                x_right + bar_width + right_count_pad, (y0 + y1) / 2,
                f"{int(right_totals.get(g, 0))}",
                ha="left", va="center",
                fontsize=count_fontsize,
                color="#333333",
            )

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    return fig, ax


# plot_chord
def plot_chord(
    data,
    left_col,
    right_col,
    left_order=None,
    right_order=None,
    left_colors=None,
    right_colors=None,
    count_col=None,
    title=None,
    figsize=(10, 10),
    inner_radius=1.0,
    arc_width=0.08,
    segment_gap_deg=2.0,
    side_gap_deg=14.0,
    alpha=0.55,
    flow_color="left",
    left_axis_label=None,
    right_axis_label=None,
    axis_label_fontsize=13,
    axis_label_fontweight="bold",
    segment_label_fontsize=10,
    segment_label_pad=0.18,
    show_ticks=True,
    tick_step="auto",
    target_n_ticks=5,
    min_tick_label_gap_deg=4.0,
    hide_zero_tick_label=False,
    tick_fontsize=7,
    n_arc_pts=24,
    default_color="#aaaaaa",
):
    """
    plot a chord / circular sankey diagram from a long-format dataframe.

    the left and right category sets sit on opposing semi-circular arcs;
    ribbons through the center connect each left segment to each right
    segment with width proportional to row counts (or ``count_col``).
    arc segments carry tick marks of cumulative counts when ``show_ticks``
    is True.

    Parameters
    ----------
    data : pd.DataFrame
        long-format dataframe with at least ``left_col`` and ``right_col``.
        if ``count_col`` is None, flow widths come from row counts; otherwise
        from the ``count_col`` column.
    left_col, right_col : str
        column names mapped to the left and right arcs.
    left_order, right_order : list or None
        ordering of categories on each side. defaults to ``cat.categories``
        for pandas Categorical, otherwise ``pd.unique()`` order.
    left_colors, right_colors : dict or None
        ``{category: color}`` per side. missing categories fall back to
        ``default_color``.
    count_col : str or None
        precomputed count column. None to count rows.
    title : str or None
        figure title.
    figsize : tuple
        figure dimensions.
    inner_radius : float
        inner radius of the colored ring.
    arc_width : float
        radial width of the colored ring.
    segment_gap_deg : float
        gap between segments within one arc, in degrees.
    side_gap_deg : float
        gap between the two arcs (split symmetrically at top and bottom),
        in degrees.
    alpha : float
        ribbon transparency.
    flow_color : {"left", "right"}
        which side's color is used for the ribbons.
    left_axis_label, right_axis_label : str or None
        bold labels placed at the midpoint of each arc (e.g. ``"DEGs"``,
        ``"Genes"``).
    segment_label_pad : float
        radial padding (in axis units) between the outer edge of the ring
        and the segment text labels. increase this to avoid clashing with
        cumulative-count tick labels on busy plots.
    show_ticks : bool
        whether to render cumulative-count tick marks on each segment.
    tick_step : int, str, tuple, or dict
        spacing of tick marks along each segment, in count units. accepts:

        - ``"auto"`` (default): pick a 1/2/5 * 10^k step per side so each
          side's largest segment has roughly ``target_n_ticks`` major ticks.
        - ``int``: use the same step on both sides.
        - ``(left_step, right_step)`` tuple / 2-list: per-side step. each
          entry may itself be ``"auto"`` or an int.
        - ``{"left": step, "right": step}`` dict: per-side override.
    target_n_ticks : int
        target number of ticks on the largest segment of a side when
        ``tick_step="auto"``. used to pick a "nice" 1/2/5 * 10^k step.
    min_tick_label_gap_deg : float
        minimum angular distance (in degrees) between two tick *labels* on
        the same side. tick marks themselves are always drawn; labels too
        close to the previous one are suppressed so the numbers stay
        readable on small segments.
    hide_zero_tick_label : bool
        if True, suppress the ``0`` label at the start of each segment
        (the tick mark is still drawn). useful when many small segments
        crowd the ring with zeros.
    n_arc_pts : int
        number of line-segments used to approximate the inner arc per
        ribbon endpoint.
    default_color : str
        fallback color for categories not present in the color dict.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """

    # plot_chord
    # api:
    # fig, ax = plot_chord(
    #     data=enrich_df,
    #     left_col="fiber_type",        # e.g. "IIB" / "IIX"
    #     right_col="category",         # e.g. "Metabolism", "Cellular Processes", ...
    #     count_col="n_pathways",
    #     left_axis_label="Fibers",
    #     right_axis_label="Categories",
    #     left_colors={"IIB": "#7e57c2", "IIX": "#ef9a9a"},
    #     right_colors=category_palette,
    #     tick_step="auto",             # or e.g. (50, 50) / {"left": 50, "right": 50}
    #     title="Up-enriched pathways → categories",
    # )

    def _nice_tick_step(max_val, target_n=5):
        """pick a 1/2/5 * 10^k step so max_val / step is roughly target_n."""
        if max_val <= 0:
            return 1
        raw = max_val / max(target_n, 1)
        mag = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            step = m * mag
            if step >= raw:
                return int(step) if step >= 1 else float(step)
        return int(10 * mag)

    def _resolve_tick_step(user, side, max_total):
        """resolve tick_step input into a numeric step for the given side."""
        if isinstance(user, dict):
            v = user.get(side, "auto")
        elif isinstance(user, (tuple, list)) and len(user) == 2:
            v = user[0 if side == "left" else 1]
        else:
            v = user
        if v is None or v == "auto":
            return _nice_tick_step(max_total, target_n=target_n_ticks)
        return v

    if flow_color not in ("left", "right"):
        raise ValueError("flow_color must be 'left' or 'right'")

    for col in (left_col, right_col):
        if col not in data.columns:
            raise ValueError(f"{col} was not found in data columns")

    # build counts
    if count_col is None:
        df = (
            data.groupby([left_col, right_col], observed=True)
            .size()
            .reset_index(name="count")
        )
        cnt_col = "count"
    else:
        if count_col not in data.columns:
            raise ValueError(f"{count_col} was not found in data columns")
        df = data[[left_col, right_col, count_col]].copy()
        cnt_col = count_col

    # default orders
    def _default_order(col):
        series = data[col]
        if hasattr(series, "cat"):
            return series.cat.categories.tolist()
        return list(pd.unique(series))

    if left_order is None:
        left_order = _default_order(left_col)
    if right_order is None:
        right_order = _default_order(right_col)

    # default colors
    if left_colors is None:
        tab = plt.cm.tab20.colors
        left_colors = {g: tab[i % len(tab)] for i, g in enumerate(left_order)}
    if right_colors is None:
        tab = plt.cm.tab20.colors
        right_colors = {
            g: tab[(i + 10) % len(tab)] for i, g in enumerate(right_order)
        }

    # totals per category
    left_totals = (
        df.groupby(left_col, observed=True)[cnt_col].sum().to_dict()
    )
    right_totals = (
        df.groupby(right_col, observed=True)[cnt_col].sum().to_dict()
    )

    # arc angular ranges (radians) — right half on the +x side, left half on -x
    side_gap = np.deg2rad(side_gap_deg)
    seg_gap = np.deg2rad(segment_gap_deg)

    right_a0 = -np.pi / 2 + side_gap / 2
    right_a1 = np.pi / 2 - side_gap / 2
    right_span = right_a1 - right_a0

    left_a0 = np.pi / 2 + side_gap / 2
    left_a1 = 3 * np.pi / 2 - side_gap / 2
    left_span = left_a1 - left_a0

    def _segment_ranges(order, totals, a_start, a_span, gap):
        n = len(order)
        if n == 0:
            return {}
        available = a_span - max(n - 1, 0) * gap
        total = sum(totals.get(g, 0) for g in order)
        ranges = {}
        cursor = a_start
        for g in order:
            cnt = totals.get(g, 0)
            ang = (cnt / total) * available if total > 0 else 0.0
            ranges[g] = (cursor, cursor + ang)
            cursor += ang + gap
        return ranges

    left_ranges = _segment_ranges(
        left_order, left_totals, left_a0, left_span, seg_gap
    )
    right_ranges = _segment_ranges(
        right_order, right_totals, right_a0, right_span, seg_gap
    )

    # fill cursors track how much of each segment has been used by ribbons
    left_fill = {g: left_ranges[g][0] for g in left_order}
    right_fill = {g: right_ranges[g][0] for g in right_order}

    # figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    # add extra room for tick labels + segment labels + axis labels stacked
    # radially outside the colored ring
    lim = inner_radius + arc_width + max(segment_label_pad + 0.30, 0.55)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    def _arc_pts(a, b, r, n):
        angles = np.linspace(a, b, n)
        return [(r * np.cos(t), r * np.sin(t)) for t in angles]

    # stable ribbon stacking
    df_sorted = df.copy()
    df_sorted[left_col] = pd.Categorical(
        df_sorted[left_col], categories=left_order, ordered=True
    )
    df_sorted[right_col] = pd.Categorical(
        df_sorted[right_col], categories=right_order, ordered=True
    )
    df_sorted = df_sorted.sort_values([left_col, right_col])

    r_in = inner_radius

    # draw ribbons
    for _, row in df_sorted.iterrows():
        lg, rg, cnt = row[left_col], row[right_col], row[cnt_col]
        lt = left_totals.get(lg, 0)
        rt = right_totals.get(rg, 0)
        if lt == 0 or rt == 0 or cnt is None or cnt == 0:
            continue

        l0, l1 = left_ranges[lg]
        r0, r1 = right_ranges[rg]

        la = (cnt / lt) * (l1 - l0)
        ra = (cnt / rt) * (r1 - r0)

        la0 = left_fill[lg]
        la1 = la0 + la
        ra0 = right_fill[rg]
        ra1 = ra0 + ra
        left_fill[lg] += la
        right_fill[rg] += ra

        # build a closed ribbon path:
        #   1) line-approximated arc from la0 -> la1 on the inner radius
        #   2) cubic Bezier through (0, 0) from la1 -> ra1
        #   3) line-approximated arc from ra1 -> ra0
        #   4) cubic Bezier through (0, 0) from ra0 -> la0
        left_arc = _arc_pts(la0, la1, r_in, n_arc_pts)
        right_arc = _arc_pts(ra1, ra0, r_in, n_arc_pts)

        p_la0 = left_arc[0]
        p_la1 = left_arc[-1]
        p_ra1 = right_arc[0]
        p_ra0 = right_arc[-1]

        verts = [p_la0]
        codes = [Path.MOVETO]

        for p in left_arc[1:]:
            verts.append(p)
            codes.append(Path.LINETO)

        # cubic bezier la1 -> ra1, both controls at origin
        verts.extend([(0.0, 0.0), (0.0, 0.0), p_ra1])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

        for p in right_arc[1:]:
            verts.append(p)
            codes.append(Path.LINETO)

        # cubic bezier ra0 -> la0, both controls at origin
        verts.extend([(0.0, 0.0), (0.0, 0.0), p_la0])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

        # explicit close
        verts.append(p_la0)
        codes.append(Path.CLOSEPOLY)

        path = Path(verts, codes)

        if flow_color == "left":
            face = left_colors.get(lg, default_color)
        else:
            face = right_colors.get(rg, default_color)

        ribbon = patches.PathPatch(
            path,
            facecolor=face,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(ribbon)

    # draw colored arc segments as wedges
    def _wedge(a_start, a_end, r, w, color):
        return patches.Wedge(
            center=(0, 0),
            r=r + w,
            theta1=np.rad2deg(a_start),
            theta2=np.rad2deg(a_end),
            width=w,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    for g in left_order:
        a0, a1 = left_ranges[g]
        if a1 > a0:
            ax.add_patch(
                _wedge(a0, a1, r_in, arc_width, left_colors.get(g, default_color))
            )

    for g in right_order:
        a0, a1 = right_ranges[g]
        if a1 > a0:
            ax.add_patch(
                _wedge(a0, a1, r_in, arc_width, right_colors.get(g, default_color))
            )

    # segment labels outside the ring — pushed out via `segment_label_pad`
    # so they don't collide with cumulative-count tick labels
    r_label = r_in + arc_width + segment_label_pad

    def _segment_label(angle, text, side):
        x = r_label * np.cos(angle)
        y = r_label * np.sin(angle)
        ha = "left" if side == "right" else "right"
        ax.text(
            x, y, str(text),
            ha=ha, va="center",
            fontsize=segment_label_fontsize,
            fontweight="bold",
        )

    for g in left_order:
        a0, a1 = left_ranges[g]
        if a1 > a0:
            _segment_label((a0 + a1) / 2, g, side="left")

    for g in right_order:
        a0, a1 = right_ranges[g]
        if a1 > a0:
            _segment_label((a0 + a1) / 2, g, side="right")

    # axis (side) labels at midpoint of each half
    r_axis = r_in + arc_width + segment_label_pad + 0.22

    if left_axis_label is not None:
        am = (left_a0 + left_a1) / 2
        ax.text(
            r_axis * np.cos(am),
            r_axis * np.sin(am),
            left_axis_label,
            ha="right",
            va="center",
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
        )

    if right_axis_label is not None:
        am = (right_a0 + right_a1) / 2
        ax.text(
            r_axis * np.cos(am),
            r_axis * np.sin(am),
            right_axis_label,
            ha="left",
            va="center",
            fontsize=axis_label_fontsize,
            fontweight=axis_label_fontweight,
        )

    # cumulative-count tick marks per segment
    # per-side tick steps are resolved via `tick_step` (scalar / "auto" /
    # tuple / dict). labels closer than `min_tick_label_gap_deg` to the
    # previously drawn label on the same side are suppressed so the
    # numbers stay readable on busy sides; the tick marks themselves are
    # always drawn.
    if show_ticks and tick_step is not None and tick_step != 0:
        r_tick_in = r_in + arc_width
        r_tick_out = r_in + arc_width + 0.035
        r_tick_text = r_in + arc_width + 0.085

        min_label_gap = np.deg2rad(min_tick_label_gap_deg)

        side_specs = [
            ("left",  left_order,  left_ranges,  left_totals),
            ("right", right_order, right_ranges, right_totals),
        ]

        for side, order, ranges, totals in side_specs:
            max_total = max((totals.get(g, 0) for g in order), default=0)
            step = _resolve_tick_step(tick_step, side, max_total)
            if step is None or step <= 0:
                continue

            last_label_ang = None

            for g in order:
                a0, a1 = ranges[g]
                tot = totals.get(g, 0)
                if tot <= 0 or a1 <= a0:
                    continue
                span = a1 - a0
                n_ticks = int(np.floor(tot / step)) + 1
                for k in range(n_ticks):
                    val = k * step
                    if val > tot:
                        continue
                    frac = val / tot
                    ang = a0 + frac * span

                    # always draw the tick mark
                    x0 = r_tick_in * np.cos(ang)
                    y0 = r_tick_in * np.sin(ang)
                    x1 = r_tick_out * np.cos(ang)
                    y1 = r_tick_out * np.sin(ang)
                    ax.plot(
                        [x0, x1], [y0, y1],
                        color="black",
                        linewidth=0.6,
                        zorder=4,
                    )

                    # decide whether to draw the label
                    if hide_zero_tick_label and val == 0:
                        continue
                    if (
                        last_label_ang is not None
                        and abs(ang - last_label_ang) < min_label_gap
                    ):
                        continue

                    rot_deg = np.rad2deg(ang) - 90
                    if rot_deg > 90:
                        rot_deg -= 180
                    elif rot_deg < -90:
                        rot_deg += 180

                    # render value as int when step is integral
                    label_text = (
                        f"{int(val)}" if float(val).is_integer() else f"{val:g}"
                    )
                    ax.text(
                        r_tick_text * np.cos(ang),
                        r_tick_text * np.sin(ang),
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=tick_fontsize,
                        rotation=rot_deg,
                    )
                    last_label_ang = ang

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    return fig, ax

