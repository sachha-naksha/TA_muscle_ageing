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
from matplotlib.lines import Line2D

from .utils import calculate_pairwise_significance


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

    # return dataframe and plot objects
    return heatmap_df, fig, ax


# plot_violin_box_combo
def plot_violin_box_combo(
    data,
    x_var,
    y_var,
    title=None,
    x_ticks=None,
    palette=None,
    rotation=45,
    show_scatter=True
):
    """
    plot violin and boxplot together for grouped data
    """

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 6))

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)

    # get y-axis range
    y_min = data[y_var].min()
    y_max = data[y_var].max()
    y_range = y_max - y_min

    # add padding around values
    padding = y_range * 0.1
    y_min_plot = y_min - padding
    y_max_plot = y_max + padding

    # round limits if range is large
    if y_range > 1.0:
        y_min_plot = np.floor(y_min_plot * 2) / 2
        y_max_plot = np.ceil(y_max_plot * 2) / 2
    else:
        y_min_plot = max(0, y_min_plot)

    # set y-axis limits
    ax.set_ylim(y_min_plot, y_max_plot)

    # choose tick spacing
    if y_range < 0.1:
        tick_interval = 0.02
    elif y_range < 0.5:
        tick_interval = 0.05
    elif y_range < 2.0:
        tick_interval = 0.1
    else:
        tick_interval = 0.5

    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # set category order
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(
            data[x_var].unique(),
            key=lambda x: float(x) if str(x).replace(".", "").isdigit() else x
        )

    # create violin plot
    sns.violinplot(
        data=data,
        x=x_var,
        y=y_var,
        order=categories,
        palette=palette,
        inner=None,
        linewidth=0,
        saturation=1.0,
        alpha=0.3,
        width=0.4,
        cut=0,
        ax=ax
    )

    # create box plot
    sns.boxplot(
        data=data,
        x=x_var,
        y=y_var,
        order=categories,
        width=0.4,
        linewidth=1.2,
        flierprops={"marker": " "},
        showmeans=False,
        boxprops={
            "facecolor": "none",
            "edgecolor": "none"
        },
        whiskerprops={"color": "none"},
        medianprops={"color": "none"},
        showcaps=False,
        ax=ax
    )

    # get box line groups
    num_boxes = len(categories)
    lines_per_box = len(ax.lines) // num_boxes

    # color boxplot elements
    for i, (name, box) in enumerate(zip(categories, ax.patches)):
        color = palette[name]

        box.set_facecolor(color)
        box.set_edgecolor("none")
        box.set_alpha(0.3)
        box.set_zorder(1)

        # add visible box border
        path = box.get_path()
        edges = mpatches.PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            alpha=1.0,
            zorder=2
        )
        ax.add_patch(edges)

        # color box lines
        box_lines = ax.lines[i * lines_per_box : (i + 1) * lines_per_box]

        for line in box_lines:
            line.set_color(color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)
            line.set_zorder(2)

    # add scatter points if needed
    if show_scatter:
        sns.stripplot(
            data=data,
            x=x_var,
            y=y_var,
            order=categories,
            palette=palette,
            size=6,
            alpha=1.0,
            linewidth=0,
            jitter=0.2,
            zorder=3,
            ax=ax
        )

    # calculate pairwise significance
    significance_info = calculate_pairwise_significance(
        data,
        categories,
        x_var,
        y_var
    )

    # get current y-axis limits
    current_ymin, current_ymax = ax.get_ylim()
    y_range_plot = current_ymax - current_ymin

    # set significance bar spacing
    bar_spacing = y_range_plot * 0.08
    bar_tips = y_range_plot * 0.02
    bar_height = current_ymax + bar_spacing * 0.5

    def add_significance_bar(start, end, height, p_value, sig_symbol):
        # draw significance bar
        ax.plot(
            [start, start, end, end],
            [height, height + bar_tips, height + bar_tips, height],
            color="black",
            linewidth=0.8
        )

        # choose p-value label
        if p_value < 0.00005:
            text = sig_symbol
        else:
            text = f"p = {p_value:.4f} {sig_symbol}"

        ax.text(
            (start + end) * 0.5,
            height + bar_tips,
            text,
            ha="center",
            va="bottom",
            fontsize=8
        )

    # add significant comparisons
    for (group1_idx, group2_idx), sig_data in significance_info.items():
        if sig_data["significance"] != "ns":
            add_significance_bar(
                group1_idx,
                group2_idx,
                bar_height,
                sig_data["p-value"],
                sig_data["significance"]
            )

            bar_height += bar_spacing

    # adjust y-axis for bars
    ax.set_ylim(current_ymin, bar_height + bar_spacing * 0.5)

    # add title if given
    if title:
        plt.title(title, pad=20)

    # format x-axis labels
    if x_ticks is None:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=rotation, ha="right")
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
        ax.spines["bottom"].set_visible(True)

    # clean ticks and spines
    ax.minorticks_off()
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis="x", which="major", top=False)
    ax.tick_params(axis="y", which="major", right=False, width=0.8)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_tick_params(width=0.8)

    # clean labels and grid
    plt.setp(ax.get_yticklabels(), weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(False)

    sns.despine(offset=5, trim=True, bottom=(x_ticks is None), right=True)

    # rotate x-axis labels
    if x_ticks is not None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    plt.close()

    # return figure
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
