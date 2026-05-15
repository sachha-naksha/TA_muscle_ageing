# =========================
# Imports
# =========================
import warnings
warnings.filterwarnings("ignore")

import io
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
from itertools import chain, repeat
from PIL import Image

from io import BytesIO
import base64
from IPython.display import HTML, display


# =========================
# GMT Parsing Functions
# =========================
def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """
    Parse a GMT file into a decoupler pathway dataframe.
    """

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )


def gmt_to_decoupler_multiple_pathways(
    gmt_paths,
    geneset_name=None,
    genesymbol_name=None,
):
    """
    Parse multiple GMT files and return a combined pathway dataframe.
    """

    all_records = []

    for pth in gmt_paths:
        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                all_records.extend(zip(repeat(name), genes))

    return pd.DataFrame.from_records(
        all_records,
        columns=[geneset_name, genesymbol_name],
    )


# =========================
# Gene Ranking
# =========================
def compute_gene_ranking(adata, geneset):
    """
    Compute Spearman correlation of each gene with pathway score
    and return ranked genes filtered by selected genes.
    """

    # Gene expression matrix
    X = adata.to_df()

    # AUCell pathway scores
    score_series = adata.obsm["score_aucell"].copy()

    # Compute correlations
    correlations = [
        spearmanr(X[gene], score_series)[0]
        for gene in X.columns
    ]

    # Create dataframe
    gene_importance = pd.DataFrame({
        "gene": X.columns,
        "spearman_corr": correlations
    })

    # Sort by correlation
    gene_importance = (
        gene_importance
        .sort_values("spearman_corr", ascending=False)
        .reset_index(drop=True)
    )

    # Add rank column
    gene_importance["rank"] = gene_importance.index + 1

    # Filter genes belonging to geneset
    ranked_gene_df = (
        gene_importance[gene_importance["gene"].isin(geneset)]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    return gene_importance, ranked_gene_df


# =========================
# Heatmap Plot
# =========================
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


# =========================
# Pairwise Significance
# =========================
def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    Calculate pairwise significance between all groups
    Returns a dictionary of p-values and significance levels
    """
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[data[x_var] == groups[i]][y_var]  # Changed from 'category' and 'senescence_score'
            group2 = data[data[x_var] == groups[j]][y_var]  # Changed from 'category' and 'senescence_score'
            
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Add significance stars
            if pvalue < 0.001:
                sig = '***'
            elif pvalue < 0.01:
                sig = '**'
            elif pvalue < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            results[(i, j)] = {'p-value': pvalue, 'significance': sig}
    
    return results
    

# =========================
# Violin + Box Plot Combo
# =========================
def plot_violin_box_combo(data, x_var, y_var, title=None, x_ticks=None, palette=None, rotation=45, show_scatter=True):
    """
    Create a combined violin-box plot with optional scatter points
    
    Parameters:
    -----------
    show_scatter : bool, default=True
        If True, shows individual data points as scatter. If False, shows only violin and box.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 6))
    
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)

    # Calculate y-axis limits based on data
    y_min = data[y_var].min()
    y_max = data[y_var].max()
    y_range = y_max - y_min
    
    # Add padding proportional to the data range (10% on each side)
    padding = y_range * 0.1
    y_min_plot = y_min - padding
    y_max_plot = y_max + padding
    
    # Only use floor/ceil if the range is large enough
    if y_range > 1.0:
        y_min_plot = np.floor(y_min_plot * 2) / 2
        y_max_plot = np.ceil(y_max_plot * 2) / 2
    else:
        y_min_plot = max(0, y_min_plot)
    
    # Set initial y-axis limits
    ax.set_ylim(y_min_plot, y_max_plot)
    
    # Set appropriate tick intervals based on data range
    if y_range < 0.1:
        tick_interval = 0.02
    elif y_range < 0.5:
        tick_interval = 0.05
    elif y_range < 2.0:
        tick_interval = 0.1
    else:
        tick_interval = 0.5
    
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # Determine order
    if x_ticks is not None:
        categories = x_ticks
    else:
        categories = sorted(data[x_var].unique(), key=lambda x: float(x) if x.replace('.','').isdigit() else x)

    # Create violin plot with explicit order
    violin = sns.violinplot(
        data=data, x=x_var, y=y_var,
        order=categories,
        palette=palette, inner=None,
        linewidth=0, saturation=1.0,
        alpha=0.3, width=0.4, cut=0
    )

    # Create box plot with explicit order
    box_plot = sns.boxplot(
        data=data, x=x_var, y=y_var,
        order=categories,
        width=0.4, linewidth=1.2,
        flierprops={'marker': ' '},
        showmeans=False,
        boxprops={
            'facecolor': 'none',
            'edgecolor': 'none'
        },
        whiskerprops={'color': 'none'},
        medianprops={'color': 'none'},
        showcaps=False,
        ax=ax
    )

    # Count number of boxes and lines per box
    num_boxes = len(categories)
    lines_per_box = len(ax.lines) // num_boxes

    # Update box plot colors after creation
    for i, (name, box) in enumerate(zip(categories, ax.patches)):
        color = palette[name]
        
        # Create filled box with transparency
        box.set_facecolor(color)
        box.set_edgecolor('none')
        box.set_alpha(0.3)
        box.set_zorder(1)
        
        # Create box edges with full opacity
        path = box.get_path()
        edges = mpatches.PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            linewidth=1.2,
            alpha=1.0,
            zorder=2
        )
        ax.add_patch(edges)
        
        # Get and color all lines for this box
        box_lines = ax.lines[i * lines_per_box : (i + 1) * lines_per_box]
        for line in box_lines:
            line.set_color(color)
            line.set_alpha(1.0)
            line.set_linewidth(1.2)
            line.set_zorder(2)

    # ========== CONDITIONAL SCATTER POINTS ==========
    if show_scatter:
        # Add individual points on top with explicit order
        sns.stripplot(
            data=data, x=x_var, y=y_var,
            order=categories,
            palette=palette, size=6,
            alpha=1.0, linewidth=0,
            jitter=0.2, zorder=3
        )
    # ================================================
    
    # Calculate significance using the ordered categories
    significance_info = calculate_pairwise_significance(data, categories, x_var, y_var)

    # Get current y limits before adding bars
    current_ymin, current_ymax = ax.get_ylim()
    y_range_plot = current_ymax - current_ymin
    
    # Make bar spacing relative to the data range
    bar_spacing = y_range_plot * 0.08
    bar_tips = y_range_plot * 0.02
    bar_height = current_ymax + bar_spacing * 0.5

    # Add significance bar function
    def add_significance_bar(start, end, height, p_value, sig_symbol):
        # Draw the bar
        ax.plot([start, start, end, end], 
                [height, height + bar_tips, height + bar_tips, height],
                color='black', linewidth=0.8)
        
        # If p-value rounds to 0.0000 (very small), show only asterisks
        if p_value < 0.00005:  # This rounds to 0.0000 with 4 decimals
            text = sig_symbol  # Just "***"
        else:
            text = f'p = {p_value:.4f} {sig_symbol}'  # "p = 0.0123 **"
        ax.text((start + end) * 0.5, height + bar_tips, 
                text, ha='center', va='bottom', fontsize=8)

    # Add significant bars (p < 0.05 only)
    for (group1_idx, group2_idx), sig_data in significance_info.items():
        if sig_data['significance'] != 'ns':
            add_significance_bar(
                group1_idx, 
                group2_idx, 
                bar_height,
                sig_data['p-value'],
                sig_data['significance']
            )
            bar_height += bar_spacing

    # Adjust y-axis limits to accommodate bars
    ax.set_ylim(current_ymin, bar_height + bar_spacing * 0.5)

    if title:
        plt.title(title, pad=20)

    if x_ticks is None:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=rotation, ha='right')
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
        ax.spines['bottom'].set_visible(True)

    # Configure ticks and spines with thinner lines
    ax.minorticks_off()
    ax.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis='x', which='major', top=False)
    ax.tick_params(axis='y', which='major', right=False, width=0.8)
    
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(width=0.8)
    
    plt.setp(ax.get_yticklabels(), weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.yaxis.grid(False)
    
    sns.despine(offset=5, trim=True, bottom=(x_ticks is None), right=True)
    
    # Force rotation of x-tick labels
    if x_ticks is not None:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    
    plt.close()
    
    return fig


# =========================
# Combine Plots
# =========================
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
