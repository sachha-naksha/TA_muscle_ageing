# pseudotime_animation.py

"""
animated pseudotime trajectories of pathway / gene activity scores.

two entry points:
    - create_animated_pathway_plot: drives a moving dot along a smoothed
      score-vs-pseudotime curve, one subplot per score column.
    - animate_gene_along_pseudotime: thin wrapper that pulls a gene's
      expression from an AnnData layer and reuses the pathway animator.

writes either MP4 (FFMpegWriter) or GIF (PillowWriter); falls back to
GIF if MP4 fails. returns the matplotlib Animation + Figure so callers
can embed in notebooks via the returned objects.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from scipy.ndimage import gaussian_filter1d

import scanpy as sc
from anndata import AnnData


def _ffmpeg_available():
    """return True if an ffmpeg binary is on PATH and runs"""
    import subprocess
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=5,
        )
        return True
    except Exception:
        return False


def create_animated_pathway_plot(
    df_cell_level: pd.DataFrame,
    score_cols: list,
    pseudotime_key: str = 'Pseudotime',
    group_by_key: str = 'Annotation',
    smoothing_method: str = 'gaussian',
    smoothing_strength: float = 120.0,
    geneset_sizes: pd.Series = None,
    groups_to_plot: list = None,
    colors_dict: dict = None,
    legend_labels_map: dict = None,
    n_subplot_cols: int = 2,
    figsize_per_subplot: tuple = (7, 4),
    nframes: int = 200,
    fps: int = 30,
    dot_size: int = 150,
    output_file: str = 'pathway_animation.mp4',
    use_gif: bool = None,
    verbose: bool = True,
):
    """
    animate a moving dot along smoothed score-vs-pseudotime curves.

    df_cell_level must contain `pseudotime_key`, `group_by_key`, and each
    column in `score_cols`. one subplot per score column.

    use_gif:
        None  -> autodetect ffmpeg, fall back to GIF if missing
        True  -> force GIF output (PillowWriter)
        False -> force MP4 output (FFMpegWriter)

    returns (anim, fig) on success, or (HTML, fig) if both writers fail.
    """
    if use_gif is None:
        use_gif = not _ffmpeg_available()

    if verbose:
        print("Starting animation creation...")

    required_cols = list(score_cols) + [pseudotime_key, group_by_key]
    missing = [c for c in required_cols if c not in df_cell_level.columns]
    if missing:
        raise KeyError(f"missing columns in df_cell_level: {missing}")

    num_scores = len(score_cols)
    n_subplot_rows = (num_scores + n_subplot_cols - 1) // n_subplot_cols

    fig, axes = plt.subplots(
        n_subplot_rows, n_subplot_cols,
        figsize=(figsize_per_subplot[0] * n_subplot_cols,
                 figsize_per_subplot[1] * n_subplot_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    all_lines_data = []

    for subplot_idx, score_col in enumerate(score_cols):
        ax = axes[subplot_idx]

        plot_df = df_cell_level[[score_col, pseudotime_key, group_by_key]].dropna()
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        if groups_to_plot:
            groups = [g for g in groups_to_plot if g in plot_df[group_by_key].unique()]
        else:
            groups = sorted(plot_df[group_by_key].unique())

        for group_idx, group in enumerate(groups):
            group_data = plot_df[plot_df[group_by_key] == group].sort_values(pseudotime_key)
            if len(group_data) < 2:
                continue

            x_vals = group_data[pseudotime_key].values
            y_vals = group_data[score_col].values

            if smoothing_method == 'gaussian' and len(y_vals) > 10:
                try:
                    y_vals_smooth = gaussian_filter1d(y_vals, sigma=smoothing_strength)
                except Exception:
                    y_vals_smooth = y_vals
                    if verbose:
                        print(f"warning: gaussian smoothing failed for {group} in {score_col}")
            else:
                y_vals_smooth = y_vals

            color = colors_dict.get(group, f'C{group_idx}') if colors_dict else f'C{group_idx}'
            label = legend_labels_map.get(group, group) if legend_labels_map else group

            ax.plot(x_vals, y_vals_smooth, color=color, linewidth=1.5, label=label, alpha=0.8)

            dot = ax.scatter(
                [], [], s=dot_size, color=color,
                zorder=100, edgecolors='white', linewidths=2,
            )

            all_lines_data.append({
                'ax': ax,
                'x': x_vals,
                'y': y_vals_smooth,
                'dot': dot,
                'color': color,
            })

        title = score_col.replace('_', ' ')
        if geneset_sizes is not None and score_col in geneset_sizes.index:
            title += f"\n(n={int(geneset_sizes[score_col])})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"Pseudotime $\rightarrow$", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    for idx in range(num_scores, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if verbose:
        print(f"Figure created with {len(all_lines_data)} animated lines")

    def init():
        for line_data in all_lines_data:
            line_data['dot'].set_offsets(np.empty((0, 2)))
        return [ld['dot'] for ld in all_lines_data]

    def animate(frame):
        progress = frame / (nframes - 1) if nframes > 1 else 0.0
        for line_data in all_lines_data:
            x = line_data['x']
            y = line_data['y']
            if len(x) > 0:
                idx = min(int(progress * (len(x) - 1)), len(x) - 1)
                line_data['dot'].set_offsets([[x[idx], y[idx]]])
        return [ld['dot'] for ld in all_lines_data]

    if verbose:
        print(f"Creating animation with {nframes} frames at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=nframes, interval=1000 / fps,
        blit=True, repeat=True,
    )

    output_file = str(output_file)
    out_dir = Path(output_file).parent
    if str(out_dir) and out_dir != Path(""):
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_gif:
            if not output_file.endswith('.gif'):
                output_file = str(Path(output_file).with_suffix('.gif'))
            if verbose:
                print(f"Saving as GIF to {output_file}...")
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=100)
            if verbose:
                print("GIF saved successfully")
        else:
            if verbose:
                print(f"Saving as MP4 to {output_file}...")
            writer = FFMpegWriter(fps=fps, codec='mpeg4', bitrate=1800)
            anim.save(output_file, writer=writer, dpi=100)
            if verbose:
                print("MP4 saved successfully")
        return anim, fig

    except Exception as e:
        if verbose:
            print(f"error saving animation: {e}")
            print("falling back to GIF...")
        try:
            output_file_gif = str(Path(output_file).with_suffix('.gif'))
            writer = PillowWriter(fps=fps)
            anim.save(output_file_gif, writer=writer, dpi=100)
            if verbose:
                print(f"GIF fallback saved to {output_file_gif}")
            return anim, fig
        except Exception as e2:
            if verbose:
                print(f"GIF fallback also failed: {e2}")
            from IPython.display import HTML
            return HTML(anim.to_jshtml()), fig


def animate_gene_along_pseudotime(
    adata: AnnData,
    genes,
    pseudotime_key: str = 'Pseudotime',
    group_by_key: str = 'Annotation',
    layer: str = 'lognorm',
    smoothing_method: str = 'gaussian',
    smoothing_strength: float = 120.0,
    groups_to_plot: list = None,
    colors_dict: dict = None,
    legend_labels_map: dict = None,
    n_subplot_cols: int = 1,
    figsize_per_subplot: tuple = (7, 4),
    nframes: int = 200,
    fps: int = 30,
    dot_size: int = 150,
    output_file: str = 'gene_animation.mp4',
    use_gif: bool = None,
    verbose: bool = True,
):
    """
    animate gene expression along pseudotime, styled like the pathway animator.

    pulls expression for each gene from `adata.layers[layer]` (set layer=None
    to use adata.X) and forwards everything else to create_animated_pathway_plot.
    """
    if isinstance(genes, str):
        genes = [genes]

    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(f"genes not in adata.var_names: {missing}")
    for c in [pseudotime_key, group_by_key]:
        if c not in adata.obs.columns:
            raise KeyError(f"'{c}' not in adata.obs.columns")

    df_cells = sc.get.obs_df(
        adata,
        keys=list(genes) + [pseudotime_key, group_by_key],
        layer=layer,
        use_raw=False,
    )

    return create_animated_pathway_plot(
        df_cell_level=df_cells,
        score_cols=list(genes),
        pseudotime_key=pseudotime_key,
        group_by_key=group_by_key,
        smoothing_method=smoothing_method,
        smoothing_strength=smoothing_strength,
        geneset_sizes=None,
        groups_to_plot=groups_to_plot,
        colors_dict=colors_dict,
        legend_labels_map=legend_labels_map,
        n_subplot_cols=n_subplot_cols,
        figsize_per_subplot=figsize_per_subplot,
        nframes=nframes,
        fps=fps,
        dot_size=dot_size,
        output_file=output_file,
        use_gif=use_gif,
        verbose=verbose,
    )
