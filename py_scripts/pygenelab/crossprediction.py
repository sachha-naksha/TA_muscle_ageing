"""
crossprediction

Transfer a trained SLIDE / Essential-Regression model from one dataset to
another: reuse the A-loadings (and C, Gamma) learnt on a *source* dataset to
score the latent factors of a *target* dataset via EssReg `predZ` -- no
retraining, no decoupler / AUCell. Includes a mouse->human ortholog bridge so
a model trained in mouse gene space can score human cells.

predZ (EssReg):  Z = X . Gamma^-1 . A . pinv(A^T Gamma^-1 A + C^-1)
The input X must be column-standardized exactly like SLIDE's calcZMatrix
(`scale(x, TRUE, TRUE)`); `standardize_cols` does this and reproduces a stored
SLIDE `z_matrix.csv` to r ~ 1.0.

Typical use
-----------
    import pygenelab as pgl
    model   = pgl.crossprediction.load_slide_model(MODEL_DIR)
    sig_lfs = pgl.crossprediction.significant_lfs(model)
    omap, stats = pgl.crossprediction.build_human_ortholog_map(
        ORTHOLOGS, model["genes"], adata.var_names)
    new_z = pgl.crossprediction.project_human(adata, model, omap)

The high-level one-liner that attaches LF scores to `adata.obs` for plotting:

    new_z = pgl.crossprediction.transfer_latent_scores(
        adata, MODEL_DIR, ORTHOLOGS, lfs=["Z12"],
        orient_by=(adata.obs["age"] == adata.obs["age"].max()).values)
"""

import os
import sys

import numpy as np
import pandas as pd

__all__ = [
    "load_slide_model",
    "significant_lfs",
    "build_human_ortholog_map",
    "standardize_cols",
    "predZ",
    "align_mouse_to_A",
    "align_human_to_A",
    "project_to_latent",
    "project_mouse",
    "project_human",
    "fit_transfer_classifier",
    "plot_transfer_roc",
    "transfer_latent_scores",
    "transferred_driver_genes",
    "plot_transferred_driver_heatmap",
]


# ----------------------------------------------------------------------
# model loading (rpy2 + base-R readRDS)
# ----------------------------------------------------------------------
def load_slide_model(model_dir, r_home=None):
    """Read a SLIDE out-folder's ``AllLatentFactors.rds`` + ``SLIDE_LFs.rds``
    with base-R ``readRDS`` (via rpy2) and return the pieces ``predZ`` needs.

    Returns a dict with:
        A (genes x K ndarray), genes (mouse symbols = rows of A),
        lfnames (Z1..ZK), C (K x K), Gamma (len p), beta (len K),
        marginal (list[int]), inter_p1 / inter_p2 (list[int]).

    Notes
    -----
    rpy2 must point at the R it was built against. In the ``decoupler_psc``
    conda env that is ``$CONDA_PREFIX/lib/R`` -- the system R 4.5 ABI differs
    (``undefined symbol: R_typeToChar``). We force ``R_HOME`` *before* importing
    rpy2; call this before any other rpy2 use. Pass ``r_home`` to override.
    """
    os.environ["R_HOME"] = r_home or os.path.join(sys.prefix, "lib", "R")
    import rpy2.robjects as ro

    readRDS = ro.r["readRDS"]
    rownames = ro.r["rownames"]
    colnames = ro.r["colnames"]

    lf = readRDS(os.path.join(model_dir, "AllLatentFactors.rds"))
    A = np.asarray(lf.rx2("A"))
    genes = list(rownames(lf.rx2("A")))      # source (mouse) gene symbols
    lfnames = list(colnames(lf.rx2("A")))    # Z1 .. ZK
    C = np.asarray(lf.rx2("C"))
    Gamma = np.asarray(lf.rx2("Gamma"))
    beta = np.asarray(lf.rx2("beta")).ravel()

    sl = readRDS(os.path.join(model_dir, "SLIDE_LFs.rds"))
    marginal = [int(v) for v in np.asarray(sl.rx2("marginal_vals"))]
    inter = sl.rx2("interaction")
    p1 = [int(v) for v in np.asarray(inter.rx2("p1"))]
    p2 = [int(v) for v in np.asarray(inter.rx2("p2"))]

    return dict(A=A, genes=genes, lfnames=lfnames, C=C, Gamma=Gamma, beta=beta,
                marginal=marginal, inter_p1=p1, inter_p2=p2)


def significant_lfs(model, use_interactions=True):
    """Significant latent factors, exactly as the R cross-prediction script:
    ``sig_lfs = unique( marginal_vals  U  interaction$p2 )``. Returns a list of
    ``"Z<k>"`` names in first-seen order. Set ``use_interactions=False`` for the
    marginal LFs only."""
    vals = list(model["marginal"])
    if use_interactions:
        vals = vals + list(model["inter_p2"])
    out = []
    for v in vals:
        z = f"Z{v}"
        if z not in out:
            out.append(z)
    return out


# ----------------------------------------------------------------------
# ortholog bridge
# ----------------------------------------------------------------------
def build_human_ortholog_map(orthologs_csv, source_genes, target_genes,
                             source_col="mouse", target_col="human"):
    """Map source (mouse) gene symbols -> human symbols present in the target
    object, using a 1:1 ortholog table.

    Returns ``(mapping, stats)`` where ``mapping`` is ``{source_gene:
    human_symbol}`` restricted to orthologs present in ``target_genes``, and
    ``stats`` records coverage (``n_total``, ``n_mapped``, ``n_present``,
    ``frac_present``).
    """
    orth = pd.read_csv(orthologs_csv)
    s2t = dict(zip(orth[source_col], orth[target_col]))
    target = set(target_genes)

    mapping = {}
    for g in source_genes:
        h = s2t.get(g)
        if h is not None and h in target:
            mapping[g] = h

    n_total = len(source_genes)
    n_mapped = sum(g in s2t for g in source_genes)
    stats = dict(
        n_total=n_total,
        n_mapped=n_mapped,
        n_present=len(mapping),
        frac_present=(len(mapping) / n_total if n_total else 0.0),
    )
    return mapping, stats


# ----------------------------------------------------------------------
# predZ core
# ----------------------------------------------------------------------
def standardize_cols(X):
    """Column z-score == SLIDE's ``scale(x, TRUE, TRUE)``. Constant /
    zero-filled columns (sd == 0) become 0 instead of NaN."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    Z = (X - mu) / np.where(sd == 0, 1.0, sd)
    Z[:, sd == 0] = 0.0
    return Z


def predZ(X_std, A, C, Gamma):
    """EssReg predZ:  Z = X Gamma^-1 A pinv(A^T Gamma^-1 A + C^-1).
    ``X_std`` must already be column-standardized."""
    Gamma = np.where(Gamma == 0, 1e-10, Gamma)
    Gi = np.diag(1.0 / Gamma)
    G = A.T @ Gi @ A + np.linalg.inv(C)
    return X_std @ Gi @ A @ np.linalg.pinv(G)


# ----------------------------------------------------------------------
# gene-axis alignment
# ----------------------------------------------------------------------
def align_mouse_to_A(expr_df, source_genes):
    """``expr_df`` (samples x source-genes) -> ndarray with columns in
    ``source_genes`` order, zero-filling genes absent from ``expr_df`` (R's
    ``orig_x[, missing] <- 0``). Returns ``(ndarray, n_shared)``."""
    out = pd.DataFrame(0.0, index=expr_df.index, columns=source_genes)
    shared = [g for g in source_genes if g in expr_df.columns]
    out[shared] = expr_df[shared].values
    return out.values, len(shared)


def align_human_to_A(adata, source_genes, ortholog_map, layer=None):
    """Pull human expression into SOURCE (mouse) gene columns via the ortholog
    map: column j (source gene ``source_genes[j]``) = the target cell's
    expression of its human ortholog, else 0. ``layer`` selects an
    ``adata.layers`` matrix (default ``adata.X``, which is lognorm here).
    Returns ``(ndarray cells x p, list_of_cell_names)``."""
    human_syms = [ortholog_map.get(g) for g in source_genes]
    needed = sorted({h for h in human_syms if h is not None})
    sub = adata[:, needed]
    M = sub.layers[layer] if (layer and layer in adata.layers) else sub.X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    hexpr = pd.DataFrame(M, index=adata.obs_names, columns=needed)

    X = np.zeros((adata.n_obs, len(source_genes)), dtype=float)
    for j, h in enumerate(human_syms):
        if h is not None:
            X[:, j] = hexpr[h].values
    return X, list(adata.obs_names)


# ----------------------------------------------------------------------
# projection
# ----------------------------------------------------------------------
def project_to_latent(X_raw, model, index=None):
    """Standardize ``X_raw`` and project onto the model latent factors.
    Returns a DataFrame (rows = ``index``, columns = Z1..ZK)."""
    Z = predZ(standardize_cols(X_raw), model["A"], model["C"], model["Gamma"])
    return pd.DataFrame(Z, index=index, columns=model["lfnames"])


def project_mouse(mouse_x_df, model):
    """Project the source (mouse) X matrix onto its own latent factors -- this
    reproduces the stored SLIDE ``z_matrix.csv``. Returns a DataFrame."""
    X_raw, _ = align_mouse_to_A(mouse_x_df, model["genes"])
    return project_to_latent(X_raw, model, index=mouse_x_df.index)


def project_human(adata, model, ortholog_map, layer=None):
    """Project human cells onto the source-model latent factors via the
    ortholog bridge. Returns a DataFrame (cells x Z1..ZK)."""
    X_raw, cells = align_human_to_A(adata, model["genes"], ortholog_map, layer=layer)
    return project_to_latent(X_raw, model, index=cells)


# ----------------------------------------------------------------------
# cross-prediction classifier + ROC
# ----------------------------------------------------------------------
def fit_transfer_classifier(orig_z, orig_y, new_z, sig_lfs):
    """Fit a linear model on the SOURCE latent factors (``orig_z[sig_lfs]`` vs
    binary ``orig_y``) and apply it, unchanged, to the TARGET latent factors --
    the R ``glm(gaussian)`` cross-prediction. Returns a dict with the fitted
    ``model``, ``orig_pred`` / ``new_pred`` and ``orig_auc`` / ``new_auc``
    (the latter requires ``new_y`` via :func:`transfer_auc`)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    lin = LinearRegression().fit(orig_z[sig_lfs].values, np.asarray(orig_y))
    orig_pred = lin.predict(orig_z[sig_lfs].values)
    new_pred = lin.predict(new_z[sig_lfs].values)
    return dict(
        model=lin,
        sig_lfs=list(sig_lfs),
        orig_pred=orig_pred,
        new_pred=new_pred,
        orig_auc=roc_auc_score(np.asarray(orig_y), orig_pred),
    )


def plot_transfer_roc(orig_y, orig_pred, new_y, new_pred,
                      *, orig_label="source", new_label="target",
                      title="SLIDE transfer ROC", figsize=(5, 5),
                      colors=("#7BDE7B", "#B83636"), ax=None):
    """ROC of the source self-prediction and the target transfer on one axis.
    Returns ``(fig, dict_of_aucs)``."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    aucs = {}
    for y_true, y_score, lab, col in [
            (orig_y, orig_pred, orig_label, colors[0]),
            (new_y, new_pred, new_label, colors[1])]:
        if y_true is None:
            continue
        auc = roc_auc_score(np.asarray(y_true), np.asarray(y_score))
        aucs[lab] = auc
        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig, aucs


# ----------------------------------------------------------------------
# driver genes of a latent factor, translated to the target species
# ----------------------------------------------------------------------
def transferred_driver_genes(model_dir, lf, orthologs_csv, target_genes,
                             top_n=None, return_pairs=False):
    """Driver genes of latent factor ``lf`` -- read from ``feature_list_<lf>.txt``
    in the SLIDE out-folder -- translated to target (human) symbols present in
    ``target_genes`` via the 1:1 ortholog table. Genes are ordered by descending
    ``|A_loading|`` (their SLIDE driver strength); ``top_n`` keeps the strongest.

    Returns a list of human symbols, or -- with ``return_pairs=True`` -- a list of
    ``(source_gene, human_gene, A_loading)`` tuples (useful to label which mouse
    driver each human gene came from).
    """
    fl = pd.read_csv(os.path.join(model_dir, f"feature_list_{lf}.txt"), sep="\t")
    fl = fl.assign(_abs=fl["A_loading"].abs()).sort_values("_abs", ascending=False)

    src_genes = fl["names"].astype(str).tolist()
    mapping, _ = build_human_ortholog_map(orthologs_csv, src_genes, target_genes)

    pairs = [(g, mapping[g], float(al))
             for g, al in zip(fl["names"].astype(str), fl["A_loading"])
             if g in mapping]
    if top_n is not None:
        pairs = pairs[:top_n]
    if return_pairs:
        return pairs
    return [human for _, human, _ in pairs]


def _lf_driver_table(model_dir, lf, orthologs_csv, target_genes, top_n=None):
    """Internal: driver genes of ``lf`` with A_loading + mouse direction, mapped
    to human. Returns a DataFrame [mouse, human, A_loading, direction, corrs]
    ordered by descending |A_loading|. ``direction`` = +1 (Red: up in KO/aging)
    or -1 (Blue: down)."""
    fl = pd.read_csv(os.path.join(model_dir, f"feature_list_{lf}.txt"), sep="\t")
    fl = fl.assign(_abs=fl["A_loading"].abs()).sort_values("_abs", ascending=False)
    mapping, _ = build_human_ortholog_map(orthologs_csv, fl["names"].astype(str), target_genes)
    rows = []
    for _, r in fl.iterrows():
        g = str(r["names"])
        if g not in mapping:
            continue
        rows.append(dict(
            mouse=g, human=mapping[g], A_loading=float(r["A_loading"]),
            direction=(1 if str(r.get("color", "")).strip().lower() == "red" else -1),
            corrs=float(r.get("corrs", np.nan)),
        ))
    tab = pd.DataFrame(rows)
    if top_n is not None:
        tab = tab.head(top_n)
    return tab.reset_index(drop=True)


def plot_transferred_driver_heatmap(
    adata, model_dir, lf, orthologs_csv, *,
    group_col, groups, group_labels=None,
    top_n=20, layer=None, cmap="RdBu_r", figsize=(5.5, 8),
    annot=True, vmax=None, title=None, show_loading_bar=False,
):
    """Biologist-facing driver heatmap for a transferred latent factor.

    One row per driver gene (ordered by SLIDE ``A_loading``); the heatmap is the
    per-group mean of each gene's *z-scored* expression across the target cells
    (matrixplot style): red = above-average in that group, blue = below. This
    shows the real expression shift (e.g. PDK4 up in old), not a within-group
    correlation with the composite score.

    ``show_loading_bar=True`` adds a side bar of ``A_loading`` (how strongly each
    gene defines ``lf``) coloured by the *mouse* direction (red = up in KO/aging,
    blue = down); default ``False`` keeps just the heatmap.

    Returns ``(table_df, fig)`` where ``table_df`` has the per-group mean-z values
    plus A_loading / direction (the direction/loading are always returned even
    when the bar is hidden).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec

    from . import geneset_activity as _ga

    tab = _lf_driver_table(model_dir, lf, orthologs_csv, adata.var_names, top_n=top_n)
    human = tab["human"].tolist()

    # per-gene mean z-scored expression per group (shared "expression shift" calc)
    cols = [str(g) for g in groups]
    labels = list(group_labels) if group_labels is not None else cols
    mat = _ga.compute_group_expression_shift(
        adata, human, group_col, groups, layer=layer, standardize=True)
    mat = mat.reindex(human)        # keep A_loading (driver-strength) order
    mat.columns = labels

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(mat.values), 99)) or 1.0

    heatmap_title = title or f"{lf} drivers — target expression by {group_col}"

    if not show_loading_bar:
        # heatmap only
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
            annot=annot, fmt=".1f", linewidths=0.5, linecolor="white",
            cbar_kws=dict(label="mean z-scored expression"),
        )
        ax.set_yticklabels(human, rotation=0, fontstyle="italic")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(group_col)
        ax.set_ylabel("")
        ax.set_title(heatmap_title)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=figsize)
        # heatmap | A_loading bar | colorbar  (separate axes; no shared y so the
        # gene labels on the heatmap are never cleared by the bar axis)
        gs = gridspec.GridSpec(1, 3, width_ratios=[max(len(cols), 2), 1.6, 0.18],
                               wspace=0.10)
        ax = fig.add_subplot(gs[0])
        axb = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])

        sns.heatmap(
            mat, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
            annot=annot, fmt=".1f", linewidths=0.5, linecolor="white",
            cbar_ax=cax, cbar_kws=dict(label="mean z-scored expression"),
        )
        ax.set_yticklabels(human, rotation=0, fontstyle="italic")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(group_col)
        ax.set_ylabel("")
        ax.set_title(heatmap_title)

        # A_loading bars aligned to the heatmap rows (row 0 at top)
        y = np.arange(len(human)) + 0.5
        bar_colors = ["#B83636" if d > 0 else "#3B6FB8" for d in tab["direction"]]
        axb.barh(y, tab["A_loading"].values, color=bar_colors, height=0.7)
        axb.set_ylim(ax.get_ylim())          # match heatmap (already top-down)
        axb.set_yticks([])
        axb.set_xlabel("A_loading (driver wt)")
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        # legend for the mouse direction encoding
        from matplotlib.patches import Patch
        axb.legend(handles=[Patch(color="#B83636", label="up in KO/aging"),
                            Patch(color="#3B6FB8", label="down")],
                   fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()

    mat["A_loading"] = tab["A_loading"].values
    mat["direction"] = tab["direction"].values
    return mat, fig


# ----------------------------------------------------------------------
# high-level convenience: attach transferred LF scores to adata.obs
# ----------------------------------------------------------------------
def transfer_latent_scores(adata, model_dir, orthologs_csv, *,
                           lfs=None, score_suffix="_transfer_score",
                           orient_by=None, layer=None, r_home=None,
                           inplace=True, verbose=True):
    """One-call transfer: load the source model, bridge orthologs, project the
    target cells, and write the requested latent-factor scores into
    ``adata.obs[f"{lf}{score_suffix}"]`` (e.g. ``adata.obs["Z12_transfer_score"]``).

    Parameters
    ----------
    lfs : list[str] or None
        Which factors to attach (e.g. ``["Z12"]``). ``None`` -> all significant
        LFs (:func:`significant_lfs`).
    orient_by : array-like[bool/int] or None
        Optional binary label (one per cell). Each attached LF is sign-flipped
        so that higher score = the positive class (AUC >= 0.5). Useful so "older"
        cells score high regardless of the arbitrary latent-factor sign.
    layer : str or None
        Target expression layer (default ``adata.X``, lognorm here).

    Returns
    -------
    new_z : DataFrame
        Full target projection (cells x Z1..ZK). LF columns are also copied to
        ``adata.obs`` (unless ``inplace=False``).
    """
    model = load_slide_model(model_dir, r_home=r_home)
    omap, stats = build_human_ortholog_map(orthologs_csv, model["genes"], adata.var_names)
    if verbose:
        print(f"ortholog coverage: {stats['n_present']}/{stats['n_total']} "
              f"A-genes ({100 * stats['frac_present']:.1f}%) transfer to human")

    new_z = project_human(adata, model, omap, layer=layer)

    if lfs is None:
        lfs = significant_lfs(model)

    if orient_by is not None:
        from sklearn.metrics import roc_auc_score
        orient_by = np.asarray(orient_by).astype(int)

    if inplace:
        for lf in lfs:
            s = new_z[lf].reindex(adata.obs_names).values
            if orient_by is not None:
                auc = roc_auc_score(orient_by, s)
                if auc < 0.5:
                    s = -s
            adata.obs[f"{lf}{score_suffix}"] = s
        if verbose:
            print(f"attached: {[f'{lf}{score_suffix}' for lf in lfs]}")

    return new_z
