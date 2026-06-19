"""
visualize_manifold.py
Publication-quality static and interactive PCA manifold figures.

Both functions share the same Z / y inputs so the projection is computed
once by the caller and passed through — no duplication of PCA logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_manifold_static(Z, y, explained_variance=None,
                         save_path="figures/pca_manifold.png"):
    """
    2D scatter (PC1 vs PC2) colored by lick label with KDE density contours.

    Parameters
    ----------
    Z                 : np.ndarray (n_bins, ≥2)   PCA scores.
    y                 : np.ndarray (n_bins,)       Binary lick labels.
    explained_variance: array-like of length ≥2, optional
                        Explained variance ratio per component; used in axis labels.
    save_path         : str or Path

    Returns
    -------
    fig : matplotlib Figure
    """
    from scipy.stats import gaussian_kde

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)

    rng        = np.random.default_rng(42)
    nolick_idx = np.where(y == 0)[0]
    lick_idx   = np.where(y == 1)[0]

    # Subsample no-lick for scatter (190k points → unreadable)
    nl_scatter = np.sort(
        rng.choice(nolick_idx, size=min(15_000, len(nolick_idx)), replace=False)
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    # ── KDE density contours for no-lick cloud ────────────────────────────
    kde_src = rng.choice(nolick_idx, size=min(5_000, len(nolick_idx)), replace=False)
    try:
        kde = gaussian_kde(Z[kde_src, :2].T, bw_method="silverman")
        x0, x1 = Z[:, 0].min(), Z[:, 0].max()
        y0, y1 = Z[:, 1].min(), Z[:, 1].max()
        xi = np.linspace(x0, x1, 250)
        yi = np.linspace(y0, y1, 250)
        xx, yy = np.meshgrid(xi, yi)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xi, yi, zz, levels=6, cmap="Blues", alpha=0.20)
        ax.contour( xi, yi, zz, levels=4, colors=["#3B6EA5"], alpha=0.35, linewidths=0.7)
    except Exception:
        pass  # singular matrix or degenerate cluster → skip contours gracefully

    # ── scatter ───────────────────────────────────────────────────────────
    ax.scatter(
        Z[nl_scatter, 0], Z[nl_scatter, 1],
        s=1, alpha=0.20, color="#3B6EA5", rasterized=True,
        label=f"No lick  (n={len(nolick_idx):,})",
    )
    ax.scatter(
        Z[lick_idx, 0], Z[lick_idx, 1],
        s=8, alpha=0.60, color="#E05C2A", zorder=5,
        label=f"Lick  (n={len(lick_idx):,})",
    )

    # ── axes & title ──────────────────────────────────────────────────────
    ev = explained_variance
    ax.set_xlabel(
        f"PC1 ({ev[0]*100:.1f}% var)" if ev is not None else "PC1", fontsize=13
    )
    ax.set_ylabel(
        f"PC2 ({ev[1]*100:.1f}% var)" if ev is not None else "PC2", fontsize=13
    )
    ax.set_title(
        "Population Dynamics — Lick vs No-lick\n"
        "Session 1044385384 · 1,153 units · 21 brain areas",
        fontsize=13,
    )
    ax.legend(fontsize=11, markerscale=4, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


def plot_manifold_interactive(Z, y, time_bins=None,
                              save_path="figures/pca_manifold_interactive.html"):
    """
    Standalone 3D interactive HTML (Plotly). PC1/PC2/PC3, colored by lick label.
    Subsamples no-lick bins to keep the file under ~20 MB while preserving all
    lick events. include_plotlyjs=True makes the file self-contained.

    Parameters
    ----------
    Z          : np.ndarray (n_bins, ≥3)  PCA scores.
    y          : np.ndarray (n_bins,)     Binary lick labels.
    time_bins  : np.ndarray (n_bins,), optional  Bin indices for hover tooltip.
    save_path  : str or Path

    Returns
    -------
    fig : plotly Figure
    """
    import plotly.graph_objects as go

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)

    rng        = np.random.default_rng(42)
    nolick_idx = np.where(y == 0)[0]
    lick_idx   = np.where(y == 1)[0]

    nl_sub = np.sort(
        rng.choice(nolick_idx, size=min(10_000, len(nolick_idx)), replace=False)
    )
    idx  = np.concatenate([nl_sub, lick_idx])
    Z_p  = Z[idx, :3]
    y_p  = y[idx]
    tb   = time_bins[idx] if time_bins is not None else idx

    colors = np.where(y_p == 1, "#E05C2A", "#3B6EA5").tolist()
    sizes  = np.where(y_p == 1, 4, 2).tolist()
    opacs  = np.where(y_p == 1, 0.85, 0.25).tolist()
    hover  = [
        f"Bin {int(b)}<br>{'Lick' if v else 'No lick'}"
        for b, v in zip(tb, y_p)
    ]

    # Split into two traces so each class gets its own opacity/size
    nl_mask = y_p == 0
    lk_mask = y_p == 1

    traces = [
        go.Scatter3d(
            x=Z_p[nl_mask, 0], y=Z_p[nl_mask, 1], z=Z_p[nl_mask, 2],
            mode="markers",
            marker=dict(size=2, color="#3B6EA5", opacity=0.25),
            text=[h for h, m in zip(hover, nl_mask) if m],
            hovertemplate="%{text}<extra></extra>",
            name=f"No lick (n={len(nolick_idx):,})",
        ),
        go.Scatter3d(
            x=Z_p[lk_mask, 0], y=Z_p[lk_mask, 1], z=Z_p[lk_mask, 2],
            mode="markers",
            marker=dict(size=4, color="#E05C2A", opacity=0.85),
            text=[h for h, m in zip(hover, lk_mask) if m],
            hovertemplate="%{text}<extra></extra>",
            name=f"Lick (n={len(lick_idx):,})",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            "3D Neural Population Manifold — Session 1044385384<br>"
            "<sup>1,153 units · 21 brain areas · "
            "rotate / zoom / hover to explore</sup>"
        ),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="rgba(250,250,250,0.95)",
        ),
        legend=dict(itemsizing="constant", font=dict(size=12)),
        margin=dict(l=0, r=0, b=0, t=80),
        width=950, height=720,
    )

    html = fig.to_html(include_plotlyjs=True, full_html=True)
    save_path.write_text(html)
    print(f"Saved: {save_path}")
    return fig


if __name__ == "__main__":
    import sys
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    print("Loading X_raw and y ...", flush=True)
    X = np.load(repo / "data/session_1044385384_X_raw.npy")
    y = np.load(repo / "data/session_1044385384_y.npy")
    print(f"  X_raw: {X.shape}  lick bins: {int(y.sum())}", flush=True)

    print("Smoothing (σ=2 bins = 100 ms) ...", flush=True)
    X_sm = gaussian_filter1d(X.astype(np.float32), sigma=2, axis=0)

    print("Fitting PCA(n_components=3, svd_solver='randomized') ...", flush=True)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=42)
    Z   = pca.fit_transform(X_sm)
    evr = pca.explained_variance_ratio_
    print(f"  EVR: {[f'{v*100:.2f}%' for v in evr]}", flush=True)

    out = repo / "figures"
    plot_manifold_static(
        Z, y,
        explained_variance=evr,
        save_path=out / "pca_manifold.png",
    )
    plot_manifold_interactive(
        Z, y,
        time_bins=np.arange(len(y)),
        save_path=out / "pca_manifold_interactive.html",
    )
    print("Done.", flush=True)
