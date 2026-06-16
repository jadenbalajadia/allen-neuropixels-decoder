"""
manifold.py
PCA dimensionality reduction and trajectory visualization helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def fit_pca(X, n_components=3):
    """
    Fit PCA on population firing rate matrix.

    Parameters
    ----------
    X            : np.ndarray (n_timepoints, n_neurons)
    n_components : int

    Returns
    -------
    Z   : np.ndarray (n_timepoints, n_components)  PC scores
    pca : fitted sklearn PCA object
    """
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    print(f"Variance explained: "
          f"{[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    return Z, pca


def plot_trajectory_2d(Z, y=None, n_points=5000, title="Neural Trajectory (PC1 vs PC2)"):
    """Plot 2D PCA trajectory, optionally colored by label."""
    fig, ax = plt.subplots(figsize=(7, 6))
    n = min(n_points, len(Z))
    sc = ax.scatter(Z[:n, 0], Z[:n, 1],
                    c=y[:n] if y is not None else np.arange(n),
                    s=2, alpha=0.5, cmap="plasma")
    plt.colorbar(sc, ax=ax, label="Lick label" if y is not None else "Time bin")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    return fig


def plot_prelick_trajectories(Z, y, window_bins=40, max_events=100):
    """
    Overlay neural trajectories in the window before each lick event.
    """
    lick_bins = np.where(y == 1)[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    for b in lick_bins[:max_events]:
        if b > window_bins:
            traj = Z[b - window_bins:b, :2]
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.2, color="steelblue", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Pre-lick Neural Trajectories (n={min(len(lick_bins), max_events)})")
    return fig
