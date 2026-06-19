"""
decoder.py
Logistic regression decoder with cross-validation and ROC analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_pipeline(n_components=20, C=1.0, max_iter=1000):
    """PCA → StandardScaler → LogisticRegression, all fit on training data inside each fold."""
    return Pipeline([
        ("pca",    PCA(n_components=n_components)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=C, max_iter=max_iter)),
    ])


def _fold_smooth_and_fit(X_raw, y, train_idx, test_idx, sigma_bins, n_components):
    """
    Single-fold helper: sort indices into temporal order, smooth each split
    independently, then fit PCA+scaler+clf on train and score on test.

    Sorting before smoothing is critical: StratifiedKFold with shuffle=True
    returns non-contiguous index sets.  Smoothing across non-adjacent rows
    is physically meaningless.  Sorting restores temporal adjacency within
    each split without ever mixing train and test rows.
    """
    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    X_train_s = gaussian_filter1d(X_raw[train_idx], sigma=sigma_bins, axis=0)
    X_test_s  = gaussian_filter1d(X_raw[test_idx],  sigma=sigma_bins, axis=0)

    pipe = build_pipeline(n_components=n_components)
    pipe.fit(X_train_s, y[train_idx])
    proba = pipe.predict_proba(X_test_s)[:, 1]
    auc   = roc_auc_score(y[test_idx], proba)
    return auc, proba, y[test_idx]


def cross_validate(X_raw, y, sigma_bins=2, n_components=20, n_splits=5, random_state=42):
    """
    Stratified k-fold cross-validation on raw spike rates.

    Each fold:
      1. Sort train and test indices into ascending time order.
      2. Smooth each sorted split independently with gaussian_filter1d —
         train and test rows never touch.
      3. Fit PCA, StandardScaler, LogisticRegression on train only.
      4. Score on held-out test fold.

    Parameters
    ----------
    X_raw        : np.ndarray (n_bins, n_units)  Raw (unsmoothed) spike rates.
    y            : np.ndarray (n_bins,)           Binary lick labels.
    sigma_bins   : float  Gaussian smoothing sigma in bins.
    n_components : int    PCA components to retain.
    n_splits     : int    Number of CV folds.
    random_state : int

    Returns
    -------
    aucs : np.ndarray  AUC-ROC per fold.
    """
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for train, test in cv.split(X_raw, y):
        auc, _, _ = _fold_smooth_and_fit(X_raw, y, train, test, sigma_bins, n_components)
        aucs.append(auc)
    aucs = np.array(aucs)
    print(f"CV AUC: {aucs.mean():.3f} ± {aucs.std():.3f}  (chance = 0.5)")
    return aucs


def plot_roc(X_raw, y, sigma_bins=2, n_components=20, n_splits=5, random_state=42):
    """
    Plot per-fold ROC curves and return the mean AUC.

    Parameters
    ----------
    X_raw        : np.ndarray (n_bins, n_units)  Raw (unsmoothed) spike rates.
    y            : np.ndarray (n_bins,)           Binary lick labels.
    sigma_bins   : float
    n_components : int
    n_splits     : int
    random_state : int

    Returns
    -------
    fig      : matplotlib Figure
    mean_auc : float
    """
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fig, ax = plt.subplots(figsize=(6, 6))
    aucs = []
    for fold, (train, test) in enumerate(cv.split(X_raw, y)):
        auc, proba, y_test = _fold_smooth_and_fit(
            X_raw, y, train, test, sigma_bins, n_components
        )
        aucs.append(auc)
        RocCurveDisplay.from_predictions(
            y_test, proba, ax=ax, alpha=0.4, name=f"fold {fold+1} AUC={auc:.2f}"
        )
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title(f"ROC — mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    ax.legend(fontsize=8)
    return fig, float(np.mean(aucs))
