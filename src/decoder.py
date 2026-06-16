"""
decoder.py
Logistic regression decoder with cross-validation and ROC analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_decoder(C=1.0, max_iter=1000):
    """Return a sklearn Pipeline: StandardScaler + LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter))
    ])


def cross_validate(Z, y, n_splits=5, random_state=42):
    """
    k-fold cross-validation on PCA-reduced activity.

    Returns
    -------
    scores : np.ndarray  accuracy per fold
    """
    clf = build_decoder()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, Z, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}  "
          f"(chance = {max(y.mean(), 1 - y.mean()):.3f})")
    return scores


def plot_roc(Z, y, n_splits=5, random_state=42):
    """Plot mean ROC curve across k folds."""
    clf = build_decoder()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fig, ax = plt.subplots(figsize=(6, 6))
    aucs = []
    for train, test in cv.split(Z, y):
        clf.fit(Z[train], y[train])
        proba = clf.predict_proba(Z[test])[:, 1]
        auc = roc_auc_score(y[test], proba)
        aucs.append(auc)
        RocCurveDisplay.from_predictions(y[test], proba, ax=ax, alpha=0.4,
                                         name=f"fold AUC={auc:.2f}")

    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title(f"ROC — mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    ax.legend(fontsize=8)
    return fig, np.mean(aucs)
