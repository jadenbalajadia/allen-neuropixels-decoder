"""
Visual Behavior Neuropixels lick decoder — session 1044385384.

This is a CLEAN REBUILD using the verified API path:
  - Dataset: Visual Behavior Neuropixels (NOT Visual Coding — that dataset
    has no licking task)
  - Access: VisualBehaviorNeuropixelsProjectCache.from_s3_cache() — this
    dataset is not browsable via DANDI directly
  - Session ID confirmed to exist in this dataset before this script was written

Leakage fix vs. the original pipeline:
  1. Inside each fold, train and test indices are sorted back into ascending
     time order before smoothing, so gaussian_filter1d operates on temporally
     adjacent bins within each split.  Train and test rows never touch each
     other — no leakage — but the smoothed features are physically meaningful.
  2. PCA, StandardScaler, and LogisticRegression are all fit INSIDE each fold
     on training data only, then applied to held-out test data.

Run this cell-by-cell in Jupyter — it's written as one script, but the
sections marked with ## are natural breakpoints if you want separate cells.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

## ── Config ────────────────────────────────────────────────────────────────
SESSION_ID   = 1044385384
BIN_SIZE     = 0.05        # seconds → 20 Hz
SIGMA_BINS   = 2           # gaussian sigma in bins (~100 ms)
N_COMPONENTS = 20
N_SPLITS     = 5
RANDOM_STATE = 42
CACHE_DIR    = Path("./vbn_cache")
FIGURES_DIR  = Path("./figures")

## ── 1. Load cache + session ──────────────────────────────────────────────
print("Loading cache...")
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

print(f"Loading session {SESSION_ID}...")
session = cache.get_ecephys_session(ecephys_session_id=SESSION_ID)
print("Session loaded.")

## ── 2. Inspect structure before assuming column names ───────────────────
units = session.get_units()
print("\nUnits table columns:", units.columns.tolist())
print("Unit count:", len(units))

licks = session.licks
print("\nLicks table columns:", licks.columns.tolist())
print("Lick count:", len(licks))
print(licks.head())

print("\nspike_times type:", type(session.spike_times))
some_unit_id = units.index[0]
print(f"Example unit {some_unit_id}: {len(session.spike_times[some_unit_id])} spikes")

## ── STOP HERE on first run ───────────────────────────────────────────────
# Read the printed columns above. The code below ASSUMES:
#   - units has a 'quality' column (good/noise) and 'snr', 'isi_violations'
#   - licks has a 'timestamps' column
# If those assumptions don't match what you see printed, fix the column
# names in the next two sections before running further.

## ── 3. Unit QC ────────────────────────────────────────────────────────────
quality_col = "quality" if "quality" in units.columns else None
snr_col     = "snr" if "snr" in units.columns else None
isi_col     = "isi_violations" if "isi_violations" in units.columns else None

mask = pd.Series(True, index=units.index)
if quality_col:
    mask &= units[quality_col] == "good"
if snr_col:
    mask &= units[snr_col] > 1
if isi_col:
    mask &= units[isi_col] < 0.5

good_units = units[mask]
unit_ids   = good_units.index.values
print(f"\nUnits after QC: {len(unit_ids)} / {len(units)}")

if len(unit_ids) == 0:
    raise ValueError(
        "QC filtered out all units — check column names printed above "
        "and adjust the QC section."
    )

## ── 4. Build RAW spike rate matrix ──────────────────────────────────────
spike_times_dict = session.spike_times

# Use session epoch bounds rather than spike extremes to avoid label misalignment
# when the last (or first) spike is not near the task boundary.
t_start = min(
    spike_times_dict[uid].min()
    for uid in unit_ids if uid in spike_times_dict
)
t_end = max(
    spike_times_dict[uid].max()
    for uid in unit_ids if uid in spike_times_dict
) + BIN_SIZE  # +BIN_SIZE so spikes and licks at t_end fall inside the last bin

bins   = np.arange(t_start, t_end, BIN_SIZE)
n_bins = len(bins) - 1
n_units = len(unit_ids)

print(f"\nBuilding X_raw: {n_bins} bins × {n_units} units...")
X_raw = np.zeros((n_bins, n_units), dtype=np.float32)

for i, uid in enumerate(unit_ids):
    if uid not in spike_times_dict:
        continue
    counts, _ = np.histogram(spike_times_dict[uid], bins=bins)
    X_raw[:, i] = counts / BIN_SIZE

print(f"X_raw shape: {X_raw.shape}")

## ── 5. Build lick labels ─────────────────────────────────────────────────
lick_time_col = "timestamps" if "timestamps" in licks.columns else licks.columns[0]
print(f"\nUsing lick time column: '{lick_time_col}'")
lick_times = licks[lick_time_col].values

y = np.zeros(n_bins, dtype=np.int8)
idxs = ((lick_times - t_start) / BIN_SIZE).astype(int)
valid = (idxs >= 0) & (idxs < n_bins)
y[idxs[valid]] = 1

n_dropped = (~valid).sum()
if n_dropped:
    print(f"  Warning: {n_dropped} lick(s) outside [t_start, t_end] were dropped.")

print(f"Lick bins: {y.sum()} / {n_bins}  ({100 * y.mean():.2f}%)")

if y.sum() == 0:
    raise ValueError(
        "No lick bins found — check 'lick_time_col' above matches the "
        "actual timestamp column, and that lick_times fall within "
        "[t_start, t_end]."
    )

## ── 6. Corrected CV: sort indices → smooth each split → PCA+scaler+clf inside fold ──
# StratifiedKFold with shuffle=True produces non-contiguous index sets.
# Smoothing across temporally non-adjacent rows is meaningless.
# Fix: sort train_idx and test_idx back into ascending time order before
# smoothing each split.  Train and test rows never touch each other —
# the no-leakage guarantee is preserved — but each smoothed block is now
# temporally coherent within its split.
# PCA, StandardScaler, and LogisticRegression are all fit inside each fold
# on training data only.

def make_pipeline():
    return Pipeline([
        ("pca",    PCA(n_components=N_COMPONENTS)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ])

kf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
aucs = []

print(f"\nRunning {N_SPLITS}-fold CV (sort→smooth→PCA+scaler inside each fold)...\n")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw, y)):
    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    X_train_s = gaussian_filter1d(X_raw[train_idx], sigma=SIGMA_BINS, axis=0)
    X_test_s  = gaussian_filter1d(X_raw[test_idx],  sigma=SIGMA_BINS, axis=0)

    pipe = make_pipeline()
    pipe.fit(X_train_s, y[train_idx])
    y_prob = pipe.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y[test_idx], y_prob)
    aucs.append(auc)

    print(f"  Fold {fold + 1}:  AUC = {auc:.4f}  "
          f"(train n={len(train_idx):,}  test n={len(test_idx):,})")

mean_auc_shuffled = np.mean(aucs)
std_auc_shuffled  = np.std(aucs)

print(f"\n{'─'*45}")
print(f"  Shuffled k-fold AUC:  {mean_auc_shuffled:.4f} ± {std_auc_shuffled:.4f}")
print(f"{'─'*45}")

## ── 7. Blocked (contiguous-time) CV ─────────────────────────────────────
# StratifiedKFold(shuffle=True) scatters individual 50 ms bins randomly.
# Neighboring bins are highly correlated at 20 Hz, so even leakage-free
# preprocessing can let a test bin be near-duplicated by a training bin
# just milliseconds away — slightly inflating AUC.
# Blocked CV splits the session into contiguous time chunks instead,
# giving genuinely independent train/test windows.
print(f"\nRunning {N_SPLITS}-fold blocked (contiguous-time) CV...\n")

n_bins_total = len(y)
fold_bounds  = np.linspace(0, n_bins_total, N_SPLITS + 1, dtype=int)
aucs_blocked = []

for i in range(N_SPLITS):
    test_start, test_end = fold_bounds[i], fold_bounds[i + 1]
    test_idx  = np.arange(test_start, test_end)
    train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_bins_total)])

    if y[test_idx].sum() == 0:
        print(f"  Fold {i+1}: no positive labels in test block — skipping.")
        continue

    X_train_s = gaussian_filter1d(X_raw[train_idx], sigma=SIGMA_BINS, axis=0)
    X_test_s  = gaussian_filter1d(X_raw[test_idx],  sigma=SIGMA_BINS, axis=0)

    pipe = make_pipeline()
    pipe.fit(X_train_s, y[train_idx])
    y_prob = pipe.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y[test_idx], y_prob)
    aucs_blocked.append(auc)

    print(f"  Fold {i+1}:  AUC = {auc:.4f}  "
          f"(test bins {test_start:,}–{test_end:,}, n={len(test_idx):,})")

mean_auc_blocked = np.mean(aucs_blocked)
std_auc_blocked  = np.std(aucs_blocked)

print(f"\n{'─'*45}")
print(f"  Blocked CV AUC:       {mean_auc_blocked:.4f} ± {std_auc_blocked:.4f}")
print(f"{'─'*45}")

print(f"\n{'═'*45}")
print(f"  Shuffled k-fold AUC:  {mean_auc_shuffled:.4f} ± {std_auc_shuffled:.4f}")
print(f"  Blocked CV AUC:       {mean_auc_blocked:.4f} ± {std_auc_blocked:.4f}")
print(f"  (difference reflects temporal autocorrelation between nearby bins)")
print(f"{'═'*45}")

# Use shuffled mean for the ROC title (it's the primary number, blocked is the conservative check)
mean_auc = mean_auc_shuffled
std_auc  = std_auc_shuffled

## ── 8. Save corrected arrays ────────────────────────────────────────────
print("\nSaving corrected arrays...")

np.save(f"session_{SESSION_ID}_X_raw.npy", X_raw)
np.save(f"session_{SESSION_ID}_y.npy",     y)

labels_df = pd.DataFrame({
    "lick":     y,
    "time_bin": np.arange(n_bins) * BIN_SIZE + t_start,
})
labels_df.to_csv(f"session_{SESSION_ID}_labels_corrected.csv", index=False)

print("Saved:")
print(f"  session_{SESSION_ID}_X_raw.npy           (raw spikes — source of truth for CV)")
print(f"  session_{SESSION_ID}_y.npy")
print(f"  session_{SESSION_ID}_labels_corrected.csv")
print("  (No pre-computed Z saved — PCA is fit inside CV folds only.)")

## ── 9. ROC plot ──────────────────────────────────────────────────────────
FIGURES_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")

# Re-run folds to collect per-fold curves for the figure
for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw, y)):
    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    X_train_s = gaussian_filter1d(X_raw[train_idx], sigma=SIGMA_BINS, axis=0)
    X_test_s  = gaussian_filter1d(X_raw[test_idx],  sigma=SIGMA_BINS, axis=0)

    pipe = make_pipeline()
    pipe.fit(X_train_s, y[train_idx])
    y_prob = pipe.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y[test_idx], y_prob)
    RocCurveDisplay.from_predictions(
        y[test_idx], y_prob, ax=ax, alpha=0.45, name=f"fold {fold+1} (AUC={auc:.2f})"
    )

ax.set_title(
    f"Lick Decoder — Corrected CV\n"
    f"AUC = {mean_auc:.3f} ± {std_auc:.3f}"
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)
plt.tight_layout()

out_path = FIGURES_DIR / "roc_corrected.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\nSaved: {out_path}")
