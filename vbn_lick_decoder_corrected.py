"""
Visual Behavior Neuropixels lick decoder — session 1044385384.

This is a CLEAN REBUILD using the verified API path:
  - Dataset: Visual Behavior Neuropixels (NOT Visual Coding — that dataset
    has no licking task)
  - Access: VisualBehaviorNeuropixelsProjectCache.from_s3_cache() — this
    dataset is not browsable via DANDI directly
  - Session ID confirmed to exist in this dataset before this script was written

Leakage fix vs. the original pipeline:
  1. NO smoothing or PCA is applied before the CV split.
  2. Smoothing (gaussian_filter1d) and PCA are both fit INSIDE each fold,
     on the training data only, then applied to test.

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
from sklearn.metrics import roc_auc_score

from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

## ── Config ────────────────────────────────────────────────────────────────
SESSION_ID   = 1044385384
BIN_SIZE     = 0.05        # seconds → 20 Hz
SIGMA_BINS   = 2           # gaussian sigma in bins (~100 ms) — adjust if you
                           # know the original script used a different value
N_COMPONENTS = 20
N_SPLITS     = 5
RANDOM_STATE = 42
LEAKY_AUC    = 0.92        # reference number from the original (leaky) run
CACHE_DIR    = Path("./vbn_cache")

## ── 1. Load cache + session ──────────────────────────────────────────────
print("Loading cache...")
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

print(f"Loading session {SESSION_ID}...")
session = cache.get_ecephys_session(ecephys_session_id=SESSION_ID)
print("Session loaded.")

## ── 2. Inspect structure before assuming column names ───────────────────
# This step exists because we got burned guessing column names earlier in
# this project. Don't skip it — read the printed output before continuing.

units = session.get_units()
print("\nUnits table columns:", units.columns.tolist())
print("Unit count:", len(units))

licks = session.licks
print("\nLicks table columns:", licks.columns.tolist())
print("Lick count:", len(licks))
print(licks.head())

# spike_times is expected to be a dict: unit_id -> array of spike timestamps
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

## ── 4. Build RAW spike rate matrix — NO smoothing here ─────────────────
spike_times_dict = session.spike_times

all_spikes = np.concatenate([spike_times_dict[uid] for uid in unit_ids if uid in spike_times_dict])
t_start = all_spikes.min()
t_end   = all_spikes.max()
bins    = np.arange(t_start, t_end, BIN_SIZE)
n_bins  = len(bins) - 1
n_units = len(unit_ids)

print(f"\nBuilding X_raw: {n_bins} bins × {n_units} units...")
X_raw = np.zeros((n_bins, n_units), dtype=np.float32)

for i, uid in enumerate(unit_ids):
    if uid in spike_times_dict:
        counts, _ = np.histogram(spike_times_dict[uid], bins=bins)
        X_raw[:, i] = counts / BIN_SIZE

print(f"X_raw shape: {X_raw.shape}")

## ── 5. Build lick labels ─────────────────────────────────────────────────
lick_time_col = "timestamps" if "timestamps" in licks.columns else licks.columns[0]
print(f"\nUsing lick time column: '{lick_time_col}'")
lick_times = licks[lick_time_col].values

y = np.zeros(n_bins, dtype=np.int8)
for lt in lick_times:
    idx = int((lt - t_start) / BIN_SIZE)
    if 0 <= idx < n_bins:
        y[idx] = 1

print(f"Lick bins: {y.sum()} / {n_bins}  ({100 * y.mean():.2f}%)")

if y.sum() == 0:
    raise ValueError(
        "No lick bins found — check 'lick_time_col' above matches the "
        "actual timestamp column, and that lick_times fall within "
        "[t_start, t_end]."
    )

## ── 6. Corrected CV: smoothing + PCA INSIDE each fold ──────────────────
kf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)

aucs = []
print(f"\nRunning {N_SPLITS}-fold CV (smooth + PCA inside each fold)...\n")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw, y)):
    X_train_raw = X_raw[train_idx]
    X_test_raw  = X_raw[test_idx]

    X_train_s = gaussian_filter1d(X_train_raw, sigma=SIGMA_BINS, axis=0)
    X_test_s  = gaussian_filter1d(X_test_raw,  sigma=SIGMA_BINS, axis=0)

    pca     = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train_s)
    Z_test  = pca.transform(X_test_s)

    clf.fit(Z_train, y[train_idx])
    y_prob = clf.predict_proba(Z_test)[:, 1]
    auc    = roc_auc_score(y[test_idx], y_prob)
    aucs.append(auc)

    print(f"  Fold {fold + 1}:  AUC = {auc:.4f}  "
          f"(train n={len(train_idx):,}  test n={len(test_idx):,})")

mean_auc = np.mean(aucs)
std_auc  = np.std(aucs)
delta    = mean_auc - LEAKY_AUC

print(f"\n{'─'*45}")
print(f"  Corrected AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  Leaky AUC:      {LEAKY_AUC:.4f}  (previous run)")
print(f"  Delta:          {delta:+.4f}")
print(f"{'─'*45}")

if abs(delta) < 0.02:
    print("  → Result is robust: leakage had minimal impact.")
elif delta < 0:
    print("  → AUC dropped: some of the previous signal was leakage.")
else:
    print("  → AUC held or improved (unexpected — double-check config).")

## ── 7. Save corrected arrays ────────────────────────────────────────────
print("\nSaving corrected arrays...")

X_smooth_full = gaussian_filter1d(X_raw, sigma=SIGMA_BINS, axis=0)
pca_viz       = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
Z_full        = pca_viz.fit_transform(X_smooth_full)

np.save(f"session_{SESSION_ID}_X_raw.npy",       X_raw)
np.save(f"session_{SESSION_ID}_Z_corrected.npy", Z_full)
np.save(f"session_{SESSION_ID}_y.npy",            y)

labels_df = pd.DataFrame({
    "lick":     y,
    "time_bin": np.arange(n_bins) * BIN_SIZE + t_start,
})
labels_df.to_csv(f"session_{SESSION_ID}_labels_corrected.csv", index=False)

print("Saved:")
print(f"  session_{SESSION_ID}_X_raw.npy         (raw spikes, new source of truth)")
print(f"  session_{SESSION_ID}_Z_corrected.npy   (PCA viewer only — not for CV)")
print(f"  session_{SESSION_ID}_y.npy")
print(f"  session_{SESSION_ID}_labels_corrected.csv")

## ── 8. ROC plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_title(
    f"Lick Decoder — Corrected CV\n"
    f"AUC = {mean_auc:.3f} ± {std_auc:.3f}  (leaky ref: {LEAKY_AUC:.2f})"
)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.text(0.6, 0.1, f"Δ = {delta:+.3f}", transform=ax.transAxes, fontsize=10)
plt.tight_layout()
plt.savefig("roc_corrected.png", dpi=150)
plt.show()
print("\nSaved: roc_corrected.png")
