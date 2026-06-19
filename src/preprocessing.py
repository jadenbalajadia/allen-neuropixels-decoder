"""
preprocessing.py
Spike binning, Gaussian smoothing, and AllenSDK loading helpers.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def bin_spikes(spike_times, unit_ids, t_start, t_end, bin_size_ms=10):
    """
    Convert spike times to a (n_units x n_bins) firing rate matrix.

    Parameters
    ----------
    spike_times : dict  {unit_id: np.array of spike times in seconds}
    unit_ids    : list  ordered list of unit IDs to include
    t_start     : float start time in seconds
    t_end       : float end time in seconds
    bin_size_ms : int   bin width in milliseconds

    Returns
    -------
    X : np.ndarray, shape (n_bins, n_units)  spike counts per bin
    t : np.ndarray, shape (n_bins,)           bin center times in seconds
    """
    bin_size_s = bin_size_ms / 1000.0
    bins = np.arange(t_start, t_end, bin_size_s)
    t = bins[:-1] + bin_size_s / 2

    X = np.zeros((len(t), len(unit_ids)))
    for i, uid in enumerate(unit_ids):
        if uid not in spike_times:
            continue
        counts, _ = np.histogram(spike_times[uid], bins=bins)
        X[:, i] = counts / bin_size_s  # convert to Hz

    return X, t


def smooth_spikes(X, sigma_ms=25, bin_size_ms=10):
    """
    Apply Gaussian smoothing along the time axis.

    Parameters
    ----------
    X           : np.ndarray (n_bins, n_units)
    sigma_ms    : float  smoothing kernel sigma in milliseconds
    bin_size_ms : int    bin width in milliseconds (to convert sigma to bins)

    Returns
    -------
    X_smooth : np.ndarray (n_bins, n_units)
    """
    sigma_bins = sigma_ms / bin_size_ms
    return gaussian_filter1d(X, sigma=sigma_bins, axis=0)


def load_arrays(data_dir):
    """
    Load pre-extracted .npy arrays from data_dir.

    Returns X_raw (spike rates, n_bins × n_units), y (lick labels), labels (DataFrame).
    Smooth X_raw with smooth_spikes() before passing to the decoder — do not use
    a pre-smoothed or pre-PCA'd array to avoid data leakage.
    """
    from pathlib import Path
    import pandas as pd

    data_dir = Path(data_dir)
    X_raw  = np.load(data_dir / "session_1044385384_X_raw.npy")
    y      = np.load(data_dir / "session_1044385384_y.npy")
    labels = pd.read_csv(data_dir / "session_1044385384_labels_corrected.csv")
    return X_raw, y, labels
