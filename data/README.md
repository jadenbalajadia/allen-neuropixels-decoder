# Data

Raw electrophysiology data is not tracked in this repo (files are too large for GitHub).

## Pre-extracted arrays

The processed `.npy` arrays used in the notebooks are available via Google Drive:
[Google Drive folder](https://drive.google.com/drive/folders/1RAW7LVEXndBrboPt53YSQgfGjjS9JFeQ?usp=drive_link)

Place downloaded files in this `data/` directory before running notebooks locally.

| File | Description |
|---|---|
| `session_1044385384_Z.npy` | PCA-reduced population activity (T × 3) |
| `session_1044385384_X_proc.npy` | Processed firing rate matrix (T × 745) |
| `session_1044385384_y.npy` | Lick behavior labels (T,) |
| `session_1044385384_labels.csv` | Trial metadata |
| `session_1044385384_filtered_units_with_regions.csv` | Unit metadata + brain regions (745 × 36) |

## Re-extracting from scratch

To re-run the full pipeline from raw data, use the AllenSDK:

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
session = cache.get_session_data(1044385384)
```

See `notebooks/02_spike_extraction.ipynb` for the full extraction pipeline.
