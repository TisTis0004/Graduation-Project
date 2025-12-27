## ▶️ Data Generation & Training Pipeline

Follow the steps **in order** to reproduce the full pipeline from raw TUH data to model training.

---

### Step 0 — Download the Dataset

Download the **TUH EEG Seizure Dataset (TUHZ)** from the official website:

- https://isip.piconepress.com/projects/tuh_eeg/

Extract the dataset to a local directory.

---

### Step 1 — Generate Dataset Metadata (JSON)

Run:

```bash
python dataset.py
```
This script:

Scans the TUH EEG directory structure

Collects paths to .edf and .csv files

Generates two JSON files:

One JSON for all patients

One JSON for seizure-only patients

These JSON files act as lightweight metadata and do not load EEG data into memory.
Step 2 — Cache EEG Windows (.pt files)

Run:

python cache_window.py


This script:

Loads EEG recordings using MNE (CPU-based)

Extracts fixed-length EEG windows (e.g. 10s windows with stride)

Labels each window as seizure or background using CSV annotations

Saves windows as PyTorch .pt files

Why this step is important:

Reading EDF files during training is very slow

Cached .pt files are much faster to load and manipulate

Enables efficient GPU training
