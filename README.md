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
This script:

Scans the TUH EEG directory structure

Collects paths to .edf and .csv files

Generates two JSON files:

One JSON for all patients

One JSON for seizure-only patients

These JSON files act as lightweight metadata and do not load EEG data into memory.

Step 2 — Cache EEG Windows (.pt files)
Run:

bash
Copy code
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

Step 3 — Build Efficient Manifest File
Run:

bash
Copy code
python manifest_efficient.py
This script:

Reads all cached .pt files

Generates manifest.jsonl

Adds an n field to each entry

json
Copy code
{"pt_path": "data/cache_windows/sample.pt", "n": 742}
Where:

pt_path → path to the cached tensor file

n → number of EEG windows stored in that file

This significantly improves disk I/O speed and prevents RAM overflow.

Step 4 — Train the Baseline Model
Open the training notebook:

text
Copy code
baseline.ipynb
Then:

Run all cells

This step:

Loads cached EEG windows using an efficient streaming DataLoader

Trains a Tiny 1D CNN for seizure vs background classification

Uses GPU automatically if available

Saves model checkpoints and training logs

✅ Execution Summary
text
Copy code
1. Download TUH EEG dataset
2. python dataset.py
3. python cache_window.py
4. python manifest_efficient.py
5. Run baseline.ipynb (Run All)
⚠️ Important Notes
Always run commands from the project root directory

EDF reading is CPU-based (expected behavior)

Training uses GPU automatically if available

Cached .pt files should be stored on SSD for best performance

Do not enable random window-level shuffling (causes slow disk access)
