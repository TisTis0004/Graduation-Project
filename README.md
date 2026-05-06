# 🧠 EEG Seizure Detection — Binary Classification Pipeline

A streamlined deep learning pipeline for detecting seizures in raw EEG recordings using the **TUH EEG Seizure Dataset (TUSZ)**. The system trains two complementary model architectures and combines them via an ensemble for maximum accuracy.

---

## 📂 Project Structure

```
Graduation-Project/
├── train_eegnet.py              # Train 1D EEGNet on raw EEG waveforms
├── train_spectrogram.py         # Train 2D CNN-LSTM on Mel spectrograms
├── README.md
│
├── data/                        # Data pipeline
│   ├── dataset.py               # Step 1: Scan TUH dataset → JSON metadata
│   ├── cache_window_binary_banana.py  # Step 2: EDF → cached .pt windows + manifest
│   ├── dataloader.py            # Sequential DataLoader (for eval)
│   ├── dataloaderV2.py          # Balanced-sampling DataLoader (for training)
│   └── minfest_effient.py       # Manifest builder utility
│
├── helper/                      # Shared logic (models, losses, transforms)
│   ├── models.py                # All architectures (CNN-LSTM, ResNet18-LSTM, etc.)
│   ├── train_helper.py          # Training loop utilities, metrics, configs
│   ├── loss.py                  # FocalLoss, LDAMLoss
│   └── T.py                    # EEGToSpectrogram (Mel spectrogram transform)
│
├── evaluation/                  # Evaluation scripts
│   ├── eval.py                  # Single-model evaluation
│   ├── evaluate_ensemble.py     # Ensemble evaluation (1D + 2D combined)
│   └── evaluate_with_voting.py  # Temporal voting post-processing
│
├── assets/                      # Training history CSVs, metadata JSONs
└── checkpoints/                 # Saved model weights (.pt)
```

---

## ▶️ Full Pipeline (Step-by-Step)

### Step 0 — Download the Dataset

Download the **TUH EEG Seizure Dataset (TUSZ)** from:
- https://isip.piconepress.com/projects/tuh_eeg/

Extract it to a local directory.

---

### Step 1 — Generate Dataset Metadata

```bash
python data/dataset.py
```

**What this does:**
- Scans the TUH EEG directory structure.
- Collects paths to `.edf` and `.csv` annotation files.
- Outputs JSON metadata files into `assets/` (e.g., `eeg_seizure_only.json`).
- Does **not** load any EEG data into memory — just lightweight path indexing.

---

### Step 2 — Cache EEG Windows + Generate Manifest

```bash
python data/cache_window_binary_banana.py
```

> **Run this TWICE** — once for train, once for eval.  
> Uncomment the appropriate config block at the bottom of the file.

**What this does:**
1. Loads raw EDF recordings using MNE.
2. Applies the **Double Banana Bipolar Montage** (18 channels) — subtracts adjacent electrodes for common-mode noise cancellation.
3. Converts from Volts → Microvolts.
4. Applies **Robust Normalization** (Median + IQR) — resistant to electrode pop artifacts.
5. Extracts 10-second windows with 5-second stride.
6. Labels each window as `seizure` or `non_seizure` (requires 40% seizure overlap to count).
7. Saves `.pt` tensor files and auto-generates `manifest.jsonl`.

> *If you get path errors, the manifest paths are relative to where you run the script. Run from the project root.*

---

### Step 3 — Downsample Background (Balance the Dataset)

```bash
python data/downsample_background.py
```

**Why this is needed:**
The TUH dataset is massively imbalanced — background windows outnumber seizure windows by 10–20x. Training on this raw ratio would cause the model to predict "background" for everything and still get 90% accuracy while missing every seizure.

**What this does:**
- Reads the cached `manifest.jsonl` from Step 2.
- Counts seizure vs. background windows across all `.pt` files.
- Randomly keeps only `bg_multiplier × seizure_count` background windows (default: 2x).
- Saves the downsampled data to a **new folder** (does not overwrite the original cache).
- Generates a new `manifest.jsonl` for the balanced dataset.

> Edit the `bg_multiplier` in the `__main__` block to control the ratio (e.g., 2x, 3x, 5x).

---

### Step 4 — Train

You have **two independent training scripts** — each is self-contained with its own augmentation pipeline:

#### Option A: Train 1D EEGNet (raw waveforms)
```bash
python train_eegnet.py
```

| Component | Details |
|-----------|---------|
| **Model** | braindecode EEGNet (F1=16, D=2, F2=32, kernel=128) |
| **Input** | Raw 1D EEG `[B, 18, 2560]` |
| **Augmentation** | Channel dropout, amplitude scaling, Gaussian noise, time shift (progressive curriculum) |
| **MixUp** | α=0.05 (very light) |
| **Loss** | FocalLoss with label smoothing (0.05) |
| **Config** | Edit the `CONFIG` section at the top of the file |

#### Option B: Train 2D CNN-LSTM (spectrograms)
```bash
python train_spectrogram.py
```

| Component | Details |
|-----------|---------|
| **Model** | CNN-LSTM with InstanceNorm + Temporal Attention Pooling |
| **Input** | Mel spectrogram `[B, 18, F, T]` (via `EEGToSpectrogram`) |
| **Augmentation** | SpecAugment (frequency masking, time masking, gain jitter) |
| **MixUp** | α=0.1 (gentle blending) |
| **Loss** | FocalLoss with label smoothing (0.05) |
| **Config** | Edit `helper/train_helper.py` (CONFIG section + `build_model()`) |

Both scripts save checkpoints to `checkpoints/` and training logs to `assets/`.

---

### Step 5 — Evaluate

```bash
# Single model evaluation:
python evaluation/eval.py

# Ensemble (combines 1D EEGNet + 2D CNN-LSTM predictions):
python evaluation/evaluate_ensemble.py

# Temporal voting (smooths predictions across consecutive windows):
python evaluation/evaluate_with_voting.py
```

**Why ensembling works:**
- The 1D model captures **temporal waveform morphology** (sharp waves, spikes).
- The 2D model captures **spectral patterns** (frequency × time).
- Averaging their probabilities cancels out individual model noise.

---

## ✅ Quick Start Summary

```
1. Download TUH EEG dataset
2. python data/dataset.py
3. python data/cache_window_binary_banana.py   (train set)
4. python data/cache_window_binary_banana.py   (eval set — uncomment eval config)
5. python data/downsample_background.py        (balance train set)
6. python train_eegnet.py                      (1D model)
7. python train_spectrogram.py                 (2D model)
8. python evaluation/evaluate_ensemble.py      (ensemble eval)
```

---

## ⚠️ Important Notes

- Always run commands from the **project root directory**.
- EDF reading is CPU-based (expected behavior).
- Training uses the **GPU automatically** if available (with AMP mixed precision).
- Cached `.pt` files should be stored on an **SSD** for optimal training speed.
- All model architectures live in `helper/models.py`.
- All shared training logic (metrics, loaders, loss) lives in `helper/train_helper.py`.
