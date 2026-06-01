# 🧠 EEG Seizure Detection — Paper Evaluation Branch

This branch contains the streamlined, clean codebase specifically tailored for reproducing the core results of our paper on EEG Seizure Detection. It includes the final model architecture (EEGNet) and the exact evaluation pipeline used to generate the paper's metrics and plots.

All experimental, developmental, and unused code has been stripped away to provide a clear and easy-to-use repository for researchers and reviewers.

---

## 📂 Project Structure

```
Graduation-Project/
├── run_paper_evaluations.bat    # Main entry point to run all paper evaluations
├── eval_single_model.py         # Evaluation script that calculates metrics and plots graphs
├── train_eegnet.py              # Contains the EEGNet model definition and metrics
├── README.md
│
├── data/                        # Data processing and caching scripts
│   ├── cache_window_unipolar_21.py # Caches 21-channel binary evaluation dataset
│   ├── cache_window_unipolar_41.py # Caches 41-channel multiclass evaluation dataset
│   ├── dataloader.py            # Custom DataLoader logic
│   ├── dataloaderV2.py          # V2 DataLoader logic
│   └── dataset.py               # Metadata parsing logic
│
├── checkpoints/                 # Final trained model weights used in the paper
│   ├── best_model_checkpoint.pt
│   └── eegnet_10sec_full_next60.pt
│
└── assets/                      # Metadata indices required for dataset loading
    ├── eeg_seizure_only_eval.json
    ├── tuh_train_index_eval.json
    ├── eeg_seizure_only.json
    └── tuh_train_index.json
```

---

## 🚀 How to Run the Evaluations

To reproduce the results reported in the paper, simply run the evaluation script:

```bash
run_paper_evaluations.bat
```

### What the script does:
The script automates the complete evaluation process into 4 steps:

1. **21-Channel Dataset Generation**: Runs `data/cache_window_unipolar_21.py` to extract features and cache the 21-channel binary classification evaluation set.
2. **21-Channel Evaluation**: Runs `eval_single_model.py` using the cached data and `checkpoints/eegnet_10sec_full_next60.pt`.
3. **41-Channel Dataset Generation**: Runs `data/cache_window_unipolar_41.py` to cache the 41-channel 9-class multiclass evaluation set.
4. **41-Channel Evaluation**: Runs `eval_single_model.py` using the 41-channel cached data and `checkpoints/best_model_checkpoint.pt`.

---

## 📊 Viewing the Results

As the pipeline runs, it will output key metrics to the console, including:
- **Macro F1-Score**
- **Balanced Accuracy**
- **ROC-AUC Score**

Once the pipeline completes, it will automatically generate high-quality plots in the `assets/` directory (which will be created if it doesn't exist):
- `assets/cm_*.png` (Confusion Matrices for both models)
- `assets/auc_*.png` (AUC curve plots for binary classification)

---

## ⚠️ Important Requirements

- Ensure you have **Python 3.8+** installed along with standard data science packages (`torch`, `numpy`, `pandas`, `mne`, `scikit-learn`, `matplotlib`, `braindecode`).
- Run the script strictly from the **project root directory** to ensure relative paths resolve correctly.
- Ensure the EDF files correspond to the metadata paths present in the `assets/` JSON files, or adjust the paths in the data caching scripts if your local dataset directory differs.
