# EEG Seizure Detection — Paper Evaluation Branch

This branch contains the streamlined, clean codebase specifically tailored for reproducing the core results of our paper on EEG Seizure Detection. It includes the final model architecture (EEGNet) and the exact evaluation pipeline used to generate the paper's metrics and plots.

All experimental, developmental, and unused code has been stripped away to provide a clear and easy-to-use repository for researchers and reviewers.

---

## Project Structure

```
Graduation-Project/
├── run_paper_evaluations.bat    # Main entry point to run all paper evaluations
├── README.md
│
├── eegnet/                      # 1D Raw EEG Model (EEGNet)
│   ├── train.py                 # Hardcoded training script for EEGNet
│   └── eval.py                  # Hardcoded eval script for EEGNet
│
├── cnn_lstm/                    # 2D Spectrogram Model (CNN-LSTM)
│   ├── train.py                 # Hardcoded training script for CNN-LSTM
│   └── eval.py                  # Hardcoded eval script for CNN-LSTM
│
├── ensemble/                    # Cross-Architecture Ensemble
│   └── eval.py                  # Hardcoded ensemble evaluation script
│
├── core/                        # Core model architectures and shared helpers
│   ├── models.py                # Contains EEGNet and Spectrogram_CNN_LSTM architectures
│   ├── train_helper.py          # Shared training loops
│   └── T.py                     # EEG to Spectrogram transforms
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

## Manual Code Testing (Hardcoded Configs)

The repository has been restructured so you can easily run tests manually without dealing with complex command-line arguments. Every model has its own dedicated folder containing simple Python scripts. 

At the top of each `train.py` and `eval.py` script, there is a **MANUAL CONFIGURATION** block. You can open the file in your editor, change variables like `CHECKPOINT_PATH`, `EPOCHS`, or `LR`, and run the script directly.

### 1. EEGNet
```bash
python eegnet/train.py
python eegnet/eval.py
```

### 2. CNN-LSTM
```bash
python cnn_lstm/train.py
python cnn_lstm/eval.py
```

### 3. Ensemble
```bash
python ensemble/eval.py
```

---

## Paper Evaluation Automated Script

### What the script does:
The script automates the complete evaluation process into 4 steps:

1. **21-Channel Dataset Generation**: Runs `data/cache_window_unipolar_21.py` to extract features and cache the 21-channel binary classification evaluation set.
2. **21-Channel Evaluation**: Runs `eegnet/eval.py` using the cached data.
3. **41-Channel Dataset Generation**: Runs `data/cache_window_unipolar_41.py` to cache the 41-channel 9-class multiclass evaluation set.
4. **41-Channel Evaluation**: Runs `cnn_lstm/eval.py` using the 41-channel cached data.

---

## Viewing the Results

As the pipeline runs, it will output key metrics to the console, including:
- **Macro F1-Score**
- **Balanced Accuracy**
- **ROC-AUC Score**

Once the pipeline completes, it will automatically generate high-quality plots in the `assets/` directory (which will be created if it doesn't exist):
- `assets/cm_*.png` (Confusion Matrices for both models)
- `assets/auc_*.png` (AUC curve plots for binary classification)

---

## Important Requirements

- Ensure you have **Python 3.8+** installed along with standard data science packages (`torch`, `numpy`, `pandas`, `mne`, `scikit-learn`, `matplotlib`, `braindecode`).
- Run the script strictly from the **project root directory** to ensure relative paths resolve correctly.
- Ensure the EDF files correspond to the metadata paths present in the `assets/` JSON files, or adjust the paths in the data caching scripts if your local dataset directory differs.
