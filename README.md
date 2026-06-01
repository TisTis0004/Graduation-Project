# MindScope: Robust Seizure Detection

MindScope is a highly robust EEG binary seizure classification framework evaluated under strict clinical realism using the TUSZ v2.0.3 canonical, patient-disjoint splits. It utilizes an 18-channel bipolar longitudinal montage ("double banana" montage) at a standardized 256 Hz sampling rate to ensure maximum clinical relevance and resilience to artifacts.

## Overview

The framework relies on a heterogeneous ensemble combining:
1. **1D Temporal Waveform Branch (EEGNet)**: Excels at isolating sharp, localized transient features such as spike-and-wave discharges directly from raw waveforms.
2. **2D Spectral Dynamics Branch (CNN-LSTM)**: Ingests 40-bin Mel-spectrograms to track longer-term frequency trajectories and rhythmic evolution.

Because these architectures extract orthogonal feature representations, their fusion yields a substantial performance leap when processing realistic, noisy, hospital EEG.

## Features

- **18-Channel Bipolar Montage**: Focuses on localized spatial field variations to suppress common-mode artifacts.
- **Robust Normalization**: Subject-Level Robust Median and Interquartile Range (IQR) normalization preserves the high-amplitude voltage characteristics unique to electrographic seizures.
- **Heterogeneous Ensemble**: Fuses 1D temporal and 2D spectral representations.
- **Temporal Post-Processing**: Sliding majority voting window and minimum duration constraints eliminate transient artifact spikes.

## Pipeline Performance

Evaluated strictly on the canonical patient-disjoint test split of the Temple University Hospital Seizure Corpus (TUSZ):

| Architecture | F1-score | AUC-ROC |
|--------------|----------|---------|
| CNN-LSTM (Best 2D) | 0.7543 | 0.8507 |
| EEGNet-1D (Best 1D) | 0.7855 | 0.8796 |
| Weighted Ensemble | 0.8029 | 0.8985 |
| **Final Complete Pipeline** | **0.8108** | **0.9035** |

*(Final Complete Pipeline includes Temporal Voting (w = 5) and Minimum Duration (min_dur = 3))*

## Repository Structure

```
├── core/
│   ├── models.py         # Defines Spectrogram_CNN_LSTM model
│   ├── T.py              # EEGToSpectrogram transformation and augmentations
│   └── train_helper.py   # Training loops, loss functions, and dataset loading
├── data/
│   ├── cache_window_banana.py # Data indexing and cache generation (run this first)
│   ├── dataloaderV2.py   # Balanced buffering dataloaders
│   └── dataset.py        # TUSZ metadata parsing
├── cnn_lstm/             # 2D Spectral Dynamics model
│   ├── train.py          # Training script
│   └── eval.py           # Evaluation script
├── eegnet/               # 1D Temporal Waveform model
│   ├── train.py          # Training script
│   └── eval.py           # Evaluation script
├── ensemble/             # Heterogeneous Ensemble
│   └── eval.py           # Evaluation of the fused system
└── checkpoints/          # Pre-trained model weights
```

## Quick Start

### 1. Data Preparation
Before training or evaluating, you must generate the cache files. Configure the dataset paths and run:
```bash
python data/cache_window_banana.py
```
This will extract 10-second windows (with 50% overlap for evaluation) from the TUSZ `.edf` recordings and save them as PyTorch tensors.

### 2. Evaluation
To evaluate the pre-trained ensemble on the cached data:
```bash
python ensemble/eval.py
```
To evaluate individual models:
```bash
python eegnet/eval.py
python cnn_lstm/eval.py
```

### 3. Training
To train the models from scratch using the cached data:
```bash
python eegnet/train.py
python cnn_lstm/train.py
```

## Weights
Pre-trained weights are provided in the `checkpoints/` directory:
- `cnn_lstm.pt`
- `eegnet.pt`
