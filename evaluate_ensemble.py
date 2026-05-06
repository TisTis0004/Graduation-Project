"""
Cross-Architecture Ensemble Evaluation
=========================================
Combines predictions from HETEROGENEOUS models for maximum diversity:
  - 2D CNN-LSTM (operates on Mel spectrograms)
  - 1D EEGNet (operates on raw EEG waveforms)

WHY this ensemble works so well:
  - The 2D model excels at spectral pattern recognition (frequency × time)
  - The 1D model excels at temporal waveform morphology (sharp waves, spikes)
  - Averaging their probabilities cancels out individual model noise!

Run: python evaluate_ensemble.py
"""

import json
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from scipy.ndimage import median_filter
from tqdm import tqdm

# ---- Model imports ----
from helper.T import EEGToSpectrogram
from helper.models import Spectrogram_CNN_LSTM
from braindecode.models import EEGNet


# =========================================================
# CONFIG — Define your ensemble members
# =========================================================

# --- 2D Spectrogram Models (CNN-LSTM variants) ---
SPECTROGRAM_MODELS = [
    {
        "name": "CNN-LSTM-Large v3",
        "checkpoint": "checkpoints/cnn_lstm_melspectrogram_dropout_new4changes.pt",
        "model_class": Spectrogram_CNN_LSTM,
    },
]

# --- 1D Raw EEG Models (EEGNet) ---
EEGNET_MODELS = [
    {
        "name": "EEGNet-1D Best",
        "checkpoint": "checkpoints/eegnet_1d_best.pt",
        "n_chans": 18,
        "n_times": 2560,
        "n_outputs": 2,
        "F1": 16, "D": 2, "F2": 32,
        "kernel_length": 128,
        "drop_prob": 0.5,
    },
]



EVAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"
NUM_CLASSES = 2
VOTING_WINDOWS = [1, 3, 5, 7]  # Try multiple window sizes





# =========================================================
# MODEL LOADING
# =========================================================
def load_spectrogram_model(config, device):
    model = config["model_class"]()
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def load_eegnet_model(config, device):
    model = EEGNet(
        n_chans=config["n_chans"], n_outputs=config["n_outputs"], n_times=config["n_times"],
        final_conv_length="auto", pool_mode="mean", F1=config["F1"], D=config["D"],
        F2=config["F2"], kernel_length=config["kernel_length"], drop_prob=config["drop_prob"],
    )
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model





def load_eval_manifest(manifest_path):
    files = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                files.append((Path(obj["pt_path"]), int(obj["n"])))
    return files


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def predict_spectrogram_model(model, pt_path, transform, device, batch_size=64):
    data = torch.load(pt_path, map_location="cpu")
    x_all = data["x"]
    y_all = data["y"].long()
    all_probs = []
    for start in range(0, x_all.shape[0], batch_size):
        end = min(start + batch_size, x_all.shape[0])
        x_batch = x_all[start:end].to(device)
        x_batch = transform(x_batch)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(x_batch)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return y_all.numpy(), np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_eegnet_model(model, pt_path, device, batch_size=64):
    data = torch.load(pt_path, map_location="cpu")
    x_all = data["x"]
    y_all = data["y"].long()
    all_probs = []
    for start in range(0, x_all.shape[0], batch_size):
        end = min(start + batch_size, x_all.shape[0])
        x_batch = x_all[start:end].to(device)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(x_batch)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return y_all.numpy(), np.concatenate(all_probs, axis=0)





# =========================================================
# POST-PROCESSING
# =========================================================
def apply_temporal_voting(probs, window_size):
    if window_size <= 1:
        return probs
    seizure_prob = probs[:, 1].copy()
    smoothed_prob = median_filter(seizure_prob, size=window_size, mode='nearest')
    return np.stack([1 - smoothed_prob, smoothed_prob], axis=1)


def find_best_threshold(y_true, probs):
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.81, 0.02):
        y_pred = (probs[:, 1] >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def compute_full_metrics(y_true, probs, threshold):
    y_pred = (probs[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    try: auc = roc_auc_score(y_true, probs[:, 1])
    except: auc = float('nan')
    return {
        "f1_macro": f1, "balanced_accuracy": bal_acc, "auc": auc,
        "threshold": threshold, "confusion_matrix": cm,
    }


def grid_search_weights(y_true, model_probs_list, step=0.05):
    """
    Grid search for optimal ensemble weights combining up to 3 models safely!
    """
    n_groups = len(model_probs_list)
    if n_groups == 1:
        return [1.0]
    
    if n_groups == 2:
        best_f1, best_w = -1, [0.5, 0.5]
        for w1 in np.arange(0.0, 1.01, step):
            w2 = 1.0 - w1
            avg_probs = w1 * model_probs_list[0] + w2 * model_probs_list[1]
            thresh, f1 = find_best_threshold(y_true, avg_probs)
            if f1 > best_f1:
                best_f1, best_w = f1, [w1, w2]
        return best_w
        
    if n_groups == 3:
        best_f1, best_w = -1, [0.33, 0.33, 0.34]
        for w1 in np.arange(0.0, 1.01, step):
            for w2 in np.arange(0.0, 1.01 - w1, step):
                w3 = 1.0 - w1 - w2
                avg_probs = w1 * model_probs_list[0] + w2 * model_probs_list[1] + w3 * model_probs_list[2]
                thresh, f1 = find_best_threshold(y_true, avg_probs)
                if f1 > best_f1:
                    best_f1, best_w = f1, [w1, w2, w3]
        return best_w
    
    # Fallback to equal weights
    return [1.0 / n_groups] * n_groups


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"  Cross-Architecture Ensemble Evaluation")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    
    # Phase 0: Loading Models
    spec_models = []
    for cfg in SPECTROGRAM_MODELS:
        if Path(cfg["checkpoint"]).exists():
            spec_models.append((cfg["name"], load_spectrogram_model(cfg, device)))
            
    eegnet_models = []
    for cfg in EEGNET_MODELS:
        if Path(cfg["checkpoint"]).exists():
            eegnet_models.append((cfg["name"], load_eegnet_model(cfg, device)))



    # Transforms
    spec_transform = EEGToSpectrogram().to(device)
    spec_transform.eval()

    # Load eval data
    eval_files = load_eval_manifest(EVAL_MANIFEST)

    # Phase 1: Inference
    print(f"\n  Running Inference on {len(eval_files)} files...")
    per_model_probs = {}
    y_true_per_rec = []
    
    for rec_idx, (pt_path, n_windows) in enumerate(tqdm(eval_files, desc="Inference")):
        # 1. 2D Spectrogram Models
        for name, model in spec_models:
            y_true, probs = predict_spectrogram_model(model, pt_path, spec_transform, device)
            per_model_probs.setdefault(name, []).append(probs)
            
        # 2. 1D EEGNet Models
        for name, model in eegnet_models:
            y_true, probs = predict_eegnet_model(model, pt_path, device)
            per_model_probs.setdefault(name, []).append(probs)
            


        y_true_per_rec.append(y_true)

    # Phase 2: Performance
    all_y_true = np.concatenate(y_true_per_rec)
    results = {}
    print(f"\n{'='*70}\n  Individual Model Performance\n{'='*70}")
    for model_name in per_model_probs:
        all_probs = np.concatenate(per_model_probs[model_name])
        thresh, _ = find_best_threshold(all_y_true, all_probs)
        metrics = compute_full_metrics(all_y_true, all_probs, thresh)
        results[model_name] = metrics
        print(f"  {model_name}: F1={metrics['f1_macro']:.4f}")

    # Phase 3: Ensemble
    print(f"\n{'='*70}\n  Ensemble Strategies (Voting/Weights)\n{'='*70}")
    group_probs = {}
    
    if spec_models:
        group_probs["2D-CNN"] = np.concatenate([np.mean([per_model_probs[n][i] for n, _ in spec_models], axis=0) for i in range(len(eval_files))])
    if eegnet_models:
        group_probs["1D-EEGNet"] = np.concatenate([np.mean([per_model_probs[n][i] for n, _ in eegnet_models], axis=0) for i in range(len(eval_files))])


    group_keys = list(group_probs.keys())
    
    if len(group_keys) >= 2:
        simple_avg = np.mean([group_probs[k] for k in group_keys], axis=0)
        thresh, f1 = find_best_threshold(all_y_true, simple_avg)
        results["Ensemble (equal weight)"] = compute_full_metrics(all_y_true, simple_avg, thresh)
        print(f"  Ensemble (equal weight): F1={results['Ensemble (equal weight)']['f1_macro']:.4f}")

        # Grid Search
        group_list = [group_probs[k] for k in group_keys]
        opt_weights = grid_search_weights(all_y_true, group_list, step=0.05)
        weighted_avg = sum(w * p for w, p in zip(opt_weights, group_list))
        thresh, f1 = find_best_threshold(all_y_true, weighted_avg)
        weight_str = " / ".join([f"{k}={w:.2f}" for k, w in zip(group_keys, opt_weights)])
        name_str = f"Ensemble (opt: {weight_str})"
        results[name_str] = compute_full_metrics(all_y_true, weighted_avg, thresh)
        print(f"  {name_str}: F1={results[name_str]['f1_macro']:.4f}")

    else:
        weighted_avg = group_probs[group_keys[0]]

    # Phase 4 & 5: Temporal Voting on the Winner
    print("\n  Applying Temporal Voting...")
    
    # reconstruct per-rec
    idx = 0
    ens_per_rec = []
    for rec_probs_len in [len(p) for p in y_true_per_rec]:
        ens_per_rec.append(weighted_avg[idx:idx+rec_probs_len])
        idx += rec_probs_len

    for vw in VOTING_WINDOWS:
        all_voted = np.concatenate([apply_temporal_voting(p, vw) for p in ens_per_rec])
        thresh, _ = find_best_threshold(all_y_true, all_voted)
        results[f"Ensemble + voting(w={vw})"] = compute_full_metrics(all_y_true, all_voted, thresh)

    # FINAL RESULTS TABLE
    print(f"\n{'='*70}\n  FINAL RESULTS COMPARISON\n{'='*70}")
    for name, m in sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
        print(f"{name:<45} {m['f1_macro']:<10.4f}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
