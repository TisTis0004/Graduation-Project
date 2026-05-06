"""
Temporal Voting Evaluation Script
==================================
Loads the best checkpoint, runs inference PER-RECORDING (preserving temporal order),
then applies majority voting across consecutive windows to eliminate false positives.

Seizures last 30-120+ seconds. If only 1 out of 5 consecutive windows says "seizure",
it's likely a false positive. This script fixes that.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from scipy.ndimage import median_filter
from tqdm import tqdm

from helper.T import EEGToSpectrogram
from helper.models import Spectrogram_CNN_LSTM 


# =========================================================
# CONFIG — Change these to match your best checkpoint
# =========================================================
# Try each of your best checkpoints to see which benefits most from voting
CHECKPOINT_PATH = "checkpoints/eegnet_1d_best.pt"
EVAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"
MODEL_CLASS = 4  # Match the model that created the checkpoint
NUM_CLASSES = 2
VOTING_WINDOW_SIZES = [1, 3, 5, 7, 9]  # Test multiple voting windows


def load_model(checkpoint_path, model_class, device):
    """Load model from checkpoint."""
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model
    
def build_eegnet(device):
    """
    Build EEGNet optimized for 10-second EEG windows at 256Hz.
    
    Architecture rationale:
    - F1=16: 16 temporal filters — captures diverse frequency patterns
    - D=2: depth multiplier 2 — learns 32 spatial filters (2 per temporal filter)
    - F2=32: 32 pointwise filters = F1 * D — separable combination
    - kernel_length=128: 0.5s at 256Hz — captures full alpha/theta cycles
    - drop_prob=0.5: aggressive dropout — EEGNet is small, needs strong regularization
    """
    model = EEGNet(
        n_chans=N_CHANS,
        n_outputs=NUM_CLASSES,
        n_times=N_TIMES,
        final_conv_length="auto",
        pool_mode="mean",
        F1=16,
        D=2,
        F2=32,
        kernel_length=128,
        drop_prob=0.5,
    )
    return model.to(device)

def load_eval_manifest(manifest_path):
    """Load list of recording files from manifest."""
    files = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                files.append((Path(obj["pt_path"]), int(obj["n"])))
    return files


@torch.no_grad()
def predict_one_recording(model, pt_path, transform, device, batch_size=64):
    """
    Run inference on ALL windows from a single recording.
    Returns predictions, probabilities, and true labels IN TEMPORAL ORDER.
    """
    data = torch.load(pt_path, map_location="cpu")
    x_all = data["x"]  # [N, C, T]
    y_all = data["y"].long()  # [N]
    
    all_probs = []
    all_preds = []
    
    # Process in batches
    n = x_all.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x_all[start:end].to(device)
        
        # Apply spectrogram transform
        x_batch = transform(x_batch)
        
        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(x_batch)
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        all_probs.append(probs)
        all_preds.append(preds)
    
    all_probs = np.concatenate(all_probs, axis=0)  # [N, 2]
    all_preds = np.concatenate(all_preds, axis=0)   # [N]
    y_true = y_all.numpy()                           # [N]
    
    return y_true, all_preds, all_probs


def apply_temporal_voting(preds, probs, window_size):
    """
    Apply temporal smoothing to predictions from a single recording.
    
    Strategy: Median filter the seizure probability, then re-threshold.
    This smooths out isolated false positives while preserving sustained seizure detections.
    """
    if window_size <= 1:
        return preds, probs
    
    # Smooth the seizure probability (column 1)
    seizure_prob = probs[:, 1].copy()
    smoothed_prob = median_filter(seizure_prob, size=window_size, mode='nearest')
    
    # Rebuild the probability array
    smoothed_probs = np.stack([1 - smoothed_prob, smoothed_prob], axis=1)
    
    return None, smoothed_probs  # Preds will be determined by threshold search


def find_best_threshold(y_true, probs):
    """Sweep thresholds to find the one that maximizes F1-macro."""
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.81, 0.02):
        y_pred_t = (probs[:, 1] >= thresh).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average="macro", zero_division=0
        )
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_with_voting(model, eval_files, transform, device, voting_window=1):
    """
    Full evaluation pipeline:
    1. Run inference per-recording (preserving temporal order)
    2. Apply temporal voting within each recording
    3. Concatenate all results and compute metrics
    """
    all_y_true = []
    all_probs = []
    
    for pt_path, n_windows in tqdm(eval_files, desc=f"Eval (vote={voting_window})"):
        y_true, raw_preds, raw_probs = predict_one_recording(
            model, pt_path, transform, device
        )
        
        # Apply temporal voting within this recording
        _, smoothed_probs = apply_temporal_voting(raw_preds, raw_probs, voting_window)
        
        all_y_true.append(y_true)
        all_probs.append(smoothed_probs)
    
    # Concatenate all recordings
    all_y_true = np.concatenate(all_y_true)
    all_probs = np.concatenate(all_probs)
    
    # Find best threshold
    best_thresh, _ = find_best_threshold(all_y_true, all_probs)
    y_pred = (all_probs[:, 1] >= best_thresh).astype(int)
    
    # Compute metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, y_pred, average="macro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(all_y_true, y_pred)
    cm = confusion_matrix(all_y_true, y_pred)
    
    try:
        auc = roc_auc_score(all_y_true, all_probs[:, 1])
    except:
        auc = float('nan')
    
    return {
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "threshold": best_thresh,
        "confusion_matrix": cm,
        "n_samples": len(all_y_true),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH, MODEL_CLASS, device)
    
    # Setup transform (eval mode — no augmentation)
    transform = EEGToSpectrogram().to(device)
    transform.eval()
    
    # Load eval manifest
    eval_files = load_eval_manifest(EVAL_MANIFEST)
    print(f"Eval recordings: {len(eval_files)}")
    
    # === Run evaluation with different voting windows ===
    print("\n" + "=" * 70)
    print("TEMPORAL VOTING COMPARISON")
    print("=" * 70)
    
    results = {}
    for window_size in VOTING_WINDOW_SIZES:
        metrics = evaluate_with_voting(model, eval_files, transform, device, window_size)
        results[window_size] = metrics
        
        label = "No voting" if window_size == 1 else f"Voting window={window_size}"
        print(f"\n--- {label} ---")
        print(f"  F1-macro:     {metrics['f1_macro']:.4f}")
        print(f"  Precision:    {metrics['precision_macro']:.4f}")
        print(f"  Recall:       {metrics['recall_macro']:.4f}")
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"  AUC:          {metrics['auc']:.4f}")
        print(f"  Threshold:    {metrics['threshold']:.2f}")
        print(f"  Confusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Window':<12} {'F1-macro':<12} {'Bal-Acc':<12} {'AUC':<12} {'Thresh':<12}")
    print("-" * 60)
    for ws, m in results.items():
        label = "None" if ws == 1 else str(ws)
        print(f"{label:<12} {m['f1_macro']:<12.4f} {m['balanced_accuracy']:<12.4f} {m['auc']:<12.4f} {m['threshold']:<12.2f}")
    
    # Find best
    best_ws = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_f1 = results[best_ws]['f1_macro']
    baseline_f1 = results[1]['f1_macro']
    improvement = best_f1 - baseline_f1
    
    print(f"\n🏆 Best: voting_window={best_ws} → F1={best_f1:.4f} (+{improvement:.4f} over baseline)")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
