import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Add root project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import Loader as OriginalLoader
from core.models import Spectrogram_CNN_LSTM

# =========================================================
# MANUAL CONFIGURATION
# Modify these variables to test manually!
# =========================================================
CHECKPOINT_PATH = "../checkpoints/best_model_checkpoint.pt"
MANIFEST_PATH = "../cache_windows_unipolar_41_multiclass_eval/manifest.jsonl"
N_CHANS = 41 # Make sure this matches your cached spectrogram dimension!
N_CLASSES = 9
LABELS = ["bckg", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "pnsz", "mysz"]
# =========================================================

import torch.nn.functional as F
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob=None, num_classes=2):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_macro"] = prec
    metrics["recall_macro"] = rec
    metrics["f1_macro"] = f1
    if y_prob is not None and num_classes == 2:
        try: metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except: metrics["auc"] = np.nan
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, leave=False, desc="Val")
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()
        x = torch.clamp(x, min=-20.0, max=20.0)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_targets.append(y.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    if N_CLASSES == 2:
        best_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.25, 0.76, 0.02):
            y_pred_t = (y_prob[:, 1] >= thresh).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(y_true, y_pred_t, average="macro", zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh
        y_pred = (y_prob[:, 1] >= best_thresh).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob, N_CLASSES)
    metrics["loss"] = total_loss / total_samples
    if N_CLASSES == 2: metrics["best_threshold"] = best_thresh
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*50}")
    print(f"  Evaluating Model: Spectrogram CNN-LSTM")
    print(f"{'='*50}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}!")
        return

    model = Spectrogram_CNN_LSTM(num_channels=N_CHANS, num_classes=N_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint) 
    model.eval()

    print("Creating val loader...")
    val_loader = OriginalLoader(ds=MANIFEST_PATH, transform=None, batch_size=64, shuffle=False, num_workers=2).return_Loader()
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)

    print("Evaluating...")
    val_metrics = evaluate(model, val_loader, criterion, device, use_amp=(device.type=="cuda"))

    print("\n" + "="*40)
    print("             SUMMARY")
    print("="*40)
    print(f"Val F1 Macro : {val_metrics.get('f1_macro', 0):.4f}")
    print(f"Val Accuracy : {val_metrics.get('accuracy', 0):.4f}")
    print(f"Val Bal Acc  : {val_metrics.get('balanced_accuracy', 0):.4f}")
    if N_CLASSES == 2:
        print(f"Val AUC      : {val_metrics.get('auc', 0.0):.4f}")
    print("="*40)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
