"""
Train 1D EEGNet (Raw Waveform)
================================
Feeds raw EEG waveforms directly into braindecode's EEGNet for binary
seizure vs background classification. No spectrogram transform needed.

Augmentations used (all defined in this file):
  - EEG1DAugmentation: Channel dropout, amplitude scaling, Gaussian noise, time shift
  - MixUp: Light waveform blending (alpha=0.05)

Usage:  python train_eegnet.py
"""

import os
import copy
import time
import random

import numpy as np
import torch
import torch.nn as nn
from braindecode.util import set_random_seeds
from braindecode.models import EEGNet
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from data.dataloaderV2 import Loader
from data.dataloader import Loader as OriginalLoader
from helper.train_helper import (
    FocalLoss,
    compute_classification_metrics,
    save_history_to_csv,
    get_current_lr,
    is_better,
    build_log_row,
    build_epoch_message,
)


# =========================================================
# CONFIG
# =========================================================
NUM_CLASSES = 2
EPOCHS = 50
LR = 8e-4
PATIENCE = 50
MONITOR = "f1_macro"
CHECKPOINT_PATH = "checkpoints/eegnet_1d_best.pt"
HISTORY_CSV_PATH = "assets/eegnet_1d_best.csv"
SEED = 42

TRAIN_MANIFEST = "cache_windows_binary_10_sec/manifest.jsonl"
VAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"

N_CHANS = 18
N_TIMES = 2560  # 10s × 256Hz
BATCH_SIZE = 64


# =========================================================
# AUGMENTATION — 1D Raw Waveform
# =========================================================
class EEG1DAugmentation(nn.Module):
    """
    Progressive Curriculum Augmentation for raw EEG waveforms.
    NOTE : IDIDNT TEST THIS SiNCE IT WAS AN IDEA BUT THE MAIN AUGG WAS HEAVY FROM START TO END
    IF U SAW THE DATA IS NOT LEARNING U CAN USE LIGHTER AUGG OR TURN THIS OFF IF U WANT 

    Phase 1 (epochs 0-20):  Heavy augmentation — forces generalization
    Phase 2 (epochs 20-40): Light augmentation — fine-tune
    Phase 3 (epochs 40+):   No augmentation — let SWA settle on clean data

    Techniques:
      1. Channel dropout   — randomly zeros entire channels (simulates electrode loss)
      2. Amplitude scaling  — global gain variation (simulates impedance changes)
      3. Gaussian noise     — additive noise proportional to channel std
      4. Random time shift  — small temporal jitter (simulates alignment variation)
    """
    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, x):
        """x: [B, C, T]"""
        # Phase 3: No augmentation
        if not self.training or self.current_epoch >= 40:
            return x

        B, C, T = x.shape
        device = x.device

        # Determine phase multipliers
        if self.current_epoch < 20:
            p_mult = 2.0       # double probability
            noise_g = 0.04     # more noise
        else:
            p_mult = 1.0       # base probability
            noise_g = 0.02     # light noise

        # 1. Channel dropout
        ch_p = 0.02 * p_mult
        ch_mask = (torch.rand(B, C, 1, device=device) > ch_p).float()
        x = x * ch_mask

        # 2. Global amplitude scaling
        if random.random() < (0.25 * p_mult):
            lo = 0.85 if p_mult > 1 else 0.9
            hi = 1.15 if p_mult > 1 else 1.1
            scale = torch.empty(B, 1, 1, device=device).uniform_(lo, hi)
            x = x * scale

        # 3. Additive Gaussian noise
        if random.random() < (0.2 * p_mult):
            noise_std = x.std(dim=-1, keepdim=True).clamp(min=1e-8) * noise_g
            noise = torch.randn_like(x) * noise_std
            x = x + noise

        # 4. Random time shift
        if random.random() < (0.2 * p_mult):
            shift = random.randint(int(-32 * p_mult), int(32 * p_mult))
            x = torch.roll(x, shifts=shift, dims=-1)

        return x


def mixup_data(x, y, alpha=0.05):
    """
    Very light MixUp for raw EEG waveforms.
    alpha=0.05 means lambda ≈ 0.95–1.0 most of the time.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# =========================================================
# MODEL
# =========================================================
def build_eegnet(device, weights=None):
    """
    EEGNet optimized for 10s EEG @ 256Hz.
    F1=16, D=2, F2=32, kernel_length=128 (0.5s), drop_prob=0.5
    """
    model = EEGNet(
        n_chans=N_CHANS,
        n_outputs=NUM_CLASSES,
        n_times=N_TIMES,
        final_conv_length="auto",
        pool_mode="mean",
        F1=16, D=2, F2=32,
        kernel_length=128,
        drop_prob=0.5,
    )
    if weights:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    return model.to(device)


# =========================================================
# TRAINING LOOP
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, augment, use_amp=True):
    model.train()
    augment.train()
    total_loss, total_samples = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, leave=False, desc="Train")
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        # Apply 1D augmentation
        x = augment(x)

        # MixUp on augmented waveforms
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.05)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

        pbar.set_postfix({
            "loss": f"{total_loss/total_samples:.4f}",
            "acc": f"{(torch.cat(all_preds)==torch.cat(all_targets)).float().mean().item():.4f}",
            "lr": f"{get_current_lr(optimizer):.2e}",
        })

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, NUM_CLASSES, topk=2)
    metrics["loss"] = total_loss / total_samples
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, desc="Eval"):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []

    pbar = tqdm(loader, leave=False, desc=desc)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).long()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_targets.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # Optimal threshold search for imbalanced eval
    best_f1, best_thresh = -1, 0.5
    for thresh in np.arange(0.30, 0.71, 0.02):
        y_pred_t = (y_prob[:, 1] >= thresh).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(y_true, y_pred_t, average="macro", zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, thresh

    y_pred = (y_prob[:, 1] >= best_thresh).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, NUM_CLASSES, topk=2)
    metrics["loss"] = total_loss / total_samples
    metrics["best_threshold"] = best_thresh
    return metrics


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using device: {device}")
    set_random_seeds(seed=SEED, cuda=(device.type == "cuda"))

    model = build_eegnet(device, weights=None)
    augment = EEG1DAugmentation().to(device)
    criterion = FocalLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("Creating loaders...")
    train_loader = Loader(ds=TRAIN_MANIFEST, batch_size=BATCH_SIZE, transform=None).return_Loader()
    val_loader = OriginalLoader(ds=VAL_MANIFEST, transform=None).return_Loader()

    best_metric, best_epoch, patience_counter = None, -1, 0
    history = []

    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        augment.set_epoch(epoch)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, augment, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, desc="Val")

        current_metric = val_metrics["f1_macro"]
        scheduler.step(current_metric)
        epoch_time = time.time() - epoch_start

        log_row = build_log_row(epoch, optimizer, train_metrics, val_metrics, epoch_time, topk=2)
        history.append(log_row)
        save_history_to_csv(history, HISTORY_CSV_PATH)

        print(build_epoch_message(epoch, EPOCHS, log_row, train_metrics, val_metrics, topk=2))
        print("Val Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        if is_better(current_metric, best_metric, mode="max"):
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_metric": best_metric,
                "monitor": MONITOR,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved best checkpoint at epoch {epoch+1} with {MONITOR}={current_metric:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} with {ckpt['monitor']}={ckpt['best_metric']:.6f}")

    total_time = time.time() - start_time
    print(f"Best epoch: {best_epoch}")
    print(f"Training completed in {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
