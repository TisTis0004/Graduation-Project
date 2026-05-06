"""
Train 2D Spectrogram CNN-LSTM
==============================
Converts raw EEG → Mel Spectrogram, then feeds it to a CNN-LSTM with
Temporal Attention Pooling for binary seizure vs background classification.

Augmentations used (all defined in this file):
  - SpectrogramAugmentation: Frequency masking, time masking, gain jitter
  - MixUp: Spectrogram blending (alpha=0.1)

Usage:  python train_spectrogram.py

To change the model architecture, modify build_model() in helper/train_helper.py
To change hyperparameters (LR, EPOCHS, etc.), edit the CONFIG section in helper/train_helper.py
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from braindecode.util import set_random_seeds
from helper.T import EEGToSpectrogram

from helper.train_helper import (
    CHECKPOINT_PATH,
    EPOCHS,
    HISTORY_CSV_PATH,
    MONITOR,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    build_epoch_message,
    build_loaders,
    build_log_row,
    build_model,
    build_training_components,
    evaluate,
    get_monitored_metric,
    is_better,
    make_checkpoint,
    save_history_to_csv,
    train_one_epoch,
)


# =========================================================
# AUGMENTATION — 2D Spectrogram (SpecAugment-style)
# =========================================================
class SpectrogramAugmentation(nn.Module):
    """
    SpecAugment-style augmentation for Mel spectrograms [B, C, F, T].
    Applied AFTER EEGToSpectrogram, BEFORE the CNN-LSTM.

    Techniques:
      1. Frequency Masking  — zeros random frequency bands (forces model to
                              not rely on a single frequency range)
      2. Time Masking       — zeros random time columns (forces temporal
                              robustness, simulates brief signal dropout)
      3. Gain Jitter        — per-channel amplitude scaling (simulates
                              electrode impedance variation across channels)

    These are the standard augmentations from Google's SpecAugment paper
    (Park et al., 2019), adapted for multi-channel EEG spectrograms.
    """
    def __init__(self, freq_mask_max=5, time_mask_max=8, gain_range=(0.85, 1.15)):
        super().__init__()
        self.freq_mask_max = freq_mask_max   # Max frequency bins to mask
        self.time_mask_max = time_mask_max   # Max time frames to mask
        self.gain_lo = gain_range[0]
        self.gain_hi = gain_range[1]

    def forward(self, x):
        """x: [B, C, F, T] — multi-channel Mel spectrogram"""
        if not self.training:
            return x

        B, C, F, T = x.shape
        device = x.device

        # 1. Frequency Masking (zero out a random band of frequencies)
        if random.random() < 0.5:
            f_width = random.randint(1, min(self.freq_mask_max, F - 1))
            f_start = random.randint(0, F - f_width)
            x = x.clone()
            x[:, :, f_start:f_start + f_width, :] = 0.0

        # 2. Time Masking (zero out a random stretch of time)
        if random.random() < 0.5:
            t_width = random.randint(1, min(self.time_mask_max, T - 1))
            t_start = random.randint(0, T - t_width)
            x = x.clone()
            x[:, :, :, t_start:t_start + t_width] = 0.0

        # 3. Gain Jitter (per-channel amplitude scaling)
        if random.random() < 0.3:
            gain = torch.empty(B, C, 1, 1, device=device).uniform_(self.gain_lo, self.gain_hi)
            x = x * gain

        return x


def mixup_data(x, y, alpha=0.1):
    """
    Spectrogram MixUp: blend two spectrograms and their labels.

    alpha=0.1 creates gentle blends (lambda ≈ 0.9–1.0).
    Forces the model to learn graded confidence instead of binary decisions.
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
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    mode = "min" if MONITOR == "val_loss" else "max"
    topk = min(2, NUM_CLASSES)

    print(f"Using device: {device}")
    print(f"AMP enabled: {use_amp}")

    set_random_seeds(seed=SEED, cuda=(device.type == "cuda"))

    model = build_model(device, weights=None)  # Change model in helper/train_helper.py
    criterion, optimizer, scheduler, scaler = build_training_components(model, device)

    # Transforms & Augmentation
    eeg_to_spec = EEGToSpectrogram().to(device)
    spec_augment = SpectrogramAugmentation().to(device)

    print('Creating loaders...')
    train_loader, val_loader = build_loaders(transform=None)

    best_metric = None
    best_epoch = -1
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---- TRAIN ----
        model.train()
        spec_augment.train()

        from helper.train_helper import get_current_lr, compute_classification_metrics
        from tqdm import tqdm

        total_loss, total_samples = 0.0, 0
        all_targets, all_preds, all_probs = [], [], []

        pbar = tqdm(train_loader, leave=False, desc="Train")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).long()

            # Step 1: EEG → Spectrogram
            with torch.no_grad():
                x = eeg_to_spec(x)

            # Step 2: SpecAugment (frequency mask, time mask, gain jitter)
            x = spec_augment(x)

            # Step 3: MixUp on the augmented spectrogram
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.1)

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

        train_metrics = compute_classification_metrics(y_true, y_pred, y_prob, NUM_CLASSES, topk)
        train_metrics["loss"] = total_loss / total_samples

        # ---- EVAL (no augmentation, uses the helper's evaluate with threshold search) ----
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            transform=eeg_to_spec,
            use_amp=use_amp,
            num_classes=NUM_CLASSES,
            topk=topk,
            desc="Val",
        )

        current_metric = get_monitored_metric(val_metrics)
        scheduler.step(current_metric)

        epoch_time = time.time() - epoch_start

        log_row = build_log_row(
            epoch=epoch,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch_time=epoch_time,
            topk=topk,
        )

        history.append(log_row)
        save_history_to_csv(history, HISTORY_CSV_PATH)

        print(build_epoch_message(epoch, EPOCHS, log_row, train_metrics, val_metrics, topk))
        print("Val Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        if is_better(current_metric, best_metric, mode=mode):
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = make_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                history=history,
                use_amp=use_amp,
            )

            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved best checkpoint at epoch {epoch + 1} with {MONITOR}={current_metric:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"with {checkpoint['monitor']}={checkpoint['best_metric']:.6f}"
        )

    total_time = time.time() - start_time
    print(f"Best epoch: {best_epoch}")
    print(f"Training completed in {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
