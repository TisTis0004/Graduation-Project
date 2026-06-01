import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from data.dataloader import Loader as OriginalLoader
from train_eegnet import compute_metrics, evaluate, FocalLoss
import matplotlib.subplots as plt_sub
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def _draw_cm(ax, cm, title, color, f1_macro, labels_list):
    total = cm.sum()
    norm_m = cm / (total + 1e-8)
    
    cmap = LinearSegmentedColormap.from_list("c", ["#ffffff", color], N=256)
    ax.imshow(norm_m, cmap=cmap, vmin=0, vmax=norm_m.max() * 1.25, aspect="auto")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / (cm[i].sum() + 1e-8)) * 100
            count = int(cm[i, j])
            txt_color = "white" if norm_m[i, j] > norm_m.max() * 0.55 else "#222"
            
            if cm.shape[0] <= 2:
                text = f"{count:,}\n({pct:.1f}%)"
            else:
                text = f"{count:,}"
                
            ax.text(j, i, text, ha="center", va="center",
                    color=txt_color, fontsize=10 if cm.shape[0] > 4 else 12, fontweight="bold")

    ax.set_xticks(range(len(labels_list)))
    ax.set_yticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(labels_list, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    stats = f"F1-macro = {f1_macro:.4f}"
    ax.text(0.5, -0.3, stats,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9", edgecolor="#ccc", linewidth=1.0))

def plot_results(metrics, model_name, labels_list):
    os.makedirs("assets", exist_ok=True)
    
    # 1. Plot Confusion Matrix
    fig_size = (8, 8) if len(labels_list) > 2 else (6, 6)
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    _draw_cm(ax, metrics["confusion_matrix"], f"Confusion Matrix: {model_name}", "#1f77b4", metrics["f1_macro"], labels_list)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    cm_path = f"assets/cm_{model_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved CM plot to {cm_path}")

    # 2. Plot AUC (only if binary)
    if len(labels_list) == 2 and "auc" in metrics:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_color("#bbb")
            
        auc_val = metrics.get("auc", 0.0)
        ax.barh(["AUC"], [1.0], color="#eeeeee", edgecolor="#ccc", height=0.45, zorder=0)
        ax.barh(["AUC"], [auc_val], color="#1f77b4", edgecolor="#1f77b4", height=0.45, zorder=1)
        
        ax.text(auc_val + 0.01, 0, f"AUC = {auc_val:.4f}", va="center", ha="left", fontsize=12, fontweight="bold", color="#1f77b4")
        ax.text(auc_val / 2, 0, f"F1-macro = {metrics['f1_macro']:.4f}", va="center", ha="center", fontsize=10, color="white", fontweight="bold")
        
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("ROC-AUC", fontsize=11)
        ax.set_title(f"AUC: {model_name}", fontsize=12, fontweight="bold", pad=8)
        ax.axvline(0.5, color="#ccc", linewidth=1.0, linestyle="--")
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
            
        plt.tight_layout()
        auc_path = f"assets/auc_{model_name}.png"
        plt.savefig(auc_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [OK] Saved AUC plot to {auc_path}")

import torch.nn as nn
from core.models import EEGNet, CNN_LSTM

def build_eegnet(device, n_chans, n_classes, n_times=2500):
    model = EEGNet(n_chans=n_chans, n_classes=n_classes)
    return model.to(device)

def build_cnn_lstm(device, n_chans, n_classes):
    model = CNN_LSTM(num_channels=n_chans, num_classes=n_classes)
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific EEGNet model")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint")
    parser.add_argument("--manifest", required=True, type=str, help="Path to validation manifest")
    parser.add_argument("--n_chans", required=True, type=int, help="Number of channels")
    parser.add_argument("--n_classes", required=True, type=int, help="Number of classes (e.g. 2 or 9)")
    parser.add_argument("--labels", type=str, help="Comma separated list of labels (e.g. 'bckg,seizure')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(args.ckpt).replace(".pt", "")

    print(f"{'='*50}")
    print(f"  Evaluating Model: {model_name}")
    print(f"{'='*50}")
    print(f"Device:    {device}")
    print(f"Channels:  {args.n_chans}")
    print(f"Classes:   {args.n_classes}")
    print(f"Manifest:  {args.manifest}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}!")
    
    # Extract labels
    if args.labels:
        labels_list = [l.strip() for l in args.labels.split(',')]
    else:
        if args.n_classes == 2:
            labels_list = ["Background", "Seizure"]
        elif args.n_classes == 9:
            labels_list = ["bckg", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "pnsz", "mysz"]
        else:
            labels_list = [f"Class {i}" for i in range(args.n_classes)]

    # Load model
    model = build_eegnet(device, n_chans=args.n_chans, n_classes=args.n_classes)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint) 
        
    model.eval()

    # Load data
    print("\nCreating val loader...")
    val_loader = OriginalLoader(ds=args.manifest, transform=None, batch_size=128, shuffle=False, num_workers=2).return_Loader()

    # Dummy criterion
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)

    # Note: evaluate expects NUM_CLASSES global variable in train_eegnet, we temporarily override it
    import train_eegnet
    train_eegnet.NUM_CLASSES = args.n_classes
    
    print("\nEvaluating...")
    val_metrics = evaluate(model, val_loader, criterion, device, use_amp=(device.type=="cuda"))

    # Print summary
    print("\n" + "="*40)
    print("             SUMMARY")
    print("="*40)
    print(f"Val F1 Macro : {val_metrics.get('f1_macro', 0):.4f}")
    print(f"Val Accuracy : {val_metrics.get('accuracy', 0):.4f}")
    print(f"Val Bal Acc  : {val_metrics.get('balanced_accuracy', 0):.4f}")
    if args.n_classes == 2:
        print(f"Val AUC      : {val_metrics.get('auc', 0.0):.4f}")
    print("="*40)
    
    print("\nGenerating Plots...")
    plot_results(val_metrics, model_name, labels_list)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
