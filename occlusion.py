"""
MindScope XAI on the Ensemble Model: Occlusion Analysis
========================================================
Three levels of occlusion, all using the same ensemble:

    Level 1 -- Channel occlusion
        Mute one of the 18 EEG channels (set to zero) at a time.
        Measure how much the ensemble's seizure confidence drops.
        Answers: which channels does the model actually need?

    Level 2 -- Time-segment occlusion
        Mute one 3-second segment at a time (0-3s, 3-6s, 6-9s, 9-10s).
        Answers: which part of the 10-second window matters most?

    Level 3 -- Band occlusion
        Remove one frequency band at a time by notch-filtering it out.
        Answers: which frequency band is most important to the model?

No gradients. No baseline path. No integration.
Just: remove it and see how much the prediction drops.

Output: xai_occlusion_Ensemble/
    channel_occlusion.png
    time_occlusion.png
    band_occlusion.png
    occlusion_summary.json

"""

import os
import sys
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from braindecode.models import EEGNet
sys.path.append(str(Path(__file__).resolve().parent))
from helper.T import EEGToSpectrogram
from helper.models import Spectrogram_CNN_LSTM


DRIVE_DATA_DIR         = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\V2 of 10 sec\cache_windows_binary_10_sec_eval"
MANIFEST_PATH          = os.path.join(DRIVE_DATA_DIR, "manifest.jsonl")
SPECTROGRAM_CHECKPOINT = r"weights\cnn_lstm_melspectrogram_dropout_new4changes.pt"
EEGNET_CHECKPOINT      = r"weights\eegnet_1d_best.pt"

ENSEMBLE_WEIGHT_2D = 0.5
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PER_CLASS      = None   
RANDOM_SEED        = 42

INPUT_CLAMP   = 20.0
N_CHANS       = 18
N_TIMES       = 2560
N_OUTPUTS     = 2
SAMPLING_RATE = 256

EEGNET_CFG = dict(
    n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES,
    final_conv_length="auto", pool_mode="mean",
    F1=16, D=2, F2=32, kernel_length=128, drop_prob=0.5,
)

CHANNEL_NAMES = [
    "fp1-f7", "f7-t3",  "t3-t5",  "t5-o1",
    "fp2-f8", "f8-t4",  "t4-t6",  "t6-o2",
    "fp1-f3", "f3-c3",  "c3-p3",  "p3-o1",
    "fp2-f4", "f4-c4",  "c4-p4",  "p4-o2",
    "fz-cz",  "cz-pz",
]

BRAIN_REGIONS = {
    "Frontal":   [0, 4, 8, 12],
    "Temporal":  [1, 2, 5, 6],
    "Central":   [9, 13, 16],
    "Parietal":  [10, 14, 17],
    "Occipital": [3, 7, 11, 15],
}

TIME_SEGMENTS = {
    "Early\n0-3s":    (0,    768),
    "Mid\n3-6s":      (768,  1536),
    "Mid-Late\n6-9s": (1536, 2304),
    "Late\n9-10s":    (2304, 2560),
}

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

BAND_COLORS = {
    "delta": "#4472C4",
    "theta": "#ED7D31",
    "alpha": "#70AD47",
    "beta":  "#C00000",
    "gamma": "#7030A0",
}

CLASS_COLORS = {"seizure": "#C00000", "background": "#2E75B6"}


# ---------------------------------------------------------------------------
# Model loading  (identical to IG script)
# ---------------------------------------------------------------------------

class EnsembleWrapper(nn.Module):
    def __init__(self, eegnet, cnn_lstm, spec_transform, weight_2d=0.5):
        super().__init__()
        self.eegnet         = eegnet
        self.cnn_lstm       = cnn_lstm
        self.spec_transform = spec_transform
        self.weight_2d      = float(weight_2d)
        self.weight_1d      = 1.0 - float(weight_2d)

    def forward(self, x):
        x_clamped = torch.clamp(x, min=-INPUT_CLAMP, max=INPUT_CLAMP)
        logits_1d = self.eegnet(x_clamped)
        probs_1d  = torch.softmax(logits_1d, dim=1)
        spec      = self.spec_transform(x_clamped)
        logits_2d = self.cnn_lstm(spec)
        probs_2d  = torch.softmax(logits_2d, dim=1)
        return self.weight_2d * probs_2d + self.weight_1d * probs_1d


def _clean_swa(sd):
    return {(k.replace("module.", "") if k.startswith("module.") else k): v
            for k, v in sd.items() if k != "n_averaged"}


def load_ensemble():
    print(f"[device] {DEVICE} | max_per_class={MAX_PER_CLASS}", flush=True)
    eegnet = EEGNet(**EEGNET_CFG).to(DEVICE)
    ck1 = torch.load(EEGNET_CHECKPOINT, map_location=DEVICE, weights_only=False)
    sd1 = ck1.get("model_state_dict", ck1) if isinstance(ck1, dict) else ck1
    if isinstance(ck1, dict) and ck1.get("swa", False):
        sd1 = _clean_swa(sd1)
    eegnet.load_state_dict(sd1)
    eegnet.eval()

    cnn_lstm = Spectrogram_CNN_LSTM().to(DEVICE)
    ck2 = torch.load(SPECTROGRAM_CHECKPOINT, map_location=DEVICE, weights_only=False)
    sd2 = ck2.get("model_state_dict", ck2) if isinstance(ck2, dict) else ck2
    cnn_lstm.load_state_dict(sd2)
    cnn_lstm.eval()

    spec = EEGToSpectrogram().to(DEVICE)
    spec.eval()

    wrapper = EnsembleWrapper(eegnet, cnn_lstm, spec, weight_2d=ENSEMBLE_WEIGHT_2D).to(DEVICE)
    wrapper.eval()
    print("[load] Ensemble ready", flush=True)
    return wrapper


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def safe_load_pt(pt_path):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        shutil.copy2(pt_path, tmp)
        return torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def resolve_pt(p):
    return os.path.join(DRIVE_DATA_DIR, Path(p).name)


def scan_and_sample():
    print("[manifest] reading...", flush=True)
    entries = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    sz_idx, bg_idx = [], []
    t0 = time.time()
    for i, e in enumerate(entries):
        pt = resolve_pt(e["pt_path"])
        if not os.path.exists(pt):
            continue
        data = safe_load_pt(pt)
        if data is None:
            continue
        for w in range(len(data["y"])):
            lbl = int(data["y"][w].item())
            (sz_idx if lbl == 1 else bg_idx).append((pt, w))
        if (i + 1) % 25 == 0:
            print(f"[scan] {i+1}/{len(entries)} files ({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(RANDOM_SEED)
    if MAX_PER_CLASS is not None:
        sz_idx = rng.sample(sz_idx, min(MAX_PER_CLASS, len(sz_idx)))
        bg_idx = rng.sample(bg_idx, min(MAX_PER_CLASS, len(bg_idx)))

    combined = [(p, i, 1) for p, i in sz_idx] + [(p, i, 0) for p, i in bg_idx]
    combined.sort(key=lambda x: x[0])
    print(f"[scan] {len(sz_idx)} seizure + {len(bg_idx)} background = {len(combined)} total", flush=True)
    return combined


# ---------------------------------------------------------------------------
# Occlusion helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_prob(wrapper, x_np):
    """Return seizure probability (class 1) for a numpy array [18, 2560]."""
    t = torch.from_numpy(x_np).unsqueeze(0).float().to(DEVICE)
    return wrapper(t)[0, 1].item()


def bandstop_filter(signal_np, low_hz, high_hz, fs=SAMPLING_RATE, order=4):
    """Zero-phase bandstop (notch) filter to remove a frequency band."""
    nyq  = fs / 2.0
    low  = max(low_hz  / nyq, 1e-4)
    high = min(high_hz / nyq, 1.0 - 1e-4)
    sos  = butter(order, [low, high], btype="bandstop", output="sos")
    return sosfiltfilt(sos, signal_np, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main occlusion loop
# ---------------------------------------------------------------------------

def run_occlusion(wrapper, selected):
    """
    For every window:
      1. Get baseline probability (original signal).
      2. For each channel: zero it out, get new probability, compute drop.
      3. For each time segment: zero it out, get new probability, compute drop.
      4. For each band: notch it out, get new probability, compute drop.
    Accumulate mean drops separately for seizure and background windows.
    """
    n_windows = len(selected)

    # Accumulators: sum of confidence drops per class
    acc = {
        cls: {
            "channel":  np.zeros(N_CHANS,             dtype=np.float64),
            "time":     np.zeros(len(TIME_SEGMENTS),   dtype=np.float64),
            "band":     np.zeros(len(BANDS),           dtype=np.float64),
            "count":    0,
        }
        for cls in ("seizure", "background")
    }

    seg_names  = list(TIME_SEGMENTS.keys())
    band_names = list(BANDS.keys())

    t0 = time.time()
    total = 0

    for pt_path, group in groupby(selected, key=lambda x: x[0]):
        windows = list(group)
        data    = safe_load_pt(pt_path)
        if data is None:
            continue

        for _, w_idx, true_label in windows:
            x_np    = data["x"][w_idx].numpy().astype(np.float32)  # [18, 2560]
            cls     = "seizure" if true_label == 1 else "background"
            p_base  = get_prob(wrapper, x_np)

            # -- Level 1: channel occlusion --
            for ch in range(N_CHANS):
                x_occ      = x_np.copy()
                x_occ[ch]  = 0.0           # mute this channel
                p_occ      = get_prob(wrapper, x_occ)
                drop       = p_base - p_occ  # positive = this channel helped
                acc[cls]["channel"][ch] += drop

            # -- Level 2: time-segment occlusion --
            for t_idx, (seg_name, (s, e)) in enumerate(TIME_SEGMENTS.items()):
                x_occ        = x_np.copy()
                x_occ[:, s:e] = 0.0        # mute this time window across all channels
                p_occ        = get_prob(wrapper, x_occ)
                drop         = p_base - p_occ
                acc[cls]["time"][t_idx] += drop

            # -- Level 3: band occlusion --
            for b_idx, (band_name, (lo, hi)) in enumerate(BANDS.items()):
                x_occ   = bandstop_filter(x_np, lo, hi)  # remove this band
                p_occ   = get_prob(wrapper, x_occ)
                drop    = p_base - p_occ
                acc[cls]["band"][b_idx] += drop

            acc[cls]["count"] += 1
            total += 1

            if total % 50 == 0 or total == n_windows:
                elapsed = time.time() - t0
                rate    = total / max(elapsed, 1e-3)
                eta     = (n_windows - total) / max(rate, 1e-3)
                print(f"[occ] {total}/{n_windows} | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining", flush=True)

    # Average the drops
    results = {}
    for cls in ("seizure", "background"):
        cnt = max(acc[cls]["count"], 1)
        results[cls] = {
            "channel": (acc[cls]["channel"] / cnt).tolist(),
            "time":    (acc[cls]["time"]    / cnt).tolist(),
            "band":    (acc[cls]["band"]    / cnt).tolist(),
            "count":   acc[cls]["count"],
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_channel_occlusion(results, out_dir):
    sz  = np.array(results["seizure"]["channel"])
    bg  = np.array(results["background"]["channel"])

    y   = np.arange(N_CHANS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.suptitle(
        "Ensemble Occlusion: Channel Importance\n"
        "Mean drop in seizure confidence when each channel is zeroed out",
        fontsize=13
    )

    for ax, vals, cls, color in zip(
        axes,
        [sz, bg],
        ["Seizure windows", "Background windows"],
        [CLASS_COLORS["seizure"], CLASS_COLORS["background"]],
    ):
        order = np.argsort(vals)
        ax.barh(y, vals[order], color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([CHANNEL_NAMES[i] for i in order], fontsize=8)
        ax.set_xlabel("Mean confidence drop (higher = more important)", fontsize=9)
        ax.set_title(cls, fontsize=10, color=color)
        ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "channel_occlusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] channel_occlusion.png saved", flush=True)


def plot_time_occlusion(results, out_dir):
    sz   = np.array(results["seizure"]["time"])
    bg   = np.array(results["background"]["time"])
    segs = list(TIME_SEGMENTS.keys())
    x    = np.arange(len(segs))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Ensemble Occlusion: Time Segment Importance\n"
        "Mean drop in seizure confidence when each 3-second segment is zeroed out",
        fontsize=12
    )
    w = 0.35
    ax.bar(x - w/2, sz, width=w, color=CLASS_COLORS["seizure"],   label=f"Seizure (n={results['seizure']['count']})",    alpha=0.85)
    ax.bar(x + w/2, bg, width=w, color=CLASS_COLORS["background"], label=f"Background (n={results['background']['count']})", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(segs, fontsize=9)
    ax.set_ylabel("Mean confidence drop", fontsize=9)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_occlusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] time_occlusion.png saved", flush=True)


def plot_band_occlusion(results, out_dir):
    sz         = np.array(results["seizure"]["band"])
    bg         = np.array(results["background"]["band"])
    band_names = list(BANDS.keys())
    x          = np.arange(len(band_names))
    colors_sz  = [BAND_COLORS[b] for b in band_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Ensemble Occlusion: Frequency Band Importance\n"
        "Mean drop in seizure confidence when each band is notch-filtered out",
        fontsize=12
    )

    for ax, vals, cls in zip(axes, [sz, bg], ["Seizure windows", "Background windows"]):
        bars = ax.bar(x, vals, color=colors_sz, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{b}\n({BANDS[b][0]}-{BANDS[b][1]} Hz)" for b in band_names],
            fontsize=8
        )
        ax.set_ylabel("Mean confidence drop", fontsize=9)
        ax.set_title(cls, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)

        # Annotate bars with values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "band_occlusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] band_occlusion.png saved", flush=True)


def save_summary(results, out_dir):
    band_names = list(BANDS.keys())
    seg_names  = list(TIME_SEGMENTS.keys())

    summary = {
        "method":        "Occlusion (channel / time-segment / band)",
        "ensemble":      "CNN-LSTM-2D + EEGNet-1D",
        "occlusion_value": "zero (channel and time) / bandstop filter (band)",
    }

    for cls in ("seizure", "background"):
        ch_vals   = results[cls]["channel"]
        t_vals    = results[cls]["time"]
        b_vals    = results[cls]["band"]
        top_ch    = CHANNEL_NAMES[int(np.argmax(ch_vals))]
        top_seg   = list(TIME_SEGMENTS.keys())[int(np.argmax(t_vals))]
        top_band  = band_names[int(np.argmax(b_vals))]

        summary[cls] = {
            "n_windows":         results[cls]["count"],
            "most_important_channel":      top_ch,
            "most_important_time_segment": top_seg,
            "most_important_band":         top_band,
            "channel_drops":  {CHANNEL_NAMES[i]: round(v, 6) for i, v in enumerate(ch_vals)},
            "time_drops":     {seg_names[i]: round(v, 6) for i, v in enumerate(t_vals)},
            "band_drops":     {band_names[i]: round(v, 6) for i, v in enumerate(b_vals)},
        }

    with open(os.path.join(out_dir, "occlusion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[save] occlusion_summary.json saved", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    out_dir = "xai_occlusion_Ensemble"
    os.makedirs(out_dir, exist_ok=True)

    wrapper  = load_ensemble()
    selected = scan_and_sample()

    print(f"\n[occ] Starting occlusion on {len(selected)} windows...", flush=True)
    print("[occ] Each window runs 18 channel + 4 time + 5 band forward passes.", flush=True)
    print(f"[occ] Total forward passes: ~{len(selected) * 27:,}", flush=True)
    t_start = time.time()

    results = run_occlusion(wrapper, selected)

    elapsed = time.time() - t_start
    print(f"\n[occ] Done. Total time: {elapsed/60:.1f} min", flush=True)

    print("\n[occ] === QUICK RESULTS ===", flush=True)
    for cls in ("seizure", "background"):
        ch   = results[cls]["channel"]
        t    = results[cls]["time"]
        b    = results[cls]["band"]
        print(f"\n  {cls.upper()} (n={results[cls]['count']})")
        print(f"    Most important channel:      {CHANNEL_NAMES[np.argmax(ch)]}  (drop={max(ch):.4f})")
        print(f"    Most important time segment: {list(TIME_SEGMENTS.keys())[np.argmax(t)]}  (drop={max(t):.4f})")
        print(f"    Most important band:         {list(BANDS.keys())[np.argmax(b)]}  (drop={max(b):.4f})")

    print("\n[plot] Generating plots...", flush=True)
    plot_channel_occlusion(results, out_dir)
    plot_time_occlusion(results, out_dir)
    plot_band_occlusion(results, out_dir)
    save_summary(results, out_dir)

    print(f"\n[done] All outputs saved to {out_dir}/", flush=True)