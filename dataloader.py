import numpy as np
import mne
import torch
from typing import Dict, Any

from __future__ import annotations

import csv
from pathlib import Path
import torch


# ---- label parsing ----
SEIZURE_TOKENS = {
    "seiz",
    "sz",
    "seizure",
    "fnsz",
    "gnsz",
    "spsz",
    "cpsz",
    "absz",
    "tnsz",
    "tcsz",
    "mysz",
}


def _is_seizure_label(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(tok in t for tok in SEIZURE_TOKENS)


def label_from_csv_bi(csv_bi_path: str | Path) -> int:
    """
    Returns:
      1 if any row looks like a seizure event
      0 otherwise
    """
    p = Path(csv_bi_path)
    if not p.exists():
        return 0

    # TUH csv_bi is small; just scan rows.
    with p.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # skip obvious headers
            joined = " ".join(row).lower()
            if "start" in joined and "stop" in joined and "label" in joined:
                continue

            # try common places where label might appear (often last col)
            cand_cols = [row[-1]] + row  # include last + all
            if any(_is_seizure_label(c) for c in cand_cols):
                return 1

    return 0


def load_edf_window_all_channels(
    edf_path: str,
    fs: int,
    window_sec: float,
    C_max: int,
) -> torch.Tensor:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick("eeg")
    # picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    # if len(picks) > 0:
    #     raw.pick(picks)

    if int(raw.info["sfreq"]) != fs:
        raw = raw.copy().resample(fs, npad="auto")

    T = int(fs * window_sec)

    data = raw.get_data(start=0, stop=T)  # [C_i, T_i] (T_i can be < T)
    data = data.astype(np.float32)

    # normalize per channel (avoid div by 0)
    data = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-6
    )

    C_i, T_i = data.shape

    # pad/truncate channels + time
    x = np.zeros((C_max, T), dtype=np.float32)
    c = min(C_i, C_max)
    t = min(T_i, T)
    x[:c, :t] = data[:c, :t]

    return torch.from_numpy(x)


def collate_edf_all_channels(
    batch: list[Dict[str, Any]],
    fs: int = 250,
    window_sec: float = 10.0,
    C_max: int = 41,  # choose something safe; we'll also show how to auto-find it below
) -> Dict[str, Any]:
    xs, ys = [], []

    for item in batch:
        ys.append(label_from_csv_bi(item["csv_bi_path"]))
        xs.append(
            load_edf_window_all_channels(
                item["edf_path"],
                fs=fs,
                window_sec=window_sec,
                C_max=41,
            )
        )

    x = torch.stack(xs, dim=0)  # [B, C_max, T]
    y = torch.tensor(ys, dtype=torch.long)

    return {"x": x, "y": y, "meta": batch}


# ====== Test =====
import torch
from torch.utils.data import DataLoader

# --- C) collate function test ---
# --- C) collate function test (UPDATED for real EDF) ---
fs = 250
window_sec = 10.0
C_max = 41
T = int(fs * window_sec)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=lambda b: collate_edf_all_channels(
        b, fs=fs, window_sec=window_sec, C_max=C_max
    ),
)
