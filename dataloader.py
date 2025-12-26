from __future__ import annotations

# =========================
# Imports
# =========================
import json
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Seizure label logic
# =========================
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
      1 if any seizure label found
      0 otherwise
    """
    p = Path(csv_bi_path)
    if not p.exists():
        return 0

    with p.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            joined = " ".join(row).lower()
            if "start" in joined and "stop" in joined and "label" in joined:
                continue

            # try last column + all columns
            candidates = [row[-1]] + row
            if any(_is_seizure_label(c) for c in candidates):
                return 1

    return 0


# =========================
# EDF loading
# =========================
def load_edf_window_all_channels(
    edf_path: str,
    fs: int,
    window_sec: float,
    C_max: int,
) -> torch.Tensor:

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick("eeg")

    if int(raw.info["sfreq"]) != fs:
        raw = raw.copy().resample(fs, npad="auto")

    T = int(fs * window_sec)

    data = raw.get_data(start=0, stop=T).astype(np.float32)

    # per-channel normalization
    data = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-6
    )

    C_i, T_i = data.shape

    x = np.zeros((C_max, T), dtype=np.float32)
    c = min(C_i, C_max)
    t = min(T_i, T)
    x[:c, :t] = data[:c, :t]

    return torch.from_numpy(x)


# =========================
# Collate function
# =========================
def collate_edf_all_channels(
    batch: List[Dict[str, Any]],
    fs: int = 250,
    window_sec: float = 10.0,
    C_max: int = 41,
) -> Dict[str, Any]:

    xs, ys = [], []

    for item in batch:
        ys.append(label_from_csv_bi(item["csv_bi_path"]))
        xs.append(
            load_edf_window_all_channels(
                item["edf_path"],
                fs=fs,
                window_sec=window_sec,
                C_max=C_max,
            )
        )

    x = torch.stack(xs, dim=0)  # [B, C, T]
    y = torch.tensor(ys, dtype=torch.long)

    return {
        "x": x,
        "y": y,
        "meta": batch,
    }


# =========================
# Dataset (JSON-based)
# =========================
class TUHZJsonDataset(Dataset):
    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as f:
            self.records: List[Dict[str, Any]] = json.load(f)

        if len(self.records) == 0:
            raise RuntimeError("JSON is empty")

        required = {"edf_path", "csv_path", "csv_bi_path"}
        missing = required - set(self.records[0].keys())
        if missing:
            raise KeyError(f"Missing keys in JSON: {missing}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "edf_path": rec["edf_path"],
            "csv_path": rec["csv_path"],
            "csv_bi_path": rec["csv_bi_path"],
            "stem": rec.get("stem"),
            "subject": rec.get("subject"),
            "session": rec.get("session"),
            "montage": rec.get("montage"),
        }


class TUHZDataloaderBinary:
    def __init__(self, json_path, fs=250, window_sec=10.0, C_max=41):
        dataset = TUHZJsonDataset(json_path)

        self.fs = fs
        self.window_sec = window_sec
        self.C_max = C_max

        self.loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda b: collate_edf_all_channels(
                b, fs=self.fs, window_sec=self.window_sec, C_max=self.C_max
            ),
        )

    def return_loader(self):
        return self.loader


# =========================
# Test DataLoader
# =========================
if __name__ == "__main__":

    json_path = "eeg_seizure_only.json"  # <-- update path
    dataset = TUHZJsonDataset(json_path)

    fs = 250
    window_sec = 10.0
    C_max = 41

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

    batch = next(iter(loader))
    print("x shape:", batch["x"].shape)  # [B, C_max, T]
    print("y shape:", batch["y"].shape)  # [B]
    print("batch keys:", batch.keys())
