from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import mne

# =========================================================
# CONFIGURATION
# =========================================================

CANONICAL_21 = [
    "fp1", "fp2",
    "f7", "f3", "fz", "f4", "f8",
    "t3", "c3", "cz", "c4", "t4",
    "t5", "p3", "pz", "p4", "t6",
    "o1", "o2",
    "a1", "a2",
]

@dataclass
class CacheConfig:
    json_path: str
    out_dir: str

    fs: int = 250
    window_sec: float = 10.0
    stride_sec: float = 5.0

    max_records: Optional[int] = None
    max_windows_per_record: Optional[int] = None

    l_freq: Optional[float] = 0.5
    h_freq: Optional[float] = 40.0

    background_labels: Tuple[str, ...] = ("bckg", "background")
    seizure_label_name: str = "seizure"
    non_seizure_label_name: str = "non_seizure"

    flat_std_thresh: float = 1e-6 
    max_flat_ratio: float = 0.3
    max_zero_ratio: float = 0.3

    per_channel_normalize: bool = True
    clip_percentile: float = 2.0


# =========================================================
# LABELS
# =========================================================

def to_binary_label(raw_label: str, background_labels: Tuple[str, ...]) -> str:
    raw_label = str(raw_label).strip().lower()
    if raw_label in set(background_labels):
        return "non_seizure"
    return "seizure"

def build_binary_label_vocab(
    seizure_label_name: str = "seizure",
    non_seizure_label_name: str = "non_seizure",
):
    labels = [non_seizure_label_name, seizure_label_name]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    return label_to_id, id_to_label


def read_label_intervals_from_csv(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")

    intervals = []
    for _, row in df.iterrows():
        s = row["start_time"]
        e = row["stop_time"]
        lab = row["label"]

        if pd.isna(s) or pd.isna(e) or pd.isna(lab) or e <= s:
            continue

        intervals.append({
            "start_time": float(s),
            "stop_time": float(e),
            "label": lab,
        })

    return intervals

def assign_raw_label_by_overlap(
    ws: float,
    we: float,
    intervals: List[Dict[str, Any]],
    default: str = "bckg",
    seizure_priority_threshold: float = 0.4, 
) -> str:
    overlap_by_label: Dict[str, float] = {}
    window_duration = we - ws

    for item in intervals:
        s = item["start_time"]
        e = item["stop_time"]
        lab = item["label"].lower()

        overlap = max(0.0, min(we, e) - max(ws, s))
        if overlap > 0:
            overlap_by_label[lab] = overlap_by_label.get(lab, 0.0) + overlap
            
    if not overlap_by_label:
        return default
        
    labeled_duration = sum(overlap_by_label.values())
    unlabeled_duration = max(0.0, window_duration - labeled_duration)
    if unlabeled_duration > 0:
        overlap_by_label[default] = overlap_by_label.get(default, 0.0) + unlabeled_duration

    for lab, overlap in overlap_by_label.items():
        if ("sz" in lab or "seiz" in lab) and (overlap / window_duration) >= seizure_priority_threshold:
            return lab 

    return max(overlap_by_label.items(), key=lambda x: x[1])[0]


# =========================================================
# CHANNEL / MONTAGE
# =========================================================

def normalize_channel_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("eeg ", "").replace("-ref", "").replace("-le", "").replace(" ", "")
    name = name.replace("t7", "t3").replace("t8", "t4").replace("p7", "t5").replace("p8", "t6")
    return re.sub(r"[^a-z0-9]", "", name)

def load_edf_signals(
    edf_path: str | Path,
    fs: int,
    l_freq: Optional[float],
    h_freq: Optional[float],
) -> Tuple[np.ndarray, List[str]]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick("eeg")

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    data = raw.get_data().astype(np.float32)
    ch_names = list(raw.ch_names)
    return data, ch_names

def extract_unipolar_channels(
    data: np.ndarray,
    original_channels: List[str],
    target_channels: List[str]
):
    norm_original = [normalize_channel_name(ch) for ch in original_channels]
    idx_map = {ch: i for i, ch in enumerate(norm_original)}

    T = data.shape[1]
    unipolar_data = np.zeros((len(target_channels), T), dtype=np.float32)
    
    missing_channels = []

    for i, ch in enumerate(target_channels):
        if ch in idx_map:
            unipolar_data[i] = data[idx_map[ch]]
        else:
            missing_channels.append(ch)
            unipolar_data[i] = np.zeros((T,), dtype=np.float32)

    meta = {
        "final_montage_channels": target_channels,
        "missing_channels": missing_channels,
        "original_channels": original_channels,
        "ignored_channels": []
    }
    return unipolar_data, meta


# =========================================================
# PREPROCESS / QUALITY
# =========================================================

def normalize_per_channel_robust(x: np.ndarray, clip_percentile: float = 2.0) -> np.ndarray:
    low  = np.percentile(x, clip_percentile, axis=1, keepdims=True)
    high = np.percentile(x, 100 - clip_percentile, axis=1, keepdims=True)
    x_clipped = np.clip(x, low, high)
    
    median = np.median(x_clipped, axis=1, keepdims=True)
    
    q25 = np.percentile(x_clipped, 25, axis=1, keepdims=True)
    q75 = np.percentile(x_clipped, 75, axis=1, keepdims=True)
    iqr = q75 - q25 + 1e-8
    
    return ((x_clipped - median) / iqr).astype(np.float32)

def is_bad_window(
    xw: np.ndarray,
    flat_std_thresh: float,
    max_flat_ratio: float,
    max_zero_ratio: float,
) -> bool:
    if not np.isfinite(xw).all():
        return True

    ch_std = xw.std(axis=1)
    flat_mask = ch_std < flat_std_thresh
    zero_mask = np.all(np.abs(xw) < 1e-8, axis=1)

    if flat_mask.mean() > max_flat_ratio:
        return True
    if zero_mask.mean() > max_zero_ratio:
        return True

    return False


# =========================================================
# CACHE ONE RECORD
# =========================================================

def cache_one_record_windows(
    rec: Dict[str, Any],
    out_dir: Path,
    cfg: CacheConfig,
    label_to_id: Dict[str, int],
) -> Optional[Tuple[Path, int, Dict[str, Any]]]:
    edf_path = Path(rec["edf_path"])
    csv_path = Path(rec["csv_path"])
    stem = rec.get("stem") or edf_path.stem

    if not edf_path.exists() or not csv_path.exists():
        return None

    intervals = read_label_intervals_from_csv(csv_path)

    raw_data, original_channels = load_edf_signals(
        edf_path=edf_path, fs=cfg.fs, l_freq=cfg.l_freq, h_freq=cfg.h_freq
    )

    # 1. Apply Unipolar 21 Channel Selection
    x_full, montage_meta = extract_unipolar_channels(
        data=raw_data, original_channels=original_channels, target_channels=CANONICAL_21
    )

    # 2. Convert Volts to Microvolts
    x_full = x_full * 1e6

    # 3. Apply Robust Normalization
    if cfg.per_channel_normalize:
        x_full = normalize_per_channel_robust(x_full, clip_percentile=cfg.clip_percentile)

    T_full = x_full.shape[1]
    win_T = int(cfg.fs * cfg.window_sec)
    stride_T = int(cfg.fs * cfg.stride_sec)

    starts = list(range(0, max(0, T_full - win_T) + 1, stride_T))
    if cfg.max_windows_per_record is not None:
        starts = starts[:cfg.max_windows_per_record]

    xs, ys = [], []
    raw_label_counts, final_label_counts = Counter(), Counter()
    bad_windows, skipped_unknown = 0, 0

    for st in starts:
        en = st + win_T
        if en > T_full: continue

        xw = x_full[:, st:en]
        
        adjusted_flat_thresh = 0.01 if cfg.per_channel_normalize else (cfg.flat_std_thresh * 1e6)
        
        if is_bad_window(
            xw, flat_std_thresh=adjusted_flat_thresh, max_flat_ratio=cfg.max_flat_ratio, max_zero_ratio=cfg.max_zero_ratio
        ):
            bad_windows += 1
            continue

        raw_label = assign_raw_label_by_overlap(st/cfg.fs, en/cfg.fs, intervals)
        final_label = to_binary_label(raw_label, cfg.background_labels)

        raw_label_counts[raw_label] += 1
        final_label_counts[final_label] += 1

        if final_label not in label_to_id:
            skipped_unknown += 1
            continue

        xs.append(torch.from_numpy(xw))
        ys.append(label_to_id[final_label])

    if len(xs) == 0:
        print(f"[SKIP] no valid windows: {stem}")
        return None

    X = torch.stack(xs, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)
    out_path = out_dir / f"{stem}.pt"

    torch.save({
        "x": X, "y": Y,
        "meta": {
            "stem": stem, "fs": cfg.fs, "window_sec": cfg.window_sec,
            "label_mode": "binary", "bad_windows_skipped": bad_windows,
            **montage_meta,
        },
    }, out_path)

    report = {
        "stem": stem, "n_windows": len(xs), "bad_windows_skipped": bad_windows,
        "final_label_counts": dict(final_label_counts),
        "missing_channels": montage_meta["missing_channels"],
    }
    return out_path, len(xs), report


# =========================================================
# BUILD CACHE
# =========================================================

def build_cache_from_json(cfg: CacheConfig) -> Path:
    json_path = Path(cfg.json_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if cfg.max_records is not None:
        records = records[:cfg.max_records]

    label_to_id, id_to_label = build_binary_label_vocab(cfg.seizure_label_name, cfg.non_seizure_label_name)

    manifest_path = out_dir / "manifest.jsonl"
    n_cached_records, total_windows = 0, 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, rec in enumerate(records, start=1):
            result = cache_one_record_windows(rec, out_dir, cfg, label_to_id)
            if result:
                out_pt, n, report = result
                n_cached_records += 1
                total_windows += n
                mf.write(json.dumps({"pt_path": str(out_pt), "n": n}, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(records)}] cached: {out_pt.name} ({n} windows)")

    print(f"\nDone. Cached records: {n_cached_records} | Total windows: {total_windows}")
    return manifest_path


if __name__ == "__main__":
    cfg = CacheConfig(
        json_path=r"assets\eeg_seizure_only_eval.json",
        out_dir=r"cache_windows_unipolar_21_eval",
        fs=250, # Must be 250Hz for the 2500 timepoint expectation of the model!
        window_sec=10,
        stride_sec=5,
        l_freq=0.5,
        h_freq=40.0,
    )
    build_cache_from_json(cfg)
