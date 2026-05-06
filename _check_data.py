import torch
import random
from pathlib import Path

files = list(Path("cache_windows_binary_10_sec").glob("*.pt"))
random.shuffle(files)
check = files[:5]

for f in check:
    d = torch.load(f, map_location="cpu")
    x = d["x"]
    y = d["y"]
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    labels = y.unique().tolist()
    print(f"{f.name}: shape={x.shape}, dtype={x.dtype}, "
          f"NaN={has_nan}, Inf={has_inf}, "
          f"min={x.min():.4f}, max={x.max():.4f}, "
          f"mean={x.mean():.4f}, std={x.std():.4f}, "
          f"labels={labels}")
