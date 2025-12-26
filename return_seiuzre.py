import pandas as pd
from pathlib import Path
import json

def csv_has_seizure(csv_path: str | Path) -> bool:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False

    df = pd.read_csv(
        csv_path,
        comment="#",          # ⬅️ THIS skips all metadata rows
        usecols=["label"],    # ⬅️ safer than column index
        dtype=str
    )

    labels = df["label"].str.strip().str.lower()

    # TUH seizure labels are usually cpsz, spsz, fnsz, gnsz, etc.
    return labels.str.contains("sz").any()


with open('tuh_train_index.json', 'r') as f:
    data=json.load(f)

seizure_records = []

for rec in data:
    if csv_has_seizure(rec["csv_path"]):
        seizure_records.append(rec)


outpath=Path('eeg_seizure_only.json')
with outpath.open('w' , encoding='utf-8') as f:
    json.dump(seizure_records , f, indent=2)
print(f"Seizure recordings: {len(seizure_records)} / {len(data)}")
