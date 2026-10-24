"""Training results shared between the training scripts and src/evaluate.py.

Each training script records one row per model in reports/train_times.csv;
retraining a model replaces its row, so the scripts can run in any order.
"""

from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
TRAIN_TIMES_CSV = REPORTS_DIR / "train_times.csv"


def save_train_time(name, train_time, val_acc, device="cpu"):
    """Add or replace this model's row in reports/train_times.csv."""
    row = pd.DataFrame([{"model": name, "train_time_s": round(train_time, 2),
                         "val_acc": round(val_acc, 4), "device": device}])
    if TRAIN_TIMES_CSV.exists():
        old = pd.read_csv(TRAIN_TIMES_CSV)
        row = pd.concat([old[old["model"] != name], row], ignore_index=True)  
    REPORTS_DIR.mkdir(exist_ok=True)
    row.to_csv(TRAIN_TIMES_CSV, index=False)
