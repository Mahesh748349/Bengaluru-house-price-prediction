import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def detect_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def rmsle(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().round(4),
            "unique": df.nunique(dropna=True),
        }
    )
    return summary.sort_values(["missing_pct", "unique"], ascending=[False, False])
