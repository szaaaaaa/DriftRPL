from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from driftrpl.io import data_dir


@dataclass(frozen=True)
class GasDriftStream:
    X: np.ndarray          # shape (T, D)
    y: np.ndarray          # shape (T,)
    batch: np.ndarray      # shape (T,)
    gas_id: np.ndarray     # shape (T,)


def load_gas_drift_stream(
    x_path: Path | None = None,
    y_path: Path | None = None,
    sort_by_batch: bool = True,
) -> GasDriftStream:
    """
    Load Gas Sensor Drift dataset (already converted to dense CSV).
    Expected files:
      data/gas_drift/X.csv  with 128 features (+ optional metadata col like gas_id)
      data/gas_drift/y.csv  with columns: y, gas_id, batch  (batch & gas_id required)

    Returns an online stream ordered by batch if sort_by_batch=True.
    """
    base = data_dir() / "gas_drift"
    x_path = x_path or (base / "X.csv")
    y_path = y_path or (base / "y.csv")

    if not x_path.exists():
        raise FileNotFoundError(f"Missing X.csv: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y.csv: {y_path}")

    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(X_df) == 0 or len(y_df) == 0:
        raise ValueError("X.csv or y.csv is empty.")
    if len(X_df) != len(y_df):
        raise ValueError(f"Row mismatch: X has {len(X_df)}, y has {len(y_df)}.")

    # y column
    if "y" not in y_df.columns:
        raise ValueError(f"y.csv must contain column 'y'. Got: {list(y_df.columns)}")
    if "batch" not in y_df.columns:
        raise ValueError(f"y.csv must contain column 'batch'. Got: {list(y_df.columns)}")
    if "gas_id" not in y_df.columns:
        raise ValueError(f"y.csv must contain column 'gas_id'. Got: {list(y_df.columns)}")

    # Select feature columns:
    # - Prefer f1..f128 if present
    # - Else fallback to numeric columns excluding obvious metadata like gas_id/batch/y
    f_cols = [c for c in X_df.columns if c.startswith("f")]
    if len(f_cols) >= 128:
        f_cols = f_cols[:128]
    else:
        drop = {"y", "batch", "gas_id"}
        cand = [c for c in X_df.columns if c not in drop]
        # If user inserted gas_id in X, remove it
        cand = [c for c in cand if c != "gas_id"]
        # Keep only numeric columns
        numeric = X_df[cand].select_dtypes(include=["number"])
        if numeric.shape[1] < 128:
            raise ValueError(
                f"X.csv does not appear to contain 128 numeric feature columns. "
                f"Numeric cols found={numeric.shape[1]}. Columns={list(X_df.columns)}"
            )
        f_cols = list(numeric.columns[:128])

    X = X_df[f_cols].to_numpy(dtype=np.float32)
    y = y_df["y"].to_numpy(dtype=np.float32)
    batch = y_df["batch"].to_numpy(dtype=np.int32)
    gas_id = y_df["gas_id"].to_numpy(dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf.")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN/Inf.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X/y length mismatch after conversion.")
    if X.shape[1] != 128:
        raise ValueError(f"Expected 128 features, got {X.shape[1]}.")

    # Optional: ensure online order
    if sort_by_batch:
        order = np.argsort(batch, kind="stable")
        X = X[order]
        y = y[order]
        batch = batch[order]
        gas_id = gas_id[order]

    return GasDriftStream(X=X, y=y, batch=batch, gas_id=gas_id)
