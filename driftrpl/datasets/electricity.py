# driftrpl/datasets/electricity.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ElectricityStream:
    y: np.ndarray               # (T,)
    batch: np.ndarray           # (T,) coarse regime id
    time: Optional[np.ndarray]  # (T,) datetime64 or None
    meta: dict


def _sniff_sep_decimal(p: Path, nbytes: int = 4096) -> tuple[str, str]:
    s = p.open("r", encoding="utf-8", errors="ignore").read(nbytes)
    # LD2011_2014.txt most commonly uses ';' as separator
    sep = ";" if s.count(";") >= s.count(",") else ","
    # when sep=';' it's very often decimal=','
    decimal = "," if sep == ";" else "."
    return sep, decimal


def load_electricity_stream(
    txt_path: Union[str, Path] = "data/electricity/LD2011_2014.txt",
    column: int = 1,
    segment_len: int = 2000,
    max_rows: Optional[int] = None,
) -> ElectricityStream:
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"Electricity txt not found: {p}")

    sep, decimal = _sniff_sep_decimal(p)

    # read header to get column names
    header = pd.read_csv(p, sep=sep, nrows=0, engine="python")
    cols = list(header.columns)
    if len(cols) < 2:
        raise ValueError(f"Unexpected electricity file format: only {len(cols)} columns.")

    time_col = cols[0]
    if column < 1 or column >= len(cols):
        raise ValueError(f"column must be in [1, {len(cols)-1}], got {column}")
    y_col = cols[column]

    df = pd.read_csv(
        p,
        sep=sep,
        decimal=decimal,
        usecols=[time_col, y_col],
        nrows=max_rows,
        engine="python",
    )

    # parse time if possible
    time = None
    try:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if not t.isna().all():
            time = t.to_numpy()
    except Exception:
        time = None

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float32)

    # drop NaNs
    mask = np.isfinite(y)
    y = y[mask]
    if time is not None:
        time = time[mask]

    if len(y) < 10:
        raise ValueError(f"Too few valid samples after cleaning: {len(y)}")

    seg = max(1, int(segment_len))
    batch = (np.arange(len(y), dtype=np.int64) // seg).astype(np.int32)

    return ElectricityStream(
        y=y,
        batch=batch,
        time=time,
        meta={
            "path": str(p),
            "sep": sep,
            "decimal": decimal,
            "time_col": time_col,
            "y_col": y_col,
            "n": int(len(y)),
            "segment_len": seg,
            "max_rows": None if max_rows is None else int(max_rows),
        },
    )
