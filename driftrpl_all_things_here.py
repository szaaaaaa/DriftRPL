import os
import sys
import math
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from driftrpl.datasets.gas_drift import load_gas_drift_stream
from driftrpl.datasets.electricity import load_electricity_stream

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility
# -----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    if path is None or path == "":
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def log_line(path: str, msg: str) -> None:
    ensure_dir(os.path.dirname(path))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def next_experiment_number(out_dir: str) -> int:
    """
    Returns the next experiment number by scanning existing experiment dirs.
    Expected dir name prefix: exp_XXXX_YYYYMMDD_HHMMSS
    """
    ensure_dir(out_dir)
    max_n = 0
    for name in os.listdir(out_dir):
        if not name.startswith("exp_"):
            continue
        parts = name.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            max_n = max(max_n, int(parts[1]))
    return max_n + 1


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def compute_recovery_mae(y_true: np.ndarray, y_pred: np.ndarray, drift_points_sup: List[int], W: int) -> float:
    """
    Mean MAE over W-step windows immediately after each supervised drift point.
    """
    vals = []
    n = len(y_true)
    for dp in drift_points_sup:
        a = int(dp)
        b = int(min(dp + W, n))
        if 0 <= a < b:
            vals.append(float(np.mean(np.abs(y_true[a:b] - y_pred[a:b]))))
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr.astype(np.float64)
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(arr.astype(np.float64), k, mode="same")


def make_eval_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drift_points_sup: List[int],
    segment_len: int,
    L: int,
    h: int,
    total_len: int,
    roll_win: int,
) -> pd.DataFrame:
    """
    Create per-step evaluation CSV for a supervised stream.
    Includes rolling MAE/RMSE and regime_id aligned to the underlying raw segments.
    """
    n = len(y_true)
    t = np.arange(n, dtype=np.int32)
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    roll_mae = _rolling_mean(abs_err, roll_win)
    roll_rmse = np.sqrt(_rolling_mean(sq_err, roll_win))

    # supervised index i maps to raw target time: t_target = (L-1)+i+h
    t_target = (L - 1) + t + h
    regime_id = np.minimum(
        t_target // max(1, segment_len),
        int(math.ceil(total_len / max(1, segment_len)) - 1),
    ).astype(np.int32)

    is_drift = np.zeros(n, dtype=np.int32)
    for dp in drift_points_sup:
        if 0 <= int(dp) < n:
            is_drift[int(dp)] = 1

    df = pd.DataFrame(
        {
            "step": t,
            "t_target_raw": t_target.astype(np.int32),
            "regime_id": regime_id,
            "is_drift": is_drift,
            "y_true": y_true.astype(np.float64),
            "y_pred": y_pred.astype(np.float64),
            "abs_err": abs_err.astype(np.float64),
            "sq_err": sq_err.astype(np.float64),
            f"rolling_mae_win{roll_win}": roll_mae.astype(np.float64),
            f"rolling_rmse_win{roll_win}": roll_rmse.astype(np.float64),
        }
    )
    return df


# -----------------------------
# Sliding windows
# -----------------------------
def make_windows(y: np.ndarray, L: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate series y[0..T-1] into supervised pairs:
      x[i] = y[i : i+L]
      y_sup[i] = y[i+L-1+h]
    So total n = T - (L-1) - h.
    """
    y = np.asarray(y).astype(np.float32)
    T = int(len(y))
    n = T - (L - 1) - h
    if n <= 0:
        raise ValueError(f"Not enough length: T={T}, L={L}, h={h} => n={n}")
    x = np.zeros((n, L), dtype=np.float32)
    y_sup = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        x[i, :] = y[i : i + L]
        y_sup[i] = y[i + (L - 1) + h]
    return x, y_sup


# -----------------------------
# Page-Hinkley
# -----------------------------
class PageHinkley:
    """
    Minimal Page-Hinkley change detector on scalar stream (loss values).
    """

    def __init__(self, delta: float = 0.005, lam: float = 50.0, alpha: float = 0.99):
        self.delta = float(delta)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.mean = 0.0
        self.m_t = 0.0
        self.M_t = 0.0
        self.t = 0

    def update(self, x: float) -> bool:
        self.t += 1
        x = float(x)
        if self.t == 1:
            self.mean = x
        else:
            self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x

        self.m_t += (x - self.mean - self.delta)
        self.M_t = min(self.M_t, self.m_t)
        return (self.m_t - self.M_t) > self.lam


# -----------------------------
# Online scaler (Welford, vector + scalar)
# -----------------------------
class OnlineScaler:
    """
    Running mean/std using Welford's algorithm.
    Supports scalar (shape=()) or vector (shape=(d,)).
    At time t, you should call transform() using stats from 0..t-1,
    and call update(x_t) AFTER processing current step to keep strict causality.
    """

    def __init__(self, shape: Optional[Tuple[int, ...]] = None, eps: float = 1e-8):
        self.eps = float(eps)
        self.shape = shape
        self.n = 0
        if shape is None or shape == ():
            self.mean = 0.0
            self.M2 = 0.0
        else:
            self.mean = np.zeros(shape, dtype=np.float64)
            self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        if self.shape is None or self.shape == ():
            xv = float(np.asarray(x))
            delta = xv - float(self.mean)
            self.mean = float(self.mean) + delta / self.n
            delta2 = xv - float(self.mean)
            self.M2 = float(self.M2) + delta * delta2
        else:
            xv = np.asarray(x, dtype=np.float64)
            delta = xv - self.mean
            self.mean = self.mean + delta / self.n
            delta2 = xv - self.mean
            self.M2 = self.M2 + delta * delta2

    def get_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            if self.shape is None or self.shape == ():
                return np.array(self.mean, dtype=np.float64), np.array(1.0, dtype=np.float64)
            return self.mean.copy(), np.ones(self.shape, dtype=np.float64)

        var = self.M2 / (self.n - 1)
        std = np.sqrt(np.maximum(var, 0.0)) + self.eps
        return (self.mean if isinstance(self.mean, np.ndarray) else np.array(self.mean, dtype=np.float64)), std

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean, std = self.get_mean_std()
        xv = np.asarray(x, dtype=np.float32)
        return ((xv - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        mean, std = self.get_mean_std()
        xn = np.asarray(x_norm, dtype=np.float32)
        return (xn * std.astype(np.float32) + mean.astype(np.float32)).astype(np.float32)


# -----------------------------
# Models with embed()
# -----------------------------
class MLPForecaster(nn.Module):
    def __init__(self, L: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(L, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        y = self.fc2(h).squeeze(-1)
        return y

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return h


class GRUForecaster(nn.Module):
    def __init__(self, L: int, hidden: int):
        super().__init__()
        self.L = int(L)
        self.hidden = int(hidden)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        _, h_last = self.gru(x_seq)
        y = self.fc(h_last[0]).squeeze(-1)
        return y

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        _, h_last = self.gru(x_seq)
        return h_last[0]


def build_model(hidden: int, L: int = 64) -> nn.Module:
    return MLPForecaster(L=L, hidden=hidden)


# -----------------------------
# DriftBench-TS (synthetic piecewise drift)
# -----------------------------
@dataclass
class SegmentConfig:
    ar: np.ndarray
    mu: float
    sigma: float
    season_amp: float
    season_period: int
    drift_type: str


def _stable_ar(rng: np.random.Generator, ar_order: int, scale: float = 0.8) -> np.ndarray:
    ar = rng.normal(0.0, 0.2, size=(ar_order,))
    ar = ar / (np.sum(np.abs(ar)) + 1e-12) * scale
    return ar.astype(np.float32)


def generate_piecewise_series(
    total_len: int,
    segment_len: int,
    ar_order: int,
    drift_types: List[str],
    seed: int,
) -> Tuple[np.ndarray, List[int], List[SegmentConfig]]:
    rng = np.random.default_rng(seed)
    n_segments = int(math.ceil(total_len / segment_len))
    drift_points_raw = [s * segment_len for s in range(1, n_segments) if s * segment_len < total_len]

    base_ar = _stable_ar(rng, ar_order)
    base_mu = 0.0
    base_sigma = 0.5
    base_amp = 1.0
    base_period = 50

    segments: List[SegmentConfig] = []
    for s in range(n_segments):
        dtype = drift_types[s % len(drift_types)]
        ar = base_ar.copy()
        mu = base_mu
        sigma = base_sigma
        amp = base_amp
        period = base_period

        if dtype == "mean":
            mu += float(rng.normal(0.0, 2.0))
        elif dtype == "var":
            sigma *= float(rng.uniform(0.6, 2.0))
        elif dtype == "season":
            amp *= float(rng.uniform(0.5, 2.5))
            period = int(rng.integers(30, 90))
        elif dtype == "corr":
            ar = ar + rng.normal(0.0, 0.15, size=(ar_order,)).astype(np.float32)
            ar = ar / (np.sum(np.abs(ar)) + 1e-12) * 0.8
        elif dtype == "mixed":
            mu += float(rng.normal(0.0, 1.5))
            sigma *= float(rng.uniform(0.7, 1.8))
            amp *= float(rng.uniform(0.7, 2.2))
            period = int(rng.integers(35, 85))
            ar = ar + rng.normal(0.0, 0.12, size=(ar_order,)).astype(np.float32)
            ar = ar / (np.sum(np.abs(ar)) + 1e-12) * 0.8
        else:
            raise ValueError(f"Unknown drift type: {dtype}")

        segments.append(
            SegmentConfig(ar=ar, mu=mu, sigma=sigma, season_amp=amp, season_period=period, drift_type=dtype)
        )

    y = np.zeros((total_len,), dtype=np.float32)
    y[:ar_order] = rng.normal(0.0, 1.0, size=(ar_order,)).astype(np.float32)

    for t in range(ar_order, total_len):
        seg_id = min(t // segment_len, n_segments - 1)
        seg = segments[seg_id]
        seasonal = seg.season_amp * math.sin(2.0 * math.pi * (t / max(seg.season_period, 1)))
        ar_part = float(np.dot(seg.ar, y[t - ar_order : t][::-1]))
        noise = float(rng.normal(0.0, seg.sigma))
        y[t] = np.float32(seg.mu + seasonal + ar_part + noise)

    return y, drift_points_raw, segments


def map_raw_drift_to_supervised_indices(drift_points_raw: List[int], L: int, h: int, total_len: int) -> List[int]:
    n_sup = total_len - h - (L - 1)
    drift_points_sup = []
    for dp in drift_points_raw:
        i0 = dp - h - (L - 1)
        if 0 <= i0 < n_sup:
            drift_points_sup.append(int(i0))
    return drift_points_sup


def build_drift_schedule(
    total_len: int,
    segment_len: int,
    ar_order: int,
    drift_types: List[str],
    seed: int,
    L: int,
    h: int,
    drift_points_raw: List[int],
    drift_points_sup: List[int],
    segments: List[SegmentConfig],
) -> Dict[str, Any]:
    return {
        "dataset": "synthetic",
        "total_len": total_len,
        "segment_len": segment_len,
        "ar_order": ar_order,
        "drift_types": drift_types,
        "seed": seed,
        "n_segments": len(segments),
        "drift_points_raw": drift_points_raw,
        "drift_points_sup": drift_points_sup,
        "segments": [asdict(seg) for seg in segments],
    }


# -----------------------------
# Online training loop
# -----------------------------
@dataclass
class OnlineConfig:
    L: int
    buffer_M: int
    batch_B: int
    steps_K: int
    lr: float
    warmup: int
    tau: float
    alpha: float
    beta: float
    eps: float
    loss_ema: float
    window_W: int
    ph_delta: float
    ph_lam: float
    ph_alpha: float
    reset_warmup: int


class BufferItem:
    def __init__(self, x: np.ndarray, y: float, z: np.ndarray, w: float, v: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.v = v


class UniformReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.items: List[BufferItem] = []
        self.rng = np.random.default_rng(seed)

    def add(self, item: BufferItem) -> None:
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items.pop(0)
            self.items.append(item)

    def sample(self, batch_size: int) -> List[BufferItem]:
        if len(self.items) == 0:
            return []
        idx = self.rng.integers(0, len(self.items), size=(batch_size,))
        return [self.items[int(i)] for i in idx]


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.items: List[BufferItem] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, item: BufferItem) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, self.seen))
            if j < self.capacity:
                self.items[j] = item

    def sample(self, batch_size: int) -> List[BufferItem]:
        if len(self.items) == 0:
            return []
        idx = self.rng.integers(0, len(self.items), size=(batch_size,))
        return [self.items[int(i)] for i in idx]


class PrototypePrioritizedBuffer:
    def __init__(self, capacity: int, tau: float, alpha: float, beta: float, eps: float, seed: int):
        self.capacity = int(capacity)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.items: List[BufferItem] = []
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _dist2(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def add_or_update(self, x: np.ndarray, y: float, z: np.ndarray) -> int:
        if len(self.items) == 0:
            self.items.append(BufferItem(x=x.copy(), y=float(y), z=z.copy(), w=1.0, v=0.0))
            return 0

        d2s = [self._dist2(z, it.z) for it in self.items]
        j = int(np.argmin(d2s))
        if d2s[j] <= (self.tau * self.tau):
            it = self.items[j]
            w_new = it.w + 1.0
            it.x = (it.x * it.w + x) / w_new
            it.y = float((it.y * it.w + y) / w_new)
            it.z = (it.z * it.w + z) / w_new
            it.w = w_new
            self.items[j] = it
            return j

        if len(self.items) < self.capacity:
            self.items.append(BufferItem(x=x.copy(), y=float(y), z=z.copy(), w=1.0, v=0.0))
            return len(self.items) - 1

        best_d2 = 1e30
        best_i, best_k = 0, 1
        m = len(self.items)
        for i in range(m):
            zi = self.items[i].z
            for k in range(i + 1, m):
                d2 = self._dist2(zi, self.items[k].z)
                if d2 < best_d2:
                    best_d2 = d2
                    best_i, best_k = i, k

        it_i = self.items[best_i]
        it_k = self.items[best_k]
        w_new = it_i.w + it_k.w
        it_i.x = (it_i.x * it_i.w + it_k.x * it_k.w) / w_new
        it_i.y = float((it_i.y * it_i.w + it_k.y * it_k.w) / w_new)
        it_i.z = (it_i.z * it_i.w + it_k.z * it_k.w) / w_new
        it_i.w = w_new
        it_i.v = max(it_i.v, it_k.v)
        self.items[best_i] = it_i
        self.items.pop(best_k)

        self.items.append(BufferItem(x=x.copy(), y=float(y), z=z.copy(), w=1.0, v=0.0))
        return len(self.items) - 1

    def update_value(self, idx: int, drift_signal: float, clip_c: float = 5.0) -> None:
        if 0 <= idx < len(self.items):
            it = self.items[idx]
            ds = float(np.clip(drift_signal, 0.0, clip_c))
            it.v = (1.0 - self.beta) * it.v + self.beta * ds

    def sample(self, batch_size: int) -> List[BufferItem]:
        if len(self.items) == 0:
            return []
        v = np.array([max(0.0, float(it.v)) for it in self.items], dtype=np.float64)
        w = (v + self.eps) ** max(0.0, self.alpha)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            idx = self.rng.integers(0, len(self.items), size=(batch_size,))
            return [self.items[int(i)] for i in idx]
        p = w / s
        idx = self.rng.choice(len(self.items), size=(batch_size,), replace=True, p=p)
        return [self.items[int(i)] for i in idx]


def _torch_batch_from_items(items: List[BufferItem], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xb = torch.from_numpy(np.stack([it.x for it in items]).astype(np.float32)).to(device)
    yb = torch.from_numpy(np.array([it.y for it in items], dtype=np.float32)).to(device)
    return xb, yb


def train_online(
    model: nn.Module,
    x_all: np.ndarray,
    y_all: np.ndarray,
    cfg: OnlineConfig,
    device: str,
    mode: str,
    seed: int,
    run_log_path: Optional[str],
    live_loss_csv_path: Optional[str],
    live_flush_every: int,
    log_every: int,
    norm_mode: str,
) -> Dict[str, np.ndarray]:
    """
    mode:
      static
      online_sgd
      sliding_window_sgd
      ph_reset_sgd
      uniform_replay
      reservoir_replay
      driftrpl

    norm_mode:
      global: x_all/y_all are already normalized before calling this function.
              output "pred" is normalized; caller can denorm using global stats.
      online:  x_all/y_all are raw. This function performs strict-causal online scaling.
              output "pred_raw" is denormalized at each step using the scaler stats from 0..t-1.
              output "pred" is the normalized prediction.
    """
    if norm_mode not in ("global", "online"):
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.train()

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    uni_buf = UniformReplayBuffer(cfg.buffer_M, seed=seed + 11)
    res_buf = ReservoirReplayBuffer(cfg.buffer_M, seed=seed + 17)
    pr_buf = PrototypePrioritizedBuffer(cfg.buffer_M, cfg.tau, cfg.alpha, cfg.beta, cfg.eps, seed=seed + 23)
    window_items: List[BufferItem] = []

    ph = PageHinkley(delta=cfg.ph_delta, lam=cfg.ph_lam, alpha=cfg.ph_alpha)
    last_reset_t = -10**9

    n = len(y_all)

    preds_norm = np.zeros((n,), dtype=np.float32)  # normalized prediction
    preds_raw = np.zeros((n,), dtype=np.float32)   # raw-space prediction (only meaningful for online; for global caller denorm)
    losses = np.zeros((n,), dtype=np.float32)

    # For online normalization, record y mean/std used at each step (pre-update) for reproducibility/debug
    y_mean_hist = np.zeros((n,), dtype=np.float32)
    y_std_hist = np.ones((n,), dtype=np.float32)

    loss_ema = 0.0

    f_loss = None
    if live_loss_csv_path is not None:
        ensure_dir(os.path.dirname(live_loss_csv_path))
        f_loss = open(live_loss_csv_path, "w", encoding="utf-8", buffering=1)
        f_loss.write("step,y_true,y_pred,loss\n")

    def _log(msg: str) -> None:
        if run_log_path is not None:
            log_line(run_log_path, msg)

    x_all_np = x_all.astype(np.float32, copy=False)
    y_all_np = y_all.astype(np.float32, copy=False)

    # GPU pre-load (global mode: fully valid; online mode: still OK because we only index current step)
    if device.lower() == "cuda":
        x_all_t = torch.from_numpy(x_all_np).to(torch_device)
        y_all_t = torch.from_numpy(y_all_np).to(torch_device)
    else:
        x_all_t = torch.from_numpy(x_all_np)
        y_all_t = torch.from_numpy(y_all_np)

    sw_rng = np.random.default_rng(seed + 100000)

    # Online scalers (strict causality: transform uses stats from 0..t-1; update AFTER step)
    x_scaler = None
    y_scaler = None
    if norm_mode == "online":
        # x is either (L,) or (d,), treat as vector per step
        x_shape = (int(x_all_np.shape[1]),) if x_all_np.ndim == 2 else ()
        x_scaler = OnlineScaler(shape=x_shape, eps=1e-8)
        y_scaler = OnlineScaler(shape=(), eps=1e-8)

    try:
        for t in range(n):
            x_t_raw = x_all_np[t]
            y_t_raw = float(y_all_np[t])

            # 1) transform with stats up to t-1
            if norm_mode == "online":
                assert x_scaler is not None and y_scaler is not None
                # snapshot mean/std BEFORE updating with current sample
                y_mean_t, y_std_t = y_scaler.get_mean_std()
                y_mean_t = float(np.asarray(y_mean_t))
                y_std_t = float(np.asarray(y_std_t))
                y_mean_hist[t] = np.float32(y_mean_t)
                y_std_hist[t] = np.float32(y_std_t)

                x_t = x_scaler.transform(x_t_raw)  # (d,)
                y_t = float(y_scaler.transform(np.array(y_t_raw, dtype=np.float32)))  # scalar normalized
            else:
                # already normalized by caller
                x_t = x_t_raw
                y_t = y_t_raw

            # 2) prediction (normalized space)
            if device.lower() == "cuda":
                x_t_t = torch.from_numpy(np.asarray(x_t, dtype=np.float32)).to(torch_device).unsqueeze(0)
            else:
                x_t_t = torch.from_numpy(np.asarray(x_t, dtype=np.float32)).unsqueeze(0)

            with torch.no_grad():
                yhat_norm = float(model(x_t_t).detach().cpu().numpy()[0])
            preds_norm[t] = np.float32(yhat_norm)

            # 3) denormalize prediction for evaluation/loss logging output
            if norm_mode == "online":
                # use pre-update y stats (strict-causal)
                yhat_raw = yhat_norm * float(y_std_hist[t]) + float(y_mean_hist[t])
                preds_raw[t] = np.float32(yhat_raw)
                loss_t = float((yhat_norm - y_t) ** 2)
                # live CSV uses raw-scale y_true/y_pred for interpretability
                y_true_for_csv = y_t_raw
                y_pred_for_csv = float(yhat_raw)
            else:
                preds_raw[t] = np.float32(yhat_norm)  # caller may denorm outside
                loss_t = float((yhat_norm - y_t) ** 2)
                y_true_for_csv = y_t
                y_pred_for_csv = yhat_norm

            losses[t] = np.float32(loss_t)

            # 4) drift signal
            if t == 0:
                loss_ema = loss_t
            else:
                loss_ema = (1.0 - cfg.loss_ema) * loss_ema + cfg.loss_ema * loss_t
            drift_signal = abs(loss_t - loss_ema)

            # live loss csv
            if f_loss is not None:
                f_loss.write(f"{t},{y_true_for_csv},{y_pred_for_csv},{loss_t}\n")
                if live_flush_every <= 1 or (t % live_flush_every == 0):
                    f_loss.flush()

            # periodic run log
            if log_every > 0 and (t % log_every == 0):
                _log(f"progress step={t}/{n} loss={loss_t:.6f} ema={loss_ema:.6f}")

            # static mode: still must update scalers for strict online (optional but consistent)
            if mode == "static":
                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            # embed uses normalized x
            with torch.no_grad():
                z_t = model.embed(x_t_t).detach().cpu().numpy()[0].astype(np.float32)

            item = BufferItem(x=np.asarray(x_t, dtype=np.float32).copy(), y=float(y_t), z=z_t.copy(), w=1.0, v=0.0)

            if mode == "online_sgd":
                if t >= cfg.warmup:
                    model.train()
                    opt.zero_grad()
                    out = model(x_t_t)
                    if device.lower() == "cuda":
                        y_target = torch.tensor([y_t], device=torch_device, dtype=torch.float32)
                    else:
                        y_target = torch.tensor([y_t], dtype=torch.float32)
                    loss = criterion(out, y_target)
                    loss.backward()
                    opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            if mode == "sliding_window_sgd":
                window_items.append(item)
                if len(window_items) > cfg.window_W:
                    window_items.pop(0)
                if t >= cfg.warmup:
                    for _ in range(cfg.steps_K):
                        idx = sw_rng.integers(0, len(window_items), size=(cfg.batch_B,))
                        batch = [window_items[int(i)] for i in idx]
                        xb, yb = _torch_batch_from_items(batch, torch_device)
                        model.train()
                        opt.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            if mode == "ph_reset_sgd":
                drift = ph.update(loss_t)
                if drift and (t - last_reset_t) > cfg.reset_warmup and t >= cfg.warmup:
                    opt = optim.Adam(model.parameters(), lr=cfg.lr)
                    ph.reset()
                    last_reset_t = t
                    _log(f"drift_detected step={t} optimizer_reset=1")

                if t >= cfg.warmup:
                    model.train()
                    opt.zero_grad()
                    out = model(x_t_t)
                    if device.lower() == "cuda":
                        y_target = torch.tensor([y_t], device=torch_device, dtype=torch.float32)
                    else:
                        y_target = torch.tensor([y_t], dtype=torch.float32)
                    loss = criterion(out, y_target)
                    loss.backward()
                    opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            if mode == "uniform_replay":
                uni_buf.add(item)
                if t >= cfg.warmup:
                    for _ in range(cfg.steps_K):
                        batch = uni_buf.sample(cfg.batch_B)
                        if not batch:
                            break
                        xb, yb = _torch_batch_from_items(batch, torch_device)
                        model.train()
                        opt.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            if mode == "reservoir_replay":
                res_buf.add(item)
                if t >= cfg.warmup:
                    for _ in range(cfg.steps_K):
                        batch = res_buf.sample(cfg.batch_B)
                        if not batch:
                            break
                        xb, yb = _torch_batch_from_items(batch, torch_device)
                        model.train()
                        opt.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            if mode == "driftrpl":
                idx = pr_buf.add_or_update(x=item.x, y=item.y, z=item.z)
                pr_buf.update_value(idx, drift_signal=drift_signal, clip_c=5.0)
                if t >= cfg.warmup:
                    for _ in range(cfg.steps_K):
                        batch = pr_buf.sample(cfg.batch_B)
                        if not batch:
                            break
                        xb, yb = _torch_batch_from_items(batch, torch_device)
                        model.train()
                        opt.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        opt.step()

                if norm_mode == "online":
                    x_scaler.update(x_t_raw)
                    y_scaler.update(np.array(y_t_raw, dtype=np.float32))
                continue

            raise ValueError(f"Unknown mode: {mode}")

        return {
            "pred": preds_norm,         # normalized predictions
            "pred_raw": preds_raw,      # raw predictions (online mode); for global mode it's same as pred
            "loss": losses,             # loss in normalized space
            "y_mean_t": y_mean_hist,    # online mode: per-step y mean used for denorm; global mode: mostly zeros
            "y_std_t": y_std_hist,      # online mode: per-step y std used for denorm; global mode: ones
        }
    finally:
        if f_loss is not None:
            f_loss.flush()
            f_loss.close()


# -----------------------------
# Plotting
# -----------------------------
def plot_curves(
    out_dir: str,
    y_true: np.ndarray,
    preds_by_method: Dict[str, np.ndarray],
    drift_points_sup: List[int],
    prefix: str,
    win: int = 50,
) -> None:
    t = np.arange(len(y_true), dtype=np.int32)

    plt.figure()
    for name, yp in preds_by_method.items():
        e = np.abs(y_true.astype(np.float64) - yp.astype(np.float64))
        plt.plot(t, _rolling_mean(e, win), label=name)
    for dp in drift_points_sup:
        plt.axvline(dp, linestyle="--", linewidth=1)
    plt.xlabel("supervised time index")
    plt.ylabel(f"rolling MAE (win={win})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_curve_mae.png"), dpi=160)
    plt.close()

    plt.figure()
    for name, yp in preds_by_method.items():
        e2 = (y_true.astype(np.float64) - yp.astype(np.float64)) ** 2
        r = np.sqrt(_rolling_mean(e2, win))
        plt.plot(t, r, label=name)
    for dp in drift_points_sup:
        plt.axvline(dp, linestyle="--", linewidth=1)
    plt.xlabel("supervised time index")
    plt.ylabel(f"rolling RMSE (win={win})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_curve_rmse.png"), dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out_dir", type=str, default="outputs_stats")
    parser.add_argument("--model", type=str, default="gru", choices=["mlp", "gru"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument("--hs", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--hidden", type=int, nargs="+", default=[32])

    # data
    parser.add_argument("--total_len", type=int, default=12000)
    parser.add_argument("--segment_len", type=int, default=2000)
    parser.add_argument("--ar_order", type=int, default=5)
    parser.add_argument("--drift_types", type=str, nargs="+", default=["mean", "var", "season", "corr", "mixed"])
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "gas_drift", "electricity"])

    # electricity
    parser.add_argument("--electricity_path", type=str, default="data/electricity/LD2011_2014.txt")
    parser.add_argument("--electricity_col", type=int, default=1)

    # online config
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--buffer_M", type=int, default=128)
    parser.add_argument("--batch_B", type=int, default=32)
    parser.add_argument("--steps_K", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)

    # driftrpl
    parser.add_argument("--tau", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--loss_ema", type=float, default=0.01)

    # sliding window
    parser.add_argument("--window_W", type=int, default=256)

    # PH reset
    parser.add_argument("--ph_delta", type=float, default=0.005)
    parser.add_argument("--ph_lam", type=float, default=30.0)
    parser.add_argument("--ph_alpha", type=float, default=0.99)
    parser.add_argument("--reset_warmup", type=int, default=50)

    # evaluation
    parser.add_argument("--recovery_W", type=int, default=200)
    parser.add_argument("--eval_roll_win", type=int, default=50)

    # saving
    parser.add_argument("--save_runs", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--live_loss", action="store_true")
    parser.add_argument("--live_flush_every", type=int, default=1)

    # logging
    parser.add_argument("--log_every", type=int, default=500)

    # representative plots
    parser.add_argument("--plot_seed", type=int, default=0)
    parser.add_argument("--plot_h", type=int, default=1)
    parser.add_argument("--plot_hidden", type=int, default=32)

    # NEW: normalization mode (minimal, backward-compatible)
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="global",
        choices=["global", "online"],
        help="global: use full-stream mean/std (fast baseline, but potential leakage). "
             "online: strict-causal running mean/std (paper-ready).",
    )

    args = parser.parse_args()

    if args.dataset == "gas_drift":
        args.hs = [1]
        args.model = "mlp"

    print(f"Running with args: {args}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False.")
        _ = torch.cuda.get_device_name(0)

    dataset_out_dir = os.path.join(args.out_dir, args.dataset)
    exp_n = next_experiment_number(dataset_out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{exp_n:04d}_{ts}"
    exp_root = os.path.join(dataset_out_dir, exp_id)
    ensure_dir(exp_root)

    exp_log = os.path.join(exp_root, "experiment.log")
    log_line(exp_log, f"experiment_id={exp_id}")
    log_line(exp_log, f"command={' '.join(sys.argv)}")
    log_line(exp_log, f"device={args.device} model={args.model} seeds={args.seeds} horizons={args.hs} hidden={args.hidden}")
    log_line(exp_log, f"norm_mode={args.norm_mode}")

    cfg_base = OnlineConfig(
        L=args.L,
        buffer_M=args.buffer_M,
        batch_B=args.batch_B,
        steps_K=args.steps_K,
        lr=args.lr,
        warmup=args.warmup,
        tau=args.tau,
        alpha=args.alpha,
        beta=args.beta,
        eps=args.eps,
        loss_ema=args.loss_ema,
        window_W=args.window_W,
        ph_delta=args.ph_delta,
        ph_lam=args.ph_lam,
        ph_alpha=args.ph_alpha,
        reset_warmup=args.reset_warmup,
    )

    methods: Dict[str, str] = {
        "Static": "static",
        "OnlineSGD": "online_sgd",
        "SlidingWindowSGD": "sliding_window_sgd",
        "PHResetSGD": "ph_reset_sgd",
        "UniformReplay": "uniform_replay",
        "ReservoirReplay": "reservoir_replay",
        "DriftRPL": "driftrpl",
    }

    save_json(
        os.path.join(exp_root, "experiment_meta.json"),
        {
            "experiment_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
            "device": args.device,
            "model": args.model,
            "seeds": args.seeds,
            "hs": args.hs,
            "hidden": args.hidden,
            "norm_mode": args.norm_mode,
            "data": {
                "total_len": args.total_len,
                "segment_len": args.segment_len,
                "ar_order": args.ar_order,
                "drift_types": args.drift_types,
            },
            "online": asdict(cfg_base),
            "evaluation": {"recovery_W": args.recovery_W, "eval_roll_win": args.eval_roll_win},
            "methods": methods,
        },
    )

    all_rows: List[Dict[str, object]] = []
    rep_preds_by_method: Dict[str, np.ndarray] = {}
    rep_y_true: Optional[np.ndarray] = None
    rep_drift_points_sup: Optional[List[int]] = None

    for h in args.hs:
        for hidden in args.hidden:
            for seed in args.seeds:
                set_all_seeds(seed)

                # -----------------------------
                # data / stream selection
                # -----------------------------
                if args.dataset == "gas_drift":
                    stream = load_gas_drift_stream()
                    X = stream.X.astype(np.float32)
                    y_all = stream.y.astype(np.float32)
                    batch_all = stream.batch.astype(np.int32)
                    assert len(X) == len(y_all) == len(batch_all)

                    x_all = X
                    drift_points_sup = [int(i) for i in np.where(np.diff(batch_all) != 0)[0] + 1]

                    # MIN FIX (hard): gas_drift input dim must match model input dim
                    L_run = int(X.shape[1])
                    cfg = OnlineConfig(
                        L=L_run,
                        buffer_M=cfg_base.buffer_M,
                        batch_B=cfg_base.batch_B,
                        steps_K=cfg_base.steps_K,
                        lr=cfg_base.lr,
                        warmup=cfg_base.warmup,
                        tau=cfg_base.tau,
                        alpha=cfg_base.alpha,
                        beta=cfg_base.beta,
                        eps=cfg_base.eps,
                        loss_ema=cfg_base.loss_ema,
                        window_W=cfg_base.window_W,
                        ph_delta=cfg_base.ph_delta,
                        ph_lam=cfg_base.ph_lam,
                        ph_alpha=cfg_base.ph_alpha,
                        reset_warmup=cfg_base.reset_warmup,
                    )

                    gas_id = getattr(stream, "gas_id", None)
                    gas_id_unique = int(len(np.unique(gas_id))) if gas_id is not None else None

                    drift_schedule = {
                        "dataset": "gas_drift",
                        "n": int(len(y_all)),
                        "drift_points_sup": drift_points_sup,
                        "batch_unique": int(len(np.unique(batch_all))),
                        "gas_id_unique": gas_id_unique,
                        "feature_dim": int(X.shape[1]),
                    }

                elif args.dataset == "electricity":
                    cfg = cfg_base
                    stream = load_electricity_stream(
                        txt_path=args.electricity_path,
                        column=args.electricity_col,
                        segment_len=args.segment_len,
                    )
                    y = stream.y.astype(np.float32)
                    x_all, y_all = make_windows(y, L=cfg.L, h=h)

                    drift_points_raw = [int(i) for i in np.where(np.diff(stream.batch) != 0)[0] + 1]
                    drift_points_sup = map_raw_drift_to_supervised_indices(
                        drift_points_raw=drift_points_raw, L=cfg.L, h=h, total_len=len(y)
                    )

                    drift_schedule = {
                        "dataset": "electricity",
                        "path": stream.meta.get("path") if hasattr(stream, "meta") else None,
                        "y_col": stream.meta.get("y_col") if hasattr(stream, "meta") else None,
                        "n_raw": int(len(y)),
                        "n_sup": int(len(y_all)),
                        "drift_points_raw": drift_points_raw,
                        "drift_points_sup": drift_points_sup,
                        "segment_len": int(args.segment_len),
                    }

                else:
                    cfg = cfg_base
                    y, drift_points_raw, segments = generate_piecewise_series(
                        total_len=args.total_len,
                        segment_len=args.segment_len,
                        ar_order=args.ar_order,
                        drift_types=args.drift_types,
                        seed=seed,
                    )
                    x_all, y_all = make_windows(y, L=cfg.L, h=h)
                    drift_points_sup = map_raw_drift_to_supervised_indices(
                        drift_points_raw=drift_points_raw, L=cfg.L, h=h, total_len=args.total_len
                    )
                    drift_schedule = build_drift_schedule(
                        total_len=args.total_len,
                        segment_len=args.segment_len,
                        ar_order=args.ar_order,
                        drift_types=args.drift_types,
                        seed=seed,
                        L=cfg.L,
                        h=h,
                        drift_points_raw=drift_points_raw,
                        drift_points_sup=drift_points_sup,
                        segments=segments,
                    )

                # -----------------------------
                # Normalization
                # global: do what you had before (fast baseline)
                # online: DO NOT normalize here; train_online will do strict-causal scaling
                # -----------------------------
                x_mean = float(np.mean(x_all))
                x_std = float(np.std(x_all))
                y_mean = float(np.mean(y_all))
                y_std = float(np.std(y_all))
                if x_std == 0:
                    x_std = 1e-8
                if y_std == 0:
                    y_std = 1e-8

                if args.norm_mode == "global":
                    x_all = (x_all - x_mean) / x_std
                    y_all = (y_all - y_mean) / y_std

                # -----------------------------
                # directories
                # -----------------------------
                seedh_dir = os.path.join(exp_root, "runs", f"seed{seed}", f"h{h}", f"hidden{hidden}")
                ensure_dir(seedh_dir)

                # -----------------------------
                # base model (shared init)
                # -----------------------------
                base = build_model(hidden, L=cfg.L) if args.model == "mlp" else GRUForecaster(L=cfg.L, hidden=hidden)
                base_state = base.state_dict()

                for display_name, mode in methods.items():
                    run_dir = os.path.join(seedh_dir, display_name)
                    ensure_dir(run_dir)
                    run_log = os.path.join(run_dir, "train.log")

                    model = build_model(hidden, L=cfg.L) if args.model == "mlp" else GRUForecaster(L=cfg.L, hidden=hidden)
                    model.load_state_dict(base_state)

                    live_loss_path = os.path.join(run_dir, "loss_live.csv") if args.live_loss else None

                    out = train_online(
                        model=model,
                        x_all=x_all,
                        y_all=y_all,
                        cfg=cfg,
                        device=args.device,
                        mode=mode,
                        seed=seed,
                        run_log_path=run_log,
                        live_loss_csv_path=live_loss_path,
                        live_flush_every=args.live_flush_every,
                        log_every=args.log_every,
                        norm_mode=args.norm_mode,
                    )

                    # -----------------------------
                    # Evaluation: always in RAW space for reporting
                    # -----------------------------
                    if args.norm_mode == "global":
                        y_pred = (out["pred"] * y_std + y_mean).astype(np.float64)
                        y_true = (y_all * y_std + y_mean).astype(np.float64)
                    else:
                        y_pred = out["pred_raw"].astype(np.float64)
                        y_true = y_all.astype(np.float64)  # raw y_all (not normalized)

                    m_mae = mae(y_true, y_pred)
                    m_rmse = rmse(y_true, y_pred)
                    m_smape = smape(y_true, y_pred)
                    rec = compute_recovery_mae(y_true, y_pred, drift_points_sup, W=args.recovery_W)

                    all_rows.append(
                        {
                            "experiment_id": exp_id,
                            "method": display_name,
                            "mode": mode,
                            "model": args.model,
                            "seed": seed,
                            "h": h,
                            "hidden": hidden,
                            "L": cfg.L,
                            "norm_mode": args.norm_mode,
                            "MAE": m_mae,
                            "RMSE": m_rmse,
                            "sMAPE": m_smape,
                            f"Recovery@{args.recovery_W}(MAE)": rec,
                            "run_dir": run_dir,
                        }
                    )

                    # per-run saving (keep original behavior)
                    if args.save_runs:
                        save_npy(os.path.join(run_dir, "pred.npy"), out["pred"])
                        save_npy(os.path.join(run_dir, "pred_raw.npy"), out["pred_raw"])
                        save_npy(os.path.join(run_dir, "loss.npy"), out["loss"])
                        # y_true.npy: keep whatever y_all is in this run (normalized if global; raw if online)
                        save_npy(os.path.join(run_dir, "y_true.npy"), y_all.astype(np.float32))
                        # save per-step y scaler stats for online mode (for auditability)
                        save_npy(os.path.join(run_dir, "y_mean_t.npy"), out["y_mean_t"])
                        save_npy(os.path.join(run_dir, "y_std_t.npy"), out["y_std_t"])

                        save_json(
                            os.path.join(run_dir, "config.json"),
                            {
                                "experiment_id": exp_id,
                                "method": display_name,
                                "mode": mode,
                                "model": args.model,
                                "seed": seed,
                                "h": h,
                                "hidden": hidden,
                                "device": args.device,
                                "norm_mode": args.norm_mode,
                                "data": {
                                    "total_len": args.total_len,
                                    "segment_len": args.segment_len,
                                    "ar_order": args.ar_order,
                                    "drift_types": args.drift_types,
                                },
                                "online": {
                                    "L": cfg.L,
                                    "buffer_M": cfg.buffer_M,
                                    "batch_B": cfg.batch_B,
                                    "steps_K": cfg.steps_K,
                                    "lr": cfg.lr,
                                    "warmup": cfg.warmup,
                                    "window_W": cfg.window_W,
                                    "tau": cfg.tau,
                                    "alpha": cfg.alpha,
                                    "beta": cfg.beta,
                                    "eps": cfg.eps,
                                    "loss_ema": cfg.loss_ema,
                                    "ph_delta": cfg.ph_delta,
                                    "ph_lam": cfg.ph_lam,
                                    "ph_alpha": cfg.ph_alpha,
                                    "reset_warmup": cfg.reset_warmup,
                                    "recovery_W": args.recovery_W,
                                    "log_every": args.log_every,
                                    "live_flush_every": args.live_flush_every,
                                },
                                "drift": (
                                    {
                                        "drift_points_sup": drift_schedule["drift_points_sup"],
                                        "batch_unique": drift_schedule.get("batch_unique", None),
                                        "gas_id_unique": drift_schedule.get("gas_id_unique", None),
                                        "n": drift_schedule.get("n", None),
                                        "feature_dim": drift_schedule.get("feature_dim", None),
                                    }
                                    if args.dataset == "gas_drift"
                                    else (
                                        {
                                            "drift_points_raw": drift_schedule["drift_points_raw"],
                                            "drift_points_sup": drift_schedule["drift_points_sup"],
                                            "n_raw": drift_schedule.get("n_raw", None),
                                            "n_sup": drift_schedule.get("n_sup", None),
                                            "segment_len": drift_schedule.get("segment_len", None),
                                        }
                                        if args.dataset == "electricity"
                                        else {
                                            "drift_points_raw": drift_schedule["drift_points_raw"],
                                            "drift_points_sup": drift_schedule["drift_points_sup"],
                                            "n_segments": drift_schedule["n_segments"],
                                        }
                                    )
                                ),
                                "normalization": {
                                    "global_x_mean": x_mean,
                                    "global_x_std": x_std,
                                    "global_y_mean": y_mean,
                                    "global_y_std": y_std,
                                },
                            },
                        )

                    # per-run eval.csv (always write in RAW space)
                    if args.dataset == "gas_drift":
                        n_ = len(y_true)
                        t_ = np.arange(n_, dtype=np.int32)
                        abs_err = np.abs(y_true - y_pred)
                        sq_err = (y_true - y_pred) ** 2
                        roll_mae = _rolling_mean(abs_err, int(args.eval_roll_win))
                        roll_rmse = np.sqrt(_rolling_mean(sq_err, int(args.eval_roll_win)))

                        is_drift = np.zeros(n_, dtype=np.int32)
                        for dp in drift_points_sup:
                            if 0 <= int(dp) < n_:
                                is_drift[int(dp)] = 1

                        df_eval = pd.DataFrame(
                            {
                                "step": t_,
                                "is_drift": is_drift,
                                "y_true": y_true.astype(np.float64),
                                "y_pred": y_pred.astype(np.float64),
                                "abs_err": abs_err.astype(np.float64),
                                "sq_err": sq_err.astype(np.float64),
                                f"rolling_mae_win{int(args.eval_roll_win)}": roll_mae.astype(np.float64),
                                f"rolling_rmse_win{int(args.eval_roll_win)}": roll_rmse.astype(np.float64),
                            }
                        )
                    else:
                        total_len_eval = args.total_len
                        if args.dataset == "electricity":
                            total_len_eval = int(drift_schedule["n_raw"])
                        df_eval = make_eval_frame(
                            y_true=y_true,
                            y_pred=y_pred,
                            drift_points_sup=drift_points_sup,
                            segment_len=args.segment_len,
                            L=cfg.L,
                            h=h,
                            total_len=total_len_eval,
                            roll_win=int(args.eval_roll_win),
                        )

                    df_eval.to_csv(os.path.join(run_dir, "eval.csv"), index=False)

                    if args.save_checkpoints:
                        torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))

                    if seed == args.plot_seed and h == args.plot_h and hidden == args.plot_hidden:
                        rep_preds_by_method[display_name] = y_pred.astype(np.float32).copy()
                        rep_y_true = y_true.astype(np.float32).copy()
                        rep_drift_points_sup = drift_points_sup

                    msg = (
                        f"done method={display_name} seed={seed} h={h} "
                        f"MAE={m_mae:.6f} RMSE={m_rmse:.6f} sMAPE={m_smape:.6f} Rec={rec:.6f}"
                    )
                    print(msg)
                    log_line(run_log, "run_end status=OK")
                    log_line(exp_log, msg)

    df = pd.DataFrame(all_rows)
    per_run_path = os.path.join(exp_root, f"results_per_run_{exp_id}.csv")
    df.to_csv(per_run_path, index=False)

    metric_cols = ["MAE", "RMSE", "sMAPE", f"Recovery@{args.recovery_W}(MAE)"]
    agg = df.groupby(["method", "h", "hidden", "norm_mode"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    agg = agg.rename(columns={"method_": "method", "h_": "h", "hidden_": "hidden", "norm_mode_": "norm_mode"})

    agg_path = os.path.join(exp_root, f"results_agg_mean_std_{exp_id}.csv")
    agg.to_csv(agg_path, index=False)

    leader_rows = []
    for hh in sorted(df["h"].unique().tolist()):
        sub = agg[agg["h"] == hh].copy()
        sub = sub.sort_values(by=["MAE_mean", f"Recovery@{args.recovery_W}(MAE)_mean"], ascending=[True, True])
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))
        leader_rows.append(sub)
    leaderboard = pd.concat(leader_rows, axis=0, ignore_index=True)
    leaderboard_path = os.path.join(exp_root, f"leaderboard_{exp_id}.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    if rep_y_true is not None and rep_drift_points_sup is not None and len(rep_preds_by_method) > 0:
        prefix = f"rep_seed{args.plot_seed}_h{args.plot_h}_hidden{args.plot_hidden}_{args.model}_{args.norm_mode}_{exp_id}"
        plot_curves(
            out_dir=exp_root,
            y_true=rep_y_true,
            preds_by_method=rep_preds_by_method,
            drift_points_sup=rep_drift_points_sup,
            prefix=prefix,
            win=int(args.eval_roll_win),
        )
        log_line(exp_log, f"plots_saved prefix={prefix}")

    log_line(exp_log, f"saved {per_run_path}")
    log_line(exp_log, f"saved {agg_path}")
    log_line(exp_log, f"saved {leaderboard_path}")

    print("experiment_dir:", exp_root)
    print("results:", per_run_path)
    print("agg:", agg_path)
    print("leaderboard:", leaderboard_path)


if __name__ == "__main__":
    main()

