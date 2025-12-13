import os
import sys
import math
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
    regime_id = np.minimum(t_target // max(1, segment_len), int(math.ceil(total_len / max(1, segment_len)) - 1)).astype(
        np.int32
    )

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
    """
    Returns:
      y: (total_len,)
      drift_points_raw: raw time indices where a new segment starts (excluding 0)
      segments: list of SegmentConfig describing each segment's generating parameters
    """
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

        segments.append(SegmentConfig(ar=ar, mu=mu, sigma=sigma, season_amp=amp, season_period=period, drift_type=dtype))

    y = np.zeros((total_len,), dtype=np.float32)
    y[:ar_order] = rng.normal(0.0, 1.0, size=(ar_order,)).astype(np.float32)

    for t in range(ar_order, total_len):
        seg_id = min(t // segment_len, n_segments - 1)
        seg = segments[seg_id]
        seasonal = seg.season_amp * math.sin(2.0 * math.pi * (t / max(seg.season_period, 1)))
        ar_part = float(np.dot(seg.ar, y[t - ar_order:t][::-1]))
        noise = float(rng.normal(0.0, seg.sigma))
        y[t] = np.float32(seg.mu + seasonal + ar_part + noise)

    return y, drift_points_raw, segments


def make_windows(y: np.ndarray, L: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    supervised index i corresponds to raw end time t_end = (L-1)+i
    target at t_target = t_end + h
    """
    n = len(y)
    n_sup = n - h - (L - 1)
    if n_sup <= 0:
        raise ValueError("Series too short for given L and h.")
    x = np.zeros((n_sup, L), dtype=np.float32)
    yt = np.zeros((n_sup,), dtype=np.float32)
    for i in range(n_sup):
        t_end = (L - 1) + i
        x[i] = y[t_end - L + 1:t_end + 1]
        yt[i] = y[t_end + h]
    return x, yt


def map_raw_drift_to_supervised_indices(drift_points_raw: List[int], L: int, h: int, total_len: int) -> List[int]:
    """
    First affected supervised index: smallest i such that t_target >= dp
      t_target = (L-1)+i + h >= dp  => i >= dp - h - (L-1)
    """
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
    n_segments = int(math.ceil(total_len / segment_len))
    # serialize segment configs safely
    seg_dicts = []
    for i, seg in enumerate(segments):
        seg_dicts.append(
            {
                "segment_id": i,
                "start_raw": int(i * segment_len),
                "end_raw": int(min((i + 1) * segment_len, total_len)),
                "drift_type": seg.drift_type,
                "mu": float(seg.mu),
                "sigma": float(seg.sigma),
                "season_amp": float(seg.season_amp),
                "season_period": int(seg.season_period),
                "ar": [float(x) for x in seg.ar.tolist()],
            }
        )

    return {
        "seed": int(seed),
        "total_len": int(total_len),
        "segment_len": int(segment_len),
        "ar_order": int(ar_order),
        "drift_types_cycle": list(drift_types),
        "n_segments": int(n_segments),
        "drift_points_raw": [int(x) for x in drift_points_raw],
        "drift_points_sup": [int(x) for x in drift_points_sup],
        "windowing": {"L": int(L), "h": int(h)},
        "segments": seg_dicts,
    }


# -----------------------------
# Models
# -----------------------------
class MLPForecaster(nn.Module):
    def __init__(self, L: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(L, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2).squeeze(-1)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return h2


class GRUForecaster(nn.Module):
    def __init__(self, L: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        _, h = self.gru(x_seq)
        last = h.squeeze(0)
        return self.head(last).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        _, h = self.gru(x_seq)
        return h.squeeze(0)


# -----------------------------
# Buffers
# -----------------------------
@dataclass
class BufferItem:
    x: np.ndarray
    y: float
    z: np.ndarray
    w: float
    v: float


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

        # merge closest pair then insert new (O(M^2), ok for small M)
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
            self.items[idx] = it

    def sample(self, batch_size: int) -> List[BufferItem]:
        if len(self.items) == 0:
            return []
        vs = np.array([it.v for it in self.items], dtype=np.float64)
        probs = (vs + self.eps) ** self.alpha
        probs = probs / (probs.sum() + 1e-12)
        idx = self.rng.choice(len(self.items), size=(batch_size,), replace=True, p=probs)
        return [self.items[int(i)] for i in idx]


# -----------------------------
# Drift detection baseline: Page-Hinkley
# -----------------------------
@dataclass
class PageHinkley:
    delta: float = 0.005
    lam: float = 30.0
    alpha: float = 0.99
    mean: float = 0.0
    cum: float = 0.0
    min_cum: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> bool:
        if not self.initialized:
            self.mean = x
            self.cum = 0.0
            self.min_cum = 0.0
            self.initialized = True
            return False
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        self.cum += (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.lam

    def reset(self) -> None:
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.initialized = False


# -----------------------------
# Online config
# -----------------------------
@dataclass
class OnlineConfig:
    L: int = 64
    buffer_M: int = 128
    batch_B: int = 32
    steps_K: int = 1
    lr: float = 3e-4
    warmup: int = 200
    tau: float = 2.5
    alpha: float = 0.8
    beta: float = 0.1
    eps: float = 1e-3
    loss_ema: float = 0.01
    window_W: int = 256
    ph_delta: float = 0.005
    ph_lam: float = 30.0
    ph_alpha: float = 0.99
    reset_warmup: int = 50


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
    """
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

    n = len(x_all)
    preds = np.zeros((n,), dtype=np.float32)
    losses = np.zeros((n,), dtype=np.float32)

    loss_ema = 0.0

    # live loss CSV handle (single open per run; flush periodically)
    f_loss = None
    if live_loss_csv_path is not None:
        ensure_dir(os.path.dirname(live_loss_csv_path))
        f_loss = open(live_loss_csv_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        f_loss.write("step,y_true,y_pred,loss\n")

    def _log(msg: str) -> None:
        if run_log_path is not None:
            log_line(run_log_path, msg)

    try:
        for t in range(n):
            x_t_np = x_all[t]
            y_t = float(y_all[t])

            x_t = torch.from_numpy(x_t_np).to(torch_device).unsqueeze(0)
            with torch.no_grad():
                yhat = float(model(x_t).cpu().numpy()[0])
            preds[t] = np.float32(yhat)

            loss_t = float((yhat - y_t) ** 2)
            losses[t] = np.float32(loss_t)

            if t == 0:
                loss_ema = loss_t
            else:
                loss_ema = (1.0 - cfg.loss_ema) * loss_ema + cfg.loss_ema * loss_t
            drift_signal = abs(loss_t - loss_ema)

            # live loss csv
            if f_loss is not None:
                f_loss.write(f"{t},{y_t},{yhat},{loss_t}\n")
                if live_flush_every <= 1 or (t % live_flush_every == 0):
                    f_loss.flush()

            # periodic run log
            if log_every > 0 and (t % log_every == 0):
                _log(f"progress step={t}/{n} loss={loss_t:.6f} ema={loss_ema:.6f}")

            if mode == "static":
                continue

            with torch.no_grad():
                z_t = model.embed(x_t).cpu().numpy()[0].astype(np.float32)

            item = BufferItem(x=x_t_np.copy(), y=y_t, z=z_t.copy(), w=1.0, v=0.0)

            if mode == "online_sgd":
                if t >= cfg.warmup:
                    model.train()
                    opt.zero_grad()
                    out = model(x_t)
                    loss = criterion(out, torch.tensor([y_t], device=torch_device))
                    loss.backward()
                    opt.step()
                continue

            if mode == "sliding_window_sgd":
                window_items.append(item)
                if len(window_items) > cfg.window_W:
                    window_items.pop(0)
                if t < cfg.warmup:
                    continue
                for _ in range(cfg.steps_K):
                    rng = np.random.default_rng(seed + 100000 + t)
                    idx = rng.integers(0, len(window_items), size=(cfg.batch_B,))
                    batch = [window_items[int(i)] for i in idx]
                    xb, yb = _torch_batch_from_items(batch, torch_device)
                    model.train()
                    opt.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    opt.step()
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
                    out = model(x_t)
                    loss = criterion(out, torch.tensor([y_t], device=torch_device))
                    loss.backward()
                    opt.step()
                continue

            if mode == "uniform_replay":
                uni_buf.add(item)
                if t < cfg.warmup:
                    continue
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
                continue

            if mode == "reservoir_replay":
                res_buf.add(item)
                if t < cfg.warmup:
                    continue
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
                continue

            if mode == "driftrpl":
                idx = pr_buf.add_or_update(x=x_t_np, y=y_t, z=z_t)
                pr_buf.update_value(idx, drift_signal=drift_signal, clip_c=5.0)
                if t < cfg.warmup:
                    continue
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
                continue

            raise ValueError(f"Unknown mode: {mode}")

        return {"pred": preds, "loss": losses}
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

    # rolling MAE
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

    # rolling RMSE
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

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="gru", choices=["mlp", "gru"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--hs", type=int, nargs="+", default=[1, 5, 10])

    # data
    parser.add_argument("--total_len", type=int, default=12000)
    parser.add_argument("--segment_len", type=int, default=2000)
    parser.add_argument("--ar_order", type=int, default=5)
    parser.add_argument("--drift_types", type=str, nargs="+", default=["mean", "var", "season", "corr", "mixed"])

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
    parser.add_argument("--eval_roll_win", type=int, default=50, help="Rolling window for per-step eval.csv and plots.")

    # saving
    parser.add_argument("--save_runs", action="store_true", help="Save per-run pred/loss/y_true/config/drift_schedule/eval.csv and live loss CSV.")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save model_final.pt per run.")
    parser.add_argument("--live_loss", action="store_true", help="Write loss_live.csv during training (append).")
    parser.add_argument("--live_flush_every", type=int, default=1, help="Flush loss_live.csv every N steps (1 = every step).")

    # logging
    parser.add_argument("--log_every", type=int, default=500, help="Write a short progress line every N steps to train.log (0 disables).")

    # representative plots
    parser.add_argument("--plot_seed", type=int, default=0)
    parser.add_argument("--plot_h", type=int, default=1)

    args = parser.parse_args()

    # device checks
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.")
        _ = torch.cuda.get_device_name(0)

    # experiment folder with counter + timestamp
    exp_n = next_experiment_number(args.out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{exp_n:04d}_{ts}"
    exp_root = os.path.join(args.out_dir, exp_id)
    ensure_dir(exp_root)

    exp_log = os.path.join(exp_root, "experiment.log")
    log_line(exp_log, f"experiment_id={exp_id}")
    log_line(exp_log, f"command={' '.join(sys.argv)}")
    log_line(exp_log, f"device={args.device} model={args.model} seeds={args.seeds} hs={args.hs}")

    cfg = OnlineConfig(
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

    # experiment meta (paper-ready: include all drift + online params)
    save_json(
        os.path.join(exp_root, "experiment_meta.json"),
        {
            "experiment_id": exp_id,
            "timestamp": ts,
            "command": " ".join(sys.argv),
            "device": args.device,
            "model": args.model,
            "seeds": args.seeds,
            "hs": args.hs,
            "data": {
                "total_len": args.total_len,
                "segment_len": args.segment_len,
                "ar_order": args.ar_order,
                "drift_types": args.drift_types,
            },
            "online": asdict(cfg),
            "evaluation": {"recovery_W": args.recovery_W, "eval_roll_win": args.eval_roll_win},
            "methods": methods,
        },
    )

    def build_model() -> nn.Module:
        if args.model == "mlp":
            return MLPForecaster(L=cfg.L, hidden=64)
        return GRUForecaster(L=cfg.L, hidden=64)

    all_rows: List[Dict[str, object]] = []

    rep_preds_by_method: Dict[str, np.ndarray] = {}
    rep_y_true: Optional[np.ndarray] = None
    rep_drift_points_sup: Optional[List[int]] = None

    for h in args.hs:
        for seed in args.seeds:
            set_all_seeds(seed)

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

            # Save a paper-grade schedule once per (seed,h) at seed/h level
            seedh_dir = os.path.join(exp_root, "runs", f"seed{seed}", f"h{h}")
            ensure_dir(seedh_dir)
            if args.save_runs:
                save_json(os.path.join(seedh_dir, "drift_schedule.json"), drift_schedule)

            base = build_model()
            base_state = base.state_dict()

            for display_name, mode in methods.items():
                run_dir = os.path.join(seedh_dir, display_name)
                ensure_dir(run_dir)

                run_log = os.path.join(run_dir, "train.log")
                log_line(run_log, f"run_start method={display_name} mode={mode} seed={seed} h={h} device={args.device}")
                log_line(run_log, f"cfg L={cfg.L} M={cfg.buffer_M} B={cfg.batch_B} K={cfg.steps_K} lr={cfg.lr} warmup={cfg.warmup}")

                live_loss_csv = None
                if args.live_loss or args.save_runs:
                    live_loss_csv = os.path.join(run_dir, "loss_live.csv")

                model = build_model()
                model.load_state_dict(base_state)

                out = train_online(
                    model=model,
                    x_all=x_all,
                    y_all=y_all,
                    cfg=cfg,
                    device=args.device,
                    mode=mode,
                    seed=seed,
                    run_log_path=run_log,
                    live_loss_csv_path=live_loss_csv,
                    live_flush_every=max(1, int(args.live_flush_every)),
                    log_every=int(args.log_every),
                )

                y_pred = out["pred"].astype(np.float64)
                y_true = y_all.astype(np.float64)

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
                        "L": cfg.L,
                        "MAE": m_mae,
                        "RMSE": m_rmse,
                        "sMAPE": m_smape,
                        f"Recovery@{args.recovery_W}(MAE)": rec,
                        "buffer_M": cfg.buffer_M,
                        "steps_K": cfg.steps_K,
                        "batch_B": cfg.batch_B,
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
                        "run_dir": run_dir,
                    }
                )

                # save per-run artifacts
                if args.save_runs:
                    save_npy(os.path.join(run_dir, "pred.npy"), out["pred"])
                    save_npy(os.path.join(run_dir, "loss.npy"), out["loss"])
                    save_npy(os.path.join(run_dir, "y_true.npy"), y_all.astype(np.float32))

                    save_json(os.path.join(run_dir, "config.json"), {
                        "experiment_id": exp_id,
                        "method": display_name,
                        "mode": mode,
                        "model": args.model,
                        "seed": seed,
                        "h": h,
                        "device": args.device,
                        "data": {
                            "total_len": args.total_len,
                            "segment_len": args.segment_len,
                            "ar_order": args.ar_order,
                            "drift_types": args.drift_types
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
                        "drift": {
                            "drift_points_raw": drift_schedule["drift_points_raw"],
                            "drift_points_sup": drift_schedule["drift_points_sup"],
                            "n_segments": drift_schedule["n_segments"],
                        },
                        "evaluation": {"eval_roll_win": args.eval_roll_win},
                    })

                    # per-run eval.csv (paper-ready, for plots/tables later without re-running)
                    df_eval = make_eval_frame(
                        y_true=y_true,
                        y_pred=y_pred,
                        drift_points_sup=drift_points_sup,
                        segment_len=args.segment_len,
                        L=cfg.L,
                        h=h,
                        total_len=args.total_len,
                        roll_win=int(args.eval_roll_win),
                    )
                    df_eval.to_csv(os.path.join(run_dir, "eval.csv"), index=False)

                if args.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))

                if seed == args.plot_seed and h == args.plot_h:
                    rep_preds_by_method[display_name] = out["pred"].copy()
                    rep_y_true = y_all.copy()
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
    agg = df.groupby(["method", "h"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    agg = agg.rename(columns={"method_": "method", "h_": "h"})

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

    # representative plots
    if rep_y_true is not None and rep_drift_points_sup is not None and len(rep_preds_by_method) > 0:
        prefix = f"rep_seed{args.plot_seed}_h{args.plot_h}_{args.model}_{exp_id}"
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
