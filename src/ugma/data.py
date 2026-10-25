from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


@dataclass(slots=True)
class SplitBounds:
    train_end: int
    val_end: int
    total: int


@dataclass(slots=True)
class SeriesBundle:
    values: np.ndarray
    timestamps: np.ndarray
    channel_names: list[str]
    topology: np.ndarray
    shift_mask: np.ndarray
    metadata: dict


class TrainStandardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, train_values: np.ndarray) -> "TrainStandardizer":
        self.mean = train_values.mean(axis=0, keepdims=True).astype(np.float32)
        self.scale = train_values.std(axis=0, keepdims=True).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class WindowDataset(Dataset):
    """Sliding windows whose complete target interval lies in one split."""

    def __init__(
        self,
        values: np.ndarray,
        seq_len: int,
        pred_len: int,
        target_start: int,
        target_end: int,
        shift_mask: np.ndarray | None = None,
    ) -> None:
        self.values = torch.from_numpy(values.astype(np.float32))
        self.seq_len = seq_len
        self.pred_len = pred_len
        first = max(seq_len, target_start)
        last = target_end - pred_len
        self.target_starts = np.arange(first, last + 1, dtype=np.int64)
        self.shift_mask = (
            np.zeros(len(values), dtype=bool) if shift_mask is None else shift_mask.astype(bool)
        )

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_start = int(self.target_starts[index])
        input_start = target_start - self.seq_len
        target_end = target_start + self.pred_len
        return {
            "x": self.values[input_start:target_start],
            "y": self.values[target_start:target_end],
            "input_start": torch.tensor(input_start, dtype=torch.long),
            "input_end": torch.tensor(target_start, dtype=torch.long),
            "target_start": torch.tensor(target_start, dtype=torch.long),
            "target_end": torch.tensor(target_end, dtype=torch.long),
            "shifted": torch.tensor(bool(self.shift_mask[target_start:target_end].any())),
        }


def chronological_bounds(length: int, train_ratio: float, val_ratio: float) -> SplitBounds:
    train_end = int(length * train_ratio)
    # Compute each split length independently.  Adding decimal ratios first can
    # round 0.7 + 0.1 just below 0.8 and move the validation boundary by one row.
    val_end = train_end + int(length * val_ratio)
    return SplitBounds(train_end=train_end, val_end=val_end, total=length)


def _row_normalize(adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=-1, keepdims=True)
    return adjacency / np.maximum(degree, 1e-8)


def generate_multiplex_process(cfg: DataConfig, seed: int) -> SeriesBundle:
    """Generate a stable nonlinear VAR process with known multiplex propagation.

    Layer 0 links a ring, layer 1 links same-role nodes, and layer 2 contains
    directed supervisory links. Complexity controls edge density and cross-layer
    coupling. The final segment contains controlled mean, scale, frequency, and
    propagation shifts.
    """

    rng = np.random.default_rng(seed)
    n, length, layers = cfg.channels, cfg.length, cfg.layers
    topology = np.zeros((layers, n, n), dtype=np.float32)

    for node in range(n):
        topology[0, node, (node - 1) % n] = 1.0
        topology[0, node, (node + 1) % n] = 1.0

    role_width = max(2, n // 3)
    for i in range(n):
        for j in range(n):
            if i != j and i % role_width == j % role_width:
                topology[min(1, layers - 1), i, j] = 1.0

    if layers >= 3:
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.08 + 0.30 * cfg.network_complexity:
                    topology[2, i, j] = 1.0

    topology = np.stack([_row_normalize(layer) for layer in topology])
    layer_weights = np.linspace(0.18, 0.08, layers, dtype=np.float32)
    layer_weights *= 0.55 + cfg.network_complexity

    values = np.zeros((length, n), dtype=np.float32)
    values[:8] = rng.normal(0.0, 0.2, size=(8, n))
    phase = rng.uniform(0, 2 * np.pi, n)
    frequency = rng.uniform(0.012, 0.045, n)
    shift_index = int(length * cfg.shift_start)

    for t in range(8, length):
        seasonal = 0.12 * np.sin(2 * np.pi * frequency * t + phase)
        local = 0.56 * values[t - 1] - 0.11 * values[t - 2] + 0.05 * values[t - 8]
        propagated = np.zeros(n, dtype=np.float32)
        for layer_id in range(layers):
            propagated += layer_weights[layer_id] * (topology[layer_id] @ values[t - 1])

        if t >= shift_index:
            severity = cfg.shift_severity
            seasonal += severity * 0.18 * np.sin(2 * np.pi * (frequency * 1.7) * t + phase)
            propagated *= 1.0 + 0.35 * severity
            offset = severity * np.linspace(-0.35, 0.35, n)
            scale = 1.0 + severity * np.linspace(0.05, 0.30, n)
        else:
            offset = 0.0
            scale = 1.0

        noise = rng.normal(0.0, 0.035 + 0.015 * (t >= shift_index), n)
        values[t] = np.tanh(scale * (local + propagated + seasonal) + offset + noise)

    shift_mask = np.arange(length) >= shift_index
    timestamps = np.arange(length, dtype=np.int64)
    return SeriesBundle(
        values=values,
        timestamps=timestamps,
        channel_names=[f"sensor_{i:02d}" for i in range(n)],
        topology=topology,
        shift_mask=shift_mask,
        metadata={
            "kind": "synthetic_multiplex",
            "layers": layers,
            "network_complexity": cfg.network_complexity,
            "shift_index": shift_index,
            "shift_severity": cfg.shift_severity,
        },
    )


def correlation_topology(train_values: np.ndarray, quantile: float = 0.70) -> np.ndarray:
    """Build three declared data-derived relation layers for topology ablations.

    These layers are statistical relations, not claimed physical multiplex edges:
    zero-lag absolute correlation, positive lead-lag correlation, and negative
    lead-lag correlation.
    """

    corr = np.nan_to_num(np.abs(np.corrcoef(train_values.T)), nan=0.0)
    lead = np.zeros_like(corr)
    for i in range(train_values.shape[1]):
        for j in range(train_values.shape[1]):
            lead[i, j] = np.corrcoef(train_values[:-1, i], train_values[1:, j])[0, 1]
    lead = np.nan_to_num(lead, nan=0.0)

    layers = []
    for scores in (corr, np.maximum(lead, 0.0), np.maximum(-lead, 0.0)):
        off_diag = scores[~np.eye(scores.shape[0], dtype=bool)]
        threshold = np.quantile(off_diag, quantile)
        adjacency = np.where(scores >= threshold, scores, 0.0).astype(np.float32)
        layers.append(_row_normalize(adjacency))
    return np.stack(layers)


def load_csv_bundle(cfg: DataConfig) -> SeriesBundle:
    frame = pd.read_csv(cfg.path)
    timestamps = frame.pop(cfg.timestamp_column).astype(str).to_numpy()
    values = frame.to_numpy(dtype=np.float32)
    bounds = chronological_bounds(len(values), cfg.train_ratio, cfg.val_ratio)
    topology = correlation_topology(values[: bounds.train_end])
    return SeriesBundle(
        values=values,
        timestamps=timestamps,
        channel_names=list(frame.columns),
        topology=topology,
        shift_mask=np.zeros(len(values), dtype=bool),
        metadata={
            "kind": "csv",
            "path": str(Path(cfg.path).resolve()),
            "topology": "three statistical relation layers derived from training data only",
        },
    )


def load_ttfbench_bundle(cfg: DataConfig) -> SeriesBundle:
    shifted_frame = pd.read_csv(cfg.path)
    reference_frame = pd.read_csv(cfg.reference_path)
    timestamps = shifted_frame.pop(cfg.timestamp_column).astype(str).to_numpy()
    reference_frame.pop(cfg.timestamp_column)
    values = shifted_frame.to_numpy(dtype=np.float32)
    reference = reference_frame.to_numpy(dtype=np.float32)
    # TTFBench reconstructs the untouched prefix through normalize/denormalize,
    # which introduces harmless floating-point roundoff (especially for ETTm1).
    # Use numerical equivalence instead of an exact-difference threshold.
    bounds = chronological_bounds(len(values), cfg.train_ratio, cfg.val_ratio)
    equal_rows = np.isclose(values, reference, rtol=1e-5, atol=1e-6).all(axis=1)
    prefix_match_fraction = float(equal_rows[: bounds.val_end].mean())
    if prefix_match_fraction < 0.99:
        raise ValueError(
            "TTFBench clean reference mismatch: fewer than 99% of train/validation rows "
            f"match ({prefix_match_fraction:.3%}). Check for an upstream dataset mislabel."
        )
    shift_mask = ~equal_rows
    topology = correlation_topology(values[: bounds.train_end])
    return SeriesBundle(
        values=values,
        timestamps=timestamps,
        channel_names=list(shifted_frame.columns),
        topology=topology,
        shift_mask=shift_mask,
        metadata={
            "kind": "ttfbench",
            "path": str(Path(cfg.path).resolve()),
            "reference_path": str(Path(cfg.reference_path).resolve()),
            "shifted_rows": int(shift_mask.sum()),
            "first_shifted_row": int(np.flatnonzero(shift_mask)[0]) if shift_mask.any() else None,
            "clean_prefix_match_fraction": prefix_match_fraction,
            "topology": "three statistical relation layers derived from training data only",
        },
    )


def load_bundle(cfg: DataConfig, seed: int) -> SeriesBundle:
    if cfg.kind == "synthetic_multiplex":
        return generate_multiplex_process(cfg, seed)
    if cfg.kind == "ttfbench":
        return load_ttfbench_bundle(cfg)
    return load_csv_bundle(cfg)


def make_datasets(bundle: SeriesBundle, cfg: DataConfig) -> tuple[dict[str, WindowDataset], TrainStandardizer, SplitBounds]:
    bounds = chronological_bounds(len(bundle.values), cfg.train_ratio, cfg.val_ratio)
    standardizer = TrainStandardizer().fit(bundle.values[: bounds.train_end])
    values = standardizer.transform(bundle.values)
    datasets = {
        "train": WindowDataset(values, cfg.seq_len, cfg.pred_len, 0, bounds.train_end, bundle.shift_mask),
        "val": WindowDataset(values, cfg.seq_len, cfg.pred_len, bounds.train_end, bounds.val_end, bundle.shift_mask),
        "test": WindowDataset(values, cfg.seq_len, cfg.pred_len, bounds.val_end, bounds.total, bundle.shift_mask),
    }
    return datasets, standardizer, bounds


def make_loaders(datasets: dict[str, WindowDataset], cfg: DataConfig, seed: int) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=split == "train",
            num_workers=cfg.num_workers,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }


def write_manifest(
    bundle: SeriesBundle,
    cfg: DataConfig,
    bounds: SplitBounds,
    output_path: str | Path,
) -> None:
    digest = hashlib.sha256(bundle.values.tobytes()).hexdigest()
    manifest = {
        "sha256_values": digest,
        "shape": list(bundle.values.shape),
        "channels": bundle.channel_names,
        "split_bounds": asdict(bounds),
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "metadata": bundle.metadata,
        "shift_prevalence": {
            "overall": float(bundle.shift_mask.mean()),
            "train": float(bundle.shift_mask[: bounds.train_end].mean()),
            "validation": float(bundle.shift_mask[bounds.train_end : bounds.val_end].mean()),
            "test": float(bundle.shift_mask[bounds.val_end :].mean()),
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
