from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowDataset
from .models.ugma import UGMAModel


@dataclass(slots=True)
class TrainingRecord:
    stage: str
    epoch: int
    train_loss: float
    validation_loss: float | None
    elapsed_seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def controlled_shift_augmentation(
    history: torch.Tensor,
    target: torch.Tensor,
    probability: float,
    severity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply train-time-only affine, trend, changepoint, and frequency shifts."""

    batch, seq_len, channels = history.shape
    pred_len = target.shape[1]
    apply = (torch.rand(batch, 1, 1, device=history.device) < probability).to(history.dtype)
    scale = torch.exp(torch.randn(batch, 1, channels, device=history.device) * (0.20 * severity))
    offset = (torch.rand(batch, 1, channels, device=history.device) * 2.0 - 1.0) * severity
    slope = (torch.rand(batch, 1, channels, device=history.device) * 2.0 - 1.0) * (0.30 * severity)
    time = torch.linspace(-1.0, 1.0, seq_len + pred_len, device=history.device)[None, :, None]
    joined = torch.cat([history, target], dim=1)
    onset = torch.randint(seq_len // 2, seq_len + pred_len, (batch, 1, 1), device=history.device)
    integer_time = torch.arange(seq_len + pred_len, device=history.device)[None, :, None]
    after_onset = (integer_time >= onset).to(history.dtype)
    change_level = (torch.rand(batch, 1, channels, device=history.device) * 2.0 - 1.0) * severity
    frequency = torch.rand(batch, 1, channels, device=history.device) * 0.18 + 0.04
    phase = torch.rand(batch, 1, channels, device=history.device) * (2.0 * torch.pi)
    oscillation = 0.35 * severity * torch.sin(2.0 * torch.pi * frequency * integer_time + phase)
    shifted = joined * scale + offset + slope * time + after_onset * (change_level + oscillation)
    joined = joined * (1.0 - apply) + shifted * apply
    return joined[:, :seq_len], joined[:, seq_len:]


def risk_ranking_loss(predicted: torch.Tensor, realized: torch.Tensor) -> torch.Tensor:
    predicted = predicted.flatten()
    realized = realized.flatten()
    permutation = torch.randperm(len(predicted), device=predicted.device)
    target_sign = torch.sign(realized - realized[permutation])
    informative = target_sign != 0
    margin = (predicted - predicted[permutation]) * target_sign
    if not informative.any():
        return predicted.sum() * 0.0
    return F.softplus(-margin[informative]).mean()


@torch.no_grad()
def risk_validation_loss(model: UGMAModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        backbone = model.backbone(x)
        normalized_x, normalized_backbone, mean, scale = model.normalize_context(x, backbone)
        predicted = model.risk_monitor(normalized_x, normalized_backbone)
        realized = ((y - mean) / scale - normalized_backbone).abs().mean(dim=1)
        total += F.mse_loss(predicted, realized, reduction="sum").item()
        count += realized.numel()
    return total / count


def train_risk_monitor(
    model: UGMAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    rank_weight: float,
    shift_probability: float,
    shift_severity: float,
    device: torch.device,
    early_stopping_patience: int | None = None,
) -> list[TrainingRecord]:
    """Pretrain the empirical risk estimator before it controls either gate."""

    optimizer = torch.optim.Adam(
        model.risk_monitor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    records: list[TrainingRecord] = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            x, y = controlled_shift_augmentation(x, y, shift_probability, shift_severity)
            with torch.no_grad():
                backbone = model.backbone(x)
                normalized_x, normalized_backbone, mean, scale = model.normalize_context(x, backbone)
                realized = ((y - mean) / scale - normalized_backbone).abs().mean(dim=1)
            optimizer.zero_grad(set_to_none=True)
            predicted = model.risk_monitor(normalized_x, normalized_backbone)
            loss = F.mse_loss(predicted, realized) + rank_weight * risk_ranking_loss(predicted, realized)
            loss.backward()
            optimizer.step()
            total += loss.item() * realized.numel()
            count += realized.numel()
        val = risk_validation_loss(model, val_loader, device)
        records.append(TrainingRecord("risk_monitor", epoch, total / count, val, time.perf_counter() - started))
        if val < best_val:
            best_val = val
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.risk_monitor.state_dict().items()
            }
        else:
            stale_epochs += 1
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            break
    if best_state is not None:
        model.risk_monitor.load_state_dict(best_state)
    return records


@torch.no_grad()
def fit_risk_calibrator(
    model: UGMAModel,
    val_loader: DataLoader,
    shift_severity: float,
    device: torch.device,
    seed: int,
) -> dict:
    """Fit a positive affine calibration using clean and perturbed validation only."""

    seed_everything(seed + 7919)
    raw_scores, realized_errors = [], []
    model.eval()
    severity_multipliers = (0.0, 0.5, 1.0, 2.0, 3.0)
    severities = tuple(multiplier * shift_severity for multiplier in severity_multipliers)
    for severity in severities:
        for batch in val_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            if severity > 0:
                x, y = controlled_shift_augmentation(x, y, 1.0, severity)
            backbone = model.backbone(x)
            normalized_x, normalized_backbone, mean, scale = model.normalize_context(x, backbone)
            raw_scores.append(model.risk_monitor(normalized_x, normalized_backbone).reshape(-1).cpu())
            realized_errors.append(
                ((y - mean) / scale - normalized_backbone).abs().mean(dim=1).reshape(-1).cpu()
            )
    raw = torch.cat(raw_scores).double()
    realized = torch.cat(realized_errors).double()
    centered_raw = raw - raw.mean()
    slope = float((centered_raw * (realized - realized.mean())).sum() / centered_raw.square().sum().clamp_min(1e-12))
    slope = max(0.0, slope)
    intercept = float(realized.mean() - slope * raw.mean())
    model.set_risk_calibration(slope, intercept)
    calibrated = (slope * raw + intercept).clamp_min(0.0)
    return {
        "method": "positive_affine",
        "fit_partition": "validation_clean_plus_controlled_shifts",
        "samples": int(len(raw)),
        "severity_multipliers": list(severity_multipliers),
        "slope": slope,
        "intercept": intercept,
        "mae_before": float((raw - realized).abs().mean()),
        "mae_after": float((calibrated - realized).abs().mean()),
    }


@torch.no_grad()
def fit_shift_monitor(
    model: UGMAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    quantile: float | None,
    device: torch.device,
) -> dict:
    train_features = []
    for batch in train_loader:
        normalized, _, _ = model.normalize_history(batch["x"].to(device))
        train_features.append(model.shift_monitor.features(normalized).cpu())
    features = torch.cat(train_features)
    model.shift_monitor.set_reference(
        features.mean(dim=0).to(device),
        features.std(dim=0, unbiased=False).to(device),
    )
    validation_scores = []
    for batch in val_loader:
        normalized, _, _ = model.normalize_history(batch["x"].to(device))
        validation_scores.append(model.shift_monitor(normalized).reshape(-1).cpu())
    threshold = (
        float(torch.quantile(torch.cat(validation_scores), quantile))
        if quantile is not None
        else None
    )
    if threshold is not None:
        model.set_safety_shift_threshold(threshold)
    return {
        "monitor": "train_fitted_diagonal_mahalanobis",
        "validation_quantile": quantile,
        "threshold": threshold,
    }


@torch.no_grad()
def prediction_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        prediction = model(x)
        total += F.mse_loss(prediction, y, reduction="sum").item()
        count += y.numel()
    return total / count


def train_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    early_stopping_patience: int | None = None,
) -> list[TrainingRecord]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    records: list[TrainingRecord] = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = F.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        val = prediction_loss(model, val_loader, device)
        records.append(TrainingRecord("backbone", epoch, total / count, val, time.perf_counter() - started))
        if val < best_val:
            best_val = val
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return records


def _sample_window_tensors(
    dataset: WindowDataset,
    max_windows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if len(dataset) <= max_windows:
        indices = np.arange(len(dataset))
    else:
        indices = np.sort(rng.choice(len(dataset), max_windows, replace=False))
    histories, futures, starts = [], [], []
    for index in indices:
        item = dataset[int(index)]
        histories.append(item["x"])
        futures.append(item["y"])
        starts.append(item["target_start"])
    x = torch.stack(histories)
    y = torch.stack(futures)
    target_starts = torch.stack(starts)
    channels = x.shape[-1]
    nodes = torch.arange(channels).repeat(len(x))
    return (
        x.transpose(1, 2).reshape(-1, x.shape[1]),
        y.transpose(1, 2).reshape(-1, y.shape[1]),
        target_starts.repeat_interleave(channels),
        nodes,
    )


def _future_nearest_positive(
    futures: torch.Tensor,
    starts: torch.Tensor,
    temporal_gap: int,
    metric: str = "cosine",
    chunk_size: int = 256,
) -> torch.Tensor:
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("positive metric must be cosine or euclidean")
    positives = []
    normalized = F.normalize(futures, dim=-1) if metric == "cosine" else None
    for begin in range(0, len(futures), chunk_size):
        end = min(begin + chunk_size, len(futures))
        if metric == "cosine":
            distance = 1.0 - normalized[begin:end] @ normalized.T
        else:
            distance = torch.cdist(futures[begin:end], futures)
        invalid = (starts[begin:end, None] - starts[None, :]).abs() < temporal_gap
        if not (~invalid).any(dim=1).all():
            raise ValueError(
                "No non-overlapping predictive positive is available; "
                "reduce the declared temporal gap or sample a longer training interval."
            )
        distance[invalid] = float("inf")
        positives.append(distance.argmin(dim=1))
    return torch.cat(positives)


def train_predictive_encoder(
    model: UGMAModel,
    train_dataset: WindowDataset,
    epochs: int,
    temporal_gap: int,
    temperature: float,
    device: torch.device,
    seed: int,
    positive_metric: str = "cosine",
    max_windows: int = 512,
) -> list[TrainingRecord]:
    histories, futures, starts, _ = _sample_window_tensors(train_dataset, max_windows, seed)
    histories, futures = model.normalize_bank_entries(histories, futures)
    # Two windows with target starts closer than seq_len + pred_len have
    # overlapping complete sample intervals [t-seq_len, t+pred_len).  Apply
    # whichever exclusion is stricter: this span or the declared gap.
    effective_temporal_gap = max(
        temporal_gap,
        train_dataset.seq_len + train_dataset.pred_len,
    )
    positive_indices = _future_nearest_positive(
        futures,
        starts,
        effective_temporal_gap,
        metric=positive_metric,
    )
    pairs = TensorDataset(histories, histories[positive_indices])
    loader = DataLoader(pairs, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    records: list[TrainingRecord] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.encoder.train()
        total, count = 0.0, 0
        for anchor, positive in loader:
            anchor, positive = anchor.to(device), positive.to(device)
            optimizer.zero_grad(set_to_none=True)
            z_anchor = model.encoder(anchor.unsqueeze(-1)).squeeze(1)
            z_positive = model.encoder(positive.unsqueeze(-1)).squeeze(1)
            logits = z_anchor @ z_positive.T / temperature
            target = torch.arange(len(anchor), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(anchor)
            count += len(anchor)
        records.append(TrainingRecord("predictive_encoder", epoch, total / count, None, time.perf_counter() - started))
    return records


@torch.no_grad()
def build_condition_bank(
    model: UGMAModel,
    train_dataset: WindowDataset,
    bank_size: int,
    device: torch.device,
    seed: int,
    max_windows: int = 4096,
) -> dict[str, float | int]:
    started = time.perf_counter()
    histories, futures, starts, nodes = _sample_window_tensors(train_dataset, max_windows, seed)
    histories, futures = model.normalize_bank_entries(histories, futures)
    model.encoder.eval()
    key_batches = []
    for begin in range(0, len(histories), 1024):
        batch = histories[begin : begin + 1024].to(device)
        key_batches.append(model.encoder(batch.unsqueeze(-1)).squeeze(1).cpu())
    keys = torch.cat(key_batches)

    if len(keys) > bank_size:
        selected_parts = []
        channel_count = int(nodes.max().item()) + 1
        base_quota, remainder = divmod(bank_size, channel_count)
        for node in range(channel_count):
            candidates = torch.where(nodes == node)[0]
            quota = min(len(candidates), base_quota + int(node < remainder))
            if quota == 0:
                continue
            node_keys = keys[candidates]
            if len(candidates) > quota:
                clustering = MiniBatchKMeans(
                    n_clusters=quota,
                    batch_size=min(2048, len(candidates)),
                    n_init=3,
                    random_state=seed + node,
                ).fit(node_keys.numpy())
                centers = torch.from_numpy(clustering.cluster_centers_).float()
                local = torch.cdist(centers, node_keys).argmin(dim=1).unique()
                if len(local) < quota:
                    used = set(local.tolist())
                    remaining = torch.tensor([i for i in range(len(node_keys)) if i not in used])
                    local = torch.cat([local, remaining[: quota - len(local)]])
                selected_parts.append(candidates[local])
            else:
                selected_parts.append(candidates)
        selected = torch.cat(selected_parts)
    else:
        selected = torch.arange(len(keys))

    model.bank.set_entries(
        keys[selected].to(device),
        futures[selected].to(device),
        nodes[selected].to(device),
        starts[selected].to(device),
        histories[selected].to(device),
    )
    bytes_used = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            model.bank.keys,
            model.bank.histories,
            model.bank.futures,
            model.bank.nodes,
            model.bank.target_starts,
        )
    )
    return {
        "entries": int(len(selected)),
        "bytes": int(bytes_used),
        "build_seconds": time.perf_counter() - started,
        "latest_target_start": int(model.bank.target_starts.max().item()),
        "entries_per_node_min": int(torch.bincount(model.bank.nodes).min().item()),
        "entries_per_node_max": int(torch.bincount(model.bank.nodes).max().item()),
        "temporal_exclusion_span": int(model.bank.temporal_gap),
    }


@torch.no_grad()
def enhancement_loss(model: UGMAModel, loader: DataLoader, device: torch.device, risk_weight: float) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        starts = batch["target_start"].to(device)
        prediction, aux = model(x, starts)
        realized_risk = model.realized_risk(y, aux)
        loss = F.mse_loss(prediction, y) + risk_weight * F.mse_loss(aux["risk"], realized_risk)
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def train_enhancement(
    model: UGMAModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    risk_weight: float,
    risk_rank_weight: float,
    shift_probability: float,
    shift_severity: float,
    device: torch.device,
    early_stopping_patience: int | None = None,
) -> list[TrainingRecord]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    records: list[TrainingRecord] = []
    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            x, y = controlled_shift_augmentation(x, y, shift_probability, shift_severity)
            optimizer.zero_grad(set_to_none=True)
            starts = batch["target_start"].to(device)
            prediction, aux = model(x, starts)
            realized_risk = model.realized_risk(y, aux)
            loss = (
                F.mse_loss(prediction, y)
                + risk_weight * F.mse_loss(aux["risk"], realized_risk)
                + risk_rank_weight * risk_ranking_loss(aux["risk"], realized_risk)
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        val = enhancement_loss(model, val_loader, device, risk_weight)
        records.append(TrainingRecord("enhancement", epoch, total / count, val, time.perf_counter() - started))
        if val < best_val:
            best_val = val
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("backbone.")
            }
        else:
            stale_epochs += 1
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            break
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    return records
