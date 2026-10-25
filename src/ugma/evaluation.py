from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .models.ugma import UGMAModel


@dataclass(slots=True)
class ForecastMetrics:
    mae: float
    rmse: float
    count: int


def _forecast_metrics(error: np.ndarray) -> ForecastMetrics:
    return ForecastMetrics(
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        count=int(error.size),
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(stats.spearmanr(left, right).statistic)


def risk_coverage_curve(risk: np.ndarray, realized_error: np.ndarray, points: int = 20) -> dict[str, list[float] | float]:
    order = np.argsort(risk)
    coverages = np.linspace(0.05, 1.0, points)
    selective_risk = []
    for coverage in coverages:
        kept = order[: max(1, int(len(order) * coverage))]
        selective_risk.append(float(realized_error[kept].mean()))
    aurc = float(np.trapz(selective_risk, coverages))
    return {"coverage": coverages.tolist(), "selective_mae": selective_risk, "aurc": aurc}


def calibration_bins(risk: np.ndarray, realized_error: np.ndarray, bins: int = 10) -> dict:
    quantiles = np.quantile(risk, np.linspace(0.0, 1.0, bins + 1))
    predicted, observed, counts = [], [], []
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (risk >= quantiles[index]) & (
            (risk <= quantiles[index + 1]) if upper_inclusive else (risk < quantiles[index + 1])
        )
        count = int(mask.sum())
        if count:
            predicted.append(float(risk[mask].mean()))
            observed.append(float(realized_error[mask].mean()))
        else:
            predicted.append(None)
            observed.append(None)
        counts.append(count)
    weights = np.asarray(counts) / max(1, sum(counts))
    differences = np.asarray(
        [abs(p - o) if p is not None else 0.0 for p, o in zip(predicted, observed)]
    )
    ece = float(np.sum(weights * differences))
    return {"predicted": predicted, "observed": observed, "counts": counts, "ece": ece}


@torch.no_grad()
def evaluate(model: UGMAModel, loader: DataLoader, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    targets, backbone_predictions, enhanced_predictions = [], [], []
    predicted_risk, realized_risk, shift_flags = [], [], []
    alphas, betas, retrieval_confidences, retrieval_similarities, safety_factors, shift_scores = [], [], [], [], [], []
    for batch in loader:
        x, y = batch["x"].to(device), batch["y"].to(device)
        starts = batch["target_start"].to(device)
        enhanced, aux = model(x, starts)
        targets.append(y.cpu().numpy())
        backbone_predictions.append(aux["backbone"].cpu().numpy())
        enhanced_predictions.append(enhanced.cpu().numpy())
        predicted_risk.append(aux["risk"].cpu().numpy())
        realized_risk.append(model.realized_risk(y, aux).cpu().numpy())
        alphas.append(aux["alpha"].cpu().numpy())
        betas.append(aux["beta"].cpu().numpy())
        retrieval_confidences.append(aux["retrieval_confidence"].cpu().numpy())
        retrieval_similarities.append(aux["retrieval_similarity"].cpu().numpy())
        safety_factors.append(aux["safety_factor"].cpu().numpy())
        shift_scores.append(aux["shift_score"].cpu().numpy())
        shift_flags.append(batch["shifted"].numpy())

    y = np.concatenate(targets)
    baseline = np.concatenate(backbone_predictions)
    enhanced = np.concatenate(enhanced_predictions)
    risk = np.concatenate(predicted_risk).reshape(-1)
    realized = np.concatenate(realized_risk).reshape(-1)
    shifted = np.concatenate(shift_flags).astype(bool)
    alpha = np.concatenate(alphas).reshape(-1)
    beta = np.concatenate(betas).reshape(-1)
    retrieval_confidence = np.concatenate(retrieval_confidences).reshape(-1)
    retrieval_similarity = np.concatenate(retrieval_similarities).reshape(-1)
    safety_factor = np.concatenate(safety_factors).reshape(-1)
    shift_score = np.concatenate(shift_scores).reshape(-1)

    baseline_metrics = _forecast_metrics(y - baseline)
    enhanced_metrics = _forecast_metrics(y - enhanced)
    risk_is_applicable = model.risk_monitor.mode != "disabled"
    if risk_is_applicable:
        pearson = stats.pearsonr(risk, realized)
        spearman = stats.spearmanr(risk, realized)
        risk_validation = {
            "applicable": True,
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "calibration": calibration_bins(risk, realized),
            "risk_coverage": risk_coverage_curve(risk, realized),
        }
    else:
        risk_validation = {
            "applicable": False,
            "reason": (
                "The retained method contains no risk monitor; risk metrics are "
                "reported only for reviewer-diagnostic controls."
            ),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "calibration": {"predicted": [], "observed": [], "counts": [], "ece": None},
            "risk_coverage": None,
        }
    summary = {
        "backbone": asdict(baseline_metrics),
        "enhanced": asdict(enhanced_metrics),
        "improvement_percent": {
            "mae": 100.0 * (baseline_metrics.mae - enhanced_metrics.mae) / baseline_metrics.mae,
            "rmse": 100.0 * (baseline_metrics.rmse - enhanced_metrics.rmse) / baseline_metrics.rmse,
        },
        "risk_validation": risk_validation,
        "gate_diagnostics": {
            "alpha_mean": float(alpha.mean()),
            "beta_mean": float(beta.mean()),
            "risk_alpha_spearman": _safe_spearman(risk, alpha),
            "risk_beta_spearman": _safe_spearman(risk, beta),
            "retrieval_confidence_mean": float(retrieval_confidence.mean()),
            "fallback_fraction_below_0_5": float((retrieval_confidence < 0.5).mean()),
            "retrieval_similarity_mean": float(retrieval_similarity.mean()),
            "safety_factor_mean": float(safety_factor.mean()),
            "safety_fallback_fraction_below_0_5": float((safety_factor < 0.5).mean()),
            "shift_score_mean": float(shift_score.mean()),
        },
    }
    if shifted.any() and (~shifted).any():
        summary["regimes"] = {
            "steady": {
                "backbone": asdict(_forecast_metrics((y - baseline)[~shifted])),
                "enhanced": asdict(_forecast_metrics((y - enhanced)[~shifted])),
            },
            "shifted": {
                "backbone": asdict(_forecast_metrics((y - baseline)[shifted])),
                "enhanced": asdict(_forecast_metrics((y - enhanced)[shifted])),
            },
        }
    arrays = {
        "target": y,
        "backbone": baseline,
        "enhanced": enhanced,
        "predicted_risk": risk,
        "realized_risk": realized,
        "shifted": shifted,
        "alpha": alpha,
        "beta": beta,
        "retrieval_confidence": retrieval_confidence,
        "retrieval_similarity": retrieval_similarity,
        "safety_factor": safety_factor,
        "shift_score": shift_score,
    }
    return summary, arrays


def paired_bootstrap(
    baseline_absolute_error: np.ndarray,
    enhanced_absolute_error: np.ndarray,
    seed: int,
    samples: int = 5000,
    block_length: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    baseline = baseline_absolute_error.reshape(len(baseline_absolute_error), -1).mean(axis=1)
    enhanced = enhanced_absolute_error.reshape(len(enhanced_absolute_error), -1).mean(axis=1)
    difference = baseline - enhanced
    if block_length is None:
        block_length = max(2, int(round(len(difference) ** (1.0 / 3.0))))
    blocks_needed = int(np.ceil(len(difference) / block_length))
    starts = rng.integers(0, len(difference), size=(samples, blocks_needed))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(difference)
    indices = indices.reshape(samples, -1)[:, : len(difference)]
    bootstrap = difference[indices].mean(axis=1)
    statistic, p_value = stats.wilcoxon(difference, alternative="greater")
    return {
        "mean_mae_reduction": float(difference.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "paired_cohens_d": float(difference.mean() / difference.std(ddof=1)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_one_sided": float(p_value),
        "bootstrap": "circular_moving_block",
        "block_length": int(block_length),
    }


@torch.no_grad()
def benchmark_inference(
    model: UGMAModel,
    batch: torch.Tensor,
    device: torch.device,
    warmup: int = 20,
    repetitions: int = 100,
) -> dict[str, float | int]:
    model.eval()
    batch = batch.to(device)
    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repetitions):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(batch)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            model(batch)
            timings.append((time.perf_counter() - started) * 1000.0)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "batch_size": len(batch),
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_std": float(np.std(timings, ddof=1)),
        "latency_ms_per_sample": float(np.mean(timings) / len(batch)),
        "peak_gpu_bytes": int(peak),
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
    }
