"""Aggregate completed runs without hiding outliers or failed configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def flatten_run(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    cfg, result = run["config"], run["results"]
    training = run.get("training", [])
    epochs_by_stage = {
        stage: sum(row.get("stage") == stage for row in training)
        for stage in ("backbone", "predictive_encoder", "risk_monitor", "enhancement")
    }
    stage_seconds = result.get("training_seconds_by_stage", {})
    stage_peaks = result.get("training_peak_gpu_bytes_by_stage", {})
    latency = result.get("latency_by_batch", {})
    environment = run.get("environment", {})
    return {
        "run_path": str(path),
        "seed": cfg["seed"],
        "dataset_kind": cfg["data"]["kind"],
        "dataset_path": cfg["data"].get("path", "generated"),
        "backbone": cfg["backbone"]["name"],
        "horizon": cfg["data"]["pred_len"],
        "network_complexity": cfg["data"].get("network_complexity"),
        "shift_severity": cfg["data"].get("shift_severity"),
        "topology_mode": cfg["ugma"].get("topology_mode", "multiplex"),
        "dfm_mode": cfg["ugma"].get("dfm_mode", "multiplex_graph_attention"),
        "use_hcb": cfg["ugma"]["use_hcb"],
        "use_dfm": cfg["ugma"]["use_dfm"],
        "use_risk_gate": cfg["ugma"]["use_risk_gate"],
        "training_protocol": cfg["ugma"].get("training_protocol", "enhanced_pretrain"),
        "risk_monitor_mode": cfg["ugma"].get("risk_monitor_mode", "shift_augmented"),
        "normalization_mode": cfg["ugma"].get("normalization_mode", "none"),
        "gate_mode": cfg["ugma"].get("gate_mode", "confidence_scaled"),
        "positive_metric": cfg["ugma"].get("positive_metric", "cosine"),
        "top_k": cfg["ugma"]["top_k"],
        "bank_size": cfg["ugma"]["bank_size"],
        "retrieval_temperature": cfg["ugma"]["retrieval_temperature"],
        "temporal_gap": cfg["ugma"]["temporal_gap"],
        "risk_loss_weight": cfg["ugma"]["risk_loss_weight"],
        "gate_bias": cfg["ugma"].get("gate_bias", -5.0),
        "retrieval_encoder": cfg["ugma"].get("retrieval_encoder", "predictive_contrastive"),
        "retrieval_scope": cfg["ugma"].get("retrieval_scope", "all"),
        "baseline_mae": result["backbone"]["mae"],
        "enhanced_mae": result["enhanced"]["mae"],
        "baseline_rmse": result["backbone"]["rmse"],
        "enhanced_rmse": result["enhanced"]["rmse"],
        "improvement_mae_percent": result["improvement_percent"]["mae"],
        "improvement_rmse_percent": result["improvement_percent"]["rmse"],
        "risk_pearson": result["risk_validation"]["pearson_r"],
        "risk_spearman": result["risk_validation"]["spearman_rho"],
        "risk_ece": result["risk_validation"]["calibration"]["ece"],
        "latency_ms_per_sample": result["efficiency"]["latency_ms_per_sample"],
        "peak_gpu_bytes": result["efficiency"]["peak_gpu_bytes"],
        "latency_batch1_ms": latency.get("1", {}).get("latency_ms_mean"),
        "latency_batch16_ms": latency.get("16", {}).get("latency_ms_mean"),
        "latency_batch32_ms": latency.get("32", {}).get("latency_ms_mean"),
        "parameters_total": result["efficiency"]["parameters_total"],
        "parameters_trainable": result["efficiency"]["parameters_trainable"],
        "bank_bytes": result["bank"]["bytes"],
        "bank_build_seconds": result["bank"]["build_seconds"],
        "backbone_epochs_run": epochs_by_stage["backbone"],
        "risk_epochs_run": epochs_by_stage["risk_monitor"],
        "enhancement_epochs_run": epochs_by_stage["enhancement"],
        "backbone_training_seconds": stage_seconds.get("backbone"),
        "risk_training_seconds": stage_seconds.get("risk_monitor"),
        "enhancement_training_seconds": stage_seconds.get("enhancement"),
        "backbone_training_peak_gpu_bytes": stage_peaks.get("backbone"),
        "risk_training_peak_gpu_bytes": stage_peaks.get("risk_monitor"),
        "enhancement_training_peak_gpu_bytes": stage_peaks.get("enhancement"),
        "git_commit": environment.get("git_commit"),
        "source_sha256": environment.get("source_sha256"),
    }


def aggregate(root: Path, output: Path, source_sha256: str | None = None) -> None:
    paths = sorted(root.rglob("run.json"))
    if not paths:
        raise FileNotFoundError(f"No run.json found below {root}")
    frame = pd.DataFrame(flatten_run(path) for path in paths)
    input_runs = len(frame)
    if source_sha256 is not None:
        frame = frame[frame["source_sha256"] == source_sha256].copy()
        if frame.empty:
            raise FileNotFoundError(
                f"No run.json below {root} has source_sha256={source_sha256}"
            )
    keys = [
        "dataset_kind", "dataset_path", "backbone", "horizon", "network_complexity",
        "shift_severity", "topology_mode", "use_hcb", "use_dfm", "dfm_mode",
        "use_risk_gate", "training_protocol", "risk_monitor_mode", "normalization_mode",
        "gate_mode", "positive_metric", "top_k", "retrieval_encoder", "retrieval_scope",
        "bank_size", "retrieval_temperature", "temporal_gap", "risk_loss_weight", "gate_bias",
    ]
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False):
        differences = group["baseline_mae"] - group["enhanced_mae"]
        if len(group) >= 2 and np.any(differences != 0):
            p_value = float(stats.wilcoxon(differences, alternative="greater").pvalue)
        else:
            p_value = float("nan")
        row = dict(zip(keys, group_key))
        row.update(
            n_runs=len(group),
            seeds=",".join(map(str, sorted(group["seed"].unique()))),
            source_sha256=";".join(sorted(group["source_sha256"].dropna().unique())),
            baseline_mae_mean=group["baseline_mae"].mean(),
            enhanced_mae_mean=group["enhanced_mae"].mean(),
            enhanced_mae_std=group["enhanced_mae"].std(ddof=1),
            baseline_rmse_mean=group["baseline_rmse"].mean(),
            enhanced_rmse_mean=group["enhanced_rmse"].mean(),
            enhanced_rmse_std=group["enhanced_rmse"].std(ddof=1),
            improvement_mean_percent=group["improvement_mae_percent"].mean(),
            improvement_std_percent=group["improvement_mae_percent"].std(ddof=1),
            improvement_median_percent=group["improvement_mae_percent"].median(),
            improvement_iqr_percent=group["improvement_mae_percent"].quantile(0.75)
            - group["improvement_mae_percent"].quantile(0.25),
            rmse_improvement_mean_percent=group["improvement_rmse_percent"].mean(),
            degraded_runs=int((group["improvement_mae_percent"] < 0).sum()),
            degraded_rmse_runs=int((group["improvement_rmse_percent"] < 0).sum()),
            paired_mae_reduction_mean=differences.mean(),
            paired_seed_cohens_d=(
                differences.mean() / differences.std(ddof=1)
                if len(differences) >= 2 and differences.std(ddof=1) > 0
                else float("nan")
            ),
            wilcoxon_p=p_value,
            backbone_epochs_mean=group["backbone_epochs_run"].mean(),
            risk_epochs_mean=group["risk_epochs_run"].mean(),
            enhancement_epochs_mean=group["enhancement_epochs_run"].mean(),
            backbone_training_seconds_mean=group["backbone_training_seconds"].mean(),
            risk_training_seconds_mean=group["risk_training_seconds"].mean(),
            enhancement_training_seconds_mean=group["enhancement_training_seconds"].mean(),
            batch1_latency_ms_mean=group["latency_batch1_ms"].mean(),
            batch16_latency_ms_mean=group["latency_batch16_ms"].mean(),
            batch32_latency_ms_mean=group["latency_batch32_ms"].mean(),
            parameters_total=group["parameters_total"].max(),
            parameters_trainable=group["parameters_trainable"].max(),
            bank_bytes=group["bank_bytes"].max(),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    finite = summary["wilcoxon_p"].notna()
    summary["wilcoxon_p_holm"] = np.nan
    summary.loc[finite, "wilcoxon_p_holm"] = holm_adjust(summary.loc[finite, "wilcoxon_p"].to_numpy())
    variant_columns = [
        "topology_mode", "use_hcb", "use_dfm", "dfm_mode", "use_risk_gate",
        "training_protocol", "risk_monitor_mode", "normalization_mode", "gate_mode",
        "positive_metric", "top_k",
        "retrieval_encoder", "retrieval_scope", "bank_size", "retrieval_temperature",
        "temporal_gap", "risk_loss_weight", "gate_bias",
    ]
    condition_columns = [
        "dataset_kind", "dataset_path", "backbone", "horizon", "network_complexity", "shift_severity"
    ]
    frame["variant"] = frame[variant_columns].astype(str).agg("|".join, axis=1)
    pairwise_rows = []
    for condition, condition_frame in frame.groupby(condition_columns, dropna=False):
        variants = sorted(condition_frame["variant"].unique())
        for left_index, left in enumerate(variants):
            for right in variants[left_index + 1 :]:
                left_frame = condition_frame[condition_frame["variant"] == left][["seed", "enhanced_mae"]]
                right_frame = condition_frame[condition_frame["variant"] == right][["seed", "enhanced_mae"]]
                matched = left_frame.merge(right_frame, on="seed", suffixes=("_left", "_right"))
                if len(matched) < 2:
                    continue
                difference = matched["enhanced_mae_left"] - matched["enhanced_mae_right"]
                p_value = (
                    float(stats.wilcoxon(difference, alternative="two-sided").pvalue)
                    if np.any(difference != 0)
                    else 1.0
                )
                row = dict(zip(condition_columns, condition))
                row.update(
                    variant_left=left,
                    variant_right=right,
                    matched_seeds=len(matched),
                    mae_left_minus_right=float(difference.mean()),
                    wilcoxon_p_two_sided=p_value,
                )
                pairwise_rows.append(row)
    pairwise = pd.DataFrame(pairwise_rows)
    if len(pairwise):
        pairwise["wilcoxon_p_holm"] = holm_adjust(pairwise["wilcoxon_p_two_sided"].to_numpy())
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_runs.csv", index=False)
    summary.to_csv(output / "group_summary.csv", index=False)
    pairwise.to_csv(output / "pairwise_variants.csv", index=False)
    (output / "aggregation.json").write_text(
        json.dumps(
            {
                "input_runs": len(frame),
                "groups": len(summary),
                "excluded_source_mismatch": input_runs - len(frame),
                "required_source_sha256": source_sha256,
                "exclusions": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/aggregate"))
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    aggregate(args.root, args.output, args.source_sha256)


if __name__ == "__main__":
    main()
