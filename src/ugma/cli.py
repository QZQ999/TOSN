from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import SplitBounds, load_bundle, make_datasets, make_loaders, write_manifest
from .evaluation import benchmark_inference, evaluate, paired_bootstrap
from .models import UGMAModel, build_backbone
from .training import (
    TrainingRecord,
    build_condition_bank,
    fit_risk_calibrator,
    fit_shift_monitor,
    seed_everything,
    train_backbone,
    train_enhancement,
    train_predictive_encoder,
    train_risk_monitor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted-initial-build"


def _source_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src" / "ugma").rglob("*.py"))
    paths += [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "external_versions.json"]
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "torch", "reformer-pytorch")
    return {name: importlib.metadata.version(name) for name in packages}


def _backbone_cache_key(cfg, values: np.ndarray, bounds: SplitBounds) -> str:
    payload = {
        "seed": cfg.seed,
        "data": {
            "seq_len": cfg.data.seq_len,
            "pred_len": cfg.data.pred_len,
            "train_end": bounds.train_end,
            "val_end": bounds.val_end,
        },
        "backbone": asdict(cfg.backbone),
        "backbone_source": hashlib.sha256(
            (PROJECT_ROOT / "src" / "ugma" / "models" / "backbones.py").read_bytes()
        ).hexdigest(),
        "data_source": hashlib.sha256(
            (PROJECT_ROOT / "src" / "ugma" / "data.py").read_bytes()
        ).hexdigest(),
        "external_versions": json.loads((PROJECT_ROOT / "external_versions.json").read_text(encoding="utf-8")),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    # Backbone fitting and model selection can only see train + validation.
    # Hashing that prefix lets a clean run and its TTFBench-shifted counterpart
    # share a checkpoint without letting any shifted test values affect the key.
    digest.update(np.ascontiguousarray(values[: bounds.val_end]).tobytes())
    return digest.hexdigest()


def run_experiment(config_path: str | Path) -> Path:
    # Capture provenance before any long-running stage.  This prevents a result
    # from claiming a later source revision if the worktree changes mid-run.
    run_git_commit = _git_commit()
    run_source_digest = _source_digest()
    cfg = load_config(config_path)
    if cfg.data.path and not Path(cfg.data.path).is_absolute():
        cfg.data.path = str(PROJECT_ROOT / cfg.data.path)
    if cfg.data.reference_path and not Path(cfg.data.reference_path).is_absolute():
        cfg.data.reference_path = str(PROJECT_ROOT / cfg.data.reference_path)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    output = PROJECT_ROOT / cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    training_peak_gpu_bytes: dict[str, int | None] = {}

    def reset_stage_peak() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def record_stage_peak(stage: str) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            training_peak_gpu_bytes[stage] = int(torch.cuda.max_memory_allocated(device))
        else:
            training_peak_gpu_bytes[stage] = None

    bundle = load_bundle(cfg.data, cfg.seed)
    datasets, _, bounds = make_datasets(bundle, cfg.data)
    loaders = make_loaders(datasets, cfg.data, cfg.seed)
    write_manifest(bundle, cfg.data, bounds, output / "data_manifest.json")

    channels = bundle.values.shape[1]
    backbone = build_backbone(
        cfg.backbone.name,
        cfg.data.seq_len,
        cfg.data.pred_len,
        channels,
        cfg.backbone.hidden_dim,
        PROJECT_ROOT,
    ).to(device)
    cache_key = _backbone_cache_key(cfg, bundle.values, bounds)
    cache_path = PROJECT_ROOT / "artifacts" / "cache" / "backbones" / f"{cache_key}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=device, weights_only=True)
        backbone.load_state_dict(cached["state_dict"])
        training_records = [TrainingRecord(**item) for item in cached["training_records"]]
        backbone_cache_hit = True
        training_peak_gpu_bytes["backbone"] = None
    else:
        reset_stage_peak()
        training_records = train_backbone(
            backbone,
            loaders["train"],
            loaders["val"],
            cfg.backbone.epochs,
            cfg.backbone.learning_rate,
            cfg.backbone.weight_decay,
            device,
            cfg.backbone.early_stopping_patience,
        )
        record_stage_peak("backbone")
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in backbone.state_dict().items()},
                "training_records": [asdict(item) for item in training_records],
            },
            cache_path,
        )
        backbone_cache_hit = False

    topology = torch.from_numpy(bundle.topology).to(device)
    if cfg.ugma.topology_mode == "none":
        topology = torch.empty((0, channels, channels), device=device)
    elif cfg.ugma.topology_mode == "single_layer":
        topology = topology[:1]
    elif cfg.ugma.topology_mode != "multiplex":
        raise ValueError("topology_mode must be none, single_layer, or multiplex")
    model = UGMAModel(
        backbone=backbone,
        seq_len=cfg.data.seq_len,
        pred_len=cfg.data.pred_len,
        channels=channels,
        d_model=cfg.ugma.d_model,
        top_k=cfg.ugma.top_k,
        retrieval_temperature=cfg.ugma.retrieval_temperature,
        topology=topology,
        topology_strength=cfg.ugma.topology_strength,
        temporal_gap=cfg.ugma.temporal_gap,
        fallback_similarity_threshold=cfg.ugma.fallback_similarity_threshold,
        use_hcb=cfg.ugma.use_hcb,
        use_dfm=cfg.ugma.use_dfm,
        use_risk_gate=cfg.ugma.use_risk_gate,
        dfm_mode=cfg.ugma.dfm_mode,
        retrieval_encoder=cfg.ugma.retrieval_encoder,
        retrieval_scope=cfg.ugma.retrieval_scope,
        gate_bias=cfg.ugma.gate_bias,
        safety_temperature=cfg.ugma.safety_temperature,
        risk_monitor_mode=cfg.ugma.risk_monitor_mode,
        normalization_mode=cfg.ugma.normalization_mode,
        gate_mode=cfg.ugma.gate_mode,
    ).to(device)
    if cfg.ugma.retrieval_encoder == "predictive_contrastive":
        reset_stage_peak()
        training_records += train_predictive_encoder(
            model,
            datasets["train"],
            cfg.ugma.encoder_epochs,
            cfg.ugma.temporal_gap,
            cfg.ugma.contrastive_temperature,
            device,
            cfg.seed,
            cfg.ugma.positive_metric,
        )
        record_stage_peak("predictive_encoder")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    reset_stage_peak()
    bank_stats = build_condition_bank(
        model,
        datasets["train"],
        cfg.ugma.bank_size,
        device,
        cfg.seed,
    )
    record_stage_peak("condition_bank")
    if cfg.ugma.training_protocol == "enhanced_pretrain":
        reset_stage_peak()
        training_records += train_risk_monitor(
            model,
            loaders["train"],
            loaders["val"],
            cfg.ugma.risk_pretrain_epochs,
            cfg.ugma.learning_rate,
            cfg.ugma.weight_decay,
            cfg.ugma.risk_rank_weight,
            cfg.ugma.shift_augmentation_probability,
            cfg.ugma.shift_augmentation_severity,
            device,
            cfg.ugma.early_stopping_patience,
        )
        record_stage_peak("risk_monitor")
        calibration_stats = fit_risk_calibrator(
            model,
            loaders["val"],
            cfg.ugma.shift_augmentation_severity,
            device,
            cfg.seed,
        )
        for parameter in model.risk_monitor.parameters():
            parameter.requires_grad_(False)
    elif cfg.ugma.training_protocol == "manuscript_joint":
        if cfg.ugma.risk_monitor_mode == "disabled":
            calibration_stats = {
                "method": "not_applicable",
                "fit_partition": None,
                "samples": 0,
                "slope": None,
                "intercept": None,
                "reason": (
                    "The retained reviewer-revised method contains no risk monitor; "
                    "enabled monitor variants are diagnostic controls only."
                ),
            }
        else:
            calibration_stats = {
                "method": "identity",
                "fit_partition": None,
                "samples": 0,
                "slope": 1.0,
                "intercept": 0.0,
                "reason": "The diagnostic monitor is jointly trained and has no post-hoc calibration.",
            }
    else:
        raise ValueError("training_protocol must be enhanced_pretrain or manuscript_joint")
    safety_stats = fit_shift_monitor(
        model,
        loaders["train"],
        loaders["val"],
        cfg.ugma.safety_risk_quantile
        if cfg.ugma.safety_risk_quantile < 1.0 and cfg.ugma.safety_monitor == "shift"
        else None,
        device,
    )
    if cfg.ugma.safety_risk_quantile < 1.0 and cfg.ugma.safety_monitor == "shift":
        safety_threshold = safety_stats["threshold"]
    elif cfg.ugma.safety_risk_quantile < 1.0 and cfg.ugma.safety_monitor == "risk":
        _, validation_arrays = evaluate(model, loaders["val"], device)
        safety_threshold = float(
            np.quantile(validation_arrays["predicted_risk"], cfg.ugma.safety_risk_quantile)
        )
        model.set_safety_risk_threshold(safety_threshold)
    else:
        safety_threshold = None
    if cfg.ugma.safety_monitor not in {"risk", "shift"}:
        raise ValueError("safety_monitor must be risk or shift")
    reset_stage_peak()
    training_records += train_enhancement(
        model,
        loaders["train"],
        loaders["val"],
        cfg.ugma.enhancement_epochs,
        cfg.ugma.learning_rate,
        cfg.ugma.weight_decay,
        cfg.ugma.risk_loss_weight,
        cfg.ugma.risk_rank_weight,
        cfg.ugma.shift_augmentation_probability,
        cfg.ugma.shift_augmentation_severity,
        device,
        cfg.ugma.early_stopping_patience,
    )
    record_stage_peak("enhancement")

    summary, arrays = evaluate(model, loaders["test"], device)
    if cfg.data.kind == "ttfbench":
        reference_cfg = replace(
            cfg.data,
            kind="csv",
            path=cfg.data.reference_path,
            reference_path=None,
        )
        reference_bundle = load_bundle(reference_cfg, cfg.seed)
        reference_datasets, _, _ = make_datasets(reference_bundle, reference_cfg)
        reference_loaders = make_loaders(reference_datasets, reference_cfg, cfg.seed)
        clean_summary, clean_arrays = evaluate(model, reference_loaders["test"], device)
        summary["ttfbench_clean_reference"] = clean_summary
        baseline_shift_delta = summary["backbone"]["mae"] - clean_summary["backbone"]["mae"]
        enhanced_shift_delta = summary["enhanced"]["mae"] - clean_summary["enhanced"]["mae"]
        summary["distribution_shift"] = {
            "baseline_mae_degradation": baseline_shift_delta,
            "enhanced_mae_degradation": enhanced_shift_delta,
            "absolute_degradation_recovered": baseline_shift_delta - enhanced_shift_delta,
            "degradation_recovery_percent": (
                100.0 * (baseline_shift_delta - enhanced_shift_delta) / baseline_shift_delta
                if baseline_shift_delta > 0
                else None
            ),
        }
        arrays.update({f"clean_{key}": value for key, value in clean_arrays.items()})
    baseline_error = np.abs(arrays["target"] - arrays["backbone"])
    enhanced_error = np.abs(arrays["target"] - arrays["enhanced"])
    summary["paired_statistics"] = paired_bootstrap(baseline_error, enhanced_error, cfg.seed)
    first_batch = next(iter(loaders["test"]))["x"]
    default_batch = first_batch[: min(16, len(first_batch))]
    summary["efficiency"] = benchmark_inference(model, default_batch, device, warmup=5, repetitions=20)
    summary["latency_by_batch"] = {
        str(batch_size): benchmark_inference(
            model,
            first_batch[: min(batch_size, len(first_batch))],
            device,
            warmup=5,
            repetitions=20,
        )
        for batch_size in (1, 16, 32)
    }
    summary["bank"] = bank_stats
    summary["risk_calibrator"] = calibration_stats
    summary["safety_fallback"] = {
        "validation_quantile": cfg.ugma.safety_risk_quantile,
        "threshold": safety_threshold,
        "temperature": cfg.ugma.safety_temperature,
        "monitor": cfg.ugma.safety_monitor,
        "shift_monitor_fit": safety_stats,
    }
    summary["training_seconds_by_stage"] = {
        stage: float(sum(item.elapsed_seconds for item in training_records if item.stage == stage))
        for stage in sorted({item.stage for item in training_records})
    }
    summary["training_peak_gpu_bytes_by_stage"] = training_peak_gpu_bytes

    record = {
        "config": asdict(cfg),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_commit": run_git_commit,
            "source_sha256": run_source_digest,
            "packages": _package_versions(),
            "backbone_cache": {
                "key": cache_key,
                "hit": backbone_cache_hit,
                "path": str(cache_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
        },
        "training": [asdict(item) for item in training_records],
        "results": summary,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **arrays)
    torch.save({"backbone": backbone.state_dict(), "ugma": model.state_dict()}, output / "checkpoint.pt")
    return output / "run.json"


def main() -> None:
    parser = argparse.ArgumentParser(prog="ugma")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    path = run_experiment(args.config)
    if not getattr(args, "quiet", False):
        result = json.loads(path.read_text(encoding="utf-8"))["results"]
        print(json.dumps({"run": str(path), **result}, indent=2))


if __name__ == "__main__":
    main()
