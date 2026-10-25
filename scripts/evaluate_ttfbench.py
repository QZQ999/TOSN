"""Evaluate one frozen training result over every compatible TTFBench shift file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ugma.config import parse_config
from ugma.data import load_bundle, make_datasets, make_loaders, write_manifest
from ugma.evaluation import evaluate, paired_bootstrap
from ugma.models import UGMAModel, build_backbone
from ugma.training import fit_shift_monitor


ROOT = Path(__file__).resolve().parents[1]


def rebuild_model(run: dict, checkpoint_path: Path, device: torch.device) -> tuple[UGMAModel, object]:
    cfg = parse_config(run["config"])
    bundle = load_bundle(cfg.data, cfg.seed)
    channels = bundle.values.shape[1]
    topology = torch.from_numpy(bundle.topology).to(device)
    if cfg.ugma.topology_mode == "none":
        topology = torch.empty((0, channels, channels), device=device)
    elif cfg.ugma.topology_mode == "single_layer":
        topology = topology[:1]
    backbone = build_backbone(
        cfg.backbone.name, cfg.data.seq_len, cfg.data.pred_len, channels,
        cfg.backbone.hidden_dim, ROOT,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
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
    ).to(device)
    state = checkpoint["ugma"]
    model.bank.set_entries(
        state["bank.keys"], state["bank.futures"], state["bank.nodes"], state["bank.target_starts"]
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--shift-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-hcb", action="store_true")
    parser.add_argument("--disable-dfm", action="store_true")
    parser.add_argument("--fallback-threshold", type=float)
    parser.add_argument("--risk-fallback-quantile", type=float)
    parser.add_argument("--shift-fallback-quantile", type=float)
    parser.add_argument("--label", default="trained_full")
    args = parser.parse_args()
    source_run = args.source_run if args.source_run.is_absolute() else ROOT / args.source_run
    shift_root = args.shift_root if args.shift_root.is_absolute() else ROOT / args.shift_root
    reference = args.reference if args.reference.is_absolute() else ROOT / args.reference
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run = json.loads(source_run.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = rebuild_model(run, source_run.parent / "checkpoint.pt", device)
    if args.disable_hcb:
        model.use_hcb = False
    if args.disable_dfm:
        model.use_dfm = False
    if args.fallback_threshold is not None:
        model.fallback_similarity_threshold = args.fallback_threshold
    clean_cfg = replace(cfg.data, kind="csv", path=str(reference), reference_path=None)
    clean_bundle = load_bundle(clean_cfg, cfg.seed)
    clean_datasets, _, _ = make_datasets(clean_bundle, clean_cfg)
    clean_loaders = make_loaders(clean_datasets, clean_cfg, cfg.seed)
    safety_threshold = None
    safety_monitor = None
    if args.shift_fallback_quantile is not None:
        safety_stats = fit_shift_monitor(
            model,
            clean_loaders["train"],
            clean_loaders["val"],
            args.shift_fallback_quantile,
            device,
        )
        safety_threshold = safety_stats["threshold"]
        safety_monitor = "shift"
    elif args.risk_fallback_quantile is not None:
        _, validation_arrays = evaluate(model, clean_loaders["val"], device)
        safety_threshold = float(
            np.quantile(validation_arrays["predicted_risk"], args.risk_fallback_quantile)
        )
        model.set_safety_risk_threshold(safety_threshold)
        safety_monitor = "risk"
    clean_summary, clean_arrays = evaluate(model, clean_loaders["test"], device)
    rows = []
    for shifted_path in sorted(shift_root.glob("*.csv")):
        data_cfg = replace(
            cfg.data,
            kind="ttfbench",
            path=str(shifted_path),
            reference_path=str(reference),
        )
        bundle = load_bundle(data_cfg, cfg.seed)
        datasets, _, bounds = make_datasets(bundle, data_cfg)
        loaders = make_loaders(datasets, data_cfg, cfg.seed)
        summary, arrays = evaluate(model, loaders["test"], device)

        baseline_delta = summary["backbone"]["mae"] - clean_summary["backbone"]["mae"]
        enhanced_delta = summary["enhanced"]["mae"] - clean_summary["enhanced"]["mae"]
        distribution = {
            "baseline_mae_degradation": baseline_delta,
            "enhanced_mae_degradation": enhanced_delta,
            "absolute_degradation_recovered": baseline_delta - enhanced_delta,
            "degradation_recovery_percent": (
                100.0 * (baseline_delta - enhanced_delta) / baseline_delta if baseline_delta > 0 else None
            ),
        }
        baseline_error = np.abs(arrays["target"] - arrays["backbone"])
        enhanced_error = np.abs(arrays["target"] - arrays["enhanced"])
        summary["paired_statistics"] = paired_bootstrap(baseline_error, enhanced_error, cfg.seed)
        summary["clean_reference"] = clean_summary
        summary["distribution_shift"] = distribution
        sample_output = output / shifted_path.stem
        sample_output.mkdir(parents=True, exist_ok=True)
        write_manifest(bundle, data_cfg, bounds, sample_output / "data_manifest.json")
        (sample_output / "evaluation.json").write_text(
            json.dumps(
                {
                    "source_run": str(source_run),
                    "source_environment": run["environment"],
                    "data_config": asdict(data_cfg),
                    "results": summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            sample_output / "predictions.npz",
            **arrays,
            **{f"clean_{key}": value for key, value in clean_arrays.items()},
        )
        rows.append({"sample": shifted_path.stem, **distribution, "shifted_mae": summary["enhanced"]["mae"]})
        print(json.dumps(rows[-1]), flush=True)
    recoveries = [row["degradation_recovery_percent"] for row in rows if row["degradation_recovery_percent"] is not None]
    aggregate = {
        "source_run": str(source_run),
        "inference_variant": {
            "label": args.label,
            "use_hcb": model.use_hcb,
            "use_dfm": model.use_dfm,
            "fallback_similarity_threshold": model.fallback_similarity_threshold,
            "risk_fallback_quantile": args.risk_fallback_quantile,
            "safety_threshold": safety_threshold,
            "shift_fallback_quantile": args.shift_fallback_quantile,
            "safety_monitor": safety_monitor,
        },
        "samples": rows,
        "recovery_percent_mean": float(np.mean(recoveries)),
        "recovery_percent_median": float(np.median(recoveries)),
        "negative_recovery_samples": int(sum(value < 0 for value in recoveries)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
