"""Materialize reviewer experiment YAMLs from the frozen matrix declaration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2026, 2027, 2028, 2029, 2030]
HORIZONS = [96, 192, 336, 720]
BACKBONES = ["iTransformer", "PatchTST", "TimeMixer", "TimesNet", "FEDformer"]


def write_config(config: dict, relative: str) -> Path:
    path = ROOT / "configs" / "generated" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def public_standard(base: dict, namespace: str = "public") -> list[Path]:
    paths = []
    for dataset in ("ETTh1", "ETTm1"):
        for backbone in BACKBONES:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    cfg = copy.deepcopy(base)
                    cfg["seed"] = seed
                    cfg["data"]["kind"] = "csv"
                    cfg["data"]["path"] = f"external/ett/ETT-small/{dataset}.csv"
                    cfg["data"]["pred_len"] = horizon
                    cfg["backbone"]["name"] = backbone
                    tag = f"{dataset}/{backbone}/h{horizon}/s{seed}"
                    cfg["output_dir"] = f"results/runs/{namespace}/{tag}"
                    paths.append(write_config(cfg, f"{namespace}/{tag}.yaml"))
    return paths


def ttfbench(base: dict) -> list[Path]:
    paths = []
    for dataset in ("ETTh1", "ETTm1"):
        shifted_files = sorted((ROOT / "external" / "dynatta" / "TTFBench" / dataset).glob("*.csv"))
        for shifted in shifted_files:
            for seed in SEEDS:
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                cfg["data"]["kind"] = "ttfbench"
                cfg["data"]["path"] = str(shifted.relative_to(ROOT)).replace("\\", "/")
                # The upstream TTFBench directory named ETTm1 contains ETTm2
                # values; keep the directory label but bind it to its verified
                # clean content instead of silently using the wrong reference.
                reference_dataset = "ETTm2" if dataset == "ETTm1" else dataset
                cfg["data"]["reference_path"] = f"external/ett/ETT-small/{reference_dataset}.csv"
                tag = f"{dataset}/{shifted.stem}/s{seed}"
                cfg["output_dir"] = f"results/runs/ttfbench/{tag}"
                paths.append(write_config(cfg, f"ttfbench/{tag}.yaml"))
    return paths


def multiplex(base: dict) -> list[Path]:
    paths = []
    for complexity in (0.15, 0.35, 0.55, 0.75, 0.90):
        for severity in (0.20, 0.50, 0.80):
            for topology_mode in ("none", "single_layer", "multiplex"):
                for seed in SEEDS:
                    cfg = copy.deepcopy(base)
                    cfg["ugma"].update(
                        risk_monitor_mode="disabled",
                        use_risk_gate=False,
                        risk_loss_weight=0.0,
                    )
                    cfg["seed"] = seed
                    cfg["data"]["network_complexity"] = complexity
                    cfg["data"]["shift_severity"] = severity
                    cfg["ugma"]["topology_mode"] = topology_mode
                    tag = f"c{complexity:.2f}/v{severity:.2f}/{topology_mode}/s{seed}"
                    cfg["output_dir"] = f"results/runs/multiplex/{tag}"
                    paths.append(write_config(cfg, f"multiplex/{tag}.yaml"))
    return paths


def dfm_ablation(base: dict) -> list[Path]:
    paths = []
    variants = (
        ("no_dfm", False, "channel_independent"),
        ("channel_independent", True, "channel_independent"),
        ("sensor_attention", True, "sensor_attention"),
        ("multiplex_graph_attention", True, "multiplex_graph_attention"),
    )
    for label, enabled, mode in variants:
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["ugma"]["use_dfm"] = enabled
            cfg["ugma"]["dfm_mode"] = mode
            tag = f"{label}/s{seed}"
            cfg["output_dir"] = f"results/runs/dfm_ablation/{tag}"
            paths.append(write_config(cfg, f"dfm_ablation/{tag}.yaml"))
    return paths


def retrieval_ablation(base: dict) -> list[Path]:
    paths = []
    variants = {
        "full": {},
        "no_hcb": {"use_hcb": False},
        "no_dfm": {"use_dfm": False},
        "softplus_risk_gate": {
            "risk_monitor_mode": "softplus_mlp",
            "use_risk_gate": True,
            "risk_loss_weight": 1.0,
        },
        "raw_cosine": {"retrieval_encoder": "raw_cosine"},
        "same_sensor": {"retrieval_scope": "same_sensor"},
        "cross_sensor": {"retrieval_scope": "cross_sensor"},
        "top1": {"top_k": 1},
    }
    for label, overrides in variants.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["ugma"].update(overrides)
            tag = f"{label}/s{seed}"
            cfg["output_dir"] = f"results/runs/retrieval_ablation_v3/{tag}"
            paths.append(write_config(cfg, f"retrieval_ablation/{tag}.yaml"))
    return paths


def sensitivity(base: dict) -> list[Path]:
    paths = []
    grid = {
        "bank_size": [64, 128, 256, 512, 1024],
        "top_k": [1, 4, 8, 16, 32],
        "retrieval_temperature": [0.05, 0.10, 0.20, 0.50],
        "temporal_gap": [96, 192, 288, 480],
        "contrastive_temperature": [0.05, 0.10, 0.20, 0.50],
        "gate_bias": [-7.0, -5.0, -3.0],
    }
    for parameter, values in grid.items():
        for value in values:
            for seed in SEEDS:
                cfg = copy.deepcopy(base)
                cfg["seed"] = seed
                cfg["ugma"][parameter] = value
                tag = f"{parameter}/{value}/s{seed}"
                cfg["output_dir"] = f"results/runs/sensitivity/{tag}"
                paths.append(write_config(cfg, f"sensitivity/{tag}.yaml"))
    return paths


def public_ablation(base: dict, namespace: str = "public_ablation") -> list[Path]:
    paths = []
    variants = {
        "final_no_monitor": {},
        "no_hcb": {"use_hcb": False, "gate_bias": -5.0},
        "no_dfm": {"use_dfm": False, "gate_bias": -5.0},
        "softplus_risk_gate": {
            "risk_monitor_mode": "softplus_mlp",
            "use_risk_gate": True,
            "risk_loss_weight": 1.0,
        },
        "submitted_relu_head": {
            "risk_monitor_mode": "manuscript_mlp",
            "use_risk_gate": True,
            "risk_loss_weight": 1.0,
        },
    }
    for label, overrides in variants.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["ugma"].update(overrides)
            tag = f"{label}/s{seed}"
            cfg["output_dir"] = f"results/runs/{namespace}/ETTh1/iTransformer/h96/{tag}"
            paths.append(write_config(cfg, f"{namespace}/ETTh1/iTransformer/h96/{tag}.yaml"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("public", "public_strict", "public_ablation", "public_ablation_strict", "ttfbench", "multiplex", "dfm", "retrieval", "sensitivity", "all"),
        default="all",
    )
    args = parser.parse_args()
    ett = yaml.safe_load((ROOT / "configs" / "ett_base.yaml").read_text(encoding="utf-8"))
    ett_strict = yaml.safe_load((ROOT / "configs" / "ett_strict.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    suites = {
        "public": lambda: public_standard(ett),
        "public_strict": lambda: public_standard(ett_strict, "public_strict"),
        "public_ablation": lambda: public_ablation(ett),
        "public_ablation_strict": lambda: public_ablation(ett_strict, "public_ablation_strict"),
        "ttfbench": lambda: ttfbench(ett),
        "multiplex": lambda: multiplex(smoke),
        "dfm": lambda: dfm_ablation(smoke),
        "retrieval": lambda: retrieval_ablation(ett_strict),
        "sensitivity": lambda: sensitivity(ett_strict),
    }
    suite_directories = {
        "public": "public",
        "public_strict": "public_strict",
        "public_ablation": "public_ablation",
        "public_ablation_strict": "public_ablation_strict",
        "ttfbench": "ttfbench",
        "multiplex": "multiplex",
        "dfm": "dfm_ablation",
        "retrieval": "retrieval_ablation",
        "sensitivity": "sensitivity",
    }
    selected = suites if args.suite == "all" else {args.suite: suites[args.suite]}
    generated = [path for generate in selected.values() for path in generate()]
    expected = {path.resolve() for path in generated}
    for suite_name in selected:
        suite_root = ROOT / "configs" / "generated" / suite_directories[suite_name]
        for stale in suite_root.rglob("*.yaml"):
            if stale.resolve() not in expected:
                stale.unlink()
    print(f"generated {len(generated)} configs under {ROOT / 'configs' / 'generated'}")


if __name__ == "__main__":
    main()
