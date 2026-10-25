"""Aggregate the matched risk-head decision experiment without hiding failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    ROOT / "results/runs/public_ablation_strict/ETTh1/iTransformer/h96"
)
SEEDS = range(2026, 2031)
VARIANTS = ("full_bias_m5", "learned_no_risk", "submitted_relu_head")


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else float("nan"),
    }


def load_variant(root: Path, variant: str, source: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        directory = root / variant / f"s{seed}"
        run_path = directory / "run.json"
        predictions_path = directory / "predictions.npz"
        if not run_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"missing matched artifact: {directory}")
        record = json.loads(run_path.read_text(encoding="utf-8"))
        actual_source = record["environment"].get("source_sha256")
        if actual_source != source:
            raise ValueError(
                f"{run_path}: source {actual_source!r} does not match {source!r}"
            )
        result = record["results"]
        arrays = np.load(predictions_path)
        risk = np.asarray(arrays["predicted_risk"])
        realized = np.asarray(arrays["realized_risk"]).reshape(-1)
        backbone = np.asarray(arrays["backbone"])
        retrieval_similarity = np.asarray(arrays["retrieval_similarity"]).reshape(-1)
        shift_score = np.asarray(arrays["shift_score"]).reshape(-1)
        forecast_roughness = np.abs(np.diff(backbone, axis=1)).mean(axis=1).reshape(-1)
        calibration_counts = result["risk_validation"]["calibration"]["counts"]
        curve = np.asarray(result["risk_validation"]["risk_coverage"]["selective_mae"])
        rows.append(
            {
                "seed": seed,
                "baseline_mae": result["backbone"]["mae"],
                "enhanced_mae": result["enhanced"]["mae"],
                "mae_improvement_percent": result["improvement_percent"]["mae"],
                "pearson": result["risk_validation"]["pearson_r"],
                "spearman": result["risk_validation"]["spearman_rho"],
                "ece": result["risk_validation"]["calibration"]["ece"],
                "occupied_calibration_bins": int(np.count_nonzero(calibration_counts)),
                "risk_exact_zero_fraction": float(np.mean(risk == 0)),
                "risk_below_1e_minus_6_fraction": float(np.mean(risk < 1e-6)),
                "risk_coverage_nondecreasing_fraction": float(np.mean(np.diff(curve) >= 0)),
                "simple_proxy_spearman": {
                    "one_minus_retrieval_similarity": float(
                        spearmanr(1.0 - retrieval_similarity, realized).statistic
                    ),
                    "negative_forecast_roughness": float(
                        spearmanr(-forecast_roughness, realized).statistic
                    ),
                    "shift_score": float(spearmanr(shift_score, realized).statistic),
                },
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    fields = (
        "mae_improvement_percent",
        "pearson",
        "spearman",
        "ece",
        "occupied_calibration_bins",
        "risk_exact_zero_fraction",
        "risk_below_1e_minus_6_fraction",
        "risk_coverage_nondecreasing_fraction",
    )
    summary = {field: mean_sd([float(row[field]) for row in rows]) for field in fields}
    proxy_names = rows[0]["simple_proxy_spearman"]
    summary["simple_proxy_spearman"] = {
        name: mean_sd([row["simple_proxy_spearman"][name] for row in rows])
        for name in proxy_names
    }
    summary["degradation_count"] = sum(row["mae_improvement_percent"] < 0 for row in rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/aggregate/risk_revision_decision/summary.json",
    )
    args = parser.parse_args()
    run_root = args.root if args.root.is_absolute() else ROOT / args.root
    variants = {
        variant: load_variant(run_root, variant, args.source) for variant in VARIANTS
    }
    full = np.asarray([row["enhanced_mae"] for row in variants["full_bias_m5"]])
    no_risk = np.asarray([row["enhanced_mae"] for row in variants["learned_no_risk"]])
    advantage = no_risk - full
    statistic, p_value = wilcoxon(advantage, alternative="two-sided")
    payload = {
        "source_sha256": args.source,
        "run_root": str(run_root),
        "seeds": list(SEEDS),
        "rows": variants,
        "summary": {variant: summarize(rows) for variant, rows in variants.items()},
        "full_risk_vs_no_risk": {
            "positive_means_full_risk_lower_mae": True,
            "mae_advantage": mean_sd(advantage.tolist()),
            "seeds_favoring_full": int(np.sum(advantage > 0)),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p_two_sided": float(p_value),
            "paired_cohens_d": (
                float(advantage.mean() / advantage.std(ddof=1))
                if advantage.std(ddof=1) > 0
                else None
            ),
        },
        "interpretation_rule": (
            "The title and central risk-guided claim require direct discrimination, "
            "non-degenerate calibration/risk-coverage, and matched forecast benefit; "
            "a low ECE from one occupied bin is not sufficient."
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["full_risk_vs_no_risk"], indent=2))


if __name__ == "__main__":
    main()
