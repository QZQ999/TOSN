"""Aggregate official external methods over the ten TTFBench ETTh1 shifts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "external" / "petsa" / "results"
SHIFT_ROOT = RESULT_ROOT / "ttfbench" / "ETTh1"
OUT = ROOT / "results" / "aggregate" / "ttfbench_external"
METHODS = {
    "static": ("static", "test_mae_all.npy"),
    "PETSA": ("PETSA", "adapt_test_mae_all.npy"),
    "TAFAS": ("TAFAS", "adapt_test_mae_all.npy"),
    "DynaTTA": ("DynaTTA", "adapt_test_mae_all.npy"),
}
CLEAN = {
    "static": RESULT_ROOT / "official_baseline" / "iTransformer" / "ETTh1_96" / "test_mae_all.npy",
    "PETSA": RESULT_ROOT / "PETSA_RANK_16_LOSSALPHA_0.2_GATINGINIT_0.02" / "iTransformer" / "ETTh1_96" / "adapt_test_mae_all.npy",
    "TAFAS": RESULT_ROOT / "TAFAS_GATINGINIT_0.01" / "iTransformer" / "ETTh1_96" / "adapt_test_mae_all.npy",
    "DynaTTA": RESULT_ROOT / "DynaTTA_official" / "iTransformer" / "ETTh1_96" / "adapt_test_mae_all.npy",
}


def _interval(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    clean = {method: np.load(path) for method, path in CLEAN.items()}
    if any(array.shape != (3389,) for array in clean.values()):
        raise ValueError("clean arrays must all contain the same 3389 windows")
    samples = sorted(path for path in SHIFT_ROOT.iterdir() if path.is_dir())
    if len(samples) != 10:
        raise ValueError(f"expected 10 TTFBench sample directories, found {len(samples)}")

    rows: list[dict] = []
    for sample in samples:
        shifted: dict[str, np.ndarray] = {}
        for method, (folder, filename) in METHODS.items():
            path = sample / folder / "iTransformer" / "ETTh1_96" / filename
            values = np.load(path)
            if values.shape != (3389,) or not np.isfinite(values).all():
                raise ValueError(f"invalid array: {path}")
            shifted[method] = values
        baseline_degradation = float(shifted["static"].mean() - clean["static"].mean())
        for method in METHODS:
            shifted_mae = float(shifted[method].mean())
            clean_mae = float(clean[method].mean())
            degradation = shifted_mae - clean_mae
            recovered = baseline_degradation - degradation
            rows.append(
                {
                    "sample": sample.name,
                    "method": method,
                    "clean_mae": clean_mae,
                    "shifted_mae": shifted_mae,
                    "direct_reduction_vs_shifted_static": float(shifted["static"].mean() - shifted_mae),
                    "clean_to_shift_degradation": degradation,
                    "static_clean_to_shift_degradation": baseline_degradation,
                    "absolute_degradation_recovered": recovered,
                    "degradation_recovery_percent": (
                        100.0 * recovered / baseline_degradation if abs(baseline_degradation) > 1e-12 else None
                    ),
                }
            )

    aggregates = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        direct = np.array([row["direct_reduction_vs_shifted_static"] for row in selected])
        degradation = np.array([row["clean_to_shift_degradation"] for row in selected])
        recovered = np.array([row["absolute_degradation_recovered"] for row in selected])
        shifted = np.array([row["shifted_mae"] for row in selected])
        ci_low, ci_high = _interval(recovered)
        if method == "static":
            statistic, p_value = None, None
        else:
            test = stats.wilcoxon(direct, alternative="greater")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        aggregates.append(
            {
                "method": method,
                "shifted_mae_mean": float(shifted.mean()),
                "shifted_mae_std": float(shifted.std(ddof=1)),
                "direct_reduction_vs_shifted_static_mean": float(direct.mean()),
                "clean_to_shift_degradation_mean": float(degradation.mean()),
                "absolute_degradation_recovered_mean": float(recovered.mean()),
                "recovered_mean_ci95_low": ci_low,
                "recovered_mean_ci95_high": ci_high,
                "ratio_of_mean_degradation_recovered_percent": float(
                    100.0
                    * recovered.mean()
                    / np.mean([row["static_clean_to_shift_degradation"] for row in selected])
                ),
                "negative_recovery_samples": int((recovered < 0).sum()),
                "wilcoxon_direct_vs_shifted_static_statistic": statistic,
                "wilcoxon_direct_vs_shifted_static_p_one_sided": p_value,
            }
        )

    payload = {
        "protocol": {
            "dataset": "TTFBench/ETTh1 (10 official synthetic_multivariate shift files)",
            "model": "one fixed official PETSA iTransformer ETTh1-96 checkpoint",
            "training_split": "60/20/20 chronological (official PETSA ETTh1 config)",
            "test_windows_per_sample": 3389,
            "inference": "online TTA for PETSA/TAFAS/DynaTTA; no adaptation for static",
            "statistical_unit": "10 shift files; overlapping windows are not treated as independent replicates",
        },
        "aggregate": aggregates,
        "per_sample": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
