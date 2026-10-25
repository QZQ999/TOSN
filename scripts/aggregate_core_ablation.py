"""Create the paired five-seed ETTh1 risk-gate ablation summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs" / "public_ablation" / "ETTh1" / "iTransformer" / "h96"
OUT = ROOT / "results" / "aggregate" / "core_risk_ablation"


def main() -> None:
    rows = []
    for seed in range(2026, 2031):
        full = json.loads((RUN_ROOT / "full_bias_m5" / f"s{seed}" / "run.json").read_text(encoding="utf-8"))
        no_risk = json.loads((RUN_ROOT / "learned_no_risk" / f"s{seed}" / "run.json").read_text(encoding="utf-8"))
        full_results, no_risk_results = full["results"], no_risk["results"]
        full_key = full["environment"]["backbone_cache"]["key"]
        no_risk_key = no_risk["environment"]["backbone_cache"]["key"]
        if full_key != no_risk_key:
            raise ValueError(f"seed {seed} does not use a shared backbone cache")
        backbone = full_results["backbone"]["mae"]
        if backbone != no_risk_results["backbone"]["mae"]:
            raise ValueError(f"seed {seed} backbone metrics differ")
        rows.append(
            {
                "seed": seed,
                "backbone_mae": backbone,
                "full_risk_mae": full_results["enhanced"]["mae"],
                "learned_no_risk_mae": no_risk_results["enhanced"]["mae"],
                "full_improvement_percent": full_results["improvement_percent"]["mae"],
                "no_risk_improvement_percent": no_risk_results["improvement_percent"]["mae"],
                "risk_advantage_mae": no_risk_results["enhanced"]["mae"] - full_results["enhanced"]["mae"],
                "risk_error_pearson": full_results["risk_validation"]["pearson_r"],
                "risk_error_spearman": full_results["risk_validation"]["spearman_rho"],
                "backbone_cache_key": full_key,
            }
        )

    advantage = np.array([row["risk_advantage_mae"] for row in rows])
    full_improvement = np.array([row["full_improvement_percent"] for row in rows])
    no_risk_improvement = np.array([row["no_risk_improvement_percent"] for row in rows])
    rng = np.random.default_rng(2026)
    bootstrap = advantage[rng.integers(0, len(advantage), size=(100_000, len(advantage)))].mean(axis=1)
    test = stats.wilcoxon(advantage, alternative="two-sided")
    summary = {
        "dataset": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "seeds": [row["seed"] for row in rows],
        "full_improvement_percent_mean": float(full_improvement.mean()),
        "full_improvement_percent_std": float(full_improvement.std(ddof=1)),
        "no_risk_improvement_percent_mean": float(no_risk_improvement.mean()),
        "no_risk_improvement_percent_std": float(no_risk_improvement.std(ddof=1)),
        "risk_advantage_mae_mean": float(advantage.mean()),
        "risk_advantage_mae_median": float(np.median(advantage)),
        "risk_advantage_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "risk_advantage_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "risk_advantage_paired_cohens_d": float(advantage.mean() / advantage.std(ddof=1)),
        "full_better_seeds": int((advantage > 0).sum()),
        "no_risk_better_seeds": int((advantage < 0).sum()),
        "wilcoxon_statistic_two_sided": float(test.statistic),
        "wilcoxon_p_two_sided": float(test.pvalue),
        "interpretation": (
            "Risk estimates correlate with realized backbone error, but the five-seed paired ablation "
            "does not show additional forecasting benefit from feeding risk into the gates."
        ),
    }
    payload = {"summary": summary, "per_seed": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
