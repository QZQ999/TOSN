"""Compare manuscript ETTh1/iTransformer horizon claims with audit runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aggregate" / "horizon_audit"
CLAIMED = {
    96: (0.410, 0.398),
    192: (0.441, 0.427),
    336: (0.465, 0.451),
    720: (0.504, 0.473),
}


def _run_path(horizon: int) -> Path:
    if horizon == 96:
        return (
            ROOT
            / "results/runs/public_ablation/ETTh1/iTransformer/h96/full_bias_m5/s2026/run.json"
        )
    return ROOT / f"results/runs/public/ETTh1/iTransformer/h{horizon}/s2026/run.json"


def main() -> None:
    rows = []
    for horizon, (claimed_backbone, claimed_enhanced) in CLAIMED.items():
        path = _run_path(horizon)
        run = json.loads(path.read_text(encoding="utf-8"))
        result = run["results"]
        backbone = result["backbone"]["mae"]
        enhanced = result["enhanced"]["mae"]
        manifest = json.loads((path.parent / "data_manifest.json").read_text(encoding="utf-8"))
        prediction_count = result["backbone"]["count"] // (horizon * len(manifest["channels"]))
        rows.append(
            {
                "horizon": horizon,
                "claimed_backbone_mae": claimed_backbone,
                "claimed_enhanced_mae": claimed_enhanced,
                "claimed_improvement_percent": 100.0 * (claimed_backbone - claimed_enhanced) / claimed_backbone,
                "audit_backbone_mae": backbone,
                "audit_enhanced_mae": enhanced,
                "audit_improvement_percent": result["improvement_percent"]["mae"],
                "audit_rmse_improvement_percent": result["improvement_percent"]["rmse"],
                "test_windows": int(prediction_count),
                "risk_error_pearson": result["risk_validation"]["pearson_r"],
                "batch1_latency_ms": result["latency_by_batch"]["1"]["latency_ms_mean"],
                "batch32_latency_ms": result["latency_by_batch"]["32"]["latency_ms_mean"],
                "batch32_peak_gpu_bytes": result["latency_by_batch"]["32"]["peak_gpu_bytes"],
                "bank_bytes": result["bank"]["bytes"],
                "enhancement_training_seconds": result["training_seconds_by_stage"]["enhancement"],
                "run_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    payload = {
        "warning": (
            "Audit runs use the initial fixed-10-epoch/LR-1e-4 backbone protocol. "
            "They are diagnostic and must not be called strict reproductions of manuscript values."
        ),
        "runtime_provenance": {
            "git_commit_loaded_at_process_start": "c0897e1e38393b4f0675ba94d7d9e2f08cd906b0",
            "source_sha256_loaded_at_process_start": "017f31e8a614f755c5bb0168a68d97ad8ee423575994abe5d80a72751694c65d",
            "h720_note": (
                "The H=720 process started at this revision, but source files were changed while it ran. "
                "Its run.json may therefore contain a later on-disk digest; this explicit start record governs."
            ),
        },
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
