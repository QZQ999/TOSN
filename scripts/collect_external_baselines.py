"""Collect auditable ETTh1-96 results from pinned external implementations.

The script does not run or alter any model.  It reads raw arrays produced by the
official PETSA/TAFAS and RAFT implementations plus this repository's U-GMA runs,
checks window counts, and writes a compact comparison with paired block tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ugma.evaluation import paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aggregate" / "external_baselines"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load(path: str) -> np.ndarray:
    return np.load(ROOT / path)


def _window_mae(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    return np.abs(prediction - target).reshape(len(target), -1).mean(axis=1)


def _record(method: str, family: str, errors: np.ndarray, baseline: np.ndarray | None) -> dict:
    row = {
        "method": method,
        "protocol_family": family,
        "windows": int(len(errors)),
        "mae": float(errors.mean()),
        "median_window_mae": float(np.median(errors)),
    }
    if baseline is None:
        row.update({"baseline_mae": None, "mae_reduction": None, "improvement_percent": None})
    else:
        if len(errors) != len(baseline):
            raise ValueError(f"paired length mismatch for {method}: {len(errors)} vs {len(baseline)}")
        reduction = float(baseline.mean() - errors.mean())
        row.update(
            {
                "baseline_mae": float(baseline.mean()),
                "mae_reduction": reduction,
                "improvement_percent": 100.0 * reduction / float(baseline.mean()),
                "paired_statistics": paired_bootstrap(baseline, errors, seed=2026),
            }
        )
    return row


def main() -> None:
    versions = json.loads((ROOT / "external_versions.json").read_text(encoding="utf-8"))

    petsa_root = ROOT / "external" / "petsa" / "results"
    official = np.load(
        petsa_root / "official_baseline" / "iTransformer" / "ETTh1_96" / "test_mae_all.npy"
    )
    petsa = np.load(
        petsa_root
        / "PETSA_RANK_16_LOSSALPHA_0.2_GATINGINIT_0.02"
        / "iTransformer"
        / "ETTh1_96"
        / "adapt_test_mae_all.npy"
    )
    tafas = np.load(
        petsa_root
        / "TAFAS_GATINGINIT_0.01"
        / "iTransformer"
        / "ETTh1_96"
        / "adapt_test_mae_all.npy"
    )
    dynatta = np.load(
        petsa_root
        / "DynaTTA_official"
        / "iTransformer"
        / "ETTh1_96"
        / "adapt_test_mae_all.npy"
    )

    ugma_full_npz = _load(
        "results/runs/public_ablation/ETTh1/iTransformer/h96/full_bias_m5/s2026/predictions.npz"
    )
    ugma_no_risk_npz = _load(
        "results/runs/public_ablation/ETTh1/iTransformer/h96/learned_no_risk/s2026/predictions.npz"
    )
    target = ugma_full_npz["target"]
    if not np.array_equal(target, ugma_no_risk_npz["target"]):
        raise ValueError("U-GMA full and no-risk runs do not share identical targets")
    ugma_backbone = _window_mae(ugma_full_npz["backbone"], target)
    ugma_full = _window_mae(ugma_full_npz["enhanced"], target)
    ugma_no_risk = _window_mae(ugma_no_risk_npz["enhanced"], target)

    raft_setting = next(
        (ROOT / "external" / "raft" / "results").glob("*custom701020*/pred.npy")
    ).parent
    raft_pred = np.load(raft_setting / "pred.npy")
    raft_true = np.load(raft_setting / "true.npy")
    raft = _window_mae(raft_pred, raft_true)

    expected = 3389
    arrays = {
        "official_iTransformer": official,
        "PETSA": petsa,
        "TAFAS": tafas,
        "DynaTTA": dynatta,
        "U-GMA_backbone": ugma_backbone,
        "U-GMA_full": ugma_full,
        "U-GMA_no_risk": ugma_no_risk,
        "RAFT": raft,
    }
    mismatches = {name: len(array) for name, array in arrays.items() if len(array) != expected}
    if mismatches:
        raise ValueError(f"expected {expected} common ETTh1 windows, got {mismatches}")

    rows = [
        _record("official iTransformer (PETSA codebase)", "PETSA official", official, None),
        _record("PETSA", "PETSA official", petsa, official),
        _record("TAFAS", "PETSA official", tafas, official),
        _record("DynaTTA", "DynaTTA adapter in PETSA/TAFAS host", dynatta, official),
        _record("U-GMA backbone", "U-GMA/TSLib", ugma_backbone, None),
        _record("U-GMA full risk gate", "U-GMA/TSLib", ugma_full, ugma_backbone),
        _record("U-GMA learned no-risk gate", "U-GMA/TSLib", ugma_no_risk, ugma_backbone),
        _record("RAFT", "RAFT official fair 70/10/20", raft, None),
    ]
    comparisons = {
        "TAFAS_minus_PETSA": paired_bootstrap(petsa, tafas, seed=2026),
        "DynaTTA_minus_static": paired_bootstrap(official, dynatta, seed=2026),
        "U-GMA_no-risk_minus_full": paired_bootstrap(ugma_full, ugma_no_risk, seed=2026),
    }
    provenance = {
        "dataset": "ETTh1",
        "sequence_length": 96,
        "prediction_length": 96,
        "split_by_protocol": {
            "PETSA_TAFAS_DynaTTA": "60/20/20 chronological (official ETTh1 config)",
            "U-GMA": "70/10/20 chronological",
            "RAFT_fair_run": "70/10/20 chronological",
        },
        "test_windows": expected,
        "external_versions": versions,
        "instrumentation_patch_sha256": _sha256(ROOT / "patches" / "petsa_metrics_export.patch"),
        "raft_compatibility_patch_sha256": _sha256(ROOT / "patches" / "raft_windows_device.patch"),
        "comparability_note": (
            "Absolute MAE across protocol families is descriptive only because official PETSA/TAFAS, "
            "RAFT, and U-GMA train different implementations/hyperparameters. Adaptation gains are paired "
            "only within a shared frozen baseline and identical 3389-window test target."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance, "methods": rows, "paired_comparisons": comparisons}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flat_fields = [
        "method",
        "protocol_family",
        "windows",
        "mae",
        "median_window_mae",
        "baseline_mae",
        "mae_reduction",
        "improvement_percent",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in flat_fields} for row in rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
