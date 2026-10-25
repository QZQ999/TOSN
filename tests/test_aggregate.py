import json

from ugma.aggregate import flatten_run


def test_flatten_run_preserves_training_and_provenance_fields(tmp_path) -> None:
    run = {
        "config": {
            "seed": 2026,
            "data": {"kind": "csv", "path": "ETTh1.csv", "pred_len": 96},
            "backbone": {"name": "iTransformer"},
            "ugma": {
                "use_hcb": True,
                "use_dfm": True,
                "use_risk_gate": True,
                "top_k": 16,
                "bank_size": 512,
                "retrieval_temperature": 0.1,
                "temporal_gap": 192,
                "risk_loss_weight": 1.0,
            },
        },
        "environment": {"git_commit": "abc", "source_sha256": "def"},
        "training": [
            {"stage": "backbone", "epoch": 1},
            {"stage": "risk_monitor", "epoch": 1},
            {"stage": "enhancement", "epoch": 1},
            {"stage": "enhancement", "epoch": 2},
        ],
        "results": {
            "backbone": {"mae": 1.0, "rmse": 2.0},
            "enhanced": {"mae": 0.9, "rmse": 1.8},
            "improvement_percent": {"mae": 10.0, "rmse": 10.0},
            "risk_validation": {
                "pearson_r": 0.5,
                "spearman_rho": 0.4,
                "calibration": {"ece": 0.1},
            },
            "efficiency": {
                "latency_ms_per_sample": 1.0,
                "peak_gpu_bytes": 123,
                "parameters_total": 100,
                "parameters_trainable": 50,
            },
            "latency_by_batch": {
                "1": {"latency_ms_mean": 3.0},
                "16": {"latency_ms_mean": 4.0},
                "32": {"latency_ms_mean": 5.0},
            },
            "bank": {"bytes": 456, "build_seconds": 0.5},
            "training_seconds_by_stage": {
                "backbone": 1.0,
                "risk_monitor": 2.0,
                "enhancement": 3.0,
            },
            "training_peak_gpu_bytes_by_stage": {
                "backbone": 10,
                "risk_monitor": 20,
                "enhancement": 30,
            },
        },
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")

    row = flatten_run(path)

    assert row["enhancement_epochs_run"] == 2
    assert row["enhancement_training_seconds"] == 3.0
    assert row["enhancement_training_peak_gpu_bytes"] == 30
    assert row["latency_batch32_ms"] == 5.0
    assert row["source_sha256"] == "def"
