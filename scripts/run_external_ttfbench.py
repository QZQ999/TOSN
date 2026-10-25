"""Run official static/PETSA/TAFAS/DynaTTA inference on TTFBench ETTh1 files."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PETSA = ROOT / "external" / "petsa"
SHIFT_ROOT = ROOT / "external" / "dynatta" / "TTFBench" / "ETTh1"
CLEAN = ROOT / "external" / "ett" / "ETT-small" / "ETTh1.csv"
CHECKPOINT = "./checkpoints/iTransformer/ETTh1_96"


def _host_file(sample: Path) -> Path:
    target = ROOT / "artifacts" / "ttfbench_hosts" / sample.stem / "ETTh1" / "ETTh1.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(sample, target)
        except OSError:
            shutil.copy2(sample, target)
    return target


def _validate_prefix(sample: Path) -> None:
    shifted = pd.read_csv(sample).iloc[:, 1:].to_numpy(dtype=np.float32)
    clean = pd.read_csv(CLEAN).iloc[:, 1:].to_numpy(dtype=np.float32)
    test_start = len(clean) - int(len(clean) * 0.2)
    matched = np.isclose(shifted[:test_start], clean[:test_start], rtol=1e-5, atol=1e-6).mean()
    if matched < 0.99:
        raise ValueError(f"{sample.name}: clean prefix match only {matched:.3%}")


def _common(host_root: Path, result_dir: Path) -> list[str]:
    return [
        "DATA.NAME", "ETTh1",
        "DATA.BASE_DIR", str(host_root),
        "DATA.PRED_LEN", "96",
        "MODEL.NAME", "iTransformer",
        "MODEL.pred_len", "96",
        "TRAIN.ENABLE", "False",
        "TRAIN.CHECKPOINT_DIR", CHECKPOINT,
        "TEST.BATCH_SIZE", "4096",
        "TRAIN.BATCH_SIZE", "256",
        "NUM_GPUS", "1",
        "VISIBLE_DEVICES", "0",
        "DATA_LOADER.NUM_WORKERS", "0",
        "RESULT_DIR", str(result_dir),
    ]


def _run(sample: Path, method: str) -> None:
    host = _host_file(sample)
    directory_label = {"static": "static", "petsa": "PETSA", "tafas": "TAFAS", "dynatta": "DynaTTA"}[method]
    base = PETSA / "results" / "ttfbench" / "ETTh1" / sample.stem / directory_label
    final = base / "iTransformer" / "ETTh1_96"
    expected = final / ("test_mae_all.npy" if method == "static" else "adapt_test_mae_all.npy")
    if expected.exists():
        print(f"SKIP {sample.stem} {method}", flush=True)
        return
    common = _common(host.parents[1], base)
    if method == "static":
        command = [sys.executable, "main.py", *common, "TTA.ENABLE", "False", "TEST.ENABLE", "True"]
    elif method == "petsa":
        command = [
            sys.executable, "main.py", *common,
            "TTA.ENABLE", "True", "TTA.SOLVER.BASE_LR", "0.001",
            "TTA.SOLVER.WEIGHT_DECAY", "0.0", "TTA.PETSA.GATING_INIT", "0.02",
            "TTA.PETSA.RANK", "16", "TTA.PETSA.LOSS_ALPHA", "0.2",
            "TEST.ENABLE", "False",
        ]
    elif method == "tafas":
        command = [
            sys.executable, "main.py", *common,
            "TTA.ENABLE", "True", "TTA.SOLVER.BASE_LR", "0.001",
            "TTA.SOLVER.WEIGHT_DECAY", "0.0", "TTA.TAFAS.GATING_INIT", "0.01",
            "TEST.ENABLE", "False",
        ]
    elif method == "dynatta":
        command = [
            sys.executable, str(ROOT / "scripts" / "run_dynatta_official.py"), *common,
            "TTA.ENABLE", "True", "TTA.SOLVER.BASE_LR", "0.005",
            "TTA.SOLVER.WEIGHT_DECAY", "0.0", "TTA.TAFAS.GATING_INIT", "0.01",
            "TEST.ENABLE", "False",
        ]
    else:
        raise ValueError(method)
    print(f"START {sample.stem} {method}", flush=True)
    completed = subprocess.run(command, cwd=PETSA)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if not expected.exists():
        raise FileNotFoundError(f"expected output was not produced: {expected}")
    values = np.load(expected)
    if values.shape != (3389,) or not np.isfinite(values).all():
        raise ValueError(f"invalid output {expected}: {values.shape}")
    print(f"DONE {sample.stem} {method} MAE={values.mean():.8f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--methods", nargs="+", choices=("static", "petsa", "tafas", "dynatta"),
        default=("static", "petsa", "tafas", "dynatta"),
    )
    args = parser.parse_args()
    samples = sorted(SHIFT_ROOT.glob("*.csv"))
    if args.limit is not None:
        samples = samples[: args.limit]
    for sample in samples:
        _validate_prefix(sample)
        for method in args.methods:
            _run(sample, method)


if __name__ == "__main__":
    main()
