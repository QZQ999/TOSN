import numpy as np
import pytest
from pathlib import Path

from ugma.config import DataConfig
from ugma.data import chronological_bounds, generate_multiplex_process, load_ttfbench_bundle, make_datasets


def synthetic_config() -> DataConfig:
    return DataConfig(
        kind="synthetic_multiplex",
        length=600,
        channels=6,
        layers=3,
        seq_len=48,
        pred_len=24,
        train_ratio=0.6,
        val_ratio=0.2,
        batch_size=16,
        network_complexity=0.5,
        shift_start=0.8,
        shift_severity=0.7,
    )


def test_every_target_is_inside_declared_split() -> None:
    cfg = synthetic_config()
    bundle = generate_multiplex_process(cfg, seed=3)
    datasets, _, bounds = make_datasets(bundle, cfg)
    ranges = {"train": (0, bounds.train_end), "val": (bounds.train_end, bounds.val_end), "test": (bounds.val_end, bounds.total)}
    for split, dataset in datasets.items():
        lower, upper = ranges[split]
        assert np.all(dataset.target_starts >= lower)
        assert np.all(dataset.target_starts + cfg.pred_len <= upper)


def test_decimal_split_boundaries_do_not_lose_a_row() -> None:
    bounds = chronological_bounds(17_420, train_ratio=0.7, val_ratio=0.1)
    assert bounds.train_end == 12_194
    assert bounds.val_end == 13_936
    assert bounds.total - bounds.val_end == 3_484


def test_synthetic_process_has_multiplex_layers_and_shift() -> None:
    cfg = synthetic_config()
    bundle = generate_multiplex_process(cfg, seed=4)
    assert bundle.topology.shape == (3, 6, 6)
    assert np.allclose(bundle.topology.sum(axis=-1), 1.0)
    assert bundle.shift_mask[: int(cfg.length * cfg.shift_start)].sum() == 0
    assert bundle.shift_mask[int(cfg.length * cfg.shift_start) :].all()


def test_ttfbench_prefix_validation_exposes_upstream_mislabel() -> None:
    root = Path(__file__).resolve().parents[1]
    shifted = root / "external" / "dynatta" / "TTFBench" / "ETTm1" / "synthetic_multivariate_243.csv"
    if not shifted.exists():
        pytest.skip("official TTFBench checkout is not present")
    common = dict(
        kind="ttfbench", path=str(shifted), timestamp_column="date", seq_len=96,
        pred_len=96, train_ratio=0.7, val_ratio=0.1, batch_size=32,
    )
    wrong = DataConfig(
        **common,
        reference_path=str(root / "external" / "ett" / "ETT-small" / "ETTm1.csv"),
    )
    with pytest.raises(ValueError, match="dataset mislabel"):
        load_ttfbench_bundle(wrong)
    verified = DataConfig(
        **common,
        reference_path=str(root / "external" / "ett" / "ETT-small" / "ETTm2.csv"),
    )
    bundle = load_ttfbench_bundle(verified)
    assert bundle.metadata["clean_prefix_match_fraction"] > 0.99
