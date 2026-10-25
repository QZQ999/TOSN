import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("priority", SCRIPTS / "reviewer_priority_20260911.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_no_future_label_access():
    starts = np.arange(1000, 1500)
    for start in starts[::32]:
        end = p.matured_end(starts, start, 96)
        assert np.all(starts[:end] + 96 <= start)
        assert np.all(starts[end:] + 96 > start)


def test_shift_leaves_train_validation_unchanged():
    values = np.random.default_rng(7).normal(size=(300, 3)).astype(np.float32)
    for kind, level in p.CASES:
        changed = p.shifted_values(values, 150, 200, kind, level)
        assert np.array_equal(changed[:200], values[:200])
        assert np.isfinite(changed).all()
        assert not np.shares_memory(changed, values)


def test_constant_risk_has_no_fake_rank_information():
    score = np.zeros((20, 2))
    realized = np.arange(40).reshape(20, 2)
    metrics = p.risk_metrics(score, realized, score)
    assert metrics["spearman"] is None
    assert metrics["constant_score"]
    assert all(np.isclose(row["selective_mae"], realized.mean()) for row in metrics["coverage"])


def test_validation_bins_do_not_depend_on_test_scores():
    vs = np.linspace(0, 1, 100)
    target = np.arange(100)
    a = p.risk_metrics(vs, target, vs)
    b = p.risk_metrics(vs + 100, target, vs)
    assert a["validation_bin_edges"] == b["validation_bin_edges"]


def test_official_gcm_starts_as_identity():
    import torch
    p.OFFICIAL = SCRIPTS.parent / "external/petsa/tta/tafas.py"
    module = p.official_gcm()(96, 7)
    x = torch.randn(3, 96, 7)
    assert torch.equal(x, module(x))


def test_compare_keeps_original_tolerance():
    assert p.compare(np.zeros(2), np.ones(2) * 1e-5)["allclose"]
    assert not p.compare(np.zeros(2), np.ones(2) * 0.01)["allclose"]
