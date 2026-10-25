import importlib.util
import json
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/reviewer_followup.py"
    spec = importlib.util.spec_from_file_location("reviewer_followup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_json_keeps_degenerate_risk_as_null(tmp_path):
    module = load_script()
    path = tmp_path / "result.json"
    module.write_json(path, {"rho": float("nan"), "count": np.int64(5), "nested": [np.float64(0.25)]})
    assert json.loads(path.read_text()) == {"rho": None, "count": 5, "nested": [0.25]}


def test_summary_preserves_negative_gain_and_rmse_definition():
    module = load_script()
    errors = {"baseline": (np.array([1., 1.]), np.array([1., 1.])),
              "full": (np.array([1., 3.]), np.array([1., 9.]))}
    result = module.summarize(errors)
    assert result["full"]["mae"] == 2.0
    assert result["full"]["rmse"] == np.sqrt(5.0)
