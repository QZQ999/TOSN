"""Run the official standalone DynaTTA adapter inside its declared TAFAS stack.

The DynaTTA repository ships only the adapter and a main file whose imports
assume the surrounding TAFAS project.  PETSA contains that same TAFAS stack.
This runner supplies the missing host imports without modifying DynaTTA's
algorithm, then saves its already-computed per-window metrics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PETSA_ROOT = ROOT / "external" / "petsa"
DYNATTA_SOURCE = ROOT / "external" / "dynatta" / "DynaTTA" / "DynaTTA.py"


def _load_official_adapter():
    spec = importlib.util.spec_from_file_location("official_dynatta", DYNATTA_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {DYNATTA_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.chdir(PETSA_ROOT)
    sys.path.insert(0, str(PETSA_ROOT))

    from config import get_norm_module_cfg
    from datasets.build import update_cfg_from_dataset
    from models.build import build_model, build_norm_module, load_best_model
    from utils.misc import set_devices, set_seeds
    from utils.parser import load_config, parse_args

    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    cfg.RESULT_DIR = os.path.join(
        cfg.RESULT_DIR, cfg.TRAIN.CHECKPOINT_DIR.split("./checkpoints/")[-1]
    )
    Path(cfg.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    model = load_best_model(cfg, model)
    if cfg.NORM_MODULE.ENABLE:
        norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)

    official = _load_official_adapter()
    adapter = official.build_adapter(cfg, model, norm_module=norm_module)
    mse, mae = adapter.adapt()
    mse_all = np.asarray(adapter.mse_all)
    mae_all = np.asarray(adapter.mae_all)
    np.save(Path(cfg.RESULT_DIR) / "adapt_test_mse_all.npy", mse_all)
    np.save(Path(cfg.RESULT_DIR) / "adapt_test_mae_all.npy", mae_all)
    source_sha = hashlib.sha256(DYNATTA_SOURCE.read_bytes()).hexdigest()
    metadata = {
        "method": "DynaTTA official adapter",
        "source_sha256": source_sha,
        "source_commit": json.loads((ROOT / "external_versions.json").read_text(encoding="utf-8"))[
            "dynatta"
        ]["commit"],
        "host_stack": "pinned PETSA repository containing the TAFAS modules imported by DynaTTA",
        "windows": int(len(mae_all)),
        "mse": float(mse),
        "mae": float(mae),
        "algorithm_source_modified": False,
    }
    (Path(cfg.RESULT_DIR) / "dynatta_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
