"""Isolated reviewer diagnostics; never overwrite the reconstructed model source.

The paper author's original implementation was NOT supplied. These experiments
test an explicitly identified reconstruction and its controls, not author code.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = "636807134ff574f308266a22cef002d41d90a1960a4522075ffbd941d8d96a45"
OUT = ROOT / "results/reviewer_followup_20260909"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    def finite(obj):
        if isinstance(obj, dict):
            return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [finite(v) for v in obj]
        if hasattr(obj, "item"):
            return finite(obj.item())
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    temp.write_text(json.dumps(finite(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def initialize():
    import torch
    from ugma.cli import _source_digest
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if _source_digest() != SOURCE:
        raise RuntimeError("Model source changed: preserve and audit the new version before resuming")
    if not torch.cuda.is_available():
        raise RuntimeError("These paired diagnostics require the original CUDA host")


def load_model(path):
    import torch
    from ugma.config import parse_config
    from ugma.data import load_bundle, make_datasets
    from ugma.models import UGMAModel, build_backbone
    from ugma.training import seed_everything
    path = ROOT / path
    run = json.loads(path.read_text(encoding="utf-8"))
    if run["environment"]["source_sha256"] != SOURCE:
        raise RuntimeError("Wrong checkpoint generation")
    cfg = parse_config(run["config"])
    seed_everything(cfg.seed)
    bundle = load_bundle(cfg.data, cfg.seed)
    datasets, scaler, bounds = make_datasets(bundle, cfg.data)
    channels = bundle.values.shape[1]
    backbone = build_backbone(cfg.backbone.name, cfg.data.seq_len, cfg.data.pred_len,
                              channels, cfg.backbone.hidden_dim, ROOT)
    kwargs = {k: getattr(cfg.ugma, k) for k in inspect.signature(UGMAModel).parameters
              if hasattr(cfg.ugma, k)}
    kwargs.update(backbone=backbone, seq_len=cfg.data.seq_len, pred_len=cfg.data.pred_len,
                  channels=channels, topology=torch.empty((0, channels, channels)))
    model = UGMAModel(**kwargs)
    state = torch.load(path.with_name("checkpoint.pt"), map_location="cpu", weights_only=True)["ugma"]
    for name in ("keys", "histories", "futures", "nodes", "target_starts"):
        model.bank._buffers[name] = torch.empty_like(state["bank." + name])
    model.load_state_dict(state, strict=True)
    model.to("cuda").eval()
    return model, cfg, bundle, datasets, scaler, bounds, run


def batch_items(dataset, indices):
    import torch
    rows = [dataset[int(i)] for i in indices]
    return (torch.stack([r["x"] for r in rows]).cuda(),
            torch.stack([r["y"] for r in rows]).cuda(),
            torch.stack([r["target_start"] for r in rows]).cuda())


def metric_errors(y, prediction):
    error = (y - prediction).double()
    return error.abs().mean(dim=1).cpu().numpy(), error.square().mean(dim=1).cpu().numpy()


def summarize(errors):
    import numpy as np
    return {key: {"mae": float(a.mean()), "rmse": float(np.sqrt(s.mean()))}
            for key, (a, s) in errors.items()}


def clean_diagnostic(job, output):
    import numpy as np
    import torch
    model, cfg, bundle, datasets, _, _, run = load_model(job["source_run"])
    source = ROOT / job["source_run"]
    with np.load(source.with_name("predictions.npz"), allow_pickle=False) as archive:
        saved = {k: archive[k] for k in ("target", "backbone", "enhanced")}
    dataset = datasets["test"]
    if len(dataset) != len(saved["target"]):
        raise RuntimeError("Test partition differs")
    full = job["kind"] == "controls"
    ids = np.arange(len(dataset)) if full else np.unique(np.linspace(0, len(dataset)-1, 128, dtype=int))
    original_forward = model.backbone.forward
    chunks = {}
    max_error = 0.0
    with torch.inference_mode():
        for begin in range(0, len(ids), 32):
            chosen = ids[begin:begin+32]
            x, y, starts = batch_items(dataset, chosen)
            baseline = torch.from_numpy(saved["backbone"][chosen]).cuda()
            model.backbone.forward = lambda history, value=baseline: value
            enhanced, aux = model(x, starts)
            if not np.array_equal(y.cpu().numpy(), saved["target"][chosen]):
                raise RuntimeError("Targets differ from saved experiment")
            actual = enhanced.cpu().numpy()
            max_error = max(max_error, float(np.max(np.abs(actual-saved["enhanced"][chosen]))))
            if not np.allclose(actual, saved["enhanced"][chosen], rtol=2e-4, atol=2e-4):
                raise RuntimeError(f"Enhancer replay mismatch {max_error}")
            if full:
                scale, mean = aux["normalization_scale"], aux["normalization_mean"]
                local = aux["alpha"][:, None, :] * aux["local"] * scale
                retrieval = aux["beta"][:, None, :] * (aux["reference"]-aux["normalized_backbone"]) * scale
                predictions = {"backbone": baseline, "full": enhanced,
                               "local_only_instant": baseline + local,
                               "retrieval_only_instant": baseline + retrieval,
                               "topk_reference_only": aux["reference"] * scale + mean}
                old_k = model.bank.top_k
                model.bank.top_k = 1
                nearest = model.bank.retrieve(model.encoder(model.normalize_history(x)[0]), starts)
                model.bank.top_k = old_k
                predictions["nearest_neighbor_only"] = torch.where(
                    nearest.valid_candidates.any(-1)[:, None, :],
                    nearest.reference * scale + mean, baseline)
                for key, value in predictions.items():
                    chunks.setdefault(key, []).append(metric_errors(y, value))
    model.backbone.forward = original_forward
    result = {"source_run": str(source), "checkpoint_sha256": digest(source.with_name("checkpoint.pt")),
              "source_sha256": SOURCE, "kind": job["kind"], "identity": job["identity"],
              "replay_max_absolute_error": max_error, "windows": len(ids),
              "interpretation": "reconstructed implementation; inference removal is not retrained ablation"}
    if full:
        errors = {k: (np.concatenate([p[0] for p in v]), np.concatenate([p[1] for p in v]))
                  for k, v in chunks.items()}
        result["metrics"] = summarize(errors)
        baseline = errors["backbone"][0]
        result["quality_groups"] = []
        masks = np.empty(baseline.shape, dtype=np.int8)
        for channel in range(baseline.shape[1]):
            for label, group in enumerate(np.array_split(np.argsort(baseline[:, channel], kind="stable"), 3)):
                masks[group, channel] = label
        for label, name in enumerate(("good", "middle", "poor")):
            mask = masks == label
            result["quality_groups"].append({"quality": name, "units": int(mask.sum()),
                "metrics": {k: {"mae": float(a[mask].mean()), "rmse": float(np.sqrt(s[mask].mean()))}
                            for k, (a, s) in errors.items()}})
        np.savez_compressed(output / "window_errors.npz", quality=masks,
                            **{k+"_mae": a for k, (a, _) in errors.items()},
                            **{k+"_mse": s for k, (_, s) in errors.items()})
    else:
        result["hardware"] = torch.cuda.get_device_name(0)
        result["parameters"] = {"backbone": sum(p.numel() for p in model.backbone.parameters()),
                                "full": sum(p.numel() for p in model.parameters())}
        result["latency"] = []
        with torch.inference_mode():
            for size in (1, 16, 32):
                x, _, _ = batch_items(dataset, np.arange(size))
                for _ in range(10):
                    model.backbone(x)
                    model(x)
                timings = {"backbone": [], "full": []}
                memory = {}
                for repetition in range(50):
                    order = ("backbone", "full") if repetition % 2 == 0 else ("full", "backbone")
                    for name in order:
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        prediction = model.backbone(x) if name == "backbone" else model(x)
                        end.record()
                        torch.cuda.synchronize()
                        timings[name].append(start.elapsed_time(end))
                        memory[name] = max(memory.get(name, 0), torch.cuda.max_memory_allocated()-before)
                        del prediction
                means = {k: float(np.mean(v)) for k, v in timings.items()}
                result["latency"].append({"batch": size, "mean_ms": means,
                    "std_ms": {k: float(np.std(v, ddof=1)) for k, v in timings.items()},
                    "extra_ms": means["full"]-means["backbone"],
                    "extra_percent": 100*(means["full"] / means["backbone"]-1),
                    "incremental_peak_allocated_bytes": memory,
                    "memory_note": "activation increment; full model resident in both cases",
                    "timings_ms": timings})
    write_json(output / "result.json", result)


def shift_diagnostic(job, output):
    import numpy as np
    import torch
    from ugma.data import WindowDataset
    model, cfg, bundle, datasets, scaler, bounds, run = load_model(job["source_run"])
    source = ROOT / job["source_run"]
    with np.load(source.with_name("predictions.npz"), allow_pickle=False) as archive:
        saved = {k: archive[k] for k in ("target", "backbone", "enhanced")}
    n, c = bundle.values.shape
    onset = bounds.val_end
    train_mean = bundle.values[:bounds.train_end].mean(0)
    train_std = bundle.values[:bounds.train_end].std(0)
    direction = np.where(np.arange(c) % 2 == 0, 1.0, -1.0)
    results = []
    cases = [("clean", 0)] + [(kind, level) for kind in ("mean", "scale", "ramp") for level in (0.5, 1.0)]
    for kind, level in cases:
        values = bundle.values.copy()
        if kind == "mean":
            values[onset:] += level * train_std * direction
        elif kind == "scale":
            values[onset:] = train_mean + (1+level) * (values[onset:]-train_mean)
        elif kind == "ramp":
            values[onset:] += np.linspace(0, level, n-onset)[:, None] * train_std * direction
        if not np.array_equal(values[:onset], bundle.values[:onset]):
            raise RuntimeError("Controlled shift contaminated training/validation")
        dataset = WindowDataset(scaler.transform(values), cfg.data.seq_len, cfg.data.pred_len,
                                onset, n, np.arange(n) >= onset)
        chunks = {k: [] for k in ("backbone", "full")}
        with torch.inference_mode():
            for begin in range(0, len(dataset), 32):
                ids = np.arange(begin, min(begin+32, len(dataset)))
                x, y, starts = batch_items(dataset, ids)
                enhanced, aux = model(x, starts)
                if kind == "clean":
                    for key, actual in (("target", y), ("backbone", aux["backbone"]), ("enhanced", enhanced)):
                        if not np.allclose(actual.cpu().numpy(), saved[key][ids], rtol=2e-4, atol=2e-4):
                            raise RuntimeError(f"Full CUDA clean replay mismatch: {key}")
                for key, prediction in (("backbone", aux["backbone"]), ("full", enhanced)):
                    chunks[key].append(metric_errors(y, prediction))
        errors = {k: (np.concatenate([v[0] for v in parts]), np.concatenate([v[1] for v in parts]))
                  for k, parts in chunks.items()}
        metrics = summarize(errors)
        row = {"kind": kind, "severity": level, "metrics": metrics, "windows": len(dataset)}
        if results:
            base_delta = metrics["backbone"]["mae"]-results[0]["metrics"]["backbone"]["mae"]
            full_delta = metrics["full"]["mae"]-results[0]["metrics"]["full"]["mae"]
            row["shift_effect"] = {"backbone_mae_increase": base_delta,
                                   "full_mae_increase": full_delta,
                                   "absolute_degradation_recovered": base_delta-full_delta}
        results.append(row)
        np.savez_compressed(output / f"{kind}_{level}_window_errors.npz",
                            **{k+"_mae": a for k, (a, _) in errors.items()},
                            **{k+"_mse": s for k, (_, s) in errors.items()})
        print(f"SHIFT {job['id']} {kind} {level} {metrics}", flush=True)
    write_json(output / "result.json", {"identity": job["identity"], "source_sha256": SOURCE,
        "source_run": str(source), "checkpoint_sha256": digest(source.with_name("checkpoint.pt")),
        "protocol": "predeclared test-only mean/scale/ramp stress; no retraining or test-based selection",
        "limitations": "synthetic perturbations of public data, not TTFBench or real industrial incidents",
        "onset_row": onset, "cases": results})


def train_control(job, output):
    import torch
    import yaml
    import ugma.cli as cli
    from ugma.models import UGMAModel
    cfg = copy.deepcopy(json.loads((ROOT / job["source_run"]).read_text(encoding="utf-8"))["config"])
    cfg["output_dir"] = str(output / "training")
    cfg["ugma"].update(job.get("overrides", {}))
    variant = job["variant"]
    if variant in ("fixed_initial_gates", "learned_constant_gate"):
        class DiagnosticModel(UGMAModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                if variant == "fixed_initial_gates":
                    for module in (self.local_gate, self.global_gate):
                        for parameter in module.parameters():
                            parameter.requires_grad_(False)
                else:
                    original = self.global_gate.forward
                    self.global_gate.forward = lambda inputs: original(torch.zeros_like(inputs))
        cli.UGMAModel = DiagnosticModel
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    script_hash = digest(__file__)
    write_json(output / "protocol.json", {"variant": variant, "base_source_sha256": SOURCE,
        "diagnostic_script_sha256": script_hash, "overrides": job.get("overrides", {}),
        "runtime_override": variant if variant in ("fixed_initial_gates", "learned_constant_gate") else None,
        "warning": "base source hash alone does not identify runtime gate controls",
        "interpretation": "reconstructed model control, not original author implementation"})
    run_path = cli.run_experiment(config_path)
    record = json.loads(Path(run_path).read_text(encoding="utf-8"))
    write_json(output / "result.json", {"identity": job["identity"], "variant": variant,
        "source_run": job["source_run"], "run_path": str(run_path),
        "base_source_sha256": SOURCE, "diagnostic_script_sha256": script_hash,
        "metrics": {k: record["results"][k] for k in ("backbone", "enhanced", "improvement_percent", "risk_validation")}})


def sensitivity_summary(output):
    import pandas as pd
    rows = []
    for path in (ROOT / "results/runs/sensitivity").rglob("run.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["environment"]["source_sha256"] != SOURCE:
            continue
        cfg = run["config"]
        row = {"sweep": path.parents[2].name,
               "config_identity": json.dumps({k: v for k, v in cfg.items() if k not in ("seed", "output_dir")}, sort_keys=True),
               "seed": cfg["seed"], "run_path": str(path),
               "improvement_percent": run["results"]["improvement_percent"]["mae"]}
        for key in ("bank_size", "top_k", "contrastive_temperature", "retrieval_temperature", "temporal_gap", "gate_bias"):
            row[key] = cfg["ugma"][key]
        rows.append(row)
    if len(rows) != 125:
        raise RuntimeError(f"Expected 125 source-matched sensitivity results; found {len(rows)}")
    frame = pd.DataFrame(rows)
    groups = []
    for (sweep, _), group in frame.groupby(["sweep", "config_identity"]):
        if len(group) != 5 or group.seed.nunique() != 5:
            raise RuntimeError("Sensitivity grouping is not a five-seed configuration")
        row = {key: group.iloc[0][key] for key in ("bank_size", "top_k", "contrastive_temperature", "retrieval_temperature", "temporal_gap", "gate_bias")}
        groups.append({**row, "sweep": sweep, "n": 5, "mean": float(group.improvement_percent.mean()),
                       "std": float(group.improvement_percent.std(ddof=1))})
    frame.to_csv(output / "all_runs.csv", index=False)
    pd.DataFrame(groups).to_csv(output / "groups.csv", index=False)
    write_json(output / "result.json", {"runs": len(rows), "groups": groups,
        "grouping": "sweep plus complete configuration except seed/output; repeated defaults are not pooled; includes contrastive temperature"})


def raft_control(job, output):
    """Official RAFT architecture in the same train/validation/test harness.

    This is not an unchanged upstream end-to-end run: optimizer budget and
    preprocessing match our paired evaluation and are recorded explicitly.
    """
    import importlib.util
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset
    from ugma.config import parse_config
    from ugma.data import load_bundle, make_datasets
    from ugma.training import seed_everything
    original = json.loads((ROOT / job["source_run"]).read_text(encoding="utf-8"))
    cfg = parse_config(original["config"])
    seed_everything(cfg.seed)
    bundle = load_bundle(cfg.data, cfg.seed)
    datasets, _, bounds = make_datasets(bundle, cfg.data)
    official_root = ROOT / "external/raft_reviewer"
    model_path = official_root / "models/RAFT.py"
    # A fresh worker has not imported TSLib; enforce this to prevent 'layers'
    # from accidentally resolving to a different external implementation.
    if "layers" in sys.modules:
        raise RuntimeError("RAFT must run in its own process")
    sys.path.insert(0, str(official_root))
    source = model_path.read_text(encoding="utf-8")
    if source.count("index = index.to(self.device)") != 1:
        raise RuntimeError("Pinned RAFT device patch no longer matches")
    source = source.replace("index = index.to(self.device)", "index = index.cpu()")
    spec = importlib.util.spec_from_file_location("reviewer_official_raft", model_path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(model_path), "exec"), module.__dict__)
    from layers.Retrieval import RetrievalTool

    class Adapter(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.pred_len = cfg.data.pred_len
            self.label_len = 0
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            row = self.dataset[index]
            return index, row["x"], row["y"], torch.zeros(1), torch.zeros(1)

    def bounded_retrieve_all(tool, data, train=False, device=torch.device("cpu")):
        pieces = []
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for ids, x, _, _, _ in loader:
                pieces.append(tool.retrieve(x.float().to(device), ids, train=train).cpu())
        return torch.cat(pieces, dim=1)
    RetrievalTool.retrieve_all = bounded_retrieve_all
    args = SimpleNamespace(gpu=0, task_name="long_term_forecast", seq_len=cfg.data.seq_len,
                           pred_len=cfg.data.pred_len, enc_in=bundle.values.shape[1], n_period=3, topm=20)
    model = module.Model(args).cuda()
    adapted = {key: Adapter(value) for key, value in datasets.items()}
    started = time.perf_counter()
    model.prepare_dataset(adapted["train"], adapted["val"], adapted["test"])
    bank_seconds = time.perf_counter()-started
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {key: DataLoader(value, batch_size=32, shuffle=key == "train", num_workers=0,
                               generator=generator if key == "train" else None)
               for key, value in adapted.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.backbone.learning_rate,
                                 weight_decay=cfg.backbone.weight_decay)
    best, stale, state, history = float("inf"), 0, None, []
    for epoch in range(1, cfg.backbone.epochs+1):
        model.train()
        total, units = 0., 0
        for ids, x, y, _, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            y = y.cuda()
            prediction = model(x.cuda(), ids, mode="train")
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            total += loss.item()*y.numel()
            units += y.numel()
        model.eval()
        validation, count = 0., 0
        with torch.inference_mode():
            for ids, x, y, _, _ in loaders["val"]:
                error = model(x.cuda(), ids, mode="valid")-y.cuda()
                validation += error.square().sum().item()
                count += error.numel()
        validation /= count
        history.append({"epoch": epoch, "train_mse": total/units, "val_mse": validation})
        print(f"RAFT {job['id']} epoch={epoch} val_mse={validation}", flush=True)
        if validation < best:
            best, stale = validation, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.backbone.early_stopping_patience:
            break
    model.load_state_dict(state)
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for ids, x, y, _, _ in loaders["test"]:
            predictions.append(model(x.cuda(), ids, mode="test").cpu().numpy())
            targets.append(y.numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    with np.load((ROOT/job["source_run"]).with_name("predictions.npz"), allow_pickle=False) as saved:
        if not np.array_equal(target, saved["target"]):
            raise RuntimeError("RAFT matched harness test targets differ")
    error = target.astype(np.float64)-prediction.astype(np.float64)
    np.savez_compressed(output / "predictions.npz", prediction=prediction, target=target)
    torch.save(state, output / "checkpoint.pt")
    write_json(output / "result.json", {"identity": job["identity"], "method": "RAFT official architecture, matched harness",
        "official_commit": "bfe2320f5d719d1e2a024330de3410b370bbb2a9",
        "model_file_sha256": digest(model_path), "retrieval_file_sha256": digest(official_root/"layers/Retrieval.py"),
        "diagnostic_script_sha256": digest(__file__), "source_run_for_targets": job["source_run"],
        "metrics": {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error**2).mean()))},
        "history": history, "bank_seconds": bank_seconds, "total_seconds": time.perf_counter()-started,
        "training": {"max_epochs": cfg.backbone.epochs, "patience": cfg.backbone.early_stopping_patience,
                     "lr": cfg.backbone.learning_rate, "weight_decay": cfg.backbone.weight_decay,
                     "batch": 32, "selection": "validation MSE only; test evaluated once"},
        "protocol_notes": ["same train-only outer standardizer, windows, and 70/10/20 split",
            "official RAFT architecture and multigranularity retrieval; model is not a frozen-backbone plug-in",
            "CPU retrieval-index compatibility fix; precomputation batch 256 and workers 0",
            "official train-window mask retained; no claim identical candidate pool to reconstructed HCB"]})


def prepare():
    with (ROOT / "results/aggregate/public_strict/all_runs.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80 or any(row["source_sha256"] != SOURCE for row in rows):
        raise RuntimeError("Expected all 80 source-matched main configurations")
    rows.sort(key=lambda r: (int(r["horizon"]), Path(r["dataset_path"]).stem,
                             r["backbone"] != "iTransformer", int(r["seed"])))
    jobs = [{"id": "sensitivity_summary", "kind": "sensitivity"}]
    # Interleave one job per family first so protocol faults surface early.
    families = {k: [] for k in ("latency", "controls", "shift", "train", "raft")}
    for row in rows:
        identity = {"dataset": Path(row["dataset_path"]).stem, "backbone": row["backbone"],
                    "horizon": int(row["horizon"]), "seed": int(row["seed"])}
        tag = f"{identity['dataset']}_{identity['backbone']}_h{identity['horizon']}_s{identity['seed']}"
        common = {"source_run": row["run_path"], "identity": identity}
        families["latency"].append({**common, "id": "latency_"+tag, "kind": "latency"})
        if identity["horizon"] != 96:
            continue
        if identity["backbone"] == "iTransformer":
            families["raft"].append({**common, "id": "raft_"+tag, "kind": "raft"})
        for kind in ("controls", "shift"):
            families[kind].append({**common, "id": kind+"_"+tag, "kind": kind})
        variants = {"fixed_initial_gates": {}, "learned_constant_gate": {}}
        # Existing ETTh1/iTransformer controls are retained rather than rerun.
        if not (identity["dataset"] == "ETTh1" and identity["backbone"] == "iTransformer"):
            variants.update({
                "submitted_relu": {"risk_monitor_mode": "manuscript_mlp", "use_risk_gate": True, "risk_loss_weight": 1.0},
                "softplus_risk": {"risk_monitor_mode": "softplus_mlp", "use_risk_gate": True, "risk_loss_weight": 1.0},
                "no_dfm": {"use_dfm": False}, "no_hcb": {"use_hcb": False},
                "raw_cosine": {"retrieval_encoder": "raw_cosine"},
                "same_sensor": {"retrieval_scope": "same_sensor"},
                "cross_sensor": {"retrieval_scope": "cross_sensor"},
            })
        for variant, overrides in variants.items():
            families["train"].append({**common, "id": variant+"_"+tag, "kind": "train",
                                      "variant": variant, "overrides": overrides})
    # Short diagnostic families run first, followed by full paired retraining.
    jobs += [families[k].pop(0) for k in families]
    for key in ("latency", "controls", "shift", "raft", "train"):
        jobs += families[key]
    payload = {"source_sha256": SOURCE, "script_sha256": digest(__file__), "jobs": jobs,
               "topology_status": "held: paper graph input and reproduction provenance need author clarification",
               "scope": "all new results are reconstruction diagnostics; no manuscript renaming decision"}
    write_json(OUT / "manifest.json", payload)
    print(json.dumps({"jobs": len(jobs), "counts": {k: sum(j["kind"] == k for j in jobs) for k in ("sensitivity", "latency", "controls", "shift", "train", "raft")}}))


def worker(index):
    initialize()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if digest(__file__) != manifest["script_sha256"]:
        raise RuntimeError("Follow-up worker changed after registration")
    job = manifest["jobs"][index]
    output = OUT / "jobs" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    if job["kind"] in ("latency", "controls"):
        clean_diagnostic(job, output)
    elif job["kind"] == "shift":
        shift_diagnostic(job, output)
    elif job["kind"] == "train":
        train_control(job, output)
    elif job["kind"] == "raft":
        raft_control(job, output)
    else:
        sensitivity_summary(output)
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["diagnostic_script_sha256"] = digest(__file__)
    result["job_id"] = job["id"]
    write_json(result_path, result)


def completed(job, script_hash):
    path = OUT / "jobs" / job["id"] / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result.get("diagnostic_script_sha256") == script_hash and result.get("job_id") == job["id"]
    except (OSError, ValueError):
        return False


def controller():
    import psutil
    from run_remote_unattended import ControllerLock
    lock = ControllerLock(OUT / "controller.lock")
    if not lock.acquire():
        return
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if digest(__file__) != manifest["script_sha256"]:
        raise RuntimeError("Script changed after manifest was frozen")
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"attempts": {}, "failures": {}}
    old_child = state.get("child_pid")
    if old_child and psutil.pid_exists(old_child):
        proc = psutil.Process(old_child)
        if "reviewer_followup.py" in " ".join(proc.cmdline()):
            return  # Never launch a second worker after a controller-only restart.
    python = str(ROOT / ".venv/Scripts/python.exe")
    jobs = manifest["jobs"]
    for index, job in enumerate(jobs):
        directory = OUT / "jobs" / job["id"]
        if completed(job, manifest["script_sha256"]) or state["attempts"].get(job["id"], 0) >= 2:
            continue
        for _ in range(2-state["attempts"].get(job["id"], 0)):
            directory.mkdir(parents=True, exist_ok=True)
            attempt = state["attempts"].get(job["id"], 0)+1
            state["attempts"][job["id"]] = attempt
            log = directory / f"attempt{attempt}.log"
            with log.open("w", encoding="utf-8") as handle:
                child = subprocess.Popen([python, "-u", str(Path(__file__)), "worker", "--index", str(index)],
                    cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state.update(phase="running", current=job["id"], child_pid=child.pid,
                             controller_pid=os.getpid(), total=len(jobs), updated=time.time())
                write_json(state_path, state)
                try:
                    code = child.wait(timeout=12*3600)
                except subprocess.TimeoutExpired:
                    parent = psutil.Process(child.pid)
                    for descendant in parent.children(recursive=True):
                        descendant.kill()
                    child.kill()
                    child.wait()
                    code = -1
            state["child_pid"] = None
            success = code == 0 and completed(job, manifest["script_sha256"])
            state["completed"] = sum(completed(j, manifest["script_sha256"]) for j in jobs)
            if success:
                state["failures"].pop(job["id"], None)
            else:
                state["failures"][job["id"]] = {"returncode": code, "log": str(log)}
            state["updated"] = time.time()
            write_json(state_path, state)
            if success:
                break
    state.update(phase="complete" if not state["failures"] else "finished_with_failures", current=None, updated=time.time())
    write_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "controller"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        worker(args.index)
    else:
        controller()
