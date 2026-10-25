"""Bounded reviewer additions. Old queues, source and checkpoints are read-only.

TAFAS uses its pinned GCM class in a delayed-full-label, immutable-forecast
evaluation harness. This is explicitly NOT the unchanged upstream TAFAS protocol.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import reviewer_followup as rf

ROOT = rf.ROOT
OUT = ROOT / "results/reviewer_priority_20260911"
OFFICIAL = ROOT / "external/reviewer_priority/tafas.py"
POLICY = "float64 query encoder and cosine dot products; stable bank-index tie order; original stored FP32 keys/futures; no retraining"
CASES = [("clean", 0.0)] + [(k, s) for k in ("mean", "scale", "ramp") for s in (0.5, 1.0)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_jobs():
    old = read(rf.OUT / "manifest.json")
    families = {k: [] for k in ("shift", "risk", "tta", "weight")}
    for parent in old["jobs"]:
        if parent["kind"] != "shift":
            continue
        ident = parent["identity"]
        tag = parent["id"][len("shift_"):]
        common = {"identity": ident, "source_run": parent["source_run"]}
        if ident["dataset"] == "ETTm1":
            families["shift"].append({**common, "id": "stable_" + tag, "kind": "shift", "reviewers": ["R2.6", "R2.8"]})
        if ident["backbone"] == "iTransformer":
            families["tta"].append({**common, "id": "tafas_delayed_" + tag, "kind": "tta", "reviewers": ["R1.8", "R2.2"]})
        if ident["dataset"] == "ETTh1" and ident["backbone"] == "iTransformer":
            riskpath = ROOT / f"results/runs/public_ablation_strict/ETTh1/iTransformer/h96/submitted_relu_head/s{ident['seed']}/run.json"
            for weight in (0.1, 10.0):
                families["weight"].append({"identity": ident, "source_run": str(riskpath), "id": f"weight_{weight}_" + tag,
                    "kind": "weight", "variant": "submitted_relu_weight", "overrides": {"risk_loss_weight": weight}, "reviewers": ["R2.9"]})
        else:
            result = read(rf.OUT / "jobs" / ("submitted_relu_" + tag) / "result.json")
            riskpath = Path(result["run_path"])
        if ident["backbone"] == "iTransformer":
            families["tta"][-1]["risk_run"] = str(riskpath)
        families["risk"].append({"identity": ident, "source_run": str(riskpath), "id": "risk_shift_" + tag, "kind": "risk", "reviewers": ["R2.3"]})
    # One example of each inference family before long training. Then finish
    # existing-checkpoint evidence before the ten additional enhancement fits.
    jobs = [families[k].pop(0) for k in ("shift", "risk", "tta")]
    for family in families.values():
        jobs.extend(family)
    assert len(jobs) == 50 and len({j["id"] for j in jobs}) == 50
    return jobs


def prepare():
    rf.initialize()
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Manifest already frozen; use controller/status, never replace it")
    jobs = manifest_jobs()
    for job in jobs:
        path = ROOT / job["source_run"]
        run = read(path)
        assert run["environment"]["source_sha256"] == rf.SOURCE
        assert run["config"]["seed"] == job["identity"]["seed"]
        assert run["config"]["data"]["pred_len"] == 96
        job["inputs"] = {str(p): rf.digest(p) for p in (path, path.with_name("checkpoint.pt"), path.with_name("predictions.npz"))}
        if job["kind"] == "tta":
            reference = ROOT / job["risk_run"]
            reference_run = read(reference)
            assert reference_run["environment"]["backbone_cache"]["key"] == run["environment"]["backbone_cache"]["key"]
            for p in (reference, reference.with_name("predictions.npz")):
                job["inputs"][str(p)] = rf.digest(p)
        if job["kind"] in ("risk", "weight"):
            assert run["config"]["ugma"]["risk_monitor_mode"] == "manuscript_mlp"
            assert run["config"]["ugma"]["use_risk_gate"]
        if job["kind"] == "weight":
            assert run["config"]["ugma"]["risk_loss_weight"] == 1.0
            cache = ROOT / run["environment"]["backbone_cache"]["path"]
            job["cache_path"] = str(cache)
            job["cache_sha256"] = rf.digest(cache)
    payload = {"created_unix": time.time(), "script_sha256": rf.digest(__file__), "source_sha256": rf.SOURCE,
        "dependency_hashes": {str(p): rf.digest(p) for p in (Path(rf.__file__), ROOT / "scripts/run_remote_unattended.py", OFFICIAL)},
        "old_manifest_sha256": rf.digest(rf.OUT / "manifest.json"), "jobs": jobs,
        "protocol": {"shift": POLICY, "risk": "original ReLU SMM; validation-fixed calibration bins; no model updates; full test, seven cases and simple proxies",
            "tta": "pinned TAFAS GCM; frozen paired backbone; Adam lr=0.005 wd=0.0001; batch32; one MSE update per matured batch; labels only after whole horizon; forecasts never revised; no partial-label PAAS; no test hyperparameter tuning",
            "weight": "ETTh1/iTransformer/Q96, five original seeds; lambda 0.1 and 10; reuse existing lambda1 results and backbone cache; all other settings unchanged"},
        "not_claimed": ["official unchanged TAFAS end-to-end reproduction", "full method evidence from monitor-disabled R0", "real industrial incident testing", "all reviewer questions resolved"]}
    rf.write_json(OUT / "manifest.json", payload)
    print(json.dumps({"total": len(jobs), "counts": {k: sum(j['kind'] == k for j in jobs) for k in ('shift', 'risk', 'tta', 'weight')}, "manifest_sha256": rf.digest(OUT / 'manifest.json')}), flush=True)


def stable_retrieval(model):
    import torch
    from ugma.models.ugma import RetrievalOutput
    assert model.bank.retrieval_scope == "all"
    encoder = copy.deepcopy(model.encoder).double()
    forward = encoder.forward
    encoder.forward = lambda history: forward(history.double())
    model.encoder = encoder
    bank = model.bank
    keys = bank.keys.double()

    def retrieve(queries, starts=None):
        b, c, d = queries.shape
        scores = queries.reshape(-1, d) @ keys.T
        if starts is not None:
            near = (starts.repeat_interleave(c)[:, None] - bank.target_starts[None, :]).abs() < bank.temporal_gap
            scores.masked_fill_(near, float("-inf"))
        indices = torch.argsort(scores, descending=True, stable=True, dim=-1)[:, :bank.top_k]
        vals = scores.gather(1, indices).float()
        valid = torch.isfinite(vals)
        safe = torch.where(valid, vals, torch.zeros_like(vals))
        safe = safe - safe.amax(-1, keepdim=True)
        weights = torch.exp(safe / bank.temperature) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        ref = torch.einsum("bk,bkq->bq", weights, bank.futures[indices]).reshape(b, c, -1).transpose(1, 2)
        return RetrievalOutput(ref, vals.reshape(b, c, -1), bank.nodes[indices].reshape(b, c, -1),
            bank.target_starts[indices].reshape(b, c, -1), valid.reshape(b, c, -1))
    bank.retrieve = retrieve


def shifted_values(values, train_end, onset, kind, level):
    import numpy as np
    result = values.copy()
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0).clip(1e-6)
    direction = np.where(np.arange(values.shape[1]) % 2 == 0, 1.0, -1.0)
    if kind == "mean":
        result[onset:] += level * std * direction
    elif kind == "scale":
        result[onset:] = mean + (1 + level) * (values[onset:] - mean)
    elif kind == "ramp":
        result[onset:] += np.linspace(0, level, len(values) - onset)[:, None] * std * direction
    elif kind != "clean":
        raise ValueError(kind)
    assert np.array_equal(result[:onset], values[:onset])
    return result


def case_dataset(cfg, bundle, scaler, bounds, kind, level):
    import numpy as np
    from ugma.data import WindowDataset
    values = shifted_values(bundle.values, bounds.train_end, bounds.val_end, kind, level)
    return WindowDataset(scaler.transform(values), cfg.data.seq_len, cfg.data.pred_len,
        bounds.val_end, len(values), np.arange(len(values)) >= bounds.val_end)


def compare(a, b):
    import numpy as np
    diff = np.abs(a - b)
    bad = diff > 2e-4 + 2e-4 * np.abs(b)
    return {"allclose": bool(not bad.any()), "max_abs": float(diff.max()), "bad_elements": int(bad.sum())}


def shift_job(job, output, limit=None):
    import numpy as np
    import torch
    model, cfg, bundle, datasets, scaler, bounds, _ = rf.load_model(job["source_run"])
    assert cfg.ugma.risk_monitor_mode == "disabled"
    stable_retrieval(model)
    with np.load((ROOT / job["source_run"]).with_name("predictions.npz")) as archive:
        saved = {k: archive[k] for k in ("target", "backbone", "enhanced")}
    cases, checks = [], {}
    for kind, level in CASES:
        dataset = case_dataset(cfg, bundle, scaler, bounds, kind, level)
        count = min(len(dataset), limit) if limit else len(dataset)
        chunks = {k: [] for k in ("backbone", "full", "local_only", "retrieval_only")}
        clean_preds = []
        with torch.inference_mode():
            for begin in range(0, count, 32):
                ids = np.arange(begin, min(begin + 32, count))
                x, y, starts = rf.batch_items(dataset, ids)
                enhanced, aux = model(x, starts)
                preds = {"backbone": aux["backbone"], "full": enhanced,
                    "local_only": aux["backbone"] + aux["alpha"][:, None, :] * aux["local"] * aux["normalization_scale"],
                    "retrieval_only": aux["backbone"] + aux["beta"][:, None, :] * (aux["reference"] - aux["normalized_backbone"]) * aux["normalization_scale"]}
                if kind == "clean":
                    assert np.array_equal(y.cpu().numpy(), saved["target"][ids])
                    assert compare(aux["backbone"].cpu().numpy(), saved["backbone"][ids])["allclose"]
                    clean_preds.append(enhanced.cpu().numpy())
                for name, pred in preds.items():
                    chunks[name].append(rf.metric_errors(y, pred))
        errors = {k: (np.concatenate([v[0] for v in parts]), np.concatenate([v[1] for v in parts])) for k, parts in chunks.items()}
        np.savez_compressed(output / f"{kind}_{level}.npz", **{k + "_mae": v[0] for k, v in errors.items()}, **{k + "_mse": v[1] for k, v in errors.items()})
        cases.append({"kind": kind, "severity": level, "windows": count, "metrics": rf.summarize(errors)})
        if kind == "clean":
            p32 = np.concatenate(clean_preds)
            p16 = []
            with torch.inference_mode():
                for begin in range(0, count, 16):
                    x, _, starts = rf.batch_items(dataset, np.arange(begin, min(begin + 16, count)))
                    p16.append(model(x, starts)[0].cpu().numpy())
            checks = {"batch16_vs_32": compare(np.concatenate(p16), p32), "new_vs_archive": compare(p32, saved["enhanced"][:count])}
            assert checks["batch16_vs_32"]["allclose"], checks
            # Good/middle/poor are descriptive retrospective groups, not routing.
            order = np.argsort(errors["backbone"][0], axis=0, kind="stable")
            groups = {}
            for label, indices in zip(("good", "middle", "poor"), np.array_split(order, 3, axis=0)):
                groups[label] = {k: float(np.take_along_axis(v[0], indices, axis=0).mean()) for k, v in errors.items()}
            checks["retrospective_error_groups"] = groups
        print(job["id"], kind, level, cases[-1]["metrics"], flush=True)
    return {"cases": cases, "replay_checks": checks, "precision_policy": POLICY, "model_variant": "monitor-disabled R0", "smoke_only": bool(limit)}


def score_arrays(model, dataset, limit=None):
    import numpy as np
    import torch
    chunks = {k: [] for k in ("risk", "realized", "history_variation", "level_distance", "forecast_roughness")}
    count = min(len(dataset), limit) if limit else len(dataset)
    with torch.inference_mode():
        for begin in range(0, count, 32):
            x, y, _ = rf.batch_items(dataset, np.arange(begin, min(begin + 32, count)))
            base = model.backbone(x)
            nx, nb, mean, scale = model.normalize_context(x, base)
            raw = model.risk_monitor(nx, nb)
            arrays = {"risk": (model.risk_calibration_slope * raw + model.risk_calibration_intercept).clamp_min(0),
                "realized": ((y - mean) / scale - nb).abs().mean(1),
                "history_variation": x.diff(dim=1).abs().mean(1), "level_distance": x.mean(1).abs(),
                "forecast_roughness": nb.diff(dim=1).abs().mean(1)}
            for key, value in arrays.items():
                chunks[key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def risk_metrics(score, realized, validation_score):
    import numpy as np
    from scipy.stats import spearmanr
    score, realized, vs = score.ravel().astype(float), realized.ravel().astype(float), validation_score.ravel().astype(float)
    varying = score.std() > 1e-12 and realized.std() > 1e-12
    edges = np.unique(np.quantile(vs, np.linspace(0, 1, 11)))[1:-1]
    ids = np.searchsorted(edges, score, side="right")
    bins = [{"bin": i, "count": int((ids == i).sum()), "predicted": float(score[ids == i].mean()), "observed": float(realized[ids == i].mean())}
            for i in range(len(edges) + 1) if (ids == i).any()]
    # Integrate tied scores fractionally; chronological tie order must not create
    # apparently informative risk-coverage curves for an all-zero ReLU head.
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=realized)
    coverage = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        take = np.clip(fraction * len(score) - np.r_[0, np.cumsum(counts)[:-1]], 0, counts)
        coverage.append({"coverage": fraction, "selective_mae": float((take * totals / counts).sum() / take.sum())})
    return {"pearson": float(np.corrcoef(score, realized)[0, 1]) if varying else None,
        "spearman": float(spearmanr(score, realized).statistic) if varying else None,
        "constant_score": bool(score.std() <= 1e-12), "zero_fraction": float((score == 0).mean()),
        "validation_bin_edges": edges.tolist(), "bins": bins, "coverage": coverage}


def risk_job(job, output, limit=None):
    import numpy as np
    model, cfg, bundle, datasets, scaler, bounds, _ = rf.load_model(job["source_run"])
    validation = score_arrays(model, datasets["val"], limit)
    np.savez_compressed(output / "validation_scores.npz", **validation)
    cases = []
    for kind, level in CASES:
        dataset = case_dataset(cfg, bundle, scaler, bounds, kind, level)
        arrays = score_arrays(model, dataset, limit)
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(output / f"{kind}_{level}_scores.npz", **arrays)
        metrics = {k: risk_metrics(v, arrays["realized"], validation[k]) for k, v in arrays.items() if k != "realized"}
        cases.append({"kind": kind, "severity": level, "windows": len(arrays["risk"]), "mean_realized_risk": float(arrays["realized"].mean()), "scores": metrics})
        print(job["id"], kind, level, "risk rho", metrics["risk"]["spearman"], flush=True)
    return {"cases": cases, "risk_units": "history-instance-normalized channel-wise backbone MAE", "proxy_units": "input proxies use train-standardized values; roughness uses instance-normalized forecasts; raw proxy calibration is descriptive, not a calibrated error predictor", "smoke_only": bool(limit)}


def matured_end(starts, prediction_start, horizon):
    import numpy as np
    return int(np.searchsorted(starts + horizon, prediction_start, side="right"))


def official_gcm():
    import torch
    module = ast.parse(OFFICIAL.read_text(encoding="utf-8"))
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GCM"]
    assert len(classes) == 1
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(OFFICIAL), "exec"), namespace)
    return namespace["GCM"]


def tta_job(job, output, limit=None):
    import numpy as np
    import torch
    model, cfg, _, datasets, _, _, _ = rf.load_model(job["source_run"])
    backbone = model.backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    dataset = datasets["test"]
    channels = dataset[0]["x"].shape[-1]
    gcm = official_gcm()
    incali, outcali = gcm(cfg.data.seq_len, channels).cuda(), gcm(cfg.data.pred_len, channels).cuda()
    optimizer = torch.optim.Adam(list(incali.parameters()) + list(outcali.parameters()), lr=0.005, weight_decay=0.0001)
    count = min(len(dataset), limit) if limit else len(dataset)
    forecasts, targets, ledger = [], [], []
    used = 0
    with np.load((ROOT / job["source_run"]).with_name("predictions.npz")) as archive:
        saved = {k: archive[k] for k in ("target", "backbone", "enhanced")}
    for begin in range(0, count, 32):
        origin = int(dataset.target_starts[begin])
        end = matured_end(dataset.target_starts, origin, cfg.data.pred_len)
        for train_begin in range(used, end, 32):
            indices = np.arange(train_begin, min(train_begin + 32, end))
            assert int(dataset.target_starts[indices].max()) + cfg.data.pred_len <= origin
            x, y, _ = rf.batch_items(dataset, indices)
            optimizer.zero_grad(set_to_none=True)
            # Official iTransformer normalizes in-place. Autograd must retain
            # the pre-mutation values for input-calibrator gradients; torch's
            # clone-on-mutation context preserves the exact forward arithmetic.
            with torch.autograd.graph.allow_mutation_on_saved_tensors():
                pred = outcali(backbone(incali(x)))
                loss = torch.nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite TAFAS adaptation loss")
                loss.backward()
            optimizer.step()
            ledger.append({"forecast_origin": origin, "train_first_index": int(indices[0]), "train_last_index": int(indices[-1]), "last_label_exclusive": int(dataset.target_starts[indices[-1]]) + cfg.data.pred_len, "loss": float(loss.detach())})
        used = end
        ids = np.arange(begin, min(begin + 32, count))
        x, y, _ = rf.batch_items(dataset, ids)
        with torch.no_grad():
            base = backbone(x)
            assert compare(base.cpu().numpy(), saved["backbone"][ids])["allclose"]
            pred = outcali(backbone(incali(x)))
        assert np.array_equal(y.cpu().numpy(), saved["target"][ids])
        assert torch.isfinite(pred).all()
        forecasts.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
        if begin % 1024 == 0:
            print(job["id"], begin, count, "updates", len(ledger), flush=True)
    prediction, target = np.concatenate(forecasts), np.concatenate(targets)
    assert np.allclose(prediction[:min(32, count)], saved["backbone"][:min(32, count)], rtol=2e-4, atol=2e-4)
    with np.load((ROOT / job["risk_run"]).with_name("predictions.npz")) as reference:
        assert np.array_equal(reference["target"][:count], target)
        assert compare(reference["backbone"][:count], saved["backbone"][:count])["allclose"]
        full_relu = reference["enhanced"][:count]
    errors = {k: (np.abs(v.astype(float) - target).mean(1), ((v.astype(float) - target)**2).mean(1)) for k, v in
        {"tafas_delayed_full_labels": prediction, "frozen_backbone": saved["backbone"][:count], "R0_offline": saved["enhanced"][:count], "SMM_ReLU_offline": full_relu}.items()}
    np.savez_compressed(output / "predictions.npz", prediction=prediction, target=target,
        **{k + "_mae": v[0] for k, v in errors.items()}, **{k + "_mse": v[1] for k, v in errors.items()})
    rf.write_json(output / "label_access_ledger.json", ledger)
    torch.save({"input_calibrator": incali.state_dict(), "output_calibrator": outcali.state_dict()}, output / "calibrators.pt")
    return {"method": "TAFAS GCM, delayed-full-label matched harness", "metrics": rf.summarize(errors), "windows": count,
        "updates": len(ledger), "official_file_sha256": rf.digest(OFFICIAL), "source_repository": "BorealisAI/PETSA", "source_commit": "87853d888e98311ac94e64be920d17b57143b20c",
        "protocol_difference": "fixed batches32, only fully matured labels, no partial-label adaptation, no PAAS, no revision of already issued forecasts; shared original split/scaler/checkpoint; upstream GCM and MSE retained; torch clone-on-mutation autograd context for original in-place backbone normalization",
        "information_difference": "TAFAS receives delayed deployment labels; R0 does not update at deployment. This difference must be reported with comparison.", "smoke_only": bool(limit)}


def weight_job(job, output):
    assert rf.digest(job["cache_path"]) == job["cache_sha256"]
    # A failed fit never leaves an apparent accepted top-level result.
    fit_output = output / ("fit_" + str(time.time_ns()))
    fit_output.mkdir()
    rf.train_control(job, fit_output)
    result = read(fit_output / "result.json")
    run = read(result["run_path"])
    assert run["environment"]["backbone_cache"]["hit"], "Unexpected backbone retraining"
    original = read(ROOT / job["source_run"])
    assert run["environment"]["backbone_cache"]["key"] == original["environment"]["backbone_cache"]["key"]
    result["lambda1_reference"] = {"path": job["source_run"], "metrics": original["results"]}
    result["backbone_cache"] = run["environment"]["backbone_cache"]
    return result


def verified_manifest():
    manifest = read(OUT / "manifest.json")
    assert rf.digest(__file__) == manifest["script_sha256"], "Runner changed after registration"
    for path, digest in manifest["dependency_hashes"].items():
        assert rf.digest(path) == digest, path
    assert rf.digest(rf.OUT / "manifest.json") == manifest["old_manifest_sha256"]
    return manifest


def worker(index, smoke=False):
    rf.initialize()
    manifest = verified_manifest()
    job = manifest["jobs"][index]
    for path, digest in job["inputs"].items():
        assert rf.digest(path) == digest, path
    output = OUT / ("smoke_" + manifest["script_sha256"][:12] if smoke else "jobs") / job["id"]
    if (output / "result.json").exists():
        raise RuntimeError("Existing result preserved; controller must validate/skip it")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if job["kind"] == "weight":
        assert not smoke
        result = weight_job(job, output)
    else:
        result = {"shift": shift_job, "risk": risk_job, "tta": tta_job}[job["kind"]](job, output, 256 if smoke else None)
    result.update(job_id=job["id"], identity=job["identity"], reviewers=job["reviewers"], inputs=job["inputs"],
        diagnostic_script_sha256=manifest["script_sha256"], source_sha256=rf.SOURCE, elapsed_seconds=time.time() - started)
    rf.write_json(output / "result.json", result)


def completed(job, script_hash):
    try:
        result = read(OUT / "jobs" / job["id"] / "result.json")
        return result.get("diagnostic_script_sha256") == script_hash and result.get("job_id") == job["id"] and not result.get("smoke_only", False)
    except (OSError, ValueError):
        return False


def controller():
    import psutil
    from run_remote_unattended import ControllerLock
    lock = ControllerLock(OUT / "controller.lock")
    if not lock.acquire():
        return
    try:
        manifest = verified_manifest()
        state_path = OUT / "state.json"
        state = read(state_path) if state_path.exists() else {"attempts": {}, "failures": {}}
        pid = state.get("child_pid")
        if pid and psutil.pid_exists(pid):
            try:
                if Path(__file__).name in " ".join(psutil.Process(pid).cmdline()):
                    return
            except psutil.NoSuchProcess:
                pass
        jobs, script_hash = manifest["jobs"], manifest["script_sha256"]
        for index, job in enumerate(jobs):
            if completed(job, script_hash):
                continue
            for attempt in range(state["attempts"].get(job["id"], 0) + 1, 3):
                directory = OUT / "jobs" / job["id"]
                directory.mkdir(parents=True, exist_ok=True)
                logfile = directory / f"attempt{attempt}.log"
                with logfile.open("a", encoding="utf-8") as handle:
                    child = subprocess.Popen([str(ROOT / ".venv/Scripts/python.exe"), "-u", str(Path(__file__)), "worker", "--index", str(index)],
                        cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    state["attempts"][job["id"]] = attempt
                    state.update(phase="running", current=job["id"], child_pid=child.pid, controller_pid=os.getpid(), total=len(jobs),
                        completed=sum(completed(j, script_hash) for j in jobs), updated=time.time())
                    rf.write_json(state_path, state)
                    try:
                        code = child.wait(timeout=6 * 3600)
                    except subprocess.TimeoutExpired:
                        parent = psutil.Process(child.pid)
                        for descendant in parent.children(recursive=True):
                            descendant.kill()
                        child.kill()
                        child.wait()
                        code = -1
                success = code == 0 and completed(job, script_hash)
                state.update(child_pid=None, updated=time.time(), completed=sum(completed(j, script_hash) for j in jobs))
                if success:
                    state["failures"].pop(job["id"], None)
                else:
                    state["failures"][job["id"]] = {"returncode": code, "log": str(logfile)}
                rf.write_json(state_path, state)
                if success:
                    break
        state.update(phase="complete" if all(completed(j, script_hash) for j in jobs) else "finished_with_failures", current=None, child_pid=None, updated=time.time())
        rf.write_json(state_path, state)
    finally:
        lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "controller", "worker", "smoke", "status"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "controller":
        controller()
    elif args.command == "status":
        manifest = verified_manifest()
        print(json.dumps({"state": read(OUT / "state.json") if (OUT / "state.json").exists() else None,
            "accepted": sum(completed(j, manifest['script_sha256']) for j in manifest['jobs']), "total": len(manifest['jobs'])}))
    else:
        worker(args.index, args.command == "smoke")
