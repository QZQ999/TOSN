"""Independent recovery queue: fixed physical batch32 for reproducible inference.

The priority queue and its failures remain immutable. Padding repeats the last
item and is discarded; these frozen models have no across-batch operations.
Logical partitions16/32 both execute physical batch32. This is not evidence for
arbitrary physical batch-size invariance. No tolerance or model weight changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import reviewer_priority_20260911 as p
import reviewer_followup as rf

ROOT = rf.ROOT
OUT = ROOT / "results/reviewer_shift_fixed_batch_20260911"
POLICY = "Fixed physical CUDA batch32, repeat-last padding discarded; logical partition16 versus32 checked; float64 query encoder/dot product and stable retrieval ties; original weights. Not arbitrary physical-batch invariance."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_physical_forward(model):
    import torch
    forward = model.forward

    def fixed(history, starts=None):
        count = len(history)
        assert 0 < count <= 32
        if count < 32:
            history = torch.cat([history, history[-1:].expand(32 - count, -1, -1)], dim=0)
            if starts is not None:
                starts = torch.cat([starts, starts[-1:].expand(32 - count)])
        pred, aux = forward(history, starts)
        return pred[:count], {k: v[:count] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == 32 else v for k, v in aux.items()}
    model.forward = fixed
    return model


def prepare():
    rf.initialize()
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Recovery manifest exists; do not overwrite")
    original = p.verified_manifest()
    jobs = [dict(j) for j in original["jobs"] if j["kind"] == "shift"]
    assert len(jobs) == 10
    for job in jobs:
        job["parent_job_id"] = job["id"]
        job["id"] = "fixed_batch_" + job["id"]
    dependencies = dict(original["dependency_hashes"])
    dependencies[str(Path(p.__file__))] = rf.digest(p.__file__)
    rf.write_json(OUT / "manifest.json", {"script_sha256": rf.digest(__file__), "source_sha256": rf.SOURCE,
        "dependency_hashes": dependencies, "old_manifest_sha256": rf.digest(rf.OUT / "manifest.json"), "jobs": jobs,
        "parent_manifest_sha256": rf.digest(p.OUT / "manifest.json"), "policy": POLICY, "created_unix": time.time()})
    print(json.dumps({"jobs": 10, "manifest_sha256": rf.digest(OUT / "manifest.json")}))


def verified():
    manifest = read(OUT / "manifest.json")
    assert manifest["script_sha256"] == rf.digest(__file__)
    assert manifest["parent_manifest_sha256"] == rf.digest(p.OUT / "manifest.json")
    for path, digest in manifest["dependency_hashes"].items():
        assert rf.digest(path) == digest, path
    return manifest


def worker(index, audit=False):
    rf.initialize()
    manifest = verified()
    job = manifest["jobs"][index]
    for path, digest in job["inputs"].items():
        assert rf.digest(path) == digest, path
    output = OUT / ("full_preflight" if audit else "jobs") / job["id"]
    if (output / "result.json").exists():
        raise RuntimeError("Existing accepted result preserved")
    output.mkdir(parents=True, exist_ok=True)
    original_load = rf.load_model

    def load(path):
        model, *rest = original_load(path)
        return (fixed_physical_forward(model), *rest)
    rf.load_model = load
    started = time.time()
    result = p.shift_job(job, output)
    result["replay_checks"]["logical_partition16_vs_32_at_fixed_physical32"] = result["replay_checks"].pop("batch16_vs_32")
    result.update(job_id=job["id"], identity=job["identity"], inputs=job["inputs"], reviewers=job["reviewers"],
        parent_job_id=job["parent_job_id"], diagnostic_script_sha256=manifest["script_sha256"], source_sha256=rf.SOURCE,
        precision_policy=POLICY, full_preflight_only=audit, elapsed_seconds=time.time() - started)
    rf.write_json(output / "result.json", result)


def controller():
    # Wait through recurring scheduler invocations; never contend with primary
    # reviewer training or reset its failure/attempt counters.
    parent_state = read(p.OUT / "state.json")
    if parent_state["phase"] not in ("complete", "finished_with_failures") or parent_state.get("child_pid"):
        return
    manifest = verified()
    # Reuse the tested lock/resume machinery, but point all runtime paths and
    # script-identity checks at this independent runner in this process only.
    p.OUT = OUT
    p.__file__ = __file__
    p.controller()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "full-preflight", "controller"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "controller":
        controller()
    else:
        worker(args.index, args.command == "full-preflight")
