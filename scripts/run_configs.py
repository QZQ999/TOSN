"""Resume-safe sequential launcher; completed run.json files are never overwritten."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-retries", type=int, default=1)
    args = parser.parse_args()
    config_root = args.config_root if args.config_root.is_absolute() else ROOT / args.config_root
    configs = sorted(config_root.rglob("*.yaml"))
    if args.limit is not None:
        configs = configs[: args.limit]
    ledger = ROOT / "results" / "experiment_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        result = ROOT / raw["output_dir"] / "run.json"
        if result.exists():
            print(f"SKIP completed {config.relative_to(ROOT)}", flush=True)
            continue
        started = time.time()
        print(f"START {config.relative_to(ROOT)}", flush=True)
        attempt = 0
        while True:
            attempt += 1
            completed = subprocess.run(
                [sys.executable, "-m", "ugma.cli", "run", "--config", str(config), "--quiet"],
                cwd=ROOT,
            )
            event = {
                "config": str(config.relative_to(ROOT)).replace("\\", "/"),
                "result": str(result.relative_to(ROOT)).replace("\\", "/"),
                "started_unix": started,
                "elapsed_seconds": time.time() - started,
                "attempt": attempt,
                "returncode": completed.returncode,
            }
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            native_crash = completed.returncode < 0 or completed.returncode >= 0x80000000
            if completed.returncode == 0 or not native_crash or attempt > args.native_retries:
                break
            print(f"RETRY native exit {completed.returncode} for {config.relative_to(ROOT)}", flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
