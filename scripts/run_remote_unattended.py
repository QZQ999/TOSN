"""Autonomous, resume-safe controller for the U-GMA R1 evidence matrix.

The controller is deliberately outside ``src/ugma`` so deploying it does not
change the experiment source digest recorded by ``ugma.cli``.  It is designed
for a single GPU host.  A priority file can restrict execution to the configs
needed for reviewer-evidence closure.  Failed jobs receive a bounded number of
attempts so one pathological configuration cannot starve the remaining queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised on Linux deployment hosts.
    import fcntl


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "636807134ff574f308266a22cef002d41d90a1960a4522075ffbd941d8d96a45"


@dataclass(frozen=True)
class Suite:
    name: str
    config_root: str
    result_root: str
    aggregate_root: str
    expected: int
    heavy: bool


SUITES = (
    Suite(
        "public_strict",
        "configs/generated/public_strict",
        "results/runs/public_strict",
        "results/aggregate/public_strict",
        200,
        True,
    ),
    Suite(
        "public_ablation_strict",
        "configs/generated/public_ablation_strict",
        "results/runs/public_ablation_strict",
        "results/aggregate/public_ablation_strict",
        25,
        True,
    ),
    Suite(
        "sensitivity",
        "configs/generated/sensitivity",
        "results/runs/sensitivity",
        "results/aggregate/sensitivity",
        125,
        True,
    ),
    Suite(
        "multiplex",
        "configs/generated/multiplex",
        "results/runs/multiplex",
        "results/aggregate/multiplex",
        225,
        False,
    ),
    Suite(
        "retrieval_ablation",
        "configs/generated/retrieval_ablation",
        "results/runs/retrieval_ablation_v3",
        "results/aggregate/retrieval_ablation",
        40,
        True,
    ),
)


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src" / "ugma").rglob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "external_versions.json"]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def read_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "output_dir" not in raw:
        raise ValueError(f"invalid config: {path}")
    return raw


def result_paths(config: Path) -> tuple[Path, Path]:
    raw = read_config(config)
    output = ROOT / raw["output_dir"]
    return output / "run.json", output / "predictions.npz"


def complete_result(config: Path) -> bool:
    run_path, predictions_path = result_paths(config)
    if not run_path.is_file() or not predictions_path.is_file():
        return False
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        results = payload["results"]
        environment = payload["environment"]
        _ = results["backbone"]["mae"]
        _ = results["enhanced"]["mae"]
        return environment.get("source_sha256") == EXPECTED_SOURCE_SHA256
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


class ControllerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.owned = False
        self.handle: Any | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on Linux deployment hosts.
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            handle.close()
            return False
        handle.seek(0)
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.truncate()
        handle.flush()
        self.handle = handle
        self.owned = True
        return True

    def release(self) -> None:
        if self.owned and self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised on Linux deployment hosts.
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
            self.handle.close()
            self.handle = None
            self.owned = False


def terminate_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=20)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def available_gpu_memory_mb() -> int | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        return int(completed.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def wait_for_gpu_memory(minimum_mb: int, stop: threading.Event) -> None:
    while not stop.is_set():
        free = available_gpu_memory_mb()
        if free is None or free >= minimum_mb:
            return
        stop.wait(30)


def config_weight(path: Path) -> float:
    """Approximate runtime weight used for shortest-evidence-first scheduling."""
    raw = read_config(path)
    data = raw.get("data", {})
    backbone = raw.get("backbone", {})
    horizon = float(data.get("pred_len", 1))
    dataset_multiplier = 4.0 if "ETTm1" in str(data.get("path", "")) else 1.0
    model_multiplier = {
        "FEDformer": 5.0,
        "TimesNet": 4.0,
        "PatchTST": 2.5,
        "TimeMixer": 2.0,
        "iTransformer": 1.0,
        "dlinear": 0.05,
    }.get(str(backbone.get("name")), 1.0)
    epochs = float(backbone.get("epochs", 1)) + float(raw.get("ugma", {}).get("enhancement_epochs", 1))
    return dataset_multiplier * model_multiplier * horizon * max(epochs, 1.0)


def read_priority_file(path: Path) -> list[Path]:
    priority_path = path if path.is_absolute() else ROOT / path
    priority_path = priority_path.resolve()
    if not priority_path.is_file():
        raise FileNotFoundError(f"priority file not found: {priority_path}")
    configs: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(priority_path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw_line.split("#", 1)[0].strip()
        if not value:
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{priority_path}:{line_number}: config escapes repository root")
        has_glob = any(character in value for character in "*?[")
        matches = sorted(ROOT.glob(value)) if has_glob else [(ROOT / value)]
        matches = [match.resolve() for match in matches if match.is_file()]
        if not matches:
            raise FileNotFoundError(f"{priority_path}:{line_number}: config pattern matched nothing: {value}")
        for config in matches:
            if not config.is_relative_to(ROOT):
                raise ValueError(f"{priority_path}:{line_number}: config escapes repository root")
            if config.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"{priority_path}:{line_number}: not a YAML config: {config}")
            if config in seen:
                raise ValueError(f"{priority_path}:{line_number}: duplicate config {config.relative_to(ROOT)}")
            seen.add(config)
            configs.append(config)
    if not configs:
        raise ValueError(f"priority file is empty: {priority_path}")
    return configs


def state_directory(path: Path) -> Path:
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"state directory escapes repository root: {resolved}")
    return resolved


class AutonomousController:
    def __init__(self, args: argparse.Namespace, suites: list[Suite]) -> None:
        self.args = args
        self.suites = suites
        self.state_dir = state_directory(args.state_dir)
        self.state_path = self.state_dir / "state.json"
        self.ledger_path = self.state_dir / "ledger.jsonl"
        self.controller_log = self.state_dir / "controller.log"
        self.attempts: dict[str, int] = {}
        self.running: dict[str, dict[str, Any]] = {}
        self.completed: set[str] = set()
        self.failures: dict[str, str] = {}
        self.started_unix = time.time()
        self.lock = threading.RLock()
        self.ledger_lock = threading.Lock()
        self.cache_locks_guard = threading.Lock()
        self.cache_locks: dict[str, threading.Lock] = {}
        self.stop = threading.Event()
        self.total_configs = 0

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"{timestamp} {message}"
        with self.lock:
            self.controller_log.parent.mkdir(parents=True, exist_ok=True)
            with self.controller_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        print(line, flush=True)

    def config_key(self, config: Path) -> str:
        return config.relative_to(ROOT).as_posix()

    def write_state(self, phase: str, suite: str | None = None) -> None:
        with self.lock:
            payload = {
                "controller_pid": os.getpid(),
                "started_unix": self.started_unix,
                "updated_unix": time.time(),
                "phase": phase,
                "suite": suite,
                "source_sha256": source_digest(),
                "total_configs": self.total_configs,
                "completed_count": len(self.completed),
                "running": self.running,
                "attempts": self.attempts,
                "failures": self.failures,
            }
        atomic_json(self.state_path, payload)

    def heartbeat(self) -> None:
        while not self.stop.wait(60):
            try:
                self.write_state("running")
            except Exception:
                self.log("HEARTBEAT_ERROR " + traceback.format_exc().replace("\n", " | "))

    def preflight(self) -> dict[str, list[Path]]:
        digest = source_digest()
        if digest != self.args.expected_source_sha256:
            raise RuntimeError(f"source digest mismatch: expected {self.args.expected_source_sha256}, found {digest}")
        suite_configs: dict[str, list[Path]] = {}
        priority = read_priority_file(self.args.priority_file) if self.args.priority_file else None
        assigned_priority: set[Path] = set()
        for suite in self.suites:
            all_paths = sorted((ROOT / suite.config_root).rglob("*.yaml"))
            if len(all_paths) != suite.expected:
                raise RuntimeError(f"{suite.name}: expected {suite.expected} configs, found {len(all_paths)}")
            if priority is None:
                paths = sorted(all_paths, key=config_weight)
            else:
                available = set(all_paths)
                paths = [config for config in priority if config in available]
                assigned_priority.update(paths)
            for config in paths:
                raw = read_config(config)
                data = raw.get("data", {})
                for field in ("path", "reference_path"):
                    value = data.get(field)
                    if value and not (ROOT / value).is_file():
                        raise FileNotFoundError(f"{config}: missing data {field}={value}")
                if complete_result(config):
                    self.completed.add(self.config_key(config))
            suite_configs[suite.name] = paths
        if priority is not None:
            unassigned = [config for config in priority if config not in assigned_priority]
            if unassigned:
                values = ", ".join(str(path.relative_to(ROOT)) for path in unassigned)
                raise ValueError(f"priority configs are outside selected suites: {values}")
        self.total_configs = sum(len(paths) for paths in suite_configs.values())
        self.log(
            f"PREFLIGHT_OK source={digest} configs={self.total_configs} already_complete={len(self.completed)} "
            f"gpu_free_mb={available_gpu_memory_mb()}"
        )
        self.write_state("preflight_complete")
        return suite_configs

    def wait_for_external(self) -> None:
        if not self.args.wait_pid_file:
            return
        pid_path = ROOT / self.args.wait_pid_file
        if not pid_path.exists():
            self.log(f"WAIT_PID_FILE_MISSING {pid_path}")
            return
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            self.log(f"WAIT_PID_FILE_INVALID {pid_path}")
            return
        self.log(f"WAIT_EXTERNAL pid={pid}")
        while psutil.pid_exists(pid) and not self.stop.wait(30):
            self.write_state("waiting_for_external")
        self.log(f"WAIT_EXTERNAL_DONE pid={pid}")

    def log_path(self, suite: Suite, config: Path, attempt: int) -> Path:
        relative = config.relative_to(ROOT / suite.config_root).with_suffix("")
        return self.state_dir / "job_logs" / suite.name / relative.parent / f"{relative.name}.attempt{attempt}.log"

    def cache_lock_for(self, config: Path) -> threading.Lock:
        """Serialize jobs that can read or write the same backbone cache entry."""
        raw = read_config(config)
        payload = {
            "seed": raw.get("seed"),
            "data": raw.get("data", {}),
            "backbone": raw.get("backbone", {}),
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        with self.cache_locks_guard:
            return self.cache_locks.setdefault(key, threading.Lock())

    def run_one(self, suite: Suite, config: Path, timeout_seconds: float) -> tuple[Path, bool, str]:
        with self.cache_lock_for(config):
            if self.stop.is_set():
                return config, False, "controller_stopping"
            key = self.config_key(config)
            with self.lock:
                attempt = self.attempts.get(key, 0) + 1
                self.attempts[key] = attempt
            return self.run_one_locked(suite, config, attempt, timeout_seconds)

    def run_one_locked(
        self,
        suite: Suite,
        config: Path,
        attempt: int,
        timeout_seconds: float,
    ) -> tuple[Path, bool, str]:
        key = self.config_key(config)
        if complete_result(config):
            with self.lock:
                self.completed.add(key)
                self.failures.pop(key, None)
            return config, True, "already_complete"

        wait_for_gpu_memory(self.args.min_free_gpu_mb, self.stop)
        if self.stop.is_set():
            return config, False, "controller_stopping"
        log_path = self.log_path(suite, config, attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": str(self.args.cpu_threads_per_job),
                "MKL_NUM_THREADS": str(self.args.cpu_threads_per_job),
                "OPENBLAS_NUM_THREADS": str(self.args.cpu_threads_per_job),
                "NUMEXPR_NUM_THREADS": str(self.args.cpu_threads_per_job),
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "ugma.cli",
            "run",
            "--config",
            str(config),
            "--quiet",
        ]
        started = time.time()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} attempt={attempt} command={command!r}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            with self.lock:
                self.running[key] = {
                    "pid": process.pid,
                    "suite": suite.name,
                    "attempt": attempt,
                    "started_unix": started,
                    "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
                }
            self.write_state("running", suite.name)
            timed_out = False
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(process.pid)
                returncode = -999
            elapsed = time.time() - started
            handle.write(
                f"END {time.strftime('%Y-%m-%dT%H:%M:%S%z')} returncode={returncode} "
                f"timed_out={timed_out} elapsed_seconds={elapsed:.3f}\n"
            )
            handle.flush()

        success = returncode == 0 and complete_result(config)
        reason = "complete" if success else ("timeout" if timed_out else f"returncode={returncode}")
        with self.lock:
            self.running.pop(key, None)
            if success:
                self.completed.add(key)
                self.failures.pop(key, None)
            else:
                self.failures[key] = reason
        event = {
            "event": "job_finished",
            "time_unix": time.time(),
            "suite": suite.name,
            "config": key,
            "attempt": attempt,
            "pid": process.pid,
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "success": success,
            "reason": reason,
            "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        }
        append_jsonl(self.ledger_path, event, self.ledger_lock)
        self.write_state("running", suite.name)
        return config, success, reason

    def run_round(
        self,
        suite: Suite,
        pending: list[Path],
        workers: int,
        timeout_seconds: float,
    ) -> list[Path]:
        failed: list[Path] = []
        self.log(f"ROUND_START suite={suite.name} jobs={len(pending)} workers={workers}")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=suite.name) as executor:
            futures = {}
            for config in pending:
                futures[executor.submit(self.run_one, suite, config, timeout_seconds)] = config
            for future in as_completed(futures):
                config = futures[future]
                try:
                    _, success, reason = future.result()
                except Exception as exc:
                    success = False
                    reason = f"controller_exception={exc!r}"
                    key = self.config_key(config)
                    with self.lock:
                        self.running.pop(key, None)
                        self.failures[key] = reason
                    append_jsonl(
                        self.ledger_path,
                        {
                            "event": "controller_job_exception",
                            "time_unix": time.time(),
                            "suite": suite.name,
                            "config": key,
                            "traceback": traceback.format_exc(),
                        },
                        self.ledger_lock,
                    )
                if not success:
                    failed.append(config)
                    self.log(f"JOB_RETRY_QUEUED suite={suite.name} config={self.config_key(config)} reason={reason}")
        self.log(f"ROUND_END suite={suite.name} failed={len(failed)}")
        return failed

    def run_suite(self, suite: Suite, configs: list[Path]) -> bool:
        if not configs:
            self.log(f"SUITE_SKIPPED name={suite.name} reason=no_priority_configs")
            return True
        pending = [config for config in configs if not complete_result(config)]
        for config in configs:
            if complete_result(config):
                self.completed.add(self.config_key(config))
        workers = self.args.heavy_workers if suite.heavy else self.args.light_workers
        timeout_hours = self.args.heavy_timeout_hours if suite.heavy else self.args.light_timeout_hours
        timeout_seconds = timeout_hours * 3600
        self.log(f"SUITE_START name={suite.name} pending={len(pending)} workers={workers}")
        round_index = 0
        while pending and not self.stop.is_set():
            eligible = [
                config
                for config in pending
                if self.attempts.get(self.config_key(config), 0) < self.args.max_attempts
            ]
            exhausted = [config for config in pending if config not in eligible]
            for config in exhausted:
                key = self.config_key(config)
                self.failures[key] = f"max_attempts={self.args.max_attempts}"
                self.log(f"JOB_GAVE_UP suite={suite.name} config={key} attempts={self.attempts.get(key, 0)}")
            if not eligible:
                break
            round_index += 1
            isolated = round_index >= self.args.isolate_after_round
            round_workers = 1 if isolated else workers
            pending = self.run_round(suite, eligible, round_workers, timeout_seconds)
            if pending:
                largest_attempt = max(self.attempts[self.config_key(config)] for config in pending)
                backoff = min(
                    self.args.max_retry_backoff_minutes * 60,
                    self.args.retry_backoff_seconds * (2 ** min(largest_attempt - 1, 8)),
                )
                self.log(
                    f"RETRY_BACKOFF suite={suite.name} failed={len(pending)} seconds={backoff:.0f} "
                    f"next_workers=1"
                )
                self.stop.wait(backoff)
        completed_count = sum(complete_result(config) for config in configs)
        expected_count = len(configs)
        if completed_count != expected_count:
            self.log(f"SUITE_INCOMPLETE name={suite.name} completed={completed_count}/{expected_count}")
            self.write_state("suite_incomplete", suite.name)
            return False
        self.log(f"SUITE_COMPLETE name={suite.name} completed={completed_count}")
        self.write_state("suite_complete", suite.name)
        return True

    def aggregate(self) -> None:
        self.log("AGGREGATION_START")
        log_path = self.state_dir / "aggregation.log"
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            for suite in self.suites:
                command = [
                    sys.executable,
                    "-m",
                    "ugma.aggregate",
                    "--root",
                    suite.result_root,
                    "--output",
                    suite.aggregate_root,
                    "--source-sha256",
                    EXPECTED_SOURCE_SHA256,
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if completed.returncode:
                    raise RuntimeError(f"aggregation failed for {suite.name}: {completed.returncode}")
        self.log("AGGREGATION_COMPLETE")

    def run(self) -> int:
        heartbeat = threading.Thread(target=self.heartbeat, name="heartbeat", daemon=True)
        heartbeat.start()
        try:
            suite_configs = self.preflight()
            if self.args.dry_run:
                self.log("DRY_RUN_COMPLETE")
                return 0
            self.wait_for_external()
            # The external benchmark may have completed one matrix member.
            all_complete = True
            for suite in self.suites:
                all_complete = self.run_suite(suite, suite_configs[suite.name]) and all_complete
            self.aggregate()
            marker_name = "ALL_EXPERIMENTS_COMPLETE.json" if all_complete else "EXPERIMENTS_FINISHED_WITH_FAILURES.json"
            marker = self.state_dir / marker_name
            atomic_json(
                marker,
                {
                    "status": "complete" if all_complete else "incomplete",
                    "completed_unix": time.time(),
                    "elapsed_seconds": time.time() - self.started_unix,
                    "source_sha256": source_digest(),
                    "total_configs": self.total_configs,
                    "completed_configs": len(self.completed),
                    "suites": {suite.name: suite.expected for suite in self.suites},
                },
            )
            self.write_state("complete" if all_complete else "incomplete")
            self.log(f"EXPERIMENT_QUEUE_FINISHED complete={all_complete} marker={marker}")
            return 0 if all_complete else 2
        except Exception:
            failure = {
                "status": "controller_crashed",
                "time_unix": time.time(),
                "traceback": traceback.format_exc(),
            }
            atomic_json(self.state_dir / "CONTROLLER_CRASH.json", failure)
            self.log("CONTROLLER_CRASH " + failure["traceback"].replace("\n", " | "))
            return 1
        finally:
            self.stop.set()
            heartbeat.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="*", choices=[s.name for s in SUITES])
    parser.add_argument("--priority-file", type=Path)
    parser.add_argument("--state-dir", type=Path, default=Path("results/unattended"))
    parser.add_argument("--heavy-workers", type=int, default=1)
    parser.add_argument("--light-workers", type=int, default=1)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument("--heavy-timeout-hours", type=float, default=24.0)
    parser.add_argument("--light-timeout-hours", type=float, default=4.0)
    parser.add_argument("--min-free-gpu-mb", type=int, default=8000)
    parser.add_argument("--isolate-after-round", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--max-retry-backoff-minutes", type=float, default=30.0)
    parser.add_argument("--wait-pid-file", type=Path)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for field in ("heavy_workers", "light_workers", "cpu_threads_per_job", "max_attempts"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    suite_by_name = {suite.name: suite for suite in SUITES}
    selected = args.suites or [suite.name for suite in SUITES]
    suites = [suite_by_name[name] for name in selected]
    lock = ControllerLock(state_directory(args.state_dir) / "controller.lock")
    if not lock.acquire():
        print("Another run_remote_unattended.py controller is already active.", flush=True)
        raise SystemExit(0)
    try:
        raise SystemExit(AutonomousController(args, suites).run())
    finally:
        lock.release()


if __name__ == "__main__":
    main()
