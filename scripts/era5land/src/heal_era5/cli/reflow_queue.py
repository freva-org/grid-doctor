#!/usr/bin/env python3
"""Queue sequential ERA5/ERA5-Land Reflow interval submissions.

This helper is designed for long-running HPC campaigns where each interval run
is safe on its own, but submitting too many Reflow array elements at once would
violate the scheduler's total job-submission limit.

The script reads one interval per line from a plain-text plan file, submits at
most ``--max-active-runs`` Reflow runs at a time, and polls each run until it
reaches a terminal state before submitting more work. Submission state is
persisted to JSON so the controller can be restarted without losing progress.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich_argparse import RichHelpFormatter

TERMINAL_STATES = frozenset(("SUCCESS", "FAILED", "CANCELLED"))
RUN_ID_RE = re.compile(r"run_id\s*=\s*(\S+)")


def log(message, *args):
    """Print one timestamped log line to stdout."""

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    text = message % args if args else message
    print(f"[{timestamp}] {text}", flush=True)


def utc_timestamp():
    """Return one ISO-8601 timestamp string in the local timezone."""

    return datetime.now().astimezone().isoformat()


def parse_args(argv=None, *, prog=None):
    """Parse command-line arguments for the queue controller."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Submit time-interval scoped Reflow runs in a controlled sequence so "
            "the scheduler never sees more than a small number of active runs."
        ),
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Text file containing one interval token pair per line, such as 1984,1994.",
    )
    parser.add_argument(
        "--command-template",
        required=True,
        help=(
            "Shell-style command template used for each submission. Supported "
            "placeholders: {interval}, {run_dir}, {index}, {label}."
        ),
    )
    parser.add_argument(
        "--run-dir-root",
        required=True,
        help="Root directory under which one run directory per interval is created.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional JSON state path. Defaults to <run-dir-root>/reflow-queue-state.json.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Seconds to sleep between status checks.",
    )
    parser.add_argument(
        "--max-active-runs",
        type=int,
        default=1,
        help="Maximum number of simultaneously active Reflow runs.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for Reflow status checks.",
    )
    parser.add_argument(
        "--store-path",
        default=None,
        help="Optional explicit Reflow SQLite manifest path.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        default=False,
        help="Keep submitting later intervals even if one run fails.",
    )
    parser.add_argument(
        "--write-sbatch",
        default=None,
        help=(
            "Optional path where an sbatch wrapper script should be written and "
            "the controller should exit without running."
        ),
    )
    parser.add_argument(
        "--sbatch-account",
        default=None,
        help="Account to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-partition",
        default=None,
        help="Partition to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-job-name",
        default="era5-reflow-queue",
        help="Job name to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-time",
        default="7-00:00:00",
        help="Wall-clock time limit to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-cpus-per-task",
        type=int,
        default=1,
        help="CPU count to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-mem",
        default="2G",
        help="Memory request to place in the generated sbatch wrapper.",
    )
    parser.add_argument(
        "--sbatch-output",
        default=None,
        help=(
            "Optional stdout/stderr path for the generated sbatch wrapper. "
            "Defaults to <run-dir-root>/controller-%%j.out."
        ),
    )
    return parser.parse_args(argv)


def load_plan(path):
    """Return normalized interval entries from a plain-text plan file."""

    entries = []
    plan_path = Path(path)
    for line_number, raw_line in enumerate(plan_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            raise ValueError(f"Invalid interval on line {line_number} of {plan_path}: {raw_line!r}")
        start, end = [token.strip() for token in line.split(",", 1)]
        if not start or not end:
            raise ValueError(f"Invalid interval on line {line_number} of {plan_path}: {raw_line!r}")
        entries.append(f"{start},{end}")
    if not entries:
        raise ValueError(f"Plan file {plan_path} does not contain any intervals.")
    return entries


def interval_label(interval):
    """Return a filesystem-safe label for one interval string."""

    return interval.replace(",", "_").replace("-", "")


def load_state(state_path, intervals, run_dir_root):
    """Load persisted queue state or create a fresh state structure."""

    path = Path(state_path)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"entries": []}

    existing = {}
    for entry in state.get("entries", []):
        existing[str(entry["interval"])] = dict(entry)

    entries = []
    for index, interval in enumerate(intervals, start=1):
        label = interval_label(interval)
        run_dir = str((Path(run_dir_root) / f"{index:03d}-{label}").resolve())
        entry = existing.get(interval, {})
        entry.setdefault("index", index)
        entry.setdefault("interval", interval)
        entry.setdefault("label", label)
        entry.setdefault("run_dir", run_dir)
        entry.setdefault("submitted", False)
        entry.setdefault("completed", False)
        entry.setdefault("run_id", None)
        entry.setdefault("status", "PENDING_SUBMISSION")
        entries.append(entry)

    return {"entries": entries}


def save_state(state_path, state):
    """Persist queue state as formatted JSON."""

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def build_submit_command(template, entry):
    """Render one submission command from the template and entry metadata."""

    rendered = template.format(
        interval=entry["interval"],
        run_dir=entry["run_dir"],
        index=entry["index"],
        label=entry["label"],
    )
    return shlex.split(rendered)


def submit_entry(template, entry, workdir):
    """Submit one queued interval and return the parsed Reflow run ID."""

    run_dir = Path(entry["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_submit_command(template, entry)
    log("Submitting interval %s with run dir %s", entry["interval"], run_dir)
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout or ""
    if output:
        for line in output.rstrip().splitlines():
            log("submit[%s] %s", entry["interval"], line)
    if completed.returncode != 0:
        raise RuntimeError(f"Submission failed for interval {entry['interval']} with exit code {completed.returncode}.")

    match = RUN_ID_RE.search(output)
    if not match:
        raise RuntimeError("Could not parse run_id from submission output for interval {}.".format(entry["interval"]))
    return match.group(1)


def status_command(python_executable, entry, store_path):
    """Build the Reflow status command for one submitted entry."""

    command = [
        python_executable,
        "-m",
        "heal_era5",
        "remap-reflow",
        "status",
        str(entry["run_id"]),
        "--run-dir",
        str(entry["run_dir"]),
        "--json",
    ]
    if store_path:
        command.extend(["--store-path", store_path])
    return command


def query_status(python_executable, entry, store_path, workdir):
    """Return the current Reflow run status string for one entry."""

    command = status_command(
        python_executable=python_executable,
        entry=entry,
        store_path=store_path,
    )
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "unknown status failure"
        raise RuntimeError("Status check failed for run {}: {}".format(entry["run_id"], detail))

    payload = json.loads(completed.stdout)
    run_info = payload.get("run", {})
    return str(run_info.get("status", "UNKNOWN")).upper()


def needs_status_refresh(entry):
    """Return whether an entry should be re-polled on controller restart."""

    if not entry.get("submitted") or not entry.get("run_id"):
        return False
    return (not entry.get("completed")) or entry.get("status") in ("FAILED", "CANCELLED", "SUBMITTED", "RUNNING")


def active_entries(entries):
    """Return submitted entries whose runs are not yet terminal."""

    active = []
    for entry in entries:
        if entry.get("submitted") and not entry.get("completed"):
            active.append(entry)
    return active


def pending_entries(entries):
    """Return entries that have not been submitted yet."""

    pending = []
    for entry in entries:
        if not entry.get("submitted"):
            pending.append(entry)
    return pending


def update_active_statuses(args, state, workdir):
    """Poll all active runs and update their persisted statuses."""

    for entry in active_entries(state["entries"]):
        status = query_status(
            python_executable=args.python_executable,
            entry=entry,
            store_path=args.store_path,
            workdir=workdir,
        )
        previous = entry.get("status")
        entry["status"] = status
        if previous != status:
            log(
                "Interval %s run %s changed state: %s -> %s",
                entry["interval"],
                entry["run_id"],
                previous,
                status,
            )
        if status in TERMINAL_STATES:
            entry["completed"] = True
            entry["completed_at"] = utc_timestamp()


def refresh_submitted_statuses(args, state, workdir):
    """Reconcile persisted controller state with current Reflow run statuses."""

    for entry in state["entries"]:
        if not needs_status_refresh(entry):
            continue
        status = query_status(
            python_executable=args.python_executable,
            entry=entry,
            store_path=args.store_path,
            workdir=workdir,
        )
        previous = entry.get("status")
        previous_completed = bool(entry.get("completed"))
        entry["status"] = status
        if status in TERMINAL_STATES:
            entry["completed"] = True
            entry.setdefault("completed_at", utc_timestamp())
        else:
            entry["completed"] = False
            entry.pop("completed_at", None)
        if previous != status or previous_completed != entry["completed"]:
            log(
                "Reconciled interval %s run %s: status=%s completed=%s",
                entry["interval"],
                entry["run_id"],
                entry["status"],
                entry["completed"],
            )


def submit_available_entries(args, state, workdir):
    """Submit queued entries until the active-run limit is reached."""

    active_count = len(active_entries(state["entries"]))
    available_slots = max(args.max_active_runs - active_count, 0)
    if available_slots == 0:
        return

    for entry in pending_entries(state["entries"])[:available_slots]:
        run_id = submit_entry(args.command_template, entry, workdir)
        entry["run_id"] = run_id
        entry["submitted"] = True
        entry["submitted_at"] = utc_timestamp()
        entry["status"] = "SUBMITTED"
        log("Submitted interval %s as run %s", entry["interval"], run_id)


def has_failed_entries(entries):
    """Return whether any completed entry ended in a failed terminal state."""

    for entry in entries:
        if entry.get("completed") and entry.get("status") in ("FAILED", "CANCELLED"):
            return True
    return False


def all_done(entries):
    """Return whether every entry has reached a terminal state."""

    for entry in entries:
        if not entry.get("completed"):
            return False
    return True


def shell_quote_join(parts):
    """Return one shell-safe command line assembled from argument tokens."""

    return " ".join(shlex.quote(part) for part in parts)


def build_controller_command(args, script_path):
    """Return the command used to run this controller with the current arguments."""

    command = [
        args.python_executable,
        str(script_path),
        "--plan",
        args.plan,
        "--command-template",
        args.command_template,
        "--run-dir-root",
        args.run_dir_root,
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-active-runs",
        str(args.max_active_runs),
        "--python-executable",
        args.python_executable,
    ]
    if args.state_path:
        command.extend(["--state-path", args.state_path])
    if args.store_path:
        command.extend(["--store-path", args.store_path])
    if args.continue_on_failure:
        command.append("--continue-on-failure")
    return command


def render_sbatch_script(args, script_path):
    """Return a reusable sbatch wrapper script for the queue controller."""

    output_path = args.sbatch_output or str(Path(args.run_dir_root).expanduser().resolve() / "controller-%j.out")
    lines = ["#!/bin/bash", f"#SBATCH --job-name={args.sbatch_job_name}"]
    if args.sbatch_account:
        lines.append(f"#SBATCH --account={args.sbatch_account}")
    if args.sbatch_partition:
        lines.append(f"#SBATCH --partition={args.sbatch_partition}")
    lines.extend(
        [
            f"#SBATCH --time={args.sbatch_time}",
            f"#SBATCH --cpus-per-task={args.sbatch_cpus_per_task}",
            f"#SBATCH --mem={args.sbatch_mem}",
            f"#SBATCH --output={output_path}",
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(script_path.parent))}",
            shell_quote_join(build_controller_command(args, script_path)),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None, *, prog=None):
    """Run the queue controller until all intervals finish or one fails."""

    args = parse_args(argv, prog=prog)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be a positive integer.")
    if args.max_active_runs <= 0:
        raise ValueError("--max-active-runs must be a positive integer.")

    workdir = Path(__file__).resolve().parents[2]
    intervals = load_plan(args.plan)
    run_dir_root = Path(args.run_dir_root).expanduser().resolve()
    state_path = (
        Path(args.state_path).expanduser().resolve() if args.state_path else run_dir_root / "reflow-queue-state.json"
    )

    state = load_state(
        state_path=state_path,
        intervals=intervals,
        run_dir_root=run_dir_root,
    )
    if args.write_sbatch:
        sbatch_path = Path(args.write_sbatch).expanduser().resolve()
        sbatch_path.write_text(
            render_sbatch_script(args, Path(__file__).resolve()),
            encoding="utf-8",
        )
        os.chmod(sbatch_path, 0o755)
        log("\n Wrote sbatch wrapper to %s", sbatch_path)
        return 0

    refresh_submitted_statuses(args, state, workdir)
    save_state(state_path, state)
    log("Loaded %d intervals from %s", len(intervals), args.plan)
    log("State file: %s", state_path)

    while True:
        update_active_statuses(args, state, workdir)
        save_state(state_path, state)

        if has_failed_entries(state["entries"]) and not args.continue_on_failure:
            log("Stopping after a failed run. Inspect %s for details.", state_path)
            return 1

        if all_done(state["entries"]):
            log("All interval runs reached terminal states.")
            return 0

        submit_available_entries(args, state, workdir)
        save_state(state_path, state)

        if all_done(state["entries"]):
            log("All interval runs reached terminal states.")
            return 0

        active = active_entries(state["entries"])
        pending = pending_entries(state["entries"])
        log(
            "Sleeping %d seconds with %d active and %d pending interval runs.",
            args.poll_seconds,
            len(active),
            len(pending),
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
