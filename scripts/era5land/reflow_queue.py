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
from typing import Dict, Iterable, List, Optional


TERMINAL_STATES = frozenset(("SUCCESS", "FAILED", "CANCELLED"))
RUN_ID_RE = re.compile(r"run_id\s*=\s*(\S+)")


def log(message, *args):
    """Print one timestamped log line to stdout."""

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    text = message % args if args else message
    print("[%s] %s" % (timestamp, text), flush=True)


def parse_args():
    """Parse command-line arguments for the queue controller."""

    parser = argparse.ArgumentParser(
        description=(
            "Submit interval-scoped Reflow runs in a controlled sequence so "
            "the scheduler never sees more than a small number of active runs."
        )
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
    return parser.parse_args()


def load_plan(path):
    """Return normalized interval entries from a plain-text plan file."""

    entries = []
    plan_path = Path(path)
    for line_number, raw_line in enumerate(plan_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            raise ValueError(
                "Invalid interval on line %d of %s: %r"
                % (line_number, plan_path, raw_line)
            )
        start, end = [token.strip() for token in line.split(",", 1)]
        if not start or not end:
            raise ValueError(
                "Invalid interval on line %d of %s: %r"
                % (line_number, plan_path, raw_line)
            )
        entries.append("%s,%s" % (start, end))
    if not entries:
        raise ValueError("Plan file %s does not contain any intervals." % plan_path)
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
        run_dir = str((Path(run_dir_root) / ("%03d-%s" % (index, label))).resolve())
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
        universal_newlines=True,
    )
    output = completed.stdout or ""
    if output:
        for line in output.rstrip().splitlines():
            log("submit[%s] %s", entry["interval"], line)
    if completed.returncode != 0:
        raise RuntimeError(
            "Submission failed for interval %s with exit code %d."
            % (entry["interval"], completed.returncode)
        )

    match = RUN_ID_RE.search(output)
    if not match:
        raise RuntimeError(
            "Could not parse run_id from submission output for interval %s."
            % entry["interval"]
        )
    return match.group(1)


def status_command(python_executable, converter_path, entry, store_path):
    """Build the Reflow status command for one submitted entry."""

    command = [
        python_executable,
        str(converter_path),
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


def query_status(python_executable, converter_path, entry, store_path, workdir):
    """Return the current Reflow run status string for one entry."""

    command = status_command(
        python_executable=python_executable,
        converter_path=converter_path,
        entry=entry,
        store_path=store_path,
    )
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "unknown status failure"
        raise RuntimeError(
            "Status check failed for run %s: %s" % (entry["run_id"], detail)
        )

    payload = json.loads(completed.stdout)
    run_info = payload.get("run", {})
    return str(run_info.get("status", "UNKNOWN")).upper()


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


def update_active_statuses(args, state, converter_path, workdir):
    """Poll all active runs and update their persisted statuses."""

    for entry in active_entries(state["entries"]):
        status = query_status(
            python_executable=args.python_executable,
            converter_path=converter_path,
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
            entry["completed_at"] = datetime.utcnow().isoformat() + "Z"


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
        entry["submitted_at"] = datetime.utcnow().isoformat() + "Z"
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


def main():
    """Run the queue controller until all intervals finish or one fails."""

    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be a positive integer.")
    if args.max_active_runs <= 0:
        raise ValueError("--max-active-runs must be a positive integer.")

    workdir = Path(__file__).resolve().parent
    converter_path = workdir / "converter.py"
    intervals = load_plan(args.plan)
    run_dir_root = Path(args.run_dir_root).expanduser().resolve()
    state_path = (
        Path(args.state_path).expanduser().resolve()
        if args.state_path
        else run_dir_root / "reflow-queue-state.json"
    )

    state = load_state(
        state_path=state_path,
        intervals=intervals,
        run_dir_root=run_dir_root,
    )
    save_state(state_path, state)
    log("Loaded %d intervals from %s", len(intervals), args.plan)
    log("State file: %s", state_path)

    while True:
        update_active_statuses(args, state, converter_path, workdir)
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
