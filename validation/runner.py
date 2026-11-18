"""Run jobs on this machine within a thread budget, and merge each run's finished jobs.

A job is done when its `<prefix>.done` marker exists: the runner writes it only after odsp
exits with 0, so a job killed or failed midway is simply run again (its partial outputs are
removed first). The driver's output and messages for each job go to `<prefix>.log`.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from odspy import InputError
from odspy.io import SUFFIXES
from odspy.merge import merge

from .core import Job, Run, merged_prefix, split

_POLL = 0.05    # seconds between checks of the running jobs


def done_marker(job: Job) -> Path:
    return Path(f"{job.prefix}.done")


def is_done(job: Job) -> bool:
    """The `<prefix>.done` marker exists."""
    return done_marker(job).exists()


def _clear(job: Job) -> None:
    """Remove what an earlier attempt of this job left (marker, outputs, log), so a rerun
    starts clean: prefixes are p<point>_b<first>, so '<name>_' or '<name>.' matches only this
    job's files."""
    folder = job.prefix.parent
    if folder.is_dir():
        for f in folder.iterdir():
            if f.name.startswith((job.prefix.name + "_", job.prefix.name + ".")):
                f.unlink()


def run_jobs(jobs: Sequence[Job], threads_budget: int, odsp: Path) -> list[Job]:
    """Run `jobs` with `odsp`, in order, starting the next job while the threads of the
    running jobs plus its own fit in `threads_budget` (a job wider than the budget runs
    alone), and write `<prefix>.done` for every job that exits with 0. Returns the jobs that
    failed (non-zero exit or could not start), in the given order."""
    if threads_budget < 1:
        raise ValueError(f"threads budget must be >= 1 (got {threads_budget})")
    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen, object]] = []
    failed: set[int] = set()
    try:
        while pending or running:
            used = sum(j.threads for j, _, _ in running)
            while pending and (not running or used + pending[0].threads <= threads_budget):
                job = pending.pop(0)
                _clear(job)
                job.prefix.parent.mkdir(parents=True, exist_ok=True)
                log = open(f"{job.prefix}.log", "w", encoding="utf-8")
                try:
                    proc = subprocess.Popen([str(odsp), *job.args], stdout=log,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                except OSError as e:
                    log.write(f"could not start {odsp}: {e}\n")
                    log.close()
                    failed.add(id(job))
                    continue
                running.append((job, proc, log))
                used += job.threads
            time.sleep(_POLL)
            still = []
            for job, proc, log in running:
                rc = proc.poll()
                if rc is None:
                    still.append((job, proc, log))
                    continue
                log.close()
                if rc == 0:
                    done_marker(job).touch()
                else:
                    failed.add(id(job))
            running = still
    finally:
        # interrupted (Ctrl-C): stop the children instead of leaving them running unmarked
        for _, proc, log in running:
            proc.kill()
            proc.wait()
            log.close()
    return [j for j in jobs if id(j) in failed]


def is_merged(scale_root: Path, model: str, run: Run) -> bool:
    """The run's jobs are all done and its merged prefix exists and is newer than every job
    (a job rerun after the merge makes the merge stale)."""
    jobs = split(model, run, scale_root)
    cfg = Path(f"{merged_prefix(scale_root, model, run.name)}_config.txt")
    if not cfg.exists() or not all(is_done(j) for j in jobs):
        return False
    made = cfg.stat().st_mtime_ns
    return all(done_marker(j).stat().st_mtime_ns <= made for j in jobs)


def _remove_merged(out: Path) -> None:
    """Remove the merged outputs of one run (only files named for this exact prefix), so a
    failed merge never leaves an older merge to be analysed."""
    if not out.parent.is_dir():
        return
    stem = out.name + "_"
    kinds = {f"{s}.csv" for s in SUFFIXES} | {"config.txt", "merge_report.txt"}
    for f in out.parent.iterdir():
        rest = f.name[len(stem):]
        if f.name.startswith(stem) and (rest in kinds or (
                rest.startswith("point") and rest.partition("_")[0][5:].isdigit()
                and rest.partition("_")[2] in kinds)):
            f.unlink()


def merge_run(scale_root: Path, model: str, run: Run) -> bool:
    """Merge the jobs of `run` into merged_prefix(scale_root, model, run.name). False (and no
    merged outputs left) when a job is not done or odspy refuses the merge; merge_run_reason
    says why."""
    return merge_run_reason(scale_root, model, run) is None


def merge_run_reason(scale_root: Path, model: str, run: Run) -> str | None:
    """As merge_run, returning None on success and the reason of a failure otherwise."""
    jobs = split(model, run, scale_root)
    out = merged_prefix(scale_root, model, run.name)
    missing = [j for j in jobs if not is_done(j)]
    if missing:
        _remove_merged(out)
        return f"{len(missing)} of {len(jobs)} job(s) not done (first: {missing[0].prefix})"
    try:
        merge(out, [j.prefix for j in jobs])
    except (InputError, OSError) as e:
        _remove_merged(out)
        return str(e)
    return None
