"""The validation campaign's command.

    python -m validation.campaign {list,run,slurm,status,merge,analyse} --scale S
        [--models a,b] [--threads N] [--force] [--root DIR] [--odsp PATH] [--conf FILE]

list     the runs of each model and the jobs they split into
run      run the jobs not done yet on this machine (at most --threads threads at once),
         then merge and analyse
slurm    write SLURM job arrays for the jobs not done yet (validation/slurm.conf)
status   jobs done, runs merged, models analysed
merge    merge every run whose jobs are all done (skipping merges that are up to date)
analyse  analyse every model whose runs are all merged, and write summary.csv/summary.md

Results go to <root>/<scale>/ (default validation/results/<scale>/); run, slurm, merge and
analyse append what they did, skipped or failed to <root>/<scale>/log.txt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

from . import models as models_pkg
from .core import SCALES, Check, Job, Run, split
from .report import write_summary
from .runner import done_marker, is_done, is_merged, merge_run_reason, run_jobs
from .slurm import DEFAULT_CONF, REPO, read_conf, write_arrays

DEFAULT_ROOT = REPO / "validation" / "results"
DEFAULT_ODSP = REPO / "bin" / "odsp"
CHECKS_FILE = "checks.json"     # one per model: the checks of its last successful analysis


@dataclass
class Plan:
    """One model's runs at this scale and the jobs of each run."""
    model: str
    module: ModuleType
    runs: list[Run]
    jobs: dict[str, list[Job]]


class Campaign:
    def __init__(self, args: argparse.Namespace, selected: dict[str, ModuleType]):
        self.scale: str = args.scale
        self.root = Path(args.root).resolve()
        self.scale_root = self.root / self.scale
        self.threads: int = args.threads
        self.force: bool = args.force
        self.odsp = Path(args.odsp).resolve()
        self.conf = Path(args.conf) if args.conf else DEFAULT_CONF
        self.selected = selected
        self.plans: list[Plan] = []
        self.ok = True

    # --- logging ---------------------------------------------------------------------------
    def log(self, stage: str, message: str) -> None:
        """Print `message` and append it to <scale>/log.txt with a time stamp."""
        print(message)
        self.scale_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.scale_root / "log.txt", "a", encoding="utf-8") as f:
            for line in message.splitlines() or [""]:
                f.write(f"{stamp} [{stage}] {line}\n")

    def fail(self, stage: str, message: str) -> None:
        self.ok = False
        self.log(stage, message)

    # --- planning --------------------------------------------------------------------------
    def plan(self, stage: str, quiet: bool = False) -> None:
        """Ask each selected model for its runs and split them; a model whose runs cannot be
        split is reported and left out."""
        for name, mod in self.selected.items():
            try:
                runs = list(mod.runs(self.scale))
                names = [r.name for r in runs]
                dup = sorted({n for n in names if names.count(n) > 1})
                if dup:
                    raise ValueError(f"run name(s) {', '.join(dup)} used more than once")
                jobs = {r.name: split(name, r, self.scale_root) for r in runs}
            except Exception as e:      # a broken model must not stop the others
                if quiet:
                    print(f"{name}: cannot plan its runs: {e}", file=sys.stderr)
                    self.ok = False
                else:
                    self.fail(stage, f"FAILED {name}: cannot plan its runs: {type(e).__name__}: {e}")
                continue
            self.plans.append(Plan(name, mod, runs, jobs))

    def all_jobs(self) -> list[Job]:
        return [j for p in self.plans for r in p.runs for j in p.jobs[r.name]]

    # --- stages ----------------------------------------------------------------------------
    def jobs(self) -> None:
        if not self.odsp.exists():
            self.fail("run", f"FAILED: {self.odsp} not found (build it with make -j10 bin/odsp)")
            return
        jobs = self.all_jobs()
        todo = jobs if self.force else [j for j in jobs if not is_done(j)]
        self.log("run", f"{len(todo)} job(s) to run, {len(jobs) - len(todo)} already done "
                        f"(budget {self.threads} threads, odsp {self.odsp})")
        failed = run_jobs(todo, self.threads, self.odsp)
        for j in failed:
            self.fail("run", f"FAILED {j.model}/{j.run} job {j.prefix}: {_error_text(j)} "
                             f"(full output in {j.prefix}.log)")
        self.log("run", f"{len(todo) - len(failed)} job(s) done, {len(failed)} failed")

    def merge(self) -> None:
        merged = current = 0
        for p in self.plans:
            for r in p.runs:
                jobs = p.jobs[r.name]
                if not self.force and is_merged(self.scale_root, p.model, r):
                    current += 1
                    continue
                todo = [j for j in jobs if not is_done(j)]
                if todo:
                    self.fail("merge", f"not merged {p.model}/{r.name}: {len(todo)} of {len(jobs)} "
                                       f"job(s) not done")
                    continue
                reason = merge_run_reason(self.scale_root, p.model, r)
                if reason is None:
                    merged += 1
                else:
                    self.fail("merge", f"FAILED merge {p.model}/{r.name}: {reason}")
        self.log("merge", f"{merged} run(s) merged, {current} already up to date")

    def analyse(self) -> None:
        figures = self.scale_root / "figures"
        for p in self.plans:
            stored = self.scale_root / p.model / CHECKS_FILE
            waiting = [r.name for r in p.runs if not is_merged(self.scale_root, p.model, r)]
            if waiting:
                stored.unlink(missing_ok=True)
                self.fail("analyse", f"skipped {p.model}: run(s) not ready (jobs not done or not "
                                     f"merged): {', '.join(waiting)}")
                continue
            figures.mkdir(parents=True, exist_ok=True)
            try:
                checks = list(p.module.analyse(self.scale, self.scale_root / p.model, figures))
                bad = [c for c in checks if not isinstance(c, Check)]
                if bad:
                    raise TypeError(f"analyse returned {type(bad[0]).__name__}, not Check")
            except Exception as e:
                stored.unlink(missing_ok=True)
                tb = "".join(traceback.format_exception(e)).rstrip()
                self.fail("analyse", f"FAILED analyse {p.model}: {type(e).__name__}: {e}\n{tb}")
                continue
            stored.write_text(json.dumps([dataclasses.asdict(c) for c in checks], indent=1),
                              encoding="utf-8")
            self.log("analyse", f"analysed {p.model}: {len(checks)} check(s)")
            for c in checks:
                if c.flagged:
                    what = (f"not below -{c.tolerance_sigma:g} (one-sided check)" if c.one_sided
                            else f"beyond the tolerance {c.tolerance_sigma:g}")
                    self.log("analyse", f"  {c.check}: sigma = {c.sigma:+.2f} {what}")
        checks = self.stored_checks()
        write_summary(checks, self.scale_root)
        self.log("analyse", f"summary: {len(checks)} check(s) -> {self.scale_root / 'summary.csv'}, "
                            "summary.md")

    def stored_checks(self) -> list[Check]:
        """The stored checks of every registered model (not only the selected ones, so
        analysing one model keeps the others' rows), in registry order."""
        checks: list[Check] = []
        for name in models_pkg.MODELS:
            stored = self.scale_root / name / CHECKS_FILE
            if stored.exists():
                checks += [Check(**d) for d in json.loads(stored.read_text(encoding="utf-8"))]
        return checks

    def slurm(self) -> None:
        try:
            conf = read_conf(self.conf)
        except (OSError, ValueError) as e:
            self.fail("slurm", f"FAILED: {e}")
            return
        jobs = self.all_jobs()
        if self.force:
            for j in jobs:          # a forced job is not done until its array task says so
                done_marker(j).unlink(missing_ok=True)
        todo = [j for j in jobs if not is_done(j)]
        out = self.scale_root / "slurm"
        scripts = write_arrays(todo, conf, out)
        self.log("slurm", f"{len(todo)} job(s) not done, {len(jobs) - len(todo)} done; wrote "
                          f"{len(scripts)} array(s) in {out}")
        for s in scripts:
            self.log("slurm", f"  submit with: sbatch {s}")
        if scripts:
            self.log("slurm", "then: python -m validation.campaign merge / analyse "
                              f"--scale {self.scale}")

    def status(self) -> None:
        for p in self.plans:
            stored = self.scale_root / p.model / CHECKS_FILE
            n = len(json.loads(stored.read_text(encoding="utf-8"))) if stored.exists() else None
            print(f"{p.model}: " + (f"analysed, {n} check(s)" if n is not None else "not analysed"))
            for r in p.runs:
                jobs = p.jobs[r.name]
                done = sum(is_done(j) for j in jobs)
                merged = ("merged" if is_merged(self.scale_root, p.model, r)
                          else "not merged" if done == len(jobs) else "")
                print(f"  {r.name}: {done}/{len(jobs)} job(s) done{', ' + merged if merged else ''}")

    def list(self) -> None:
        for p in self.plans:
            print(f"{p.model}:")
            for r in p.runs:
                jobs = p.jobs[r.name]
                print(f"  {r.name}: {len(jobs)} job(s) x {r.threads} thread(s); odsp "
                      f"{shlex.join(r.args)} --seed {r.seed}")
        jobs = self.all_jobs()
        print(f"{len(jobs)} job(s), {sum(j.threads for j in jobs)} thread-job(s)")


def _error_text(job: Job) -> str:
    """The driver's error message from a job's output: from the first line containing
    'error' to the end (joined on one line, at most 300 characters), else the last line."""
    try:
        lines = [l.strip() for l in Path(f"{job.prefix}.log").read_text().splitlines() if l.strip()]
    except OSError:
        lines = []
    if not lines:
        return "odsp exited with an error"
    first = next((i for i, l in enumerate(lines) if "error" in l.lower()), len(lines) - 1)
    text = " ".join(lines[first:])
    return text if len(text) <= 300 else text[:297] + "..."


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m validation.campaign",
                                description="Run the ODSP validation campaign.",
                                epilog="See validation/README.md.")
    p.add_argument("stage", choices=("list", "run", "slurm", "status", "merge", "analyse"))
    p.add_argument("--scale", required=True, choices=SCALES,
                   help="smoke: seconds (tests); small: about an hour on one machine; full: cluster")
    p.add_argument("--models", help="comma-separated model names (default: all)")
    p.add_argument("--threads", type=int, default=10,
                   help="local thread budget for run: jobs start while their threads fit [10]")
    p.add_argument("--force", action="store_true",
                   help="rerun (run, slurm) or re-merge (merge, run) what is already done")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="results root [validation/results]")
    p.add_argument("--odsp", default=str(DEFAULT_ODSP), help="the odsp binary [bin/odsp]")
    p.add_argument("--conf", help="SLURM configuration [validation/slurm.conf]")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage; 0 when everything it attempted succeeded, 1 otherwise (2: usage)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    registry = models_pkg.MODELS
    if args.models:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
        unknown = [n for n in names if n not in registry]
        if unknown:
            parser.error(f"unknown model(s) {', '.join(unknown)} "
                         f"(known: {', '.join(registry) or 'none'})")
        selected = {n: registry[n] for n in names}
    else:
        selected = dict(registry)
    os.environ.setdefault("MPLBACKEND", "Agg")      # figures are files, never windows
    c = Campaign(args, selected)
    stage = args.stage
    reading = stage in ("list", "status")
    c.plan("run" if stage == "run" else stage, quiet=reading)
    if stage == "list":
        c.list()
    elif stage == "status":
        c.status()
    elif stage == "slurm":
        c.slurm()
    elif stage == "merge":
        c.merge()
    elif stage == "analyse":
        c.analyse()
    else:
        c.jobs()
        c.merge()
        c.analyse()
    return 0 if c.ok else 1


if __name__ == "__main__":
    sys.exit(main())
