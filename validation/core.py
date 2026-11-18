"""The campaign's data model: a model's logical runs, the jobs they split into, and the checks
its analysis returns.

A model module (validation/models/<name>.py) defines `NAME: str`, `runs(scale) -> list[Run]`
and `analyse(scale, data, figures) -> list[Check]`, where `data` is results/<scale>/<model>/
holding one merged prefix per run (named after the run) and `figures` the folder for its
figures. `split` turns one run into the jobs that run it: one job per sweep point
(--sweep-point) and, for an ensemble with `block`, one job per replica block
(--first-replica); `odspy.merge.merge` joins the jobs' outputs back into the run's prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from odspy.merge import sweep_grid_size

SCALES = ("smoke", "small", "full")      # smoke: seconds, for pytest; small: ~1 h locally; full: cluster

# Options split sets on every job; a model passing them would clash with the split.
_RESERVED = ("--csv", "--seed", "--threads", "--sweep-point", "--first-replica")


@dataclass(frozen=True)
class Run:
    name: str                  # unique within the model; merged prefix results/<scale>/<model>/<name>
    args: tuple[str, ...]      # odsp arguments without --csv/--seed/--threads/--sweep-point/--first-replica
    seed: int
    threads: int = 1
    block: int | None = None   # ensemble replicas per job; None: no replica split


@dataclass(frozen=True)
class Job:
    model: str
    run: str
    args: tuple[str, ...]      # complete odsp command line after "odsp"
    prefix: Path               # results/<scale>/<model>/parts/<run>/p<point>_b<first>
    threads: int


@dataclass
class Check:
    model: str
    check: str                 # e.g. "ising.tc"
    quantity: str              # what was measured, in words
    measured: float
    error: float
    predicted: float
    predicted_error: float     # 0 for exact values
    source: str                # key into theory.CITATIONS
    sizes: str
    statistics: str            # replicas / samples / chains
    warnings: list[str] = field(default_factory=list)
    figure: str | None = None
    tolerance_sigma: float = 3.0   # regression tolerance at smoke/small scale

    @property
    def sigma(self) -> float:
        """(measured - predicted) / hypot(error, predicted_error); with both errors zero, 0 for
        an exact match and +-inf otherwise."""
        diff = self.measured - self.predicted
        scale = math.hypot(self.error, self.predicted_error)
        if scale == 0:
            return 0.0 if diff == 0 else math.copysign(math.inf, diff)
        return diff / scale


def _value(args: tuple[str, ...], key: str) -> str | None:
    """The value of option `key` in an argument list ('--key value' or '--key=value'), the last
    one when repeated, as the driver reads it; None when absent."""
    value = None
    for i, a in enumerate(args):
        if a == key and i + 1 < len(args):
            value = args[i + 1]
        elif a.startswith(key + "="):
            value = a[len(key) + 1:]
    return value


def _without(args: tuple[str, ...], key: str) -> tuple[str, ...]:
    """`args` with every occurrence of option `key` and its value removed."""
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a == key:
            skip = True
        elif not a.startswith(key + "="):
            out.append(a)
    return tuple(out)


def merged_prefix(scale_root: Path, model: str, run: str) -> Path:
    """Where the merged outputs of a run go: results/<scale>/<model>/<run>."""
    return Path(scale_root) / model / run


def split(model: str, run: Run, scale_root: Path) -> list[Job]:
    """The jobs of one run: one per sweep point when its args contain --sweep (counted with
    sweep_grid_size), times one per replica block of `run.block` replicas covering --replicas
    when `block` is set (ensembles only). Every job gets --seed run.seed --threads
    run.threads --csv <prefix>; the prefix is parts/<run>/p<point>_b<first replica> under the
    model's folder (point 0 and first replica 0 for a run without that split)."""
    for key in _RESERVED:
        if any(a == key or a.startswith(key + "=") for a in run.args):
            raise ValueError(f"{model}/{run.name}: {key} is set by the campaign, not by the model")
    if run.threads < 1:
        raise ValueError(f"{model}/{run.name}: threads must be >= 1 (got {run.threads})")
    sweep = _value(run.args, "--sweep")
    points: list[int | None] = [None]
    if sweep is not None:
        config = {"sweep": sweep}
        sweep2 = _value(run.args, "--sweep2")
        if sweep2 is not None:
            config["sweep2"] = sweep2
        points = list(range(sweep_grid_size(config) or 0))
    blocks: list[tuple[int, int] | None] = [None]
    if run.block is not None:
        if _value(run.args, "--task") != "ensemble":
            raise ValueError(f"{model}/{run.name}: block applies only to runs with --task ensemble")
        if run.block < 1:
            raise ValueError(f"{model}/{run.name}: block must be >= 1 (got {run.block})")
        replicas = _value(run.args, "--replicas")
        if replicas is None or not replicas.isdigit() or int(replicas) < 1:
            raise ValueError(f"{model}/{run.name}: block needs --replicas R (R >= 1) in the args")
        total = int(replicas)
        blocks = [(k, min(run.block, total - k)) for k in range(0, total, run.block)]
    base = _without(run.args, "--replicas") if run.block is not None else run.args
    parts = Path(scale_root) / model / "parts" / run.name
    jobs: list[Job] = []
    for point in points:
        for blk in blocks:
            prefix = parts / f"p{point or 0}_b{blk[0] if blk else 0}"
            args = base
            if point is not None:
                args += ("--sweep-point", str(point))
            if blk is not None:
                args += ("--replicas", str(blk[1]), "--first-replica", str(blk[0]))
            args += ("--seed", str(run.seed), "--threads", str(run.threads), "--csv", str(prefix))
            jobs.append(Job(model, run.name, args, prefix, run.threads))
    return jobs
