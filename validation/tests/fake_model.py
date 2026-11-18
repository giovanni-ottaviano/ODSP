"""A tiny voter model for the framework tests: one sweep run of 3 points and one ensemble run
of 7 replicas split into blocks of 3, and an analysis returning one Check (the exit
probability of the voter model on the complete graph, 1/2 from a balanced start)."""

from __future__ import annotations

import math
import types
from pathlib import Path

from odspy import find_outputs

from validation.core import Check, Run


def make_model(name: str, bad: bool = False) -> types.ModuleType:
    """A model module named `name`; with `bad`, its sweep run carries an invalid option."""
    mod = types.ModuleType(name)
    mod.NAME = name

    def runs(scale: str) -> list[Run]:
        sweep = ("--model", "voter", "--graph", "complete", "--task", "ensemble",
                 "--replicas", "2", "--sweep", "n=10,20,30")
        if bad:
            sweep += ("--bogus-option", "1")
        return [
            Run("sweep", sweep, seed=1),
            Run("ens", ("--model", "voter", "--graph", "complete", "--n", "20",
                        "--task", "ensemble", "--replicas", "7"), seed=2, block=3),
        ]

    def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
        sweep = find_outputs(data / "sweep").get("sweep")
        if sweep is None or len(sweep) != 3:
            raise RuntimeError(f"{data / 'sweep'}: expected a sweep table of 3 rows")
        reps = find_outputs(data / "ens").get("replicas")
        up = reps["m"] > 0
        p = float(up.mean())
        return [Check(model=name, check=f"{name}.exit", quantity="exit probability to +1",
                      measured=p, error=math.sqrt(max(p * (1 - p), 0.25) / len(up)),
                      predicted=0.5, predicted_error=0.0, source="voter.exit",
                      sizes="N=20", statistics=f"{len(up)} replicas", warnings=[], figure=None)]

    mod.runs = runs
    mod.analyse = analyse
    return mod


_default = make_model("fake")
NAME = _default.NAME
runs = _default.runs
analyse = _default.analyse
