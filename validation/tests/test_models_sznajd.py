"""The Sznajd model (validation/models/sznajd.py): exact exit probability on the complete graph,
sharpening of the exit-probability step on the square lattice."""

from __future__ import annotations

import numpy as np

from validation.core import _value
from validation.models import MODELS

from .smoke import run_smoke

CHECKS = ["sznajd.complete_exit", "sznajd.lattice_step", "sznajd.lattice_half"]


def test_sznajd_smoke(tmp_path, odsp_bin):
    assert "sznajd" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "sznajd", CHECKS)
    by = {r["check"]: r for r in rows}
    c = by["sznajd.complete_exit"]
    assert float(c["predicted"]) == 1.0 and c["source"] == "sznajd_complete"
    assert "n(n-1)" in c["quantity"]
    assert "replaces" in c["warnings"] and "step at 1/2" in c["warnings"]
    s = by["sznajd.lattice_step"]
    assert float(s["predicted"]) == 0.0 and s["source"] == "sznajd_lattice"
    assert s["one_sided"] == "below"                        # passes when b < 0 by 3 errors
    assert float(s["tolerance_sigma"]) == 3.0
    assert "one-sided" in s["quantity"]
    assert "(fit L >= 12)" in s["sizes"]                    # R12: all sizes but the smallest
    assert "fit over all sizes (L >= 8)" in s["warnings"]
    h = by["sznajd.lattice_half"]
    assert float(h["predicted"]) == 0.5 and "binomial" in h["statistics"]
    assert "E(1/2) at every L" in h["warnings"]


def test_sznajd_runs():
    mod = MODELS["sznajd"]
    for scale in ("smoke", "small", "full"):
        runs = {r.name: r for r in mod.runs(scale)}
        assert set(runs) == {"complete"} | {f"lattice_L{L}" for L in mod.SCALE[scale]["lattice"][0]}
        assert _value(runs["complete"].args, "--graph") == "complete"
        for name, r in runs.items():
            assert _value(r.args, "--model") == "sznajd" and _value(r.args, "--init") == "random"
            if name.startswith("lattice"):
                L = int(name[len("lattice_L"):])
                assert _value(r.args, "--dims") == f"{L}x{L}"
                assert _value(r.args, "--boundary") == "periodic"
                # the grid is symmetric about 1/2 and contains 1/2
                ups = np.array([float(v) for v in _value(r.args, "--sweep")[3:].split(",")])
                assert ups.size == np.unique(ups).size and 0.5 in ups
                assert np.allclose(np.sort(1 - ups), ups, atol=1e-12)
    full = mod.SCALE["full"]["lattice"][0]
    assert len(full) >= 5 and max(full) >= 8 * min(full)


def test_width_estimator():
    mod = MODELS["sznajd"]
    f = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    E = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    # E = 0.1 between 0.3 and 0.4 (at 0.35), E = 0.9 between 0.6 and 0.7 (at 0.65)
    assert abs(mod.transition_width(f, E) - 0.3) < 1e-12
    # noise: the first upward crossing of each level counts
    E2 = np.array([0.0, 0.12, 0.08, 0.95, 1.0])
    f10 = 0.3 + 0.1 * 0.1 / 0.12
    f90 = 0.5 + 0.1 * (0.9 - 0.08) / (0.95 - 0.08)
    assert abs(mod.transition_width(f, E2) - (f90 - f10)) < 1e-12
