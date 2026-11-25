"""The 2D majority-vote model's checks (validation/models/majorityvote.py), at smoke scale."""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.models import MODELS
from validation.theory import ISING_BINDER, ISING_EXPONENTS, MV_QC

from .smoke import run_smoke

CHECKS = ["mv.qc", "mv.binder", "mv.inv_nu", "mv.gamma_nu", "mv.beta_nu"]


def test_mv_smoke(tmp_path, odsp_bin):
    from validation.models._critical import FIT_RULE

    assert "mv" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "mv", CHECKS)
    by = {r["check"]: r for r in rows}
    predicted = {r["check"]: float(r["predicted"]) for r in rows}
    assert predicted["mv.qc"] == pytest.approx(MV_QC[0], abs=1e-12)
    assert float(by["mv.qc"]["predicted_error"]) == pytest.approx(MV_QC[1])
    assert by["mv.qc"]["source"] == "mv_qc"
    assert "q_c from the Binder crossing" in by["mv.qc"]["quantity"]
    # R14: U* of the 2D Ising class (periodic square lattice), universal within the class
    assert predicted["mv.binder"] == pytest.approx(ISING_BINDER[0], abs=1e-12)
    assert float(by["mv.binder"]["predicted_error"]) == pytest.approx(ISING_BINDER[1])
    assert by["mv.binder"]["source"] == "mv_binder"
    assert "universal" in by["mv.binder"]["warnings"]
    assert predicted["mv.inv_nu"] == pytest.approx(1.0 / ISING_EXPONENTS["nu"])
    assert predicted["mv.gamma_nu"] == pytest.approx(7.0 / 4.0)
    assert predicted["mv.beta_nu"] == pytest.approx(-1.0 / 8.0)
    # the campaign rule (R12): smoke sizes 8, 12, 16, 24 -> exponent fits over L >= 12
    for name in ("mv.inv_nu", "mv.gamma_nu", "mv.beta_nu"):
        assert by[name]["source"] == "mv_exponents"
        assert "(fit L >= 12)" in by[name]["sizes"], by[name]["sizes"]
        assert FIT_RULE in by[name]["warnings"]
        assert "finite-size diagnostic: fit over all sizes (L >= 8)" in by[name]["warnings"]
    assert "|dU/dq| at the cited q_c" in by["mv.inv_nu"]["quantity"]
    # R13: chi' at a fixed scaling variable, the replaced estimator named in the row
    g = by["mv.gamma_nu"]
    assert "chi' = N(<x^2> - <|x|>^2) near its broad maximum" in g["quantity"]
    assert "(q - q_c) L^(1/nu) = 0.2" in g["quantity"]
    assert "peak_vs_size not used" in g["warnings"]
    assert "chi' at fixed (q - q_c) L^(1/nu) = 0.2" in g["warnings"]   # as in the quantity
    assert "fixed from pilot scans at L = 16, 32" in g["warnings"]
    assert "before any exponent fit" in g["warnings"]
    # q_c is a measured value: its uncertainty, propagated into every exponent read at it
    for name in ("mv.inv_nu", "mv.gamma_nu", "mv.beta_nu"):
        assert "uncertainty of the cited q_c" in by[name]["warnings"], name
    assert "insensitive" not in by["mv.inv_nu"]["warnings"]
    # the slopes at x = -0.1 and +0.1: chords symmetric about q_c (the grid has 0.15 between
    # 0.1 and 0.2, which must not be taken as the far point)
    assert ("slopes at x = -0.1, 0.1 (chords to the grid points at x = -0.2, 0, 0.2)"
            in by["mv.inv_nu"]["warnings"])


def test_mv_runs_are_noise_sweeps():
    """Stationary noise sweeps of Oliveira's rule on periodic L x L lattices, ordered starts,
    >= 8 chains per point, the cited q_c itself a point of every size."""
    from validation.models import majorityvote as mv

    for scale in ("smoke", "small", "full"):
        runs = mv.runs(scale)
        assert runs
        for r in runs:
            a = list(r.args)
            assert a[a.index("--model") + 1] == "majority-vote"
            assert a[a.index("--init") + 1] == "up"
            assert a[a.index("--task") + 1] == "stationary"
            assert a[a.index("--observable") + 1] == "m"
            assert int(a[a.index("--replicas") + 1]) >= 8
            assert "--boundary" not in a or a[a.index("--boundary") + 1] == "periodic"
            sweep = a[a.index("--sweep") + 1]
            assert sweep.startswith("noise="), sweep
            assert not any("temperature" in x for x in a)
            q = [float(v) for v in sweep.split("=", 1)[1].split(",")]
            assert MV_QC[0] in q and np.all(np.diff(q) > 0) and 0 < min(q) and max(q) < 0.5
    stats = mv.MODEL.scales
    assert len(stats["small"].sizes) >= 4
    full = stats["full"].sizes
    assert len(full) >= 5 and max(full) / min(full) >= 8


def test_mv_model_choices_fixed():
    """R13: chi' is read at one fixed, exact grid point of the scaling variable; no collapse
    (all points exact); the campaign fit rule (R12) holds for every scale."""
    from validation.models import majorityvote as mv

    m = mv.MODEL
    assert m.control == "noise" and m.control_label == "q"
    assert m.tc == MV_QC[0] and m.tc_error == MV_QC[1]
    assert m.gamma_at is not None and m.gamma_at > 0 and m.gamma_at in m.grid
    assert m.exact_x is None and 0.0 in m.grid
    assert m.gamma_note
    for s in m.scales.values():
        assert s.fit_from == sorted(s.sizes)[1] and s.chains >= 8
    assert set(mv.__all__) == {"NAME", "runs", "analyse"}


def test_critical_point_shift():
    """The change of a fitted exponent when the critical point moves by +-delta: each size's
    value is carried along its own log-derivative, y_L exp(d_L delta L^(1/nu))."""
    from validation.models._critical import critical_point_shift

    L = np.array([16.0, 32.0, 64.0, 128.0])
    y = L ** 1.75
    e = 0.01 * y
    assert critical_point_shift(L, y, e, np.zeros(4), 1e-3, 1.0, 16) == (0.0, 0.0)
    d = np.full(4, 0.5)                 # d ln y / dx at x = 0
    up, down = critical_point_shift(L, y, e, d, 1e-3, 1.0, 32)
    # equal relative errors: the weighted power fit is the log-linear fit of the kept sizes
    keep = L >= 32
    for delta, got in ((1e-3, up), (-1e-3, down)):
        b = np.polyfit(np.log(L[keep]), np.log(y[keep]) + d[keep] * delta * L[keep], 1)[0]
        assert got == pytest.approx(b - 1.75, rel=1e-3, abs=1e-6)
    assert up > 0 > down and math.isfinite(up)


def test_log_slope_derivative():
    """d ln|dU/dx| / dx at the middle of 3 points from the chords on either side: exact for an
    exponential (both chords carry the same factor), any sign."""
    from validation.models._critical import log_slope_derivative

    x = np.array([-0.2, 0.0, 0.2])
    for c in (5.5, -3.0):
        assert log_slope_derivative(x, 0.6 - np.exp(c * x)) == pytest.approx(c, rel=1e-12)
        assert log_slope_derivative(x, 2.0 * np.exp(c * x)) == pytest.approx(c, rel=1e-12)
    assert log_slope_derivative(x, 0.6 - 0.4 * x) == pytest.approx(0.0, abs=1e-12)


def test_grid_neighbours_guard_the_ends():
    """A point needs `reach` grid points on each side; at an end it is refused with a clear
    message instead of wrapping around (grid[-1]) or raising IndexError."""
    from validation.models._critical import grid_neighbours

    grid = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2)
    assert grid_neighbours(grid, 0.0) == (-0.1, 0.1)
    assert grid_neighbours(grid, 0.0, reach=2) == (-0.2, 0.2)
    for x, reach in ((-0.3, 1), (0.2, 1), (-0.2, 2), (0.1, 2)):
        with pytest.raises(ValueError, match="grid point"):
            grid_neighbours(grid, x, reach)


def test_full_scale_chains_ruling_r15():
    """R15: 16 chains per point at full scale for ising, potts and mv (8 at small)."""
    from validation.models import ising, majorityvote, potts

    for model in (ising.MODEL, potts.MODEL, majorityvote.MODEL):
        assert model.scales["full"].chains == 16, model.name
        assert model.scales["small"].chains == 8, model.name


def test_mv_tau_model_covers_measured_maximum():
    """The tau model sets burn-in and interval: it must cover the largest tau(|m|) measured
    anywhere on the grid (tau peaks on the disordered side, not at q_c), in sweeps: small runs
    (two seeds) 328 at L = 32 and 1637 at L = 64, and the extra L = 128 runs."""
    from validation.models import majorityvote as mv
    from validation.models._critical import tau_of

    for L, tau in mv.TAU_MEASURED_MAX.items():
        assert tau_of(mv.MODEL, L) >= tau, (L, tau, tau_of(mv.MODEL, L))
    assert mv.TAU_MEASURED_MAX[32] >= 327 and mv.TAU_MEASURED_MAX[64] >= 1637
    assert 128 in mv.TAU_MEASURED_MAX
