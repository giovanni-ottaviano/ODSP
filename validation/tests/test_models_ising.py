"""The 2D Ising model's checks (validation/models/ising.py), at smoke scale."""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.models import MODELS
from validation.theory import ISING_BINDER, ISING_EXPONENTS, ISING_TC

from .smoke import run_smoke

CHECKS = ["ising.tc", "ising.binder", "ising.inv_nu", "ising.gamma_nu", "ising.beta_nu",
          "ising.collapse_tc", "ising.collapse_nu", "ising.energy", "ising.magnetization"]


def test_ising_smoke(tmp_path, odsp_bin):
    from validation.models._critical import FIT_RULE

    assert "ising" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "ising", CHECKS)
    by = {r["check"]: r for r in rows}
    predicted = {r["check"]: float(r["predicted"]) for r in rows}
    assert predicted["ising.tc"] == pytest.approx(ISING_TC, abs=1e-12)
    assert predicted["ising.collapse_tc"] == pytest.approx(ISING_TC, abs=1e-12)
    assert predicted["ising.binder"] == pytest.approx(ISING_BINDER[0], abs=1e-12)
    assert predicted["ising.inv_nu"] == pytest.approx(1.0 / ISING_EXPONENTS["nu"])
    assert predicted["ising.collapse_nu"] == pytest.approx(ISING_EXPONENTS["nu"])
    assert predicted["ising.gamma_nu"] == pytest.approx(7.0 / 4.0)
    assert predicted["ising.beta_nu"] == pytest.approx(-1.0 / 8.0)
    assert predicted["ising.energy"] == 1.0 and predicted["ising.magnetization"] == 1.0
    binder = next(r for r in rows if r["check"] == "ising.binder")
    assert float(binder["predicted_error"]) == pytest.approx(ISING_BINDER[1])
    # The campaign rule (fixed before the full run): exponent fits and the collapse over all
    # sizes but the smallest (smoke: 8, 12, 16, 24 -> L >= 12), with the all-sizes fits and
    # collapse as a finite-size diagnostic computed from this run's data.
    for name in ("ising.inv_nu", "ising.gamma_nu", "ising.beta_nu"):
        assert "(fit L >= 12)" in by[name]["sizes"], by[name]["sizes"]
        assert FIT_RULE in by[name]["warnings"]
        assert "finite-size diagnostic: fit over all sizes (L >= 8)" in by[name]["warnings"]
    for name in ("ising.collapse_tc", "ising.collapse_nu"):
        w = by[name]["warnings"]
        assert "(collapse L >= 12)" in by[name]["sizes"], by[name]["sizes"]
        assert FIT_RULE in w
        assert "finite-size diagnostic: collapse over all sizes (L >= 8): S = " in w
        assert "x_c = " in w and "nu = " in w
        assert "S = 2.7" not in w and "S = 1.05" not in w   # no stale small-run numbers
    assert ("peak_vs_size refused (flat 2D Ising chi' maximum); chi' at fixed (T - T_c) L = 2 "
            "used instead, same target 7/4") in by["ising.gamma_nu"]["warnings"]
    assert "2.1 < T < 2.5 left out" in by["ising.energy"]["warnings"]


def test_scaled_grid_holds_tc_and_scales_with_size():
    """Each size's temperatures are T_c + x L^(-1/nu) on one grid of x, with T_c itself a
    point (the estimators read |m| there and take the slope through the 3 points around it)."""
    from validation.models._critical import scaled_grid

    xs = (-2.0, -1.0, 0.0, 1.0, 2.0)
    for L in (8, 64):
        T = scaled_grid(ISING_TC, 1.0, L, xs)
        assert len(T) == len(xs) and T[2] == ISING_TC
        assert np.allclose((np.array(T) - ISING_TC) * L, xs)
    T = scaled_grid(1.0, 5.0 / 6.0, 32, xs)
    assert np.allclose((np.array(T) - 1.0) * 32 ** 1.2, xs)


def test_crossing_value_error():
    """The value of U at a crossing gets a parametric-bootstrap error that matches the
    spread of the curves at the crossing (two straight lines with known errors)."""
    from odspy import SizeCurve
    from validation.models._critical import crossing_with_value_error

    x = np.linspace(-1.0, 1.0, 5)
    a = SizeCurve(16, "L", x, 0.6 - 0.1 * x, np.full(5, 0.01), "temperature", "binder", "a")
    b = SizeCurve(32, "L", x, 0.6 - 0.2 * x, np.full(5, 0.01), "temperature", "binder", "b")
    c, y_err = crossing_with_value_error(a, b, target=0.0, n_boot=2000, seed=1)
    assert c.found and abs(c.x) < 1e-12 and c.y == pytest.approx(0.6)
    # At a grid point each curve is read directly (error 0.01); the crossing of the two
    # lines sits where U_a = U_b, a weighted mix of both: error between 0.007 and 0.02.
    assert 0.007 < y_err < 0.02 and math.isfinite(c.err)


def test_size_grids_have_no_multipoint_coincidences():
    """Every size's grid holds T_c and the other exact points at the same scaled position
    (x = (T - T_c) L^(1/nu)), and no increasing affine map -- i.e. no (x_c, nu) of a collapse
    -- takes more than 2 of one size's collapse points onto points of another (odspy's
    collapse settles on such coincidences; validation/models/_critical.py)."""
    from itertools import combinations

    from validation.models import ising, potts
    from validation.models._critical import OFFSET, size_grid

    for model in (ising.MODEL, potts.MODEL):
        exact = model.grid if model.exact_x is None else model.exact_x
        for stats in model.scales.values():
            grids = []
            for i, L in enumerate(stats.sizes):
                T = np.array(size_grid(model, L, i))
                assert np.all(np.diff(T) > 0)
                x = (T - model.tc) * L ** (1.0 / model.nu)
                assert T[list(model.grid).index(0.0)] == model.tc
                for k, g in enumerate(model.grid):
                    if g in exact:
                        assert x[k] == pytest.approx(g, abs=1e-9)
                    else:
                        assert 1e-6 < abs(x[k] - g) <= OFFSET + 1e-9
                grids.append(np.array([t for t, g in zip(T, model.grid) if g not in exact]))
            if model.exact_x is None:
                continue
            for A, B in combinations(grids, 2):
                tol = 1e-9 * (B.max() - B.min())
                worst = 0
                for i, j in combinations(range(len(A)), 2):
                    for k, l in combinations(range(len(B)), 2):
                        slope = (B[l] - B[k]) / (A[j] - A[i])
                        if abs(slope - 1.0) < 1e-12:
                            continue
                        mapped = B[k] + slope * (A - A[i])
                        hits = sum(np.min(np.abs(B - t)) < tol for t in mapped)
                        worst = max(worst, hits)
                assert worst <= 2, (model.name, stats.sizes)


def test_ising_module_exports_only_the_model_interface():
    """The FSS helpers live in validation/models/_critical.py; ising.py does not re-export
    them for the tests."""
    from validation.models import ising

    assert set(ising.__all__) == {"NAME", "runs", "analyse"}


def test_campaign_fit_rule():
    """One rule for every model with finite-size-scaling fits, fixed before the full run:
    every exponent fit and the collapse use all sizes but the smallest. No per-model or
    per-scale choice is left to tune: Stats has no fit_from / collapse_from fields and the
    models no notes justifying a choice."""
    import dataclasses

    from validation.models import ising, potts
    from validation.models._critical import Model, Stats

    names = {f.name for f in dataclasses.fields(Stats)}
    assert not names & {"fit_from", "collapse_from"}
    assert not {f.name for f in dataclasses.fields(Model)} & {"fit_note", "collapse_note"}
    for model in (ising.MODEL, potts.MODEL):
        for stats in model.scales.values():
            assert stats.fit_from == sorted(stats.sizes)[1]


def test_ising_gamma_point_is_exact():
    """chi' is read at (T - T_c) L = 2 exactly in every size (not at an offset point)."""
    from validation.models import ising

    assert ising.MODEL.gamma_at in ising.MODEL.exact_x


def test_burn_in_caveat():
    """A data-driven check of the tau model: burn-in shorter than 20 tau_abs (tau_abs in
    samples times the interval, from the table) at any point is named."""
    from pathlib import Path

    from odspy.io import Provenance, Table

    from validation.models._critical import burn_in_caveat

    def table(tau_abs):
        cols = {"temperature": np.array([2.2, 2.3]), "tau_abs": np.array(tau_abs, float)}
        prov = Provenance(config={"burn-in": "1000", "interval": "5", "samples": "100"})
        return Table(Path("x_sweep.csv"), "sweep", prov, cols, list(cols))

    assert burn_in_caveat(table([4.0, 9.0])) is None          # 1000 / 45 = 22 >= 20
    text = burn_in_caveat(table([4.0, 11.0]))                 # 1000 / 55 = 18 < 20
    assert text is not None and "x_sweep.csv" in text and "temperature=2.3" in text
    assert "18.2" in text and "20" in text


def test_all_sizes_collapse_text_withholds_poor_errors():
    """The all-sizes collapse diagnostic prints errors only when odspy gives them: a poor
    collapse (errors withheld, nan) says so instead of printing '± nan'."""
    from odspy.collapse import CollapseResult

    from validation.models._critical import all_sizes_collapse_text

    def result(poor, errors):
        return CollapseResult(xc=2.26801, nu=0.9805, a=0.0, errors=errors, quality=3.18,
                              poor=poor, searched=True, fixed=["a"], method="test",
                              points_used=18, points_total=20)

    text = all_sizes_collapse_text(result(True, {"xc": math.nan, "nu": math.nan}), 8)
    assert "nan" not in text
    assert text == ("finite-size diagnostic: collapse over all sizes (L >= 8): S = 3.18 over 18 "
                    "of 20 points, x_c = 2.26801, nu = 0.9805; errors withheld (flagged poor "
                    "by odspy)")
    text = all_sizes_collapse_text(result(False, {"xc": 0.0015, "nu": 0.028}), 8)
    assert "x_c = 2.26801 ± 0.0015, nu = 0.9805 ± 0.028" in text and "withheld" not in text
