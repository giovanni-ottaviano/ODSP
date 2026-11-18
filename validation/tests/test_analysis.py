"""Tests of the shared helpers the model analyses use (validation/analysis.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from odspy import fit

from validation.analysis import (exponent_check, plot_exponent, plot_ratio, ratio_check,
                                 replica_ratio, save_figure, weighted_ratio)


def test_weighted_ratio():
    r = weighted_ratio([2.0, 4.4], [0.2, 0.44], [2.0, 4.0])
    assert r.ratios == pytest.approx([1.0, 1.1])
    assert r.errors == pytest.approx([0.1, 0.11])
    w = np.array([1 / 0.1 ** 2, 1 / 0.11 ** 2])
    assert r.value == pytest.approx((w * [1.0, 1.1]).sum() / w.sum())
    assert r.error == pytest.approx(1 / math.sqrt(w.sum()))
    assert r.dof == 1 and r.chi2 == pytest.approx(((np.array([1.0, 1.1]) - r.value) ** 2 * w).sum())
    # a theory error adds in quadrature to the ratio's error: (sim / th^2) * th_err
    r = weighted_ratio([3.0], [0.3], [2.0], [0.2])
    assert r.value == pytest.approx(1.5) and r.error == pytest.approx(math.hypot(0.15, 0.15))
    assert r.dof == 0 and r.p_value is None
    with pytest.raises(ValueError):
        weighted_ratio([1.0, 2.0], [0.1], [1.0, 2.0])
    with pytest.raises(ValueError):
        weighted_ratio([1.0], [0.0], [1.0])


def test_ratio_check():
    good = weighted_ratio([1.0, 1.02, 0.99], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0])
    c = ratio_check(good, model="m", check="m.r", quantity="T_sim / T_exact", source="voter_complete",
                    sizes="N=1,2,3", statistics="10 replicas", warnings=["given"], figure="m.r")
    assert (c.model, c.check, c.predicted, c.predicted_error) == ("m", "m.r", 1.0, 0.0)
    assert c.measured == good.value and c.error == good.error and c.figure == "m.r"
    assert c.warnings == ["given"]
    bad = weighted_ratio([1.0, 1.5, 0.5], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0])
    c = ratio_check(bad, model="m", check="m.r", quantity="q", source="voter_complete", sizes="",
                    statistics="")
    assert len(c.warnings) == 1 and "inconsistent" in c.warnings[0] and "chi2/dof" in c.warnings[0]


def test_replica_ratio():
    rng = np.random.default_rng(3)
    R, K = 400, 6
    exact = np.geomspace(0.5, 0.1, K)
    common = rng.normal(0, 0.05, (R, 1))                      # correlated across times
    V = exact * (1 + common + rng.normal(0, 0.05, (R, K)))
    r = replica_ratio(V, exact)
    X = V / exact
    w = 1 / X.var(axis=0, ddof=1)
    z = X @ w / w.sum()                                       # one weighted ratio per replica
    assert r.value == pytest.approx(z.mean()) and r.error == pytest.approx(z.std(ddof=1) / math.sqrt(R))
    assert r.ratios == pytest.approx(X.mean(axis=0))
    assert r.errors == pytest.approx(X.std(axis=0, ddof=1) / math.sqrt(R))
    assert r.dof == K - 1 and r.p_value is not None and r.p_value > 0.001
    assert r.statistic == "T^2/dof (Hotelling)" and not r.warnings
    # the correlated error is larger than the independent-point one would be
    assert r.error > 1 / math.sqrt(np.sum(1 / r.errors ** 2))
    # one time off by 10% is inconsistent with a common ratio, and ratio_check says so
    bad = replica_ratio(V * np.where(np.arange(K) == 2, 1.1, 1.0), exact)
    assert bad.p_value < 1e-6
    c = ratio_check(bad, model="m", check="m.r", quantity="q", source="voter_interface_2d",
                    sizes="", statistics="", report_consistency=True)
    assert len(c.warnings) == 2 and "T^2/dof (Hotelling)" in c.warnings[0]
    assert "inconsistent" in c.warnings[1]
    good = ratio_check(r, model="m", check="m.r", quantity="q", source="voter_interface_2d",
                       sizes="", statistics="", report_consistency=True)
    assert len(good.warnings) == 1 and good.warnings[0].startswith("consistency over 6 points")
    # too few replicas for the covariance: no consistency test, and a warning saying so
    few = replica_ratio(V[:10], exact)
    assert few.p_value is None and len(few.warnings) == 1 and "R >= 2K" in few.warnings[0]
    with pytest.raises(ValueError):
        replica_ratio(V, exact[:-1])


def test_exponent_check():
    x = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    y = 3.0 * x ** 1.0
    r = fit(x, y * (1 + 0.01 * np.array([1, -1, 1, -1, 1])), 0.02 * y)
    c = exponent_check(r, 1.0, model="m", check="m.b", quantity="b in T ~ N^b",
                       source="voter_lattice", sizes="N=10..160", statistics="s",
                       warnings=["given"], figure="m.b")
    assert c.measured == r.params["b"] and c.error == r.errors["b"] and c.predicted == 1.0
    assert c.warnings[0] == "given"
    assert any(w.startswith("fit: chi2/dof") and "p = " in w for w in c.warnings)
    # a poor fit and the fit's own warnings are carried along
    yb = y * (1 + 0.3 * np.array([1, -1, 1, -1, 1]))
    rb = fit(np.append(x, np.nan), np.append(yb, 1.0), np.append(0.01 * y, 1.0))
    cb = exponent_check(rb, 1.0, model="m", check="m.b", quantity="q", source="voter_lattice",
                        sizes="", statistics="")
    assert any("poor fit" in w for w in cb.warnings)
    assert any("NaN" in w for w in cb.warnings)
    assert any("errors scaled" in w for w in cb.warnings)


def test_figures(tmp_path):
    x = np.array([10.0, 20.0, 40.0, 80.0])
    y = 2.0 * x
    r = fit(x, y * (1 + 0.01 * np.array([1, -1, 1, -1])), 0.02 * y)
    name = save_figure(plot_exponent(r, 1.0, "N", "T", title="t"), tmp_path / "figs", "m.b")
    assert name == "m.b"
    assert (tmp_path / "figs" / "m.b.pdf").exists() and (tmp_path / "figs" / "m.b.png").exists()
    ratio = weighted_ratio(y, 0.02 * y, y)
    fig = plot_ratio(x, ratio, "N", title="t", absolute=(y, 0.02 * y, x, y, "T"))
    assert save_figure(fig, tmp_path / "figs", "m.r") == "m.r"
    assert (tmp_path / "figs" / "m.r.png").exists()
    # power+c fits are drawn too (linear axes)
    rc = fit(x, 0.5 * x + 3 + 0.01 * np.array([1, -1, 1, -1]), np.full(4, 0.02), model="power+c")
    assert save_figure(plot_exponent(rc, 1.0, "ln t", "1/rho"), tmp_path / "figs", "m.c") == "m.c"
