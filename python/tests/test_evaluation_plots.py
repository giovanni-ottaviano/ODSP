import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from odspy import find_crossings, find_outputs, fit, fit_timeseries, search_collapse  # noqa: E402
from odspy.collapse import collapse  # noqa: E402
from odspy.io import Provenance  # noqa: E402
from odspy.plots import plot_collapse, plot_crossings, plot_fit, save  # noqa: E402
from test_collapse import NU, XC, scaling_curves  # noqa: E402
from test_crossings import curve, model  # noqa: E402

LABEL = "errors not trustworthy: correlated points (rerun with --save-trajectories true)"


@pytest.fixture(autouse=True)
def no_leaks():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    plt.close("all")
    assert not leaked, "figure leaked"


def texts(fig):
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts] + [ax.get_title()]
    return out


def joined(fig):
    return " ".join(" ".join(t.split()) for t in texts(fig))


def poor_fit(xrange=None):
    x = np.logspace(0, 2, 15)
    y = 2 * x ** 0.5 * (1 + 0.15 * np.sin(x))
    return fit(x, y, 0.01 * y, "power", xrange=xrange)


def test_fit_figure_shows_quality():
    r = poor_fit()
    assert r.poor
    fig = plot_fit(r, "t", "m")
    assert "POOR FIT" in texts(fig)
    assert f"{r.chi2 / r.dof:.3g}" in joined(fig) and "χ²/dof" in joined(fig)
    assert "errors scaled by sqrt(chi2/dof)" in joined(fig)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_good_fit_has_no_poor_flag():
    x = np.linspace(1, 10, 12)
    r = fit(x, 1 + 2 * x, np.full(12, 0.1), "line")
    fig = plot_fit(r, "x", "y")
    assert "POOR FIT" not in joined(fig)
    plt.close(fig)


def test_faded_points_kept():
    r = poor_fit(xrange=(3, 30))
    assert 0 < r.used.sum() < len(r.x)
    fig = plot_fit(r, "t", "m")
    ax = fig.axes[0]
    n = sum(len(c.lines[0].get_xdata()) for c in ax.containers)
    assert n == len(r.x)
    assert sorted(c.lines[0].get_alpha() for c in ax.containers) == [0.3, 1.0]
    plt.close(fig)


def test_untrustworthy_label_drawn(outputs):
    out = find_outputs(outputs["ens"])
    r = fit_timeseries(out.get("timeseries_ensemble"), "length", trajectories=None)
    fig = plot_fit(r, "t", "length", out.provenance)
    assert LABEL in texts(fig)
    assert r.chi2 is None and "χ²/dof =" not in joined(fig)
    assert "goodness of fit not assessed" not in joined(fig)
    if r.scaled:                                       # only as the fit's own caveat
        assert "not a goodness of fit" in joined(fig)
    assert "POOR FIT" not in joined(fig)
    save(fig, Path(outputs["ens"]).parent / "fig", "fit_ts")


def test_unweighted_and_undefined_errors_draw():
    x = np.linspace(1, 10, 10)
    r = fit(x, 1 + 2 * x + 0.1 * np.cos(5 * x), None, "line", unweighted=True)
    fig = plot_fit(r, "x", "y")
    assert "goodness of fit not assessed" in joined(fig)
    plt.close(fig)
    # out-of-range points with NaN/zero errors: no bar, no failure
    y = 1 + 2 * x
    e = np.full(10, 0.1)
    e[0], e[1] = np.nan, 0.0
    fig = plot_fit(fit(x, y, e, "line", xrange=(3, 10)), "x", "y")
    plt.close(fig)


def test_crossing_and_collapse_figures(tmp_path):
    x = np.linspace(2.15, 2.39, 13)
    curves = [curve(L, x, model(L, x)) for L in (8, 16)]
    fig = plot_crossings(curves, find_crossings(curves, n_boot=200), Provenance())
    assert len(fig.axes) == 2 and "no crossing in range" not in joined(fig)
    plt.close(fig)
    apart = [curve(8, x, 1.0 + 0 * x), curve(16, x, 2.0 + 0 * x)]
    fig = plot_crossings(apart, find_crossings(apart, n_boot=50))
    assert "no crossing in range" in joined(fig)
    save(fig, tmp_path, "cross")

    cs = scaling_curves()
    good = search_collapse(cs, XC + 0.01, NU * 1.1, 0.2)
    fig = plot_collapse(good, "(T − Tc) L^(1/ν)", "m L^(−a)")
    assert "POOR COLLAPSE" not in joined(fig) and "S = " in joined(fig)
    assert f"from {good.points_used} of {good.points_total} points" in joined(fig)
    plt.close(fig)
    bad = collapse(cs, XC, 2.5, 0.25)
    assert bad.poor
    fig = plot_collapse(bad, "x", "y")
    assert "POOR COLLAPSE" in texts(fig)
    plt.close(fig)


# --- fix round 1 ------------------------------------------------------------------------------

from odspy.fits import FitResult  # noqa: E402


def result(**kw):
    x = np.logspace(0, 2, 8)
    base = dict(model="power", params={"A": 1.0, "b": 0.5}, errors={"A": 0.01, "b": 0.01},
                chi2=12.0, dof=6, p_value=0.06, scaled=False, poor=False,
                method="weighted least squares", x=x, y=x ** 0.5, yerr=0.01 * x ** 0.5,
                used=np.ones(8, bool), xrange=(1.0, 100.0), warnings=[], notes=[])
    base.update(kw)
    return FitResult(**base)


def test_bootstrap_fit_named_hotelling():
    r = result(method="bootstrap over replicas", chi2=12.0, dof=6, p_value=0.06)
    fig = plot_fit(r, "t", "y")
    j = joined(fig)
    assert "T²/dof" in j and "Hotelling" in j and "χ²/dof" not in j
    assert "p = 0.06" in j
    plt.close(fig)


def test_weighted_fit_named_chi2():
    fig = plot_fit(result(), "t", "y")
    assert "χ²/dof" in joined(fig) and "T²/dof" not in joined(fig)
    plt.close(fig)


def segments(ax):
    segs = []
    for c in ax.containers:
        for lc in c.lines[2]:
            segs += [np.asarray(s) for s in lc.get_segments()]
    return segs


def test_undefined_errors_draw_no_bar():
    x = np.logspace(0, 2, 8)
    e = 0.01 * x ** 0.5
    e = e.copy()
    e[0], e[1], e[7] = np.nan, 0.0, -1.0          # out of the fit range (3, 30)
    r = result(yerr=e, used=(x >= 3) & (x <= 30), xrange=(3.0, 30.0))
    fig = plot_fit(r, "t", "y")
    for ax in fig.axes:
        n_segments = 0
        for seg in segments(ax):
            if seg.ndim != 2:
                continue
            n_segments += 1
            ys = seg[:, 1]
            assert np.isnan(ys).any() or ys[0] != ys[1], "zero-length bar drawn"
        assert n_segments > 0
    assert "error undefined for 3 points" in joined(fig)
    plt.close(fig)


def test_residual_labels():
    fig = plot_fit(result(), "t", "y")
    assert fig.axes[1].get_ylabel() == "(data − fit)/σ"
    plt.close(fig)
    x = np.linspace(1, 10, 10)
    r = fit(x, 1 + 2 * x + 0.1 * np.cos(5 * x), None, "line", unweighted=True)
    fig = plot_fit(r, "x", "y")
    assert fig.axes[1].get_ylabel() == "data − fit"
    plt.close(fig)


def test_poor_headline_has_threshold():
    fig = plot_fit(poor_fit(), "t", "m")
    assert "< 0.01" in joined(fig)
    plt.close(fig)


def test_untrustworthy_without_binary():
    r = result(chi2=None, p_value=None, scaled=True, poor=False, method="weighted, correlated points",
               warnings=[LABEL], notes=["errors scaled by sqrt(chi2/dof) = 7.7 of the independent-point "
                                        "fit (not a goodness of fit: points correlated)"])
    fig = plot_fit(r, "t", "y")
    j = joined(fig)
    assert LABEL in texts(fig)
    assert "χ²/dof" not in j and "T²/dof" not in j
    assert "errors scaled by sqrt(chi2/dof) = " not in j.replace(r.notes[0], "")
    assert "POOR FIT" not in j and "goodness of fit not assessed" not in j
    plt.close(fig)


def test_correlation_label_drawn_for_unweighted_fit():
    from odspy.trajectories import UNTRUSTWORTHY_G, mark_untrustworthy
    x = np.arange(1.0, 8.0)
    r = mark_untrustworthy(fit(x, np.exp(-x / 2), model="power", unweighted=True), UNTRUSTWORTHY_G)
    fig = plot_fit(r, "r", "G")
    t = texts(fig)
    plt.close(fig)
    assert UNTRUSTWORTHY_G in t
    assert LABEL not in t


def test_hotelling_scaling_drawn():
    from test_trajectories import ensemble, tables
    rng = np.random.default_rng(24)
    ts, tr = tables(3 + ensemble(rng, 60))
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    fig = plot_fit(r, "t", "y")
    j = joined(fig)
    plt.close(fig)
    assert "E[T²] = " in j and "errors scaled by sqrt(T²/E[T²])" in j
    assert "sqrt(chi2/dof)" not in j
