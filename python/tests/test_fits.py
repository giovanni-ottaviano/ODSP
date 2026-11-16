import math

import numpy as np
import pytest

from odspy.errors import InputError
from odspy.fits import FitResult, fit, plain_loglog_slope

X20 = np.logspace(1, 4, 20)


def legacy_slope(xs, ys):
    """Direct transcription of benchmarks/analyse.py's slope()."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def test_power_recovers_exponent():
    rng = np.random.default_rng(11)
    covered, pvals = 0, []
    for _ in range(200):
        y = 3 * X20 ** 0.5 * (1 + rng.normal(0, 0.02, X20.size))
        r = fit(X20, y, 0.02 * y, model="power")
        assert isinstance(r, FitResult) and r.method == "weighted least squares"
        covered += abs(r.params["b"] - 0.5) < 2 * r.errors["b"]
        pvals.append(r.p_value)
    assert covered / 200 >= 0.90
    assert 0.3 < np.mean(pvals) < 0.7


def test_wrong_model_flagged():
    rng = np.random.default_rng(12)
    y0 = 2 + X20 ** 0.3
    y = y0 * (1 + rng.normal(0, 0.005, X20.size))
    r = fit(X20, y, 0.005 * y0, model="power")
    assert r.poor is True
    assert r.p_value < 0.01
    assert "poor fit" in r.summary()


def test_error_scaling():
    rng = np.random.default_rng(13)
    x = np.logspace(1, 4, 100)
    y = 3 * x ** 0.5 * (1 + rng.normal(0, 0.06, x.size))
    under = fit(x, y, 0.02 * y)              # errors underestimated by 3x
    assert under.scaled is True
    assert under.chi2 / under.dof > 4
    # The unscaled error with the true error bars is 3x the unscaled one with the
    # underestimated bars; the scaled error should land close to it.
    true = fit(x, y, 0.06 * y)
    true_unscaled = true.errors["b"] / (math.sqrt(true.chi2 / true.dof) if true.scaled else 1)
    unscaled_under = true_unscaled / 3
    assert under.errors["b"] == pytest.approx(3 * unscaled_under, rel=0.3)
    assert "scaled" in under.summary()
    assert "sqrt(chi2/dof)" in under.summary()


def test_no_scaling_when_chi2_small():
    rng = np.random.default_rng(14)
    y = 3 * X20 ** 0.5 * (1 + rng.normal(0, 0.01, X20.size))
    r = fit(X20, y, 0.05 * y)                # errors overestimated: chi2/dof < 1
    assert r.chi2 / r.dof < 1
    assert r.scaled is False
    assert "scaled" not in r.summary()


def test_refuses_missing_errors():
    y = 3 * X20 ** 0.5
    with pytest.raises(InputError, match="unweighted"):
        fit(X20, y)
    bad = 0.01 * y
    bad[3] = 0.0
    with pytest.raises(InputError, match="unweighted"):
        fit(X20, y, bad)
    bad[3] = np.nan
    with pytest.raises(InputError, match="unweighted"):
        fit(X20, y, bad)
    r = fit(X20, y, unweighted=True)
    assert r.method == "unweighted (no error bars)"
    assert r.chi2 is None and r.p_value is None and r.poor is False
    assert r.params["b"] == pytest.approx(0.5, abs=1e-8)
    assert "unweighted" in r.summary()


def test_bad_error_outside_range_is_fine():
    y = 3 * X20 ** 0.5
    e = 0.01 * y
    e[0] = 0.0
    r = fit(X20, y, e, xrange=(20, 1e4))
    assert not r.used[0]


def test_nan_points_counted():
    rng = np.random.default_rng(15)
    y = 3 * X20 ** 0.5 * (1 + rng.normal(0, 0.02, X20.size))
    e = 0.02 * y
    y[[4, 9]] = np.nan
    e[9] = np.nan                            # a NaN error at a NaN point is not an error
    r = fit(X20, y, e)
    assert "2 points with NaN values excluded" in r.warnings
    assert r.used.sum() == 18 and not r.used[4] and not r.used[9]
    assert r.x.size == 20
    assert "2 points with NaN values excluded" in r.summary()


def test_too_few_points():
    y = 3 * X20 ** 0.5
    with pytest.raises(InputError, match="need at least 3 points in range, have 2"):
        fit(X20[:2], y[:2], 0.01 * y[:2])
    with pytest.raises(InputError, match="need at least 4 points in range, have 1"):
        fit(X20, y, 0.01 * y, model="power+c", xrange=(9, 11))
    with pytest.raises(InputError, match="need at least 3 points in range, have 1"):
        fit(np.array([1.0]), np.array([2.0]), np.array([0.1]), model="line")
    with pytest.raises(InputError, match="need at least 3 points in range, have 0"):
        fit(np.array([]), np.array([]), np.array([]))


def test_bad_arguments():
    y = 3 * X20 ** 0.5
    with pytest.raises(InputError, match="model"):
        fit(X20, y, 0.01 * y, model="exp")
    with pytest.raises(InputError, match="length"):
        fit(X20, y[:-1], 0.01 * y)
    with pytest.raises(InputError, match="xrange"):
        fit(X20, y, 0.01 * y, xrange=(100, 10))


def test_nonpositive_x_excluded_for_power():
    x = np.concatenate([[0.0], X20])
    y = np.concatenate([[0.0], 3 * X20 ** 0.5])
    r = fit(x, y, unweighted=True)
    assert not r.used[0]
    assert "1 point with x <= 0 excluded (power needs x > 0)" in r.warnings


def test_range_marks_points():
    rng = np.random.default_rng(16)
    y = 3 * X20 ** 0.5 * (1 + rng.normal(0, 0.02, X20.size))
    r = fit(X20, y, 0.02 * y, xrange=(100, 2000))
    assert r.x.size == 20 and r.y.size == 20 and r.yerr.size == 20
    inside = (X20 >= 100) & (X20 <= 2000)
    assert np.array_equal(r.used, inside)
    assert r.xrange == (100, 2000)
    assert r.dof == inside.sum() - 2
    assert r.residuals().shape == (20,)


def test_other_models():
    x = np.linspace(1, 50, 30)
    e = np.full(x.size, 0.01)
    rng = np.random.default_rng(17)
    r = fit(x, 1.5 + 0.7 * np.log(x) + rng.normal(0, 0.01, x.size), e, model="log")
    assert r.params["a"] == pytest.approx(1.5, abs=4 * r.errors["a"])
    assert r.params["b"] == pytest.approx(0.7, abs=4 * r.errors["b"])
    r = fit(x, 1.5 + 0.7 * x + rng.normal(0, 0.01, x.size), e, model="line")
    assert r.params["b"] == pytest.approx(0.7, abs=4 * r.errors["b"])
    assert r.predict(np.array([2.0])) == pytest.approx(r.params["a"] + 2 * r.params["b"])
    y0 = 2 + X20 ** 0.3
    y = y0 * (1 + rng.normal(0, 0.005, X20.size))
    r = fit(X20, y, 0.005 * y0, model="power+c")
    assert r.params["b"] == pytest.approx(0.3, abs=4 * r.errors["b"])
    assert r.params["c"] == pytest.approx(2, abs=4 * r.errors["c"])
    assert r.poor is False


def test_plain_slope_matches_legacy():
    rng = np.random.default_rng(18)
    for _ in range(20):
        x = np.sort(rng.uniform(0.1, 1000, 15))
        y = rng.uniform(0.01, 50, 15)
        assert plain_loglog_slope(x, y) == pytest.approx(legacy_slope(x, y), abs=1e-12)
        sel = (x >= 5) & (x <= 500)
        assert plain_loglog_slope(x, y, xrange=(5, 500)) == pytest.approx(
            legacy_slope(x[sel], y[sel]), abs=1e-12)
    with pytest.raises(InputError):
        plain_loglog_slope([1.0], [2.0])


def test_nan_x_counted_with_range():
    rng = np.random.default_rng(19)
    x = X20.copy()
    y = 3 * x ** 0.5 * (1 + rng.normal(0, 0.02, x.size))
    e = 0.02 * y
    x[5] = np.nan
    for xr in (None, (10, 1e4), (100, 2000)):
        r = fit(x, y, e, xrange=xr)
        assert "1 point with NaN values excluded" in r.warnings
        assert r.x.size == 20 and np.isnan(r.x[5]) and not r.used[5]
    # a NaN y outside the range is not in range: not counted
    y2 = y.copy()
    y2[0] = np.nan
    r = fit(X20, y2, e, xrange=(100, 2000))
    assert not any("NaN" in w for w in r.warnings)


@pytest.mark.parametrize("kw", [dict(p_threshold="a"), dict(p_threshold=None), dict(p_threshold=1.5),
                                dict(model=["power"]), dict(model=3), dict(unweighted="no")])
def test_fit_rejects_bad_argument_types(kw):
    x = np.linspace(1, 10, 10)
    with pytest.raises(InputError):
        fit(x, x ** 0.5, np.full(10, 0.01), **kw)


@pytest.mark.parametrize("seed", ["a", 1.5, -1, True, None])
def test_seeded_calls_reject_bad_seeds(seed):
    from odspy import find_crossings, search_collapse
    from synthetic import crossing_model, curve, scaling_curves
    x = np.linspace(2.2, 2.34, 8)
    with pytest.raises(InputError, match="seed must be a non-negative integer"):
        find_crossings([curve(L, x, crossing_model(L, x)) for L in (8, 16)], n_boot=10, seed=seed)
    with pytest.raises(InputError, match="seed must be a non-negative integer"):
        search_collapse(scaling_curves(), 2.27, 1.0, 0.25, n_boot=10, seed=seed, starts=1)
