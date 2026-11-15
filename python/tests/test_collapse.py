import math
import re
import sys

import numpy as np
import pytest

from odspy import CollapseResult, InputError, SizeCurve, collapse, search_collapse

XC, NU, A = 2.27, 1.0, 0.25
SIGMA = 0.01  # noise on the scaled ordinate Y = y L^(-a)
SIZES = (8, 16, 32)


def curve(size, x, y, yerr):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.full(len(x), yerr) if np.isscalar(yerr) else np.asarray(yerr, float)
    return SizeCurve(size, "L", x, y, yerr, "temperature", "m", "test")


def scaling_curves(seed=3, sizes=SIZES):
    """y = L^a tanh((x - xc) L^(1/nu)) with known Gaussian noise of sd SIGMA L^a."""
    rng = np.random.default_rng(seed)
    x = XC + np.linspace(-0.08, 0.08, 33)
    out = []
    for L in sizes:
        f = np.tanh((x - XC) * L ** (1 / NU)) + rng.normal(0, SIGMA, len(x))
        out.append(curve(L, x, L ** A * f, L ** A * SIGMA))
    return out


def reference_quality(scaled):
    """Houdayer-Hartmann S by explicit loops: for every point, a weighted straight-line
    fit through the bracketing points of each other size."""
    terms = []
    for i, (_, X, Y, E) in enumerate(scaled):
        for j in range(len(X)):
            px, py, pe = [], [], []
            for k, (_, Xo, Yo, Eo) in enumerate(scaled):
                if k == i or not Xo[0] <= X[j] <= Xo[-1]:
                    continue
                m = min(max(int(np.searchsorted(Xo, X[j], side="right")) - 1, 0), len(Xo) - 2)
                px += [Xo[m], Xo[m + 1]]
                py += [Yo[m], Yo[m + 1]]
                pe += [Eo[m], Eo[m + 1]]
            if not px:
                continue
            coef, cov = np.polyfit(px, py, 1, w=1 / np.array(pe), cov="unscaled")
            v = np.array([X[j], 1.0])
            master, var = np.polyval(coef, X[j]), v @ cov @ v
            terms.append((Y[j] - master) ** 2 / (E[j] ** 2 + var))
    return float(np.mean(terms)), len(terms)


def test_exact_collapse_quality():
    curves = scaling_curves()
    good = collapse(curves, XC, NU, A)
    print(f"S(true) = {good.quality:.4f}")
    assert isinstance(good, CollapseResult)
    assert 0.5 < good.quality < 2 and not good.poor
    assert (good.xc, good.nu, good.a) == (XC, NU, A)
    assert not good.searched and good.errors == {} and good.fixed == []
    assert "Houdayer" in good.method
    bad = collapse(curves, XC, 1.5, A)
    print(f"S(nu=1.5) = {bad.quality:.4f}")
    assert bad.quality > 3 and bad.poor


def test_scaled_coordinates():
    curves = scaling_curves()
    res = collapse(curves, XC, 0.8, 0.3)
    assert [s[0] for s in res.scaled] == list(SIZES)
    for c, (size, X, Y, E) in zip(curves, res.scaled):
        np.testing.assert_allclose(X, (c.x - XC) * size ** (1 / 0.8))
        np.testing.assert_allclose(Y, c.y * size ** -0.3)
        np.testing.assert_allclose(E, c.yerr * size ** -0.3)


def test_quality_hand_computed():
    # two sizes, X = x L: A at X = 0, 1, 2 and B at X = 0, 1, 2; only X = 1 disagrees
    a = curve(1, [0, 1, 2], [0, 1, 2], 0.1)
    b = curve(2, [0, 0.5, 1], [0, 1.2, 2], 0.1)
    res = collapse([a, b], 0.0, 1.0)
    # each X = 1 point: (0.2)^2 / (0.1^2 + 0.1^2) = 2; the other four contribute 0
    assert res.quality == pytest.approx(4 / 6)
    assert res.warnings == []


def test_quality_matches_reference_loop():
    curves = scaling_curves(seed=8, sizes=(8, 12, 16, 24))
    for xc, nu, a in ((XC, NU, A), (2.26, 1.2, 0.1), (2.29, 0.7, 0.4)):
        res = collapse(curves, xc, nu, a)
        ref, n = reference_quality(res.scaled)
        assert res.quality == pytest.approx(ref, rel=1e-9)


def test_points_without_overlap_are_counted():
    a = curve(1, [0, 1, 2, 3], [0, 1, 2, 3], 0.1)
    b = curve(1.5, [2, 3, 4], [2, 3, 4], 0.1)  # X = 3, 4.5, 6: overlaps A only at X = 3
    res = collapse([a, b], 0.0, 1.0)
    ref, n = reference_quality(res.scaled)
    assert n == 2 and res.quality == pytest.approx(ref)
    assert any("5 of 7 points" in w and "not in S" in w for w in res.warnings)


def test_no_overlap_is_poor_not_good():
    a = curve(1, [0, 1], [0, 1], 0.1)
    b = curve(2, [5, 6], [5, 6], 0.1)
    res = collapse([a, b], 0.0, 1.0)
    assert math.isnan(res.quality) and res.poor
    assert any("no point" in w for w in res.warnings)


def test_nonfinite_and_zero_error_points_excluded_and_counted():
    curves = scaling_curves()
    c = curves[1]
    c.y[3] = np.nan
    c.yerr[5] = np.nan
    c.yerr[7] = 0.0
    res = collapse(curves, XC, NU, A)
    assert len(res.scaled[1][1]) == len(c.x) - 3
    assert any("size 16: 2 non-finite points excluded" in w for w in res.warnings)
    assert any("size 16: 1 points with zero yerr excluded" in w for w in res.warnings)


@pytest.mark.parametrize("kwargs, match", [
    (dict(nu=0.0), "nu"),
    (dict(nu=-1.0), "nu"),
    (dict(xc=math.nan), "xc"),
    (dict(a=math.inf), "a"),
])
def test_bad_parameters(kwargs, match):
    args = dict(xc=XC, nu=NU, a=A) | kwargs
    with pytest.raises(InputError, match=match):
        collapse(scaling_curves(), **args)


def test_bad_curves():
    curves = scaling_curves()
    with pytest.raises(InputError, match="at least two"):
        collapse(curves[:1], XC, NU)
    with pytest.raises(InputError, match="yerr"):
        collapse([curves[0], curve(16, curves[1].x, curves[1].y, None)], XC, NU)
    with pytest.raises(InputError, match="duplicate"):
        collapse([curves[0], curves[0]], XC, NU)
    neg = curve(16, curves[1].x, curves[1].y, -0.1)
    with pytest.raises(InputError, match="negative"):
        collapse([curves[0], neg], XC, NU)
    with pytest.raises(InputError, match="size"):
        collapse([curves[0], curve(0, curves[1].x, curves[1].y, 0.1)], XC, NU)
    with pytest.raises(InputError, match="fewer than 2"):
        collapse([curves[0], curve(16, [2.27], [0.0], 0.1)], XC, NU)
    other = SizeCurve(16, "L", curves[1].x, curves[1].y, curves[1].yerr, "temperature",
                      "binder", "test")
    with pytest.raises(InputError, match="different"):
        collapse([curves[0], other], XC, NU)


def test_search_recovers_parameters():
    curves = scaling_curves()
    res = search_collapse(curves, 2.25, 0.8, 0.2)
    print(f"xc={res.xc:.5f}+-{res.errors['xc']:.5f} nu={res.nu:.4f}+-{res.errors['nu']:.4f} "
          f"a={res.a:.4f}+-{res.errors['a']:.4f} S={res.quality:.3f}")
    assert res.searched and res.fixed == []
    assert set(res.errors) == {"xc", "nu", "a"}
    for name, true in (("xc", XC), ("nu", NU), ("a", A)):
        err = res.errors[name]
        assert 0 < err < math.inf
        assert abs(getattr(res, name) - true) < 3 * err
    assert not res.poor
    assert "Nelder-Mead" in res.method
    # this family is commensurate (XC on the x grid, L = 8:16:32, nu = 1): a few bootstrap
    # draws end pinned on coincident scaled points at the grid and count as failed
    failed = int(re.search(r"bootstrap \(n=200, (\d+) failed\)", res.method).group(1))
    assert failed <= 0.2 * 200
    assert not any("optimum held by coincident" in w for w in res.warnings)
    assert res.quality == pytest.approx(collapse(curves, res.xc, res.nu, res.a).quality)


def test_fixed_a():
    curves = scaling_curves()
    res = search_collapse(curves, 2.25, 0.8, 0.2, fit_a=False, n_boot=50)
    assert res.a == 0.2 and res.fixed == ["a"]
    assert set(res.errors) == {"xc", "nu"}
    assert "a fixed" in res.method


def test_search_is_reproducible():
    curves = scaling_curves()
    r1 = search_collapse(curves, 2.25, 0.8, 0.2, n_boot=20, seed=5)
    r2 = search_collapse(curves, 2.25, 0.8, 0.2, n_boot=20, seed=5)
    assert (r1.xc, r1.nu, r1.a, r1.errors) == (r2.xc, r2.nu, r2.a, r2.errors)


@pytest.mark.parametrize("kwargs, match", [
    (dict(n_boot=1), "n_boot"),
    (dict(n_boot=2.5), "n_boot"),
    (dict(nu0=0.0), "nu"),
    (dict(xc0=math.nan), "xc"),
])
def test_search_bad_arguments(kwargs, match):
    args = dict(xc0=XC, nu0=NU, a0=A) | kwargs
    with pytest.raises(InputError, match=match):
        search_collapse(scaling_curves(), **args)


def test_search_start_without_overlap():
    a = curve(1, [0, 1], [0, 1], 0.1)
    b = curve(2, [5, 6], [5, 6], 0.1)
    with pytest.raises(InputError, match="overlap"):
        search_collapse([a, b], 0.0, 1.0)


def test_search_does_not_stall_on_a_kink():
    # with a single Nelder-Mead run this data set stalls at nu = 0.90, S = 2.29,
    # worse than S at the true parameters (1.77)
    curves = scaling_curves(seed=101)
    res = search_collapse(curves, 2.25, 0.8, 0.2, n_boot=2)
    truth = collapse(curves, XC, NU, A).quality
    print(f"S(search) = {res.quality:.4f}, S(true) = {truth:.4f}, nu = {res.nu:.4f}")
    assert res.quality <= truth


def _flaky(monkeypatch, fail):
    """Make the minimiser report failure on the calls (1 = main fit) for which fail(n)."""
    mod = sys.modules["odspy.collapse"]  # odspy.collapse is the function
    real = mod._minimise
    calls = {"n": 0}

    def flaky(*args):
        r = real(*args)
        calls["n"] += 1
        if fail(calls["n"]):
            r.success = False
            r.message = "Maximum number of function evaluations has been exceeded."
        return r

    monkeypatch.setattr(mod, "_minimise", flaky)


def _no_pins(monkeypatch):
    """Isolate the forced failures from draws pinned on coincident scaled points."""
    monkeypatch.setattr(sys.modules["odspy.collapse"], "_pinned", lambda *a: 0)


def test_few_failed_minimisations_are_counted(monkeypatch):
    _no_pins(monkeypatch)
    _flaky(monkeypatch, lambda n: n == 5)  # 1 of 10 draws (call 1 is the main fit)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=10)
    assert not res.poor
    assert any("1 of 10 bootstrap minimisations did not converge" in w for w in res.warnings)
    assert "bootstrap (n=10, 1 failed)" in res.method
    assert all(math.isfinite(v) and v > 0 for v in res.errors.values())


def test_many_failed_minimisations_withhold_errors(monkeypatch):
    _no_pins(monkeypatch)
    _flaky(monkeypatch, lambda n: n % 3 == 0)  # 3 of 9 draws: above 20%
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=9)
    assert set(res.errors) == {"xc", "nu", "a"}
    assert all(math.isnan(v) for v in res.errors.values())
    assert any("3 of 9 bootstrap minimisations did not converge" in w and "20%" in w
               for w in res.warnings)
    assert "bootstrap (n=9, 3 failed)" in res.method


def test_unconverged_search_is_poor_without_errors(monkeypatch):
    _flaky(monkeypatch, lambda n: n == 1)  # only the main fit
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=5)
    assert res.poor
    assert set(res.errors) == {"xc", "nu", "a"}
    assert all(math.isnan(v) for v in res.errors.values())
    assert any(w.startswith("search did not converge") for w in res.warnings)
    assert "errors withheld" in res.method


def test_all_failed_minimisations_withhold_errors(monkeypatch):
    _flaky(monkeypatch, lambda n: n > 1)  # every draw, not the main fit
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=3)
    assert all(math.isnan(v) for v in res.errors.values())
    assert any("3 of 3 bootstrap" in w for w in res.warnings)


def _fake_minimize(monkeypatch, runs):
    """Replace scipy's minimize in odspy.collapse by a scripted sequence of results."""
    from scipy.optimize import OptimizeResult

    mod = sys.modules["odspy.collapse"]
    seq = iter([OptimizeResult(x=np.array([k + 1.0, 1.0]), fun=f, success=ok,
                               message="ok" if ok else "maxfev") for k, (f, ok) in
                enumerate(runs)])
    seen = []

    def fake(fun, x0, method, options):
        seen.append(np.array(x0))
        return next(seq)

    monkeypatch.setattr(mod, "minimize", fake)
    return mod, seen


def test_minimise_restarts_after_a_failed_first_run(monkeypatch):
    mod, seen = _fake_minimize(monkeypatch, [(5.0, False), (3.0, True), (3.0, True)])
    r = mod._minimise(lambda p: 0.0, np.array([0.0, 1.0]), np.array([0.1, 0.1]))
    assert r.success and r.fun == 3.0 and len(seen) == 3
    np.testing.assert_array_equal(seen[1], [1.0, 1.0])  # restarted from the failed run's best


def test_minimise_does_not_repeat_a_stuck_run(monkeypatch):
    # a failed run that ends where it started would only be repeated identically
    mod, seen = _fake_minimize(monkeypatch, [(5.0, False), (5.0, False)])
    r = mod._minimise(lambda p: 0.0, np.array([1.0, 1.0]), np.array([0.1, 0.1]))
    assert not r.success and len(seen) == 1


def test_minimise_prefers_a_converged_run(monkeypatch):
    # a later run with lower S that did not converge must not replace a converged one
    mod, seen = _fake_minimize(monkeypatch, [(3.0, True), (2.5, False), (3.0, True)])
    r = mod._minimise(lambda p: 0.0, np.array([0.0, 1.0]), np.array([0.1, 0.1]))
    assert r.success and r.fun == 3.0


def lines_family():
    """Three straight lines that are not a scaling family (reviewer's reproduction)."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 21)
    return [curve(8, x, 0.0 + 2 * x + rng.normal(0, .04, 21), .04),
            curve(16, x, 0.5 + 5 * x + rng.normal(0, .04, 21), .04),
            curve(32, x, 1.5 - 3 * x + rng.normal(0, .04, 21), .04)]


@pytest.mark.parametrize("start", [(0.2, 0.5, 0.0), (0.9, 0.3, 0.0)])
def test_non_collapsing_family_is_poor(start):
    res = search_collapse(lines_family(), *start, n_boot=10)
    print(f"start {start}: xc={res.xc:.4g} nu={res.nu:.4g} a={res.a:.4g} S={res.quality:.3g} "
          f"from {res.points_used} of {res.points_total}")
    assert res.poor
    assert res.points_total == 63


def test_little_overlap_is_poor():
    a = curve(1, [0, 1, 2, 3], [0, 1, 2, 3], 0.1)
    b = curve(1.5, [2, 3, 4], [3, 4, 5], 0.1)  # only X = 3 overlaps, and agrees: S = 0
    res = collapse([a, b], 0.0, 1.0)
    assert res.quality == pytest.approx(0, abs=1e-12) and res.poor
    assert (res.points_used, res.points_total) == (2, 7)
    assert "S from 2 of 7 points" in res.method
    assert any(w.startswith("too little overlap: S rests on 2 of 7 points") for w in res.warnings)


def test_good_collapse_reports_its_points():
    res = collapse(scaling_curves(), XC, NU, A)
    assert not res.poor and res.points_total == 99
    assert res.points_used / res.points_total >= 0.5
    assert f"S from {res.points_used} of 99 points" in res.method
    assert not any("too little overlap" in w for w in res.warnings)


def shifted_windows(seed):
    """The genuine family sampled in windows that barely overlap at the true parameters
    (28 of 63 points): an optimum held on the 50% overlap limit (reviewer's wall.py)."""
    rng = np.random.default_rng(seed)
    out = []
    for L, (lo, hi) in {8: (2.15, 2.27), 16: (2.25, 2.31), 32: (2.28, 2.31)}.items():
        x = np.linspace(lo, hi, 21)
        f = np.tanh((x - XC) * L ** (1 / NU)) + rng.normal(0, SIGMA, len(x))
        out.append(curve(L, x, L ** A * f, L ** A * SIGMA))
    return out


@pytest.mark.parametrize("seed", [1, 2])
def test_several_starts_near_the_overlap_limit_give_no_confident_errors(seed):
    res = search_collapse(shifted_windows(seed), XC, NU, A, n_boot=5)
    assert res.poor
    assert all(math.isnan(v) for v in res.errors.values())


@pytest.mark.parametrize("seed", [1, 2, 4])
def test_search_held_by_the_overlap_limit_is_poor(seed):
    res = search_collapse(shifted_windows(seed), XC, NU, A, starts=1, n_boot=5)
    print(f"seed {seed}: S={res.quality:.3g} {res.points_used}/{res.points_total} "
          f"poor={res.poor} {res.errors}")
    assert res.poor
    assert all(math.isnan(v) for v in res.errors.values())
    assert any(w.startswith("search constrained by the overlap limit") for w in res.warnings)


def test_poor_collapse_withholds_errors():
    # a held at a wrong value: S ~ 900, the search converges but the collapse is poor
    res = search_collapse(scaling_curves(), 2.25, 0.8, 1.5, fit_a=False, n_boot=10)
    assert res.poor and res.quality > 3
    assert set(res.errors) == {"xc", "nu"}
    assert all(math.isnan(v) for v in res.errors.values())
    assert any(w.startswith("errors withheld: poor collapse (S = ") for w in res.warnings)
    assert "errors withheld" in res.method


def test_degenerate_bootstrap_spread_withholds_errors(monkeypatch):
    col = sys.modules["odspy.collapse"]
    real = col._minimise
    calls = []

    def stuck(objective, start, steps):
        r = real(objective, start, steps)
        calls.append(1)
        if len(calls) > 1:         # bootstrap re-minimisations that never move
            r.x = np.asarray(start, float) * (1 + 1e-15 * len(calls))
            r.fun = objective(r.x)
        return r
    monkeypatch.setattr(col, "_minimise", stuck)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=10)
    assert all(math.isnan(v) for v in res.errors.values())
    assert any("errors withheld" in w and "bootstrap" in w for w in res.warnings)


def windows(seed, lo, hi):
    rng = np.random.default_rng(seed)
    out = []
    for L in SIZES:
        x = np.linspace(lo, hi, 21)
        f = np.tanh((x - XC) * L ** (1 / NU)) + rng.normal(0, SIGMA, len(x))
        out.append(curve(L, x, L ** A * f, L ** A * SIGMA))
    return out


def test_xc_outside_the_measured_range_is_flagged():
    res = search_collapse(windows(1, 2.18, 2.26), 2.24, 1.0, 0.25, n_boot=10)
    assert res.xc > 2.26
    assert res.poor
    assert all(math.isnan(v) for v in res.errors.values())
    assert any(w.startswith("x_c = ") and "outside the measured range [2.18, 2.26]: extrapolated"
               in w for w in res.warnings)


def test_xc_inside_the_range_is_not_flagged():
    res = search_collapse(windows(1, 2.18, 2.30), 2.24, 1.0, 0.25, n_boot=10)
    assert not res.poor and not any("outside the measured range" in w for w in res.warnings)


PINNED = "optimum held by coincident scaled points"


# (2.33, 1.0) ends at xc = 2.3499994, a few 1e-7 off the grid value: still pinned
@pytest.mark.parametrize("start", [(2.35, 1.0), (2.33, 0.9), (2.2, 1.0), (2.33, 1.0)])
def test_search_pinned_on_coincident_scaled_points_is_poor(outputs, start):
    # stat2 (L = 8, 12, 16, T = 2.1:2.4:0.05): from these starts the search ends exactly on a
    # grid temperature with nu = 1, where scaled points of different sizes coincide (S jumps
    # there) and finite errors were reported (reviewer's probe_a2/probe_a4)
    from odspy import families
    curves = families([outputs["stat2"]], "abs_susceptibility")
    res = search_collapse(curves, start[0], start[1], 1.75, starts=1, n_boot=20)
    assert res.poor
    assert all(math.isnan(v) for v in res.errors.values())
    assert any(PINNED in w and "try another start" in w for w in res.warnings)
    assert "coincident" in res.method


def test_several_starts_on_the_coarse_grid_give_no_confident_errors(outputs):
    # the same data with the default starts: the search may leave the pin for another
    # minimum, but on this 7-temperature grid its errors must not come back confident
    from odspy import families
    curves = families([outputs["stat2"]], "abs_susceptibility")
    res = search_collapse(curves, 2.35, 1.0, 1.75, n_boot=20)
    # the pinned optimum is never reported as a good one: either the result is poor or it is
    # another minimum, shown without errors and with the pinned one named as set aside
    assert not (abs(res.xc - 2.35) < 0.01 * 0.3 and not res.poor)
    assert all(math.isnan(v) for v in res.errors.values())
    assert res.poor or any(w.startswith("set aside") for w in res.warnings)


def test_healthy_optimum_on_the_same_data_is_not_flagged(outputs):
    # from 2.27 the search finds a lower S off the grid (xc = 2.252, nu = 0.40): not pinned.
    # On this coarse grid (7 temperatures) many bootstrap draws do end pinned on it; they
    # count as failed and, being more than 20%, withhold the errors
    from odspy import families
    curves = families([outputs["stat2"]], "abs_susceptibility")
    res = search_collapse(curves, 2.27, 1.0, 1.75, n_boot=20)
    assert not res.poor
    assert not any(PINNED in w for w in res.warnings)
    assert "coincident" not in res.method
    if not all(math.isfinite(v) for v in res.errors.values()):
        assert any("bootstrap minimisations" in w and "coincident" in w and "20%" in w
                   for w in res.warnings)


def test_coincident_pairs_counts_cross_size_coincidences():
    mod = sys.modules["odspy.collapse"]
    data = mod._Data(scaling_curves())
    # x grid centred on XC, sizes 8:16:32 and nu = 1: commensurate, many coincidences
    assert mod._coincident_pairs(data, XC, 1.0, A) > 2
    assert mod._coincident_pairs(data, XC + 1.234e-3, 1.037, A) <= 2
    assert mod._pinned(data, XC, 1.0, A) == mod._coincident_pairs(data, XC, 1.0, A)
    assert mod._pinned(data, XC + 1.234e-3, 1.037, A) == 0


def test_bootstrap_draws_on_coincident_points_count_as_failed(monkeypatch):
    mod = sys.modules["odspy.collapse"]
    calls = {"n": 0}

    def fake(data, xc, nu, a):
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 7      # the main fit is fine, every draw is pinned

    monkeypatch.setattr(mod, "_pinned", fake)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=5)
    assert all(math.isnan(v) for v in res.errors.values())
    assert any("5 of 5 bootstrap minimisations" in w and "coincident" in w for w in res.warnings)
    assert "bootstrap (n=5, 5 failed)" in res.method


@pytest.mark.parametrize("seed", [11, 14, 20])
def test_off_grid_optimum_with_a_few_coincidences_is_not_pinned(seed):
    # these optima sit off the x grid (x_c = 2.26991, 2.27011, 2.27015) at a corner of S's
    # jumps where 3 cross-size pairs coincide: generic, not held by the grid
    res = search_collapse(scaling_curves(seed=seed), 2.25, 0.8, 0.2, n_boot=20)
    assert not any(PINNED in w for w in res.warnings)
    assert not res.poor



# ------------------------------------------------------------- several starts

def _count_minimise(monkeypatch):
    mod = sys.modules["odspy.collapse"]
    real = mod._minimise
    calls = []

    def counted(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(mod, "_minimise", counted)
    return calls


def test_starts_one_is_a_single_search(monkeypatch):
    calls = _count_minimise(monkeypatch)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=1, n_boot=3)
    assert len(calls) == 1 + 3
    assert "1 start" in res.method


def test_default_search_tries_fifteen_starts(monkeypatch):
    calls = _count_minimise(monkeypatch)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, n_boot=3)
    assert len(calls) == 15 + 3
    assert "best of 15 starts" in res.method


@pytest.mark.parametrize("starts", [0, 16, 2.5, True])
def test_starts_must_be_between_1_and_15(starts):
    with pytest.raises(InputError, match="starts"):
        search_collapse(scaling_curves(), 2.25, 0.8, 0.2, starts=starts, n_boot=3)


def _second_minimum(monkeypatch, shift, factor):
    """Make the 2nd start end at the best fit moved by `shift` in xc, its objective value
    scaled so that S becomes `factor` x the best S (the other starts run for real)."""
    mod = sys.modules["odspy.collapse"]
    real = mod._minimise
    calls = []

    def scripted(objective, start, steps):
        calls.append(1)
        r = real(objective, start, steps)
        if len(calls) == 2:
            first = real(objective, np.array([2.25, 0.8, 0.2]), steps)
            x = np.array(first.x, float)
            x[0] += shift
            S = factor * first.fun / (1 - first.fun)
            r.x, r.fun, r.success = x, S / (1 + S), True
        return r
    monkeypatch.setattr(mod, "_minimise", scripted)


def test_comparable_minimum_elsewhere_warns(monkeypatch):
    _second_minimum(monkeypatch, shift=0.01, factor=1.05)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, n_boot=4)
    assert any(w.startswith("start-dependent: another minimum at xc=") for w in res.warnings)


def test_much_worse_minimum_elsewhere_does_not_warn(monkeypatch):
    _second_minimum(monkeypatch, shift=0.01, factor=3.0)
    res = search_collapse(scaling_curves(), 2.25, 0.8, 0.2, n_boot=4)
    assert not any("start-dependent" in w for w in res.warnings)


def test_several_starts_escape_a_local_minimum():
    # seed 3 from (2.2, 0.5, 0.5): one start stops in a local minimum (S = 2.12, nu = 0.83);
    # the grid of starts reaches the global one (S = 0.93, nu = 1.03; truth nu = 1)
    one = search_collapse(scaling_curves(seed=3), 2.2, 0.5, 0.5, starts=1, n_boot=4)
    many = search_collapse(scaling_curves(seed=3), 2.2, 0.5, 0.5, n_boot=4)
    assert one.quality > 2.0 and abs(one.nu - 1.0) > 0.1
    assert many.quality < 1.0 and abs(many.nu - 1.0) < 0.1
    assert not any("start-dependent" in w for w in many.warnings)



def test_comparable_minimum_within_the_errors_does_not_warn(monkeypatch):
    # a second start ending 1e-5 away in x_c (far inside the ~2e-4 error) is the same answer
    _second_minimum(monkeypatch, shift=1e-5, factor=1.05)
    res = search_collapse(scaling_curves(seed=11), 2.25, 0.8, 0.2, n_boot=20)
    assert all(math.isfinite(v) for v in res.errors.values())
    assert not any("start-dependent" in w for w in res.warnings)



@pytest.mark.parametrize("seed, start", [(1, (2.2708, 1.437, 0.527)), (2, (2.2704, 1.454, 0.513))])
def test_optimum_near_the_overlap_limit_is_poor_from_its_own_start(seed, start):
    # the nu ~ 1.44 optimum (truth 1.0) rests on 33 of 63 points: a move of 1% of the x span
    # loses the overlap, so it is held by the limit even when no start ends flagged nearby
    for starts in (15, 1):
        res = search_collapse(shifted_windows(seed), *start, starts=starts, n_boot=5)
        assert res.poor, (starts, res.nu, res.warnings)
        assert all(math.isnan(v) for v in res.errors.values())


@pytest.mark.parametrize("start", [(2.25, 0.8, 0.2), (2.27, 1.0, 0.25)])
def test_verdict_does_not_count_flagged_starts(start):
    # clean seed 10: several starts end flagged (on pins of the commensurate grid), none with a
    # lower S than the reported optimum: the verdict is the optimum's own, not poor
    res = search_collapse(scaling_curves(seed=10), *start, n_boot=10)
    assert not res.poor, res.warnings
    assert "flagged" in res.method


def test_every_start_flagged_reports_an_in_range_optimum(outputs):
    from odspy import families
    curves = families([outputs["stat2"]], "abs_susceptibility")
    res = search_collapse(curves, 2.4, 1.0, 1.75, n_boot=5)
    assert res.poor
    assert 2.1 <= res.xc <= 2.4 and 0 < res.nu < 10
    assert any("starts" in w and "flagged" in w for w in res.warnings)


@pytest.mark.parametrize("sigmas, warns", [(1.5, False), (3.0, True)])
def test_start_dependence_boundary_is_two_errors(monkeypatch, sigmas, warns):
    ref = search_collapse(scaling_curves(seed=11), 2.25, 0.8, 0.2, n_boot=20)
    assert math.isfinite(ref.errors["xc"])
    _second_minimum(monkeypatch, shift=sigmas * ref.errors["xc"], factor=1.05)
    res = search_collapse(scaling_curves(seed=11), 2.25, 0.8, 0.2, n_boot=20)
    assert any("start-dependent" in w for w in res.warnings) == warns



def test_lowest_s_on_a_flagged_optimum_makes_the_result_poor():
    # clean seed 11 from the true values: the lowest S is the pin at the true x_c (a measured
    # value on this commensurate grid, flagged); the reported off-pin optimum (83 of 99 points)
    # passes every check, but it is only the best of the rest
    res = search_collapse(scaling_curves(seed=11), 2.27, 1.0, 0.25, n_boot=10)
    assert res.points_used >= 0.6 * res.points_total
    assert res.poor
    assert any(w.startswith("the lowest S found is a flagged optimum") for w in res.warnings)
    assert all(math.isnan(v) for v in res.errors.values())


def test_valley_on_marginal_overlap_is_poor_from_every_start():
    # shifted_windows seed 2: the starts slide along a valley (nu 1.45 .. 1.9, truth 1.0) of
    # optima resting on about half of the points: poor however the search lands
    for start in ((2.2704, 1.454, 0.513), (2.265, 2.0, 0.6), (2.29, 1.2, 0.5)):
        res = search_collapse(shifted_windows(2), *start, n_boot=5)
        assert res.poor, (start, res.nu)
        assert all(math.isnan(v) for v in res.errors.values())


def test_collapse_on_barely_enough_points_is_poor():
    # shifted_windows seed 6 from this exact start: nu 1.585 (truth 1.0) rests on 33 of 63
    # points and passed every other check; rounding the start flipped the verdict. The share
    # of points in S is the optimum's own property: below 60% the collapse is poor
    res = search_collapse(shifted_windows(6), 2.2210878519, 1.5327656431, 0.0331486478, n_boot=20)
    assert res.points_used < 0.6 * res.points_total
    assert res.poor
    assert all(math.isnan(v) for v in res.errors.values())
    assert any("60%" in w for w in res.warnings)
