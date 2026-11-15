import math
from pathlib import Path

import numpy as np
import pytest

from odspy import InputError, SizeCurve, crossings_plain, find_crossings, read_table

ROOT = Path(__file__).resolve().parents[2]
METHOD = "linear interpolation, parametric bootstrap (n=2000)"


def curve(size, x, y, yerr=0.002):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.full(len(x), yerr) if np.isscalar(yerr) else np.asarray(yerr, float)
    return SizeCurve(size, "L", x, y, yerr, "temperature", "m", "test")


def model(L, x):
    return 0.6 + 0.05 * (x - 2.27) * L


def test_known_crossing():
    x = np.linspace(2.15, 2.39, 13)
    rng = np.random.default_rng(11)
    sigma = 0.002
    xs, errs = [], []
    for rep in range(50):
        curves = [curve(L, x, model(L, x) + rng.normal(0, sigma, len(x)), sigma)
                  for L in (8, 16, 32)]
        res = find_crossings(curves, n_boot=500, seed=1)
        assert len(res) == 2
        for c in res:
            assert c.found and c.note == "" and c.n_boot == 500
            if rep == 0:
                assert abs(c.x - 2.27) < 3 * c.err
        xs.append([c.x for c in res])
        errs.append([c.err for c in res])
    xs, errs = np.array(xs), np.array(errs)
    z = np.abs(xs - 2.27) / errs
    print(f"fraction |z|<3: {(z < 3).mean():.3f}, max z {z.max():.2f}")
    assert (z < 3).mean() >= 0.95
    for k in range(2):
        scatter, mean_err = xs[:, k].std(ddof=1), errs[:, k].mean()
        print(f"pair {k}: scatter {scatter:.5f}, mean err {mean_err:.5f}")
        assert scatter / 2 < mean_err < scatter * 2


def test_default_bootstrap_and_fields():
    x = np.linspace(2.15, 2.39, 13)
    res = find_crossings([curve(L, x, model(L, x)) for L in (8, 16)])
    (c,) = res
    assert (c.size_a, c.size_b) == (8, 16)
    assert c.method == METHOD and c.n_boot == 2000
    assert c.x == pytest.approx(2.27) and c.y == pytest.approx(0.6)
    assert c.err > 0 and c.warnings == []


def test_no_crossing_in_range():
    x = np.linspace(2.0, 2.5, 11)
    res = find_crossings([curve(8, x, 0.5 + 0 * x), curve(16, x, 0.7 + 0 * x)])
    (c,) = res
    assert c.found is False
    assert c.note == "no crossing in range"
    assert math.isnan(c.x) and math.isnan(c.err) and math.isnan(c.y)


def test_crossing_outside_overlap_not_extrapolated():
    # difference changes sign at x=3.0, beyond both curves
    x = np.linspace(2.0, 2.5, 11)
    res = find_crossings([curve(8, x, 0.5 + 0.1 * (x - 3)), curve(16, x, 0.5 + 0.2 * (x - 3))])
    assert not res[0].found and res[0].note == "no crossing in range"


def test_different_grids():
    def f(L, x):
        return 0.6 + 0.3 * np.tanh(3 * (x - 2.27)) * (L / 8)
    xa = np.arange(2.20, 2.341, 0.04)
    xb = xa + 0.02
    common = np.concatenate([xa, xb])
    common.sort()
    inter = find_crossings([curve(8, xa, f(8, xa)), curve(16, xb, f(16, xb))], seed=3)[0]
    same = find_crossings([curve(8, common, f(8, common)),
                           curve(16, common, f(16, common))], seed=3)[0]
    assert inter.found and same.found
    assert abs(inter.x - same.x) <= max(inter.err, same.err)
    assert abs(inter.x - 2.27) < 0.01


def test_overlap_only():
    # curve b covers only part of curve a; crossing lies outside the overlap
    xa = np.linspace(2.0, 2.6, 13)
    xb = np.linspace(2.4, 2.6, 5)
    a = curve(8, xa, model(8, xa))
    b = curve(16, xb, model(16, xb))
    assert not find_crossings([a, b])[0].found


def test_several_sign_changes():
    x = np.linspace(0, 1, 41)
    a = curve(8, x, np.cos(10 * x), 0.001)
    b = curve(16, x, 0 * x, 0.001)
    res = find_crossings([a, b], n_boot=300)
    assert len(res) == 3 and all(c.found for c in res)
    assert [c.x for c in res] == pytest.approx(
        [math.pi / 20, 3 * math.pi / 20, 5 * math.pi / 20], abs=0.005)


def test_curve_touching_zero_exactly():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # difference: +1, 0 (exact), -1, -2, -3 -> a single crossing at x=1
    a = curve(8, x, [1.0, 0.0, -1.0, -2.0, -3.0], 0.01)
    b = curve(16, x, np.zeros(5), 0.01)
    res = find_crossings([a, b], n_boot=200)
    assert len(res) == 1 and res[0].found and res[0].x == pytest.approx(1.0)
    # tangency (touches zero, no sign change) is not a crossing
    t = curve(8, x, [1.0, 0.0, 1.0, 2.0, 3.0], 0.01)
    assert not find_crossings([t, b], n_boot=200)[0].found
    # zero at the last point only
    e = curve(8, x, [3.0, 2.0, 1.0, 0.5, 0.0], 0.01)
    r = find_crossings([e, b], n_boot=200)
    assert len(r) == 1 and r[0].found and r[0].x == pytest.approx(4.0)


def test_nonfinite_points_excluded_and_reported():
    x = np.linspace(2.15, 2.39, 13)
    ya = model(8, x)
    ya[3] = np.nan
    yerr = np.full(13, 0.002)
    yerr[5] = np.nan
    res = find_crossings([curve(8, x, ya, yerr), curve(16, x, model(16, x))])
    (c,) = res
    assert c.found and c.x == pytest.approx(2.27, abs=0.01)
    assert any("2 non-finite" in w and "8" in w for w in c.warnings)


def test_no_yerr_gives_nan_err_and_warning():
    x = np.linspace(2.15, 2.39, 13)
    res = find_crossings([curve(8, x, model(8, x), None), curve(16, x, model(16, x))])
    (c,) = res
    assert c.found and math.isnan(c.err) and c.n_boot == 0
    assert any("yerr" in w for w in c.warnings)
    assert "bootstrap" not in c.method


def test_bootstrap_draws_without_crossing_reported():
    # crossing between the last two points, huge noise: many draws lose it
    x = np.array([0.0, 1.0, 2.0])
    a = curve(8, x, [1.0, 0.5, 0.0], 5.0)
    b = curve(16, x, [0.0, 0.5, 0.2], 5.0)
    res = find_crossings([a, b], n_boot=400, seed=2)
    assert res[0].found
    assert any("no crossing" in w and "bootstrap" in w for w in res[0].warnings)


def test_seed_reproducible():
    x = np.linspace(2.15, 2.39, 13)
    cs = [curve(L, x, model(L, x)) for L in (8, 16)]
    assert find_crossings(cs, seed=5)[0].err == find_crossings(cs, seed=5)[0].err


def test_bad_input():
    x = np.linspace(0, 1, 5)
    with pytest.raises(InputError):
        find_crossings([curve(8, x, x)])
    with pytest.raises(InputError):
        find_crossings([curve(8, x, x), curve(8, x, x)])
    with pytest.raises(InputError):
        find_crossings([curve(8, x, x), curve(16, x, x)], n_boot=1)
    # too few finite points: no traceback, not found, warned
    one = curve(16, [0.0, 1.0], [np.nan, 1.0])
    r = find_crossings([curve(8, x, x), one])
    assert not r[0].found and r[0].warnings


def test_plain_matches_legacy():
    results = ROOT / "benchmarks" / "results"
    if not results.exists():
        pytest.skip("benchmarks/results absent")
    tables = {}
    for L in (16, 32, 64):
        t = read_table(results / f"ising_L{L}_sweep.csv")
        tables[L] = [{n: float(t[n][i]) for n in t.names} for i in range(len(t))]

    def legacy(tables, xkey, ykey):
        Ls = sorted(tables)
        out = []
        for a, b in zip(Ls, Ls[1:]):
            xa = [r[xkey] for r in tables[a]]
            da = [r[ykey] - s[ykey] for r, s in zip(tables[a], tables[b])]
            for i in range(len(xa) - 1):
                if da[i] == 0 or da[i] * da[i + 1] < 0:
                    out.append((a, b, xa[i] - da[i] * (xa[i + 1] - xa[i]) / (da[i + 1] - da[i])))
        return out

    for ykey in ("binder", "xi_over_L"):
        got = crossings_plain(tables, "temperature", ykey)
        assert got == legacy(tables, "temperature", ykey)
    assert crossings_plain(tables, "temperature", "binder")


def test_negative_yerr_raises_input_error():
    x = np.linspace(2.15, 2.39, 13)
    err = np.full(13, 0.002)
    err[4] = -0.001
    bad = curve(8, x, model(8, x), err)
    with pytest.raises(InputError, match=r"size 8.*negative.*x=2\.23"):
        find_crossings([bad, curve(16, x, model(16, x))])
    # zero yerr is legitimate
    err[4] = 0.0
    res = find_crossings([curve(8, x, model(8, x), err), curve(16, x, model(16, x))], n_boot=50)
    assert res[0].found


def test_unsorted_x_is_sorted_with_y_and_yerr():
    x = np.linspace(2.15, 2.39, 13)
    ref = find_crossings([curve(8, x, model(8, x)), curve(16, x, model(16, x))], seed=4)[0]
    p = np.random.default_rng(0).permutation(13)
    err = np.linspace(0.001, 0.003, 13)
    shuf = find_crossings([curve(8, x[p], model(8, x)[p], err[p]),
                           curve(16, x[p], model(16, x)[p], err[p])], seed=4)[0]
    sort = find_crossings([curve(8, x, model(8, x), err), curve(16, x, model(16, x), err)],
                          seed=4)[0]
    assert shuf.x == pytest.approx(ref.x) and shuf.found
    assert shuf.err == sort.err


def test_duplicate_x_raises_input_error():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    with pytest.raises(InputError, match=r"size 8.*duplicate x.*1"):
        find_crossings([curve(8, x, [0, 1, 2, 3.0]), curve(16, [0.0, 1.0, 2.0], [1, 1, 1.0])])
