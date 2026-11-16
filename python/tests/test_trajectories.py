from pathlib import Path

import numpy as np
import pytest

from odspy.errors import InputError
from odspy.fits import fit
from odspy.io import Provenance, Table, find_outputs
from odspy.trajectories import fit_timeseries

LABEL = "errors not trustworthy: correlated points (rerun with --save-trajectories true)"
TIMES = np.logspace(0, 4, 12)


def tables(V, times=TIMES, name="y"):
    """Synthetic (timeseries_ensemble, trajectories) tables from a replicas x times matrix."""
    R, T = V.shape
    cols = {"time": times, f"{name}_mean": V.mean(0),
            f"{name}_err": V.std(0, ddof=1) / np.sqrt(R)}
    ts = Table(Path("syn_timeseries.csv"), "timeseries_ensemble", Provenance(), cols, list(cols))
    tcols = {"replica": np.repeat(np.arange(R, dtype=float), T),
             "time": np.tile(times, R), name: V.ravel()}
    tr = Table(Path("syn_trajectories.csv"), "trajectories", Provenance(), tcols, list(tcols))
    return ts, tr


def ensemble(rng, R, times=TIMES, scale=0.1, noise=0.01):
    """y_r(t) = t^0.5 + u_r + noise: a per-replica offset u_r, so the points are
    strongly correlated in time and the offset biases the fitted exponent."""
    u = rng.normal(0, scale, R)
    return times ** 0.5 + u[:, None] + noise * rng.normal(size=(R, times.size))


def test_timeseries_without_trajectories_labelled(outputs):
    out = find_outputs(outputs["ens"])
    r = fit_timeseries(out.get("timeseries_ensemble"), "length", trajectories=None)
    assert r.p_value is None and r.chi2 is None and r.poor is False
    assert LABEL in r.warnings
    assert r.method == "weighted, correlated points"
    assert LABEL in r.summary()


def test_timeseries_with_real_trajectories(outputs):
    out = find_outputs(outputs["ens"])
    r = fit_timeseries(out.get("timeseries_ensemble"), "length", xrange=(1, 63),
                       trajectories=out.get("trajectories"), n_boot=200)
    assert r.method == "bootstrap over replicas"
    assert r.used.sum() == 10                       # times 1, 2, 3, 4, 6, 10, ..., 63
    assert np.isfinite(r.errors["b"]) and r.errors["b"] > 0
    assert isinstance(r.p_value, float)             # R = 24 >= 2K = 20
    assert LABEL not in r.warnings


def test_bootstrap_covers_truth():
    rng = np.random.default_rng(21)
    boot, naive = 0, 0
    n = 100
    for i in range(n):
        V = ensemble(rng, 40)
        ts, tr = tables(V)
        r = fit_timeseries(ts, "y", trajectories=tr, n_boot=300, seed=i)
        assert r.method == "bootstrap over replicas"
        boot += abs(r.params["b"] - 0.5) < r.errors["b"]
        nf = fit(ts["time"], ts["y_mean"], ts["y_err"])     # independent-point weighted fit
        naive += abs(nf.params["b"] - 0.5) < nf.errors["b"]
    print(f"bootstrap coverage {boot / n:.2f}, naive coverage {naive / n:.2f}")
    assert 0.55 <= boot / n <= 0.80
    assert naive / n < 0.40


def test_correlated_chi2_needs_enough_replicas():
    rng = np.random.default_rng(22)
    ts, tr = tables(ensemble(rng, 10))
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert r.p_value is None and r.chi2 is None
    assert ("goodness of fit not assessed: R = 10 replicas for K = 12 correlated points "
            "(need R >= 2K)") in r.warnings
    ts, tr = tables(ensemble(rng, 60))
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert isinstance(r.p_value, float) and isinstance(r.chi2, float)
    assert r.dof == 10
    assert not any(w.startswith("goodness of fit not assessed") for w in r.warnings)


def test_correlated_p_values_calibrated():
    # Correct model, correlated points: the p-values should be roughly uniform,
    # so a correct fit is rarely flagged.
    rng = np.random.default_rng(23)
    ps = []
    for i in range(100):
        ts, tr = tables(ensemble(rng, 40))
        ps.append(fit_timeseries(ts, "y", trajectories=tr, n_boot=20, seed=i).p_value)
    ps = np.array(ps)
    print(f"mean p {ps.mean():.3f}, fraction p < 0.01 {np.mean(ps < 0.01):.3f}")
    assert 0.3 < ps.mean() < 0.7
    assert np.mean(ps < 0.01) <= 0.05


def test_correlated_wrong_model_flagged():
    rng = np.random.default_rng(24)
    V = 3 + ensemble(rng, 60)                       # y = 3 + t^0.5: not a pure power law
    ts, tr = tables(V)
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert r.poor is True
    assert "poor fit" in r.summary()


def test_transform_and_zero_spread():
    rng = np.random.default_rng(25)
    V = ensemble(rng, 30)
    V[:, -1] = 100.0                                # all replicas equal at the last time
    ts, tr = tables(V)
    r = fit_timeseries(ts, "y", trajectories=tr, transform=lambda y: y ** 2, n_boot=50)
    assert r.params["b"] == pytest.approx(1.0, abs=0.1)
    assert not r.used[-1]
    assert "1 point with zero or missing error excluded (all replicas equal, or one replica)" \
        in r.warnings
    r = fit_timeseries(ts, "y", transform=lambda y: y ** 2)
    assert r.params["b"] == pytest.approx(1.0, abs=0.1)
    assert not r.used[-1]


def test_bad_trajectories():
    rng = np.random.default_rng(26)
    ts, tr = tables(ensemble(rng, 10))
    with pytest.raises(InputError, match="no column 'z'"):
        fit_timeseries(ts, "z", trajectories=tr)
    ts2, _ = tables(ensemble(rng, 10), times=TIMES * 2)
    with pytest.raises(InputError, match="times"):
        fit_timeseries(ts2, "y", trajectories=tr)
    ragged = Table(tr.path, tr.kind, tr.provenance,
                   {k: v[:-1] for k, v in tr.columns.items()}, tr.names)
    with pytest.raises(InputError, match="times"):
        fit_timeseries(ts, "y", trajectories=ragged)
    with pytest.raises(InputError, match="n_boot"):
        fit_timeseries(ts, "y", trajectories=tr, n_boot=1)


def test_bootstrap_yerr_finite_with_nonfinite_resamples():
    # 1/y is infinite in the resamples that miss replica 0 (mean 0 at the first time).
    rng = np.random.default_rng(27)
    V = ensemble(rng, 30)
    V[:, 0] = 0.0
    V[0, 0] = 1.0
    ts, tr = tables(V)
    r = fit_timeseries(ts, "y", trajectories=tr, transform=lambda y: 1 / y, n_boot=200,
                       model="line", xrange=(0, 10))
    assert np.all(np.isfinite(r.yerr[r.used]))
    assert any("bootstrap resamples discarded (non-finite" in w for w in r.warnings)


def test_bootstrap_yerr_finite_when_mean_crosses_zero():
    rng = np.random.default_rng(28)
    V = ensemble(rng, 30)
    V[:, 3] = rng.normal(0, 1, 30)
    V[:, 3] -= V[:, 3].mean() - 0.2                 # full-sample mean 0.2, resamples cross 0
    ts, tr = tables(V)
    r = fit_timeseries(ts, "y", trajectories=tr, transform=lambda y: np.sqrt(y), n_boot=200)
    assert r.used[3]
    assert np.all(np.isfinite(r.yerr[r.used]))
    w = [s for s in r.warnings if "bootstrap resamples discarded (non-finite" in s]
    assert len(w) == 1 and w[0].endswith("of 200 bootstrap resamples discarded "
                                         "(non-finite values at fitted times)")


def test_nan_time_counted_in_timeseries_fits():
    rng = np.random.default_rng(29)
    times = TIMES.copy()
    times[4] = np.nan
    ts, tr = tables(ensemble(rng, 30), times=times)
    for traj in (None, tr):
        r = fit_timeseries(ts, "y", trajectories=traj, xrange=(1, 1e4), n_boot=50)
        assert "1 point with NaN values excluded" in r.warnings
        assert not r.used[4] and r.x.size == 12


@pytest.mark.parametrize("with_traj", [False, True])
def test_refusal_names_the_zero_spread_exclusions(with_traj):
    V = np.ones((6, TIMES.size))                      # every replica identical: zero spread
    ts, tr = tables(V)
    with pytest.raises(InputError) as e:
        fit_timeseries(ts, "y", trajectories=tr if with_traj else None)
    msg = str(e.value)
    assert "have 0" in msg and "zero or missing error excluded" in msg and "12 points" in msg


def test_poor_bootstrap_fit_scales_its_errors(monkeypatch):
    rng = np.random.default_rng(24)
    ts, tr = tables(3 + ensemble(rng, 60))          # wrong model: T^2 far above E[T^2]
    unscaled = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    R, K, P = 60, int(unscaled.used.sum()), 2
    q = K - P
    expected = (R - 1) * q / (R - q - 2)
    assert unscaled.chi2 > expected
    assert unscaled.scaled is True
    assert unscaled.chi2_expected == pytest.approx(expected)
    # the bootstrap sd times sqrt(T^2 / E[T^2]): the same bootstrap with T^2 unavailable
    def singular(*a, **k):
        raise np.linalg.LinAlgError("singular")
    with monkeypatch.context() as m:
        m.setattr(np.linalg, "cholesky", singular)
        raw = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert raw.chi2 is None and raw.scaled is False
    factor = np.sqrt(unscaled.chi2 / expected)
    for k in ("A", "b"):
        assert unscaled.errors[k] == pytest.approx(raw.errors[k] * factor)
    s = unscaled.summary()
    assert "errors scaled by sqrt(T²/E[T²]) = " in s
    assert "T²/dof (Hotelling, not centred on dof)" in s and "E[T²] = " in s
    assert "chi2/dof" not in s


def test_good_bootstrap_fit_is_not_scaled():
    rng = np.random.default_rng(24)
    ts, tr = tables(ensemble(rng, 60))
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert r.chi2 <= r.chi2_expected                 # the fixture: T^2 below its expectation
    assert r.scaled is False and "errors scaled" not in r.summary()


def test_unassessed_bootstrap_fit_says_errors_exclude_misfit():
    rng = np.random.default_rng(22)
    ts, tr = tables(ensemble(rng, 10))                # R = 10 < 2K = 24
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=50)
    assert r.scaled is False
    assert any("do not include any misfit" in n for n in r.notes)


def test_correlated_fit_not_converging_is_not_called_singular(monkeypatch):
    import odspy.trajectories as T
    from odspy.fits import _FitFailed
    rng = np.random.default_rng(25)
    ts, tr = tables(ensemble(rng, 60))
    real = T._core

    def gls_fails(model, x, y, sigma=None, p0=None):
        if sigma is not None and np.ndim(sigma) == 2:
            raise _FitFailed("maxfev reached")
        return real(model, x, y, sigma, p0)
    monkeypatch.setattr(T, "_core", gls_fails)
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=30)
    assert r.chi2 is None
    assert any("did not converge (maxfev reached)" in w for w in r.warnings)
    assert not any("singular" in w for w in r.warnings)


def test_bootstrap_errors_drop_the_plain_fits_unknown_errors_note(monkeypatch):
    import odspy.fits as F
    rng = np.random.default_rng(26)
    ts, tr = tables(ensemble(rng, 60))
    real = F._core

    def no_errors(model, x, y, sigma=None, p0=None):
        params, cov, notes = real(model, x, y, sigma, p0)
        return params, np.full_like(cov, np.inf), notes + [f"{F.ERRORS_UNKNOWN} (singular covariance)"]
    monkeypatch.setattr(F, "_core", no_errors)       # the plain fit only: the bootstrap uses its own
    r = fit_timeseries(ts, "y", trajectories=tr, n_boot=30)
    assert all(np.isfinite(v) for v in r.errors.values())
    assert not any(F.ERRORS_UNKNOWN in w for w in r.warnings + r.notes)
