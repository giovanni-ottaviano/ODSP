"""Fits to ensemble time series, whose points are correlated in time.

Every time point of an ensemble time series comes from the same replicas, so a
chi2 treating the points as independent means nothing. With the per-replica
file (`_trajectories.csv`, from --save-trajectories true) the parameter errors
come from a bootstrap over whole replicas, and a correlated chi2 is reported
when there are enough replicas. Without it the fit is made, no goodness of fit
is reported, and the result is labelled as untrustworthy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .errors import InputError
from .fits import _MODELS, WEIGHTED, FitResult, _check_model, _core, _fit, _FitFailed
from .io import Table

UNTRUSTWORTHY = "errors not trustworthy: correlated points (rerun with --save-trajectories true)"
UNTRUSTWORTHY_G = ("errors not trustworthy: correlated points (G(r) at different r share the "
                   "same configurations)")
UNTRUSTWORTHY_RUN = "errors not trustworthy: one autocorrelated trajectory"
UNTRUSTWORTHY_PREFIX = "errors not trustworthy"
BOOTSTRAP = "bootstrap over replicas"
CORRELATED = "weighted, correlated points"
_SPREAD = "{points} with zero or missing error excluded (all replicas equal, or one replica)"


def _apply(transform, m: np.ndarray) -> np.ndarray:
    if transform is None:
        return m
    with np.errstate(all="ignore"):
        y = np.asarray(transform(m), dtype=float)
    if y.shape != m.shape:
        raise InputError(f"transform must act elementwise: got shape {y.shape} for {m.shape}")
    return y


def _slope(transform, m: np.ndarray) -> np.ndarray:
    """d transform / dm at m (central difference); 1 without a transform."""
    if transform is None:
        return np.ones_like(m)
    h = 1e-6 * np.maximum(np.abs(m), 1e-12)
    return (_apply(transform, m + h) - _apply(transform, m - h)) / (2 * h)


def _replica_matrix(traj: Table, observable: str) -> tuple[np.ndarray, np.ndarray]:
    """(values[R, T], times[T]) from a trajectories table."""
    cols = [traj[name] for name in (observable, "replica", "time")]
    try:
        vals, rep, tim = (np.asarray(c, dtype=float) for c in cols)
    except (TypeError, ValueError):
        raise InputError(f"{traj.path}: non-numeric values in the replica, time or "
                         f"'{observable}' column") from None
    if vals.size == 0:
        raise InputError(f"{traj.path}: no rows")
    # Group by replica, keeping each replica's rows in file order (the driver writes
    # times in recording order; a NaN time must stay in place to be counted later).
    order = np.argsort(rep, kind="stable")
    reps = np.unique(rep)
    R = reps.size
    if vals.size % R:
        raise InputError(f"{traj.path}: replicas have different recording times")
    T = vals.size // R
    times = tim[order].reshape(R, T)
    if not np.all(rep[order].reshape(R, T) == reps[:, None]) or \
            not np.all((times == times[0]) | (np.isnan(times) & np.isnan(times[0]))):
        raise InputError(f"{traj.path}: replicas have different recording times")
    return vals[order].reshape(R, T), times[0]


def _without_trajectories(ts, observable, model, xrange, transform, p_threshold) -> FitResult:
    t = ts["time"]
    m, e = ts[f"{observable}_mean"], ts[f"{observable}_err"]
    y = _apply(transform, m)
    with np.errstate(all="ignore"):
        ey = np.abs(_slope(transform, m)) * e
        spread = np.isfinite(y) & ~(np.isfinite(ey) & (ey > 0))
    r = _fit(t, y, ey, model, xrange, False, p_threshold, exclude=[(spread, _SPREAD)])
    return mark_untrustworthy(r, UNTRUSTWORTHY)


def mark_untrustworthy(r: FitResult, label: str) -> FitResult:
    """A fit to correlated points, made as if they were independent: no chi2 or p-value
    (they would mean nothing), never flagged poor, `label` (an "errors not trustworthy: …"
    text) added to the warnings; a weighted fit's method becomes CORRELATED."""
    if r.scaled and r.chi2 is not None:
        r.notes.append(f"errors scaled by sqrt(chi2/dof) = {np.sqrt(r.chi2 / r.dof):.3g} of "
                       "the independent-point fit (not a goodness of fit: points correlated)")
    r.chi2 = r.p_value = None
    r.poor = False
    if r.method == WEIGHTED:
        r.method = CORRELATED
    if label not in r.warnings:
        r.warnings.append(label)
    return r


def fit_timeseries(ts: Table, observable: str, model="power", xrange=None,
                   trajectories: Table | None = None, transform=None,
                   n_boot=1000, seed=0, p_threshold=0.01) -> FitResult:
    """Fit `model` to `<observable>_mean` against `time` of an ensemble time series.

    `transform` (elementwise, e.g. ``lambda rho: 1 / rho``) is applied to the mean
    series before fitting.

    Without `trajectories`: weighted fit with `<observable>_err` (propagated through
    `transform` to first order); no chi2 or p-value, labelled UNTRUSTWORTHY.

    With `trajectories` (R replicas): `params` from an unweighted fit to the
    full-sample means; `errors` = standard deviation of the parameters over
    `n_boot` resamples of whole replicas (numpy Generator(seed)); `yerr` the
    bootstrap standard deviation of each (transformed) mean. Resamples that are
    non-finite at a fitted time (e.g. 1/rho with rho = 0) are discarded from both
    and counted in `warnings`. Goodness of fit, only
    when R >= 2K for K fitted times: chi2 = min over the parameters of r^T C^-1 r,
    C = covariance of the replica values at the fitted times / R (through
    `transform` to first order); because C is itself estimated from the R
    replicas, p comes from Hotelling's T^2 (F distribution with K - P and
    R - K + P degrees of freedom) rather than from chi2(K - P), which would flag
    most correct fits as poor. When T^2 exceeds its expectation for a correct model,
    E[T^2] = (R-1)q/(R-q-2) with q = K - P, the bootstrap errors (which see only the
    replica noise) are multiplied by sqrt(T^2/E[T^2]) and `scaled` is set; when T^2
    cannot be assessed, a note says the errors do not include any misfit.

    Points with zero spread over the replicas (all equal, e.g. after every
    replica stopped) are excluded and counted in `warnings`.
    """
    names = _check_model(model)[0]
    if not ts.has("time"):
        raise InputError(f"{ts.path}: no 'time' column: fit_timeseries needs an ensemble "
                         "time series (<prefix>_timeseries.csv from --task ensemble)")
    if trajectories is None:
        return _without_trajectories(ts, observable, model, xrange, transform, p_threshold)

    try:
        n_boot = int(n_boot)
    except (TypeError, ValueError):
        raise InputError(f"n_boot must be an integer >= 2, got {n_boot!r}") from None
    if n_boot < 2:
        raise InputError(f"n_boot must be an integer >= 2, got {n_boot}")
    V, times = _replica_matrix(trajectories, observable)
    t = np.asarray(ts["time"], float)
    if t.shape != times.shape or not np.allclose(t, times, equal_nan=True):
        raise InputError(f"{trajectories.path} and {ts.path}: recording times differ "
                         "(not from the same run?)")
    R = V.shape[0]
    if R < 2:
        raise InputError(f"{trajectories.path}: need at least 2 replicas for a bootstrap, "
                         f"have {R}")
    m = V.mean(axis=0)
    y = _apply(transform, m)
    with np.errstate(invalid="ignore"):
        spread = np.isfinite(y) & ~(np.ptp(V, axis=0) > 0)
    base = _fit(t, y, None, model, xrange, True, p_threshold, exclude=[(spread, _SPREAD)])
    used = base.used
    xu = t[used]
    p0 = [base.params[k] for k in names]

    rng = np.random.default_rng(seed)
    boot_params, boot_y, nonfinite, failed = [], [], 0, 0
    for _ in range(n_boot):
        yb = _apply(transform, V[rng.integers(0, R, R)].mean(axis=0))
        if not np.all(np.isfinite(yb[used])):
            nonfinite += 1          # discarded entirely: neither errors nor yerr see it
            continue
        boot_y.append(yb)
        try:
            params, _, _ = _core(model, xu, yb[used], None, p0)
        except _FitFailed:
            failed += 1
            continue
        if np.all(np.isfinite(params)):
            boot_params.append(params)
        else:
            failed += 1
    warnings = list(base.warnings)
    if len(boot_params) < 2:
        raise InputError(f"fit_timeseries: only {len(boot_params)} of {n_boot} bootstrap "
                         "resamples could be fitted")
    if nonfinite:
        warnings.append(f"{nonfinite} of {n_boot} bootstrap resamples discarded "
                        "(non-finite values at fitted times)")
    if failed:
        warnings.append(f"{failed} of {n_boot} bootstrap resamples discarded "
                        "(fit did not converge)")
    errors = np.std(np.array(boot_params), axis=0, ddof=1)
    with np.errstate(invalid="ignore"):
        yerr = np.std(np.array(boot_y), axis=0, ddof=1)
    yerr[~np.isfinite(yerr)] = np.nan   # unknown at unfitted points with non-finite resamples
    notes = [f"errors: standard deviation over {len(boot_params)} bootstrap resamples of "
             f"R = {R} replicas (seed {seed})"]

    K, P = int(used.sum()), len(names)
    dof = K - P
    chi2 = p = expected = None
    scaled = False
    if R < 2 * K:
        warnings.append(f"goodness of fit not assessed: R = {R} replicas for K = {K} "
                        "correlated points (need R >= 2K)")
    else:
        d = _slope(transform, m[used])
        C = np.cov(V[:, used], rowvar=False) / R * np.outer(d, d)
        try:
            pg, _, _ = _core(model, xu, y[used], C, p0)
            L = np.linalg.cholesky(C)
            with np.errstate(all="ignore"):
                z = np.linalg.solve(L, y[used] - _MODELS[model][2](xu, *pg))
            chi2 = float(z @ z)
            if not np.isfinite(chi2):
                raise _FitFailed("non-finite chi2")
        except (_FitFailed, np.linalg.LinAlgError):
            chi2 = None
            warnings.append("goodness of fit not assessed: covariance of the replica values "
                            "at the fitted times is singular")
        if chi2 is not None:
            p = float(stats.f.sf(chi2 * (R - dof) / ((R - 1) * dof), dof, R - dof))
            notes.append(f"chi2 correlated: minimised over the parameters with the replica "
                         f"covariance at the K = {K} fitted times; p from Hotelling's T^2 "
                         f"(R = {R} replicas)")
            # E[T^2] for a correct model: T^2 ~ (R-1)q/(R-q) F(q, R-q), q = K - P
            expected = (R - 1) * dof / (R - dof - 2) if R - dof - 2 > 0 else None
            if expected is not None and chi2 > expected:
                # the bootstrap spread sees only the replica noise, not the misfit
                errors = errors * np.sqrt(chi2 / expected)
                scaled = True
    if chi2 is None:
        notes.append("errors: bootstrap spread only; they do not include any misfit of the "
                     "model (goodness of fit not assessed)")
    return FitResult(
        model=model, params=base.params,
        errors={k: float(v) for k, v in zip(names, errors)},
        chi2=chi2, dof=dof, p_value=p, scaled=scaled,
        poor=p is not None and p < p_threshold, method=BOOTSTRAP,
        x=t, y=y, yerr=yerr, used=used, xrange=base.xrange,
        warnings=warnings, p_threshold=p_threshold, notes=notes,
        chi2_expected=expected if chi2 is not None else None,
    )
