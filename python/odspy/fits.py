"""Least-squares fits that carry their own quality (chi2/dof, p-value, flags).

Models: "power" (A x^b), "power+c" (A x^b + c), "log" (a + b ln x),
"line" (a + b x). Weighted fits need an error bar on every point in range;
an unweighted fit must be asked for explicitly and is labelled as such.
"""

from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

from .errors import InputError

WEIGHTED = "weighted least squares"
UNWEIGHTED = "unweighted (no error bars)"


def _power(x, A, b):
    return A * np.power(x, b)


def _power_c(x, A, b, c):
    return A * np.power(x, b) + c


# name -> (parameter names, formula, function, linear basis or None, needs x > 0)
_MODELS = {
    "power": (("A", "b"), "y = A x^b", _power, None, True),
    "power+c": (("A", "b", "c"), "y = A x^b + c", _power_c, None, True),
    "log": (("a", "b"), "y = a + b ln x", lambda x, a, b: a + b * np.log(x),
            lambda x: np.column_stack([np.ones_like(x), np.log(x)]), True),
    "line": (("a", "b"), "y = a + b x", lambda x, a, b: a + b * x,
             lambda x: np.column_stack([np.ones_like(x), x]), False),
}


class _FitFailed(Exception):
    pass


def _check_model(model: str):
    if model not in _MODELS:
        raise InputError(f"unknown fit model '{model}' (expected one of: {', '.join(_MODELS)})")
    return _MODELS[model]


def _loglog_start(x, y):
    """(A, b) from an OLS of ln y on ln x over the points with x, y > 0."""
    ok = (x > 0) & (y > 0)
    if ok.sum() >= 2 and np.ptp(np.log(x[ok])) > 0:
        b, lna = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        return math.exp(lna), b
    return (float(np.mean(y) / np.mean(x)) if np.mean(x) != 0 else 1.0), 1.0


def _start(model, x, y):
    A, b = _loglog_start(x, y)
    return [A, b] if model == "power" else [A, b, 0.0]


def _core(model, x, y, sigma=None, p0=None):
    """Fit `model` to (x, y). sigma: None (unweighted, covariance from the residual
    scatter), 1-D error bars, or a 2-D covariance matrix (absolute in both cases).
    Returns (params, covariance, notes)."""
    names, _, func, basis, _ = _MODELS[model]
    notes: list[str] = []
    if basis is not None:
        X = basis(x)
        if sigma is None:
            Xw, yw = X, y
        elif np.ndim(sigma) == 1:
            Xw, yw = X / sigma[:, None], y / sigma
        else:
            try:
                L = np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError:
                raise _FitFailed("covariance matrix is not positive definite") from None
            Xw, yw = np.linalg.solve(L, X), np.linalg.solve(L, y)
        params, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        try:
            cov = np.linalg.inv(Xw.T @ Xw)
        except np.linalg.LinAlgError:
            cov = np.full((len(names), len(names)), np.inf)
            notes.append("parameter errors could not be estimated (degenerate x values)")
        if sigma is None:
            dof = len(y) - len(names)
            cov = cov * (np.sum((yw - Xw @ params) ** 2) / dof)
        return params, cov, notes
    if p0 is None:
        p0 = _start(model, x, y)
    with _warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        _warnings.simplefilter("always")
        try:
            params, cov = curve_fit(func, x, y, p0=p0, sigma=sigma,
                                    absolute_sigma=sigma is not None, maxfev=20000)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            raise _FitFailed(str(e)) from None
    if any(issubclass(w.category, OptimizeWarning) for w in caught) or \
            not np.all(np.isfinite(cov)):
        cov = np.full((len(names), len(names)), np.inf)
        notes.append("parameter errors could not be estimated (singular covariance)")
    return np.asarray(params, float), np.asarray(cov, float), notes


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    errors: dict[str, float]
    chi2: float | None
    dof: int
    p_value: float | None
    scaled: bool
    poor: bool
    method: str
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None
    used: np.ndarray
    xrange: tuple[float, float]
    warnings: list[str]
    p_threshold: float = 0.01
    notes: list[str] = field(default_factory=list)   # how errors / chi2 were obtained
    # expected value of `chi2` for a correct model when it is not dof: Hotelling's T^2 of a
    # bootstrap time-series fit, E[T^2] = (R-1)q/(R-q-2); None for an ordinary chi2
    chi2_expected: float | None = None

    def predict(self, x) -> np.ndarray:
        func = _MODELS[self.model][2]
        with np.errstate(all="ignore"):
            return func(np.asarray(x, float), *(self.params[k] for k in _MODELS[self.model][0]))

    def residuals(self) -> np.ndarray:
        with np.errstate(all="ignore"):
            r = self.y - self.predict(self.x)
            return r if self.yerr is None else r / self.yerr

    def summary(self) -> str:
        names, formula = _MODELS[self.model][:2]
        n = int(self.used.sum())
        lo, hi = self.xrange
        lines = [f"{self.model} fit {formula}: {n} of {self.x.size} points with "
                 f"x in [{lo:.6g}, {hi:.6g}], {self.method}"]
        for k in names:
            lines.append(f"  {k} = {self.params[k]:.6g} +/- {self.errors[k]:.2g}")
        hotelling = self.chi2_expected is not None
        if self.chi2 is not None and hotelling:
            lines.append(f"  T²/dof (Hotelling, not centred on dof) = {self.chi2:.4g}/{self.dof} = "
                         f"{self.chi2 / self.dof:.3g}, E[T²] = {self.chi2_expected:.4g}, "
                         f"p = {self.p_value:.3g}")
        elif self.chi2 is not None:
            lines.append(f"  chi2/dof = {self.chi2:.4g}/{self.dof} = {self.chi2 / self.dof:.3g}, "
                         f"p = {self.p_value:.3g}")
        elif self.method == UNWEIGHTED:
            lines.append("  goodness of fit not assessed (unweighted fit); "
                         "errors from the scatter of the residuals")
        if self.poor:
            lines.append(f"  poor fit: p = {self.p_value:.3g} < {self.p_threshold:g}")
        if self.scaled and self.chi2 is not None and hotelling:
            lines.append("  errors scaled by sqrt(T²/E[T²]) = "
                         f"{math.sqrt(self.chi2 / self.chi2_expected):.3g}")
        elif self.scaled and self.chi2 is not None:
            lines.append(f"  errors scaled by sqrt(chi2/dof) = {math.sqrt(self.chi2 / self.dof):.3g}")
        # (a scaled result without a reported chi2 explains its scaling in `notes`)
        lines += [f"  {s}" for s in self.notes]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _as_array(v, name: str) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"fit: {name} must be numeric") from None
    if a.ndim != 1:
        raise InputError(f"fit: {name} must be one-dimensional")
    return a


def _because(warns: list[str]) -> str:
    """The exclusions behind a refusal, appended to its message."""
    return f" ({'; '.join(warns)})" if warns else ""


def _fit(x, y, yerr=None, model="power", xrange=None, unweighted=False, p_threshold=0.01,
         exclude=(), prior=()) -> FitResult:
    """`fit`, plus `exclude`: (mask, message) pairs; in-range finite points in a mask are
    dropped and counted with message.format(n=..., points=...); and `prior`: warnings
    from the caller (e.g. points it could not compute), put first in the result's
    warnings. A refusal (too few points) names all of these warnings."""
    names, _, _, _, needs_pos = _check_model(model)
    x, y = _as_array(x, "x"), _as_array(y, "y")
    e = None if yerr is None else _as_array(yerr, "yerr")
    if y.size != x.size or (e is not None and e.size != x.size):
        raise InputError(f"fit: x, y and yerr must have the same length "
                         f"(have {x.size}, {y.size}" + ("" if e is None else f", {e.size}") + ")")
    with np.errstate(invalid="ignore"):
        if xrange is None:
            used = np.ones(x.size, bool)
        else:
            try:
                lo, hi = (float(v) for v in xrange)
            except (TypeError, ValueError):
                raise InputError(f"fit: xrange must be a pair (lo, hi), got {xrange!r}") from None
            if not (lo <= hi):
                raise InputError(f"fit: xrange must have lo <= hi, got ({lo:g}, {hi:g})")
            used = (x >= lo) & (x <= hi)
        warns: list[str] = list(prior)
        # A point with a non-finite x cannot be placed against the range (comparisons with
        # NaN are False), so it is always counted; a NaN y only when its x is in range.
        bad = ~np.isfinite(x) | (used & ~np.isfinite(y))
        if bad.any():
            warns.append(f"{_plural(int(bad.sum()), 'point')} with NaN values excluded")
            used &= ~bad
        if needs_pos:
            bad = used & (x <= 0)
            if bad.any():
                warns.append(f"{_plural(int(bad.sum()), 'point')} with x <= 0 excluded "
                             f"({model} needs x > 0)")
                used &= ~bad
        for mask, message in exclude:
            bad = used & mask
            if bad.any():
                warns.append(message.format(points=_plural(int(bad.sum()), "point")))
                used &= ~bad
        if not unweighted:
            if e is None:
                raise InputError("fit: no error bars (yerr) given: pass yerr, or "
                                 "unweighted=True for an explicitly unweighted fit")
            bad = used & ~(np.isfinite(e) & (e > 0))
            if bad.any():
                raise InputError(f"fit: {_plural(int(bad.sum()), 'point')} in range with "
                                 "missing, NaN or non-positive errors (yerr): fix them, narrow "
                                 "xrange, or pass unweighted=True for an unweighted fit"
                                 + _because(warns))
    n, k = int(used.sum()), len(names) + 1
    if n < k:
        raise InputError(f"need at least {k} points in range, have {n}" + _because(warns))
    xu, yu = x[used], y[used]
    sigma = None if unweighted else e[used]
    try:
        params, cov, notes = _core(model, xu, yu, sigma)
    except _FitFailed as err:
        raise InputError(f"fit: {model} fit did not converge ({err})") from None
    warns += notes
    errors = np.sqrt(np.diag(cov))
    dof = n - len(names)
    chi2 = p = None
    scaled = False
    if not unweighted:
        with np.errstate(all="ignore"):
            chi2 = float(np.sum(((yu - _MODELS[model][2](xu, *params)) / sigma) ** 2))
        p = float(stats.chi2.sf(chi2, dof))
        if chi2 / dof > 1:
            errors = errors * math.sqrt(chi2 / dof)
            scaled = True
    if xrange is None:
        xr = (float(xu.min()), float(xu.max()))
    else:
        xr = (lo, hi)
    return FitResult(
        model=model,
        params={kk: float(v) for kk, v in zip(names, params)},
        errors={kk: float(v) for kk, v in zip(names, errors)},
        chi2=chi2, dof=dof, p_value=p, scaled=scaled,
        poor=p is not None and p < p_threshold,
        method=UNWEIGHTED if unweighted else WEIGHTED,
        x=x, y=y, yerr=None if unweighted else e, used=used, xrange=xr,
        warnings=warns, p_threshold=p_threshold,
    )


def fit(x, y, yerr=None, model="power", xrange=None, unweighted=False,
        p_threshold=0.01) -> FitResult:
    """Least-squares fit of `model` to the points with x in `xrange` (default: all).

    Weighted by `yerr` unless `unweighted=True` (then no chi2 or p-value, errors
    from the residual scatter). Points outside the range stay in the result
    (`used` marks the fitted ones); NaN points are excluded and counted in
    `warnings`. Errors are multiplied by sqrt(chi2/dof) when it exceeds 1
    (`scaled`). `poor` when p < p_threshold.
    """
    return _fit(x, y, yerr, model, xrange, unweighted, p_threshold)


def plain_loglog_slope(x, y, xrange=None) -> float:
    """Legacy (benchmarks/analyse.py): unweighted OLS slope of ln y on ln x over the
    points with x, y > 0 (and lo <= x <= hi when `xrange` is given). No error, no
    quality measure: use `fit` for that."""
    pts = list(zip(np.asarray(x, float).tolist(), np.asarray(y, float).tolist()))
    if xrange is not None:
        lo, hi = xrange
        pts = [(a, b) for a, b in pts if lo <= a <= hi]
    pts = [(math.log(a), math.log(b)) for a, b in pts if a > 0 and b > 0]
    n = len(pts)
    if n < 2:
        raise InputError(f"plain_loglog_slope: need at least 2 points with x, y > 0, have {n}")
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((a - mx) ** 2 for a, _ in pts)
    if den == 0:
        raise InputError("plain_loglog_slope: all x values are equal")
    return sum((a - mx) * (b - my) for a, b in pts) / den
