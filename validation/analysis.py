"""Small helpers shared by the model analyses (validation/models/*.py): a ratio of simulation
to theory over several sizes or points, an exponent from an odspy fit, and their figures.

Both kinds of check have predicted value 1 for a ratio and the theory's exponent for a fit;
what odspy says about the measurement (excluded points, fit quality, scaled errors) goes into
`Check.warnings`, so the summary table carries it next to the number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from scipy import stats

from odspy import FitResult
from odspy.io import Provenance
from odspy.plots.style import color, finish, new_fig, save

from validation.core import Check

P_THRESHOLD = 0.01      # below this p-value a ratio's points are called inconsistent


@dataclass
class Ratio:
    """Weighted mean of r_i = sim_i / theory_i: value +- error, the per-point ratios and their
    errors, and the consistency statistic of the points around one common ratio (`statistic`
    names it; p None when it is not assessed), plus any warnings about it."""
    value: float
    error: float
    ratios: np.ndarray
    errors: np.ndarray
    chi2: float
    dof: int
    p_value: float | None
    statistic: str = "chi2/dof"
    warnings: list[str] = field(default_factory=list)


def weighted_ratio(sim, sim_err, theory, theory_err=None) -> Ratio:
    """r_i = sim_i / theory_i with error hypot(sim_err_i, r_i theory_err_i) / theory_i (theory
    errors default to 0, for exact predictions), and their inverse-variance weighted mean."""
    sim, err, th = (np.asarray(v, float) for v in (sim, sim_err, theory))
    th_err = np.zeros_like(th) if theory_err is None else np.asarray(theory_err, float)
    if not (sim.ndim == 1 and sim.shape == err.shape == th.shape == th_err.shape and sim.size):
        raise ValueError("sim, sim_err, theory (and theory_err) must be 1-D of the same length")
    r = sim / th
    e = np.hypot(err, r * th_err) / np.abs(th)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(e)) and np.all(e > 0)):
        raise ValueError(f"ratios need finite values and positive errors (ratios {r}, errors {e})")
    w = 1.0 / e ** 2
    value = float(np.sum(w * r) / np.sum(w))
    chi2 = float(np.sum(w * (r - value) ** 2))
    dof = r.size - 1
    return Ratio(value, float(1.0 / math.sqrt(np.sum(w))), r, e, chi2, dof,
                 float(stats.chi2.sf(chi2, dof)) if dof > 0 else None)


def replica_ratio(V, exact) -> Ratio:
    """sim/theory over K correlated points measured on the same R replicas: V[R, K] holds each
    replica's value at each point, `exact` the K predictions (e.g. a time series against an
    exact curve). x_ki = V_ki / exact_i; r_i = mean_k x_ki. The weighted mean uses weights
    w_i = 1 / var_k(x_ki); being linear in the replicas, it is the mean of one number per
    replica, z_k = sum_i w_i x_ki / sum_i w_i, so its error is sd(z)/sqrt(R): exactly the
    bootstrap over whole replicas, correlations between points included.

    Consistency with one common ratio (when R >= 2K): Hotelling's
    T^2 = min_mu (r - mu)^T C^-1 (r - mu), C = covariance of x over replicas / R, with
    q = K - 1 and (R - q) T^2 / ((R - 1) q) ~ F(q, R - q), as odspy's fit_timeseries does."""
    V, ex = np.asarray(V, float), np.asarray(exact, float)
    if V.ndim != 2 or ex.ndim != 1 or V.shape[1] != ex.size or V.shape[0] < 2 or ex.size < 1:
        raise ValueError("V must be (R >= 2, K) and exact of length K")
    X = V / ex
    R, K = X.shape
    var = X.var(axis=0, ddof=1)
    if not (np.all(np.isfinite(X)) and np.all(var > 0)):
        raise ValueError("replica values must be finite with non-zero spread at every point")
    w = 1.0 / var
    z = X @ w / w.sum()
    r = X.mean(axis=0)
    chi2, p, warns = math.nan, None, []
    q = K - 1
    if q < 1:
        pass
    elif R < 2 * K:
        warns.append(f"consistency over the {K} points not assessed: R = {R} replicas (need "
                     "R >= 2K)")
    else:
        Ci = np.linalg.inv(np.cov(X, rowvar=False) / R)
        one = np.ones(K)
        mu = float(one @ Ci @ r / (one @ Ci @ one))
        chi2 = float((r - mu) @ Ci @ (r - mu))
        p = float(stats.f.sf(chi2 * (R - q) / ((R - 1) * q), q, R - q))
    return Ratio(float(z.mean()), float(z.std(ddof=1) / math.sqrt(R)), r,
                 np.sqrt(var / R), chi2, q, p, "T^2/dof (Hotelling)", warns)


def ratio_check(ratio: Ratio, *, model: str, check: str, quantity: str, source: str, sizes: str,
                statistics: str, warnings: Sequence[str] = (), figure: str | None = None,
                report_consistency: bool = False) -> Check:
    """A Check of a weighted ratio against 1 (exact). Points inconsistent with one common
    ratio (p < P_THRESHOLD) are flagged in the warnings; the error is not inflated. With
    `report_consistency` the consistency statistic is reported even when it is fine."""
    warns = list(warnings) + list(ratio.warnings)
    stat = f"{ratio.statistic} = {ratio.chi2:.3g}/{ratio.dof}, p = {ratio.p_value:.2g}" \
        if ratio.p_value is not None else ""
    if report_consistency and stat:
        warns.append(f"consistency over {ratio.dof + 1} points: {stat}")
    if ratio.p_value is not None and ratio.p_value < P_THRESHOLD:
        warns.append(f"ratios inconsistent across points: {stat}")
    return Check(model=model, check=check, quantity=quantity, measured=ratio.value,
                 error=ratio.error, predicted=1.0, predicted_error=0.0, source=source,
                 sizes=sizes, statistics=statistics, warnings=warns, figure=figure)


def fit_quality(r: FitResult) -> list[str]:
    """What a FitResult says about itself, as warning texts: its own warnings, chi2/dof and p
    (Hotelling's T^2 for a bootstrap time-series fit), poor fit, scaled errors."""
    out = list(r.warnings)
    if r.chi2 is not None:
        stat = "T^2/dof (Hotelling)" if r.chi2_expected is not None else "chi2/dof"
        out.append(f"fit: {stat} = {r.chi2:.3g}/{r.dof}, p = {r.p_value:.2g}")
        if r.poor:
            out.append(f"poor fit: p = {r.p_value:.2g} < {r.p_threshold:g}")
        if r.scaled:
            ref = r.chi2_expected if r.chi2_expected is not None else r.dof
            out.append(f"errors scaled by {math.sqrt(r.chi2 / ref):.3g} (misfit)")
    elif r.scaled:
        out.append("errors scaled for misfit")
    return out


def exponent_check(r: FitResult, predicted: float, *, model: str, check: str, quantity: str,
                   source: str, sizes: str, statistics: str, param: str = "b",
                   warnings: Sequence[str] = (), figure: str | None = None) -> Check:
    """A Check of fit parameter `param` (the exponent b of "power" / "power+c") against an exact
    prediction; the fit's warnings and quality follow the caller's warnings."""
    return Check(model=model, check=check, quantity=quantity, measured=r.params[param],
                 error=r.errors[param], predicted=float(predicted), predicted_error=0.0,
                 source=source, sizes=sizes, statistics=statistics,
                 warnings=list(warnings) + fit_quality(r), figure=figure)


def save_figure(fig: Figure, figures: Path, name: str) -> str:
    """Save `fig` as figures/<name>.pdf and .png (odspy's save) and return `name`, the
    Check.figure value."""
    save(fig, figures, name)
    return name


def plot_ratio(x, ratio: Ratio, xlabel: str, title: str | None = None,
               prov: Provenance | None = None, absolute=None, logx: bool = True) -> Figure:
    """sim/theory at each x with its error bar, the line 1 (theory) and the weighted mean's band.
    `absolute` = (sim, sim_err, theory_x, theory_y, ylabel) adds a left panel with the data
    against the theory curve (log axes when `logx`)."""
    x = np.asarray(x, float)
    fig, axes = new_fig(1, 2 if absolute is not None else 1)
    ax = axes[0, -1]
    if absolute is not None:
        sim, sim_err, tx, ty, ylabel = absolute
        a0 = axes[0, 0]
        a0.errorbar(x, sim, yerr=sim_err, fmt="o", ms=4, color=color(0), capsize=0, label="simulation")
        a0.plot(tx, ty, "-", color=color(1), lw=1, label="theory")
        scale = "log" if logx else "linear"
        a0.set(xscale=scale, yscale=scale, xlabel=xlabel, ylabel=ylabel)
        a0.legend()
    ax.errorbar(x, ratio.ratios, yerr=ratio.errors, fmt="o", ms=4, color=color(0), capsize=0,
                label="simulation / theory")
    ax.axhline(1.0, color=color(1), lw=1, label="theory")
    lo, hi = ratio.value - ratio.error, ratio.value + ratio.error
    ax.axhspan(lo, hi, color=color(0), alpha=0.15, lw=0,
               label=f"weighted mean {ratio.value:.4f} ± {ratio.error:.2g}")
    ax.set(xscale="log" if logx else "linear", xlabel=xlabel, ylabel="simulation / theory")
    ax.legend()
    return finish(fig, prov, title)


def plot_exponent(r: FitResult, predicted: float, xlabel: str, ylabel: str,
                  title: str | None = None, prov: Provenance | None = None,
                  param: str = "b", theory=None, legend: str = "upper left") -> Figure:
    """The fitted points with error bars (unused ones faded), the fit, and the theory curve:
    `theory` = (x, y, label) when given, else the same model with the exponent fixed at
    `predicted` and the other parameters refitted by weighted least squares. Log-log axes for
    "power", linear for "power+c". `legend`: its location (the default suits rising curves)."""
    x, y = np.asarray(r.x, float), np.asarray(r.y, float)
    e = None if r.yerr is None else np.asarray(r.yerr, float)
    used = np.asarray(r.used, bool) & np.isfinite(x) & np.isfinite(y)
    fig, axes = new_fig()
    ax = axes[0, 0]
    for sel, alpha, label in ((used, 1.0, "fitted"), (~used & np.isfinite(y), 0.35, "not fitted")):
        if sel.any():
            ax.errorbar(x[sel], y[sel], yerr=None if e is None else e[sel], fmt="o", ms=4,
                        color=color(0), alpha=alpha, capsize=0, label=label)
    xs = np.geomspace(x[used].min(), x[used].max(), 200) if r.model == "power" else \
        np.linspace(x[used].min(), x[used].max(), 200)
    ax.plot(xs, r.predict(xs), "-", color=color(0), lw=1,
            label=f"fit: {param} = {r.params[param]:.3f} ± {r.errors[param]:.2g}")
    if theory is not None:
        ax.plot(theory[0], theory[1], "--", color=color(1), lw=1, label=theory[2])
    else:   # exponent fixed, amplitude (and constant) by weighted least squares
        w = np.ones(used.sum()) if e is None else 1.0 / e[used] ** 2
        xb = x[used] ** predicted
        if r.model == "power+c":
            X = np.column_stack([xb, np.ones_like(xb)])
            A, c = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y[used]))
        else:
            A, c = np.sum(w * xb * y[used]) / np.sum(w * xb * xb), 0.0
        ax.plot(xs, A * xs ** predicted + c, "--", color=color(1), lw=1,
                label=f"theory: {param} = {predicted:g}")
    if r.model == "power":
        ax.set(xscale="log", yscale="log")
        ax.xaxis.set_minor_formatter(NullFormatter())    # minor labels collide within a decade
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc=legend)
    return finish(fig, prov, title)
