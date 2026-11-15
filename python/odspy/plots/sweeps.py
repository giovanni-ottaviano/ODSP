"""Sweep tables: ensemble sweeps (sd/sqrt(replicas)) and stationary sweeps (jackknife)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import InputError
from ..helpers import _label, is_first_passage, stationary_caveats
from ..io import Table
from ..sizes import size_of
from .style import (BETWEEN_CHAIN, JACKKNIFE, SD_RT, color, finish, grid, hide_unused, masked,
                    new_fig, note_undefined, numeric)

CENSORED = "censored: mean biased low"
STATIONARY = [("mean", "⟨m⟩"), ("abs_mean", "⟨|m|⟩"), ("susceptibility", "susceptibility"),
              ("abs_susceptibility", "|m| susceptibility"), ("binder", "Binder U"),
              ("specific_heat", "specific heat"), ("xi_over_L", "ξ/L")]
_TIME_COLS = ("mean_consensus_time", "mean_consensus_time_lower_bound")


def _warn(warnings, text):
    if warnings is not None:
        warnings.append(text)


@dataclass
class Series:
    column: str
    label: str
    y: np.ndarray
    err: np.ndarray | None
    err_label: str


def is_stationary(t: Table) -> bool:
    return t.provenance.get("task") == "stationary" or t.has("chains") or t.has("tau_abs")


def _stationary_err_label(t: Table) -> str:
    chains = t["chains"] if t.has("chains") else None
    if chains is not None and np.nanmax(chains) > 1:
        return BETWEEN_CHAIN
    try:
        if int(float(t.provenance.get("replicas", "1"))) > 1 and chains is None:
            return BETWEEN_CHAIN
    except ValueError:
        pass
    return JACKKNIFE


def series_of(t: Table, only: str | None = None) -> list[Series]:
    out = []
    if is_stationary(t):
        lab = _stationary_err_label(t)
        for col, name in STATIONARY:
            if t.has(col) and numeric(t, col) is not None and (only in (None, col)):
                err = numeric(t, col + "_err") if t.has(col + "_err") else None
                out.append(Series(col, name, t[col], err, lab))
    else:
        reps = t["replicas"] if t.has("replicas") else None
        for n in t.names:
            if not n.startswith("mean_") or n in _TIME_COLS or numeric(t, n) is None:
                continue
            o = n[5:]
            if only not in (None, n):
                continue
            err = None
            if reps is not None and t.has("sd_" + o):
                with np.errstate(divide="ignore", invalid="ignore"):
                    err = np.where(reps >= 2, t["sd_" + o] / np.sqrt(reps), np.nan)
            out.append(Series(n, n, t[n], err, SD_RT))
    return out


def _flagged(t: Table, keys: list[str]) -> tuple[np.ndarray, list[str]]:
    """Points flagged by stationary_caveats (blocks short relative to tau), and its messages."""
    flag = np.zeros(len(t), bool)
    if not is_stationary(t):
        return flag, []
    msgs = stationary_caveats(t)
    for i in range(len(t)):
        label = ", ".join(f"{k}={_label(t[k][i])}" for k in keys)
        if any(m.startswith(label + ": jackknife blocks") for m in msgs):
            flag[i] = True
    return flag, msgs


def _xvalues(t: Table, name: str):
    """Numeric x for a key column, tick labels for categorical ones, axis label, log flag."""
    col = t[name]
    if col.dtype != object:
        return col.astype(float), None, name, False
    sizes = []
    for v in col:
        try:
            sizes.append(size_of({"dims": str(v)}, "auto" if False else "N")[0])
        except InputError:
            return np.arange(len(col), dtype=float), [str(v) for v in col], name, False
    return np.array(sizes, float), None, f"{name}: sites N", True


def _draw(ax, x, s: Series, flag, c, label_prefix="", single=False):
    y = masked(s.y)
    err = s.err
    good = ~flag
    lab = label_prefix + s.err_label if err is not None else label_prefix or None
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    ax.plot(xs, ys, "-", color=c, lw=0.9, alpha=0.6)
    e = np.zeros_like(y) if err is None else err      # NaN: no bar is drawn
    ax.errorbar(x[good], y[good], yerr=e[good], fmt="o", ms=4.5, color=c, capsize=3,
                label=lab)
    if flag.any():
        ax.errorbar(x[flag], y[flag], yerr=e[flag], fmt="o", ms=5.5, mfc="white", color=c,
                    capsize=3, label=f"{label_prefix}open marker: blocks short vs τ")


def _setx(ax, x, ticks, xlabel, logx):
    ax.set_xlabel(xlabel)
    if ticks is not None:
        ax.set_xticks(x, ticks, rotation=30)
    elif logx and x.size > 1 and (x > 0).all():
        ax.set_xscale("log")


def _tau_series(t: Table) -> list[Series]:
    return [Series(c, c, t[c], None, "") for c in ("tau", "tau_abs") if t.has(c)]


def plot_sweep(t: Table, warnings: list[str] | None = None):
    """Ensemble or stationary sweep (one outer key); tables with sweep2 go to plot_sweep2."""
    if len(t) == 0:
        _warn(warnings, f"{t.path.name}: no rows; figures skipped")
        return []
    if t.provenance.get("sweep2"):
        return plot_sweep2(t, warnings=warnings)
    key = t.names[0]
    x, ticks, xlabel, logx = _xvalues(t, key)
    flag, msgs = _flagged(t, [key])
    for m in msgs:
        _warn(warnings, f"{t.path.name}: {m}")
    series = series_of(t)
    for s in series:
        note_undefined(warnings, f"{t.path.name} {s.column}", s.err,
                       "fewer than 2 replicas or chains" if not is_stationary(t)
                       else "not computed")
    figs = []
    if series:
        nr, nc = grid(len(series))
        fig, axes = new_fig(nr, nc)
        for ax, s in zip(axes.ravel(), series):
            _draw(ax, x, s, flag, color(0))
            _setx(ax, x, ticks, xlabel, logx)
            ax.set_ylabel(s.label)
            if s.err is not None or flag.any():
                ax.legend(loc="best")
        hide_unused(axes, len(series))
        figs.append(("sweep", finish(fig, t.provenance)))
    elif not is_stationary(t):
        _warn(warnings, f"{t.path.name}: no 'mean_<observable>' columns besides consensus time")
    if is_stationary(t):
        taus = _tau_series(t)
        if taus:
            fig, axes = new_fig(1, 1)
            ax = axes[0, 0]
            for k, s in enumerate(taus):
                ax.plot(x, masked(s.y), "o-", ms=4, color=color(k), label=s.label)
            _setx(ax, x, ticks, xlabel, logx)
            ax.set(ylabel="integrated autocorrelation time (samples)")
            if flag.any():
                ax.plot(x[flag], masked(taus[0].y)[flag], "o", mfc="none", ms=10, color="k",
                        label="ring: blocks short vs τ")
            ax.legend(loc="best")
            figs.append(("sweep_tau", finish(fig, t.provenance)))
    else:
        figs += _consensus_figures(t, key, x, ticks, xlabel, logx, warnings)
    return figs


def _consensus_figures(t, key, x, ticks, xlabel, logx, warnings):
    if not t.has("consensus_fraction"):
        return []
    frac = t["consensus_fraction"]
    figs = []
    panels = 1 + t.has("mean_consensus_time")
    fig, axes = new_fig(1, panels)
    ax = axes[0, 0]
    ax.plot(x, frac, "o-", ms=4, color=color(0), lw=0.9)
    _setx(ax, x, ticks, xlabel, logx)
    ax.set(ylabel="consensus fraction", ylim=(-0.02, 1.02))
    if t.has("mean_consensus_time"):
        ax = axes[0, 1]
        reps = t["replicas"] if t.has("replicas") else np.full(len(t), np.nan)
        finished = np.round(reps * frac)
        cens = frac < 1 - 1e-12
        mean = t["mean_consensus_time"].copy().astype(float)
        mean[frac <= 0] = np.nan       # no run finished: there is no mean
        err = None
        if t.has("sd_consensus_time"):
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.where(finished >= 2, t["sd_consensus_time"] / np.sqrt(finished), np.nan)
        order = np.argsort(x, kind="stable")
        if t.has("median_consensus_time"):
            ax.plot(x[order], masked(t["median_consensus_time"])[order], "s--", ms=4,
                    color=color(2), lw=0.9, label="median (gap: fewer than half finished)")
        if t.has("mean_consensus_time_lower_bound"):
            ax.plot(x[order], masked(t["mean_consensus_time_lower_bound"])[order], "v:", ms=4,
                    color=color(3), lw=0.9, label="lower bound on the mean")
        if err is None:
            e = np.zeros(len(t))
        else:
            e = err                          # NaN: no bar is drawn
            note_undefined(warnings, f"{t.path.name} consensus time", err[np.isfinite(mean)],
                           "fewer than 2 finished runs, or no replicas column")
        ok = ~cens & np.isfinite(mean)
        ax.plot(x[order], masked(mean)[order], "-", color=color(0), lw=0.8, alpha=0.5)
        ax.errorbar(x[ok], mean[ok], yerr=e[ok], fmt="o", ms=4.5, color=color(0), capsize=3,
                    label="mean over finished runs ± sd/√finished runs")
        bad = cens & np.isfinite(mean)
        if bad.any():
            ax.errorbar(x[bad], mean[bad], yerr=e[bad], fmt="o", ms=5, mfc="white",
                        color=color(0), alpha=0.35, capsize=3, label=CENSORED)
            for xi, yi, fi in zip(x[bad], mean[bad], frac[bad]):
                ax.annotate(f"{CENSORED}\n(consensus in {fi:.0%})", (xi, yi), fontsize=6,
                            color="#555555", xytext=(-6, 8), textcoords="offset points",
                            ha="right")
        for i in np.flatnonzero(cens):
            _warn(warnings, f"{t.path.name}: {CENSORED}: {key}={_label(t[key][i])} "
                            f"(only {frac[i]:.0%} of replicas reached consensus; "
                            "the lower bound counts every unfinished run at the cap, --steps)")
        _setx(ax, x, ticks, xlabel, logx)
        ax.set_ylabel("first-passage time" if is_first_passage(t) else "consensus time")
        pos = [v for v in (mean, t["median_consensus_time"] if t.has("median_consensus_time")
                           else None) if v is not None]
        if all((np.nan_to_num(v, nan=1.0) > 0).all() for v in pos):
            ax.set_yscale("log")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2)
    figs.append(("sweep_consensus", finish(fig, t.provenance)))
    return figs


def plot_sweep2(t: Table, column: str | None = None, warnings: list[str] | None = None):
    """One curve per value of the outer key against the inner key, and a heatmap of one column
    over the two keys (the first available one when `column` is None)."""
    if len(t) == 0 or len(t.names) < 2:
        _warn(warnings, f"{t.path.name}: no rows; figures skipped")
        return []
    outer, inner = t.names[0], t.names[1]
    xin, ticks, xlabel, logx = _xvalues(t, inner)
    if column is not None and not t.has(column):
        raise InputError(f"{t.path}: no column '{column}' (has: {', '.join(t.names)})")
    flag, msgs = _flagged(t, [outer, inner])
    for m in msgs:
        _warn(warnings, f"{t.path.name}: {m}")
    series = series_of(t, only=column)
    for s in series:
        note_undefined(warnings, f"{t.path.name} {s.column}", s.err,
                       "fewer than 2 replicas or chains" if not is_stationary(t)
                       else "not computed")
    if column is not None and not series:
        series = [Series(column, column, t[column], None, "")] if numeric(t, column) is not None else []
    if not series:
        _warn(warnings, f"{t.path.name}: no plottable column for the sweep2 table")
        return []
    outers = list(dict.fromkeys(_label(v) for v in t[outer]))
    olab = np.array([_label(v) for v in t[outer]])
    nr, nc = grid(len(series))
    fig, axes = new_fig(nr, nc)
    for ax, s in zip(axes.ravel(), series):
        for k, o in enumerate(outers):
            m = olab == o
            _draw(ax, xin[m], Series(s.column, s.label, s.y[m],
                                     None if s.err is None else s.err[m], s.err_label),
                  flag[m], color(k), label_prefix=f"{outer}={o}: " if False else "")
        _setx(ax, xin, ticks, xlabel, logx)
        ax.set_ylabel(s.label)
    hide_unused(axes, len(series))
    handles = [plt_line(color(k), f"{outer}={o}") for k, o in enumerate(outers)]
    if series[0].err is not None:
        handles.append(plt_line("#888888", series[0].err_label, marker=False))
    if flag.any():
        handles.append(plt_line("#888888", "open marker: blocks short vs τ", hollow=True))
    fig.legend(handles=handles, loc="outside upper center", ncols=min(len(handles), 4))
    figs = [("sweep2", finish(fig, t.provenance))]
    figs.append(("sweep2_heatmap", _heatmap(t, series[0], outer, inner, warnings)))
    return figs


def plt_line(c, label, marker=True, hollow=False):
    from matplotlib.lines import Line2D
    if not marker:
        return Line2D([], [], color=c, lw=0, label=label)
    return Line2D([], [], color=c, marker="o", ms=5, mfc="white" if hollow else c,
                  label=label)


def _heatmap(t: Table, s: Series, outer: str, inner: str, warnings=None):
    olab = [_label(v) for v in t[outer]]
    ilab = [_label(v) for v in t[inner]]
    rows = list(dict.fromkeys(olab))
    inn = t[inner]
    if inn.dtype != object:
        cols_vals = sorted(set(float(v) for v in inn))
        cols = [_label(v) for v in cols_vals]
    else:
        cols = list(dict.fromkeys(ilab))
    z = np.full((len(rows), len(cols)), np.nan)
    seen: set = set()
    dup = 0
    for o, i, v in zip(olab, ilab, s.y):
        if (o, i) in seen:
            dup += 1
        seen.add((o, i))
        z[rows.index(o), cols.index(i)] = v
    if dup:
        _warn(warnings, f"{t.path.name}: {dup} duplicate ({outer}, {inner}) cells in the "
                        "heatmap; the last value of each is shown")
    fig, axes = new_fig(1, 1, size=(5.5, 3.8))
    ax = axes[0, 0]
    im = ax.imshow(np.ma.masked_invalid(z), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(cols)), cols, rotation=60, fontsize=6)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set(xlabel=inner, ylabel=outer, title=f"{s.label} (blank: no measurement)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=s.label)
    return finish(fig, t.provenance)
