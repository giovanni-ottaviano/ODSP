"""Run and ensemble time series."""

from __future__ import annotations

import numpy as np

from ..io import Table
from .style import (COARSENING, SE, color, finish, grid, hide_unused, log_axes, masked, new_fig,
                    numeric, positive)
from .style import warn as _warn


def plot_timeseries_run(t: Table, warnings: list[str] | None = None):
    """Each recorded observable against the sweep number (linear); coarsening observables also
    on log-log axes (sweep 0 is not drawn there; the axis label says so)."""
    xname = t.names[0]
    x = numeric(t, xname)
    obs = [n for n in t.names[1:] if numeric(t, n) is not None]
    if x is None or not obs or len(t) == 0:
        _warn(warnings, f"{t.path.name}: no numeric observables or no rows; figure skipped")
        return []
    figs = []
    nr, nc = grid(len(obs))
    fig, axes = new_fig(nr, nc)
    for ax, n in zip(axes.ravel(), obs):
        ax.plot(x, masked(t[n]), color=color(0), lw=1.2)
        ax.set_xlabel(xname)
        ax.set_ylabel(n)
    hide_unused(axes, len(obs))
    figs.append(("timeseries", finish(fig, t.provenance)))
    coarse = [n for n in obs if n.split("_")[0] in COARSENING]
    if coarse:
        fig, axes = new_fig(1, len(coarse))
        for ax, n in zip(axes.ravel(), coarse):
            xs, ys, k = positive(x, t[n], warnings, t.path.name, n)
            ax.plot(xs, ys, color=color(0), lw=1.2)
            log = log_axes(ax, [(xs, ys)], warnings, t.path.name, n)
            ax.set(xlabel=f"{xname} ({xname}=0 not shown)" if log else xname, ylabel=n)
            if k and log:
                ax.set_title(f"{k} values ≤ 0 not shown", fontsize=7)
        figs.append(("timeseries_loglog", finish(fig, t.provenance)))
    return figs


def _observables(t: Table) -> list[str]:
    return [n[:-5] for n in t.names if n.endswith("_mean") and numeric(t, n) is not None]


def plot_timeseries_ensemble(t: Table, warnings: list[str] | None = None):
    """Mean +- standard error of each observable, the running-only mean, running_fraction and
    survival S(t); coarsening observables on log-log axes (t = 0 is not drawn there)."""
    time = numeric(t, t.names[0])
    obs = _observables(t)
    if time is None or len(t) == 0:
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    figs = []
    if obs:
        nr, nc = grid(len(obs))
        fig, axes = new_fig(nr, nc)
        for ax, o in zip(axes.ravel(), obs):
            log = o.split("_")[0] in COARSENING
            sel = time > 0 if log else np.ones(len(time), bool)
            tt = time[sel]
            m = masked(t[o + "_mean"])[sel]
            hidden = 0
            if log:
                _, m, hidden = positive(tt, m, warnings, t.path.name, o + "_mean")
            ax.plot(tt, m, color=color(0), lw=1.3, label="mean")
            if t.has(o + "_err"):
                e = masked(t[o + "_err"])[sel]
                ax.fill_between(tt, (m - e).filled(np.nan), (m + e).filled(np.nan),
                                color=color(0), alpha=0.25, lw=0, label=f"mean {SE}")
            if t.has(o + "_mean_running"):
                run = masked(t[o + "_mean_running"])[sel]
                if log:
                    _, run, _ = positive(tt, run, warnings, t.path.name, o + "_mean_running")
                ax.plot(tt, run, color=color(1), lw=1, ls="--",
                        label="mean over still-running replicas")
            if log:
                log = log_axes(ax, [(tt, m)] + ([(tt, run)] if t.has(o + "_mean_running") else []),
                               warnings, t.path.name, o + "_mean")
                if log and hidden:
                    ax.set_title(f"{hidden} mean values ≤ 0 not shown", fontsize=7)
            ax.set_xlabel("time (t=0 not shown on log axes)" if log else "time")
            ax.set_ylabel(o)
        hide_unused(axes, len(obs))
        # one legend above the panels (the same entries in each): inside an axes it covers curves
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), fontsize=7)
        figs.append(("timeseries_ensemble", finish(fig, t.provenance)))
    panels = [n for n in ("running_fraction", "survival") if t.has(n)]
    if panels:
        fig, axes = new_fig(1, len(panels))
        for ax, n in zip(axes.ravel(), panels):
            ax.step(time, masked(t[n]), where="post", color=color(2), lw=1.3)
            ax.set(xlabel="time", ylabel=n, ylim=(-0.02, 1.02))
            if n == "survival":
                ax.set_ylabel("survival S(t)")
        figs.append(("timeseries_survival", finish(fig, t.provenance)))
    if not figs:
        _warn(warnings, f"{t.path.name}: no '<observable>_mean' columns; figure skipped")
    return figs


def plot_trajectories(t: Table, warnings: list[str] | None = None):
    """Every replica's trajectory of each observable (thin lines, all drawn)."""
    if len(t) == 0 or not t.has("replica") or not t.has("time"):
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    obs = [n for n in t.names if n not in ("replica", "time") and numeric(t, n) is not None]
    if not obs:
        _warn(warnings, f"{t.path.name}: no observable columns; figure skipped")
        return []
    rep, time = t["replica"], t["time"]
    ids = np.unique(rep)
    nr, nc = grid(len(obs))
    fig, axes = new_fig(nr, nc)
    for ax, o in zip(axes.ravel(), obs):
        log = o.split("_")[0] in COARSENING
        hidden = 0
        drawn = []
        for r in ids:
            m = rep == r
            if log:
                xs, ys, k = positive(time[m], t[o][m], None, "", o)
                hidden += k
            else:
                xs, ys = time[m], masked(t[o])[m]
            drawn.append((xs, ys))
            ax.plot(xs, ys, lw=0.6, alpha=0.35, color=color(0))
        if log:
            log = log_axes(ax, drawn, warnings, t.path.name, o)
            if log and hidden:
                _warn(warnings, f"{t.path.name}: {hidden} {o} values <= 0 not shown on log axes")
        ax.set_xlabel("time (t=0 not shown on log axes)" if log else "time")
        ax.set_ylabel(o)
        ax.set_title(f"{ids.size} replicas" + (f"; {hidden} values ≤ 0 not shown" if log and hidden
                                              else ""), fontsize=8)
    hide_unused(axes, len(obs))
    return [("trajectories", finish(fig, t.provenance))]
