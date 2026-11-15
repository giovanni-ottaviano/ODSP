"""Per-replica final values: histograms, stop times, survival with censoring, stripes."""

from __future__ import annotations

import math

import numpy as np

from ..helpers import FIRST_PASSAGE, is_first_passage
from ..io import Table
from .style import (BINOMIAL, WILSON, binomial_yerr, color, finish, grid, hide_unused, new_fig,
                    numeric)

BOOKKEEPING = ("replica", "reached_consensus", "absorbed", "stop_time", "freeze_time", "stripes")


def _warn(warnings, text):
    if warnings is not None:
        warnings.append(text)


def kaplan_meier(time: np.ndarray, event: np.ndarray):
    """Survival curve S(t) = P(T > t) allowing for censored runs (event False). A run
    censored at t is still at risk at t; a run censored at +inf stays at risk for ever."""
    order = np.argsort(time, kind="stable")
    time, event = time[order], event[order]
    n = len(time)
    ts, ss, s = [0.0], [1.0], 1.0
    i = 0
    while i < n:
        j = i
        while j < n and time[j] == time[i]:
            j += 1
        d = int(event[i:j].sum())
        if d:
            s *= 1.0 - d / (n - i)
            ts.append(float(time[i]))
            ss.append(s)
        i = j
    return np.array(ts), np.array(ss)


def survival_curve(stop, reached, absorbed=None):
    """P(no consensus by t) from the per-replica stop times.

    Not for first-passage files (`helpers.is_first_passage`): their stop_time is the cap,
    not the first-passage time, and `plot_replicas` does not draw the curve for them.

    A run that reached consensus is an event at its stop time. A run that stopped in an
    absorbing state without consensus (froze) never reaches it: its consensus time is
    infinite, it stays at risk for ever, and the curve levels off at the frozen fraction.
    Only a run that hit the cap (not absorbed) is censored, at its stop time. Without an
    `absorbed` column every unfinished run is taken as censored."""
    stop = np.asarray(stop, float)
    done = np.asarray(reached, float) > 0.5
    frozen = np.zeros_like(done) if absorbed is None else (np.asarray(absorbed, float) > 0.5) & ~done
    return kaplan_meier(np.where(frozen, np.inf, stop), done)


def _finite(x: np.ndarray, label: str, warnings, ctx: str) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.all():
        _warn(warnings, f"{ctx}: {int((~ok).sum())} non-finite '{label}' values left out of the "
                        "histogram")
    return x[ok]


def plot_replicas(t: Table, warnings: list[str] | None = None):
    R = len(t)
    if R == 0:
        _warn(warnings, f"{t.path.name}: no rows; figures skipped")
        return []
    ctx = t.path.name
    figs = []
    obs = [n for n in t.names if n not in BOOKKEEPING and numeric(t, n) is not None]
    if obs:
        nr, nc = grid(len(obs))
        fig, axes = new_fig(nr, nc)
        for ax, n in zip(axes.ravel(), obs):
            v = _finite(t[n], n, warnings, ctx)
            if v.size:
                ax.hist(v, bins=min(30, max(5, int(math.sqrt(v.size)) + 1)), color=color(0),
                        alpha=0.85)
            ax.set(xlabel=f"final {n}", ylabel="replicas")
            ax.set_title(f"{v.size} replicas", fontsize=8)
        hide_unused(axes, len(obs))
        figs.append(("replicas_final", finish(fig, t.provenance)))
    first_passage = t.has("stop_time") and is_first_passage(t)
    if first_passage:
        _warn(warnings, f"{ctx}: stop-time histogram and survival curve skipped: {FIRST_PASSAGE} "
                        "(the driver's own survival column in _timeseries.csv, when present, is "
                        "drawn with the time series)")
    if t.has("stop_time") and t.has("reached_consensus") and not first_passage:
        stop = t["stop_time"]
        done = t["reached_consensus"] > 0.5
        absorbed = t["absorbed"] > 0.5 if t.has("absorbed") else None
        frozen = np.zeros(R, bool) if absorbed is None else absorbed & ~done
        capped = ~done & ~frozen
        n_frozen, n_capped = int(frozen.sum()), int(capped.sum())
        fig, axes = new_fig(1, 2, size=(4.6, 3.4))
        ax = axes[0, 0]
        ok = np.isfinite(stop)
        if not ok.all():
            _warn(warnings, f"{ctx}: {int((~ok).sum())} non-finite stop_time values left out of "
                            "the stop-time histogram and survival curve")
        bins = np.histogram_bin_edges(stop[ok], bins=min(30, max(5, int(math.sqrt(R)) + 1))) \
            if ok.any() else 10
        groups = [(stop[ok & done], color(0), "reached consensus")]
        if n_frozen:
            groups.append((stop[ok & frozen], color(3), "froze without consensus"))
        if n_capped:
            groups.append((stop[ok & capped], color(1), "censored (hit the cap before consensus)"))
        ax.hist([g[0] for g in groups], bins=bins, stacked=True, color=[g[1] for g in groups],
                label=[g[2] for g in groups])
        ax.set(xlabel="stop time", ylabel="replicas")
        ax.legend(loc="upper right")
        ax = axes[0, 1]
        ts, ss = survival_curve(stop[ok], done[ok], None if absorbed is None else absorbed[ok])
        ax.step(ts, ss, where="post", color=color(0), lw=1.4, label="survival (Kaplan-Meier)")
        if n_frozen:
            fs = stop[ok & frozen]
            ax.plot(fs, np.interp(fs, ts, ss) if len(ts) > 1 else np.ones_like(fs), "x",
                    color=color(3), ms=6, mew=1.2,
                    label=f"frozen without consensus: never ({n_frozen} of {R}; "
                          "the curve levels off at their fraction)")
        if n_capped:
            cs = stop[ok & capped]
            ax.plot(cs, np.interp(cs, ts, ss) if len(ts) > 1 else np.ones_like(cs), "|",
                    color=color(1), ms=10, mew=1.5,
                    label=f"censored runs: hit the cap ({n_capped} of {R})")
        ax.set(xlabel="time", ylabel="P(no consensus yet)", ylim=(-0.02, 1.02))
        ax.legend(loc="best")
        figs.append(("replicas_stoptime", finish(fig, t.provenance)))
    if t.has("freeze_time") and t.has("stop_time"):
        fig, axes = new_fig(1, 1)
        ax = axes[0, 0]
        both = np.isfinite(t["stop_time"]) & np.isfinite(t["freeze_time"])
        if not both.all():
            _warn(warnings, f"{ctx}: {int((~both).sum())} replicas with a non-finite "
                            "stop_time or freeze_time left out of the freeze scatter")
        ax.scatter(t["stop_time"][both], t["freeze_time"][both], s=12, color=color(0), alpha=0.7)
        hi = np.nanmax(np.r_[t["stop_time"], t["freeze_time"], 1.0])
        ax.plot([0, hi], [0, hi], color="#888888", lw=0.8, label="freeze_time = stop_time")
        ax.set(xlabel="stop_time", ylabel="freeze_time")
        ax.legend(loc="best")
        figs.append(("replicas_freeze", finish(fig, t.provenance)))
    if t.has("stripes"):
        s = t["stripes"]
        ok = np.isfinite(s)
        Rv = int(ok.sum())
        if Rv:
            p = float((s[ok] > 0.5).mean())
            err, edge = binomial_yerr([p], Rv)
            fig, axes = new_fig(1, 1, size=(2.8, 3.2))
            ax = axes[0, 0]
            ax.errorbar([0], [p], yerr=err, fmt="o", color=color(0), capsize=4,
                        label=f"stripe fraction, R={Rv}: "
                              + (WILSON.replace(" where p = 0 or 1", f" (p = {p:g})") if edge
                                 else BINOMIAL))
            ax.set(xlim=(-1, 1), ylim=(-0.02, 1.02), ylabel="fraction of replicas with a stripe",
                   xticks=[])
            ax.legend(loc="best")
            figs.append(("replicas_stripes", finish(fig, t.provenance)))
    if not figs:
        _warn(warnings, f"{ctx}: no numeric columns to draw; figures skipped")
    return figs
