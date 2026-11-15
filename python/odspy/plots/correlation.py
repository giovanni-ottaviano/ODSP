"""G(r) curves."""

from __future__ import annotations

import numpy as np

from ..io import Table
from .style import (BETWEEN_CHAIN, JACKKNIFE, SE, color, finish, log_axes, masked, new_fig,
                    note_undefined)
from matplotlib import cm
from matplotlib.colors import Normalize


def _warn(warnings, text):
    if warnings is not None:
        warnings.append(text)


def plot_correlation(t: Table, warnings: list[str] | None = None):
    """G(r): linear and log-log. Run tables: one curve per recording sweep; ensemble tables:
    one curve per time, with error bars; stationary tables: the curve with its error."""
    if len(t) == 0 or t.kind not in ("correlation_run", "correlation_ensemble",
                                     "correlation_stationary"):
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    group = {"correlation_run": "sweep", "correlation_ensemble": "time"}.get(t.kind)
    ycol = "G" if t.kind == "correlation_run" else "G_mean"
    r, y = t["r"], t[ycol]
    err = t["G_err"] if t.has("G_err") else None
    if t.kind == "correlation_ensemble":
        elabel = f"{SE} over replicas (G_err)"
    elif t.kind == "correlation_stationary":
        chains = t.provenance.get("replicas", "1")
        elabel = (BETWEEN_CHAIN if chains not in ("1", None) else JACKKNIFE) + " (G_err)"
    else:
        elabel = None
    if err is not None:
        note_undefined(warnings, f"{t.path.name} G_err", err, "no error estimate")
    fig, axes = new_fig(1, 2, size=(4.6, 3.4))
    if group:
        g = t[group]
        vals = list(dict.fromkeys(g))
        norm = Normalize(0, max(len(vals) - 1, 1))
        cmap = cm.viridis
        curves = [(v, g == v, cmap(norm(k))) for k, v in enumerate(vals)]
    else:
        curves = [(None, np.ones(len(r), bool), color(0))]
    for j, ax in enumerate(axes.ravel()):
        drawn = []
        for v, m, c in curves:
            if j == 1:
                m = m & (r > 0)
            yy = masked(y[m])
            if j == 1:
                yy = np.ma.masked_less_equal(yy, 0)
                drawn.append((r[m], yy))
            ax.errorbar(r[m], yy, yerr=None if err is None else err[m],
                        marker="o", ms=2.5, lw=0.9, color=c, capsize=1.5,
                        label=elabel if v is None else None, elinewidth=0.6)
        ax.set(xlabel="r", ylabel="G(r)")
        if j == 1 and log_axes(ax, drawn, warnings, t.path.name, ycol):
            ax.set(xlabel="r (r=0 not shown)")
            ax.set_title("non-positive G not shown", fontsize=7)
    if group:
        fig.colorbar(cm.ScalarMappable(norm=Normalize(0, len(curves) - 1 or 1), cmap=cmap),
                     ax=axes.ravel().tolist(),
                     label=f"{group} (curve index; {vals[0]:g} to {vals[-1]:g})")
        if err is not None:
            axes[0, 0].set_title(f"error bars: {elabel}", fontsize=7)
    elif err is not None:
        axes[0, 0].legend(loc="best")
    return [("correlation", finish(fig, t.provenance))]
