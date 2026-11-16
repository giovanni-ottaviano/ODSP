"""Shared look of the figures: palette, error-bar labels, captions, saving."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..io import Provenance, Table

# Okabe-Ito, colour-blind safe
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
SE = "± standard error"
JACKKNIFE = "± jackknife error"
BETWEEN_CHAIN = "± between-chain error"
SD_RT = "± sd/√replicas"
POISSON = "± √count (Poisson)"
BINOMIAL = "± √(p(1−p)/R)"
WILSON = "95% Wilson interval where p = 0 or 1"
COARSENING = ("length", "rho", "domains", "persistence")

_RC = {"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
       "axes.spines.right": False, "font.size": 9, "legend.fontsize": 7.5,
       "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=PALETTE)}


def warn(warnings, text: str) -> None:
    """Append `text` to the caller's warning list, if it passed one."""
    if warnings is not None:
        warnings.append(text)


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def new_fig(nrows: int = 1, ncols: int = 1, size=(4.2, 3.2), **kw):
    """A figure with a grid of axes (always a 2D array), sized per panel."""
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                                 layout="constrained", squeeze=False, **kw)
    return fig, axes


def grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(n, max_cols)
    return math.ceil(n / cols), cols


def hide_unused(axes, n: int) -> None:
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)


def masked(y) -> np.ma.MaskedArray:
    """NaN and inf become masked values, so lines show gaps and are never interpolated."""
    return np.ma.masked_invalid(np.asarray(y, float))


def numeric(t: Table, name: str):
    col = t[name]
    return None if col.dtype == object else col


def caption(fig: Figure, prov: Provenance | None) -> None:
    text = prov.caption() if prov is not None else ""
    if not text:
        return
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, 0.24 / fig.get_figheight(), 1, 1 - 0.24 / fig.get_figheight()))
    fig.text(0.5, 0.008, text, ha="center", va="bottom", fontsize=6.5, color="#555555")


def finish(fig: Figure, prov: Provenance | None, title: str | None = None) -> Figure:
    if title:
        fig.suptitle(title, fontsize=10)
    caption(fig, prov)
    return fig


def save(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write <stem>.pdf and <stem>.png into out_dir (created), then close the figure. If
    either cannot be written, neither is left behind and the error is raised."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            try:
                fig.savefig(p, dpi=150 if ext == "png" else None)
            except Exception:
                for q in paths + [p]:      # no half-written or orphan files of this figure
                    q.unlink(missing_ok=True)
                raise
            paths.append(p)
    finally:
        plt.close(fig)
    return paths


def note_undefined(warnings, ctx: str, err, reason: str) -> int:
    """Count the NaN entries of an error array and say so; returns the count."""
    n = int(np.isnan(np.asarray(err, float)).sum()) if err is not None else 0
    if n and warnings is not None:
        warnings.append(f"{ctx}: error undefined for {n} points ({reason}); drawn without "
                        "an error bar")
    return n


def positive(x, y, warnings, ctx: str, what: str):
    """x, y with non-positive/NaN y and non-positive x masked for log axes; returns (x, y, n_hidden) where
    n_hidden counts finite points that cannot be drawn on a log axis. Warns when n_hidden > 0."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    bad = finite & (y <= 0) & (x > 0)      # t = 0 is announced in the axis label
    n = int(bad.sum())
    if n and warnings is not None:
        warnings.append(f"{ctx}: {n} {what} values <= 0 not shown on log axes")
    keep = finite & ~bad & (x > 0)
    return np.ma.masked_where(~keep, x), np.ma.masked_where(~keep, y), n


def _drawable(x, y) -> bool:
    """True if some unmasked finite point has x > 0 and y > 0."""
    x = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(x, float)), np.nan)
    y = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(y, float)), np.nan)
    with np.errstate(invalid="ignore"):
        return bool(np.any(np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)))


def log_axes(ax, pairs, warnings, ctx: str, what: str) -> bool:
    """Log-log axes for `ax` if any of the (x, y) pairs drawn on it has a point with
    x > 0 and y > 0; otherwise the panel stays linear (log axes would have nothing to
    show, and matplotlib cannot draw them), says so on the panel and warns. Returns True
    when log axes were set."""
    if any(_drawable(x, y) for x, y in pairs):
        ax.set(xscale="log", yscale="log")
        return True
    if warnings is not None:
        warnings.append(f"{ctx}: {what}: no values > 0 to show on log axes; that panel is "
                        "left empty, on linear axes")
    ax.text(0.5, 0.5, "no values > 0:\nnothing to show on log axes", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="#555555")
    return False


def binomial_yerr(p, R):
    """Error bars of binomial fractions p out of R trials as a (2, n) array (below, above):
    √(p(1−p)/R) where 0 < p < 1, the 95% Wilson interval where p is 0 or 1 (where that
    error would be 0); NaN where R is not known. Returns (yerr, number of Wilson bars)."""
    from ..helpers import wilson_interval
    p = np.atleast_1d(np.asarray(p, float))
    R = np.broadcast_to(np.asarray(R, float), p.shape)
    err = np.full((2, p.size), np.nan)
    n_edge = 0
    for i, (pi, ri) in enumerate(zip(p, R)):
        if not (np.isfinite(pi) and np.isfinite(ri) and ri >= 1):
            continue
        if 0 < pi < 1:
            err[:, i] = math.sqrt(pi * (1 - pi) / ri)
        else:
            lo, hi = wilson_interval(int(round(pi * ri)), int(round(ri)))
            err[:, i] = (max(0.0, pi - lo), max(0.0, hi - pi))
            n_edge += 1
    return err, n_edge
