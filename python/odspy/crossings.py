"""Crossings of the curves of consecutive system sizes (e.g. Binder cumulant)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .errors import InputError, rng_from
from .sizes import SizeCurve

NO_CROSSING = "no crossing in range"
MAX_LOST_FRACTION = 0.05


@dataclass
class Crossing:
    size_a: float
    size_b: float
    x: float  # nan when not found
    err: float  # nan when not found or without error bars
    y: float  # nan when not found
    found: bool
    note: str  # "" or "no crossing in range"
    method: str
    n_boot: int
    warnings: list[str] = field(default_factory=list)


def _method(n_boot: int) -> str:
    if n_boot:
        return f"linear interpolation, parametric bootstrap (n={n_boot})"
    return "linear interpolation, no error estimate"


def _clean(c: SizeCurve, use_err: bool):
    """Finite points of a curve sorted by x; returns (x, y, yerr or None, number excluded).

    Raises InputError for a negative yerr or a repeated x within the curve.
    """
    x, y = np.asarray(c.x, float), np.asarray(c.y, float)
    where = f"size {c.size:g} ({c.source})"
    err_all = None if c.yerr is None else np.asarray(c.yerr, float)
    if err_all is not None:
        neg = np.flatnonzero(err_all < 0)
        if len(neg):
            raise InputError(f"{where}: negative yerr {err_all[neg[0]]:g} at "
                             f"x={x[neg[0]]:g} ({len(neg)} negative values)")
    good = np.isfinite(x) & np.isfinite(y)
    err = None
    if use_err and err_all is not None:
        err = err_all
        good &= np.isfinite(err)
    order = np.argsort(x[good], kind="stable")
    x, y = x[good][order], y[good][order]
    if err is not None:
        err = err[good][order]
    rep = np.flatnonzero(np.diff(x) == 0)
    if len(rep):
        raise InputError(f"{where}: duplicate x value {x[rep[0]]:g} within one curve")
    return x, y, err, int((~good).sum())


def _grid_weights(x: np.ndarray, grid: np.ndarray):
    """Index and weight so that y(grid) = y[i]*(1-w) + y[i+1]*w (linear interpolation)."""
    i = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(x) - 2)
    dx = x[i + 1] - x[i]
    w = np.where(dx > 0, (grid - x[i]) / np.where(dx > 0, dx, 1.0), 0.0)
    return i, np.clip(w, 0.0, 1.0)


def _locate(grid: np.ndarray, d: np.ndarray) -> list[float]:
    """x of each sign change of d(grid); a node where d is exactly zero counts once
    if the sign really changes there (or it lies on the edge)."""
    out = []
    n = len(grid)
    for i in range(n):
        if d[i] == 0:
            if i > 0 and d[i - 1] == 0:
                continue
            if 0 < i < n - 1:
                nxt = next((d[j] for j in range(i + 1, n) if d[j] != 0), None)
                if nxt is None or d[i - 1] * nxt > 0:
                    continue
            out.append(float(grid[i]))
        elif i < n - 1 and d[i] * d[i + 1] < 0:
            out.append(float(grid[i] - d[i] * (grid[i + 1] - grid[i]) / (d[i + 1] - d[i])))
    return out


def _boot_nearest(grid: np.ndarray, d: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """For each draw (row of d), the crossing nearest each original crossing x0; nan if none.

    Returns an array (n_draws, len(x0)).
    """
    a, b = d[:, :-1], d[:, 1:]
    flip = (a * b < 0) | (a == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = grid[:-1] - a * (grid[1:] - grid[:-1]) / (b - a)
    xc = np.where(a == 0, grid[:-1], xc)
    xc = np.where(flip, xc, np.nan)
    out = np.full((d.shape[0], len(x0)), np.nan)
    has = flip.any(axis=1)
    for k, c in enumerate(x0):
        dist = np.where(flip, np.abs(xc - c), np.inf)
        j = dist.argmin(axis=1)
        out[has, k] = xc[np.arange(d.shape[0]), j][has]
    return out


def _pair(a: SizeCurve, b: SizeCurve, n_boot: int, rng) -> list[Crossing]:
    warns: list[str] = []
    have_err = a.yerr is not None and b.yerr is not None
    if not have_err:
        warns.append("yerr missing for "
                     + " and ".join(f"size {c.size:g}" for c in (a, b) if c.yerr is None)
                     + ": no error estimate")
    xa, ya, ea, da = _clean(a, have_err)
    xb, yb, eb, db = _clean(b, have_err)
    for c, dropped in ((a, da), (b, db)):
        if dropped:
            warns.append(f"size {c.size:g}: {dropped} non-finite points excluded")
    nb = n_boot if have_err else 0

    def none(extra=None):
        return [Crossing(a.size, b.size, math.nan, math.nan, math.nan, False, NO_CROSSING,
                         _method(nb), nb, warns + ([extra] if extra else []))]

    for c, x in ((a, xa), (b, xb)):
        if len(x) < 2:
            return none(f"size {c.size:g}: fewer than 2 usable points")
    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    if not lo < hi:
        return none("x ranges do not overlap")
    grid = np.unique(np.concatenate([xa, xb]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    ia, wa = _grid_weights(xa, grid)
    ib, wb = _grid_weights(xb, grid)

    def diff(Ya, Yb):
        return (Ya[..., ia] * (1 - wa) + Ya[..., ia + 1] * wa
                - Yb[..., ib] * (1 - wb) - Yb[..., ib + 1] * wb)

    d = diff(ya, yb)
    found = _locate(grid, d)
    if not found:
        return none()
    ygrid = ya[ia] * (1 - wa) + ya[ia + 1] * wa
    out = []
    errs = [math.nan] * len(found)
    if have_err:
        draws_a = rng.normal(ya, ea, size=(n_boot, len(ya)))
        draws_b = rng.normal(yb, eb, size=(n_boot, len(yb)))
        near = _boot_nearest(grid, diff(draws_a, draws_b), np.array(found))
        for k in range(len(found)):
            vals = near[:, k]
            ok = np.isfinite(vals)
            lost = n_boot - int(ok.sum())
            if lost / n_boot > MAX_LOST_FRACTION:
                warns.append(f"crossing at x={found[k]:.6g}: {lost} of {n_boot} bootstrap "
                             "draws had no crossing and are not in err")
            if ok.sum() >= 2:
                errs[k] = float(np.std(vals[ok], ddof=1))
            else:
                warns.append(f"crossing at x={found[k]:.6g}: too few bootstrap draws "
                             "with a crossing for an error")
    for k, xc in enumerate(found):
        yc = float(np.interp(xc, grid, ygrid))
        mine = [w for w in warns if not w.startswith("crossing at") or f"x={found[k]:.6g}:" in w]
        out.append(Crossing(a.size, b.size, xc, errs[k], yc, True, "", _method(nb), nb, mine))
    return out


def find_crossings(curves: list[SizeCurve], n_boot: int = 2000, seed: int = 0) -> list[Crossing]:
    """Crossings of the curves of consecutive sizes (one per sign change, each reported).

    Both curves are linearly interpolated onto the merged x grid of their overlap; a
    crossing outside the overlap is never extrapolated. The error is the standard
    deviation over `n_boot` parametric bootstrap draws N(y, yerr).
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise InputError(f"n_boot must be an integer >= 2, got {n_boot!r}")
    if len(curves) < 2:
        raise InputError("crossings need at least two system sizes")
    curves = sorted(curves, key=lambda c: c.size)
    sizes = [c.size for c in curves]
    if len(set(sizes)) != len(sizes):
        raise InputError(f"duplicate system sizes: {sizes}")
    if len({(c.xkey, c.column) for c in curves}) > 1:
        raise InputError("curves have different x or y columns: "
                         + ", ".join(f"{c.xkey}/{c.column}" for c in curves))
    rng = rng_from(seed)
    out: list[Crossing] = []
    for a, b in zip(curves, curves[1:]):
        out.extend(_pair(a, b, int(n_boot), rng))
    return out


def crossings_plain(tables: dict[float, list[dict]], xkey: str, ykey: str
                    ) -> list[tuple[float, float, float]]:
    """Legacy: exact transcription of benchmarks/analyse.py crossings() (pairs rows by index)."""
    Ls = sorted(tables)
    out = []
    for a, b in zip(Ls, Ls[1:]):
        xa = [r[xkey] for r in tables[a]]
        da = [r[ykey] - s[ykey] for r, s in zip(tables[a], tables[b])]
        for i in range(len(xa) - 1):
            if da[i] == 0 or da[i] * da[i + 1] < 0:
                out.append((a, b, xa[i] - da[i] * (xa[i + 1] - xa[i]) / (da[i + 1] - da[i])))
    return out
