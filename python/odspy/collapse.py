"""Finite-size data collapse with the Houdayer-Hartmann quality S.

Scaling: X = (x - xc) L^(1/nu), Y = y L^(-a), Yerr = yerr L^(-a).

Quality (Houdayer & Hartmann, PRB 70, 014418, 2004): for every point of every size,
the master curve and its error at that X come from a weighted straight-line fit
through the two neighbouring (bracketing) points of each OTHER size whose X range
contains it (with one other size this is plain linear interpolation);
S = mean of (Y - master)^2 / (Yerr^2 + master_err^2) over the points that have
such neighbours. S is about 1 for a collapse consistent with the error bars.

S is only meaningful if it rests on most of the data: a collapse is marked poor when
S > 3, when S is undefined, or when the points in S are fewer than half of all points
or fewer than 3 for some size ("too little overlap"). The search cannot escape into
such low-overlap regions: there the objective is worse than anywhere with enough overlap.

A searched optimum can also be held by the measurement grid: with x_c on a measured x
value and L^(1/nu) commensurate between sizes (e.g. L = 8, 12, 16 at nu = 1), scaled points
of different sizes coincide exactly, the bracketing neighbours change there and S jumps;
the minimiser settles on the jump, and S describes the grid, not the data. The search
calls an optimum pinned when x_c lies on a measured x value (within ONGRID_TOL = 1e-4 of
the x span: looser than the minimiser's own xatol = 1e-6, so an optimum that stopped a few
1e-7 off the grid value still counts) and more than MAX_COINCIDENT = 2 pairs of scaled
points from different sizes coincide (|X_i - X_j| < COINCIDENT_TOL = 1e-6 times the span
of X over all points). A pinned
main optimum is marked poor and its errors are withheld; a bootstrap re-minimisation that
ends pinned counts as failed. Both conditions are needed: a healthy optimum typically sits
on one to three such pairs off the grid (S is piecewise smooth, and its minimum often lies
on a jump or at a corner where jumps meet). The guard cannot tell a pin at the true x_c
from a pin elsewhere: when the true x_c is itself a grid value it fires there too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .crossings import _clean
from .errors import InputError
from .sizes import SizeCurve

POOR_QUALITY = 3.0
MIN_FRACTION = 0.5  # a collapse needs at least this fraction of all points in S ...
MIN_PER_SIZE = 3  # ... and at least this many points of every size
MAX_BOOT_FAILED = 0.2  # above this fraction of failed bootstrap minimisations: no errors
MIN_SPREAD = 1e-6  # a bootstrap sd below this fraction of the initial step is not an error
COINCIDENT_TOL = 1e-6  # scaled X closer than this fraction of the X span coincide ...
MAX_COINCIDENT = 2  # ... and more than this many cross-size pairs pin the optimum ...
ONGRID_TOL = 1e-4  # ... when x_c is this close (fraction of the x span) to a measured x
PINNED = ("optimum held by coincident scaled points (x_c on a measured value with "
          "commensurate L^(1/nu)): S describes the grid, not the data; try another start")
QUALITY = ("Houdayer-Hartmann quality S (weighted line through the bracketing points "
           "of the other sizes)")


@dataclass
class CollapseResult:
    xc: float
    nu: float
    a: float
    errors: dict[str, float]  # {} when not searched
    quality: float  # Houdayer-Hartmann S (about 1 for a good collapse); nan if no overlap
    poor: bool  # S > 3 or undefined, too little overlap, or (searched) see search_collapse
    searched: bool
    fixed: list[str]  # names held fixed in the search
    method: str
    warnings: list[str] = field(default_factory=list)
    scaled: list[tuple[float, np.ndarray, np.ndarray, np.ndarray | None]] = field(
        default_factory=list)  # (size, X, Y, Yerr)
    points_used: int = 0  # points in S (with neighbours in another size)
    points_total: int = 0  # usable points of all sizes


class _Data:
    """Cleaned points of all sizes, ready for repeated evaluation of S."""

    def __init__(self, curves):
        if not isinstance(curves, (list, tuple)) or len(curves) < 2:
            raise InputError("a collapse needs at least two system sizes")
        if not all(isinstance(c, SizeCurve) for c in curves):
            raise InputError("collapse expects a list of SizeCurve")
        curves = sorted(curves, key=lambda c: c.size)
        sizes = [c.size for c in curves]
        bad = [s for s in sizes if not (math.isfinite(s) and s > 0)]
        if bad:
            raise InputError(f"system sizes must be positive and finite, got {bad}")
        if len(set(sizes)) != len(sizes):
            raise InputError(f"duplicate system sizes: {sizes}")
        if len({(c.xkey, c.column) for c in curves}) > 1:
            raise InputError("curves have different x or y columns: "
                             + ", ".join(f"{c.xkey}/{c.column}" for c in curves))
        missing = [c for c in curves if c.yerr is None]
        if missing:
            raise InputError("the collapse quality needs error bars; yerr missing for "
                             + ", ".join(f"size {c.size:g} ({c.source})" for c in missing))
        self.warnings: list[str] = []
        self.sizes = np.array(sizes, float)
        self.x, self.y, self.e = [], [], []
        for c in curves:
            x, y, e, dropped = _clean(c, True)
            if dropped:
                self.warnings.append(f"size {c.size:g}: {dropped} non-finite points excluded")
            zero = e == 0
            if zero.any():
                self.warnings.append(f"size {c.size:g}: {int(zero.sum())} points with zero "
                                     "yerr excluded (no weight in S)")
                x, y, e = x[~zero], y[~zero], e[~zero]
            if len(x) < 2:
                raise InputError(f"size {c.size:g} ({c.source}): fewer than 2 usable points")
            self.x.append(x)
            self.y.append(y)
            self.e.append(e)
        self.gid = np.concatenate([np.full(len(x), k) for k, x in enumerate(self.x)])
        self.n = len(self.gid)

    def scale(self, xc, nu, a, ys=None):
        ys = self.y if ys is None else ys
        out = []
        for L, x, y, e in zip(self.sizes, self.x, ys, self.e):
            f = L ** -a
            out.append(((x - xc) * L ** (1 / nu), y * f, e * f))
        return out

    def quality(self, xc, nu, a, ys=None):
        """(S, boolean mask of the points in S); S is nan when no point is in S."""
        sc = self.scale(xc, nu, a, ys)
        X = np.concatenate([s[0] for s in sc])
        Y = np.concatenate([s[1] for s in sc])
        E = np.concatenate([s[2] for s in sc])
        s0 = np.zeros(self.n)
        su = np.zeros(self.n)
        suu = np.zeros(self.n)
        sy = np.zeros(self.n)
        suy = np.zeros(self.n)
        has = np.zeros(self.n, bool)
        for k, (Xk, Yk, Ek) in enumerate(sc):
            inside = (X >= Xk[0]) & (X <= Xk[-1]) & (self.gid != k)
            if not inside.any():
                continue
            j = np.clip(np.searchsorted(Xk, X, side="right") - 1, 0, len(Xk) - 2)
            for m in (j, j + 1):
                w = np.where(inside, 1.0 / Ek[m] ** 2, 0.0)
                u = Xk[m] - X  # line centred at the point itself: master = intercept
                s0 += w
                su += w * u
                suu += w * u * u
                sy += w * Yk[m]
                suy += w * u * Yk[m]
            has |= inside
        if not has.any():
            return math.nan, has
        s0, su, suu, sy, suy = s0[has], su[has], suu[has], sy[has], suy[has]
        det = s0 * suu - su * su
        master = (suu * sy - su * suy) / det
        var = suu / det
        chi = (Y[has] - master) ** 2 / (E[has] ** 2 + var)
        return float(chi.mean()), has

    def shortfall(self, has):
        """Number of points missing to reach the overlap thresholds (0 = enough overlap)."""
        per_size = np.bincount(self.gid[has], minlength=len(self.sizes))
        total = max(0, math.ceil(MIN_FRACTION * self.n) - int(has.sum()))
        return total + int(np.maximum(0, MIN_PER_SIZE - per_size).sum())


def _pinned(data, xc, nu, a) -> int:
    """The number of coincident cross-size pairs if the optimum is held by the grid
    (x_c on a measured x value and more than MAX_COINCIDENT pairs), else 0."""
    x = np.concatenate(data.x)
    if not np.any(np.abs(x - xc) < ONGRID_TOL * float(np.ptp(x))):
        return 0
    pairs = _coincident_pairs(data, xc, nu, a)
    return pairs if pairs > MAX_COINCIDENT else 0


def _coincident_pairs(data, xc, nu, a) -> int:
    """Pairs of scaled points from different sizes whose X differ by less than
    COINCIDENT_TOL times the span of X over all points."""
    X = [s[0] for s in data.scale(xc, nu, a)]
    tol = COINCIDENT_TOL * float(np.ptp(np.concatenate(X)))
    n = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            n += int((np.abs(X[i][:, None] - X[j][None, :]) < tol).sum())
    return n


def _finite(name, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be finite, got {value!r}")
    return v


def _params(xc, nu, a, suffix=""):
    xc, nu, a = (_finite(n + suffix, v) for n, v in (("xc", xc), ("nu", nu), ("a", a)))
    if nu <= 0:
        raise InputError(f"nu{suffix} must be positive, got {nu:g}")
    return xc, nu, a


def _result(data, xc, nu, a, errors, searched, fixed, method, extra, poor_because=()):
    """poor_because: reasons (warnings) that make the result poor whatever S is."""
    S, has = data.quality(xc, nu, a)
    count = int(has.sum())
    warns = list(data.warnings)
    low = data.shortfall(has) > 0
    if count == 0:
        warns.append("no point has neighbours in another size at these parameters: "
                     "S undefined")
    elif count < data.n:
        warns.append(f"{data.n - count} of {data.n} points have no neighbours in another "
                     "size and are not in S")
    if low and count:
        warns.append(f"too little overlap: S rests on {count} of {data.n} points (a collapse "
                     f"needs at least {MIN_FRACTION:.0%} of the points and {MIN_PER_SIZE} "
                     "per size)")
    warns += list(poor_because) + extra
    scaled = [(float(L), X, Y, E) for L, (X, Y, E) in zip(data.sizes, data.scale(xc, nu, a))]
    poor = not math.isfinite(S) or S > POOR_QUALITY or low or bool(poor_because)
    method += f"; S from {count} of {data.n} points"
    return CollapseResult(xc, nu, a, errors, S, poor, searched, fixed, method, warns, scaled,
                          count, data.n)


def collapse(curves, xc: float, nu: float, a: float = 0.0) -> CollapseResult:
    """Scale the curves with the given (xc, nu, a) and measure the collapse quality S."""
    xc, nu, a = _params(xc, nu, a)
    data = _Data(curves)
    return _result(data, xc, nu, a, {}, False, [], QUALITY + "; parameters given", [])


MAX_RESTARTS = 5


def _minimise(objective, start, steps):
    """Nelder-Mead, restarted with a fresh simplex around the lowest point so far until a
    converged run no longer improves on the best converged one (or MAX_RESTARTS):
    S is only piecewise smooth (it jumps where the bracketing points of a size change,
    with three or more sizes) and a single simplex can stall on a jump far from the
    minimum. A run that did not
    converge is restarted, and never replaces a converged best."""
    best = None  # best run, converged runs ranked above non-converged ones
    x = np.asarray(start, float)
    for _ in range(MAX_RESTARTS + 1):
        simplex = np.array([x] + [x + np.eye(len(x))[i] * steps[i] for i in range(len(x))])
        r = minimize(objective, x, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "xatol": 1e-6, "fatol": 1e-6,
                              "maxiter": 400 * len(x), "maxfev": 800 * len(x)})
        settled = (r.success and best is not None and best.success
                   and not r.fun < best.fun - 1e-6)
        lowest = r if best is None or r.fun <= best.fun else best
        if best is None or (r.success, -r.fun) > (best.success, -best.fun):
            best = r
        if settled or np.array_equal(lowest.x, x):
            break  # converged, or a restart would repeat the same run exactly
        x = lowest.x
    return best


def search_collapse(curves, xc0: float, nu0: float, a0: float = 0.0, fit_a: bool = True,
                    n_boot: int = 200, seed: int = 0) -> CollapseResult:
    """Minimise S over (xc, nu[, a]) with Nelder-Mead from the starting values.

    The objective is S/(1+S) where the overlap thresholds hold and 1 + (missing
    points)/(all points) where they do not, so any parameters with enough overlap
    beat any without; the minimiser cannot escape into low overlap.

    The result is poor, and its errors are withheld (nan), when the search did not
    converge, when its optimum is held by the overlap limit (a move of 1% of the initial
    simplex step in xc or nu takes the overlap below the limit: there S and the
    parameters describe the limit, not the data), when the optimum is held by coincident
    scaled points (x_c on a measured x value and more than MAX_COINCIDENT cross-size pairs
    with |dX| below COINCIDENT_TOL of the X span: see the module doc), when the collapse is poor (S > 3 or
    undefined) or when the optimum xc lies outside the measured x range of all sizes
    (an extrapolation). Errors are also withheld when the bootstrap spread of a parameter
    is below MIN_SPREAD times its initial simplex step (the re-minimisations did not move).

    Errors: standard deviation of the parameters re-minimised (from the best fit) on
    `n_boot` parametric bootstrap resamples, every point drawn from N(y, yerr).
    A bootstrap minimisation that does not converge, ends with too little overlap, ends at
    the overlap limit or ends on coincident scaled points is left out and counted; if more than MAX_BOOT_FAILED of
    them fail, the survivors would understate the errors, so no errors are given (nan).
    """
    xc0, nu0, a0 = _params(xc0, nu0, a0, "0")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise InputError(f"n_boot must be an integer >= 2, got {n_boot!r}")
    if not isinstance(fit_a, (bool, np.bool_)):
        raise InputError(f"fit_a must be True or False, got {fit_a!r}")
    data = _Data(curves)
    names = ["xc", "nu", "a"] if fit_a else ["xc", "nu"]
    fixed = [] if fit_a else ["a"]

    def objective(p, ys=None):
        xc, nu = p[0], p[1]
        a = p[2] if fit_a else a0
        if not (nu > 0 and np.all(np.isfinite(p))):
            return math.inf
        S, has = data.quality(xc, nu, a, ys)
        missing = data.shortfall(has)
        if missing or not math.isfinite(S):
            return 1.0 + max(missing, 1) / data.n
        return S / (1.0 + S)

    start = np.array([xc0, nu0, a0][:len(names)])
    if not data.quality(xc0, nu0, a0)[1].any():
        raise InputError(f"no overlap between sizes at the starting values xc={xc0:g}, "
                         f"nu={nu0:g}, a={a0:g}")
    span = max(float(np.ptp(np.concatenate(data.x))), 1e-12)
    steps = np.array([0.1 * span, 0.1 * nu0, max(0.1 * abs(a0), 0.05)][:len(names)])
    probes = 0.01 * steps[:2]
    a_of = (lambda q: q[2]) if fit_a else (lambda q: a0)

    def at_limit(q):
        """True if a small move in xc or nu from q leaves too little overlap."""
        for i in (0, 1):
            for sign in (1, -1):
                r = np.array(q, float)
                r[i] += sign * probes[i]
                if r[1] <= 0 or data.shortfall(data.quality(r[0], r[1], a_of(q))[1]):
                    return True
        return False

    extra: list[str] = []
    best = _minimise(objective, start, steps)
    p = best.x
    a = float(p[2]) if fit_a else a0
    head = (QUALITY + "; Nelder-Mead minimisation of S"
            + ("" if fit_a else f" (a fixed at {a0:g})"))
    errors = {n: math.nan for n in names}
    if not best.success:
        short, whys = "search did not converge", [f"search did not converge: {best.message}"]
    elif best.fun >= 1.0:  # _result reports the too little overlap and marks it poor
        short, whys = "search ended with too little overlap", []
    elif at_limit(p):
        short = "search constrained by the overlap limit"
        whys = [short + ": a small move of xc or nu leaves fewer than "
                f"{MIN_FRACTION:.0%} of the points (or {MIN_PER_SIZE} per size) in S, so "
                "the optimum and S describe the limit, not the data"]
    else:  # these reasons can hold together: every one is reported
        shorts, whys = [], []
        pairs = _pinned(data, float(p[0]), float(p[1]), a)
        if pairs:
            shorts.append("optimum held by coincident scaled points")
            whys.append(f"{PINNED} ({pairs} pairs of points of different sizes coincide in X "
                        f"at the optimum; more than {MAX_COINCIDENT} pin it)")
        S = data.quality(float(p[0]), float(p[1]), a)[0]
        lo, hi = (float(f(np.concatenate(data.x))) for f in (np.min, np.max))
        if not lo <= p[0] <= hi:
            shorts.append("x_c outside the measured range")
            whys.append(f"x_c = {p[0]:.6g} outside the measured range [{lo:.6g}, {hi:.6g}]: "
                        "extrapolated")
        elif not math.isfinite(S) or S > POOR_QUALITY:
            shorts.append(f"poor collapse (S = {S:.3g})")
            whys.append(f"errors withheld: poor collapse (S = {S:.3g})")
        short = "; ".join(shorts)
    if short:
        return _result(data, float(p[0]), float(p[1]), a, errors, True, fixed,
                       head + f"; errors withheld: {short}", extra, whys)
    rng = np.random.default_rng(seed)
    boot, failed = [], 0
    for _ in range(int(n_boot)):
        ys = [rng.normal(y, e) for y, e in zip(data.y, data.e)]
        r = _minimise(lambda q: objective(q, ys), p, steps)
        if (r.success and np.all(np.isfinite(r.x)) and r.fun < 1.0 and not at_limit(r.x)
                and not _pinned(data, r.x[0], r.x[1], a_of(r.x))):
            boot.append(r.x)
        else:
            failed += 1
    if failed > MAX_BOOT_FAILED * n_boot:
        extra.append(f"{failed} of {n_boot} bootstrap minimisations did not converge, "
                     "ended at or below the overlap limit or on coincident scaled points "
                     f"(more than {MAX_BOOT_FAILED:.0%}): "
                     "errors not given, the converged ones would understate them")
    elif len(boot) < 2:
        extra.append("too few converged bootstrap minimisations for errors")
    else:
        sd = np.std(np.array(boot), axis=0, ddof=1)
        flat = [n for n, v, st in zip(names, sd, steps) if not v >= MIN_SPREAD * st]
        if flat:
            extra.append(f"errors withheld: the bootstrap spread of {', '.join(flat)} is below "
                         f"{MIN_SPREAD:g} of the initial search step (the re-minimisations "
                         "did not move), which is not an error estimate")
        else:
            errors = {n: float(v) for n, v in zip(names, sd)}
    if failed and failed <= MAX_BOOT_FAILED * n_boot:
        extra.append(f"{failed} of {n_boot} bootstrap minimisations did not converge, "
                     "ended at or below the overlap limit or on coincident scaled points, "
                     "and are not in the errors")
    method = head + (f"; errors: sd over parametric bootstrap (n={n_boot}, {failed} failed) "
                     "re-minimisations")
    return _result(data, float(p[0]), float(p[1]), a, errors, True, fixed, method, extra)
