"""Finite-size scaling of a 2D critical point from stationary temperature sweeps, shared by
validation/models/ising.py, potts.py and majorityvote.py (a leading underscore: not a model
module itself). The control parameter is the temperature unless the model names another
(Model.control: the majority-vote noise q); "T" below stands for it, and the ordered phase
is on its low side for every model here.

Runs: one stationary sweep per lattice size L (--dims LxL --sweep temperature=...), each split
into one job per temperature. The temperatures of size L are T_c + x L^(-1/nu) for one list
of x (the scaling variable, Model.grid): the window around T_c shrinks with L as the critical
region does, so every size has the same points across the crossing, the 3 points of the slope
at T_c and the susceptibility peak. A common grid for all sizes (--sweep dims=... --sweep2
temperature=...) would need the largest size's resolution at every size and so waste most of
the runs. x = 0 is T_c itself (exactly), so |x|(T_c) is read without interpolation, and the
slope at T_c comes from the 3 symmetric points x = 0, +-s. Since the points sit at the same
x in every size, the parabola of a peak and the curvature of U in the slope enter every size
alike and cannot bias an exponent.

The collapse (Model.exact_x set) cannot use such points: its quality S compares the scaled
points X = (T - x_c) L^(1/nu) of different sizes, and where many of them coincide exactly S
jumps and the search settles on the jump (odspy.collapse: "optimum held by coincident scaled
points"). Points at the same x coincide in every size at the exact (T_c, nu) itself -- the
search then returns T_c to 1e-9 (seen in the small run) -- and any per-size factor or shift
of the same list of x only moves that coincidence: T is affine in x and so is X, so two such
grids coincide at every point for one (x_c, nu), which lies within the statistical errors of
the exact values for factors small enough not to disturb the other estimators. So the
points the other estimators read (Model.exact_x: T_c, the slope points, ...) stay exactly
scaled and are left out of the collapse, and every other point gets its own offset of up to
+-OFFSET in x (a fixed quasi-random sequence, quadratic in the point's index, so the offsets
of two sizes are not affinely related). Then no (x_c, nu) maps more than 2 of one size's
collapse points onto points of another (validation/tests/test_models_ising.py checks it),
and the near-coincidences are spread over +-OFFSET in x, wider than the statistical
resolution of x_c and nu. The offsets are fixed and known; every point is read at its own T.

Fit ranges (the campaign rule, fixed before the full run, the same for every model with
finite-size-scaling fits): every exponent fit and the collapse use all sizes except the
smallest (L >= the second size, Stats.fit_from); the T_c / Binder crossing is that of the two
largest sizes. The fits and the collapse over all sizes are computed as well, from the same
data, and their values go into the rows' warnings as a finite-size diagnostic.

Statistics per point: `chains` independent chains (errors leave-one-chain-out); burn-in,
interval and samples per chain from the measured integrated autocorrelation time of |x| at
T_c, tau(L) = A L^z sweeps (Model.tau): burn-in `burn_tau` tau, interval ~tau/5 and samples
for about `n_eff` effectively independent samples per point over all chains (2 tau per
independent sample). burn_in_caveat checks the tau model against the data: a point whose
burn-in is shorter than BURN_TAU_MIN measured tau(|x|) is named in the warnings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from odspy import (SizeCurve, families, find_crossings, find_outputs, fit,
                   peak_vs_size, search_collapse, slope_at_crossing_vs_size)
from odspy.helpers import stationary_caveats
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import exponent_check, plot_exponent, save_figure
from validation.core import Check, Run

N_BOOT = 2000           # parametric bootstrap draws (crossings, slopes, peaks)
N_BOOT_COLLAPSE = 200   # bootstrap re-minimisations of the collapse
# The largest offset in x of the points that are not in Model.exact_x (module doc).
OFFSET = 0.2
# The campaign rule for fit ranges (module doc), quoted in every row it applies to.
FIT_RULE = "smallest size excluded by the campaign rule, fixed before the full run"
# Burn-in below this many measured tau(|x|) at a point is named in the warnings.
BURN_TAU_MIN = 20.0


def scaled_grid(tc: float, nu: float, L: int, xs: Sequence[float]) -> tuple[float, ...]:
    """Temperatures T_c + x L^(-1/nu) for each x; x = 0 gives T_c exactly."""
    step = float(L) ** (-1.0 / nu)
    return tuple(tc if x == 0 else tc + x * step for x in xs)


def grid_x(xs: Sequence[float], index: int, exact: Sequence[float] | None) -> tuple[float, ...]:
    """The scaling variable of each point of the index-th size (module doc): the x in `exact`
    (all of them when None) as they are, every other x plus its own offset."""
    out = []
    for i, x in enumerate(xs):
        if exact is None or x in exact:
            out.append(x)
        else:
            # quadratic in i: the offsets of two sizes must not differ by a constant or by a
            # multiple of x, which would leave the grids affinely related
            u = (0.6180339887 * (i + 1) + 0.4142135624 * (index + 1) * (i + 1) ** 2) % 1.0
            out.append(x + OFFSET * (2.0 * u - 1.0))
    return tuple(out)


def size_grid(model: Model, L: int, index: int) -> tuple[float, ...]:
    """The temperatures of size L, the index-th size of its scale."""
    return scaled_grid(model.tc, model.nu, L, grid_x(model.grid, index, model.exact_x))


@dataclass(frozen=True)
class Stats:
    """One scale of a model's finite-size scaling runs."""
    sizes: tuple[int, ...]
    chains: int
    n_eff: int          # effectively independent samples per point, all chains together
    burn_tau: float     # burn-in per chain, in units of tau(L)
    threads: int        # per job (the chains of one point share them)

    @property
    def fit_from(self) -> int:
        """The smallest L in the exponent fits and the collapse: the second size (the
        campaign rule, module doc)."""
        return sorted(self.sizes)[1]


@dataclass(frozen=True)
class Model:
    """What the shared finite-size-scaling analysis needs to know about one model."""
    name: str                       # "ising" / "potts"
    args: tuple[str, ...]           # odsp options of the model (no dims, temperature, task)
    observable: str                 # the order parameter: "m" / "order"
    tc: float
    nu: float
    gamma_nu: float
    beta_nu: float
    tc_source: str                  # theory.CITATIONS keys
    exp_source: str
    grid: tuple[float, ...]         # x values of the scaling variable (must contain 0)
    tau: tuple[float, float]        # tau(L) = A L^z sweeps at T_c, |x| (measured)
    scales: dict[str, Stats]
    seed: int
    title: str                      # e.g. "2D Ising, Metropolis"
    order_label: str                # y label of the order parameter
    corrections: str                # what is known about finite-size corrections (warnings)
    # gamma/nu from chi' = N(<x^2> - <|x|>^2): None -- odspy.peak_vs_size (parabola through
    # the 3 highest points); a grid value x -- chi' at that fixed scaling variable, power fit
    # (for a maximum too flat for the 3-point parabola, see ising.py).
    gamma_at: float | None = None
    gamma_note: str = ""            # with gamma_at: the estimator it replaces and why (warnings)
    # The x placed exactly (same scaled position in every size: read by the crossing, slope,
    # |x|(T_c), ...), the others offset and used by the collapse; None: all exact, no
    # collapse.
    exact_x: tuple[float, ...] | None = None
    control: str = "temperature"    # the swept odsp parameter
    control_label: str = "T"        # its symbol in rows and figures
    # The uncertainty of tc: 0 for an exact critical point; for a measured one (majority vote)
    # it is the predicted error of <name>.tc, and its effect on the exponents read at tc goes
    # into their warnings (critical_point_shift).
    tc_error: float = 0.0

    @property
    def tc_kind(self) -> str:
        """"exact" or "cited" (a measured critical point), as the rows call tc."""
        return "exact" if self.tc_error == 0 else "cited"


def tau_of(model: Model, L: int) -> float:
    A, z = model.tau
    return A * float(L) ** z


def point_stats(model: Model, s: Stats, L: int) -> tuple[int, int, int]:
    """(burn-in, samples per chain, interval) of every point of size L."""
    tau = tau_of(model, L)
    interval = max(2, int(round(tau / 5.0)))
    samples = max(200, int(math.ceil(2.0 * s.n_eff * tau / (s.chains * interval))))
    burn = max(200, int(math.ceil(s.burn_tau * tau)))
    return burn, samples, interval


def run_name(L: int) -> str:
    return f"fss_L{L}"


def fss_runs(model: Model, scale: str) -> list[Run]:
    s = model.scales[scale]
    out = []
    for i, L in enumerate(s.sizes):
        burn, samples, interval = point_stats(model, s, L)
        temps = size_grid(model, L, i)
        args = model.args + ("--dims", f"{L}x{L}", "--task", "stationary", "--observable",
                             model.observable, "--burn-in", str(burn), "--samples", str(samples),
                             "--interval", str(interval), "--replicas", str(s.chains),
                             "--sweep", f"{model.control}=" + ",".join(f"{T:.12g}" for T in temps))
        out.append(Run(run_name(L), args, seed=model.seed + L, threads=s.threads))
    return out


# ---------------------------------------------------------------------------------------------

def sweep_table(prefix: Path) -> Table:
    table = find_outputs(prefix).get("sweep")
    if table is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    return table


def crossing_with_value_error(a: SizeCurve, b: SizeCurve, target: float, n_boot: int = N_BOOT,
                              seed: int = 0):
    """The crossing of curves a and b nearest `target` (odspy.find_crossings, with its
    bootstrap error in x) and the parametric-bootstrap error of the curves' value there:
    each draw takes every point from N(y, yerr), finds the drawn curves' crossing nearest
    the measured one and reads its y. Returns (Crossing, y error); (Crossing, nan) when no
    crossing was found."""
    found = [c for c in find_crossings([a, b], n_boot=n_boot, seed=seed) if c.found]
    if not found:
        return find_crossings([a, b], n_boot=2, seed=seed)[0], math.nan
    best = min(found, key=lambda c: abs(c.x - target))
    rng = np.random.default_rng(seed)
    ys = []
    for _ in range(n_boot):
        drawn = [SizeCurve(c.size, c.label, c.x, np.asarray(c.y, float)
                           + np.asarray(c.yerr, float) * rng.standard_normal(len(c.y)),
                           c.yerr, c.xkey, c.column, c.source) for c in (a, b)]
        hits = [c for c in find_crossings(drawn, n_boot=2, seed=0) if c.found]
        if hits:
            ys.append(min(hits, key=lambda c: abs(c.x - best.x)).y)
    y_err = float(np.std(ys, ddof=1)) if len(ys) >= 0.95 * n_boot else math.nan
    return best, y_err


def _collapse_points(curves: list[SizeCurve], model: Model, sizes: Sequence[int]) -> list[SizeCurve]:
    """The curves at their offset points only (not in Model.exact_x; module doc)."""
    out = []
    for c in curves:
        grid = size_grid(model, int(c.size), list(sizes).index(int(c.size)))
        T = [t for t, x in zip(grid, model.grid) if x not in model.exact_x]
        xs = np.asarray(c.x, float)
        keep = np.array([np.min(np.abs(np.asarray(T) - t)) < 1e-9 for t in xs])
        if keep.sum() != len(T):
            raise RuntimeError(f"{c.source}: the offset points of the grid are not all in the data")
        out.append(SizeCurve(c.size, c.label, xs[keep], np.asarray(c.y)[keep],
                             None if c.yerr is None else np.asarray(c.yerr)[keep], c.xkey,
                             c.column, c.source))
    return out


def _at_x(curves: list[SizeCurve], model: Model, sizes: Sequence[int], x: float = 0.0):
    """(sizes, values, errors) of each curve at its grid point of scaling variable x (x = 0:
    T_c itself), located by the temperature fss_runs gave it."""
    L, y, e = [], [], []
    for c in curves:
        i_size = list(sizes).index(int(c.size))
        T = size_grid(model, int(c.size), i_size)[list(model.grid).index(x)]
        xs = np.asarray(c.x, float)
        i = int(np.argmin(np.abs(xs - T)))
        if abs(xs[i] - T) > 1e-9:
            raise RuntimeError(f"{c.source}: no point at T = {T:.12g} (x = {x:g})")
        L.append(c.size); y.append(float(c.y[i])); e.append(float(c.yerr[i]))
    return np.array(L), np.array(y), np.array(e)


def local_exponents(sizes, values, errors) -> str:
    """Effective exponents ln(y2/y1) / ln(L2/L1) of successive sizes, with their errors: how
    the exponent drifts with L (finite-size corrections)."""
    sizes, values, errors = (np.asarray(v, float) for v in (sizes, values, errors))
    parts = []
    for i in range(len(sizes) - 1):
        r = math.log(sizes[i + 1] / sizes[i])
        b = math.log(abs(values[i + 1] / values[i])) / r
        e = math.hypot(errors[i] / values[i], errors[i + 1] / values[i + 1]) / r
        parts.append(f"{sizes[i]:g}/{sizes[i + 1]:g}: {b:.3f} ± {e:.2g}")
    return "; ".join(parts)


def _refit(r, fit_from: float):
    """The power fit of r's points (those r used) restricted to L >= fit_from."""
    y = np.where(r.used, np.asarray(r.y, float), np.nan)
    return fit(r.x, y, r.yerr, "power", xrange=(fit_from, math.inf))


@dataclass
class Data:
    """The merged runs of one model, read once."""
    model: Model
    scale: str
    stats: Stats
    prefixes: list[Path]
    tables: list[Table]
    binder: list[SizeCurve]
    order: list[SizeCurve]
    chi: list[SizeCurve]
    caveats: list[str]

    @property
    def sizes_text(self) -> str:
        return "L=" + ",".join(str(L) for L in self.stats.sizes)

    def statistics(self, what: str) -> str:
        s = self.stats
        return (f"{s.chains} chains per point, ~{s.n_eff} independent samples per point "
                f"(burn-in {s.burn_tau:g} tau); errors leave-one-chain-out; {what}")


def burn_in_caveat(table: Table) -> str | None:
    """The points of a stationary sweep table whose burn-in (sweeps) is shorter than
    BURN_TAU_MIN tau(|x|) as measured there (tau_abs, in samples, times the interval): a
    check of the tau model that set the burn-in. None when every point passes."""
    config = table.provenance.config
    burn, interval = float(config.get("burn-in", "nan")), float(config.get("interval", "nan"))
    tau = np.asarray(table["tau_abs"], float) * interval
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = burn / tau
    bad = np.flatnonzero(np.isfinite(ratio) & (ratio < BURN_TAU_MIN))
    if bad.size == 0:
        return None
    key = table.names[0]
    points = ", ".join(f"{key}={float(table[key][i]):.6g}" for i in bad)
    return (f"{table.path.name}: burn-in of {burn:g} sweeps is less than {BURN_TAU_MIN:g} "
            f"measured tau(|x|) at {points} (burn-in / tau down to {np.min(ratio[bad]):.1f}): "
            "the tau model underestimates tau there")


def read(model: Model, scale: str, data: Path) -> Data:
    s = model.scales[scale]
    prefixes = [data / run_name(L) for L in s.sizes]
    tables = [sweep_table(p) for p in prefixes]
    caveats = list(dict.fromkeys(c for t in tables for c in stationary_caveats(t)))
    caveats += [c for c in (burn_in_caveat(t) for t in tables) if c is not None]
    # chi' = N (<x^2> - <|x|>^2): the plain N (<x^2> - <x>^2) of the Ising m has no peak in a
    # finite system (m changes sign), and near and below T_c the pooled <x> of chains whose
    # sign flips rarely is not 0, which biases it low by ~ 2 tau(x) / (all samples), a bias
    # that grows with L; chi' depends on |x| only. For the Potts order parameter (>= 0) the
    # two are the same number.
    return Data(model, scale, s, prefixes, tables, families(prefixes, "binder"),
                families(prefixes, "abs_mean"), families(prefixes, "abs_susceptibility"),
                caveats)


def _curves_axes(ax, curves: list[SizeCurve], ylabel: str, m: Model) -> None:
    for i, c in enumerate(curves):
        ax.errorbar(c.x, c.y, yerr=c.yerr, fmt="o-", ms=3, lw=0.8, color=color(i), capsize=0,
                    label=f"L = {c.size:g}")
    c = m.control_label
    ax.axvline(m.tc, color="#555555", lw=1, ls="--", label=f"{m.tc_kind} {c}_c = {m.tc:.6g}")
    ax.set(xlabel=c, ylabel=ylabel)
    ax.legend(fontsize=6.5)


def crossing_checks(d: Data, figures: Path, binder_target=None,
                    binder_note: str = "") -> tuple[list[Check], float]:
    """<name>.tc (<name>.qc for the noise q; and <name>.binder when `binder_target` = (U*,
    error, source) is given, with `binder_note` in its warnings): the Binder crossing of the
    two largest sizes. Returns the checks and the crossing T."""
    m = d.model
    sym = m.control_label
    curves = d.binder
    pairs = [(curves[i], curves[i + 1]) for i in range(len(curves) - 1)]
    # Each pair's crossing (nearest the previous pair's, starting from the centre of the
    # grid of the smallest pair) and the value of U there, for the drift.
    rows = []
    target = float(np.median(curves[0].x))
    for k, (a, b) in enumerate(pairs):
        c, y_err = crossing_with_value_error(a, b, target, N_BOOT, seed=k)
        rows.append((c, y_err))
        if c.found:
            target = c.x
    best, y_err = rows[-1]
    a, b = pairs[-1]
    if not best.found or not math.isfinite(best.err):
        raise RuntimeError(f"{m.name}: the Binder curves of L = {a.size:g} and {b.size:g} do "
                           f"not cross (with an error) inside the scanned {m.control} range")
    warns: list[str] = []
    # how often the two largest cross (where they cross does not depend on the bootstrap)
    n_pair = sum(1 for c in find_crossings([a, b], n_boot=2, seed=0) if c.found)
    if n_pair > 1:
        warns.append(f"L = {a.size:g}/{b.size:g} curves cross {n_pair} times; the crossing "
                     "nearest the previous pair's is used")
    warns += best.warnings
    drift = "; ".join(f"{c.size_a:g}/{c.size_b:g}: " + (f"{sym} = {c.x:.5f} ± {c.err:.2g}, "
                                                        f"U = {c.y:.4f} ± {e:.2g}" if c.found
                                                        else "none")
                      for c, e in rows)
    warns.append(f"crossings of every pair (finite-size drift): {drift}")
    warns.append(m.corrections)
    warns += d.caveats
    statistics = d.statistics(f"crossing errors from {N_BOOT} parametric bootstrap draws")
    checks = []

    def figure(name: str, which: str, value: float, value_err: float, label: str) -> str:
        fig, axes = new_fig(1, 2)
        a0, a1 = axes[0]
        _curves_axes(a0, curves, "Binder cumulant U", m)
        a0.errorbar([best.x], [best.y], xerr=[best.err], fmt="*", ms=9, color="k", capsize=2,
                    label="crossing of the two largest")
        a0.legend(fontsize=6.5)
        good = [(c, e) for c, e in rows if c.found]
        inv = [1.0 / c.size_b for c, _ in good]
        ys = [c.x if which == "x" else c.y for c, _ in good]
        es = [c.err if which == "x" else e for c, e in good]
        a1.errorbar(inv, ys, yerr=es, fmt="o", ms=4, color=color(0), capsize=0,
                    label="crossing of L and the next size")
        a1.axhline(value, color=color(1), lw=1, label=label)
        if value_err > 0:
            a1.axhspan(value - value_err, value + value_err, color=color(1), alpha=0.2, lw=0)
        a1.set(xlabel="1 / (larger L of the pair)", ylabel=f"{sym} at the crossing"
               if which == "x" else "U at the crossing")
        a1.set_xlim(left=0)
        a1.legend(fontsize=6.5)
        finish(fig, d.tables[-1].provenance, f"{m.title}: Binder crossings")
        return save_figure(fig, figures, name)

    name = f"{m.name}.{sym.lower()}c"   # ising.tc, mv.qc
    checks.append(Check(model=m.name, check=name,
                        quantity=f"{sym}_c from the Binder crossing of L = {a.size:g} and "
                                 f"{b.size:g}",
                        measured=best.x, error=best.err, predicted=m.tc,
                        predicted_error=m.tc_error, source=m.tc_source, sizes=d.sizes_text,
                        statistics=statistics, warnings=list(warns),
                        figure=figure(name, "x", m.tc, m.tc_error,
                                      f"{m.tc_kind} {sym}_c = {m.tc:.6g}")))
    if binder_target is not None:
        U, U_err, source = binder_target
        if not math.isfinite(y_err):
            raise RuntimeError(f"{m.name}: no bootstrap error for U at the crossing")
        name = f"{m.name}.binder"
        checks.append(Check(model=m.name, check=name,
                            quantity=f"Binder cumulant U = 1 - <m^4>/(3 <m^2>^2) at the crossing "
                                     f"of L = {a.size:g} and {b.size:g}",
                            measured=best.y, error=y_err, predicted=U, predicted_error=U_err,
                            source=source, sizes=d.sizes_text, statistics=statistics,
                            warnings=([binder_note] if binder_note else []) + list(warns),
                            figure=figure(name, "y", U, U_err, f"U* = {U:.5f} (L -> inf)")))
    return checks, best.x


def all_sizes_fit_text(r) -> str:
    """The finite-size diagnostic of an exponent: the power fit over all sizes."""
    used = np.asarray(r.used, bool)
    sizes = np.asarray(r.x, float)[used]
    text = (f"finite-size diagnostic: fit over all sizes (L >= {sizes.min():g}): "
            f"{r.params['b']:.4f} ± {r.errors['b']:.2g}")
    if r.chi2 is not None:
        text += f", chi2/dof = {r.chi2:.3g}/{r.dof}, p = {r.p_value:.2g}"
    return text


def _exponent(d: Data, figures: Path, r, predicted: float, name: str, quantity: str,
              ylabel: str, source: str, what: str, extra: Sequence[str] = ()) -> Check:
    """An exponent Check from an odspy size fit r (its own fit over all sizes), refitted over
    L >= fit_from (the campaign rule); r's exponent and the local exponents of successive
    sizes go into the warnings."""
    m, s = d.model, d.stats
    used = np.asarray(r.used, bool)
    rr = _refit(r, s.fit_from)
    # exponent_check adds the warnings of the fit it reports (rr); those of the fit over all
    # sizes (excluded points and why) are added here.
    warns = list(r.notes) + list(r.warnings) + list(extra)
    warns.append(f"exponent fitted over L >= {s.fit_from}: {FIT_RULE}")
    warns.append(all_sizes_fit_text(r))
    warns.append("effective exponents of successive sizes: "
                 + local_exponents(np.asarray(r.x)[used], np.asarray(r.y)[used],
                                   np.asarray(r.yerr)[used]))
    warns.append(m.corrections)
    warns += d.caveats
    fig = plot_exponent(rr, predicted, "L", ylabel, prov=d.tables[-1].provenance,
                        title=f"{m.title}: {quantity}",
                        legend="upper left" if predicted > 0 else "upper right")
    return exponent_check(rr, predicted, model=m.name, check=name, quantity=quantity,
                          source=source, sizes=d.sizes_text + f" (fit L >= {s.fit_from})",
                          statistics=d.statistics(what), warnings=warns,
                          figure=save_figure(fig, figures, name))


def critical_point_shift(sizes, values, errors, dlog_dx, delta: float, nu: float,
                         fit_from: float) -> tuple[float, float]:
    """The change of the power-fit exponent (over L >= fit_from) of values read at a fixed
    scaling variable x = (T - T_c) L^(1/nu) when T_c moves by +delta and by -delta: the point
    read then sits at x - delta L^(1/nu) of the moved T_c, so each value is carried to the
    moved x along its own log-derivative dlog_dx = d ln y / dx at fixed L, y exp(dlog_dx
    delta L^(1/nu)) (errors scaled alike). First order in delta; returns (b(+delta) - b,
    b(-delta) - b)."""
    L, y, e, d = (np.asarray(v, float) for v in (sizes, values, errors, dlog_dx))
    rng = (fit_from, math.inf)
    b0 = fit(L, y, e, "power", xrange=rng).params["b"]
    out = []
    for sign in (1.0, -1.0):
        factor = np.exp(d * sign * delta * L ** (1.0 / nu))
        out.append(float(fit(L, y * factor, e * factor, "power", xrange=rng).params["b"] - b0))
    return out[0], out[1]


def grid_neighbours(grid: Sequence[float], x: float, reach: int = 1) -> tuple[float, float]:
    """The grid points `reach` places below and above grid point x. ValueError when x has
    fewer than `reach` grid points on either side (no wrap-around at the ends)."""
    i = list(grid).index(x)
    if not reach <= i < len(grid) - reach:
        raise ValueError(f"x = {x:g} needs {reach} grid point(s) on each side in the grid "
                         f"{tuple(grid)}")
    return grid[i - reach], grid[i + reach]


def log_slope_derivative(x, U) -> float:
    """d ln|dU/dx| / dx at the middle of 3 points x[0] < x[1] < x[2] of a curve U: the
    chords on either side of x[1] are its slopes at their midpoints, and the difference of
    their logs over the distance of the midpoints is the log-derivative (exact for an
    exponential)."""
    x, U = np.asarray(x, float), np.asarray(U, float)
    s_lo = (U[1] - U[0]) / (x[1] - x[0])
    s_hi = (U[2] - U[1]) / (x[2] - x[1])
    return float((math.log(abs(s_hi)) - math.log(abs(s_lo))) / (0.5 * (x[2] - x[0])))


def _tc_shift_text(d: Data, sizes, values, errors, dlog, how: str) -> str:
    """The warning of an exponent read at a fixed scaling variable when tc is a measured
    value (Model.tc_error > 0): its shift for tc +- tc_error (critical_point_shift), with
    d ln y / dx of each size (dlog, in the order of sizes) obtained as `how` says."""
    m = d.model
    up, down = critical_point_shift(sizes, values, errors, dlog, m.tc_error, m.nu,
                                    d.stats.fit_from)
    c = m.control_label
    return (f"uncertainty of the cited {c}_c (± {m.tc_error:.2g}) propagated: the exponent "
            f"moves by {up:+.2g} for {c}_c + {m.tc_error:.2g} and by {down:+.2g} for "
            f"{c}_c - {m.tc_error:.2g} (each size's value carried along its d ln y/dx, {how}); "
            "not in the error bar")


def _value_shift_text(d: Data, curves: list[SizeCurve], x: float, sizes, values, errors) -> str:
    """_tc_shift_text for values read at grid point x: d ln y / dx from the two
    neighbouring grid points."""
    m = d.model
    lo, hi = grid_neighbours(m.grid, x)
    _, y_lo, _ = _at_x(curves, m, d.stats.sizes, lo)
    _, y_hi, _ = _at_x(curves, m, d.stats.sizes, hi)
    dlog = (np.log(y_hi) - np.log(y_lo)) / (hi - lo)
    return _tc_shift_text(d, sizes, values, errors, dlog,
                          f"from the grid points at x = {lo:g}, {hi:g}")


def _slope_shift_text(d: Data, r) -> str:
    """_tc_shift_text for the slopes of U at tc (x = 0, from the points at x = 0, +-s): their
    d ln|dU/dx| / dx from the slopes at x = +-s, i.e. the chords from x = 0 to the grid
    points at x = +-2s (log_slope_derivative), which the grid must hold."""
    m = d.model
    s_lo, s_hi = grid_neighbours(m.grid, 0.0)
    far_lo, far_hi = 2.0 * s_lo, 2.0 * s_hi
    if far_lo not in m.grid or far_hi not in m.grid:
        raise ValueError(f"{m.name}: propagating the tc uncertainty into 1/nu needs the grid "
                         f"points x = {far_lo:g}, {far_hi:g} (twice the slope points)")
    xs = (far_lo, 0.0, far_hi)
    read = [_at_x(d.binder, m, d.stats.sizes, x) for x in xs]
    by_size = {int(L): log_slope_derivative(xs, [y[k] for _, y, _ in read])
               for k, L in enumerate(read[0][0])}
    dlog = np.array([by_size[int(L)] for L in r.x])
    values = np.where(r.used, np.asarray(r.y, float), np.nan)
    return _tc_shift_text(d, r.x, values, r.yerr, dlog,
                          f"from the slopes at x = {s_lo:g}, {s_hi:g} (chords to the grid points "
                          f"at x = {far_lo:g}, 0, {far_hi:g})")


def exponent_checks(d: Data, figures: Path) -> list[Check]:
    """<name>.inv_nu (Binder slope at T_c), <name>.gamma_nu (chi' peak or chi' at a fixed
    scaling variable, see Model.gamma_at), <name>.beta_nu (|x| at T_c), each a power fit
    against L. With a measured T_c (Model.tc_error), the shift of each exponent for T_c +- its
    error goes into its warnings."""
    m = d.model
    c, kind = m.control_label, m.tc_kind
    measured_tc = m.tc_error > 0
    out = []
    r = slope_at_crossing_vs_size(d.binder, m.tc, n_boot=N_BOOT, seed=1)
    out.append(_exponent(d, figures, r, 1.0 / m.nu, f"{m.name}.inv_nu",
                         f"1/nu from |dU/d{c}| at the {kind} {c}_c ~ L^(1/nu)",
                         f"|dU/d{c}| at {c}_c", m.exp_source,
                         f"slopes through the 3 points around {c}_c, {N_BOOT} bootstrap draws",
                         extra=[_slope_shift_text(d, r)] if measured_tc else ()))
    if m.gamma_at is None:
        r = peak_vs_size(d.chi, n_boot=N_BOOT, seed=2)
        out.append(_exponent(d, figures, r, m.gamma_nu, f"{m.name}.gamma_nu",
                             "gamma/nu from the peak of chi' = N(<x^2> - <|x|>^2) ~ L^(gamma/nu)",
                             "peak of chi'", m.exp_source,
                             f"peak heights from a parabola through the 3 highest points, "
                             f"{N_BOOT} bootstrap draws"))
    else:
        L, y, e = _at_x(d.chi, m, d.stats.sizes, m.gamma_at)
        extra = [m.gamma_note] if m.gamma_note else []
        if measured_tc:
            extra.append(_value_shift_text(d, d.chi, m.gamma_at, L, y, e))
        out.append(_exponent(d, figures, fit(L, y, e, "power"), m.gamma_nu,
                             f"{m.name}.gamma_nu",
                             f"gamma/nu from chi' = N(<x^2> - <|x|>^2) near its broad maximum, "
                             f"fixed ({c} - {c}_c) L^(1/nu) = {m.gamma_at:g}, ~ L^(gamma/nu)",
                             f"chi' at ({c} - {c}_c) L^(1/nu) = {m.gamma_at:g}", m.exp_source,
                             "values at that grid point of each size", extra=extra))
    L, y, e = _at_x(d.order, m, d.stats.sizes)
    r = fit(L, y, e, "power")
    out.append(_exponent(d, figures, r, -m.beta_nu, f"{m.name}.beta_nu",
                         f"-beta/nu from {m.order_label} at the {kind} {c}_c ~ L^(-beta/nu)",
                         f"{m.order_label} at {c}_c", m.exp_source,
                         f"values at the {c}_c point of each size",
                         extra=[_value_shift_text(d, d.order, 0.0, L, y, e)] if measured_tc
                         else ()))
    return out


def all_sizes_collapse_text(res, lmin: float) -> str:
    """The finite-size diagnostic of a collapse: the collapse over all sizes (L >= lmin),
    with errors only where odspy gives them (a poor collapse has them withheld)."""
    def value(key: str, v: float, digits: int) -> str:
        e = res.errors.get(key, math.nan)
        return f"{v:.{digits}f}" + (f" ± {e:.2g}" if math.isfinite(e) else "")

    return (f"finite-size diagnostic: collapse over all sizes (L >= {lmin:g}): "
            f"S = {res.quality:.3g} over {res.points_used} of {res.points_total} points, "
            f"x_c = {value('xc', res.xc, 5)}, nu = {value('nu', res.nu, 4)}"
            + ("; errors withheld (flagged poor by odspy)" if res.poor else ""))


def collapse_checks(d: Data, figures: Path, xc0: float, nu0: float) -> list[Check]:
    """<name>.collapse_tc and <name>.collapse_nu: odspy.search_collapse of the Binder cumulant
    with a = 0 (U is dimensionless), from the measured crossing and 1/nu, over L >= fit_from
    (the campaign rule) without the exactly placed points (module doc); the same collapse
    over all sizes goes into the warnings as a finite-size diagnostic."""
    m, s = d.model, d.stats
    if m.exact_x is None:
        raise ValueError(f"{m.name}: a collapse needs Model.exact_x (offset grid points)")
    lmin = s.fit_from
    curves = _collapse_points([c for c in d.binder if c.size >= lmin], m, s.sizes)
    res = search_collapse(curves, xc0, nu0, 0.0, fit_a=False, n_boot=N_BOOT_COLLAPSE, seed=0)
    every = search_collapse(_collapse_points(d.binder, m, s.sizes), xc0, nu0, 0.0, fit_a=False,
                            n_boot=N_BOOT_COLLAPSE, seed=0)
    warns = list(res.warnings)
    warns.append(f"Houdayer-Hartmann S = {res.quality:.3g} over {res.points_used} of "
                 f"{res.points_total} points; started from x_c = {xc0:.5f} (crossing), "
                 f"nu = {nu0:.4f} (1/(fitted 1/nu))")
    if res.poor:
        warns.append("collapse flagged poor by odspy (errors withheld when nan)")
    warns.append(f"collapse over L >= {lmin}: {FIT_RULE}")
    warns.append(all_sizes_collapse_text(every, min(s.sizes)))
    warns.append("collapse over the offset grid points only (x = "
                 + ", ".join(f"{x:g}" for x in m.grid if x not in m.exact_x)
                 + f" +- up to {OFFSET:g}): the points at x = "
                 + ", ".join(f"{x:g}" for x in m.exact_x)
                 + " sit at the same scaled position in every size and would pin the search "
                 "at the exact (T_c, nu) (odspy collapse: coincident scaled points)")
    warns.append(m.corrections)
    warns += d.caveats
    statistics = d.statistics(f"collapse errors from {N_BOOT_COLLAPSE} bootstrap "
                              "re-minimisations")
    sizes = d.sizes_text + f" (collapse L >= {lmin})"

    def figure(name: str) -> str:
        fig, axes = new_fig(1, 2)
        a0, a1 = axes[0]
        _curves_axes(a0, [c for c in d.binder if c.size >= lmin], "Binder cumulant U", m)
        for i, c in enumerate(curves):
            a0.plot(c.x, c.y, "o", ms=6, mfc="none", color=color(i), lw=0,
                    label="in the collapse" if i == 0 else None)
        a0.legend(fontsize=6.5)
        for i, (L, X, Y, E) in enumerate(res.scaled):
            a1.errorbar(X, Y, yerr=E, fmt="o", ms=3, color=color(i), capsize=0,
                        label=f"L = {L:g}")
        a1.set(xlabel="(T - x_c) L^(1/nu)", ylabel="U")
        a1.set_title(f"x_c = {res.xc:.5f} ± {res.errors.get('xc', math.nan):.2g}, "
                     f"nu = {res.nu:.3f} ± {res.errors.get('nu', math.nan):.2g} "
                     f"(exact {m.tc:.5f}, {m.nu:.4g})", fontsize=7)
        a1.legend(fontsize=6.5)
        finish(fig, d.tables[-1].provenance, f"{m.title}: collapse of the Binder cumulant")
        return save_figure(fig, figures, name)

    out = []
    for key, pred, label, what in (("xc", m.tc, "tc", "x_c (critical temperature)"),
                                   ("nu", m.nu, "nu", "nu")):
        name = f"{m.name}.collapse_{label}"
        out.append(Check(model=m.name, check=name,
                         quantity=f"{what} from the collapse of U against (T - x_c) L^(1/nu)",
                         measured=getattr(res, key), error=res.errors.get(key, math.nan),
                         predicted=pred, predicted_error=0.0,
                         source=m.tc_source if key == "xc" else m.exp_source, sizes=sizes,
                         statistics=statistics, warnings=list(warns), figure=figure(name)))
    return out
