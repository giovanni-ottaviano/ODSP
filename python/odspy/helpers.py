"""Named helpers for the usual analyses, each carrying its caveats with it.

Rules followed throughout: a censored consensus time is never shown as a bare mean
(the point is excluded and named, or replaced by a median), and nothing is dropped
silently (every excluded point is counted in the result's `warnings`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import InputError
from .fits import FitResult, _fit
from .io import Table, find_outputs
from .sizes import SizeCurve, size_of
from .trajectories import fit_timeseries

CENSORED = "censored: mean biased low"
FIRST_PASSAGE_WHY = "the model can leave consensus; stop_time is the cap"
FIRST_PASSAGE = f"first-passage times: not recoverable from the files ({FIRST_PASSAGE_WHY})"
FIRST_PASSAGE_NOTE = ("y is the mean first-passage time (the first time consensus is reached; "
                      "the model can leave consensus), from the driver's sweep table")


# ---------------------------------------------------------------- small utilities

def _check_boot(n_boot) -> int:
    try:
        n = int(n_boot)
    except (TypeError, ValueError):
        raise InputError(f"n_boot must be an integer >= 2, got {n_boot!r}") from None
    if n < 2:
        raise InputError(f"n_boot must be an integer >= 2, got {n}")
    return n


def _num(table: Table, name: str) -> np.ndarray:
    col = table[name]
    if col.dtype == object:
        raise InputError(f"{table.path}: column '{name}' is not numeric")
    return col


def _config(table: Table) -> dict[str, str]:
    return table.provenance.config


def _label(v) -> str:
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def is_first_passage(table: Table) -> bool:
    """True for the output of a model that can leave consensus (noisy voter, Ising/Potts
    at T > 0, q-voter with p > 0, majority vote with noise). The driver then records the
    FIRST time consensus is reached, runs every replica to the cap and writes stop_time =
    the cap: the first-passage time is in its printed summary and sweep row, not in the
    replicas file. Such a model never reports an absorbing state (absorbed = 0 in every
    row), so a replicas file is first-passage when some replica reached consensus without
    being absorbed and none was absorbed; a sweep table when absorbed_fraction is 0 at
    every point and consensus_fraction is not. The "none absorbed" condition matters for
    bounded-confidence models (Deffuant, Hegselmann-Krause): their absorbing state is
    detected one sweep after the last large change, so a replica reaching consensus in
    the last sweep before the cap has reached = 1, absorbed = 0, while its siblings are
    absorbed. A `consensus_time` column (written by current drivers for first-passage runs
    only) settles it; without it, a file where no replica reached consensus cannot be told
    apart and is reported False."""
    if table.has("consensus_time"):   # current drivers write it only for first-passage runs
        return True
    if table.has("reached_consensus") and table.has("absorbed"):
        reached, absorbed = _num(table, "reached_consensus") > 0.5, _num(table, "absorbed") > 0.5
        return bool(np.any(reached & ~absorbed) and not np.any(absorbed))
    if table.has("consensus_fraction") and table.has("absorbed_fraction"):
        frac, absorbed = _num(table, "consensus_fraction"), _num(table, "absorbed_fraction")
        return bool(np.all(absorbed == 0) and np.any(frac > 0))
    return False


def replica_times(table: Table):
    """Per-replica consensus times as the driver counts them, or None when they are not in
    the file. Absorbing models: stop_time (the consensus time of a replica that reached
    consensus, the stop of the others). First-passage models: the `consensus_time` column of
    current drivers for replicas that reached consensus, stop_time (the cap, a censoring
    time) for the others; older first-passage files lack the column -> None."""
    stop = _num(table, "stop_time")
    if not is_first_passage(table):
        return stop
    if not table.has("consensus_time"):
        return None
    reached = _num(table, "reached_consensus") > 0.5
    return np.where(reached, _num(table, "consensus_time"), stop)


FIRST_PASSAGE_FROM_COLUMN = ("first-passage times from the consensus_time column (the model "
                             "can leave consensus: the first time each replica reached it; "
                             "stop_time is the cap)")


# ---------------------------------------------------------------- consensus time vs size

def _sweep_points(src, size: str):
    """[(size, label, table, row index, point index, point outputs)] of one source."""
    out = find_outputs(src)
    table = out.get("sweep")
    if table is None:
        raise InputError(f"{src}: no sweep table (<prefix>_sweep.csv); run with --sweep")
    if len(table) == 0:
        raise InputError(f"{table.path}: sweep table has no rows")
    if not table.provenance.config:
        table.provenance = out.provenance
    first = table.names[0]
    if _config(table).get("sweep2"):
        raise InputError(f"{src}: --sweep2 inputs are not supported here; sweep the size "
                         "with --sweep dims=... or give one input per size")
    pts = []
    for i in range(len(table)):
        if first in ("dims", "n"):
            cell = table[first][i]
            text = str(int(cell)) if isinstance(cell, (float, np.floating)) and cell == int(cell) \
                else str(cell)
            value, label = size_of({first: text}, size, f"{table.path} ({first}={text})")
        else:
            value, label = size_of(_config(table), size, str(src))
        pts.append((value, label, table, i, i, out.points.get(i)))
    return pts


def _driver_median(stop: np.ndarray, reached: np.ndarray) -> float:
    """The driver's median (Ensemble.hpp): the ceil(R/2)-th smallest time over all R
    replicas, unfinished ones counting as infinite; NaN when fewer than ceil(R/2)
    finished."""
    half = (stop.size + 1) // 2
    done = np.sort(stop[reached > 0.5])
    return float(done[half - 1]) if done.size >= half else math.nan


def _driver_medians(stop: np.ndarray, reached: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`_driver_median` of each bootstrap resample (rows of `idx`), NaN if undetermined."""
    half = (stop.size + 1) // 2
    v = np.sort(np.where(reached > 0.5, stop, np.inf)[idx], axis=1)[:, half - 1]
    return np.where(np.isfinite(v), v, math.nan)


def consensus_time_vs_size(sources, size="auto", censored="exclude", model="power",
                           n_boot=1000, seed=0) -> FitResult:
    """Fit the mean consensus time against system size.

    x = size, y = mean_consensus_time, yerr = sd_consensus_time / sqrt(replicas).
    Points with consensus_fraction < 1 are a censored mean (biased low): with
    censored="exclude" they are excluded (`used` False) and named in a warning
    "censored: mean biased low". With censored="median" y is the driver's median (ceil(R/2)-th smallest) over all
    replicas (unfinished ones counting as infinitely long) from the per-point
    replicas files, with bootstrap errors over replicas; points where the median is
    not determined (fewer than half finished) stay excluded.

    For a model that can leave consensus (`is_first_passage`) the times are first-passage
    times: the sweep table's mean (censored="exclude") is the driver's and is used, noted as
    such; censored="median" is refused, since the per-point replicas files hold the cap,
    not the first-passage times.
    """
    if censored not in ("exclude", "median"):
        raise InputError(f"censored must be 'exclude' or 'median', got '{censored}'")
    if isinstance(sources, (str, Path)):
        sources = [sources]
    if not sources:
        raise InputError("no inputs given")
    n_boot = _check_boot(n_boot)
    pts = [p for s in sources for p in _sweep_points(s, size)]
    if len({p[1] for p in pts}) > 1:
        raise InputError("inputs mix lattice (L) and node-count (N) sizes; pass size='N'")
    pts.sort(key=lambda p: p[0])
    x = np.array([p[0] for p in pts], float)
    rep = np.array([_num(p[2], "replicas")[p[3]] for p in pts], float)
    frac = np.array([_num(p[2], "consensus_fraction")[p[3]] for p in pts], float)
    exclude = []
    pre: list[str] = []   # warnings added ahead of the fit's own
    notes: list[str] = []
    first_passage = any(is_first_passage(t) for t in {id(p[2]): p[2] for p in pts}.values()) \
        or any(p[5] is not None and p[5].get("replicas") is not None
               and is_first_passage(p[5].get("replicas")) for p in pts)
    if first_passage:
        if censored == "median" and any(p[5] is not None and p[5].get("replicas") is not None
                                        and replica_times(p[5].get("replicas")) is None for p in pts):
            raise InputError(f"censored='median' needs each replica's consensus time: "
                             f"{FIRST_PASSAGE} (files from a driver without the consensus_time "
                             "column). Use censored='exclude': the sweep table's mean and median "
                             "are the driver's first-passage times")
        notes.append(FIRST_PASSAGE_NOTE)
    if censored == "exclude":
        y = np.array([_num(p[2], "mean_consensus_time")[p[3]] for p in pts], float)
        sd = np.array([_num(p[2], "sd_consensus_time")[p[3]] for p in pts], float)
        with np.errstate(all="ignore"):
            yerr = sd / np.sqrt(rep)
        cens = frac < 1
        if cens.any():
            named = ", ".join(f"size {_label(s)} (consensus_fraction {f:.3g})"
                              for s, f in zip(x[cens], frac[cens]))
            exclude.append((cens, CENSORED + ": {points} excluded: " + named.replace("{", "{{")
                            .replace("}", "}}")))
    else:
        y = np.full(x.size, np.nan)
        yerr = np.full(x.size, np.nan)
        rng = np.random.default_rng(seed)
        undetermined = []
        for k, p in enumerate(pts):
            if p[5] is None or p[5].get("replicas") is None:
                raise InputError(f"{p[2].path}: censored='median' needs the per-point replicas "
                                 f"file for row {p[4]} (<prefix>_point{p[4]}_replicas.csv), "
                                 "not found")
            rt = p[5].get("replicas")
            stop, reached = replica_times(rt), _num(rt, "reached_consensus")
            if stop is None:
                raise InputError(f"{rt.path}: {FIRST_PASSAGE}")
            med = _driver_median(stop, reached)
            if not math.isfinite(med):
                undetermined.append(k)
                continue
            idx = rng.integers(0, stop.size, size=(n_boot, stop.size))
            meds = _driver_medians(stop, reached, idx)
            finite = np.isfinite(meds)
            y[k] = med
            yerr[k] = float(np.std(meds[finite], ddof=1)) if finite.sum() > 1 else np.nan
            if not finite.all():
                notes.append(f"size {_label(x[k])}: {int((~finite).sum())} of {n_boot} "
                             "bootstrap resamples had an undetermined median (discarded)")
        if undetermined:
            named = ", ".join(f"size {_label(x[k])}" for k in undetermined)
            pre.append("median undetermined (fewer than half of the replicas finished), "
                       f"excluded: {named}")
        if (frac < 1).any():
            notes.append("median as the driver defines it (ceil(R/2)-th smallest of all replicas, "
                         "unfinished runs counted as infinitely long), not the censored mean")
    r = _fit(x, y, yerr, model, None, False, 0.01, exclude=exclude, prior=pre)
    r.notes += notes
    return r


# ---------------------------------------------------------------- coarsening

def coarsening_fit(prefix, observable, model="power", xrange=None, transform=None) -> FitResult:
    """Fit an ensemble time series (e.g. rho ~ t^-a), using `_trajectories.csv`
    for bootstrap errors when it exists; otherwise the result is labelled
    "errors not trustworthy: correlated points"."""
    out = find_outputs(prefix)
    ts = out.get("timeseries_ensemble")
    if ts is None:
        raise InputError(f"{prefix}: no ensemble time series (<prefix>_timeseries.csv from "
                         "--task ensemble)")
    return fit_timeseries(ts, observable, model, xrange, out.get("trajectories"),
                          transform)


# ---------------------------------------------------------------- finite-size scaling

def _curves(curves) -> list[SizeCurve]:
    if isinstance(curves, SizeCurve):
        curves = [curves]
    curves = list(curves)
    if not curves or not all(isinstance(c, SizeCurve) for c in curves):
        raise InputError("expected a list of SizeCurve (from odspy.families)")
    return curves


def _need_err(c: SizeCurve) -> np.ndarray:
    if c.yerr is None:
        raise InputError(f"curve of size {_label(c.size)} ({c.source}): no error column; "
                         "the bootstrap needs one")
    e = np.asarray(c.yerr, float)
    if not np.all(np.isfinite(e)) or np.any(e < 0):
        raise InputError(f"curve of size {_label(c.size)} ({c.source}): missing or negative "
                         "errors")
    return e


def _peak(x: np.ndarray, y: np.ndarray):
    """(height, bracketed): the vertex of the parabola through the three highest points.
    Not bracketed (height = the highest point, a lower bound) when the maximum is the
    first or last point, the three highest points are not neighbours, or the parabola
    has no maximum inside their span."""
    order = np.argsort(y, kind="stable")
    top = np.sort(order[-3:])
    best = int(order[-1])
    xs, ys = x[top], y[top]
    edge = best in (0, x.size - 1) or np.any(np.diff(top) != 1)
    if not edge and np.ptp(xs) > 0:
        a, b, c = np.polyfit(xs, ys, 2)
        if a < 0:
            xv = -b / (2 * a)
            if xs.min() <= xv <= xs.max():
                return float(c + b * xv + a * xv * xv), True
    return float(ys.max()), False


def _boot_stat(stat, x, y, e, n_boot, rng):
    """(value, bootstrap sd) of stat(x, y), y resampled as N(y, e)."""
    v = stat(x, y)
    draws = np.array([stat(x, y + e * rng.standard_normal(y.size)) for _ in range(n_boot)])
    return v, float(np.std(draws, ddof=1))


def _scaling_fit(sizes, vals, errs, model, warns, exclude=()):
    return _fit(np.array(sizes, float), np.array(vals, float), np.array(errs, float), model,
                exclude=exclude, prior=warns)


def peak_vs_size(curves, n_boot=1000, seed=0) -> FitResult:
    """Peak height of each curve (parabola through its three highest points,
    bootstrap errors from the point errors), then a power fit against size: the
    exponent `b` is gamma/nu for a susceptibility.

    A size whose peak is not bracketed (maximum at the edge of the scanned range, or
    the three highest points not neighbours) has only a lower bound for its height: it
    is excluded from the fit (`used` False), named in a warning. For the others, the
    fraction of bootstrap draws that were bracketed is reported when below 100%."""
    curves = _curves(curves)
    n_boot = _check_boot(n_boot)
    rng = np.random.default_rng(seed)
    sizes, vals, errs, warns, edge = [], [], [], [], []
    for c in curves:
        e = _need_err(c)
        x, y = np.asarray(c.x, float), np.asarray(c.y, float)
        if x.size < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise InputError(f"curve of size {_label(c.size)} ({c.source}): need at least 3 "
                             "finite points to locate a peak")
        v, ok = _peak(x, y)
        draws = [_peak(x, y + e * rng.standard_normal(y.size)) for _ in range(n_boot)]
        s = float(np.std([d[0] for d in draws], ddof=1))
        frac = float(np.mean([d[1] for d in draws]))
        name = f"{c.label}={_label(c.size)}"
        if not ok:
            warns.append(f"peak at the edge of the scanned range for size {name}: height "
                         "is a lower bound (excluded from the fit)")
        elif frac < 1:
            warns.append(f"size {name}: the peak was bracketed in {100 * frac:.0f}% of "
                         f"{n_boot} bootstrap draws")
        sizes.append(c.size); vals.append(v); errs.append(s); edge.append(not ok)
    mask = np.array(edge)
    r = _scaling_fit(sizes, vals, errs, "power", warns,
                     [(mask, "{points} with the peak at the scanned edge excluded")])
    r.notes.append("peak heights from a parabola through the 3 highest points, errors by "
                   f"bootstrap ({n_boot} resamples)")
    return r


def _local_slope(x0, k=3):
    def stat(x, y):
        near = np.argsort(np.abs(x - x0), kind="stable")[:k]
        return float(np.polyfit(x[near], y[near], 1)[0])
    return stat


def slope_at_crossing_vs_size(curves, x0, n_boot=1000, seed=0) -> FitResult:
    """Local linear slope of each curve at `x0` (least squares through the three
    points nearest x0, bootstrap errors), then a power fit of |slope| against size:
    the exponent `b` is 1/nu."""
    curves = _curves(curves)
    n_boot = _check_boot(n_boot)
    try:
        x0 = float(x0)
    except (TypeError, ValueError):
        raise InputError(f"x0 must be a number, got {x0!r}") from None
    rng = np.random.default_rng(seed)
    stat = _local_slope(x0)
    sizes, vals, errs, warns, edge = [], [], [], [], []
    for c in curves:
        e = _need_err(c)
        x, y = np.asarray(c.x, float), np.asarray(c.y, float)
        if x.size < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise InputError(f"curve of size {_label(c.size)} ({c.source}): need at least 3 "
                             "finite points for a local slope")
        if not (x.min() <= x0 <= x.max()):
            raise InputError(f"curve of size {_label(c.size)} ({c.source}): x0={x0:g} outside "
                             f"[{x.min():g}, {x.max():g}]")
        near = np.argsort(np.abs(x - x0), kind="stable")[:3]
        one_sided = not (np.any(x[near] <= x0) and np.any(x[near] >= x0))
        if one_sided:
            warns.append(f"size {c.label}={_label(c.size)}: the 3 points nearest x0={x0:g} do "
                         "not lie on both sides of it (extrapolated slope; excluded from "
                         "the fit)")
        v, s = _boot_stat(stat, x, y, e, n_boot, rng)
        sizes.append(c.size); vals.append(abs(v)); errs.append(s); edge.append(one_sided)
    r = _scaling_fit(sizes, vals, errs, "power", warns,
                     [(np.array(edge), "{points} with a one-sided slope estimate excluded")])
    r.notes.append(f"slope from a line through the 3 points nearest x0={x0:g}, errors by "
                   f"bootstrap ({n_boot} resamples); |slope| fitted")
    return r


def _value_at(x0):
    def stat(x, y):
        return float(np.interp(x0, x, y))
    return stat


def log_growth_vs_size(curves_or_values, x0=None, n_boot=1000, seed=0) -> FitResult:
    """Fit y = a + b ln(size), e.g. the specific heat at T_c.

    `curves_or_values`: a list of SizeCurve (needs `x0`: each curve is read at x0 by
    linear interpolation, bootstrap errors from the point errors), or a tuple
    (sizes, values, errors).
    """
    if isinstance(curves_or_values, tuple) and len(curves_or_values) == 3:
        try:
            sizes, vals, errs = (np.asarray(a, float) for a in curves_or_values)
        except (TypeError, ValueError):
            raise InputError("log_growth_vs_size: (sizes, values, errors) must be numeric") \
                from None
        warns = []
        note = "values given directly"
    else:
        curves = _curves(curves_or_values)
        if x0 is None:
            raise InputError("log_growth_vs_size: x0 is required with curves")
        n_boot = _check_boot(n_boot)
        try:
            x0 = float(x0)
        except (TypeError, ValueError):
            raise InputError(f"x0 must be a number, got {x0!r}") from None
        rng = np.random.default_rng(seed)
        stat = _value_at(x0)
        sizes, vals, errs, warns = [], [], [], []
        for c in curves:
            e = _need_err(c)
            x, y = np.asarray(c.x, float), np.asarray(c.y, float)
            if x.size < 2 or not (x.min() <= x0 <= x.max()):
                raise InputError(f"curve of size {_label(c.size)} ({c.source}): x0={x0} "
                                 "outside the curve")
            v, s = _boot_stat(stat, x, y, e, n_boot, rng)
            sizes.append(c.size); vals.append(v); errs.append(s)
        note = f"values at x0={x0:g} by linear interpolation, errors by bootstrap"
    r = _fit(sizes, vals, errs, "log", prior=warns)
    r.notes.append(note)
    return r


# ---------------------------------------------------------------- stationary caveats

def error_bar_uncertainty(k: int) -> float:
    """Relative uncertainty of an error bar from k independent groups: 1/sqrt(2(k-1))."""
    return 1.0 / math.sqrt(2.0 * (k - 1)) if k > 1 else math.nan


def _error_bar_note(chains: int, blocks: int) -> str:
    k = chains if chains > 1 else blocks
    pct = math.floor(100.0 * error_bar_uncertainty(k) + 0.5)   # lround, not banker's rounding
    return (f"error bars from {k} " + ("chains" if chains > 1 else "jackknife blocks")
            + f" are themselves uncertain by about +-{pct}%"
            + (" (more --replicas narrow them)" if chains > 1 else ""))


def _cfg_int(config, key, default):
    try:
        return int(float(config[key])) if key in config else default
    except ValueError:
        raise InputError(f"configuration: {key}={config[key]!r} is not a number") from None


def stationary_caveats(table: Table) -> list[str]:
    """Caveats of a stationary sweep table: points whose jackknife blocks
    (samples / blocks) are shorter than 10 tau_abs, and how uncertain the error bars
    themselves are (one line per distinct number of chains)."""
    config = _config(table)
    blocks = _cfg_int(config, "blocks", 20)
    samples = _cfg_int(config, "samples", 0)
    names = table.names
    keys = names[:2] if config.get("sweep2") else names[:1]
    out: list[str] = []
    notes: list[str] = []
    n = len(table)
    for i in range(n):
        label = ", ".join(f"{k}={_label(table[k][i])}" for k in keys)
        if table.has("tau_abs") and samples > 0 and blocks > 0:
            tau = float(_num(table, "tau_abs")[i])
            block = samples / blocks
            if math.isfinite(tau) and block < 10.0 * tau:
                out.append(f"{label}: jackknife blocks of {block:.3g} samples are not much "
                           f"longer than the autocorrelation time (tau_abs = {tau:.3g}); "
                           "errors may be underestimated (use more --samples or a larger "
                           "--interval)")
        chains = int(_num(table, "chains")[i]) if table.has("chains") \
            else _cfg_int(config, "replicas", 1)
        if chains > 1 or blocks > 1:
            note = _error_bar_note(chains, blocks)
            if note not in notes:
                notes.append(note)
    return out + notes


# ---------------------------------------------------------------- recovering driver numbers

def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a binomial fraction k/n (sensible at k = 0 or n,
    where the Wald error √(p(1−p)/n) is 0)."""
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # exact ends where rounding would leave them a hair inside [0, 1]
    return (0.0 if k == 0 else lo), (1.0 if k == n else hi)


@dataclass
class Recovered:
    values: dict[str, tuple[float, float]]   # name -> (value, error)
    method: str
    missing: list[str] = field(default_factory=list)   # printed by the driver, not recoverable
    notes: list[str] = field(default_factory=list)     # how a value or its error is given


def _final_means(t: Table, values: dict, R: int) -> None:
    skip = {"replica", "reached_consensus", "absorbed", "stop_time", "consensus_time"}   # bookkeeping
    for name in t.names:
        if name in skip or t[name].dtype == object:
            continue
        col = t[name]
        values[f"mean_{name}"] = (float(col.mean()), float(col.std()) / math.sqrt(R))


def _fixation_missing(t: Table, missing: list[str]) -> None:
    fix = Path(str(t.path).replace("_replicas.csv", "_fixation.csv"))
    if not fix.exists():
        missing.append("fixation per opinion (no _fixation.csv)")


def summarise_replicas(t: Table, config: dict[str, str] | None = None) -> Recovered:
    """The driver's ensemble summary from a `_replicas.csv` table: fractions (binomial
    error; at 0 or 1 no error, the Wilson interval is noted instead), consensus time over
    the finished runs, its median and lower bound, and the mean and standard error of every
    final observable.

    The lower bound is the driver's: every run that did not reach consensus counts as the
    cap (`steps` of the configuration, in the unit of stop_time), whether it hit the cap or
    froze without consensus (then its consensus time is infinite). `config` defaults to the
    table's own; without `steps` the bound is undefined (NaN) when some run did not finish.

    For a model that can leave consensus (`is_first_passage`) stop_time is the cap for every
    replica: the statistics are those of the first-passage times in the `consensus_time`
    column (the driver's own first-passage summary). Files from older drivers lack it; then
    no consensus-time statistic is computed and the times are listed in `missing` (the
    driver's printed summary and sweep row hold them)."""
    for name in ("reached_consensus", "absorbed", "stop_time"):
        if not t.has(name):
            raise InputError(f"{t.path}: no column '{name}': summarise_replicas needs a "
                             "<prefix>_replicas.csv table")
    R = len(t)
    if R == 0:
        raise InputError(f"{t.path}: no rows")
    config = t.provenance.config if config is None else config
    reached, absorbed, stop = (_num(t, n) for n in ("reached_consensus", "absorbed", "stop_time"))
    done = reached > 0.5
    n_done = int(done.sum())
    frozen = ~done & (absorbed > 0.5)
    n_frozen, n_capped = int(frozen.sum()), int((~done & ~frozen).sum())
    nan = math.nan
    missing: list[str] = []
    notes: list[str] = []

    def frac(name, count):
        p = count / R
        if 0 < count < R:
            return p, math.sqrt(p * (1 - p) / R)
        lo, hi = wilson_interval(count, R)
        notes.append(f"{name} = {p:g}: the binomial error √(p(1−p)/R) is 0 here and not "
                       f"shown; 95% Wilson interval [{lo:.3g}, {hi:.3g}] (R = {R})")
        return p, nan

    values: dict[str, tuple[float, float]] = {
        "consensus_fraction": frac("consensus_fraction", n_done),
        "absorbed_fraction": frac("absorbed_fraction", int((absorbed > 0.5).sum())),
    }
    first_passage = is_first_passage(t)
    times = replica_times(t)
    if times is None:
        # stop_time is the cap for every replica: the first-passage times are not in the file
        notes.append("consensus_fraction: replicas that reached consensus at least once before "
                     "the cap (first passage; the model can leave consensus); final values are "
                     "measured at the cap; no mean, sd, median or lower bound of the consensus "
                     "time is computed (stop_time is the cap for every replica)")
        # (the report prefixes "not recoverable from the files:"; there is no fixation)
        missing.append(f"first-passage times ({FIRST_PASSAGE_WHY})")
        _final_means(t, values, R)
        return Recovered(values, "recomputed from the per-replica file (population sd, as the "
                         "driver)", missing, notes)
    if first_passage:
        notes.append(FIRST_PASSAGE_FROM_COLUMN + "; consensus_fraction: replicas that reached "
                     "consensus at least once before the cap; final values are measured at the cap")
    if n_done:
        done_times = times[done]
        mean = float(done_times.mean())
        sd = float(done_times.std())   # population sd, as the driver
        values["mean_consensus_time"] = (mean, sd / math.sqrt(n_done))
        values["sd_consensus_time"] = (sd, nan)
    # the driver's median: the ceil(R/2)-th smallest time over all replicas (unfinished
    # ones count as infinite), undefined when fewer than half finished
    med = _driver_median(times, reached)   # NaN when fewer than half finished
    values["median_consensus_time"] = (med, nan)
    if n_done == R:
        lower = float(times.mean())
    else:
        try:
            cap = float(config["steps"])
        except (KeyError, ValueError):
            cap = nan
        lower = float((times[done].sum() + cap * (R - n_done)) / R)
    values["mean_consensus_time_lower_bound"] = (lower, nan)
    if n_done < R:
        parts = []
        if n_frozen:
            parts.append(f"{n_frozen} of {R} runs froze without consensus (their consensus "
                         "time is infinite)")
        if n_capped:
            parts.append(f"{n_capped} of {R} runs hit the cap")
        bound = (f"true mean >= {lower:.6g}, each unfinished run counted at the cap "
                 f"steps = {cap:g}" if math.isfinite(lower) else
                 "lower bound undefined: the cap (--steps) is not in the file's configuration")
        missing.append(f"mean_consensus_time: {CENSORED} ({'; '.join(parts)}; {bound})")
    _final_means(t, values, R)
    if not first_passage:            # no fixation where consensus is not absorbing
        _fixation_missing(t, missing)
    return Recovered(values, "recomputed from the per-replica file (population sd, as the "
                     "driver)", missing, notes)


def _jackknife(x: np.ndarray, blocks: int, estimator) -> float:
    """Block jackknife of estimator(m1, abs_m1, m2, m4), transcribed from
    stats::jackknife_error (include/Statistics.hpp): equal contiguous blocks, a
    remainder at the end ignored, 0 with fewer than 2 blocks' worth of samples."""
    if blocks < 2 or x.size < 2 * blocks:
        return 0.0
    length = x.size // blocks
    used = length * blocks
    b = x[:used].reshape(blocks, length)
    sums = np.stack([b.sum(1), np.abs(b).sum(1), (b ** 2).sum(1), (b ** 4).sum(1)])
    loo = (sums.sum(1, keepdims=True) - sums) / (used - length)
    theta = np.array([estimator(*loo[:, k]) for k in range(blocks)])
    var = np.sum((theta - theta.mean()) ** 2) * (blocks - 1) / blocks
    return math.sqrt(var)


def _tau_int(x: np.ndarray, c: float = 6.0) -> float:
    """Transcription of stats::integrated_autocorrelation_time (Sokal windowing)."""
    n = x.size
    if n < 2:
        return 0.5
    d = x - x.mean()

    def autocov(t):
        return float(np.dot(d[:n - t], d[t:])) / (n - t)

    c0 = autocov(0)
    if c0 <= 0.0:
        return 0.5
    tau = 0.5
    for t in range(1, n // 2):
        tau += autocov(t) / c0
        if t >= c * tau:
            break
    return max(tau, 0.5)


def summarise_samples(t: Table, config: dict[str, str]) -> Recovered:
    """The driver's stationary estimators from a `_samples.csv` table.

    mean, abs_mean, susceptibility N(<x^2>-<x>^2), abs_susceptibility
    N(<x^2>-<|x|>^2) and Binder 1-<x^4>/(3<x^2>^2) of the sampled observable. With
    several chains the samples are pooled and the error is the leave-one-chain-out
    jackknife (as the driver); with one chain, the block jackknife with the config's
    `blocks` (default 20).
    """
    for name in ("sample", "chain"):
        if not t.has(name):
            raise InputError(f"{t.path}: no column '{name}': summarise_samples needs a "
                             "<prefix>_samples.csv table")
    obs = [n for n in t.names if n not in ("sample", "chain", "sweep")]
    if not obs:
        raise InputError(f"{t.path}: no observable column")
    x_all = np.asarray(_num(t, obs[-1]), float)
    chain = _num(t, "chain")
    if x_all.size == 0 or not np.all(np.isfinite(x_all)):
        raise InputError(f"{t.path}: no samples, or non-finite values in '{obs[-1]}'")
    ids = np.unique(chain)
    R = ids.size
    parts = [x_all[chain == c] for c in ids]   # chain after chain, file order within a chain
    if any(p.size != parts[0].size for p in parts):
        raise InputError(f"{t.path}: chains have different numbers of samples")
    x = np.concatenate(parts)
    N, _ = size_of(config, "N", str(t.path))
    blocks = _cfg_int(config, "blocks", 20)
    jk_blocks = R if R > 1 else blocks

    estimators = {
        "mean": lambda m1, a1, m2, m4: m1,
        "abs_mean": lambda m1, a1, m2, m4: a1,
        "susceptibility": lambda m1, a1, m2, m4: N * (m2 - m1 * m1),
        "abs_susceptibility": lambda m1, a1, m2, m4: N * (m2 - a1 * a1),
        "binder": lambda m1, a1, m2, m4: 1.0 - m4 / (3.0 * m2 * m2) if m2 > 0.0 else 0.0,
    }
    m1, a1, m2, m4 = x.mean(), np.abs(x).mean(), (x ** 2).mean(), (x ** 4).mean()
    values = {name: (float(f(m1, a1, m2, m4)), _jackknife(x, jk_blocks, f))
              for name, f in estimators.items()}
    method = (f"{R} chains pooled; error = leave-one-chain-out jackknife (as the driver)"
              if R > 1 else f"single chain; error = block jackknife, {blocks} blocks")
    # tau of x and |x|, averaged over chains as the driver does; no error is printed
    values["tau"] = (float(np.mean([_tau_int(p) for p in parts])), math.nan)
    values["tau_abs"] = (float(np.mean([_tau_int(np.abs(p)) for p in parts])), math.nan)
    missing = ["xi", "xi_over_L", "energy", "specific_heat"]
    return Recovered(values, method, missing)
