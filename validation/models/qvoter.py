"""The q-voter model with independence (--model qvoter, panel with repetition, the defaults)
on the complete graph, where it is exactly the birth-death chain of
theory.qvoter_independence_rates at any N.

qvoter.exact_finite_n: the stationary <|m|> against p for q = 3 at one moderate N, as
sim / exact (the exact chain's stationary law), weighted over p, predicted 1. The errors are
the driver's leave-one-chain-out jackknife over independent chains.
qvoter.pc_q3, qvoter.pc_q4: p_c from the crossing of the Binder cumulants of the two largest
N (odspy.find_crossings on stationary sweeps over p, >= 4 N), against the mean-field
p_c = (q-1)/(q-1+2^(q-1)) (theory.qvoter_pc). On the complete graph the crossing of N and 2N
approaches p_c from below as ~ -0.6/N (from the exact chain), while its statistical error
shrinks only as ~ N^-1/4 at fixed statistics per point, so the largest pair is chosen where
the exact chain's crossing, sampled on the same p grid and interpolated as the data are, is
<= ~0.3 sigma from mean field: 1600/3200 at small (-2.7e-4 against sigma ~9e-4), 12800/25600
at full (-2.0e-5 for q = 3, +0.2e-5 for q = 4 on the 0.001 grid; -3.8e-5 / -3.1e-5 without
the interpolation; sigma ~1.2e-4 expected). The measured crossings of every pair and the
exact chain's crossing of the compared pair are reported in the warnings.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from odspy import SizeCurve, families, find_crossings, find_outputs
from odspy.helpers import stationary_caveats
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import Ratio, ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run
from validation.theory import (qvoter_independence_rates, qvoter_mean_field_m, qvoter_pc,
                               stationary_distribution)

NAME = "qvoter"

FINITE_N_Q = 3
N_BOOT = 2000          # parametric bootstrap draws for the crossing errors


def _grid(lo: float, hi: float, step: float) -> tuple[float, ...]:
    k = int(round((hi - lo) / step))
    return tuple(round(lo + i * step, 6) for i in range(k + 1))


# Per scale. finite_n: (N, p grid, chains, samples, interval, burn-in, threads per job).
# pc: (sizes, {q: p grid}, chains, samples, interval, burn-in, threads per job). The p grids
# of the crossings bracket p_c (1/3 and 3/11) and the exact chain's crossings of every pair.
# Linear interpolation between grid points shifts a crossing by up to ~+4.5e-5 (exact chain:
# small 1600/3200, 0.0025 grid) and +1.5e-5..+3.3e-5 (full, 0.001 grid); exact_crossing
# applies the same interpolation, so the offset it reports includes this shift. tau_abs grows
# ~ sqrt(N) sweeps near p_c (~35 sweeps at N = 3200, so ~100 at N = 25600): burn-in is ~30 of
# them at small and ~50 at full.
# Full-scale error of the compared crossing, scaled from small (sigma ~9e-4 at 1600/3200 with
# 2e5 chain-sweeps per point): x20 chain-sweeps, sigma_U ~ (tau/sweeps)^1/2 with tau ~ sqrt(N),
# Binder slope difference ~ sqrt(N): ~1.4e-4 at 6400/12800 (where the exact offset would be
# ~ -0.45 sigma) and ~1.2e-4 at 12800/25600, hence N = 25600.
SCALE = {
    "smoke": {
        "finite_n": (50, _grid(0.20, 0.45, 0.05), 4, 500, 2, 300, 1),
        "pc": ((50, 100, 200, 400), {3: _grid(0.29, 0.35, 0.01), 4: _grid(0.23, 0.29, 0.01)},
               4, 400, 5, 300, 1),
    },
    "small": {
        "finite_n": (100, _grid(0.20, 0.45, 0.025), 20, 4000, 2, 1000, 2),
        "pc": ((400, 800, 1600, 3200),
               {3: _grid(0.3225, 0.3425, 0.0025), 4: _grid(0.2625, 0.2825, 0.0025)},
               20, 1000, 10, 1000, 2),
    },
    "full": {
        "finite_n": (200, _grid(0.20, 0.45, 0.0125), 50, 20000, 2, 2000, 4),
        "pc": ((800, 1600, 3200, 6400, 12800, 25600),
               {3: _grid(0.3295, 0.3375, 0.001), 4: _grid(0.2685, 0.2765, 0.001)},
               50, 4000, 20, 5000, 4),
    },
}

SEEDS = {"finite_n": 721, "pc_q3": 722, "pc_q4": 723}


def _values(xs) -> str:
    return ",".join(f"{x:g}" for x in xs)


def _stationary(q: int, chains: int, samples: int, interval: int, burn: int,
                *extra: str) -> tuple[str, ...]:
    return ("--model", "qvoter", "--q", str(q), "--graph", "complete", "--task", "stationary",
            "--observable", "m", "--burn-in", str(burn), "--samples", str(samples),
            "--interval", str(interval), "--replicas", str(chains), *extra)


def runs(scale: str) -> list[Run]:
    s = SCALE[scale]
    N, grid, chains, samples, interval, burn, threads = s["finite_n"]
    out = [Run("finite_n", _stationary(FINITE_N_Q, chains, samples, interval, burn, "--n", str(N),
                                       "--sweep", "p=" + _values(grid)),
               seed=SEEDS["finite_n"], threads=threads)]
    Ns, grids, chains, samples, interval, burn, threads = s["pc"]
    for q, grid in grids.items():
        out.append(Run(f"pc_q{q}", _stationary(q, chains, samples, interval, burn,
                                               "--sweep", "n=" + _values(Ns),
                                               "--sweep2", "p=" + _values(grid)),
                       seed=SEEDS[f"pc_q{q}"], threads=threads))
    return out


# ---------------------------------------------------------------------------------------------

def exact_moments(N: int, q: int, p: float) -> tuple[float, float]:
    """(<|m|>, Binder 1 - <m^4>/(3 <m^2>^2)) of the exact stationary law, m = (2n - N)/N."""
    pi = stationary_distribution(*qvoter_independence_rates(N, q, p))
    m = (2.0 * np.arange(N + 1) - N) / N
    m2, m4 = float(pi @ m ** 2), float(pi @ m ** 4)
    return float(pi @ np.abs(m)), 1.0 - m4 / (3.0 * m2 * m2)


def exact_crossing(Na: int, Nb: int, q: int, grid) -> float:
    """Where the exact chain's Binder curves of Na and Nb cross when sampled on `grid` and
    linearly interpolated (the estimator applied to the data): the crossing nearest the
    mean-field p_c, nan if none."""
    grid = np.asarray(grid, float)
    curves = [SizeCurve(N, "N", grid, np.array([exact_moments(N, q, p)[1] for p in grid]), None,
                        "p", "binder", "exact chain") for N in (Na, Nb)]
    found = [c.x for c in find_crossings(curves, n_boot=2) if c.found]
    return min(found, key=lambda x: abs(x - qvoter_pc(q))) if found else math.nan


def _sweep(prefix: Path) -> Table:
    table = find_outputs(prefix).get("sweep")
    if table is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    return table


def _ratio_axes(ax, x, ratio: Ratio, xlabel: str) -> None:
    ax.errorbar(x, ratio.ratios, yerr=ratio.errors, fmt="o", ms=4, color=color(0),
                capsize=0, label="simulation / exact")
    ax.axhline(1.0, color="#555555", lw=1, label="exact")
    ax.axhspan(ratio.value - ratio.error, ratio.value + ratio.error, color=color(0), alpha=0.15,
               lw=0, label=f"weighted mean {ratio.value:.4f} ± {ratio.error:.2g}")
    ax.set(xlabel=xlabel, ylabel="simulation / exact")
    ax.legend(fontsize=7)


def _finite_n(scale: str, data: Path, figures: Path) -> Check:
    t = _sweep(data / "finite_n")
    N, q = int(t.provenance.config["n"]), int(t.provenance.config["q"])
    p = np.asarray(t["p"], float)
    sim, err = np.asarray(t["abs_mean"], float), np.asarray(t["abs_mean_err"], float)
    exact = np.array([exact_moments(N, q, x)[0] for x in p])
    ratio = weighted_ratio(sim, err, exact)
    name = "qvoter.exact_finite_n"
    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0]
    ps = np.linspace(p.min(), p.max(), 101)
    a0.errorbar(p, sim, yerr=err, fmt="o", ms=4, color=color(0), capsize=0, label="simulation")
    a0.plot(ps, [exact_moments(N, q, x)[0] for x in ps], "-", color=color(1), lw=1,
            label=f"exact chain, N = {N}")
    a0.plot(ps, [qvoter_mean_field_m(q, x) for x in ps], ":", color="#555555", lw=1,
            label="mean field (N -> inf)")
    a0.axvline(qvoter_pc(q), color="#999999", lw=0.8, ls="--")
    a0.set(xlabel="p (independence)", ylabel="<|m|>")
    a0.legend(fontsize=7)
    _ratio_axes(a1, p, ratio, "p (independence)")
    finish(fig, t.provenance, f"q-voter q = {q}, complete graph N = {N}: stationary <|m|>")
    chains = int(np.min(t["chains"]))
    return ratio_check(ratio, model=NAME, check=name, source="qvoter_independence",
                       quantity="<|m|>_sim / <|m|>_exact (stationary law of the exact birth-death "
                                f"chain), q = {q}, weighted over p",
                       sizes=f"N={N} (complete), p={p.min():g}..{p.max():g} ({p.size} points)",
                       statistics=f"{chains} chains x {t.provenance.config.get('samples')} samples "
                                  "per p; leave-one-chain-out jackknife",
                       warnings=stationary_caveats(t), report_consistency=True,
                       figure=save_figure(fig, figures, name))


def _pc(scale: str, data: Path, figures: Path, q: int) -> Check:
    Ns, grids = SCALE[scale]["pc"][:2]
    grid = grids[q]
    pc = qvoter_pc(q)
    prefix = data / f"pc_q{q}"
    t = _sweep(prefix)
    curves = families([prefix], "binder")
    absm = families([prefix], "abs_mean")
    crossings = find_crossings(curves, n_boot=N_BOOT, seed=0)
    warns: list[str] = []
    largest = sorted(c.size for c in curves)[-2:]
    pair = [c for c in crossings if [c.size_a, c.size_b] == largest]
    usable = [c for c in pair if c.found and math.isfinite(c.err)]
    name = f"qvoter.pc_q{q}"
    if not usable:
        raise RuntimeError(f"{prefix}: the Binder curves of N = {largest[0]:g} and "
                           f"{largest[1]:g} do not cross inside p = {min(grid):g}..{max(grid):g} "
                           "(with an error)")
    best = min(usable, key=lambda c: abs(c.x - pc))
    if len(pair) > 1:
        warns.append(f"N = {largest[0]:g}/{largest[1]:g} curves cross {len(pair)} times (at p = "
                     + ", ".join(f"{c.x:.5g}" for c in pair) + "); the crossing nearest the "
                     "mean-field p_c is used")
    warns += best.warnings
    for c in crossings:
        if [c.size_a, c.size_b] != largest:
            warns += [f"N = {c.size_a:g}/{c.size_b:g}: {w}" for w in c.warnings]
    drift = "; ".join(f"{c.size_a:g}/{c.size_b:g}: " + (f"{c.x:.5f} ± {c.err:.2g}" if c.found
                                                        else "none")
                      for c in crossings)
    warns.append(f"crossings of every pair (drift towards p_c from below expected): {drift}")
    ex = exact_crossing(int(largest[0]), int(largest[1]), q, grid)
    sigma_off = (ex - pc) / best.err
    warns.append(f"exact finite-N chain: the N = {largest[0]:g}/{largest[1]:g} Binder curves "
                 f"cross at p = {ex:.5f} on this grid, {ex - pc:+.2g} from mean field "
                 f"({sigma_off:+.2f} sigma of this measurement)")
    if abs(sigma_off) > 0.5:
        warns.append("the finite-N offset is not negligible against the error: larger N needed")
    warns += stationary_caveats(t)

    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0]
    for i, c in enumerate(curves):
        a0.errorbar(c.x, c.y, yerr=c.yerr, fmt="o-", ms=3, lw=0.8, color=color(i), capsize=0,
                    label=f"N = {c.size:g}")
    a0.axvline(pc, color="#555555", lw=1, ls="--", label=f"mean field p_c = {pc:.5f}")
    a0.errorbar([best.x], [best.y], xerr=[best.err], fmt="*", ms=9, color="k", capsize=2,
                label=f"crossing {best.x:.5f} ± {best.err:.2g}")
    a0.set(xlabel="p (independence)", ylabel="Binder cumulant U")
    a0.legend(fontsize=6.5)
    for i, c in enumerate(absm):
        a1.errorbar(c.x, c.y, yerr=c.yerr, fmt="o", ms=3, color=color(i), capsize=0,
                    label=f"N = {c.size:g}")
    ps = np.linspace(min(grid), max(grid), 200)
    a1.plot(ps, [qvoter_mean_field_m(q, x) for x in ps], "-", color="k", lw=1,
            label="mean field |m| (N -> inf)")
    a1.axvline(pc, color="#555555", lw=1, ls="--")
    a1.set(xlabel="p (independence)", ylabel="<|m|>")
    a1.legend(fontsize=6.5)
    finish(fig, t.provenance, f"q-voter q = {q}, complete graph: Binder crossings")
    chains = int(np.min(t["chains"]))
    return Check(model=NAME, check=name,
                 quantity=f"p_c from the Binder crossing of N = {largest[0]:g} and {largest[1]:g} "
                          f"(q = {q}, independence, complete graph)",
                 measured=best.x, error=best.err, predicted=pc, predicted_error=0.0,
                 source="qvoter_independence",
                 sizes="N=" + _values(Ns) + f" (complete), p={min(grid):g}..{max(grid):g}",
                 statistics=f"{chains} chains x {t.provenance.config.get('samples')} samples per "
                            f"point; crossing error from {N_BOOT} parametric bootstrap draws",
                 warnings=warns, figure=save_figure(fig, figures, name))


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_finite_n(scale, data, figures), _pc(scale, data, figures, 3),
            _pc(scale, data, figures, 4)]
