"""The 2D Ising model on the periodic square lattice (--model ising, J = 1, h = 0, the default
Metropolis single-spin-flip dynamics, ordered start --init up), against the exact solution.

Finite-size scaling (validation/models/_critical.py): one stationary temperature sweep per
L at T = T_c + x / L (1/nu = 1), the same x for every size at x = 0, +-1 (crossing, slope,
<|m|>(T_c)) and x = 2 (chi'), offset by up to +-0.2 per size elsewhere (the points of the
collapse).
ising.tc: the crossing of the Binder cumulants U = 1 - <m^4>/(3 <m^2>^2) of the two largest
L (odspy.find_crossings), against T_c = 2/ln(1 + sqrt 2).
ising.binder: U at that crossing (parametric bootstrap error), against U* of Salas & Sokal.
ising.inv_nu: |dU/dT| at the exact T_c (odspy.slope_at_crossing_vs_size) ~ L^(1/nu), 1/nu = 1.
ising.gamma_nu: chi' = N(<m^2> - <|m|>^2) at the fixed scaling variable (T - T_c) L = 2, at
its broad maximum, ~ L^(7/4) (power fit). Planned as odspy.peak_vs_size of chi', but the 2D
Ising chi' maximum is flat to ~3% over (T - T_c) L = 1..3, comparable to its errors (2-2.5%
at ~500 independent samples), so the parabola through the 3 highest points came out convex
or unbracketed at 2 of the 4 sizes of the small run. Finite-size scaling gives
chi'(T_c + x/L) = L^(gamma/nu) g(x) at any fixed x; x = 2 is where chi' is largest and has its
smallest errors; it is placed exactly in every size (Model.exact_x). The
alternatives at T_c are worse: chi' has ~10% errors there (a small difference of <m^2> and
<|m|>^2), and the plain chi = N(<m^2> - <m>^2) is biased low by N <m>^2 of the pooled chains
(the sign of m flips only every ~1e4 sweeps at L = 64, a relative bias of a few % that grows
with L).
ising.beta_nu: <|m|> at the exact T_c ~ L^(-1/8).
ising.collapse_tc, ising.collapse_nu: odspy.search_collapse of U against (T - x_c) L^(1/nu)
(a = 0), started from the crossing and the fitted 1/nu.
Fit ranges: the campaign rule, fixed before the full run (validation/models/_critical.py):
every exponent fit and the collapse use all sizes but the smallest (L >= 12 at smoke, >= 16
at small, >= 32 at full); the crossing is that of the two largest sizes. The fits and the
collapse over all sizes are reported in the warnings as a finite-size diagnostic, with the
crossings of every pair and the effective exponents of successive sizes.

ising.energy, ising.magnetization: energy per site at T in [1.5, 3.5] and <|m|> at T <= 2.1 at
the largest L, against Onsager's ising_energy and onsager_magnetization, as sim/exact
weighted over T. The energy leaves out 2.1 < T < 2.5: towards T_c the finite lattice's energy
approaches its O(1/L) difference from the infinite one at T_c. At the temperatures used the
correlation length is xi <= 6 (exact: T = 2.1, xi ~ 3.6; T = 2.5, xi ~ 6.0), so the finite-size
difference ~ exp(-L/xi) is below 1e-4 already at L = 64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from odspy.helpers import stationary_caveats

from validation.analysis import plot_ratio, ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run
from validation.models._critical import (Model, Stats, burn_in_caveat, collapse_checks,
                                         crossing_checks, exponent_checks, fss_runs, read,
                                         sweep_table)
from validation.theory import (ISING_BINDER, ISING_EXPONENTS, ISING_TC, ising_energy,
                               onsager_magnetization)

__all__ = ["NAME", "runs", "analyse"]

NAME = "ising"

# tau(L) of |m| at T_c, Metropolis, in sweeps: 38, 206, 665 at L = 16, 32, 64 (8 chains of
# 20000 samples each, interval 5; a free fit gives z ~ 2.06). The model 0.1 L^2.17 gives 41,
# 186, 840: ~10% low at L = 32, conservative at L >= 64 (where the cost is). Each point's
# burn-in is checked against its measured tau(|m|) (_critical.burn_in_caveat).
MODEL = Model(
    name=NAME,
    args=("--model", "ising", "--init", "up"),
    observable="m",
    tc=ISING_TC,
    nu=ISING_EXPONENTS["nu"],
    gamma_nu=ISING_EXPONENTS["gamma/nu"],
    beta_nu=ISING_EXPONENTS["beta/nu"],
    tc_source="ising_tc",
    exp_source="ising_exponents",
    # x = (T - T_c) L: U changes across [-2, 4] (the collapse); x = +-1 give the slope at
    # T_c (half the error of +-0.5; on the scaled grid the curvature of U enters every size
    # alike); chi' is largest, and flat, around x = 2 (scan at L = 16). x = -1, 0, 1, 2 are
    # exact; the other 5 are offset and make the collapse.
    grid=(-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 4.0),
    tau=(0.1, 2.17),
    scales={
        "smoke": Stats(sizes=(8, 12, 16, 24), chains=8, n_eff=1000, burn_tau=30, threads=1),
        "small": Stats(sizes=(8, 16, 32, 64), chains=8, n_eff=500, burn_tau=30, threads=2),
        "full": Stats(sizes=(16, 32, 64, 128, 256), chains=10, n_eff=1200, burn_tau=30,
                      threads=8),
    },
    seed=8100,
    title="2D Ising (Metropolis)",
    order_label="<|m|>",
    corrections="finite-size corrections are not fitted: the drift of the crossings and the "
                "effective exponents of successive sizes above show their size",
    gamma_at=2.0,
    gamma_note="peak_vs_size refused (flat 2D Ising chi' maximum); chi' at fixed (T - T_c) L = 2 "
               "used instead, same target 7/4",
    exact_x=(-1.0, 0.0, 1.0, 2.0),
)

# The exact curves: (L, chains, burn-in, samples per chain, interval, threads). Away from
# T_c tau is at most tens of sweeps.
EXACT = {
    "smoke": (16, 4, 300, 300, 2, 1),
    "small": (64, 8, 2000, 1000, 5, 2),
    "full": (256, 10, 5000, 4000, 10, 4),
}
ENERGY_T = (1.5, 1.7, 1.9, 2.0, 2.1, 2.5, 2.7, 3.0, 3.25, 3.5)
# Why the energy leaves out 2.1 < T < 2.5 (module doc), quoted in its row.
ENERGY_WINDOW = ("2.1 < T < 2.5 left out: towards T_c the finite lattice's energy approaches its "
                 "O(1/L) difference from Onsager's infinite-lattice energy (critical shift); at "
                 "the T used the exact correlation length is xi <= 6, so the finite-size "
                 "difference ~ exp(-L/xi) is below 1e-4 at L >= 64")
MAGNETIZATION_T_MAX = 2.1
EXACT_SEED = 8001


def runs(scale: str) -> list[Run]:
    L, chains, burn, samples, interval, threads = EXACT[scale]
    exact = Run("exact", MODEL.args + (
        "--dims", f"{L}x{L}", "--task", "stationary", "--observable", "m", "--burn-in",
        str(burn), "--samples", str(samples), "--interval", str(interval), "--replicas",
        str(chains), "--sweep", "temperature=" + ",".join(f"{T:g}" for T in ENERGY_T)),
        seed=EXACT_SEED, threads=threads)
    return fss_runs(MODEL, scale) + [exact]


# ---------------------------------------------------------------------------------------------

def _exact_check(scale: str, data: Path, figures: Path, kind: str) -> Check:
    t = sweep_table(data / "exact")
    L = EXACT[scale][0]
    T = np.asarray(t["temperature"], float)
    if kind == "energy":
        keep = np.ones(T.size, bool)
        col, theory, ylabel = "energy", ising_energy, "energy per site"
        quantity = "energy per site e_sim / e_exact (Onsager), weighted over T"
        source, name = "ising_energy", "ising.energy"
    else:
        keep = T <= MAGNETIZATION_T_MAX + 1e-12
        col, theory, ylabel = "abs_mean", onsager_magnetization, "<|m|>"
        quantity = "<|m|>_sim / m_exact (Onsager-Yang spontaneous magnetization), weighted over T"
        source, name = "ising_magnetization", "ising.magnetization"
    T = T[keep]
    sim = np.asarray(t[col], float)[keep]
    err = np.asarray(t[col + "_err"], float)[keep]
    exact = np.array([theory(x) for x in T])
    ratio = weighted_ratio(sim, err, exact)
    Ts = np.linspace(T.min(), T.max(), 300)
    fig = plot_ratio(T, ratio, "T", title=f"2D Ising, L = {L}: {ylabel} against the exact "
                                          "solution", prov=t.provenance,
                     absolute=(sim, err, Ts, [theory(x) for x in Ts], ylabel), logx=False)
    chains = int(np.min(t["chains"]))
    warnings = stationary_caveats(t)
    burn = burn_in_caveat(t)
    if burn is not None:
        warnings.append(burn)
    if kind == "energy":
        warnings.insert(0, ENERGY_WINDOW)
    return ratio_check(ratio, model=NAME, check=name, source=source, quantity=quantity,
                       sizes=f"L={L}, T=" + ",".join(f"{x:g}" for x in T),
                       statistics=f"{chains} chains x {t.provenance.config.get('samples')} "
                                  "samples per T; leave-one-chain-out errors",
                       warnings=warnings, report_consistency=True,
                       figure=save_figure(fig, figures, name))


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    d = read(MODEL, scale, data)
    checks, tc = crossing_checks(d, figures, binder_target=(*ISING_BINDER, "ising_binder"))
    exps = exponent_checks(d, figures)
    inv_nu = next(c for c in exps if c.check == "ising.inv_nu").measured
    checks += exps
    checks += collapse_checks(d, figures, tc, 1.0 / inv_nu)
    checks += [_exact_check(scale, data, figures, "energy"),
               _exact_check(scale, data, figures, "magnetization")]
    return checks
