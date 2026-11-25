"""Majority rule at zero temperature on the periodic square lattice (--model majority --tie
random): the probability of freezing into stripes and the coarsening law.

The rule as implemented (include/MajorityModel.hpp, the driver's ensemble task): a uniformly
random node takes the majority opinion of its 4 nearest neighbours and, on a 2-2 tie, either
opinion with probability 1/2; one sweep is N such updates. This is zero-temperature Glauber
dynamics of the Ising model (a spin flips with probability 1 if the flip lowers the energy, 1/2
if it leaves it unchanged, 0 otherwise), the dynamics of Spirin-Krapivsky-Redner (2001, 2002)
and Olejarz-Krapivsky-Redner (2012), with the same time unit (one attempt per spin per unit
time). The start is Barros-Krapivsky-Redner's quench from T = infinity: independent +-1 spins,
each +1 with probability 1/2 (--init bernoulli --up 0.5, ruling R18). The exactly balanced
start (--init random: N/2 spins up) is not used: it raises the stripe fraction measurably at
the sizes simulated (0.3665 +- 0.0076 at L = 64 in the first small run, +3.6 sigma; a
Bernoulli start gives 0.3380 +- 0.0047 in the reviewer's independent program).

majority.stripes: a replica stops when it is absorbed: at consensus, or frozen (the driver
checks exactly, after a sweep without change, that no node can change; with random ties a
frozen state has every node in a strict local majority). On the square torus the only frozen
states other than consensus are flat (horizontal or vertical) stripes of width >= 2, which have
`stripes` = 1. Diagonal stripes also have `stripes` = 1 but are not frozen under random ties:
their 2-2 boundary spins keep flipping until the stripe disappears (they reach the ground state
on a time scale ~ L^3, Olejarz et al.). So the fraction of replicas ending absorbed with
stripes = 1 is the fraction frozen into flat stripes, which Barros, Krapivsky and Redner (2009)
predict to be P_01 + P_10 = 0.3388... for L -> infinity (the critical continuum-percolation
crossing probabilities on the square torus, computed from Pinson's formula in
theory.torus_wrapping_probability). Each size is a binomial fraction; the largest L is the
measured value and the other sizes are reported as drift (ruling R17: no extrapolation).
The step cap (stripes_cap) makes the expected number of capped replicas over the whole run at
most 0.01; a capped replica (a diagonal state still on its way to consensus) counts as not
frozen in stripes, with a warning.

majority.coarsening (ruling R19): the domain size l(t) = 1/rho(t), rho the density of
unsatisfied (disagreeing) bonds, i.e. the excess energy per bond at T = 0, which scales as the
inverse domain size (Bray 1994). b in l = A t^b + c against 1/2 (Lifshitz-Allen-Cahn), the
additive constant being the leading, lattice-scale correction (Huse 1986). The window
t in [100, L^2/32] sweeps was fixed before the full run, after a reviewer's pilot
showed the bias of the `length` estimator (FIT_RULE). Errors: bootstrap over whole replicas
(odspy.coarsening_fit with the trajectories). Diagnostics in the warnings: the plain power fit
over the same window, the plain local slopes of half-decade windows, and the exponent of the
driver's `length` (where G(r) first drops to 1/2), which is not used because G subtracts each
sample's m^2, biasing the length low as domains grow (first small run: b = 0.459 +- 0.008).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from odspy import coarsening_fit, find_outputs
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import exponent_check, plot_exponent, save_figure
from validation.core import Check, Run
from validation.theory import MAJORITY_STRIPE_PROBABILITY

NAME = "majority"

# Stripe cap model: a fraction P_DIAGONAL of the replicas reaches a diagonal stripe, which
# decays as exp(-t / tau), tau = TAU_DIAG L^3 sweeps (pilots: 0.059(4) L^3 at L = 64,
# 0.05(2) L^3 at L = 128; OKR 2012: tau ~ L^3; P1 = 0.042 by percolation).
TAU_DIAG = 0.06
P_DIAGONAL = 0.042
CAPPED_EXPECTED = 0.01      # expected capped replicas over the whole stripes run, at most

T_LO = 100                  # sweeps: start of the coarsening fit window
WINDOW_DIVISOR = 32         # the window ends at L^2 / WINDOW_DIVISOR sweeps
FIT_RULE = (f"fit window fixed before the full run (ruling R19): t = {T_LO}..L^2/{WINDOW_DIVISOR} "
            f"sweeps, fit form l = A t^b + c; t < {T_LO} excluded because of early-time corrections "
            "(domains still near the lattice spacing)")

# Per scale. stripes: one entry per size, (L, replicas, block (replicas per job; None: no
# split), threads per job). coarsening: (L, replicas, block, threads).
SCALE = {
    "smoke": {
        "stripes": ((8, 300, None, 1), (12, 300, None, 1), (16, 300, None, 1)),
        "coarsening": (128, 16, None, 1),
    },
    "small": {
        "stripes": ((16, 4000, 1000, 1), (32, 4000, 1000, 1), (64, 4000, 500, 1)),
        "coarsening": (256, 60, 6, 1),
    },
    "full": {
        # L = 256: blocks of 20 replicas on 2 threads, ~25 min each unless a replica sits in a
        # long-lived diagonal stripe (see validation/README.md for the worst case)
        "stripes": ((16, 40000, None, 1), (32, 40000, 10000, 1), (64, 20000, 2000, 1),
                    (128, 20000, 500, 4), (256, 10000, 20, 2)),
        "coarsening": (512, 200, 25, 4),
    },
}

SEEDS = {"stripes": 1001, "coarsening": 1002}


def stripes_cap(L: int, replicas: int, sizes: int) -> int:
    """The step cap (sweeps) of the stripes run at size L: the smallest whole number of sweeps
    with sizes * replicas * P_DIAGONAL * exp(-cap / tau) <= CAPPED_EXPECTED, tau = TAU_DIAG L^3,
    so that over the `sizes` sizes of the run (each with at most `replicas`) fewer than 0.01
    replicas are expected to be capped."""
    tau = TAU_DIAG * L ** 3
    return math.ceil(tau * math.log(sizes * replicas * P_DIAGONAL / CAPPED_EXPECTED))


def _base(L: int) -> tuple[str, ...]:
    return ("--model", "majority", "--tie", "random", "--task", "ensemble", "--graph", "lattice",
            "--dims", f"{L}x{L}", "--boundary", "periodic", "--init", "bernoulli", "--up", "0.5")


def runs(scale: str) -> list[Run]:
    s = SCALE[scale]
    out = []
    n = len(s["stripes"])
    for L, R, block, threads in s["stripes"]:
        out.append(Run(f"stripes_L{L}", _base(L) + ("--steps", str(stripes_cap(L, R, n)),
                                                    "--observables", "stripes",
                                                    "--replicas", str(R)),
                       seed=SEEDS["stripes"] + L, threads=threads, block=block))
    L, R, block, threads = s["coarsening"]
    out.append(Run("coarsening", _base(L) + ("--steps", str(L * L // WINDOW_DIVISOR),
                                             "--log-times", "10", "--observables", "rho,length",
                                             "--save-trajectories", "true", "--replicas", str(R)),
                   seed=SEEDS["coarsening"], threads=threads, block=block))
    return out


# ---------------------------------------------------------------------------------------------

def _replicas(prefix: Path) -> Table:
    t = find_outputs(prefix).get("replicas")
    if t is None:
        raise RuntimeError(f"{prefix}: no replicas table")
    return t


def _stripes(scale: str, data: Path, figures: Path) -> Check:
    sizes, fracs, errs, counts, warns = [], [], [], [], []
    prov = None
    for L, *_ in SCALE[scale]["stripes"]:
        t = _replicas(data / f"stripes_L{L}")
        prov = t.provenance
        absorbed = np.asarray(t["absorbed"], float) > 0.5
        stripes = np.asarray(t["stripes"], float) > 0.5
        consensus = np.asarray(t["reached_consensus"], float) > 0.5
        n = int(absorbed.size)
        if n == 0:
            raise RuntimeError(f"{data / f'stripes_L{L}'}: no replicas")
        capped = int(np.sum(~absorbed))
        if capped:
            warns.append(f"L={L}: {capped} of {n} replicas capped at --steps (not absorbed; "
                         f"{int(np.sum(stripes & ~absorbed))} of them in a stripe state), counted "
                         "as not frozen in stripes (diagonal states on their way to consensus)")
        # absorbed: consensus (stripes = 0) or frozen flat stripes (stripes = 1)
        odd = int(np.sum(absorbed & (stripes == consensus)))
        if odd:
            warns.append(f"L={L}: {odd} absorbed replicas neither at consensus with stripes = 0 "
                         "nor frozen with stripes = 1")
        k = int(np.sum(stripes & absorbed))
        p = k / n
        sizes.append(L)
        fracs.append(p)
        Pc = MAJORITY_STRIPE_PROBABILITY
        errs.append(math.sqrt(p * (1 - p) / n) if 0 < k < n else math.sqrt(Pc * (1 - Pc) / n))
        counts.append(n)
        if k in (0, n):
            warns.append(f"L={L}: fraction {p:g}; binomial error under the prediction used")
    P = MAJORITY_STRIPE_PROBABILITY
    drift = ", ".join(f"L={L} {p:.4f} ± {e:.2g}" for L, p, e in zip(sizes, fracs, errs))
    warns.insert(0, f"stripe fraction at every L (drift): {drift}; the largest L is the measured "
                    "value (ruling R17: no extrapolation); a residual finite-size shift may remain")
    name = "majority.stripes"
    fig, axes = new_fig()
    ax = axes[0, 0]
    x = 1.0 / np.asarray(sizes, float)
    ax.errorbar(x, fracs, yerr=errs, fmt="o", ms=4, color=color(0), capsize=0, label="simulation")
    ax.axhline(P, color=color(1), lw=1, label=f"percolation crossing P01 + P10 = {P:.4f}")
    ax.set(xlim=(0, 1.1 * x.max()), xlabel="1 / L", ylabel="fraction frozen in stripes")
    ax.legend()
    fig = finish(fig, prov, "majority rule at T = 0, periodic L x L: frozen stripes")
    return Check(model=NAME, check=name,
                 quantity="fraction of replicas frozen into flat (horizontal or vertical) stripes "
                          "at the largest L, from independent +-1 spins with P(+1) = 1/2 (a quench "
                          "from T = infinity); absorbed with stripes = 1 (diagonal stripes are not "
                          "frozen under random ties and decay to the ground state)",
                 measured=fracs[-1], error=errs[-1], predicted=P, predicted_error=0.0,
                 source="majority_stripes", sizes="L=" + ",".join(map(str, sizes)),
                 statistics=f"{counts[-1]} replicas at L={sizes[-1]} (replicas per L: "
                            + ",".join(map(str, counts)) + "); binomial errors",
                 warnings=warns, figure=save_figure(fig, figures, name))


def _inverse(v):
    return 1.0 / v


def _local_slopes(prefix: Path, t: np.ndarray, lo: float, hi: float, observable: str,
                  transform) -> list[str]:
    """Plain power-law slopes over successive half-decade windows of [lo, hi], bootstrap errors."""
    edges = [e for e in lo * 10 ** (0.5 * np.arange(0, 2 * math.log10(hi / lo) + 1))
             if e < hi * (1 - 1e-9)] + [hi]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.sum((t >= a) & (t <= b * (1 + 1e-9))) >= 3:
            r = coarsening_fit(prefix, observable, model="power", xrange=(a, b * (1 + 1e-9)),
                               transform=transform)
            out.append(f"{r.params['b']:.3f} ± {r.errors['b']:.2g} (t = {a:.3g}..{b:.3g})")
    return out


def _coarsening(scale: str, data: Path, figures: Path) -> Check:
    L, R, _, _ = SCALE[scale]["coarsening"]
    prefix = data / "coarsening"
    out = find_outputs(prefix)
    ts = out.get("timeseries_ensemble")
    if ts is None or out.get("trajectories") is None:
        raise RuntimeError(f"{prefix}: needs the ensemble time series and trajectories")
    t = np.asarray(ts["time"], float)
    lo, hi = float(T_LO), float(L * L // WINDOW_DIVISOR)
    window = (lo, hi * (1 + 1e-9))
    inside = (t >= lo) & (t <= window[1])
    ell = 1.0 / np.asarray(ts["rho_mean"], float)
    warns = [FIT_RULE + f": t = {lo:g}..{hi:g} sweeps, {int(inside.sum())} recorded times, "
             f"l = 1/rho = {ell[inside][0]:.3g}..{ell[inside][-1]:.3g} (proportional to the "
             "domain size)"]
    running = np.asarray(ts["running_fraction"], float)[inside]
    if np.any(running < 1):
        warns.append(f"some replicas were absorbed inside the window (running fraction down to "
                     f"{running.min():.3g}); their final rho enters the mean")
    r = coarsening_fit(prefix, "rho", model="power+c", xrange=window, transform=_inverse)
    rp = coarsening_fit(prefix, "rho", model="power", xrange=window, transform=_inverse)
    warns.append(f"diagnostic: plain power fit l = A t^b over the same window: b = "
                 f"{rp.params['b']:.3f} ± {rp.errors['b']:.2g}")
    local = _local_slopes(prefix, t, lo, hi, "rho", _inverse)
    if local:
        warns.append("diagnostic: plain local slopes of l = 1/rho over half-decade windows "
                     "(drift): " + ", ".join(local))
    if "length_mean" in ts.columns:
        rl = coarsening_fit(prefix, "length", model="power", xrange=window)
        warns.append(f"`length` not used: G(r) subtracts each sample's per-sample m^2, which biases "
                     f"the length low as domains grow; over the same window its plain power fit "
                     f"gives b = {rl.params['b']:.3f} ± {rl.errors['b']:.2g} (1/rho: "
                     f"{rp.params['b']:.3f})")
    warns.append("the estimator 1/rho and the fit form were fixed after the reviewer's pilot "
                 "showed the `length` bias, before the full run (ruling R19)")
    name = "majority.coarsening"
    fig = plot_exponent(r, 0.5, "t (sweeps)", "l = 1/rho", prov=ts.provenance,
                        title=f"majority rule at T = 0, {L}x{L}: coarsening length")
    chk = exponent_check(r, 0.5, model=NAME, check=name, source="majority_coarsening",
                         quantity="b in l(t) = A t^b + c, l = 1/rho (rho: density of disagreeing "
                                  "nearest-neighbour bonds, the excess energy per bond; t in "
                                  "sweeps)",
                         sizes=f"L={L} (2D), t = {lo:g}..{hi:g} sweeps",
                         statistics=f"{R} replicas; bootstrap over whole replicas",
                         warnings=warns, figure=save_figure(fig, figures, name))
    return chk


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_stripes(scale, data, figures), _coarsening(scale, data, figures)]
