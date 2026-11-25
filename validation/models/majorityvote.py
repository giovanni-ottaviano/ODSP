"""The majority-vote model with noise on the periodic square lattice (--model majority-vote,
ordered start --init up), against the critical noise of Yu (2017) and the 2D Ising class.

Rule match. include/MajorityVoteModel.hpp: a uniformly random node takes the majority opinion
of its 4 nearest neighbours (RegularLattice, periodic: 2D = 4 neighbours, the node itself not
counted) with probability 1 - noise and the minority opinion otherwise; on a tie it takes
either opinion with probability 1/2; one sweep (the driver's time unit for --burn-in and
--interval) is N such random sequential updates. This is Oliveira's (1992) flip probability
w(s_i -> -s_i) = (1/2)[1 - (1 - 2q) s_i sgn(sum_j s_j)], sgn(0) = 0: a node agreeing with
the majority flips with probability q, one disagreeing with probability 1 - q, a tied one with
1/2, so after the update it holds the majority opinion with probability 1 - q. Yu (2017)
uses the same Eq. (1), random sequential updates (each site one chance per Monte Carlo step on
average) and periodic L x L lattices, and finds q_c = 0.07518(3) (theory.MV_QC, the value and
predicted error of mv.qc; Oliveira 1992: 0.075(1)).

Finite-size scaling as in validation/models/_critical.py, with the noise q as the control
parameter (low q ordered, as low T): one stationary noise sweep per L at q = q_c + x / L
(1/nu = 1), the same x for every size (no collapse, so no offsets), q_c the cited value.
mv.qc: the crossing of the Binder cumulants U = 1 - <m^4>/(3 <m^2>^2) of the two largest L.
mv.binder (ruling R14): U at that crossing against U* of the 2D Ising class on the periodic
square lattice (Salas & Sokal), universal within the class for this geometry.
mv.inv_nu: |dU/dq| at the cited q_c ~ L^(1/nu), 1/nu = 1.
mv.gamma_nu (ruling R13, fixed before any exponent was fitted): chi' = N(<m^2> - <|m|>^2) at
the fixed scaling variable (q - q_c) L = 0.2 ~ L^(7/4), as ising.gamma_nu (the MV is in the 2D
Ising class and has the same flat chi' maximum, so odspy.peak_vs_size is not used). x = 0.2
is where chi' was largest in pilot scans at L = 16 (q = 0.050..0.110 in steps of 0.005: chi'
12.3 at q = 0.090, x = 0.24; 11.8 at x = 0.16) and L = 32 (q = 0.065..0.095 in steps of
0.0025: 42.1 at x = 0.15, 41.6 at x = 0.23), chosen before the campaign runs.
mv.beta_nu: <|m|> at the cited q_c ~ L^(-1/8).
q_c is a measured value, not exact: the shift of each exponent for q_c +- 3e-5 is estimated
from the data (_critical.critical_point_shift; for 1/nu from the slopes of U at x = +-0.1) and
reported in its warnings, not in the error bar (Yu 2017: a 0.1% change of q_c moves gamma/nu
by ~2% and beta/nu by ~8% over L <= 260).
Fit ranges: the campaign rule (R12), fixed before the full run: every exponent fit uses all
sizes but the smallest (L >= 12 at smoke, >= 16 at small, >= 32 at full); the crossing is
that of the two largest sizes; the fits over all sizes, the effective exponents of successive
sizes and the crossings of every pair are in the warnings as a finite-size diagnostic.
"""

from __future__ import annotations

from pathlib import Path

from validation.core import Check, Run
from validation.models._critical import (Model, Stats, crossing_checks, exponent_checks,
                                         fss_runs, read)
from validation.theory import ISING_BINDER, ISING_EXPONENTS, MV_QC

__all__ = ["NAME", "runs", "analyse"]

NAME = "mv"

# tau(L) of |m|, in sweeps. tau peaks on the disordered side near x = 0.1-0.15, not at q_c,
# so the model is calibrated on the largest tau_abs measured anywhere on the grid
# (TAU_MEASURED_MAX). At q_c alone: 56, 331, 1000 at L = 16, 32, 64 (8 chains of 20000
# samples). The 9 grid points of the two small runs (seeds 9100 and 19100, 8 chains of ~630
# samples) gave maxima of 16.2, 65.9, 328 and 1637 at L = 8, 16, 32, 64, each a max of 9
# estimates with ~17% statistical error. Extra runs at L = 128 (10 chains of 1600 samples,
# interval 200, 50000 sweeps of burn-in) gave 4180 at x = 0.1 and 4460 at x = 0.15. The model
# 0.20 L^2.17 (z of Ising Metropolis, the MV being in the Ising class) covers every one of
# them: 18, 82, 369, 1661 and 7490 at L = 8..128, tight at L = 64 and ~1.7x the L = 128
# runs. Each point's burn-in is also checked against its own measured tau(|m|)
# (_critical.burn_in_caveat).
TAU_MEASURED_MAX: dict[int, float] = {8: 16.2, 16: 65.9, 32: 327.5, 64: 1637.2, 128: 4460.0}

MODEL = Model(
    name=NAME,
    args=("--model", "majority-vote", "--init", "up"),
    observable="m",
    tc=MV_QC[0],
    nu=ISING_EXPONENTS["nu"],
    gamma_nu=ISING_EXPONENTS["gamma/nu"],
    beta_nu=ISING_EXPONENTS["beta/nu"],
    tc_source="mv_qc",
    exp_source="mv_exponents",
    # x = (q - q_c) L: U falls from its ordered value 2/3 (x <= -0.3) towards 0 across
    # [-0.3, 0.45] (pilot scans at L = 16, 32); x = +-0.1 give the slope at q_c; chi' is
    # largest, and flat, around x = 0.2 (module doc). Every point is exact (no collapse).
    grid=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.15, 0.2, 0.3, 0.45),
    tau=(0.20, 2.17),
    scales={
        "smoke": Stats(sizes=(8, 12, 16, 24), chains=8, n_eff=1000, burn_tau=30, threads=1),
        "small": Stats(sizes=(8, 16, 32, 64), chains=8, n_eff=500, burn_tau=30, threads=2),
        "full": Stats(sizes=(16, 32, 64, 128, 256), chains=16, n_eff=1200, burn_tau=30,
                      threads=8),
    },
    seed=9100,
    title="2D majority vote (Oliveira)",
    order_label="<|m|>",
    corrections="finite-size corrections are not fitted: the drift of the crossings and the "
                "effective exponents of successive sizes above show their size",
    gamma_at=0.2,
    gamma_note="peak_vs_size not used (ruling R13: the MV is in the 2D Ising class, whose chi' "
               "maximum is too flat for the 3-point parabola, see ising.gamma_nu); chi' at "
               "fixed (q - q_c) L^(1/nu) = 0.2 used instead, same target 7/4; x = 0.2 fixed "
               "from pilot scans at L = 16, 32 (chi' largest at x ~ 0.15-0.24) before any "
               "exponent fit",
    control="noise",
    control_label="q",
    tc_error=MV_QC[1],
)

# Why U* of the Ising model is the MV's prediction (R14), quoted in the mv.binder row.
BINDER_NOTE = ("U* is that of the 2D Ising class on the periodic square L x L lattice (Salas & "
               "Sokal): universal for the models of the class with the same geometry and "
               "boundary conditions (Privman & Fisher 1984), and the MV is in the class "
               "(Oliveira 1992, Yu 2017); it assumes isotropic large-scale correlations, as "
               "the square-lattice symmetry gives")


def runs(scale: str) -> list[Run]:
    return fss_runs(MODEL, scale)


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    d = read(MODEL, scale, data)
    checks, _ = crossing_checks(d, figures, binder_target=(*ISING_BINDER, "mv_binder"),
                                binder_note=BINDER_NOTE)
    return checks + exponent_checks(d, figures)
