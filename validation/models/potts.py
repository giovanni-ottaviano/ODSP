"""The 3-state Potts model on the periodic square lattice (--model potts --q 3, J = 1, the
default heat-bath dynamics, ordered start --init all --value 0), against the exact critical
point and exponents.

The order parameter is the driver's observable `order` = (q f_max - 1)/(q - 1), f_max the
largest of the q opinion shares: 0 for equal shares, 1 when ordered. It is max_i of the
projections of the q-state order-parameter vector on the q state directions, homogeneous of
degree 1 in that vector, so its moments scale with the magnetic exponents (beta/nu, gamma/nu)
of the vector order parameter. The stationary sweep columns are computed from it as for m:
abs_mean = <order> (it is >= 0), binder = 1 - <order^4>/(3 <order^2>^2) (2/3 when ordered; a
non-zero constant in the disordered phase, since order >= 0: still a dimensionless ratio
whose curves cross at T_c), abs_susceptibility = susceptibility = N(<order^2> - <order>^2).

Finite-size scaling as in validation/models/_critical.py with T = T_c + x L^(-6/5), the same
x for every size (no collapse, so no offsets):
potts.tc: the crossing of the Binder cumulants of the two largest L, against
T_c = 1/ln(1 + sqrt 3).
potts.inv_nu: |dU/dT| at the exact T_c ~ L^(1/nu), 1/nu = 6/5.
potts.gamma_nu: the peak of the susceptibility ~ L^(26/15).
potts.beta_nu: <order> at the exact T_c ~ L^(-2/15).
Fit ranges: the campaign rule, fixed before the full run (validation/models/_critical.py):
every exponent fit uses all sizes but the smallest (L >= 12 at smoke and small, >= 24 at
full); the crossing is that of the two largest sizes. Finite-size corrections are larger than
for the Ising model; the fits over all sizes, the effective exponents of successive sizes and
the crossings of every pair are reported in the warnings as a finite-size diagnostic.
"""

from __future__ import annotations

from pathlib import Path

from validation.core import Check, Run
from validation.models._critical import (Model, Stats, crossing_checks, exponent_checks,
                                         fss_runs, read)
from validation.theory import POTTS3_EXPONENTS, POTTS3_TC

NAME = "potts"

# tau(L) of the order parameter at T_c, heat bath, in sweeps: 80, 386, 1640 at L = 16, 32, 64
# (8 chains of 20000 samples each, interval 5): 0.2 L^2.18.
MODEL = Model(
    name=NAME,
    args=("--model", "potts", "--q", "3", "--init", "all", "--value", "0"),
    observable="order",
    tc=POTTS3_TC,
    nu=POTTS3_EXPONENTS["nu"],
    gamma_nu=POTTS3_EXPONENTS["gamma/nu"],
    beta_nu=POTTS3_EXPONENTS["beta/nu"],
    tc_source="potts_tc",
    exp_source="potts_exponents",
    # x = (T - T_c) L^(6/5): U falls from 2/3 to its disordered plateau across [-2, 3]; the
    # susceptibility peaks sharply at x ~ 1 (scans at L = 12, 24): 0.5, 1, 1.5 resolve it
    # (with steps of 1 the parabola's error depends on where the peak falls between points,
    # which moves with L through the corrections); x = +-0.5 give the slope at T_c (the U
    # errors grow fast for x > 0, so +-1 would not be more precise).
    grid=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    tau=(0.2, 2.18),
    scales={
        "smoke": Stats(sizes=(8, 12, 16, 24), chains=8, n_eff=1000, burn_tau=30, threads=1),
        "small": Stats(sizes=(6, 12, 24, 48), chains=8, n_eff=600, burn_tau=30, threads=2),
        "full": Stats(sizes=(12, 24, 48, 96, 192), chains=16, n_eff=1500, burn_tau=30,
                      threads=8),
    },
    seed=8200,
    title="2D 3-state Potts (heat bath)",
    order_label="<order>",
    corrections="finite-size corrections are not fitted (they are larger than for the Ising "
                "model): the drift of the crossings and the effective exponents of successive "
                "sizes above show their size",
)


def runs(scale: str) -> list[Run]:
    return fss_runs(MODEL, scale)


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    d = read(MODEL, scale, data)
    checks, _ = crossing_checks(d, figures)
    return checks + exponent_checks(d, figures)
