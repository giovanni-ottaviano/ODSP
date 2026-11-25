"""Majority rule at T = 0 (validation/models/majority.py): frozen stripes and coarsening."""

from __future__ import annotations

import math

from validation.core import _value
from validation.models import MODELS
from validation.theory import MAJORITY_STRIPE_PROBABILITY

from .smoke import run_smoke

CHECKS = ["majority.stripes", "majority.coarsening"]


def test_majority_smoke(tmp_path, odsp_bin):
    assert "majority" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "majority", CHECKS)
    by = {r["check"]: r for r in rows}
    s = by["majority.stripes"]
    assert float(s["predicted"]) == MAJORITY_STRIPE_PROBABILITY
    assert s["source"] == "majority_stripes"
    assert "binomial" in s["statistics"]
    assert "independent" in s["quantity"]              # R18: the T = infinity start of BKR
    # R17: the largest L is the measured value, every L as drift, residual-shift caveat
    assert "stripe fraction at every L" in s["warnings"]
    assert "residual finite-size shift" in s["warnings"]
    assert "capped" not in s["warnings"]               # every replica absorbed at smoke scale
    c = by["majority.coarsening"]
    assert float(c["predicted"]) == 0.5
    assert c["source"] == "majority_coarsening"
    # R19: l = 1/rho, l = A t^b + c over t in [100, L^2/32]; plain power and local slopes as
    # diagnostics; why `length` is not used, with its measured exponent
    assert "1/rho" in c["quantity"] and "A t^b + c" in c["quantity"]
    assert "t = 100..512 sweeps" in c["sizes"]          # smoke L = 128: L^2/32 = 512
    assert "fit window fixed before the full run" in c["warnings"]
    assert "early-time corrections" in c["warnings"]
    assert "diagnostic: plain power" in c["warnings"]
    assert "local slopes" in c["warnings"]
    assert "`length` not used" in c["warnings"] and "per-sample m^2" in c["warnings"]


def test_majority_runs_match_the_cited_protocol():
    mod = MODELS["majority"]
    for scale in ("smoke", "small", "full"):
        sizes = [L for L, *_ in mod.SCALE[scale]["stripes"]]
        runs = {r.name: r for r in mod.runs(scale)}
        assert set(runs) == {f"stripes_L{L}" for L in sizes} | {"coarsening"}
        for r in runs.values():
            a = r.args
            # zero-temperature Glauber: majority with random tie-breaking, periodic L x L, from
            # independent +-1 spins with P(+1) = 1/2 (a quench from T = infinity, R18)
            assert _value(a, "--model") == "majority" and _value(a, "--tie") == "random"
            assert _value(a, "--boundary") == "periodic" and _value(a, "--init") == "bernoulli"
            assert _value(a, "--up") == "0.5" and _value(a, "--task") == "ensemble"
        for L, R, *_ in mod.SCALE[scale]["stripes"]:
            st = runs[f"stripes_L{L}"]
            assert L % 2 == 0 and _value(st.args, "--dims") == f"{L}x{L}"
            assert int(_value(st.args, "--steps")) == mod.stripes_cap(L, R, len(sizes))
            assert _value(st.args, "--observables") == "stripes"
        co = runs["coarsening"]
        L = mod.SCALE[scale]["coarsening"][0]
        assert _value(co.args, "--observables") == "rho,length"
        assert _value(co.args, "--save-trajectories") == "true"
        assert int(_value(co.args, "--steps")) == L * L // 32
    full = [L for L, *_ in mod.SCALE["full"]["stripes"]]
    assert len(full) >= 4 and max(full) >= 8 * min(full)


def test_stripes_cap_rule():
    mod = MODELS["majority"]
    # expected capped replicas over the run: sizes x R x P(diagonal) x exp(-cap / tau) <= 0.01,
    # tau = TAU_DIAG L^3 sweeps
    for L, R, n in ((64, 4000, 3), (256, 10000, 5)):
        cap = mod.stripes_cap(L, R, n)
        tau = mod.TAU_DIAG * L ** 3
        assert n * R * mod.P_DIAGONAL * math.exp(-cap / tau) <= 0.01
        assert n * R * mod.P_DIAGONAL * math.exp(-(cap - tau) / tau) > 0.01     # not wasteful
    # a normal job at L = 256 stays short: small replica blocks
    L, R, block, threads = mod.SCALE["full"]["stripes"][-1]
    assert L == 256 and block <= 25
