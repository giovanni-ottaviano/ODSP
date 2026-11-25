"""Bounded-confidence models on the complete graph: Deffuant (--model deffuant) and
Hegselmann-Krause (--model hk). Consensus thresholds and major-cluster counts.

The rules as implemented (include/DeffuantModel.hpp, include/HegselmannKrauseModel.hpp,
include/BoundedConfidenceModel.hpp), with opinions uniform on [0, 1] (--init uniform):
  * Deffuant: one update draws a uniform directed edge (i, j) -- on the complete graph a
    uniform ordered pair -- and if |x_i - x_j| < eps both move by mu (x_j - x_i), mu = 1/2
    (they meet halfway); a sweep is N updates. This is the process whose rate equation Ben-Naim,
    Krapivsky and Redner (BKR 2003) integrate: random pairs, strict |x1 - x2| < 1, midpoint
    compromise, opinions uniform on [-Delta, Delta]; Delta = 1/(2 eps).
  * Hegselmann-Krause: an agent moves to the mean of the opinions within eps of its own (itself
    included, |x_j - x_i| <= eps). --update synchronous (the original model, the driver's
    default): every agent at once, from the same old opinions; one round is one sweep.
    --update ordered (ruling R23, a core feature the user approved for this check): agents one
    at a time from the current opinions, every agent exactly once per sweep in index order --
    with i.i.d. initial opinions a random order fixed for the whole run, Fortunato's ordered
    sweeps. --update sweep (ruling R20): the same with a fresh random order each sweep (a
    diagnostic here). (--update random, with replacement, is not used: in a sweep it leaves
    about N/e agents un-updated, and an agent not updated within the first few sweeps can be
    left behind by the clusters that form and move away.)
  * Stop: a replica stops when converged -- after a sweep's worth of updates without a change
    larger than --tolerance, every pair that can still interact agrees within it (exact test,
    so it never stops before the final clusters are formed) -- or at the step cap --steps.
  * Clusters (include/OpinionClusters.hpp): sorted opinions split wherever two consecutive
    values differ by more than 1e-3; the driver writes every replica's final clusters
    (prefix_clusters.csv: position, size), the observable `clusters` counts those holding at
    least 1% of the nodes and `largest_cluster` is the share of the largest one.

Deffuant: the step cap. Converging a pair of isolated stragglers (two agents within eps of each
other and of nobody else) takes O(N) sweeps -- they meet once every ~N/2 sweeps -- while the
clusters holding a finite fraction of the agents form within ~100 sweeps (pilot: median stop
~70 sweeps without stragglers). The Deffuant runs therefore stop at D_CAP = 1000 sweeps, and
every replica's `clusters` and `largest_cluster` are recorded at log-spaced times: a replica
whose number of clusters with >= 1% of the nodes still changes between SETTLE_FROM and D_CAP
sweeps is reported in the warnings. Capped replicas are counted in the warnings, not discarded.
The reviewer replayed one point without the cap (N = 8000, eps = 0.27042, 400 replicas of which
150 capped): no replica's major count changed.

Consensus definitions. BKR call a cluster major when its mass exceeds 1 in their units (mass
2 Delta in all), i.e. when it holds more than a fraction 1/(2 Delta) = eps of the agents, and
minor when its mass is below 1e-2 (-> 3e-4: a fraction ~1e-4 of the agents); the central
cluster in between. Two definitions of consensus follow, and they have different thresholds:
  * one major cluster (exactly one cluster with more than eps N agents): BKR, the central
    cluster splits into two major clusters at Delta_2(1) = 1.871, eps* = 1/(2 x 1.871) =
    0.26724; error 1.8e-4 from half their integration mesh (0.00125 in Delta). This is the
    consensus transition of Lorenz (2007, minor clusters neglected) and of the "eps_c ~ 0.27"
    quoted for Deffuant. ODSP's `clusters == 1` (no other cluster with >= 1% of the nodes) is
    reported next to it: near eps* the minor clusters hold far less than 1% (BKR), so the two
    agree there.
  * complete consensus, no minor clusters (largest_cluster = 1): eps_c = 1/2 (BKR, Delta = 1;
    Fortunato 2004, "all agents in one cluster"). Approached slowly: a minor edge cluster of
    mass ~ (Delta - 1)^3 (BKR) survives at finite N unless N (Delta - 1)^3 << 1, so the half
    point sits far below 1/2 at accessible N (Fortunato 2004, N = 10^4: ~0.48). It is measured
    as a diagnostic (its half point at every N in deffuant.threshold's warnings), not compared.

deffuant.threshold: eps where P(one major cluster) = 1/2 at the largest N, by linear
interpolation at the first upward crossing of 1/2 on the eps grid of that N; error by a
parametric bootstrap of the binomial P at each grid point (N_BOOT draws). Ruling R17: the half
points at the other N are reported as drift, with the caveat that a residual finite-size shift
may remain; no extrapolation. Grid: eps = eps* + u h(N), u = -2..4 in steps of 1/2,
h(N) = 0.006 (1000/N)^(1/2) (pilot at N = 1000, 4000: the transition narrows as N^(-1/2) and
its half point sits ~1-2 h above eps*). Ruling R22: the row stays flagged when the half point
at the largest N misses eps*; it gives the gap absolutely and relative to eps*, the monotone
drift over N, that no finite-N form is published (so nothing is extrapolated), the reviewer's
independent reproduction of the probabilities, and that BKR's error covers their Delta mesh
only (not their opinion grid of 400 states per unit, nor the rounding of 1.871).

deffuant.clusters_n2, _n4, _n6: replace the agreed target "major clusters ~ 1/(2 eps)". BKR's
form is a pattern periodic in Delta with period L ~ 2.155: 1/(L eps) major clusters
asymptotically, i.e. spaced L eps apart, not 2 eps, and exactly the count of their bifurcation
table at finite Delta. One row per point (ruling R21): the mean number of major clusters (more
than eps N agents) over the replicas at the largest N against BKR's exact count, at the centre
(in Delta) of each "even window" -- from the k-th type-2 bifurcation to the next type-1 one,
where the final state is 2k major clusters and only minor ones besides
(theory.deffuant_even_windows): in BKR's rate equation the count there does not depend on the mass cut (at finite N it
can: see the n4 note below), whereas in the
odd windows the central cluster's mass crosses every cut. Points: eps = 0.1938, 0.1044, 0.0718
(2, 4, 6 major clusters). The fourth window (8 clusters, eps = 0.0548) is not used: its upper
end is not in BKR's table (it was built by adding one period), so no published prediction
exists there -- a restriction decided after seeing the data (ruling R21), for that source-based
reason. Errors from the replica spread; a point where every replica has the same count gets
the variance of a count off by one in one replica of R, so that no error is zero. Every N's
value in the warnings (drift).

hk.threshold: eps where P(complete consensus) = 1/2 at the largest N (as deffuant.threshold;
R17), with --update ordered. Consensus as Fortunato (2005): every agent with the same opinion
within 1e-9 -- the driver's reached_consensus with the HK default tolerance 1e-9. Fortunato
updated the agents one at a time in ordered sweeps, the same order in every sweep: --update
ordered. He writes that random updating "would not have influence on the final number of
opinion clusters", with a footnote that on special graphs it can change the probability of
complete consensus; he makes no claim about P(consensus) on the complete graph, and there the
order does matter: the reviewer measured P(consensus) = 0.585 +- 0.025 (fixed order) against
0.415 +- 0.025 (fresh random order each sweep) at N = 2000, eps = 0.2064, and synchronous
updating shifts the half point further (0.2046 ordered, 0.2086 synchronous at N = 2000). The
half points in random-order sweeps (hk_fresh runs) and synchronous rounds (hk_sync runs, the
same grid) are reported in the warnings as diagnostics of this update-scheme dependence.
The threshold depends on N (Fortunato; Lorenz 2007: no upper bound on convergence times as N
grows), so it is compared at the source's N: Fortunato's largest, N = 10000, is the largest N
at full scale. His interval [0.195, 0.202] brackets the N -> infinity step; the measured value
is the N = 10000 half point compared against it, and reading the bracket as 0.1985 +- 0.0020
(midpoint, half-width / sqrt(3)) is a modelling choice. Grid: eps = 0.2 + u h(N), u = -3..4
in steps of 1/2, h(N) = 0.006 (1000/N)^(1/2).
Step cap of the sequential modes: H_SWEEP_CAP sweeps (pilot: longest run ~800 sweeps at
N = 4000).

hk.clusters: the 2R conjecture (Blondel, Hendrickx and Tsitsiklis 2007, 2009): clusters spaced
roughly 2 eps apart, ~1/(2 eps) of them; BHT 2009 observe, with synchronous updating, a number
of clusters growing linearly in 1/eps "with a coefficient slightly smaller than 1/2" (spacing
~2.2 eps); Wang et al. (2017) estimate 1/(2.29 eps) from the linear stability of the
mean-field (continuum, continuous-time) equation, which has no update order. So this check
keeps --update synchronous. No value with an error exists, so the check is one-sided
(Check.one_sided = "below"): the slope b of a weighted line fit n = a + b / eps of the mean
number of clusters (all clusters, every size, as BHT count them) must lie below 1/2, passed
when sigma < -3. It detects overcounting only: a count that is too low passes. Points: 15
values of 1/(2 eps) uniformly spaced in [3, 10] (eps = 0.05..0.167): from three clusters,
below which the consensus and polarisation transitions dominate, to BKR's Delta = 10 as for
Deffuant; uniform in 1/eps so that the fit averages over the steps of the count. The implied
spacing 1/b and the comparison with Wang et al.'s 1/2.29 are in the warnings.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from odspy import find_outputs, fit
from odspy.io import Outputs
from odspy.plots.style import color, finish, new_fig

from validation.analysis import fit_quality, save_figure
from validation.core import Check, Run
from validation.theory import (BKR_PERIOD, deffuant_asymptotic_clusters, deffuant_major_threshold,
                               deffuant_window_points, hk_threshold)

NAME = "boundedconfidence"

D_CAP = 1000                 # sweeps: Deffuant step cap (see the module docstring)
SETTLE_FROM = 316            # recorded sweep (--log-times 2) from which nothing may change
H_CAP = 100_000              # rounds: synchronous HK cap (pilot: longest ~2700 rounds at N = 16000)
H_SWEEP_CAP = 20_000         # sweeps: HK ordered/sweep cap (pilot: longest ~800 sweeps, N = 4000)
GRID_U_D = tuple(k / 2 for k in range(-4, 9))        # deffuant threshold: eps* + u h(N)
GRID_V_C = tuple(k / 2 for k in range(0, 9))         # complete consensus: 1/2 - v h_c(N)
GRID_U_H = tuple(k / 2 for k in range(-6, 9))        # hk threshold: 0.2 + u h(N)
HK_COUNT_X = tuple(3.0 + 0.5 * k for k in range(15))  # 1/(2 eps) of the hk count points
HK_CENTRE = 0.2
N_BOOT = 2000
WANG_SLOPE = 1 / 2.29        # Wang et al. (2017): clusters ~ 1/(2.29 eps)

# Findings of the Task 11 review about one row, carried into it as statements.
REVIEW_NOTES = {4: ["Task 11 review: the excess is a finite-N branch -- the central cluster's mass "
                    "is bimodal across replicas (at N = 8000 about 80% below mass 0.1 and 18% above 1, BKR "
                    "units); the share of replicas with a central cluster of mass > 1 falls from 26% "
                    "to 18% between N = 2000 and 8000. The count itself depends on the mass cut on "
                    "that branch (neighbouring clusters near mass 1): at N = 8000 the mean major "
                    "count is 4.155 / 4.103 / 4.048 for cuts 0.3 / 1 (BKR's own, used here) / 1.4, "
                    "and the replicas whose count changes between cuts 0.3 and 1.4 fall from 90 to "
                    "39 of 400 between N = 1000 and 8000; no ODSP bug"]}

# Per scale and model: sizes N; replicas per threshold point, per count point (and, Deffuant,
# per complete-consensus point; HK, per point of the synchronous diagnostic); block: replicas
# per job (None: no split), seq_block for HK's ordered and sweep runs. Smoke: seconds; small: a few minutes at 10 threads; full: >= 4
# sizes over a factor >= 8. HK's ordered and sweep modes cost O(N) per update on the complete
# graph (sorted-block mean), ~0.5 s per replica at N = 4000 near the threshold.
SCALE = {
    "smoke": {"deffuant": {"sizes": (200, 400), "replicas": 60, "count": 40, "complete": 40,
                           "block": None},
              "hk": {"sizes": (200, 400), "replicas": 60, "count": 40, "sync": 60, "fresh": 60,
                     "block": None, "seq_block": None}},
    "small": {"deffuant": {"sizes": (1000, 2000, 4000, 8000), "replicas": 400, "count": 400,
                           "complete": 200, "block": 100},
              "hk": {"sizes": (500, 1000, 2000, 4000), "replicas": 400, "count": 400,
                     "sync": 1000, "fresh": 400, "block": 100, "seq_block": 25}},
    "full": {"deffuant": {"sizes": (2000, 4000, 8000, 16000, 32000), "replicas": 4000,
                          "count": 2000, "complete": 1000, "block": 500},
             "hk": {"sizes": (1250, 2500, 5000, 10000), "replicas": 2000, "count": 2000,
                    "sync": 2000, "fresh": 500, "block": 250, "seq_block": 250}},
}

SEEDS = {"deffuant_threshold": 1111, "deffuant_complete": 1112, "deffuant_clusters": 1113,
         "hk_threshold": 1124, "hk_clusters": 1122, "hk_sync": 1123, "hk_fresh": 1121}


def _h(N: int) -> float:
    return 0.006 * math.sqrt(1000.0 / N)


def _decimals(values) -> str:
    return ",".join(f"{round(v, 9):.9f}" for v in values)


def deffuant_grid(N: int) -> list[float]:
    eps, _ = deffuant_major_threshold()
    return [eps + u * _h(N) for u in GRID_U_D]


def complete_grid(N: int) -> list[float]:
    h = 0.02 * (1000.0 / N) ** (1.0 / 3.0)
    return [0.5 - v * h for v in reversed(GRID_V_C)]


def hk_grid(N: int) -> list[float]:
    return [HK_CENTRE + u * _h(N) for u in GRID_U_H]


def hk_count_eps() -> list[float]:
    return [1.0 / (2.0 * x) for x in HK_COUNT_X]


def _deffuant(N: int) -> tuple[str, ...]:
    return ("--model", "deffuant", "--mu", "0.5", "--graph", "complete", "--n", str(N),
            "--init", "uniform", "--task", "ensemble", "--steps", str(D_CAP),
            "--observables", "clusters,largest_cluster", "--log-times", "2",
            "--save-trajectories", "true")


def _hk(N: int, update: str) -> tuple[str, ...]:
    cap = H_SWEEP_CAP if update in ("sweep", "ordered") else H_CAP
    return ("--model", "hk", "--update", update, "--graph", "complete", "--n", str(N),
            "--init", "uniform", "--task", "ensemble", "--steps", str(cap),
            "--observables", "clusters,largest_cluster")


def runs(scale: str) -> list[Run]:
    out = []
    d = SCALE[scale]["deffuant"]
    for N in d["sizes"]:
        for kind, grid, R in (("threshold", deffuant_grid(N), d["replicas"]),
                              ("complete", complete_grid(N), d["complete"]),
                              ("clusters", [e for e, _ in deffuant_window_points()], d["count"])):
            out.append(Run(f"deffuant_{kind}_N{N}", _deffuant(N) + (
                "--replicas", str(R), "--sweep", "epsilon=" + _decimals(grid)),
                seed=SEEDS[f"deffuant_{kind}"] + N, block=d["block"]))
    h = SCALE[scale]["hk"]
    for N in h["sizes"]:
        for kind, update, grid, R in (("threshold", "ordered", hk_grid(N), h["replicas"]),
                                      ("fresh", "sweep", hk_grid(N), h["fresh"]),
                                      ("sync", "synchronous", hk_grid(N), h["sync"]),
                                      ("clusters", "synchronous", hk_count_eps(), h["count"])):
            out.append(Run(f"hk_{kind}_N{N}", _hk(N, update) + (
                "--replicas", str(R), "--sweep", "epsilon=" + _decimals(grid)),
                seed=SEEDS[f"hk_{kind}"] + N,
                block=h["seq_block"] if update in ("ordered", "sweep") else h["block"]))
    return out


# ---------------------------------------------------------------------------------------------

class _Point:
    """One sweep point of an ensemble run: eps and, per replica (sorted by label), the final
    cluster sizes and the replicas-table columns."""

    def __init__(self, eps: float, N: int, outs: Outputs, where: str):
        self.eps, self.N = eps, N
        rep = outs.get("replicas")
        cl = outs.get("clusters")
        if rep is None or cl is None:
            raise RuntimeError(f"{where}: needs the replicas and clusters tables")
        labels = np.asarray(rep["replica"], int)
        order = np.argsort(labels)
        self.labels = labels[order]
        self.R = int(labels.size)
        self.consensus = np.asarray(rep["reached_consensus"], float)[order] > 0.5
        self.absorbed = np.asarray(rep["absorbed"], float)[order] > 0.5
        self.clusters = np.asarray(rep["clusters"], float)[order]          # >= 1% of N
        self.largest = np.asarray(rep["largest_cluster"], float)[order]
        c_rep = np.asarray(cl["replica"], int)
        c_size = np.asarray(cl["size"], float)
        index = {int(r): k for k, r in enumerate(self.labels)}
        self.major = np.zeros(self.R)            # clusters with more than eps N agents (BKR)
        self.every = np.zeros(self.R)            # all clusters, every size
        for r, s in zip(c_rep, c_size):
            k = index[int(r)]
            self.every[k] += 1
            if s > eps * N:
                self.major[k] += 1
        self.traj = outs.get("trajectories")

    def unsettled(self) -> int:
        """Replicas whose number of clusters with >= 1% of the nodes changed between
        SETTLE_FROM and the last recorded time; 0 without trajectories. (The largest share is
        not used: it grows whenever stragglers join a cluster, which no count depends on.)"""
        if self.traj is None:
            return 0
        rep = np.asarray(self.traj["replica"], int)
        t = np.asarray(self.traj["time"], float)
        c = np.asarray(self.traj["clusters"], float)
        end = t.max()
        bad = 0
        for r in np.unique(rep):
            m = rep == r
            a = m & (t == SETTLE_FROM)
            b = m & (t == end)
            if not (a.any() and b.any()):
                raise RuntimeError(f"replica {r}: no record at t = {SETTLE_FROM} or {end:g}")
            if c[a][0] != c[b][0]:
                bad += 1
        return bad


def _points(prefix: Path, N: int) -> list[_Point]:
    outs = find_outputs(prefix)
    sw = outs.get("sweep")
    if sw is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    eps = np.asarray(sw["epsilon"], float)
    if sorted(outs.points) != list(range(eps.size)):
        raise RuntimeError(f"{prefix}: point files do not match the sweep table")
    return [_Point(float(e), N, outs.points[i], f"{prefix} point {i}") for i, e in enumerate(eps)]


def half_crossing(eps, P) -> float:
    """eps where P first reaches 1/2 from below, by linear interpolation; NaN if P starts at or
    above 1/2 or never reaches it."""
    eps, P = np.asarray(eps, float), np.asarray(P, float)
    above = np.nonzero(P >= 0.5)[0]
    if above.size == 0 or above[0] == 0:
        return math.nan
    i = int(above[0])
    return float(eps[i - 1] + (0.5 - P[i - 1]) * (eps[i] - eps[i - 1]) / (P[i] - P[i - 1]))


def _crossing(eps: np.ndarray, k: np.ndarray, R: np.ndarray, rng: np.random.Generator,
              strict: bool = True) -> tuple[float, float, int]:
    """(eps_1/2, bootstrap sd, discarded draws): k successes of R at each eps (sorted), each
    redrawn as Binomial(R, k/R). A grid that does not bracket 1/2 raises when `strict`, else
    gives NaN (diagnostics)."""
    P = k / R
    e = half_crossing(eps, P)
    if not math.isfinite(e):
        if not strict:
            return math.nan, math.nan, 0
        raise RuntimeError(f"the eps grid does not bracket P = 1/2 (P = {np.round(P, 3)})")
    draws = rng.binomial(R.astype(int)[None, :], P[None, :], size=(N_BOOT, P.size)) / R
    es = np.array([half_crossing(eps, d) for d in draws])
    ok = np.isfinite(es)
    return e, float(np.std(es[ok], ddof=1)), int(np.sum(~ok))


def _fmt_list(Ns, vals, errs, digits: int = 4) -> str:
    return ", ".join(f"N={N} {v:.{digits}f} ± {e:.2g}" if math.isfinite(v) else
                     f"N={N} not bracketed by the grid" for N, v, e in zip(Ns, vals, errs))


def _threshold_curves(scale: str, data: Path, model: str, kind: str, success,
                      strict: bool = True) -> dict:
    """Per N: eps, k, R, eps_1/2 and its error, capped and unsettled replicas, provenance."""
    rng = np.random.default_rng(SEEDS[f"{model}_{kind}"])
    out = {"N": [], "curves": [], "half": [], "err": [], "lost": [], "capped": [],
           "unsettled": [], "prov": None}
    for N in SCALE[scale][model]["sizes"]:
        pts = sorted(_points(data / f"{model}_{kind}_N{N}", N), key=lambda p: p.eps)
        eps = np.array([p.eps for p in pts])
        k = np.array([float(np.sum(success(p))) for p in pts])
        R = np.array([float(p.R) for p in pts])
        e, s, lost = _crossing(eps, k, R, rng, strict)
        out["N"].append(N)
        out["curves"].append((eps, k / R, R))
        out["half"].append(e)
        out["err"].append(s)
        out["lost"].append(lost)
        out["capped"].append(int(sum(np.sum(~p.absorbed) for p in pts)))
        out["unsettled"].append(int(sum(p.unsettled() for p in pts)))
    out["prov"] = find_outputs(data / f"{model}_{kind}_N{out['N'][-1]}").provenance
    return out


def _run_warnings(c: dict, label: str, capped_ok: bool) -> list[str]:
    warns = []
    for N, lost in zip(c["N"], c["lost"]):
        if lost:
            warns.append(f"{label} N={N}: {lost} of {N_BOOT} bootstrap draws did not bracket 1/2")
    capped = sum(c["capped"])
    if capped:
        warns.append(f"{label}: {capped} replicas stopped at the step cap before converging ("
                     + ", ".join(f"N={N}: {k}" for N, k in zip(c["N"], c["capped"]) if k) + ")"
                     + ("; their clusters were settled (see below)" if capped_ok else
                        "; their final state is used as is"))
    unsettled = sum(c["unsettled"])
    if capped_ok:
        warns.append(f"{label}: " + (f"{unsettled} replicas whose number of >= 1% clusters "
                                     f"changed between {SETTLE_FROM} and {D_CAP} sweeps ("
                                     + ", ".join(f"N={N}: {k}" for N, k in
                                                 zip(c["N"], c["unsettled"]) if k) + ")"
                                     if unsettled else
                                     f"no replica's number of >= 1% clusters changed "
                                     f"between {SETTLE_FROM} and {D_CAP} sweeps (settled)"))
    return warns


def _plot_threshold(c: dict, predicted: float, perr: float, ylabel: str, title: str):
    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0, 0], axes[0, 1]
    for i, (N, (eps, P, R)) in enumerate(zip(c["N"], c["curves"])):
        a0.errorbar(eps, P, yerr=np.sqrt(np.maximum(P * (1 - P), 0.25 / R) / R), fmt="o-", ms=3,
                    lw=0.8, color=color(i), capsize=0, label=f"N = {N}")
    a0.axhline(0.5, color="#888888", lw=0.6, ls=":")
    a0.axvline(predicted, color=color(7), lw=1, label=f"theory {predicted:.4f}")
    a0.axvspan(predicted - perr, predicted + perr, color=color(7), alpha=0.12, lw=0)
    a0.set(xlabel="confidence bound eps", ylabel=ylabel)
    a0.legend(fontsize=6.5)
    a1.errorbar(c["N"], c["half"], yerr=c["err"], fmt="o", ms=4, color=color(0), capsize=0,
                label="P = 1/2 (interpolated)")
    a1.axhline(predicted, color=color(7), lw=1, label="theory")
    a1.axhspan(predicted - perr, predicted + perr, color=color(7), alpha=0.12, lw=0)
    a1.set(xscale="log", xlabel="N", ylabel="eps at P = 1/2")
    a1.legend()
    return finish(fig, c["prov"], title)


def _deffuant_threshold(scale: str, data: Path, figures: Path) -> Check:
    eps0, perr = deffuant_major_threshold()
    c = _threshold_curves(scale, data, "deffuant", "threshold", lambda p: p.major == 1)
    odsp = _threshold_curves(scale, data, "deffuant", "threshold", lambda p: p.clusters == 1,
                              strict=False)
    comp = _threshold_curves(scale, data, "deffuant", "complete", lambda p: p.largest >= 1.0,
                              strict=False)
    warns = [f"threshold at every N (drift): {_fmt_list(c['N'], c['half'], c['err'])}; the "
             "largest N is the measured value (ruling R17: no extrapolation); a residual "
             "finite-size shift may remain",
             "one major cluster = exactly one cluster with more than eps N agents (BKR's major "
             "cluster, mass > 1 in their units); with ODSP's `clusters` == 1 (no other cluster "
             f"with >= 1% of N) instead: {_fmt_list(odsp['N'], odsp['half'], odsp['err'])}",
             "diagnostic, not compared: complete consensus (no minor clusters, largest_cluster "
             f"= 1) has its half point at {_fmt_list(comp['N'], comp['half'], comp['err'])}; it "
             "tends to 1/2 as N -> infinity (BKR, Delta = 1; Fortunato 2004), slowly (minor edge "
             "clusters of mass ~ (Delta - 1)^3)"]
    half = np.asarray(c["half"])
    gap = half[-1] - eps0
    steps = np.diff(half)
    errs = np.asarray(c["err"])
    noise = 2.0 * np.hypot(errs[1:], errs[:-1])
    trend = ("monotone within errors (each step toward eps* or within 2 errors of zero)"
             if np.all(steps * np.sign(gap) <= noise) else "not monotone")
    warns += [f"gap at the largest N (N={c['N'][-1]}): {gap:+.4f}, {100 * gap / eps0:+.2f}% of "
              f"eps*; the half point over N is {trend}; no published finite-N form; not "
              "extrapolated (ruling R22: the row stays flagged when the gap exceeds the "
              "tolerance)",
              "independent program (Task 11 review): P(one major) = 0.520 +- 0.025 at eps = "
              "0.270419, N = 8000, against ODSP's 0.515 +- 0.025 in the small run",
              "BKR's error (half their integration mesh, 0.00125 in Delta) covers the Delta mesh "
              "only: not their opinion grid (400 states per unit) nor the rounding of 1.871",
              "step cap verified by replay without the cap (Task 11 review, N = 8000, eps = "
              "0.27042): 0 of 400 replicas change their major count (150 were capped)"]
    warns += _run_warnings(c, "one-major runs", True) + _run_warnings(comp, "complete-consensus "
                                                                      "runs", True)
    name = "deffuant.threshold"
    fig = _plot_threshold(c, eps0, perr, "P(one major cluster)",
                          "Deffuant (mu = 1/2), complete graph: one major cluster")
    N = c["N"][-1]
    return Check(model=NAME, check=name,
                 quantity="eps where P(one major cluster) = 1/2 at the largest N (exactly one "
                          "final cluster with more than eps N agents; linear interpolation on "
                          "the eps grid), opinions uniform on [0, 1], mu = 1/2",
                 measured=c["half"][-1], error=c["err"][-1], predicted=eps0,
                 predicted_error=perr, source="deffuant_rate_equation",
                 sizes="N=" + ",".join(map(str, c["N"])) + f" (measured at N={N})",
                 statistics=f"{int(c['curves'][-1][2].min())} replicas per point, "
                            f"{len(GRID_U_D)} points per N; parametric bootstrap of the "
                            "binomial P",
                 warnings=warns, figure=save_figure(fig, figures, name))


def _count_stat(p: _Point) -> tuple[float, float]:
    """Mean number of major clusters over the replicas and its error. A point without spread
    gets the variance of a count off by one in one replica of R, so that no error is zero."""
    R = p.R
    var = float(np.var(p.major, ddof=1)) if R > 1 else 0.0
    var = max(var, (1.0 / R) * (1.0 - 1.0 / R))
    return float(np.mean(p.major)), math.sqrt(var / R)


def _deffuant_clusters(scale: str, data: Path, figures: Path) -> list[Check]:
    """One Check per BKR window (ruling R21), all sharing one figure."""
    expected = {round(e, 9): n for e, n in deffuant_window_points()}
    per_N = []                      # (N, points sorted by decreasing eps, BKR counts)
    for N in SCALE[scale]["deffuant"]["sizes"]:
        pts = sorted(_points(data / f"deffuant_clusters_N{N}", N), key=lambda p: -p.eps)
        per_N.append((N, pts, [expected[round(p.eps, 9)] for p in pts]))
    N, pts, ns = per_N[-1]
    capped = sum(int(np.sum(~p.absorbed)) for q in per_N for p in q[1])
    unsettled = [(q[0], sum(p.unsettled() for p in q[1])) for q in per_N]
    common = ["replaces the agreed target '1/(2 eps) major clusters': BKR's rate equation gives "
              "a pattern periodic in Delta = 1/(2 eps) with period L = 2.155, i.e. 1/(L eps) "
              "major clusters spaced L eps apart asymptotically, and the exact count of their "
              "bifurcation table at finite Delta, compared here at the centre of each window with "
              "2k major clusters and only minor ones besides",
              "only the windows BKR tabulate (2, 4, 6 major clusters); the window with 8, whose "
              "upper end is not in their table, was dropped -- decided after seeing the data "
              "(ruling R21), because no published prediction exists there"]
    if capped:
        common.append(f"{capped} replicas of the count runs stopped at the step cap of {D_CAP} "
                      "sweeps before converging (isolated stragglers)")
    bad = [f"N={n}: {k}" for n, k in unsettled if k]
    common.append(f"{sum(k for _, k in unsettled)} replicas of the count runs whose number of "
                  f">= 1% clusters changed between {SETTLE_FROM} and {D_CAP} sweeps "
                  f"({', '.join(bad)}; straggler groups of >= 1% of N, far below the eps N cut)"
                  if bad else f"no replica's number of >= 1% clusters changed between "
                              f"{SETTLE_FROM} and {D_CAP} sweeps (settled)")

    name = "deffuant.clusters"
    fig, axes = new_fig()
    a0 = axes[0, 0]
    xs = np.linspace(1.5, 8.0, 200)
    a0.plot(xs, 2 * xs / BKR_PERIOD, "-", color=color(7), lw=0.8, label="1/(L eps), L = 2.155")
    a0.plot(xs, xs, "--", color="#888888", lw=0.8, label="1/(2 eps)")
    for i, (Nq, pq, _) in enumerate(per_N):
        d = np.array([1 / (2 * p.eps) for p in pq])
        st = np.array([_count_stat(p) for p in pq])
        a0.errorbar(d + 0.04 * (i - len(per_N) / 2), st[:, 0], yerr=st[:, 1], fmt="o", ms=3,
                    color=color(i), capsize=0, label=f"N = {Nq}")
    a0.plot([1 / (2 * p.eps) for p in pts], ns, "_", ms=14, mew=2, color=color(7),
            label="BKR count")
    a0.set(xlabel="Delta = 1/(2 eps)", ylabel="major clusters (> eps N agents)")
    a0.legend(fontsize=6.5)
    fig = finish(fig, find_outputs(data / f"deffuant_clusters_N{N}").provenance,
                 "Deffuant (mu = 1/2), complete graph: major clusters")
    figure = save_figure(fig, figures, name)

    checks = []
    for k, (p, n) in enumerate(zip(pts, ns)):
        m, e = _count_stat(p)
        drift, exact = [], []
        for Nq, pq, _ in per_N:
            mq, eq = _count_stat(pq[k])
            drift.append(f"N={Nq} {mq:.3f} ± {eq:.2g}")
            exact.append(f"N={Nq} {np.mean(pq[k].major == n):.3f}")
        warns = list(common) + [
            "count at every N (drift): " + ", ".join(drift) + "; the largest N is the measured "
            "value (no extrapolation); a residual finite-size shift may remain",
            f"share of the replicas with BKR's count ({n}) exactly at every N: "
            + ", ".join(exact),
            f"at N={N}: {np.mean(p.major == n):.3f} of the replicas have exactly {n}; "
            f"1/(L eps) = {deffuant_asymptotic_clusters(p.eps):.2f}, 1/(2 eps) = "
            f"{1 / (2 * p.eps):.2f}; ODSP `clusters` (>= 1% of N) {np.mean(p.clusters):.3f}"]
        checks.append(Check(
            model=NAME, check=f"deffuant.clusters_n{n}",
            quantity=f"mean number of major clusters (more than eps N agents) at eps = "
                     f"{p.eps:.4f} (Delta = {1 / (2 * p.eps):.3f}, centre of BKR's window with "
                     f"{n}), largest N; BKR rate-equation count {n}",
            measured=m, error=e, predicted=float(n), predicted_error=0.0,
            source="deffuant_rate_equation",
            sizes="N=" + ",".join(str(q[0]) for q in per_N) + f" (measured at N={N}); "
                  f"eps={p.eps:.4f}",
            statistics=f"{p.R} replicas; error from the replica spread",
            warnings=warns + REVIEW_NOTES.get(n, []), figure=figure))
    return checks


def _hk_threshold(scale: str, data: Path, figures: Path) -> Check:
    eps0, perr = hk_threshold()
    c = _threshold_curves(scale, data, "hk", "threshold", lambda p: p.consensus)
    sync = _threshold_curves(scale, data, "hk", "sync", lambda p: p.consensus, strict=False)
    fresh = _threshold_curves(scale, data, "hk", "fresh", lambda p: p.consensus, strict=False)
    N = c["N"][-1]
    warns = [f"threshold at every N (drift): {_fmt_list(c['N'], c['half'], c['err'])}; the "
             "largest N is the measured value (ruling R17: no extrapolation); a residual "
             "finite-size shift may remain",
             "Fortunato's [0.195, 0.202] brackets the N -> infinity step (from N = 5000 and "
             "10000); the measured value is the half point at the largest N (10000 at full "
             "scale) compared against it, and reading the bracket as 0.1985 +- 0.0020 "
             "(midpoint, half-width / sqrt(3)) is a modelling choice",
             "update: fixed-order sweeps (every agent once per sweep, in index order -- with "
             "i.i.d. initial opinions a random order fixed for the run; ruling R23), Fortunato's "
             "ordered sweeps; he writes only that random updating leaves the final number of "
             "clusters unchanged (footnote: on special graphs it can change the probability of "
             "complete consensus)",
             "diagnostic, not compared: in random-order sweeps (a fresh order each sweep, same "
             f"grid) the half points are {_fmt_list(fresh['N'], fresh['half'], fresh['err'])}; "
             "with synchronous updating (the original HK) they are "
             f"{_fmt_list(sync['N'], sync['half'], sync['err'])} -- the threshold depends on the "
             "update scheme; Lorenz (2007): synchronous density-based model in consensus for "
             "eps >= 0.19"]
    if N != 10000:
        warns.append(f"largest N = {N}, not Fortunato's 10000 (full scale only)")
    warns += (_run_warnings(c, "fixed-order runs", False)
              + _run_warnings(fresh, "random-order runs", False)
              + _run_warnings(sync, "synchronous runs", False))
    name = "hk.threshold"
    fig = _plot_threshold(c, eps0, perr, "P(complete consensus)",
                          "Hegselmann-Krause (fixed-order sweeps), complete graph: consensus")
    return Check(model=NAME, check=name,
                 quantity="eps where P(complete consensus) = 1/2 at the largest N (every opinion "
                          "within 1e-9 of every other at convergence; linear interpolation on "
                          "the eps grid), HK in sweeps with a fixed order (every agent once per "
                          "sweep, index order), opinions uniform on [0, 1]",
                 measured=c["half"][-1], error=c["err"][-1], predicted=eps0,
                 predicted_error=perr, source="hk_threshold",
                 sizes="N=" + ",".join(map(str, c["N"])) + f" (measured at N={N})",
                 statistics=f"{int(c['curves'][-1][2].min())} replicas per point, "
                            f"{len(GRID_U_H)} points per N; parametric bootstrap of the "
                            "binomial P",
                 warnings=warns, figure=save_figure(fig, figures, name))


def _hk_clusters(scale: str, data: Path, figures: Path) -> Check:
    per_N = []
    for N in SCALE[scale]["hk"]["sizes"]:
        pts = sorted(_points(data / f"hk_clusters_N{N}", N), key=lambda p: -p.eps)
        x = np.array([1.0 / p.eps for p in pts])
        y = np.array([np.mean(p.every) for p in pts])
        s = np.array([math.sqrt(max(np.var(p.every, ddof=1), (1 / p.R) * (1 - 1 / p.R)) / p.R)
                      for p in pts])
        y1 = np.array([np.mean(p.clusters) for p in pts])
        r = fit(x, y, s, model="line")
        r1 = fit(x, y1, s, model="line")
        capped = int(sum(np.sum(~p.absorbed) for p in pts))
        per_N.append((N, pts, x, y, s, r, r1, capped))
    N, pts, x, y, s, r, r1, _ = per_N[-1]
    b, eb = r.params["b"], r.errors["b"]
    sigma = (b - 0.5) / eb
    warns = [f"slope at every N: " + _fmt_list([q[0] for q in per_N],
                                              [q[5].params["b"] for q in per_N],
                                              [q[5].errors["b"] for q in per_N]),
             ("one-sided check passed: b < 1/2 at " if sigma < -3 else
              "ONE-SIDED CHECK FAILED: b is not below 1/2 by 3 errors (") + f"{-sigma:.1f} errors"
             + ("" if sigma < -3 else ")"),
             f"implied cluster spacing 1/b = {1 / b:.3f} eps (BHT 2009: slightly more than 2, "
             f"~2.2 on a semi-infinite interval; Wang et al. 2017 mean field: 2.29); against "
             f"Wang et al.'s slope 1/2.29 = {WANG_SLOPE:.4f}: {(b - WANG_SLOPE) / eb:+.1f} "
             "errors (information: an estimate without an error)",
             "update: synchronous, as BHT 2009 (Wang et al. 2017: a mean-field equation without "
             "update order); one-sided, so it detects overcounting only",
             f"counting only ODSP's `clusters` (>= 1% of N): slope {r1.params['b']:.4f} ± "
             f"{r1.errors['b']:.2g} (small central clusters left out)"]
    warns += fit_quality(r)
    capped = sum(q[7] for q in per_N)
    if capped:
        warns.append(f"{capped} replicas stopped at the cap of {H_CAP} rounds before converging; "
                     "their final state is used as is")
    fig, axes = new_fig()
    ax = axes[0, 0]
    for i, (Nq, _, xq, yq, sq, rq, *_rest) in enumerate(per_N):
        ax.errorbar(xq, yq, yerr=sq, fmt="o", ms=3, color=color(i), capsize=0, label=f"N = {Nq}")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, r.predict(xs), "-", color=color(len(per_N) - 1), lw=1,
            label=f"fit N = {N}: b = {b:.4f} ± {eb:.2g}")
    ax.plot(xs, xs / 2, "--", color="#888888", lw=0.8, label="1/(2 eps)")
    ax.plot(xs, WANG_SLOPE * xs, ":", color=color(7), lw=1, label="1/(2.29 eps)")
    ax.set(xlabel="1 / eps", ylabel="clusters (all sizes)")
    ax.legend(fontsize=6.5)
    name = "hk.clusters"
    fig = finish(fig, find_outputs(data / f"hk_clusters_N{N}").provenance,
                 "Hegselmann-Krause (synchronous), complete graph: clusters")
    R = min(p.R for p in pts)
    return Check(model=NAME, check=name,
                 quantity="one-sided: slope b of n = a + b/eps, n the mean number of final "
                          "clusters (all sizes), weighted line fit over 1/(2 eps) = 3..10, "
                          "largest N, synchronous HK (as BHT); predicted b < 1/2 (fewer than "
                          "1/(2 eps) clusters, spacing above 2 eps), passed when sigma < -3; "
                          "detects overcounting only",
                 measured=b, error=eb, predicted=0.5, predicted_error=0.0, source="hk_clusters",
                 sizes="N=" + ",".join(str(q[0]) for q in per_N) + f" (fit at N={N}), "
                       f"{len(HK_COUNT_X)} values of eps",
                 statistics=f"{R} replicas per point; errors from the replica spread",
                 warnings=warns, figure=save_figure(fig, figures, name), one_sided="below")


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_deffuant_threshold(scale, data, figures), *_deffuant_clusters(scale, data, figures),
            _hk_threshold(scale, data, figures), _hk_clusters(scale, data, figures)]
