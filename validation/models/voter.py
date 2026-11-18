"""The voter model (--model voter, node update): exact results on the complete graph, the exit
probability, consensus-time scaling on lattices in d = 1, 2, 3 and on Barabasi-Albert
networks, and the decay of the active-link density on the square lattice.

Each check is a ratio of simulation to exact theory (predicted 1) or an exponent fixed by theory
(2 for T ~ L^2 in 1D, 1 elsewhere once the known logarithms or the exact finite-size meeting
time are divided out). The active-link density is compared with the exact rho(t) as a ratio;
its exponent form is kept as a diagnostic.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from odspy import consensus_time_vs_size, find_outputs, fit
from odspy.io import Table
from odspy.trajectories import fit_timeseries

from validation.analysis import (exponent_check, plot_exponent, plot_ratio, ratio_check,
                                 replica_ratio, save_figure, weighted_ratio)
from validation.core import Check, Run
from validation.theory import mean_absorption_time, torus_eigentime, voter_rates, voter_rho_2d

NAME = "voter"

EXIT_UP = (0.1, 0.3, 0.5, 0.7, 0.9)
BA_M = 3
RHO_T_MIN = 10          # sweeps: 1/rho is fitted from here on (the first sweeps are transient)

# Sizes and statistics per scale. Each entry: sizes, replicas, block (replicas per job,
# None: no split) and threads per job. Smoke: seconds for pytest; small: a few minutes at
# 10 threads; full: a cluster, exponents from >= 5 sizes spanning a factor >= 8 in L or N.
SCALE = {
    "smoke": {
        "complete": ((16, 32, 64), 200, None, 1),
        "exit": (8, 200, None, 1),
        "lattice_1d": ((16, 32, 64, 128), 100, None, 1),
        "lattice_2d": ((4, 6, 8, 12), 100, None, 1),
        "lattice_3d": ((3, 4, 5, 6), 100, None, 1),
        "ba": ((100, 200, 400, 800), 100, None, 1),
        "rho_2d": ((32, 100), 24, None, 1),               # (L, sweeps), replicas
    },
    "small": {
        "complete": ((32, 64, 128, 256, 512, 1024), 2000, 500, 1),
        "exit": (16, 2000, 500, 1),
        "lattice_1d": ((32, 64, 128, 256, 512), 1000, 200, 1),
        "lattice_2d": ((8, 12, 16, 24, 32, 48), 800, 100, 1),
        "lattice_3d": ((6, 8, 10, 12, 16), 600, 60, 1),
        "ba": ((500, 1000, 2000, 4000, 8000), 500, 50, 1),
        "rho_2d": ((128, 1000), 200, 20, 1),
    },
    "full": {
        "complete": ((100, 316, 1000, 3162, 10000), 10000, 1000, 4),
        "exit": (32, 20000, 2000, 4),
        "lattice_1d": ((128, 256, 512, 1024, 2048), 10000, 500, 4),
        "lattice_2d": ((16, 32, 64, 128, 256), 2000, 100, 4),
        "lattice_3d": ((6, 8, 12, 16, 24, 32, 48), 2000, 100, 4),
        "ba": ((1000, 3162, 10000, 31623, 100000), 4000, 250, 4),
        "rho_2d": ((512, 10000), 400, 20, 4),
    },
}

SEEDS = {"complete": 601, "exit": 602, "lattice_1d": 603, "lattice_2d": 604, "lattice_3d": 605,
         "ba": 606, "rho_2d": 607}


def _dims(L: int, d: int) -> str:
    return "x".join([str(L)] * d)


def _ensemble(*extra: str) -> tuple[str, ...]:
    return ("--model", "voter", "--task", "ensemble", *extra)


def runs(scale: str) -> list[Run]:
    s = SCALE[scale]

    def run(name: str, args: tuple[str, ...], replicas: int, block, threads: int) -> Run:
        return Run(name, args + ("--replicas", str(replicas)), seed=SEEDS[name], threads=threads,
                   block=block)

    out = []
    Ns, R, block, th = s["complete"]
    # Caps (--steps) sit ~20 mean consensus times or more above the largest size's mean, so
    # censoring has probability ~e^-20 (the tails are exponential with decay time ~ the mean).
    out.append(run("complete", _ensemble("--graph", "complete", "--time-unit", "updates",
                                         "--steps", str(30 * max(Ns) ** 2),
                                         "--sweep", "n=" + ",".join(map(str, Ns))), R, block, th))
    L, R, block, th = s["exit"]
    out.append(run("exit", _ensemble("--graph", "lattice", "--dims", _dims(L, 2),
                                     "--steps", str(_cap_2d(L * L)),
                                     "--sweep", "up=" + ",".join(map(str, EXIT_UP))), R, block, th))
    for d, cap in ((1, lambda L: 2 * L * L), (2, lambda L: _cap_2d(L * L)), (3, lambda L: 20 * L ** 3)):
        name = f"lattice_{d}d"
        Ls, R, block, th = s[name]
        out.append(run(name, _ensemble("--graph", "lattice", "--steps", str(cap(max(Ls))),
                                       "--sweep", "dims=" + ",".join(_dims(L, d) for L in Ls)),
                       R, block, th))
    Ns, R, block, th = s["ba"]
    out.append(run("ba", _ensemble("--graph", "ba", "--m", str(BA_M), "--regenerate-graph", "true",
                                   "--steps", str(10 * max(Ns)),
                                   "--sweep", "n=" + ",".join(map(str, Ns))), R, block, th))
    (L, steps), R, block, th = s["rho_2d"]
    out.append(run("rho_2d", _ensemble("--graph", "lattice", "--dims", _dims(L, 2), "--steps",
                                       str(steps), "--log-times", "10", "--observables", "rho",
                                       "--save-trajectories", "true"), R, block, th))
    return out


def _cap_2d(N: int) -> int:
    """5 N ln N sweeps: ~20 times the mean 2D consensus time (~0.25 N ln N)."""
    return int(5 * N * math.log(N)) + 1


# ---------------------------------------------------------------------------------------------

def _sweep(prefix: Path) -> Table:
    table = find_outputs(prefix).get("sweep")
    if table is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    return table


def _censored_warning(table: Table, key: str) -> list[str]:
    frac = np.asarray(table["consensus_fraction"], float)
    if np.all(frac == 1):
        return []
    named = ", ".join(f"{key}={v} ({f:.3g})" for v, f in zip(table[key], frac) if f < 1)
    return [f"runs capped at --steps before consensus: consensus_fraction < 1 at {named}"]


def _complete(scale: str, data: Path, figures: Path) -> Check:
    t = _sweep(data / "complete")
    N = np.asarray(t["n"], int)
    R = np.asarray(t["replicas"], float)
    sim, err = np.asarray(t["mean_consensus_time"], float), np.asarray(t["sd_consensus_time"], float) / np.sqrt(R)
    # --init random --up 0.5 starts with exactly floor(0.5 N) = N // 2 nodes at +1
    theory = np.array([mean_absorption_time(*voter_rates(int(n)))[int(n) // 2] for n in N])
    ratio = weighted_ratio(sim, err, theory)
    grid = np.unique(np.geomspace(N.min(), N.max(), 40).astype(int))
    curve = [mean_absorption_time(*voter_rates(int(n)))[int(n) // 2] for n in grid]
    name = "voter.complete_time"
    fig = plot_ratio(N, ratio, "N", title="voter, complete graph: mean consensus time",
                     prov=t.provenance, absolute=(sim, err, grid, curve, "T (single updates)"))
    return ratio_check(ratio, model=NAME, check=name, source="voter_complete",
                       quantity="mean consensus time / exact T(N/2) (single updates, start n = N/2)",
                       sizes="N=" + ",".join(map(str, N)), statistics=f"{int(R.min())} replicas per N",
                       warnings=_censored_warning(t, "n"), figure=save_figure(fig, figures, name))


def _exit(scale: str, data: Path, figures: Path) -> Check:
    t = _sweep(data / "exit")
    L = SCALE[scale]["exit"][0]
    N = L * L
    up = np.asarray(t["up"], float)
    R = np.asarray(t["replicas"], float)
    sim = np.asarray(t["exit_prob_up"], float)
    # floor(up * N) nodes start at +1, computed in double precision as the driver does
    p = np.array([int(f * N) / N for f in up])
    err = np.sqrt(p * (1 - p) / R)     # binomial error under the prediction (never 0)
    ratio = weighted_ratio(sim, err, p)
    name = "voter.exit_probability"
    fig = plot_ratio(up, ratio, "initial fraction of +1", logx=False, prov=t.provenance,
                     title=f"voter, {L}x{L} lattice: exit probability",
                     absolute=(sim, err, [0.0, 1.0], [0.0, 1.0], "P(+1 consensus)"))
    return ratio_check(ratio, model=NAME, check=name, source="voter_exit",
                       quantity="P(+1 consensus) / initial fraction of +1 nodes (2D lattice)",
                       sizes=f"L={L} (2D), up=" + ",".join(f"{f:g}" for f in up),
                       statistics=f"{int(R.min())} replicas per point; binomial errors",
                       warnings=_censored_warning(t, "up"), figure=save_figure(fig, figures, name))


def _size_fit(prefix: Path, factor=None):
    """Consensus time against N (sweep table), y multiplied by factor(N) when given, fitted to
    A N^b. Capped runs switch to the driver's median (robust while fewer than half are
    capped). Returns (FitResult, warnings beyond the fit's own, replicas per size)."""
    t = _sweep(prefix)
    warns = _censored_warning(t, t.names[0])
    censored = "median" if warns else "exclude"
    if warns:
        warns.append("y is the median consensus time (censored='median'), not the mean")
    r = consensus_time_vs_size(prefix, size="N", censored=censored)
    reps = int(np.min(t["replicas"]))
    if factor is None:
        return r, warns, reps
    u = np.asarray(r.used, bool)
    g = factor(r.x)
    warns += r.warnings      # exclusions made before the refit
    return fit(r.x[u], r.y[u] * g[u], r.yerr[u] * g[u]), warns, reps


def _lattice(scale: str, data: Path, figures: Path, d: int) -> Check:
    """d = 1: T ~ L^b, b = 2. d = 2, 3: T ~ N ln N and T ~ N, both of the form T ~ M_N, the
    exact finite-torus meeting time of two random walks (torus_eigentime; Cox 1989), which
    carries the slowly converging corrections (M_N / (N ln N) drifts by 6% over L = 8..48 in
    2D, M_N / N by 1/L in 3D) that would bias b by 0.02-0.04. So b is fitted to
    T N / M_N ~ N^b, b = 1; since M_N / (N ln N) -> 1/pi in 2D and M_N / N -> 1.516 in 3D,
    this is the same exponent as T / ln N ~ N and T ~ N asymptotically. In 1D M_N = (L^2-1)/6
    and the plain fit is used."""
    name = f"voter.lattice_{d}d"
    Ls = SCALE[scale][f"lattice_{d}d"][0]
    if d == 1:
        r, warns, reps = _size_fit(data / "lattice_1d")
        quantity, ylabel, b = "b in T ~ L^b, L = N (T: mean consensus time, sweeps)", "T (sweeps)", 2.0
    else:
        def factor(N: np.ndarray) -> np.ndarray:
            return np.array([n / torus_eigentime(round(n ** (1 / d)), d) for n in N])
        r, warns, reps = _size_fit(data / f"lattice_{d}d", factor)
        form = "N ln N / pi" if d == 2 else "1.516 N"
        quantity = (f"b in T N / M_N ~ N^b (T: mean consensus time, sweeps; M_N: exact torus "
                    f"meeting time, ~ {form})")
        ylabel, b = "T N / M_N (sweeps)", 1.0
    prov = _sweep(data / f"lattice_{d}d").provenance
    fig = plot_exponent(r, b, "N" if d > 1 else "L", ylabel, prov=prov,
                        title=f"voter, {d}D lattice: consensus time")
    return exponent_check(r, b, model=NAME, check=name, quantity=quantity, source="voter_lattice",
                          sizes="L=" + ",".join(map(str, Ls)) + f" ({d}D)",
                          statistics=f"{reps} replicas per size", warnings=warns,
                          figure=save_figure(fig, figures, name))


def _ba(scale: str, data: Path, figures: Path) -> Check:
    name = "voter.ba_heterogeneous"
    Ns = SCALE[scale]["ba"][0]
    r, warns, reps = _size_fit(data / "ba", np.log)
    fig = plot_exponent(r, 1.0, "N", "T ln N", prov=_sweep(data / "ba").provenance,
                        title=f"voter (node update), Barabasi-Albert m={BA_M}")
    return exponent_check(r, 1.0, model=NAME, check=name, source="voter_heterogeneous",
                          quantity="b in T ln N ~ N^b (T ~ N mu1^2/mu2 ~ N/ln N; T in sweeps)",
                          sizes="N=" + ",".join(map(str, Ns)) + f", m={BA_M}",
                          statistics=f"{reps} replicas per size, new network per replica",
                          warnings=warns, figure=save_figure(fig, figures, name))


def _rows(table: Table, keep: np.ndarray, log_time: bool) -> Table:
    cols = {k: np.asarray(v)[keep] for k, v in table.columns.items()}
    if log_time:
        cols["time"] = np.log(cols["time"])
    return dataclasses.replace(table, columns=cols)


def _replica_series(traj: Table, observable: str) -> tuple[np.ndarray, np.ndarray]:
    """(values[R, T], times[T]) from a trajectories table (rows: replica, time, observables)."""
    rep, time = np.asarray(traj["replica"], float), np.asarray(traj["time"], float)
    order = np.lexsort((time, rep))
    R = np.unique(rep).size
    if rep.size % R:
        raise RuntimeError(f"{traj.path}: replicas have different recording times")
    times = time[order].reshape(R, -1)
    if not np.all(times == times[0]):
        raise RuntimeError(f"{traj.path}: replicas have different recording times")
    return np.asarray(traj[observable], float)[order].reshape(R, -1), times[0]


def _rho(scale: str, data: Path, figures: Path) -> list[Check]:
    """voter.rho_2d: rho_sim(t) / rho_exact(t) over t >= RHO_T_MIN, predicted 1. The exact
    rho(t) (theory.voter_rho_2d) has no free parameter, so this tests the normalisation of rho
    and the time unit too. The times share replicas, so the error is that of a bootstrap over
    whole replicas and the consistency is Hotelling's T^2 (replica_ratio).

    voter.rho_2d_exponent (diagnostic): b in 1/rho = A (ln t)^b + c. Asymptotically
    1/rho = (2/pi) ln t + const (Frachebourg-Krapivsky Eq. 23), b = 1, but the approach is
    logarithmic (the exact curve fits b ~ 1.05 at t <= 10^4), so the prediction is the same
    estimator applied to the exact curve: fit_timeseries takes its parameters from an
    unweighted fit to the means, so the exact curve is fitted unweighted at the same times.
    This form cannot see rho -> lambda rho and hardly t -> kappa t; the ratio is the check."""
    (L, steps), _, _, _ = SCALE[scale]["rho_2d"]
    out = find_outputs(data / "rho_2d")
    ts, traj = out.get("timeseries_ensemble"), out.get("trajectories")
    if ts is None or traj is None:
        raise RuntimeError(f"{data / 'rho_2d'}: needs the ensemble time series and trajectories")
    window = np.asarray(ts["time"], float) >= RHO_T_MIN
    warns = []
    running = np.asarray(ts["running_fraction"], float)[window]
    if np.any(running < 1):
        warns.append(f"some replicas reached consensus inside the window (running fraction "
                     f"down to {running.min():.3g}); their rho = 0 enters the mean")
    sizes = f"L={L} (2D), t = {RHO_T_MIN}..{steps} sweeps"

    V, times = _replica_series(traj, "rho")
    keep = times >= RHO_T_MIN
    t, V = times[keep], V[:, keep]
    R = V.shape[0]
    exact = voter_rho_2d(t)
    ratio = replica_ratio(V, exact)
    name = "voter.rho_2d"
    tt = np.geomspace(t.min(), t.max(), 100)
    fig = plot_ratio(t, ratio, "t (sweeps)", prov=ts.provenance,
                     title=f"voter, {L}x{L} lattice: active-link density vs exact",
                     absolute=(V.mean(axis=0), V.std(axis=0, ddof=1) / np.sqrt(R), tt,
                               voter_rho_2d(tt), "rho (active links)"))
    ratio_chk = ratio_check(ratio, model=NAME, check=name, source="voter_interface_2d",
                            quantity="rho(t) / exact rho(t) (density of active links, weighted "
                                     "over the recorded times)",
                            sizes=sizes, statistics=f"{R} replicas; errors over whole replicas",
                            warnings=warns, report_consistency=True,
                            figure=save_figure(fig, figures, name))

    r = fit_timeseries(_rows(ts, window, True), "rho", model="power+c",
                       trajectories=_rows(traj, np.asarray(traj["time"], float) >= RHO_T_MIN, True),
                       transform=lambda v: 1.0 / v, seed=0)
    u = np.asarray(r.used, bool)
    b_exact = fit(r.x[u], 1.0 / voter_rho_2d(np.exp(r.x[u])), model="power+c",
                  unweighted=True).params["b"]
    name = "voter.rho_2d_exponent"
    xs = np.linspace(r.x[u].min(), r.x[u].max(), 60)
    fig = plot_exponent(r, b_exact, "ln t (t in sweeps)", "1 / rho", prov=ts.provenance,
                        title=f"voter, {L}x{L} lattice: active-link density",
                        theory=(xs, 1.0 / voter_rho_2d(np.exp(xs)),
                                f"exact rho(t), same fit: b = {b_exact:.3f}"),
                        legend="lower right")
    ax = fig.axes[0]
    ax.plot(xs, 2 / np.pi * xs + 8 * np.log(2) / np.pi, ":", color="#555555", lw=1,
            label="asymptote (2 ln t + ln 256)/pi")
    ax.legend(loc="lower right", fontsize=6.5)
    exp_chk = exponent_check(r, b_exact, model=NAME, check=name, source="voter_interface_2d",
                             quantity="diagnostic: b in 1/rho = A (ln t)^b + c; predicted: the "
                                      "same (unweighted) fit to the exact rho(t) (-> 1 as t -> inf)",
                             sizes=sizes, statistics=f"{R} replicas; bootstrap over replicas",
                             warnings=warns, figure=save_figure(fig, figures, name))
    return [ratio_chk, exp_chk]


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_complete(scale, data, figures), _exit(scale, data, figures),
            _lattice(scale, data, figures, 1), _lattice(scale, data, figures, 2),
            _lattice(scale, data, figures, 3), _ba(scale, data, figures),
            *_rho(scale, data, figures)]
