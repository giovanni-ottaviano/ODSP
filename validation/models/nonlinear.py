"""The nonlinear voter model (--model nonlinear-voter: the chosen node flips with probability
f^alpha, f its disagreeing fraction of neighbours) on the complete graph, where it is exactly
the birth-death chain of theory.nonlinear_voter_rates at any N.

nonlinear.exit: P(+1 consensus) against the initial fraction of +1 nodes for alpha = 0.5
(coexistence: the drift pulls m to 0, the exit probability is flattened towards 1/2) and
alpha = 2 (ordering: a sharp step at 1/2), as sim / exact (theory.exit_probability) weighted
over both, predicted 1.
nonlinear.time: the mean consensus time from n = N/2, in single updates, against the exact
theory.mean_absorption_time, weighted over N for alpha = 2 (T ~ N ln N updates, i.e. ~ ln N
sweeps) and over the few small N where the alpha = 0.5 time (exponential in N) is still
practical, predicted 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

from odspy import find_outputs
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import Ratio, ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run
from validation.theory import exit_probability, mean_absorption_time, nonlinear_voter_rates

NAME = "nonlinear"

EXIT_N = 16           # alpha = 0.5 still reaches consensus in ~75 sweeps here
EXIT_UP = {0.5: (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375),
           2.0: (0.25, 0.375, 0.5, 0.625, 0.75)}   # alpha = 2: P >= 0.018 (enough exits)
CAP_FACTOR = 40       # --steps cap = 40 x the largest exact mean time: P(capped) ~ e^-40

# Per scale: exit replicas per point and their block (replicas per job); alpha = 2 sizes;
# alpha = 0.5 sizes; time replicas per N and their block; threads per job.
SCALE = {
    "smoke": {"exit": 1000, "exit_block": None, "n2": (32, 64, 128, 256), "n05": (8, 12, 16), "reps": 200,
              "block": None, "threads": 1},
    "small": {"exit": 20000, "exit_block": 5000, "n2": (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
              "n05": (8, 12, 16, 20), "reps": 2000, "block": 500, "threads": 1},
    "full": {"exit": 200000, "exit_block": 25000, "n2": (64, 256, 1024, 4096, 16384, 65536),
             "n05": (8, 12, 16, 20, 24), "reps": 10000, "block": 1000, "threads": 4},
}

SEEDS = {"exit_alpha05": 711, "exit_alpha2": 712, "time_alpha2": 713, "time_alpha05": 714}


def _tag(alpha: float) -> str:
    return "alpha" + f"{alpha:g}".replace(".", "")


def _exact_time(N: int, alpha: float) -> float:
    """Exact mean number of single updates to consensus from n = N // 2 (--init random with
    --up 0.5 starts with floor(0.5 N) nodes at +1)."""
    return float(mean_absorption_time(*nonlinear_voter_rates(N, alpha))[N // 2])


def _ensemble(alpha: float, *extra: str) -> tuple[str, ...]:
    return ("--model", "nonlinear-voter", "--alpha", f"{alpha:g}", "--graph", "complete",
            "--task", "ensemble", "--time-unit", "updates", *extra)


def runs(scale: str) -> list[Run]:
    s = SCALE[scale]
    out = []
    for alpha, ups in EXIT_UP.items():
        cap = CAP_FACTOR * float(np.max(mean_absorption_time(*nonlinear_voter_rates(EXIT_N, alpha))))
        out.append(Run(f"exit_{_tag(alpha)}",
                       _ensemble(alpha, "--n", str(EXIT_N), "--steps", str(int(cap)),
                                 "--sweep", "up=" + ",".join(f"{f:g}" for f in ups),
                                 "--replicas", str(s["exit"])),
                       seed=SEEDS[f"exit_{_tag(alpha)}"], threads=s["threads"],
                       block=s["exit_block"]))
    for alpha, key in ((2.0, "n2"), (0.5, "n05")):
        Ns = s[key]
        cap = CAP_FACTOR * _exact_time(max(Ns), alpha)
        out.append(Run(f"time_{_tag(alpha)}",
                       _ensemble(alpha, "--steps", str(int(cap)),
                                 "--sweep", "n=" + ",".join(map(str, Ns)),
                                 "--replicas", str(s["reps"])),
                       seed=SEEDS[f"time_{_tag(alpha)}"], threads=s["threads"], block=s["block"]))
    return out


# ---------------------------------------------------------------------------------------------

def _sweep(prefix: Path) -> Table:
    table = find_outputs(prefix).get("sweep")
    if table is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    return table


def _censored(table: Table, key: str) -> list[str]:
    frac = np.asarray(table["consensus_fraction"], float)
    if np.all(frac == 1):
        return []
    named = ", ".join(f"{key}={v:g} ({f:.3g})" for v, f in zip(table[key], frac) if f < 1)
    return [f"{table.path.name}: runs capped at --steps before consensus at {named}; the mean "
            "is biased low there"]


def _exit(scale: str, data: Path, figures: Path) -> Check:
    rows, warns = [], []
    for alpha in EXIT_UP:
        t = _sweep(data / f"exit_{_tag(alpha)}")
        up = np.asarray(t["up"], float)
        R = np.asarray(t["replicas"], float)
        exact_all = exit_probability(*nonlinear_voter_rates(EXIT_N, alpha))
        # floor(up N) nodes start at +1, computed in double precision as the driver does
        p = np.array([exact_all[int(f * EXIT_N)] for f in up])
        rows.append((alpha, up, np.asarray(t["exit_prob_up"], float), np.sqrt(p * (1 - p) / R),
                     p, exact_all, t))
        warns += _censored(t, "up")
    ratio = weighted_ratio(np.concatenate([r[2] for r in rows]), np.concatenate([r[3] for r in rows]),
                           np.concatenate([r[4] for r in rows]))
    name = "nonlinear.exit"
    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0]
    k = 0
    for i, (alpha, up, sim, err, p, exact_all, _) in enumerate(rows):
        a0.errorbar(up, sim, yerr=err, fmt="os"[i], ms=4, color=color(i), capsize=0,
                    label=f"alpha = {alpha:g}")
        a0.plot(np.arange(EXIT_N + 1) / EXIT_N, exact_all, "-", color=color(i), lw=1)
        a1.errorbar(up, ratio.ratios[k:k + up.size], yerr=ratio.errors[k:k + up.size],
                    fmt="os"[i], ms=4, color=color(i), capsize=0, label=f"alpha = {alpha:g}")
        k += up.size
    a0.plot([0, 1], [0, 1], ":", color="#777777", lw=1, label="voter (alpha = 1)")
    a0.set(xlabel="initial fraction of +1", ylabel="P(+1 consensus)",
           title="lines: exact chain")
    a0.title.set_fontsize(8)
    a0.legend(fontsize=7)
    _ratio_axes(a1, ratio, "initial fraction of +1", logx=False)
    finish(fig, rows[0][6].provenance, f"nonlinear voter, complete graph N = {EXIT_N}: exit "
                                       "probability")
    R = int(min(np.min(r[6]["replicas"]) for r in rows))
    return ratio_check(ratio, model=NAME, check=name, source="nonlinear_voter",
                       quantity="P(+1 consensus) / exact chain exit probability, weighted over "
                                "alpha = 0.5, 2 and the initial fractions",
                       sizes=f"N={EXIT_N} (complete); up=" + "; ".join(
                           f"alpha {a:g}: " + ",".join(f"{f:g}" for f in ups)
                           for a, ups in EXIT_UP.items()),
                       statistics=f"{R} replicas per point; binomial errors at the exact p",
                       warnings=warns, figure=save_figure(fig, figures, name))


def _ratio_axes(ax, ratio: Ratio, xlabel: str, logx: bool) -> None:
    ax.axhline(1.0, color="#555555", lw=1, label="exact")
    ax.axhspan(ratio.value - ratio.error, ratio.value + ratio.error, color=color(0), alpha=0.15,
               lw=0, label=f"weighted mean {ratio.value:.4f} ± {ratio.error:.2g}")
    ax.set(xscale="log" if logx else "linear", xlabel=xlabel, ylabel="simulation / exact")
    ax.legend(fontsize=7)


def _time(scale: str, data: Path, figures: Path) -> Check:
    sets, warns = [], []
    for alpha in (2.0, 0.5):
        t = _sweep(data / f"time_{_tag(alpha)}")
        N = np.asarray(t["n"], int)
        R = np.asarray(t["replicas"], float)
        sim = np.asarray(t["mean_consensus_time"], float)
        err = np.asarray(t["sd_consensus_time"], float) / np.sqrt(R)
        exact = np.array([_exact_time(int(n), alpha) for n in N])
        sets.append((alpha, N, sim, err, exact, t))
        warns += _censored(t, "n")
    ratio = weighted_ratio(np.concatenate([s[2] for s in sets]), np.concatenate([s[3] for s in sets]),
                           np.concatenate([s[4] for s in sets]))
    name = "nonlinear.time"
    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0]
    alpha, N, sim, err, exact, t2 = sets[0]
    grid = np.unique(np.geomspace(N.min(), N.max(), 40).astype(int))
    curve = np.array([_exact_time(int(n), alpha) for n in grid]) / grid
    a0.errorbar(N, sim / N, yerr=err / N, fmt="o", ms=4, color=color(0), capsize=0,
                label="simulation, alpha = 2")
    a0.plot(grid, curve, "-", color=color(1), lw=1, label="exact chain")
    A, B = np.polyfit(np.log(grid), curve, 1)
    a0.plot(grid, A * np.log(grid) + B, ":", color="#555555", lw=1,
            label=f"{A:.2f} ln N {B:+.2f}")
    a0.set(xscale="log", xlabel="N", ylabel="T (sweeps = updates / N)",
           title="alpha = 2: T grows like ln N (straight on a log axis)")
    a0.title.set_fontsize(8)
    a0.xaxis.set_minor_formatter(NullFormatter())     # minor labels collide within a decade
    a0.legend(fontsize=7)
    k = 0
    for i, (alpha, N, sim, err, exact, _) in enumerate(sets):
        a1.errorbar(N, ratio.ratios[k:k + N.size], yerr=ratio.errors[k:k + N.size], fmt="os"[i],
                    ms=4, color=color(i), capsize=0, label=f"alpha = {alpha:g}")
        k += N.size
    _ratio_axes(a1, ratio, "N", logx=True)
    finish(fig, t2.provenance, "nonlinear voter, complete graph: mean consensus time")
    R = int(min(np.min(s[5]["replicas"]) for s in sets))
    return ratio_check(ratio, model=NAME, check=name, source="nonlinear_voter",
                       quantity="mean consensus time / exact T(N/2) (single updates), weighted "
                                "over N for alpha = 2 and alpha = 0.5",
                       sizes="; ".join(f"alpha {s[0]:g}: N=" + ",".join(map(str, s[1]))
                                       for s in sets),
                       statistics=f"{R} replicas per N", warnings=warns,
                       figure=save_figure(fig, figures, name))


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_exit(scale, data, figures), _time(scale, data, figures)]
