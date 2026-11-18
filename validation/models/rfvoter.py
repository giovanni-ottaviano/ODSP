"""The rejection-free voter model (--model rf-voter): an exact sampler of the node-update voter
chain, so its consensus times must have the voter model's distribution. Both run on the same
1D ring (where the rejection-free jump regime dominates) with the same seed, in single
updates, so the jump regime's geometric waiting times are tested at full resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from odspy import find_outputs
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import P_THRESHOLD, ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run

NAME = "rfvoter"

# (ring length L, replicas, block, threads) per scale
SCALE = {
    "smoke": (32, 200, None, 1),
    "small": (128, 4000, 500, 1),
    "full": (256, 100000, 10000, 4),
}
SEED = 611


def runs(scale: str) -> list[Run]:
    L, R, block, threads = SCALE[scale]
    # mean consensus time ~ 0.1 L^2 sweeps = 0.1 L^3 updates, exponential tail of about the
    # same decay time: a cap of 2 L^3 updates censors with probability ~e^-20
    args = ("--graph", "lattice", "--dims", str(L), "--task", "ensemble", "--replicas", str(R),
            "--time-unit", "updates", "--steps", str(2 * L ** 3))
    return [Run(name, ("--model", model) + args, seed=SEED, threads=threads, block=block)
            for name, model in (("voter", "voter"), ("rfvoter", "rf-voter"))]


def _times(prefix: Path) -> tuple[np.ndarray, Table]:
    table = find_outputs(prefix).get("replicas")
    if table is None:
        raise RuntimeError(f"{prefix}: no replicas file")
    return np.asarray(table["stop_time"], float), table


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    L = SCALE[scale][0]
    name = "rfvoter.same_statistics"
    t_v, tab_v = _times(data / "voter")
    t_rf, tab_rf = _times(data / "rfvoter")
    warns = []
    for label, tab in (("voter", tab_v), ("rf-voter", tab_rf)):
        open_ = int(np.sum(np.asarray(tab["reached_consensus"], float) < 0.5))
        if open_:
            warns.append(f"{label}: {open_} replicas capped at --steps (mean biased low)")
    # ratio of means; the voter mean's error enters as the "theory" error
    ratio = weighted_ratio([t_rf.mean()], [t_rf.std(ddof=1) / np.sqrt(t_rf.size)],
                           [t_v.mean()], [t_v.std(ddof=1) / np.sqrt(t_v.size)])
    ks = stats.ks_2samp(t_rf, t_v)
    if ks.pvalue < P_THRESHOLD:
        warns.append(f"two-sample KS test of the consensus-time distributions: D = "
                     f"{ks.statistic:.3g}, p = {ks.pvalue:.2g}")
    fig, axes = new_fig()
    ax = axes[0, 0]
    for i, (label, t) in enumerate((("voter", t_v), ("rf-voter", t_rf))):
        s = np.sort(t)
        ax.step(s / L ** 3, np.arange(1, s.size + 1) / s.size, where="post", color=color(i),
                lw=1, label=f"{label}: mean {t.mean() / L ** 3:.4g} L^3")
    ax.set(xscale="log", xlabel="consensus time / L^3 (single updates)", ylabel="cumulative fraction")
    ax.legend(title=f"ratio {ratio.value:.4f} ± {ratio.error:.2g}; KS p = {ks.pvalue:.2g}",
              loc="upper left")
    finish(fig, tab_rf.provenance, f"rf-voter vs voter, 1D ring L={L}")
    return [ratio_check(ratio, model=NAME, check=name, source="rejection_free",
                        quantity="mean consensus time rf-voter / voter (same ring, same seed)",
                        sizes=f"L={L} (1D)", statistics=f"{t_rf.size} + {t_v.size} replicas",
                        warnings=warns, figure=save_figure(fig, figures, name))]
