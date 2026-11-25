"""The Sznajd model (--model sznajd): exit probability on the complete graph, exact at finite N,
and the sharpening of the exit-probability step on the periodic square lattice.

The rule as implemented (include/SznajdModel.hpp): one update draws a uniform directed edge
(i, j); if i and j agree, every other neighbour of i and of j adopts their opinion; a
disagreeing pair does nothing. A sweep is N updates. The start (--init random --up f) has
exactly floor(f N) nodes at +1, placed at random.

sznajd.complete_exit: on the complete graph every other node is a neighbour of i and j, so the
first agreeing pair ends the run, and P(+1) = n(n-1) / [n(n-1) + (N-n)(N-n-1)] exactly
(theory.sznajd_complete_exit, derived there), -> f^2 / (f^2 + (1-f)^2) as N -> infinity. This
replaces the agreed target "a step at 1/2 in mean field", which does not hold for this rule.
Compared as the weighted ratio of simulation to theory over N and f (predicted 1), binomial
errors under the prediction; points where the prediction is exactly 0 or 1 (n <= 1 or
n >= N - 1) must match exactly and are left out of the ratio.

sznajd.lattice_step: on L x L, E(f) = P(+1 | initial fraction f) tends to a step at 1/2 (Stauffer,
Sousa and Moss de Oliveira 2000). Its width w(L) = f(E = 0.9) - f(E = 0.1), each by linear
interpolation at the first upward crossing of the level, is fitted to A L^b over every size but
the smallest (ruling R12; the all-sizes fit in the warnings). No published value of b exists for
this rule that we could verify, so the check is one-sided (Check.one_sided = "below"): b < 0
(the width shrinks), predicted 0, passed when sigma < -3 and flagged otherwise; the warnings
say which. Errors of w: parametric bootstrap of the
binomial E at each point. The f grid of each L is f = 1/2 + u h(L), u = -2..2 in steps of 1/4,
h(L) = 0.1 (16/L)^(3/4) (from a pilot at L = 16, 32, 64: the transition keeps the same number of
grid points at every L, so the interpolation bias is the same relative fraction of w at every L
and does not bias b); E is plotted against the exact initial fraction floor(f N) / N.

sznajd.lattice_half: E(1/2) = 1/2 exactly, by the +-1 symmetry of the rule and of a start with
exactly N/2 nodes at +1 (L even), at the largest L with a binomial error; every L in the warnings.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from odspy import find_outputs, fit
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import fit_quality, plot_ratio, ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run
from validation.theory import sznajd_complete_exit, sznajd_mean_field_exit

NAME = "sznajd"

COMPLETE_UP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GRID_U = tuple(k / 4 for k in range(-8, 9))       # f = 1/2 + u h(L)
LEVELS = (0.1, 0.9)                                # the width runs between these values of E
N_BOOT = 2000

# Per scale. complete: (N values, replicas, block, threads); lattice: (L values, replicas per
# point, block, threads). Smoke: seconds; small: a few minutes at 10 threads; full: >= 5 sizes
# over a factor 8 for the width exponent.
SCALE = {
    "smoke": {"complete": ((20, 100), 400, None, 1), "lattice": ((8, 12, 16, 24), 200, None, 1)},
    "small": {"complete": ((20, 100, 1000), 4000, None, 1),
              "lattice": ((16, 24, 32, 48, 64), 1000, None, 1)},
    "full": {"complete": ((20, 100, 1000, 3000), 40000, 10000, 1),
             "lattice": ((16, 24, 32, 48, 64, 96, 128), 10000, 2500, 1)},
}

SEEDS = {"complete": 1101, "lattice": 1102}


def half_width(L: int) -> float:
    """The grid scale h(L) = 0.1 (16 / L)^(3/4)."""
    return 0.1 * (16.0 / L) ** 0.75


def grid(L: int) -> list[str]:
    """The initial fractions of size L as exact decimal strings, symmetric about 1/2."""
    h = half_width(L)
    return [f"{0.5 + round(u * h, 9):.9f}" for u in GRID_U]


def runs(scale: str) -> list[Run]:
    s = SCALE[scale]
    Ns, R, block, threads = s["complete"]
    out = [Run("complete", ("--model", "sznajd", "--task", "ensemble", "--graph", "complete",
                            "--init", "random", "--replicas", str(R),
                            "--sweep", "n=" + ",".join(map(str, Ns)),
                            "--sweep2", "up=" + ",".join(f"{f:g}" for f in COMPLETE_UP)),
               seed=SEEDS["complete"], threads=threads, block=block)]
    Ls, R, block, threads = s["lattice"]
    for L in Ls:
        # consensus took at most 0.75 L^2 sweeps in a pilot (mean ~ 0.08 L^2 at f = 1/2)
        out.append(Run(f"lattice_L{L}", ("--model", "sznajd", "--task", "ensemble", "--graph",
                                         "lattice", "--dims", f"{L}x{L}", "--boundary", "periodic",
                                         "--init", "random", "--steps", str(10 * L * L),
                                         "--replicas", str(R), "--sweep", "up=" + ",".join(grid(L))),
                       seed=SEEDS["lattice"] + L, threads=threads, block=block))
    return out


# ---------------------------------------------------------------------------------------------

def _sweep(prefix: Path) -> Table:
    table = find_outputs(prefix).get("sweep")
    if table is None:
        raise RuntimeError(f"{prefix}: no sweep table")
    return table


def _capped(table: Table, label: str) -> list[str]:
    frac = np.asarray(table["consensus_fraction"], float)
    if np.all(frac == 1):
        return []
    return [f"{label}: runs capped at --steps before consensus (consensus_fraction down to "
            f"{frac.min():.4g}); they count as not reaching +1"]


def _complete(scale: str, data: Path, figures: Path) -> Check:
    t = _sweep(data / "complete")
    N = np.asarray(t["n"], int)
    up = np.asarray(t["up"], float)
    R = np.asarray(t["replicas"], float)
    sim = np.asarray(t["exit_prob_up"], float)
    n = np.array([int(f * k) for f, k in zip(up, N)])        # floor(f N) in double, as the driver
    P = np.array([sznajd_complete_exit(int(a), int(b)) for a, b in zip(n, N)])
    warns = ["replaces the agreed target 'a step at 1/2 in mean field': ODSP's outflow rule "
             "converts every other node at the first agreeing pair, so P(+1) = "
             "n(n-1)/[n(n-1) + (N-n)(N-n-1)] -> f^2/(f^2 + (1-f)^2) (theory.sznajd_complete_exit)"]
    warns += _capped(t, "complete graph")
    exact = (P == 0) | (P == 1)
    bad = exact & (sim != P)
    if np.any(bad):
        warns.append("points with an exact 0 or 1 prediction differ: " + ", ".join(
            f"N={k} f={f:g} ({s:g})" for k, f, s in zip(N[bad], up[bad], sim[bad])))
    if np.any(exact):
        warns.append(f"{int(exact.sum())} points with prediction exactly 0 or 1 (n <= 1 or "
                     "n >= N-1) left out of the ratio" + ("" if np.any(bad) else "; they match exactly"))
    keep = ~exact
    err = np.sqrt(P * (1 - P) / R)                            # binomial under the prediction
    ratio = weighted_ratio(sim[keep], err[keep], P[keep])
    name = "sznajd.complete_exit"
    ff = np.linspace(0.0, 1.0, 101)
    fig = plot_ratio(n[keep] / N[keep], ratio, "initial fraction of +1", logx=False,
                     prov=t.provenance, title="Sznajd, complete graph: exit probability",
                     absolute=(sim[keep], err[keep], ff, [sznajd_mean_field_exit(f) for f in ff],
                               "P(+1 consensus)"))
    return ratio_check(ratio, model=NAME, check=name, source="sznajd_complete",
                       quantity="P(+1 consensus) / exact n(n-1)/[n(n-1) + (N-n)(N-n-1)] "
                                "(n = floor(f N) nodes at +1), weighted over N and f",
                       sizes="N=" + ",".join(map(str, sorted(set(N.tolist())))) + ", f="
                             + ",".join(f"{f:g}" for f in COMPLETE_UP),
                       statistics=f"{int(R.min())} replicas per point; binomial errors",
                       warnings=warns, report_consistency=True,
                       figure=save_figure(fig, figures, name))


def _crossing(f: np.ndarray, E: np.ndarray, level: float) -> float:
    """f where E first reaches `level` from below, by linear interpolation; NaN if E starts at
    or above the level or never reaches it."""
    above = np.nonzero(E >= level)[0]
    if above.size == 0 or above[0] == 0:
        return math.nan
    i = int(above[0])
    return float(f[i - 1] + (level - E[i - 1]) * (f[i] - f[i - 1]) / (E[i] - E[i - 1]))


def transition_width(f, E) -> float:
    """w = f(E = 0.9) - f(E = 0.1), each at the first upward crossing of the level."""
    f, E = np.asarray(f, float), np.asarray(E, float)
    return _crossing(f, E, LEVELS[1]) - _crossing(f, E, LEVELS[0])


def _lattice_curves(scale: str, data: Path):
    """Per size: (L, exact initial fractions, E, replicas per point, f = 1/2 row, table)."""
    out = []
    for L in SCALE[scale]["lattice"][0]:
        t = _sweep(data / f"lattice_L{L}")
        N = L * L
        f = np.array([int(v * N) / N for v in np.asarray(t["up"], float)])
        order = np.argsort(f)
        E = np.asarray(t["exit_prob_up"], float)[order]
        R = np.asarray(t["replicas"], float)[order]
        out.append((L, f[order], E, R, t))
    return out


def _width(f, E, R, rng) -> tuple[float, float, int]:
    """(w, bootstrap sd, discarded draws): E at each point redrawn as Binomial(R, E) / R."""
    w = transition_width(f, E)
    if not math.isfinite(w):
        raise RuntimeError(f"the f grid does not bracket E = {LEVELS} (E = {np.round(E, 3)})")
    draws = rng.binomial(R.astype(int)[None, :], E[None, :], size=(N_BOOT, E.size)) / R
    ws = np.array([transition_width(f, d) for d in draws])
    ok = np.isfinite(ws)
    return w, float(np.std(ws[ok], ddof=1)), int(np.sum(~ok))


def _lattice(scale: str, data: Path, figures: Path) -> list[Check]:
    curves = _lattice_curves(scale, data)
    rng = np.random.default_rng(SEEDS["lattice"])
    Ls, ws, es, warns = [], [], [], []
    for L, f, E, R, t in curves:
        w, e, lost = _width(f, E, R, rng)
        Ls.append(L)
        ws.append(w)
        es.append(e)
        warns += _capped(t, f"L={L}")
        if lost:
            warns.append(f"L={L}: {lost} of {N_BOOT} bootstrap draws did not bracket both levels")
    L_arr, w_arr, e_arr = (np.asarray(v, float) for v in (Ls, ws, es))
    r = fit(L_arr, w_arr, e_arr, model="power", xrange=(Ls[1], Ls[-1]))       # R12
    r_all = fit(L_arr, w_arr, e_arr, model="power")
    b, eb = r.params["b"], r.errors["b"]
    sigma = b / eb
    widths = ", ".join(f"L={L} {w:.4f} ± {e:.2g}" for L, w, e in zip(Ls, ws, es))
    step_warns = list(warns) + [
        f"widths w(L): {widths}",
        f"finite-size diagnostic: fit over all sizes (L >= {Ls[0]}): b = {r_all.params['b']:.3f} "
        f"± {r_all.errors['b']:.2g}",
        ("one-sided check passed: b < 0 at " if sigma < -3 else
         "ONE-SIDED CHECK FAILED: b is not below 0 by 3 errors (") + f"{-sigma:.1f} errors"
        + ("" if sigma < -3 else ")")] + fit_quality(r)

    fig, axes = new_fig(1, 2)
    a0, a1 = axes[0, 0], axes[0, 1]
    for k, (L, f, E, R, t) in enumerate(curves):
        a0.errorbar(f, E, yerr=np.sqrt(E * (1 - E) / R), fmt="o-", ms=3, lw=0.8, color=color(k),
                    capsize=0, label=f"L = {L}")
    for lev in LEVELS:
        a0.axhline(lev, color="#888888", lw=0.6, ls=":")
    a0.set(xlabel="initial fraction of +1", ylabel="P(+1 consensus)")
    a0.legend(fontsize=6.5)
    used = np.asarray(r.used, bool)
    a1.errorbar(L_arr[used], w_arr[used], yerr=e_arr[used], fmt="o", ms=4, color=color(0),
                capsize=0, label="fitted")
    a1.errorbar(L_arr[~used], w_arr[~used], yerr=e_arr[~used], fmt="o", ms=4, color=color(0),
                alpha=0.35, capsize=0, label="not fitted")
    xs = np.geomspace(L_arr.min(), L_arr.max(), 100)
    a1.plot(xs, r.predict(xs), "-", color=color(0), lw=1, label=f"fit: b = {b:.3f} ± {eb:.2g}")
    a1.set(xscale="log", yscale="log", xlabel="L", ylabel="width f(E=0.9) - f(E=0.1)")
    a1.legend()
    name = "sznajd.lattice_step"
    fig = finish(fig, curves[-1][4].provenance, "Sznajd, periodic L x L: exit-probability step")
    sizes = "L=" + ",".join(map(str, Ls)) + f" (fit L >= {Ls[1]})"
    step = Check(model=NAME, check=name,
                 quantity="one-sided: b in w ~ L^b, w = f(E = 0.9) - f(E = 0.1) the width of the "
                          "exit-probability step (linear interpolation); predicted b < 0 (the "
                          "step sharpens), passed when sigma < -3",
                 measured=b, error=eb, predicted=0.0, predicted_error=0.0, source="sznajd_lattice",
                 sizes=sizes, statistics=f"{int(curves[-1][3].min())} replicas per point, "
                                         f"{len(GRID_U)} points per L; parametric bootstrap of "
                                         "the binomial E",
                 warnings=step_warns, figure=save_figure(fig, figures, name),
                 one_sided="below")

    halves = []
    for L, f, E, R, t in curves:
        i = int(np.argmin(np.abs(f - 0.5)))
        if f[i] != 0.5:
            raise RuntimeError(f"L={L}: no point at f = 1/2")
        halves.append((L, E[i], math.sqrt(0.25 / R[i]), int(R[i])))
    L_h, E_h, e_h, R_h = halves[-1]
    name = "sznajd.lattice_half"
    fig, axes = new_fig()
    ax = axes[0, 0]
    ax.errorbar([h[0] for h in halves], [h[1] for h in halves], yerr=[h[2] for h in halves],
                fmt="o", ms=4, color=color(0), capsize=0, label="simulation")
    ax.axhline(0.5, color=color(1), lw=1, label="1/2 (symmetry)")
    ax.set(xscale="log", xlabel="L", ylabel="P(+1 consensus) at f = 1/2")
    ax.legend()
    fig = finish(fig, curves[-1][4].provenance, "Sznajd, periodic L x L: E(1/2)")
    half = Check(model=NAME, check=name,
                 quantity="P(+1 consensus) from exactly N/2 nodes at +1, largest L",
                 measured=E_h, error=e_h, predicted=0.5, predicted_error=0.0,
                 source="sznajd_lattice", sizes=f"L={L_h}",
                 statistics=f"{R_h} replicas; binomial error under the prediction",
                 warnings=list(warns) + ["E(1/2) at every L: " + ", ".join(
                     f"L={L} {E:.4f} ± {e:.2g}" for L, E, e, _ in halves)],
                 figure=save_figure(fig, figures, name))
    return [step, half]


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    return [_complete(scale, data, figures), *_lattice(scale, data, figures)]
