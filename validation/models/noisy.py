"""The noisy voter model (--model noisy-voter, opinions +/-1) on the complete graph: its
stationary law is that of the exact birth-death chain on n, the number of +1 nodes
(theory.noisy_voter_rates, theory.stationary_distribution), at any N.

noisy.stationary: <m^2> of the sampled m = (2n - N)/N against the exact chain's, as a weighted
ratio over noise N in {0.3, 1, 3} (bimodal, flat-ish and unimodal laws), predicted 1. Each
noise value is its own run of independent chains; the samples within a chain are
autocorrelated, so the error is the leave-one-chain-out jackknife of the pooled <m^2>, which
for chains of equal length is the spread of the chain means / sqrt(chains). The figure shows
the driver's histogram of m (`--bins`) against the exact probability mass of each bin.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from odspy import error_bar_uncertainty, find_outputs
from odspy.io import Table
from odspy.plots.style import color, finish, new_fig

from validation.analysis import ratio_check, save_figure, weighted_ratio
from validation.core import Check, Run
from validation.theory import noisy_voter_rates, stationary_distribution

NAME = "noisy"

NOISE_N = (0.3, 1.0, 3.0)        # noise * N: Beta-like laws with parameter ~ noise N / 2
BURN_IN_RELAX = 20                # burn-in in units of the relaxation time 1/noise (sweeps)
SAMPLES_PER_RELAX = 30            # samples per relaxation time (sets --interval)

# Per scale: N, chains (--replicas), samples per chain, threads per job. The relaxation time
# of m is 1/noise sweeps (each update randomises with probability noise; copying conserves
# <m>), so burn-in and sampling interval scale with it; tau_int(m^2) is ~0.4 / noise sweeps.
SCALE = {
    "smoke": (32, 8, 1000, 1),
    "small": (100, 40, 4000, 4),
    "full": (256, 100, 10000, 4),
}

SEEDS = {0.3: 701, 1.0: 702, 3.0: 703}


def _run_name(noise_n: float) -> str:
    return "noiseN_" + f"{noise_n:g}".replace(".", "p")


def _schedule(N: int, noise_n: float) -> tuple[float, int, int]:
    """(noise, burn-in sweeps, interval sweeps) for noise * N = noise_n."""
    noise = noise_n / N
    relax = 1.0 / noise
    return noise, math.ceil(BURN_IN_RELAX * relax), max(1, round(relax / SAMPLES_PER_RELAX))


def runs(scale: str) -> list[Run]:
    N, chains, samples, threads = SCALE[scale]
    out = []
    for eps in NOISE_N:
        noise, burn, interval = _schedule(N, eps)
        out.append(Run(_run_name(eps),
                       ("--model", "noisy-voter", "--noise", repr(noise), "--graph", "complete",
                        "--n", str(N), "--task", "stationary", "--observable", "m",
                        "--burn-in", str(burn), "--samples", str(samples),
                        "--interval", str(interval), "--replicas", str(chains),
                        "--bins", str(N + 1)),
                       seed=SEEDS[eps], threads=threads))
    return out


# ---------------------------------------------------------------------------------------------

def chain_values(samples: Table, N: int) -> np.ndarray:
    """n (number of +1 nodes) of every sample, as an array [chains, samples per chain]."""
    chain = np.asarray(samples["chain"], int)
    m = np.asarray(samples["m"], float)
    ids = np.unique(chain)
    parts = [m[chain == c] for c in ids]
    if any(p.size != parts[0].size for p in parts):
        raise RuntimeError(f"{samples.path}: chains have different numbers of samples")
    n = np.rint((np.stack(parts) + 1.0) * N / 2.0)
    if np.max(np.abs((2 * n - N) / N - np.stack(parts))) > 1e-6:
        raise RuntimeError(f"{samples.path}: m is not of the form (2n - N)/N with N = {N}")
    return n.astype(int)


def bin_index(N: int, lo: float, hi: float, bins: int) -> np.ndarray:
    """The histogram bin of each lattice value m = (2n - N)/N, n = 0..N, with the driver's rule
    (stats::histogram: equal bins over [lo, hi], floor((m - lo)/width), clamped to the ends)."""
    m = (2.0 * np.arange(N + 1) - N) / N
    width = (hi - lo) / bins
    idx = np.floor((m - lo) / width) if width > 0 else np.zeros(N + 1)
    return np.clip(idx, 0, bins - 1).astype(int)


def _point(prefix: Path, noise_n: float) -> dict:
    out = find_outputs(prefix)
    samples, hist = out.get("samples"), out.get("histogram")
    if samples is None or hist is None:
        raise RuntimeError(f"{prefix}: needs the stationary samples and histogram")
    cfg = samples.provenance.config
    N = int(cfg["n"])
    noise = _schedule(N, noise_n)[0]     # the exact value passed (the config may round it)
    if not math.isclose(float(cfg["noise"]), noise, rel_tol=1e-5):
        raise RuntimeError(f"{prefix}: noise {cfg['noise']} in the outputs, expected {noise!r}")
    n = chain_values(samples, N)
    chains = n.shape[0]
    m2 = ((2.0 * n - N) / N) ** 2
    per_chain = m2.mean(axis=1)
    sim = float(per_chain.mean())
    # leave-one-chain-out jackknife of the pooled mean (= sd of chain means / sqrt(chains))
    loo = (per_chain.sum() - per_chain) / (chains - 1)
    err = float(math.sqrt((chains - 1) / chains * np.sum((loo - loo.mean()) ** 2)))
    pi = stationary_distribution(*noisy_voter_rates(N, noise))
    m = (2.0 * np.arange(N + 1) - N) / N
    exact = float(pi @ m ** 2)

    lo, hi = float(hist["bin_low"][0]), float(hist["bin_high"][-1])
    bins = len(hist)
    idx = bin_index(N, lo, hi, bins)
    mass = np.bincount(idx, weights=pi, minlength=bins)
    counts = np.stack([np.bincount(idx[row], minlength=bins) for row in n])   # [chains, bins]
    total = counts.sum()
    width = (hi - lo) / bins
    frac = counts / counts.sum(axis=1, keepdims=True)
    warns = []
    if not np.array_equal(counts.sum(axis=0), np.asarray(hist["count"], int)):
        warns.append(f"{prefix.name}: histogram recomputed from the samples differs from the "
                     "driver's (bin edges); the figure uses the recomputed one")
    burn = float(cfg["burn-in"])
    if burn * noise < 10:
        warns.append(f"{prefix.name}: burn-in {burn:g} sweeps is under 10 relaxation times "
                     f"(1/noise = {1 / noise:.3g} sweeps)")
    return {"N": N, "noise": noise, "chains": chains, "samples": n.shape[1], "sim": sim,
            "err": err, "exact": exact, "centres": lo + width * (np.arange(bins) + 0.5),
            "width": width, "density": counts.sum(axis=0) / total / width,
            "density_err": frac.std(axis=0, ddof=1) / math.sqrt(chains) / width,
            "exact_density": mass / width, "prov": samples.provenance, "warnings": warns}


def analyse(scale: str, data: Path, figures: Path) -> list[Check]:
    points = [_point(data / _run_name(eps), eps) for eps in NOISE_N]
    ratio = weighted_ratio([p["sim"] for p in points], [p["err"] for p in points],
                           [p["exact"] for p in points])
    N, chains = points[0]["N"], points[0]["chains"]
    name = "noisy.stationary"
    fig, axes = new_fig(1, len(points))
    for ax, p, eps in zip(axes[0], points, NOISE_N):
        ax.errorbar(p["centres"], p["density"], yerr=p["density_err"], fmt="o", ms=2.5,
                    color=color(0), capsize=0, label="simulation")
        ax.plot(p["centres"], p["exact_density"], "-", color=color(1), lw=1, label="exact chain")
        ax.set(xlabel="m", ylabel="P(m)", ylim=(0, None),
               title=f"noise N = {eps:g}: <m²> ratio {p['sim'] / p['exact']:.4f} ± "
                     f"{p['err'] / p['exact']:.2g}")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7)
    finish(fig, points[0]["prov"], f"noisy voter, complete graph N = {N}: stationary law of m")
    warns = [w for p in points for w in p["warnings"]]
    warns.append(f"error bars from {chains} chains are themselves uncertain by about "
                 f"+-{100 * error_bar_uncertainty(chains):.0f}%")
    return [ratio_check(ratio, model=NAME, check=name, source="noisy_voter",
                        quantity="<m^2>_sim / <m^2>_exact (exact birth-death chain), weighted "
                                 "over noise N = " + ", ".join(f"{e:g}" for e in NOISE_N),
                        sizes=f"N={N} (complete)",
                        statistics=f"{chains} chains x {points[0]['samples']} samples per noise; "
                                   "leave-one-chain-out jackknife",
                        warnings=warns, report_consistency=True,
                        figure=save_figure(fig, figures, name))]
