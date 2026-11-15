"""Fixation, clusters, domains, histogram and samples."""

from __future__ import annotations

import math

import numpy as np

from ..errors import InputError
from ..io import Outputs, Table
from ..sizes import size_of
from .style import (BINOMIAL, POISSON, SE, WILSON, binomial_yerr, color, finish, grid,
                    hide_unused, log_axes, masked, new_fig, numeric, note_undefined)


def _warn(warnings, text):
    if warnings is not None:
        warnings.append(text)


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


def _opinion_probs(t: Table):
    """(opinions, probability, binomial error bars (2, n), number of Wilson bars). R is
    recovered as sum(count)/sum(p); see style.binomial_yerr for p = 0 or 1."""
    p = t["probability"]
    cnt = t["replicas"] if t.has("replicas") else None
    R = np.nan
    if cnt is not None and p.sum() > 0:
        R = cnt.sum() / p.sum()
    err, edge = binomial_yerr(p, R if R >= 1 else np.nan)
    return t[t.names[0]], p, err, edge


def _binomial_label(edge: int) -> str:
    return BINOMIAL + (f"; {WILSON}" if edge else "")


def plot_fixation(t: Table, warnings: list[str] | None = None):
    if len(t) == 0 or not t.has("probability"):
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    op, p, err, edge = _opinion_probs(t)
    fig, axes = new_fig(1, 1)
    ax = axes[0, 0]
    pos = np.arange(len(p))
    ax.bar(pos, p, color=color(0), alpha=0.85, width=0.7)
    has_err = np.isfinite(err).any()
    note_undefined(warnings, f"{t.path.name} fixation", err[0], "no replica count")
    ax.errorbar(pos, p, yerr=err, fmt="none", ecolor="k",
                capsize=3, label=_binomial_label(edge) if has_err else None)
    ax.set_xticks(pos[: 25], [_fmt(o) for o in op[: 25]], rotation=45 if len(p) > 6 else 0)
    if len(p) > 25:
        _warn(warnings, f"{t.path.name}: only the first 25 of {len(p)} opinions are labelled")
    ax.set(xlabel="opinion", ylabel="probability of ending in opinion", ylim=(0, 1.05))
    if has_err:
        ax.legend(loc="best")
    return [("fixation", finish(fig, t.provenance))]


def _points(outputs: Outputs, kind: str):
    return [(i, o.get(kind)) for i, o in sorted(outputs.points.items()) if o.get(kind) is not None]


def _sweep_x(outputs: Outputs, indices, warnings):
    """x values of the swept key for the point indices, and its name."""
    sw = outputs.get("sweep")
    if sw is not None and len(sw) > max(indices, default=0):
        xcol = sw[sw.names[0]]
        if xcol.dtype != object:
            return xcol[list(indices)], sw.names[0]
        sizes = []
        for i in indices:
            try:
                sizes.append(size_of({"dims": str(xcol[i])}, "N")[0])
            except InputError:
                sizes = None
                break
        if sizes is not None:
            return np.array(sizes, float), f"{sw.names[0]} (sites N)"
    _warn(warnings, "no sweep table: point files are plotted against their index")
    return np.array(list(indices), float), "point index"


def _need_points(outputs, kind, warnings, what):
    pts = _points(outputs, kind)
    if not pts:
        _warn(warnings, f"{what}: no per-point {kind} files; figure skipped")
    return pts


def plot_fixation_vs_key(outputs: Outputs, warnings: list[str] | None = None):
    pts = _need_points(outputs, "fixation", warnings, "fixation vs key")
    if not pts:
        return []
    x, xname = _sweep_x(outputs, [i for i, _ in pts], warnings)
    ops = []
    for _, t in pts:
        for o in t[t.names[0]]:
            if o not in ops:
                ops.append(o)
    if len(ops) > 8:
        _warn(warnings, f"fixation vs key: {len(ops)} distinct opinions; only the first 8 drawn")
    fig, axes = new_fig(1, 1)
    ax = axes[0, 0]
    edges = 0
    for k, o in enumerate(ops[:8]):
        y = np.zeros(len(pts))
        e = np.full((2, len(pts)), np.nan)
        for j, (_, t) in enumerate(pts):
            op, p, err, _ = _opinion_probs(t)
            m = np.flatnonzero(op == o)
            if m.size:
                y[j], e[:, j] = p[m[0]], err[:, m[0]]
                edges += int(p[m[0]] in (0.0, 1.0) and np.isfinite(err[0, m[0]]))
        note_undefined(warnings, f"fixation vs key, opinion {_fmt(o)}", e[0], "no replica count")
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=3,
                    color=color(k), label=f"opinion {_fmt(o)}")
    ax.set(xlabel=xname, ylabel="probability of ending in opinion")
    ax.set_title(f"error bars {_binomial_label(edges)}", fontsize=8)
    ax.legend(loc="best")
    return [("fixation_vs_key", finish(fig, outputs.provenance))]


def _handles(axes):
    hs, ls = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        hs += h
        ls += l
    return hs, ls


def _total_size(outputs: Outputs, where: str):
    try:
        return size_of(outputs.provenance.config, "N", where)[0]
    except InputError:
        return None


def plot_clusters(t: Table, warnings: list[str] | None = None):
    """Cluster positions per replica (marker area = size) and the size distribution."""
    if len(t) == 0:
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    fig, axes = new_fig(1, 2, size=(4.6, 3.4))
    ax = axes[0, 0]
    size = t["size"]
    ax.scatter(t["replica"], t["position"], s=10 + 140 * size / size.max(), color=color(0),
               alpha=0.6, label="marker area ∝ cluster size")
    ax.set(xlabel="replica", ylabel="cluster position")
    ax.legend(loc="best")
    ax = axes[0, 1]
    ax.hist(size, bins=min(30, max(5, int(math.sqrt(size.size)) + 1)), color=color(0))
    ax.set(xlabel="cluster size (nodes)", ylabel="clusters")
    return [("clusters", finish(fig, t.provenance))]


def _per_replica_counts(t: Table, R: int, threshold: float):
    counts = np.zeros(R)
    for r, s in zip(t["replica"].astype(int), t["size"]):
        if 0 <= r < R and s >= threshold:
            counts[r] += 1
    return counts


def _n_replicas(outputs: Outputs, i: int, t: Table) -> int:
    rep = outputs.points[i].get("replicas")
    if rep is not None and len(rep):
        return len(rep)
    sw = outputs.get("sweep")
    if sw is not None and sw.has("replicas") and len(sw) > i:
        return int(sw["replicas"][i])
    return int(t["replica"].max()) + 1 if len(t) else 1


def plot_clusters_vs_key(outputs: Outputs, warnings: list[str] | None = None):
    """Major clusters (>= 10% of N) per replica and P(all in one cluster) against the swept key."""
    pts = _need_points(outputs, "clusters", warnings, "clusters vs key")
    if not pts:
        return []
    N = _total_size(outputs, "clusters vs key")
    if N is None:
        _warn(warnings, "clusters vs key: no system size in the configuration; figure skipped")
        return []
    x, xname = _sweep_x(outputs, [i for i, _ in pts], warnings)
    mean, se, p1, pe = [], [], [], []
    for i, t in pts:
        R = _n_replicas(outputs, i, t)
        major = _per_replica_counts(t, R, 0.1 * N)
        allc = _per_replica_counts(t, R, 0.0)
        mean.append(major.mean())
        se.append(major.std(ddof=1) / math.sqrt(R) if R > 1 else np.nan)
        p1.append(float((allc == 1).mean()))
        pe.append(R)
    note_undefined(warnings, "clusters vs key", se, "a single replica")
    fig, axes = new_fig(1, 2)
    ax = axes[0, 0]
    ax.errorbar(x, mean, yerr=se, marker="o", ms=4, capsize=3, color=color(0),
                label=f"major clusters (size ≥ 10% of N) {SE}")
    ax.set(xlabel=xname, ylabel="major clusters per replica")
    ax = axes[0, 1]
    pe, edge = binomial_yerr(p1, pe)
    ax.errorbar(x, p1, yerr=pe, marker="o", ms=4, capsize=3, color=color(1),
                label=f"P(all in one cluster) {_binomial_label(edge)}")
    ax.set(xlabel=xname, ylabel="P(one cluster)", ylim=(-0.02, 1.02))
    fig.legend(*_handles(axes), loc="outside upper center", ncols=2)
    return [("clusters_vs_key", finish(fig, outputs.provenance))]


def log_bin(sizes: np.ndarray):
    """Bins of width x2 starting at 1: (edges, counts, density, Poisson error of the density).
    Density = count / (number of integer sizes in the bin * total domains)."""
    top = max(1, int(sizes.max()))
    edges = [1]
    while edges[-1] <= top:
        edges.append(edges[-1] * 2)
    edges = np.array(edges, float)
    counts, _ = np.histogram(sizes, bins=edges - 0.5)
    width = np.diff(edges)
    total = sizes.size
    return edges, counts, counts / (width * total), np.sqrt(counts) / (width * total)


def plot_domains(t: Table, warnings: list[str] | None = None):
    """Domain-size distribution (log-binned, x2, pooled over replicas), largest-domain
    fraction per replica when N is known, and wrap counts."""
    if len(t) == 0:
        _warn(warnings, f"{t.path.name}: no domains; figure skipped")
        return []
    size = t["size"]
    cfg = t.provenance.config
    try:
        N = size_of(cfg, "N", str(t.path))[0]
    except InputError:
        N = None
        _warn(warnings, f"{t.path.name}: no system size in the configuration; "
                        "largest-domain fraction not drawn")
    npan = 2 + (N is not None)
    fig, axes = new_fig(1, npan)
    ax = axes[0, 0]
    edges, counts, dens, err = log_bin(size)
    centre = np.sqrt(edges[:-1] * edges[1:])  # geometric centre of the bin
    keep = counts > 0
    ax.errorbar(centre[keep], dens[keep], yerr=err[keep], fmt="o", ms=4, capsize=3,
                color=color(0), label=f"{size.size} domains, bins ×2, {POISSON}")
    log_axes(ax, [(centre[keep], dens[keep])], warnings, t.path.name, "domain sizes")
    ax.set(xlabel="domain size", ylabel="density per unit size")
    ax.legend(loc="upper right")
    k = 1
    if N is not None:
        ax = axes[0, k]
        k += 1
        reps = t["replica"].astype(int)
        ids = np.unique(reps)
        frac = np.array([size[reps == r].max() / N for r in ids])
        ax.hist(frac, bins=min(20, max(5, int(math.sqrt(frac.size)) + 1)), color=color(1))
        ax.set(xlabel="largest-domain fraction", ylabel="replicas")
        ax.set_title(f"{ids.size} replicas (replicas without domains rows not counted)",
                     fontsize=7)
    ax = axes[0, k]
    if t.has("wraps"):
        w = t["wraps"].astype(int)
        vals, c = np.unique(w, return_counts=True)
        ax.bar(vals, c, color=color(2))
        ax.set_xticks(vals)
        ax.set(xlabel="times the domain wraps around the torus", ylabel="domains")
    else:
        ax.set_visible(False)
    return [("domains", finish(fig, t.provenance))]


def plot_domains_vs_key(outputs: Outputs, warnings: list[str] | None = None):
    pts = _need_points(outputs, "domains", warnings, "domains vs key")
    if not pts:
        return []
    N = _total_size(outputs, "domains vs key")
    x, xname = _sweep_x(outputs, [i for i, _ in pts], warnings)
    npan = 1 + (N is not None)
    fig, axes = new_fig(1, npan)
    nd, nde, lg, lge = [], [], [], []
    for i, t in pts:
        R = _n_replicas(outputs, i, t)
        reps = t["replica"].astype(int)
        per = np.bincount(reps, minlength=R)[:R].astype(float)
        nd.append(per.mean())
        nde.append(per.std(ddof=1) / math.sqrt(R) if R > 1 else np.nan)
        if N is not None:
            big = np.array([t["size"][reps == r].max() / N if (reps == r).any() else np.nan
                            for r in range(R)])
            ok = np.isfinite(big)
            lg.append(big[ok].mean() if ok.any() else np.nan)
            lge.append(big[ok].std(ddof=1) / math.sqrt(ok.sum()) if ok.sum() > 1 else np.nan)
    note_undefined(warnings, "domains vs key (domains per replica)", nde, "a single replica")
    note_undefined(warnings, "domains vs key (largest-domain fraction)", lge if lge else None,
                   "fewer than 2 replicas with domains")
    ax = axes[0, 0]
    ax.errorbar(x, nd, yerr=nde, marker="o", ms=4, capsize=3, color=color(0),
                label=f"domains per replica {SE}")
    ax.set(xlabel=xname, ylabel="domains per replica")
    if N is not None:
        ax = axes[0, 1]
        ax.errorbar(x, lg, yerr=lge, marker="o", ms=4, capsize=3, color=color(1),
                    label=f"largest-domain fraction {SE}")
        ax.set(xlabel=xname, ylabel="largest-domain fraction")
    else:
        _warn(warnings, "domains vs key: no system size; largest-domain fraction not drawn")
    fig.legend(*_handles(axes), loc="outside upper center", ncols=2)
    return [("domains_vs_key", finish(fig, outputs.provenance))]


def plot_histogram(t: Table, warnings: list[str] | None = None):
    if len(t) == 0:
        _warn(warnings, f"{t.path.name}: no rows; figure skipped")
        return []
    lo, hi, cnt, dens = t["bin_low"], t["bin_high"], t["count"], t["density"]
    centre = 0.5 * (lo + hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(cnt > 0, dens / np.sqrt(cnt), np.nan)
    fig, axes = new_fig(1, 1)
    ax = axes[0, 0]
    ax.stairs(dens, np.r_[lo, hi[-1]], color=color(0), lw=1.3)
    note_undefined(warnings, f"{t.path.name} histogram", err, "empty bins")
    ax.errorbar(centre, dens, yerr=err, fmt="none", ecolor=color(1), capsize=2,
                label=f"{POISSON} on the count (ignores autocorrelation)")
    ax.set(xlabel=_hist_label(t), ylabel="density")
    ax.legend(loc="best")
    return [("histogram", finish(fig, t.provenance))]


def _hist_label(t: Table) -> str:
    return t.provenance.get("observable", "observable") or "observable"


def plot_samples(t: Table, warnings: list[str] | None = None):
    """Traces of each sampled observable, one line per chain; tau in the title when known."""
    from ..helpers import summarise_samples
    if len(t) == 0:
        _warn(warnings, f"{t.path.name}: no samples; figure skipped")
        return []
    obs = [n for n in t.names if n not in ("sample", "chain", "sweep") and numeric(t, n) is not None]
    if not obs or not t.has("chain"):
        _warn(warnings, f"{t.path.name}: no observable columns; figure skipped")
        return []
    chains = np.unique(t["chain"])
    shown = chains[:8]
    if chains.size > shown.size:
        _warn(warnings, f"{t.path.name}: {chains.size} chains, traces drawn for the first "
                        f"{shown.size} ({chains.size - shown.size} skipped)")
    title = None
    try:
        rec = summarise_samples(t, t.provenance.config)
        tau = rec.values.get("tau", (math.nan, math.nan))[0]
        if math.isfinite(tau):
            title = f"integrated autocorrelation time of {obs[-1]}: τ ≈ {tau:.3g} samples"
    except InputError:
        pass
    fig, axes = new_fig(len(obs), 1, size=(7, 2.4))
    x = t["sample"]
    for ax, n in zip(axes.ravel(), obs):
        for k, c in enumerate(shown):
            sel = t["chain"] == c
            ax.plot(x[sel], masked(t[n])[sel], lw=0.8, color=color(k), label=f"chain {int(c)}")
        ax.set(xlabel="sample", ylabel=n)
        if len(shown) > 1:
            ax.legend(loc="upper right", ncols=min(4, len(shown)))
    return [("samples", finish(fig, t.provenance, title))]
