"""Standard figures for ODSP outputs: ``figures_for(outputs)`` and ``save``."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..errors import InputError
from ..io import Outputs
from .correlation import plot_correlation
from .distributions import (plot_clusters, plot_clusters_vs_key, plot_domains,
                            plot_domains_vs_key, plot_fixation, plot_fixation_vs_key,
                            plot_histogram, plot_samples)
from .evaluation import plot_collapse, plot_crossings, plot_fit
from .replicas import plot_replicas
from .states import plot_states
from .style import caption, save
from .sweeps import plot_sweep, plot_sweep2
from .timeseries import (plot_timeseries_ensemble, plot_timeseries_run,
                         plot_trajectories)

__all__ = ["caption", "figures_for", "iter_figures", "plot_clusters", "plot_clusters_vs_key", "plot_collapse", "plot_correlation",
           "plot_crossings", "plot_domains", "plot_domains_vs_key", "plot_fixation", "plot_fixation_vs_key",
           "plot_fit", "plot_histogram", "plot_replicas", "plot_samples", "plot_states", "plot_sweep",
           "plot_sweep2", "plot_timeseries_ensemble", "plot_timeseries_run", "plot_trajectories", "save"]


def _why(e: Exception) -> str:
    """The reason a figure was skipped: input problems as they are, anything else with its type."""
    return str(e) if isinstance(e, (InputError, KeyError)) else f"{type(e).__name__}: {e}"


def _close_new(before: set[int]) -> None:
    for n in set(plt.get_fignums()) - before:     # half-drawn figures
        plt.close(n)


def _table_figures(out: Outputs, warnings: list[str]):
    """Yield (name, Figure) for each table of `out`, one plotting function at a time."""
    simple = {
        "timeseries_run": plot_timeseries_run, "timeseries_ensemble": plot_timeseries_ensemble, "trajectories": plot_trajectories,
        "replicas": plot_replicas, "sweep": plot_sweep, "fixation": plot_fixation,
        "clusters": plot_clusters, "domains": plot_domains, "histogram": plot_histogram,
        "samples": plot_samples, "correlation_run": plot_correlation,
        "correlation_ensemble": plot_correlation, "correlation_stationary": plot_correlation,
    }
    for kind, t in out.tables.items():
        before = set(plt.get_fignums())
        try:
            if kind in simple:
                figs = simple[kind](t, warnings=warnings)
            elif kind == "states":
                figs = plot_states(t, out.provenance, warnings=warnings)
            else:
                warnings.append(f"no standard figure for {kind} ({t.path})")
                figs = []
        except Exception as e:      # one table's figures must not cost the others or the report
            _close_new(before)
            warnings.append(f"{t.path}: figures for {kind} skipped: {_why(e)}")
            figs = []
        yield from figs


def iter_figures(outputs: Outputs, per_point: bool = False, warnings: list[str] | None = None):
    """Yield (name, Figure) for every standard figure that applies, one table at a time, so a
    caller that saves and closes each figure as it comes never holds more than a few open
    (`odspy plot` does). Warnings are appended to `warnings` as the figures are drawn.

    Per-point files get their own figures only with per_point=True; otherwise the aggregate
    figures against the swept key are drawn and a warning says what was skipped."""
    if warnings is None:
        warnings = []
    count = 0
    for item in _table_figures(outputs, warnings):
        count += 1
        yield item
    if outputs.points:
        if per_point:
            for i, o in outputs.points.items():
                for n, f in _table_figures(o, warnings):
                    count += 1
                    yield f"point{i}_{n}", f
        else:
            n_files = sum(len(o.tables) for o in outputs.points.values())
            warnings.append(f"{n_files} per-point files ({len(outputs.points)} points) were read; "
                            "their own figures were skipped (per_point=True draws them)")
        for kind, fn in (("fixation", plot_fixation_vs_key), ("clusters", plot_clusters_vs_key),
                         ("domains", plot_domains_vs_key)):
            if any(o.get(kind) is not None for o in outputs.points.values()):
                before = set(plt.get_fignums())
                try:
                    figs = fn(outputs, warnings=warnings)
                except Exception as e:
                    _close_new(before)
                    warnings.append(f"{kind} vs key skipped: {_why(e)}")
                    figs = []
                for item in figs:
                    count += 1
                    yield item
    if not count:
        warnings.append("nothing to draw for these outputs")


def figures_for(outputs: Outputs, per_point: bool = False):
    """All standard figures that apply -> (list of (name, Figure), list of warnings).

    Every figure stays open until the caller closes it (``save`` does): with many sweep
    points and per_point=True, prefer ``iter_figures`` and save each figure as it comes.
    Per-point files get their own figures only with per_point=True; otherwise the aggregate
    figures against the swept key are drawn and a warning says what was skipped."""
    warnings: list[str] = []
    figs = list(iter_figures(outputs, per_point, warnings))
    return figs, warnings
