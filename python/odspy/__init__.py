"""odspy: analysis tools for ODSP simulation outputs."""

from .collapse import CollapseResult, collapse, search_collapse
from .crossings import Crossing, crossings_plain, find_crossings
from .errors import InputError
from .fits import FitResult, fit, plain_loglog_slope
from .helpers import (Recovered, coarsening_fit, consensus_time_vs_size, error_bar_uncertainty,
                      is_first_passage, log_growth_vs_size, peak_vs_size, slope_at_crossing_vs_size,
                      stationary_caveats, summarise_replicas, summarise_samples)
from .io import find_outputs, read_table
from .merge import merge
from .plots import figures_for, iter_figures, plot_collapse, plot_crossings, plot_fit, save
from .sizes import SizeCurve, families, size_of
from .trajectories import fit_timeseries

__all__ = ["CollapseResult", "Crossing", "FitResult", "InputError", "Recovered", "SizeCurve",
           "coarsening_fit", "collapse", "consensus_time_vs_size", "crossings_plain",
           "error_bar_uncertainty", "families", "find_crossings", "figures_for", "find_outputs", "fit",
           "is_first_passage", "iter_figures", "merge",
           "fit_timeseries", "log_growth_vs_size", "peak_vs_size", "plain_loglog_slope", "plot_collapse",
           "plot_crossings", "plot_fit",
           "read_table", "save", "search_collapse", "size_of", "slope_at_crossing_vs_size",
           "stationary_caveats", "summarise_replicas", "summarise_samples"]
