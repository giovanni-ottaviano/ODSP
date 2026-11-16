"""The `odspy` command: plot, crossings, fit and collapse.

Exit status 0 when the command ran (a flagged-poor evaluation is reported, not an error),
2 on an input error (message on stderr as ``odspy: error: ...``), 3 on an internal error
(``odspy: internal error: ...``, a bug to report).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

from .collapse import collapse, search_collapse
from .crossings import find_crossings
from .errors import InputError
from .fits import fit
from .helpers import (FIRST_PASSAGE_NOTE, consensus_time_vs_size, stationary_caveats,
                      summarise_replicas, summarise_samples)
from .io import Outputs, find_outputs
from .plots import iter_figures, plot_collapse, plot_crossings, plot_fit, save
from .report import (CORRELATED_PAIRS, Report, collapse_lines, crossing_lines, fit_summary,
                     recovered, write)
from .sizes import families
from .trajectories import (UNTRUSTWORTHY_G, UNTRUSTWORTHY_RUN, fit_timeseries,
                           mark_untrustworthy)

COMMANDS = ("plot", "crossings", "fit", "collapse")
MODELS = ("power", "power+c", "log", "line")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odspy", description="Figures and evaluations of ODSP outputs (odsp --csv PREFIX).",
        epilog="If the first argument is not a command, 'plot' is implied.")
    sub = p.add_subparsers(dest="command", metavar="{plot,crossings,fit,collapse}")

    def common(q):
        q.add_argument("prefix", nargs="+", metavar="PREFIX",
                       help="output prefix given to odsp --csv (or one of its files)")
        q.add_argument("--out", metavar="DIR", default=None,
                       help="where figures and the report go (default: the first prefix's folder)")

    q = sub.add_parser("plot", help="every standard figure that applies")
    common(q)
    q.add_argument("--per-point", action="store_true", help="also draw each sweep point's own files")

    q = sub.add_parser("crossings", help="crossings of a column between system sizes")
    common(q)
    q.add_argument("--column", default="binder")
    q.add_argument("--size", choices=("auto", "L", "N"), default="auto")
    q.add_argument("--boot", type=int, default=2000, help="bootstrap draws for the error")

    q = sub.add_parser("fit", help="fit a model to a column")
    common(q)
    q.add_argument("--x", default=None, help="x column (a sweep key, or 'time' for a time series); "
                   "not needed for --y mean_consensus_time (x = system size)")
    q.add_argument("--y", required=True, help="y column, or the observable of a time series")
    q.add_argument("--range", dest="xrange", default=None, metavar="lo:hi", help="fit only x in [lo, hi]")
    q.add_argument("--model", choices=MODELS, default="power")
    q.add_argument("--unweighted", action="store_true")
    q.add_argument("--censored", choices=("exclude", "median"), default="exclude",
                   help="mean_consensus_time vs size: censored points excluded, or the median")
    q.add_argument("--size", choices=("auto", "L", "N"), default="auto")
    q.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (time series, median)")
    q.add_argument("--curve", type=float, default=None, metavar="VALUE",
                   help="correlation file with several curves: the time (or sweep) of the one to fit "
                        "(default: the last)")

    q = sub.add_parser("collapse", help="finite-size-scaling data collapse of a column")
    common(q)
    q.add_argument("--column", default="binder")
    q.add_argument("--xc", type=float, required=True)
    q.add_argument("--nu", type=float, required=True)
    q.add_argument("--a", type=float, default=0.0)
    q.add_argument("--search", action="store_true", help="minimise S from the given values")
    q.add_argument("--fix-a", action="store_true", help="with --search, keep a fixed")
    q.add_argument("--starts", type=int, default=None, metavar="N",
                   help="with --search, starting points from a grid around --xc/--nu "
                        "(1 to 15, default 15; 1 = the given values alone)")
    q.add_argument("--size", choices=("auto", "L", "N"), default="auto")
    q.add_argument("--boot", type=int, default=200, help="bootstrap minimisations for the errors")
    return p


def _range(text: str | None):
    if text is None:
        return None
    try:
        lo, hi = (float(v) for v in text.split(":"))
    except ValueError:
        raise InputError(f"--range must be lo:hi (two numbers), got '{text}'") from None
    return lo, hi


def _stem(prefix: str) -> str:
    return Path(find_outputs(prefix).prefix).name


def _safe(name: str) -> str:
    """A column or observable name made safe for a file name."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", name) or "_"


def _outdir(args) -> Path:
    if args.out is not None:
        return Path(args.out)
    return Path(find_outputs(args.prefix[0]).prefix).parent


def _save(rep: Report, fig, out: Path, stem: str) -> None:
    rep.figures.extend(save(fig, out, stem))


# ------------------------------------------------------------------------------- plot

def _summaries(rep: Report, o: Outputs) -> None:
    """The driver's own numbers for a single ensemble or stationary run, and the notes that
    the files cannot support."""
    ts = o.get("timeseries_ensemble")
    if ts is not None:
        if o.get("trajectories") is None:
            rep.warn(f"{ts.path.name}: no {Path(o.prefix).name}_trajectories.csv: fits to this time "
                     "series cannot give trustworthy errors (rerun with --save-trajectories true)")
        else:
            rep.add("Trajectories: every replica's time series is present "
                    f"({Path(o.prefix).name}_trajectories.csv); fits use a bootstrap over replicas.")
    if o.points:
        return
    rt = o.get("replicas")
    if rt is not None:
        r = summarise_replicas(rt)
        rep.add()
        rep.lines.extend(recovered(r, "Ensemble summary"))
    st = o.get("samples")
    if st is not None:
        r = summarise_samples(st, st.provenance.config or o.provenance.config)
        rep.add()
        rep.lines.extend(recovered(r, "Stationary summary"))
    sw = o.get("sweep")
    if sw is not None and sw.provenance.config.get("task") == "stationary":
        rep.warn(*stationary_caveats(sw))


def cmd_plot(args) -> tuple[Report, str]:
    rep, out = Report(), _outdir(args)
    stem = _stem(args.prefix[0])
    for prefix in args.prefix:
        o = find_outputs(prefix)
        name = Path(o.prefix).name
        files = [t.path for t in o.tables.values()]
        files += [t.path for pt in o.points.values() for t in pt.tables.values()]
        rep.files("Files read", files)
        warns: list[str] = []
        # each figure is saved and closed as soon as it is drawn (never more than a few open)
        for fig_name, fig in iter_figures(o, per_point=args.per_point, warnings=warns):
            try:
                _save(rep, fig, out, f"{name}_{fig_name}")
            except OSError:
                raise
            except Exception as e:      # one figure that cannot be drawn must not stop the rest
                warns.append(f"figure {name}_{fig_name} not saved: {type(e).__name__}: {e}")
        rep.warn(*warns)
        _summaries(rep, o)
    return rep, stem


# ---------------------------------------------------------------------------- crossings

def cmd_crossings(args) -> tuple[Report, str]:
    rep, out = Report(), _outdir(args)
    curves = families(args.prefix, args.column, size=args.size)
    rep.files("Files read", [c.source for c in dict((c.source, c) for c in curves).values()])
    cs = find_crossings(curves, n_boot=args.boot)
    rep.add(f"Crossings of {args.column} against {curves[0].xkey} between consecutive sizes:")
    rep.lines.extend(crossing_lines(cs, curves[0].label))
    for c in cs:
        rep.warn(*c.warnings)
    if len({(c.size_a, c.size_b) for c in cs if c.found and math.isfinite(c.err)}) > 1:
        rep.add(f"  note: {CORRELATED_PAIRS}")
    prov = find_outputs(args.prefix[0]).provenance
    name = f"{_stem(args.prefix[0])}_crossings_{_safe(args.column)}"
    _save(rep, plot_crossings(curves, cs, prov), out, name)
    return rep, name


# --------------------------------------------------------------------------------- fit

def _fit_one(rep: Report, r, out: Path, stem: str, xlabel: str, ylabel: str, prov) -> None:
    text, warns = fit_summary(r)
    rep.add(text)
    rep.warn(*warns)
    _save(rep, plot_fit(r, xlabel, ylabel, prov), out, stem)


def _need(table, name: str) -> None:
    if not table.has(name):
        raise InputError(f"{table.path}: no column '{name}' (columns: {', '.join(table.names)})")


CURVE_ONLY = "--curve is only for correlation files with several curves (one per time or sweep)"


def _one_curve(t, curve, rep):
    """Rows of the G(r) curve to fit in a correlation table holding one curve per time (or
    sweep): the last by default, or the one at `curve`. Pooling the curves would fit points
    from different times as if they were one curve."""
    group = {"correlation_run": "sweep", "correlation_ensemble": "time"}.get(t.kind)
    values = sorted(set(np.asarray(t[group], dtype=float).tolist())) if group and t.has(group) else []
    if len(values) < 2:
        if curve is not None:
            raise InputError(f"{CURVE_ONLY}; {t.path.name} holds one")
        return np.ones(len(t), dtype=bool)
    if curve is None:
        chosen = values[-1]
    else:
        # a typed value equal to a curve's up to rounding, never within reach of two curves
        tol = min(1e-9 * max(1.0, abs(curve)), 0.25 * float(np.min(np.diff(values))))
        match = [v for v in values if abs(v - curve) <= tol]
        if not match:
            shown = ", ".join(f"{v:.10g}" for v in values[:12]) + (", ..." if len(values) > 12 else "")
            raise InputError(f"{t.path}: no curve with {group} = {curve:.10g} (has {shown})")
        chosen = match[0]
    rep.add(f"{t.path.name} holds {len(values)} curves (one per {group}); fitted the curve at "
            f"{group} = {chosen:.10g}" + (" (the last; --curve picks another)" if curve is None else ""))
    return np.asarray(t[group], dtype=float) == chosen   # exact: chosen comes from the column


def cmd_fit(args) -> tuple[Report, str]:
    rep, out = Report(), _outdir(args)
    xrange = _range(args.xrange)
    stem = _stem(args.prefix[0])
    first = find_outputs(args.prefix[0])
    if args.curve is not None and (args.y == "mean_consensus_time" or args.x == "time"
                                   or first.provenance.config.get("sweep2")):
        raise InputError(CURVE_ONLY)
    if args.y == "mean_consensus_time" and args.x in (None, "size"):
        rep.files("Files read", [t.path for p in args.prefix
                                 for t in find_outputs(p).tables.values()])
        r = consensus_time_vs_size(args.prefix, size=args.size, censored=args.censored,
                                   model=args.model, n_boot=args.boot)
        if xrange is not None:
            rep.warn("--range is ignored for mean_consensus_time against size "
                     "(all non-censored sizes are fitted)")
        label = {"auto": "system size"}.get(args.size, args.size)
        name = f"{stem}_fit_{_safe(args.y)}"
        ylabel = args.y
        if FIRST_PASSAGE_NOTE in r.notes:
            ylabel = "mean first-passage time (first time consensus is reached)"
        _fit_one(rep, r, out, name, label, ylabel, first.provenance)
        return rep, name
    if args.x is None:
        raise InputError("fit needs --x (the x column), except for --y mean_consensus_time")
    if len(args.prefix) > 1:
        raise InputError("fit takes one PREFIX (several only for --y mean_consensus_time)")
    if args.x == "time":
        ts = first.get("timeseries_ensemble")
        if ts is None:
            raise InputError(f"{first.prefix}: no ensemble time series "
                             f"({first.prefix}_timeseries.csv of an ensemble run); --x time needs one")
        obs = args.y[:-len("_mean")] if args.y.endswith("_mean") else args.y
        _need(ts, f"{obs}_mean")
        traj = first.get("trajectories")
        rep.files("Files read", [t.path for t in (ts, traj) if t is not None])
        r = fit_timeseries(ts, obs, model=args.model, xrange=xrange, trajectories=traj,
                           n_boot=args.boot)
        name = f"{stem}_fit_{_safe(obs)}"
        _fit_one(rep, r, out, name, "time", obs, first.provenance)
        return rep, name
    if first.provenance.config.get("sweep2"):
        curves = families([args.prefix[0]], args.y, xkey=args.x, size=args.size)
        rep.files("Files read", [curves[0].source])
        for c in curves:
            rep.add(f"size {c.label}={c.size:g}:")
            r = fit(c.x, c.y, c.yerr, model=args.model, xrange=xrange, unweighted=args.unweighted)
            _fit_one(rep, r, out, f"{stem}_fit_{_safe(args.y)}_{c.label}{c.size:g}", args.x,
                     args.y, first.provenance)
        return rep, f"{stem}_fit_{_safe(args.y)}"
    tables = list(first.tables.values())
    hit = [t for t in tables if t.has(args.x) and t.has(args.y)]
    if not hit:
        raise InputError("no table has both columns '%s' and '%s': " % (args.x, args.y)
                         + "; ".join(f"{t.path} ({', '.join(t.names)})" for t in tables))
    t = hit[0]
    rep.files("Files read", [t.path])
    if args.curve is not None and not t.kind.startswith("correlation"):
        raise InputError(f"{CURVE_ONLY}; {t.path.name} is not one")
    rows = _one_curve(t, args.curve, rep) if t.kind.startswith("correlation") else np.ones(len(t), bool)
    x = np.asarray(t[args.x])[rows]
    y = np.asarray(t[args.y])[rows]
    if x.dtype == object or y.dtype == object:
        raise InputError(f"{t.path}: columns '{args.x}' and '{args.y}' must be numeric")
    # the error of <o>_mean is <o>_err (as G_mean / G_err); otherwise <y>_err
    base = args.y[:-len("_mean")] if args.y.endswith("_mean") else None
    ename = base + "_err" if base and t.has(base + "_err") else args.y + "_err"
    yerr = t[ename].astype(float)[rows] if t.has(ename) and not args.unweighted else None
    r = fit(x.astype(float), y.astype(float), yerr, model=args.model, xrange=xrange,
            unweighted=args.unweighted)
    # points of one table that share their configurations: chi2 and p would mean nothing
    label = UNTRUSTWORTHY_RUN if t.kind == "timeseries_run" else \
        UNTRUSTWORTHY_G if t.kind.startswith("correlation") else None
    if label is not None:
        mark_untrustworthy(r, label)
    name = f"{stem}_fit_{_safe(args.y)}"
    _fit_one(rep, r, out, name, args.x, args.y, first.provenance)
    return rep, name


# ----------------------------------------------------------------------------- collapse

def cmd_collapse(args) -> tuple[Report, str]:
    rep, out = Report(), _outdir(args)
    curves = families(args.prefix, args.column, size=args.size)
    rep.files("Files read", sorted({c.source for c in curves}))
    if args.search:
        r = search_collapse(curves, args.xc, args.nu, args.a, fit_a=not args.fix_a, n_boot=args.boot,
                            starts=15 if args.starts is None else args.starts)
    else:
        if args.fix_a or args.boot != 200 or args.starts is not None:
            rep.warn("--fix-a, --boot and --starts only apply with --search; ignored")
        r = collapse(curves, args.xc, args.nu, args.a)
    rep.lines.extend(collapse_lines(r))
    rep.warn(*r.warnings)
    stem = _stem(args.prefix[0])
    prov = find_outputs(args.prefix[0]).provenance
    xlabel = f"({curves[0].xkey} - xc) {curves[0].label}^(1/nu)"
    name = f"{stem}_collapse_{_safe(args.column)}"
    _save(rep, plot_collapse(r, xlabel, f"{args.column} {curves[0].label}^(-a)", prov), out, name)
    return rep, name


HANDLERS = {"plot": cmd_plot, "crossings": cmd_crossings, "fit": cmd_fit, "collapse": cmd_collapse}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "plot")
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.code and argv and argv[0] in COMMANDS and any(Path(".").glob(f"{argv[0]}_*.csv")):
            print(f"odspy: note: '{argv[0]}' is read as the {argv[0]} command; for the outputs "
                  f"with prefix {argv[0]} write ./{argv[0]}", file=sys.stderr)
        return e.code if isinstance(e.code, int) else 2
    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2
    try:
        # each command writes its own report: <stem>_report.txt (plot),
        # <stem>_fit_<y>_report.txt, <stem>_crossings_<column>_report.txt,
        # <stem>_collapse_<column>_report.txt
        rep, name = HANDLERS[args.command](args)
        out = _outdir(args)
        path = out / f"{name}_report.txt"
        rep.add()
        rep.files("Figures written", rep.figures)
        rep.add(f"Report saved: {path}")
        text = rep.render()
        write(path, text)
    except (InputError, OSError) as e:
        print(f"odspy: error: {e}", file=sys.stderr)
        return 2
    except Exception as e:      # last resort: never a traceback
        print(f"odspy: internal error: {type(e).__name__}: {e} (please report)", file=sys.stderr)
        return 3
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
