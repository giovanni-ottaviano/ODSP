"""Evaluation figures: a fit, size-curve crossings, a data collapse.

Every figure puts its quality up front (red headline lines above the panels, a text column
beside them), not in a corner: POOR FIT, POOR COLLAPSE, "no crossing in range", the label for
untrustworthy errors. Points outside a fit range are faded, never removed; an undefined error
draws no bar."""

from __future__ import annotations

import math
import textwrap

import numpy as np
from matplotlib.figure import Figure

from ..collapse import CollapseResult
from ..crossings import Crossing
from ..fits import UNWEIGHTED, FitResult
from ..io import Provenance
from ..sizes import SizeCurve
from ..trajectories import CORRELATED, UNTRUSTWORTHY, UNTRUSTWORTHY_PREFIX
from .style import caption, color, masked, new_fig, note_undefined, positive

RED = "#C00000"
FADE = 0.3
LINE_IN = 0.26            # inches reserved per headline line
SIDE = 0.36               # fraction of the width given to the text column


def _fmt(v: float) -> str:
    return f"{v:.5g}" if math.isfinite(v) else "not estimated"


def _clean_err(err) -> np.ndarray | None:
    """Error bars with undefined (NaN, inf, non-positive) entries as NaN: no bar is drawn."""
    if err is None:
        return None
    e = np.asarray(err, float)
    return np.where(np.isfinite(e) & (e > 0), e, np.nan)


def _wrap(text: str, width: int = 46) -> str:
    return "\n".join(textwrap.wrap(text, width, subsequent_indent="  ")) or text


def _layout(fig: Figure, prov: Provenance | None, headlines, side: bool):
    """Red headline lines at the top (one text object each, never wrapped), caption at the
    bottom, and the right part of the figure left free for the text column."""
    caption(fig, prov)
    engine = fig.get_layout_engine()
    rect = engine.get()["rect"] if engine is not None else (0, 0, 1, 1)
    h = fig.get_figheight()
    top = LINE_IN * len(headlines) / h
    for i, (text, col, size) in enumerate(headlines):
        fig.text(0.5, 1 - (0.004 + LINE_IN * i / h), text, ha="center", va="top", color=col,
                 fontsize=size, fontweight="bold")
    if engine is not None:
        x0, y0, w, hh = rect
        engine.set(rect=(x0, y0, w * (1 - SIDE) if side else w, hh - top))


def _column(fig: Figure, lines: list[tuple[str, str]], top_in: float) -> None:
    """The text column on the right: (text, colour) paragraphs, wrapped."""
    y = 1 - (top_in + 0.1) / fig.get_figheight()
    for text, col in lines:
        t = fig.text(1 - SIDE + 0.01, y, _wrap(text), ha="left", va="top", fontsize=7.5, color=col,
                     family="monospace" if text.startswith("  ") else None)
        n = t.get_text().count("\n") + 1
        y -= (n * 0.135 + 0.07) / fig.get_figheight()


# --- fits ------------------------------------------------------------------------------------

def _fit_lines(r: FitResult, extra_warnings: list[str]):
    lines = [(f"model: {r.model}   method: {r.method}", "black"),
             (f"{int(r.used.sum())} of {r.x.size} points in the fit range "
              f"[{r.xrange[0]:.6g}, {r.xrange[1]:.6g}]; the others are faded", "black")]
    for k, v in r.params.items():
        e = r.errors.get(k, math.nan)
        lines.append((f"  {k} = {v:.6g} ± {_fmt(e)}", "black"))
    if r.chi2 is not None:
        ratio = r.chi2 / r.dof if r.dof > 0 else math.nan
        name = "T²/dof (Hotelling, not centred on dof)" if r.method != "weighted least squares" \
            else "χ²/dof"
        p = "not assessed" if r.p_value is None else f"{r.p_value:.3g}"
        exp = "" if r.chi2_expected is None else f",  E[T²] = {r.chi2_expected:.4g}"
        lines.append((f"{name} = {r.chi2:.4g}/{r.dof} = {ratio:.3g}{exp},  p = {p}", "black"))
    elif r.method == UNWEIGHTED:
        lines.append(("goodness of fit not assessed (unweighted fit); errors from the scatter "
                      "of the residuals", "black"))
    if r.scaled and r.chi2 is not None and r.chi2_expected is not None:
        lines.append(("errors scaled by sqrt(T²/E[T²]) = "
                      f"{math.sqrt(r.chi2 / r.chi2_expected):.3g}", "black"))
    elif r.scaled and r.chi2 is not None and r.dof > 0 and r.chi2 > 0:
        lines.append((f"errors scaled by sqrt(chi2/dof) = {math.sqrt(r.chi2 / r.dof):.3g}", "black"))
    lines += [(s, "black") for s in r.notes]
    lines += [(f"warning: {w}", "black") for w in list(r.warnings) + extra_warnings
              if not w.startswith(UNTRUSTWORTHY_PREFIX)]
    return lines


def plot_fit(result: FitResult, xlabel: str, ylabel: str, prov: Provenance | None = None) -> Figure:
    """Data with the fitted curve (top) and the residuals (bottom)."""
    r = result
    fig, _ = new_fig(1, 1, size=(9.0, 5.2))
    fig.clear()
    ax, rax = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for a in (ax, rax):
        a.grid(True, alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    own: list[str] = []
    logx = r.model != "line" and bool(np.any(np.isfinite(r.x) & (r.x > 0)))
    logy = r.model in ("power", "power+c") and bool(np.any(np.isfinite(r.y) & (r.y > 0)))
    x = np.asarray(r.x, float)
    y = np.asarray(r.y, float)
    err = _clean_err(r.yerr)
    shown = np.isfinite(x) & np.isfinite(y)
    if logx:
        shown &= x > 0
    if logy:
        shown &= y > 0
    hidden = int((np.isfinite(x) & np.isfinite(y) & ~shown).sum())
    if hidden:
        own.append(f"{hidden} point(s) with values <= 0 not shown on the log axes")
    res = r.residuals()
    for sel, alpha, label in ((r.used & shown, 1.0, "fitted points"),
                              (~r.used & shown, FADE, "outside the fit range")):
        if not sel.any():
            continue
        kw = dict(fmt="o", ms=4, color=color(0), alpha=alpha, elinewidth=1, capsize=0, label=label)
        ax.errorbar(x[sel], y[sel], yerr=None if err is None else err[sel], **kw)
        rax.errorbar(x[sel], res[sel], yerr=None if r.yerr is None else
                     np.where(np.isfinite(err[sel]), 1.0, np.nan), **kw)
    note_undefined(own, "fit points", None if err is None else err[shown], "no error bar")
    lo, hi = r.xrange
    xs = x[shown]
    if xs.size:
        a_lo, a_hi = max(lo, xs.min()), min(hi, xs.max())
        if np.isfinite(a_lo) and np.isfinite(a_hi) and a_lo < a_hi:
            for a in (ax, rax):
                a.axvspan(a_lo, a_hi, color=color(5), alpha=0.12, lw=0)
        grid = np.geomspace(xs.min(), xs.max(), 200) if logx and xs.min() > 0 else \
            np.linspace(xs.min(), xs.max(), 200)
        yf = r.predict(grid)
        ok = np.isfinite(yf) & ((yf > 0) if logy else True)
        ax.plot(grid[ok], np.where(ok, yf, np.nan)[ok], color=color(1), lw=1.4, label="fit")
    rax.axhline(0, color="#555555", lw=0.8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    rax.set_xlabel(xlabel)
    rax.set_ylabel("(data − fit)/σ" if r.yerr is not None else "data − fit")
    ax.legend(loc="best")
    heads = []
    labels = [w for w in r.warnings if w.startswith(UNTRUSTWORTHY_PREFIX)]
    if r.method == CORRELATED and not labels:
        labels = [UNTRUSTWORTHY]
    untrust = bool(labels)
    for w in labels:
        heads.append((w, RED, 8.5))
    if r.poor:
        heads.append(("POOR FIT", RED, 13))
    if r.chi2 is not None and r.dof > 0:
        name = "χ²/dof" if r.method == "weighted least squares" else "T²/dof"
        lim = f" < {r.p_threshold:g}" if r.poor else ""
        heads.append((f"{name} = {r.chi2 / r.dof:.3g}" +
                      ("" if r.p_value is None else f",  p = {r.p_value:.3g}{lim}"),
                      RED if r.poor else "black", 9))
    elif not untrust:
        heads.append(("goodness of fit not assessed", "black", 9))
    _layout(fig, prov, heads, side=True)
    _column(fig, _fit_lines(r, own), LINE_IN * len(heads))
    return fig


# --- crossings -------------------------------------------------------------------------------

def plot_crossings(curves: list[SizeCurve], crossings: list[Crossing],
                   prov: Provenance | None = None) -> Figure:
    """Curves with the crossing points (error bars in x), and the crossings against 1/size."""
    fig, axes = new_fig(1, 2, size=(5.0, 4.2))
    ax, dax = axes[0]
    label = curves[0].label if curves else "L"
    for i, c in enumerate(curves):
        ax.errorbar(c.x, masked(c.y), yerr=_clean_err(c.yerr), fmt="-o", ms=2.5, lw=1,
                    elinewidth=0.8, color=color(i), label=f"{label} = {c.size:g}")
    found = [c for c in crossings if c.found]
    for c in found:
        xe = c.err if math.isfinite(c.err) and c.err > 0 else None
        ax.errorbar([c.x], [c.y], xerr=None if xe is None else [xe], fmt="D", ms=6, color="black",
                    mfc="white", elinewidth=1.4, capsize=3, zorder=5)
    ax.set_xlabel(curves[0].xkey if curves else "x")
    ax.set_ylabel(curves[0].column if curves else "y")
    if curves:
        ax.legend(loc="best")
    # drift: where the pairs cross, against 1/(geometric mean size)
    if found:
        s = np.array([1 / math.sqrt(c.size_a * c.size_b) for c in found])
        xs = np.array([c.x for c in found])
        es = np.array([c.err for c in found])
        dax.errorbar(s, xs, yerr=_clean_err(es), fmt="D", ms=6, color="black", mfc="white",
                     elinewidth=1.4, capsize=3)
    dax.set_xlabel(f"1/√({label}ₐ {label}ᵦ)  (pair of sizes)")
    dax.set_ylabel("crossing position")
    dax.set_title("drift of the crossing with size", fontsize=8)
    dax.set_xlim(left=0)
    heads = []
    missing = [c for c in crossings if not c.found]
    if missing:
        pairs = ", ".join(f"{label} = {c.size_a:g}/{c.size_b:g}" for c in missing)
        heads.append((f"no crossing in range: {pairs}", RED, 10))
    if not crossings:
        heads.append(("no crossing in range (fewer than two sizes)", RED, 10))
    noerr = [c for c in found if not (math.isfinite(c.err) and c.err > 0)]
    if noerr:
        heads.append(("crossing error undefined for " +
                      ", ".join(f"{label} = {c.size_a:g}/{c.size_b:g}" for c in noerr) +
                      ": no error bar drawn", RED, 8.5))
    for w in dict.fromkeys(w for c in crossings for w in c.warnings):
        heads.append((w, RED, 8))
    methods = list(dict.fromkeys(c.method for c in crossings))
    if methods:
        heads.append(("errors: " + "; ".join(methods), "black", 8))
    _layout(fig, prov, heads, side=False)
    return fig


# --- collapse --------------------------------------------------------------------------------

def plot_collapse(result: CollapseResult, xlabel: str, ylabel: str,
                  prov: Provenance | None = None) -> Figure:
    """The scaled curves on one plot, S and the parameters beside it."""
    r = result
    fig, axes = new_fig(1, 1, size=(9.0, 4.6))
    ax = axes[0, 0]
    for i, (size, X, Y, E) in enumerate(r.scaled):
        ax.errorbar(X, masked(Y), yerr=_clean_err(E), fmt="o", ms=3, elinewidth=0.8,
                    color=color(i), label=f"size {size:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if r.scaled:
        ax.legend(loc="best")
    s = "undefined" if not math.isfinite(r.quality) else f"{r.quality:.3g}"
    heads = []
    if r.poor:
        heads.append(("POOR COLLAPSE", RED, 12))
    heads.append((f"S = {s} from {r.points_used} of {r.points_total} points", "black", 9.5))
    lines = [(r.method, "black")]
    withheld = [k for k in ("xc", "nu", "a") if k in r.errors and not math.isfinite(r.errors[k])]
    for k, v in (("xc", r.xc), ("nu", r.nu), ("a", r.a)):
        if k in r.fixed:
            lines.append((f"  {k} = {v:.6g} (fixed)", "black"))
        elif r.searched and k in r.errors and math.isfinite(r.errors[k]):
            lines.append((f"  {k} = {v:.6g} ± {r.errors[k]:.2g}", "black"))
        elif r.searched:
            lines.append((f"  {k} = {v:.6g} (error withheld)", RED))
        else:
            lines.append((f"  {k} = {v:.6g} (given)", "black"))
    if r.searched and (withheld or not r.errors):
        reason = r.method.split("errors withheld:", 1)[1].split(";")[0].strip() \
            if "errors withheld:" in r.method \
            else "; ".join(w for w in r.warnings if "error" in w.lower()) or "see warnings"
        lines.append((f"errors withheld: {reason}", RED))
    lines += [(f"warning: {w}", "black") for w in r.warnings]
    _layout(fig, prov, heads, side=True)
    _column(fig, lines, LINE_IN * len(heads))
    return fig
