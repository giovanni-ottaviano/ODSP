"""The text report of an `odspy` command: what was read, what was written, the numbers,
and every warning under one heading."""

from __future__ import annotations

import math
from pathlib import Path

from .collapse import CollapseResult
from .crossings import Crossing
from .fits import FitResult
from .helpers import Recovered


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings: list[str] = []
        self.figures: list[Path] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def warn(self, *texts: str) -> None:
        for t in texts:
            if t not in self.warnings:
                self.warnings.append(t)

    def files(self, heading: str, paths) -> None:
        paths = [str(p) for p in paths]
        if paths:
            self.add(f"{heading}:")
            self.lines.extend(f"  {p}" for p in paths)

    def render(self) -> str:
        out = list(self.lines)
        if self.warnings:
            out += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(out) + "\n"


def _fmt(v: float, e: float) -> str:
    if not math.isfinite(v):
        return "undefined"
    if math.isfinite(e):
        return f"{v:.6g} ± {e:.2g}"
    return f"{v:.6g}"


def recovered(r: Recovered, title: str) -> list[str]:
    lines = [f"{title} ({r.method}):"]
    lines += [f"  {k} = {_fmt(*ve)}" for k, ve in r.values.items()]
    lines += [f"  note: {n}" for n in r.notes]
    if r.missing:
        lines.append("  not recoverable from the files: " + "; ".join(r.missing))
    return lines


def fit_summary(r: FitResult) -> tuple[str, list[str]]:
    """The fit's summary without its inline warnings, and those warnings (they are listed
    under Warnings: instead)."""
    keep, warns = [], []
    for ln in r.summary().splitlines():
        if ln.startswith("  warning: "):
            warns.append(ln[len("  warning: "):])
        else:
            keep.append(ln)
    return "\n".join(keep), warns


def crossing_lines(cs: list[Crossing], label: str) -> list[str]:
    """One line per size pair; a pair whose curves cross several times gets them on
    indented lines under a header that says they are not separate measurements."""
    out = []
    pairs: dict[tuple[float, float], list[Crossing]] = {}
    for c in cs:
        pairs.setdefault((c.size_a, c.size_b), []).append(c)
    for (a, b), group in pairs.items():
        pair = f"{label}={a:g} vs {label}={b:g}"
        found = [c for c in group if c.found]
        if not found:
            out.append(f"  {pair}: {group[0].note or 'no crossing in range'}")
            continue
        if len(found) > 1:
            out.append(f"  {pair}: the curves cross {len(found)} times (noise where they nearly "
                       "coincide, or a difference that changes sign more than once); they are not "
                       "independent measurements of one crossing point")
            indent = "    "
        else:
            indent = "  " + pair + ": "
        for c in found:
            if math.isfinite(c.err):
                out.append(f"{indent}x = {c.x:.6g} ± {c.err:.2g}, y = {c.y:.6g}  ({c.method})")
            else:
                out.append(f"{indent}x = {c.x:.6g} (no error: {c.method}), y = {c.y:.6g}")
    return out


CORRELATED_PAIRS = ("crossings of adjacent pairs share a curve, so their errors are correlated: do "
                    "not combine them as independent (e.g. for the error of a drift)")


def collapse_lines(r: CollapseResult) -> list[str]:
    s = "undefined" if not math.isfinite(r.quality) else f"{r.quality:.3g}"
    lines = [f"collapse: S = {s}; {r.method}"]      # the method ends with "S from n of m points"
    for k, v in (("xc", r.xc), ("nu", r.nu), ("a", r.a)):
        if k in r.fixed:
            lines.append(f"  {k} = {v:.6g} (fixed)")
        elif r.searched and k in r.errors and math.isfinite(r.errors[k]):
            lines.append(f"  {k} = {v:.6g} ± {r.errors[k]:.2g}")
        elif r.searched:
            lines.append(f"  {k} = {v:.6g} (error withheld)")
        else:
            lines.append(f"  {k} = {v:.6g} (given)")
    if r.poor:
        lines.append("  POOR COLLAPSE (S > 3 or undefined, too little overlap, x_c outside the "
                     "measured range, or an optimum held by coincident scaled points: see "
                     "Warnings)")
    return lines


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
