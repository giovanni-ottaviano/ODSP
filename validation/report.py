"""The campaign's summary table: one row per check, as CSV and as a Markdown table."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from .core import Check

COLUMNS = ("model", "check", "quantity", "measured", "error", "predicted", "predicted_error",
           "source", "sigma", "tolerance_sigma", "one_sided", "sizes", "statistics", "warnings",
           "figure")


def _num(x: float) -> str:
    """Round-trip form for the CSV ('nan' and 'inf' as such)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return repr(x)


def _short(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}g}"


def _cell(text: str) -> str:
    """A Markdown table cell: '|' escaped, line breaks flattened."""
    return " ".join(str(text).split()).replace("|", "\\|")


def write_summary(checks: Sequence[Check], out_dir: Path) -> None:
    """Write summary.csv (full precision, the columns of COLUMNS; warnings joined by '; ')
    and summary.md (measured ± error, predicted ± error, σ to two decimals, and '!' after σ
    when the check is flagged: |σ| beyond its tolerance, or a one-sided check not significantly
    below its prediction; one-sided rows say so next to σ) into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for c in checks:
            w.writerow([c.model, c.check, c.quantity, _num(c.measured), _num(c.error),
                        _num(c.predicted), _num(c.predicted_error), c.source, _num(c.sigma),
                        _num(c.tolerance_sigma), c.one_sided or "", c.sizes, c.statistics,
                        "; ".join(c.warnings),
                        c.figure or ""])
    head = ("model", "check", "quantity", "measured", "predicted", "source", "σ", "sizes",
            "statistics", "warnings")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in checks:
        predicted = _short(c.predicted)
        if c.predicted_error:
            predicted += f" ± {_short(c.predicted_error, 2)}"
        sigma = f"{c.sigma:+.2f}" + (f" (one-sided: passes below -{c.tolerance_sigma:g})"
                                      if c.one_sided else "") + (" !" if c.flagged else "")
        row = (c.model, c.check, c.quantity, f"{_short(c.measured)} ± {_short(c.error, 2)}",
               predicted, c.source, sigma, c.sizes, c.statistics, "; ".join(c.warnings) or "-")
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    text = ["# Validation summary", "",
            f"{len(checks)} check(s). σ = (measured − predicted) / hypot(error, predicted error); "
            "'!' marks |σ| above the check's tolerance (a regression flag at smoke and small "
            "scale, see summary.csv for the tolerances); a one-sided check (an inequality, e.g. "
            "an exponent that must be negative) passes only when σ is below minus its tolerance "
            "and is marked '!' otherwise.", ""] + lines
    (out_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
