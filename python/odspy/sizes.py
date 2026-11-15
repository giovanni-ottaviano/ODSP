"""Group sweep tables into curves of one observable, one curve per system size."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InputError
from .io import Table, find_outputs


@dataclass
class SizeCurve:
    size: float
    label: str  # "L" or "N"
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None
    xkey: str
    column: str
    source: str


def _parse_dims(text: str, where: str) -> list[int]:
    try:
        dims = [int(p) for p in text.strip().lower().split("x")]
    except ValueError:
        raise InputError(f"{where}: cannot parse dims '{text}' (expected e.g. 16x16)") from None
    if not dims or any(d <= 0 for d in dims):
        raise InputError(f"{where}: invalid dims '{text}'")
    return dims


def size_of(config: dict[str, str], size: str = "auto", where: str = "config") -> tuple[float, str]:
    """System size from a configuration: (value, label) with label 'L' or 'N'."""
    if size not in ("auto", "L", "N"):
        raise InputError(f"{where}: size must be 'auto', 'L' or 'N', got '{size}'")
    has_dims, has_n = "dims" in config, "n" in config
    if not has_dims and not has_n:
        raise InputError(f"{where}: configuration has neither 'dims' nor 'n'; cannot get a size")
    use_dims = has_dims and (not has_n or config.get("graph", "lattice") == "lattice")
    if not use_dims:
        raw = config["n"]
        try:
            n = float(raw)
        except ValueError:
            n = math.nan
        if not math.isfinite(n) or n != int(n) or n <= 0:
            raise InputError(f"{where}: invalid n '{raw}' (expected a positive integer)")
        if size == "L":
            raise InputError(f"{where}: size='L' needs a lattice (dims), this input has n")
        return int(n), "N"
    dims = _parse_dims(config["dims"], where)
    total = math.prod(dims)
    if size == "N":
        return total, "N"
    if size == "L":
        side = total ** (1 / len(dims))
        return (dims[0] if len(set(dims)) == 1 else side), "L"
    if len(set(dims)) != 1:
        raise InputError(f"{where}: non-cubic dims {config['dims']}; pass size='N' "
                         "(number of sites) or size='L' (geometric mean side)")
    return dims[0], "L"


def _numeric(table: Table, name: str) -> np.ndarray:
    col = table[name]
    if col.dtype == object:
        raise InputError(f"{table.path}: column '{name}' is not numeric")
    return col


def _config(table: Table) -> dict[str, str]:
    return table.provenance.config


def _curve(table, rows, xkey, column, err_column, size, label):
    x = _numeric(table, xkey)[rows]
    order = np.argsort(x, kind="stable")
    y = _numeric(table, column)[rows][order]
    err = None
    if err_column is not None:
        err = _numeric(table, err_column)[rows][order]
    elif table.has(column + "_err"):
        err = _numeric(table, column + "_err")[rows][order]
    return SizeCurve(size, label, x[order], y, err, xkey, column, str(table.path))


def _sweep_table(src) -> Table:
    out = find_outputs(src)
    table = out.get("sweep")
    if table is None:
        raise InputError(f"{src}: no sweep table (<prefix>_sweep.csv); run with --sweep")
    if len(table) == 0:
        raise InputError(f"{table.path}: sweep table has no rows")
    if not table.provenance.config:
        table.provenance = out.provenance
    return table


def families(sources: list[str | Path], column: str, xkey: str | None = None,
             size: str = "auto", err_column: str | None = None) -> list[SizeCurve]:
    """One SizeCurve per system size, sorted by size."""
    if not sources:
        raise InputError("no inputs given")
    if size not in ("auto", "L", "N"):
        raise InputError(f"size must be 'auto', 'L' or 'N', got '{size}'")
    names = [str(s) for s in sources]
    tables = [_sweep_table(s) for s in sources]
    models = {n: _config(t).get("model") for n, t in zip(names, tables)}
    if len(set(models.values())) > 1:
        raise InputError("inputs have different models: "
                         + ", ".join(f"{n} ({m})" for n, m in models.items()))
    keys = {n: tuple(t.names[:2] if _config(t).get("sweep2") else t.names[:1])
            for n, t in zip(names, tables)}
    if len(set(keys.values())) > 1:
        raise InputError("inputs sweep different parameters: "
                         + ", ".join(f"{n} ({'/'.join(k)})" for n, k in keys.items()))

    curves: list[SizeCurve] = []
    if _config(tables[0]).get("sweep2"):
        if len(tables) > 1:
            raise InputError("inputs with --sweep2 cannot be combined: "
                             + ", ".join(names))
        table, outer, inner = tables[0], tables[0].names[0], tables[0].names[1]
        if outer not in ("dims", "n"):
            raise InputError(f"{names[0]}: outer sweep key is '{outer}', expected 'dims' or 'n'")
        xk = xkey or inner
        cells = table[outer]
        seen: dict[float, str] = {}
        for cell in dict.fromkeys(cells):
            text = str(int(cell)) if isinstance(cell, float) and cell == int(cell) else str(cell)
            value, label = size_of({outer: text}, size, f"{table.path} ({outer}={text})")
            if value in seen:
                raise InputError(f"{names[0]}: {outer}={text} and {outer}={seen[value]} "
                                 f"give the same size {value}")
            seen[value] = text
            rows = np.array([c == cell for c in cells])
            curves.append(_curve(table, rows, xk, column, err_column, value, label))
    else:
        owners: dict[tuple[float, str], str] = {}
        for name, table in zip(names, tables):
            swept = table.names[0]
            if swept in ("dims", "n"):
                raise InputError(f"{name}: sweeps '{swept}' in a single table; use "
                                 "--sweep2 with the size as the outer key or separate runs")
            value, label = size_of(_config(table), size, name)
            if (value, label) in owners:
                raise InputError(f"inputs {owners[(value, label)]} and {name} have the "
                                 f"same size {label}={value}")
            owners[(value, label)] = name
            rows = np.ones(len(table), dtype=bool)
            curves.append(_curve(table, rows, xkey or swept, column, err_column, value, label))
    if len({c.label for c in curves}) > 1:
        raise InputError("inputs mix lattice (L) and node-count (N) sizes; pass size='N': "
                         + ", ".join(f"{n} ({c.label})" for n, c in zip(names, curves)))
    return sorted(curves, key=lambda c: c.size)
