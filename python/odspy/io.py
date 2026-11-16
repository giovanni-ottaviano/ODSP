"""Readers for the CSV files written by ``odsp --csv P``."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import InputError

SUFFIXES = (
    "timeseries", "correlation", "trajectories", "replicas", "sweep", "fixation",
    "clusters", "domains", "histogram", "samples", "states",
)
_FILE_RE = re.compile(r"_(?:point(\d+)_)?(" + "|".join(SUFFIXES) + r")\.csv$")
_CAPTION_PARAMS = ("temperature", "noise", "epsilon", "p", "q")


@dataclass
class Provenance:
    version: str | None = None
    commit: str | None = None
    written: str | None = None
    command: str | None = None
    config: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def caption(self) -> str:
        parts = []
        if self.config.get("model"):
            parts.append(self.config["model"])
        swept = {s.split("=", 1)[0] for s in (self.get("sweep"), self.get("sweep2")) if s}
        if self.config.get("dims"):
            parts.append("dims swept" if "dims" in swept else self.config["dims"])
        elif self.config.get("n"):
            parts.append("n swept" if "n" in swept else "n=" + self.config["n"])
        for key in _CAPTION_PARAMS:
            if key in self.config and key not in swept:
                parts.append(f"{key}={self.config[key]}")
                break
        text = ", ".join(parts)
        if self.version:
            tag = f"ODSP {self.version}" + (f", git {self.commit}" if self.commit else "")
            text = f"{text} ({tag})" if text else f"({tag})"
        return text


def _parse_comment(line: str, prov: Provenance) -> None:
    body = line[1:].strip()
    m = re.match(r"ODSP (\S+?), git (\S+?), written (\S+)$", body)
    if m:
        prov.version, prov.commit, prov.written = m.groups()
    elif body.startswith("command:"):
        prov.command = body[len("command:"):].strip()
    elif line.startswith("#   ") and "=" in body:
        k, v = body.split("=", 1)
        prov.config[k.strip()] = v.strip()


def _parse_config_lines(lines: list[str], path: Path) -> Provenance:
    prov = Provenance()
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            _parse_comment(line, prov)
        elif "=" in line:
            k, v = line.split("=", 1)
            prov.config[k.strip()] = v.strip()
    return prov


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()   # a BOM (spreadsheet export) is dropped
    except FileNotFoundError:
        raise InputError(f"{path}: file not found") from None
    except (OSError, UnicodeDecodeError) as e:
        raise InputError(f"{path}: cannot read ({e})") from None


def read_config(path: str | Path) -> dict[str, str]:
    path = Path(path)
    return _parse_config_lines(_read_lines(path), path).config


@dataclass
class Table:
    path: Path
    kind: str
    provenance: Provenance
    columns: dict[str, np.ndarray]
    names: list[str]

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.columns[name]
        except KeyError:
            raise InputError(
                f"{self.path}: no column '{name}' (has: {', '.join(self.names)})"
            ) from None

    def has(self, name: str) -> bool:
        return name in self.columns

    def __len__(self) -> int:
        if not self.names:
            return 0
        return len(self.columns[self.names[0]])


def _to_column(cells: list[str]) -> np.ndarray:
    vals = []
    for c in cells:
        c = c.strip()
        if c == "":
            vals.append(np.nan)
            continue
        try:
            vals.append(float(c))
        except ValueError:
            return np.array([x.strip() for x in cells], dtype=object)
    return np.array(vals, dtype=float)


def _kind_of(path: Path, suffix: str, first: str) -> str:
    if suffix == "timeseries":
        table = {"sweep": "run", "time": "ensemble"}
        if first not in table:
            raise InputError(f"{path}: unexpected first column '{first}' in timeseries "
                             "(expected 'sweep' or 'time')")
        return "timeseries_" + table[first]
    if suffix == "correlation":
        table = {"sweep": "run", "time": "ensemble", "r": "stationary"}
        if first not in table:
            raise InputError(f"{path}: unexpected first column '{first}' in correlation "
                             "(expected 'sweep', 'time' or 'r')")
        return "correlation_" + table[first]
    return suffix


def read_table(path: str | Path) -> Table:
    path = Path(path)
    m = _FILE_RE.search(path.name)
    if not m:
        raise InputError(f"{path}: not an ODSP output file (expected <prefix>_<kind>.csv "
                         f"with kind one of {', '.join(SUFFIXES)})")
    suffix = m.group(2)
    prov = Provenance()
    header: list[str] | None = None
    rows: list[list[str]] = []
    reader_lines = _read_lines(path)
    for lineno, raw in enumerate(reader_lines, start=1):
        if not raw.strip():
            continue
        if raw.startswith("#"):
            if header is None and suffix == "states" and raw.startswith("# node,"):
                header = [c.strip() for c in raw[2:].split(",")]
            else:
                _parse_comment(raw, prov)
            continue
        cells = next(csv.reader([raw]))
        if header is None:
            header = [c.strip() for c in cells]
            continue
        if len(cells) != len(header):
            raise InputError(f"{path}: line {lineno} has {len(cells)} cells, "
                             f"expected {len(header)}")
        rows.append(cells)
    if not header:
        raise InputError(f"{path}: no header line found")
    if len(set(header)) != len(header):
        raise InputError(f"{path}: duplicate column names in header")
    columns = {}
    for j, name in enumerate(header):
        try:
            columns[name] = _to_column([r[j] for r in rows])
        except Exception as e:  # pragma: no cover - defensive
            raise InputError(f"{path}: column '{name}': {e}") from None
    return Table(path, _kind_of(path, suffix, header[0]), prov, columns, header)


@dataclass
class Outputs:
    prefix: str
    provenance: Provenance
    tables: dict[str, Table]
    points: dict[int, "Outputs"] = field(default_factory=dict)

    def get(self, kind: str) -> Table | None:
        return self.tables.get(kind)


def _reduce_prefix(prefix: str | Path) -> str:
    s = str(prefix)
    m = _FILE_RE.search(s)
    if m and s.endswith(".csv"):
        s = s[:m.start()]
    return s


def _hint(given: str, base: Path) -> str:
    """For a prefix with no outputs: what it should look like, and the prefixes of the ODSP
    outputs in its folder that are closest to it."""
    import difflib
    out = ""
    if given.endswith(".csv"):
        out += (f"; {given} " + ("is not an ODSP output file" if Path(given).exists() else "does not exist")
                + f" (files are named <prefix>_<kind>.csv, kind one of {', '.join(SUFFIXES)})")
    folder = base.parent if str(base.parent) else Path(".")
    try:
        names = [f.name for f in folder.iterdir() if f.name.endswith(".csv")] if folder.is_dir() else []
    except OSError:
        names = []
    found = sorted({f[:m.start()] for f in names if (m := _FILE_RE.search(f))})
    stem = base.name[:-4] if base.name.endswith(".csv") else base.name
    close = difflib.get_close_matches(stem, found, n=3, cutoff=0.5)
    if close:
        where = "" if str(base.parent) in ("", ".") else str(base.parent) + "/"
        out += "; did you mean " + " or ".join(where + c for c in close) + "?"
    return out


def find_outputs(prefix: str | Path) -> Outputs:
    pfx = _reduce_prefix(prefix)
    base = Path(pfx)
    folder = base.parent
    stem = base.name + "_"
    top: dict[str, Table] = {}
    pts: dict[int, dict[str, Table]] = {}
    files: list[Path] = []
    if folder.is_dir():
        files = sorted(f for f in folder.iterdir()
                       if f.name.startswith(stem) and f.name.endswith(".csv"))
    for f in files:
        m = re.fullmatch(r"(?:point(\d+)_)?(" + "|".join(SUFFIXES) + r")\.csv",
                         f.name[len(stem):])
        if not m:
            continue
        t = read_table(f)
        if m.group(1) is None:
            top[t.kind] = t
        else:
            pts.setdefault(int(m.group(1)), {})[t.kind] = t
    if not top and not pts:
        raise InputError(f"no ODSP outputs found for prefix {pfx} (looked for {pfx}_*.csv)"
                         + _hint(str(prefix), base))
    cfg_path = Path(pfx + "_config.txt")
    cfg = None
    if cfg_path.exists():
        cfg = _parse_config_lines(_read_lines(cfg_path), cfg_path)

    def fill(tables: dict[str, Table]) -> None:
        if cfg is None:
            return
        for t in tables.values():
            if not t.provenance.config and t.provenance.version is None:
                t.provenance = cfg

    fill(top)
    for tables in pts.values():
        fill(tables)
    every = list(top.values()) + [t for tables in pts.values() for t in tables.values()]
    first = every[0]
    prov = first.provenance if first.provenance.config or cfg is None else cfg
    point_outputs = {
        i: Outputs(f"{pfx}_point{i}", prov, tables) for i, tables in sorted(pts.items())
    }
    return Outputs(pfx, prov, top, point_outputs)
