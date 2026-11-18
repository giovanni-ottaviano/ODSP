"""Join the outputs of one run split across jobs into one run.

A large ensemble is split across jobs with ``--first-replica k --replicas R_b``: each job
runs replicas k..k+R_b-1 of the same ensemble (same seed), so its per-replica rows are the
single run's rows. A sweep is split with ``--sweep-point i``: each job runs the flattened
point i of the declared sweep (and may itself be one replica block of that point).
``merge`` joins the pieces back into one prefix that the rest of odspy reads as if it came
from one run: per-replica files are concatenated in replica order, aggregate files are
recombined exactly from the blocks' replica counts, means and errors, sweep rows
(statistics such as the median, which cannot be recombined from aggregates) are recomputed
from the merged replicas files, and the points' one-row sweep tables are stacked in point
order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InputError
from .helpers import _driver_median, _num, is_first_passage, replica_times
from .io import (SUFFIXES, Outputs, Provenance, Table, _read_lines, _reduce_prefix, _to_column,
                 find_outputs, read_config)

# Configuration keys that may differ between the inputs of one merge: where the files went,
# how many threads ran them, and which part of the run each input holds.
_FREE_KEYS = ("csv", "threads", "first-replica", "sweep-point")
_VERSION_RE = re.compile(r"# (ODSP \S+?, git \S+?), written \S+$")
_KEY_LINE_RE = re.compile(r"(#   )?([A-Za-z0-9_.-]+) = (.*)$")
_MERGED_TAG = "# merged by odspy from: "
_FILE_NAME_RE = re.compile(r"(?:point(\d+)_)?(" + "|".join(SUFFIXES) + r")\.csv")
# Per-replica files: one row (or several) per replica, labelled by replica.
_PER_REPLICA = ("replicas", "trajectories", "clusters", "domains")


def sweep_grid_size(config: dict[str, str]) -> int | None:
    """The number of sweep points of a configuration (`None` without `sweep`), counted as the
    driver's parse_sweep does: a list has its non-empty items, a range a:b:s has
    floor((b - a)/s + 1e-9) + 1 values, and `sweep2` multiplies the counts."""
    def count(text: str) -> int:
        key, _, rest = text.partition("=")
        if not key or not rest:
            raise InputError(f"bad sweep '{text}': expected key=v1,v2,... or key=start:stop:step")
        if ":" in rest:
            try:
                a, b, step = (float(v) for v in rest.split(":"))
            except ValueError:
                raise InputError(f"bad sweep range '{text}': expected start:stop:step") from None
            if not step > 0 or b < a:
                raise InputError(f"bad sweep range '{text}': stop >= start and step > 0 expected")
            return int(math.floor((b - a) / step + 1e-9)) + 1
        n = sum(1 for item in rest.split(",") if item)
        if n == 0:
            raise InputError(f"bad sweep '{text}': no values")
        return n

    if not config.get("sweep"):
        return None
    n = count(config["sweep"])
    if config.get("sweep2"):
        n *= count(config["sweep2"])
    return n


def _fmt(x: float) -> str:
    """Shortest round-trip form, like the driver: integers without '.0', 'nan', 'inf'."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == int(x) and abs(x) < 1e16:
        return str(int(x))
    return repr(x)


@dataclass
class _Csv:
    """One output file split into its preamble ('#' lines), header and data lines."""
    path: Path
    preamble: list[str]
    header: str
    data: list[str]

    @property
    def names(self) -> list[str]:
        h = self.header[2:] if self.header.startswith("# ") else self.header
        return h.split(",")


def _read_csv(path: Path, states: bool) -> _Csv:
    preamble: list[str] = []
    header: str | None = None
    data: list[str] = []
    for line in _read_lines(path):
        if not line.strip():
            continue
        if header is None:
            if line.startswith("#") and not (states and line.startswith("# node,")):
                preamble.append(line)
            else:
                header = line
        elif not line.startswith("#"):
            data.append(line)
    if header is None:
        raise InputError(f"{path}: no header line found")
    return _Csv(path, preamble, header, data)


@dataclass
class _Input:
    name: str                          # the prefix as given (reduced), for messages
    prefix: Path
    config: dict[str, str]
    files: dict[tuple[int | None, str], Path]   # (point index or None, suffix) -> file
    first: int
    replicas: int
    point: int | None = None           # the --sweep-point of the job, None for a whole run

    @property
    def last(self) -> int:
        return self.first + self.replicas - 1

    def range(self) -> str:
        return f"{self.name} (replicas {self.first}..{self.last})"


def _load(name: str) -> _Input:
    prefix = Path(name)
    cfg_path = Path(name + "_config.txt")
    if not cfg_path.exists():
        raise InputError(f"{name}: no configuration file {cfg_path}")
    config = read_config(cfg_path)
    files: dict[tuple[int | None, str], Path] = {}
    folder = prefix.parent
    stem = prefix.name + "_"
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.name.startswith(stem):
                m = _FILE_NAME_RE.fullmatch(f.name[len(stem):])
                if m:
                    files[(None if m.group(1) is None else int(m.group(1)), m.group(2))] = f
    if not files:
        raise InputError(f"{name}: no ODSP outputs found (looked for {name}_*.csv)")
    point = None
    if "sweep-point" in config:
        try:
            point = int(config["sweep-point"])
        except ValueError:
            raise InputError(f"{cfg_path}: bad sweep-point '{config['sweep-point']}'") from None
    task = config.get("task")
    # A stationary run has no replica blocks: only its sweep points can be joined (a whole
    # stationary run, one input, is copied).
    if task not in ("ensemble", "stationary"):
        raise InputError(f"{name}: merge joins --task ensemble outputs (and --task stationary "
                         f"outputs of --sweep-point jobs or of one whole run), this is task "
                         f"'{task}'")
    first, replicas = 0, 0
    if task == "ensemble":
        try:
            first = int(config.get("first-replica", "0"))
            replicas = int(config["replicas"])
        except (KeyError, ValueError):
            raise InputError(f"{cfg_path}: no valid 'replicas' / 'first-replica' entries") from None
    return _Input(name, prefix, config, files, first, replicas, point)


def _versions(inp: _Input) -> set[str]:
    """The 'ODSP <ver>, git <commit>' lines of an input's files (provenance=false files have none)."""
    found = set()
    for path in [Path(inp.name + "_config.txt"), *inp.files.values()]:
        for line in _read_lines(path):
            if not line.startswith("#"):
                break
            m = _VERSION_RE.match(line)
            if m:
                found.add(m.group(1))
                break
    return found


def _check_inputs(inputs: list[_Input], names: str) -> None:
    keys = sorted({k for inp in inputs for k in inp.config} - set(_FREE_KEYS) - {"replicas"})
    for k in keys:
        values = [inp.config.get(k) for inp in inputs]
        if len(set(values)) > 1:
            listed = "; ".join(f"{inp.name}: {'(absent)' if v is None else v}"
                               for inp, v in zip(inputs, values))
            raise InputError(f"cannot merge {names}: configuration key '{k}' differs ({listed})")
    versions = [_versions(inp) for inp in inputs]
    if len(set().union(*versions)) > 1:
        listed = "; ".join(f"{inp.name}: {', '.join(sorted(v)) or '(none)'}"
                           for inp, v in zip(inputs, versions))
        raise InputError(f"cannot merge {names}: written by different ODSP versions ({listed})")


def _check_blocks(blocks: list[_Input], names: str) -> None:
    for a, b in zip(blocks, blocks[1:]):
        if a.last + 1 < b.first:
            missing = (f"replica {a.last + 1}" if b.first == a.last + 2
                       else f"replicas {a.last + 1}..{b.first - 1}")
            raise InputError(f"cannot merge {names}: {a.range()} and {b.range()} are not "
                             f"contiguous ({missing} missing)")
        if a.last + 1 > b.first:
            raise InputError(f"cannot merge {names}: {a.range()} and {b.range()} overlap")
    kinds = {k for k in blocks[0].files if k[1] != "fixation"}
    for b in blocks[1:]:
        other = {k for k in b.files if k[1] != "fixation"}
        if other != kinds:
            extra = sorted(other ^ kinds, key=str)[0]
            label = ("" if extra[0] is None else f"point{extra[0]}_") + extra[1]
            raise InputError(f"cannot merge {names}: {blocks[0].name} and {b.name} hold different "
                             f"files ({label}.csv is in one only)")


# ------------------------------------------------------------------ provenance

def _update_lines(lines: list[str], out: str, replicas: int | None, first: int,
                  merged_from: str) -> list[str]:
    """The first input's preamble (or configuration file) as the merged run's: replica count
    (kept when None), first replica (dropped when 0) and prefix updated, sweep-point dropped,
    and a line naming the inputs after the version line."""
    result = []
    for line in lines:
        if line.startswith(_MERGED_TAG):   # an input that was itself a merge: the new line replaces it
            continue
        m = _KEY_LINE_RE.fullmatch(line)
        if m and (m.group(1) or not line.startswith("#")):
            indent, key = m.group(1) or "", m.group(2)
            if key == "sweep-point":
                continue
            if key == "replicas":
                if replicas is not None:
                    line = f"{indent}replicas = {replicas}"
            elif key == "first-replica":
                if first == 0:
                    continue
                line = f"{indent}first-replica = {first}"
            elif key == "csv":
                line = f"{indent}csv = {out}"
        elif line.startswith(("# configuration", "# odsp configuration")) and "odsp --config " in line:
            head, _, tail = line.partition("odsp --config ")
            line = head + "odsp --config " + out + "_config.txt" + ("):" if tail.endswith("):") else "")
        result.append(line)
        if _VERSION_RE.match(line):
            result.append(_MERGED_TAG + merged_from)
    return result


# ------------------------------------------------------------------ per kind

def _check_headers(files: list[_Csv], names: str) -> None:
    for f in files[1:]:
        if f.header != files[0].header:
            raise InputError(f"cannot merge {names}: {files[0].path} and {f.path} have different "
                             f"columns ({files[0].header} / {f.header})")


def _check_labels(f: _Csv, inp: _Input, names: str) -> None:
    """A replicas file holds the block's replicas, labelled from its first replica on."""
    labels = [line.split(",", 1)[0] for line in f.data]
    if labels != [str(inp.first + i) for i in range(inp.replicas)]:
        raise InputError(f"cannot merge {names}: {f.path} does not hold replicas "
                         f"{inp.first}..{inp.last} (as its configuration says)")


def _concat(files: list[_Csv], names: str) -> tuple[str, list[str]]:
    _check_headers(files, names)
    return files[0].header, [line for f in files for line in f.data]


def _join_states(files: list[_Csv], names: str) -> tuple[str, list[str]]:
    rows = [[line.split(",") for line in f.data] for f in files]
    for f, r in zip(files[1:], rows[1:]):
        if [c[0] for c in r] != [c[0] for c in rows[0]]:
            raise InputError(f"cannot merge {names}: {files[0].path} and {f.path} have different "
                             "node columns")
    header = ",".join(["# node"] + [n for f in files for n in f.names[1:]])
    data = [",".join([cells[0][0]] + [c for r in cells for c in r[1:]]) for cells in zip(*rows)]
    return header, data


def _sum_fixation(files: list[_Csv], names: str) -> tuple[str, list[str]]:
    _check_headers(files, names)
    if files[0].names != ["opinion", "replicas", "probability"]:
        raise InputError(f"{files[0].path}: unexpected fixation columns {files[0].header}")
    counts: dict[str, int] = {}
    for f in files:
        for line in f.data:
            opinion, count = line.split(",")[:2]
            counts[opinion] = counts.get(opinion, 0) + int(float(count))
    total = sum(counts.values())
    data = [f"{op},{n},{_fmt(n / total)}" for op, n in sorted(counts.items(), key=lambda e: float(e[0]))]
    return files[0].header, data


def _pool(sizes: list[int], means: list[float], errs: list[float]) -> tuple[float, float]:
    """Mean and error of the union of blocks from each block's (R_b, mean, err), with the
    driver's error sd_pop / sqrt(R). The spread uses deviations from the merged mean (no
    sum-of-squares cancellation)."""
    total = sum(sizes)
    mean = sum(r * m for r, m in zip(sizes, means)) / total
    var = sum(r * ((e * math.sqrt(r)) ** 2 + (m - mean) ** 2)
              for r, m, e in zip(sizes, means, errs)) / total
    return mean, math.sqrt(var) / math.sqrt(total)


def _recombine(files: list[_Csv], sizes: list[int], keys: int, names: str) -> tuple[str, list[str]]:
    """An ensemble time series (keys = 1: time) or G(r) table (keys = 2: time, r), rebuilt
    from the blocks' rows."""
    _check_headers(files, names)
    cols = files[0].names
    if keys == 1:
        rest = cols[3:]
        ok = cols[:3] == ["time", "running_fraction", "survival"] and len(rest) % 3 == 0 and all(
            rest[i + 1] == rest[i][:-5] + "_err" and rest[i + 2] == rest[i][:-5] + "_mean_running"
            and rest[i].endswith("_mean") for i in range(0, len(rest), 3))
    else:
        ok = cols == ["time", "r", "G_mean", "G_err"]
    if not ok:
        raise InputError(f"{files[0].path}: unexpected columns {files[0].header}")
    rows = [[line.split(",") for line in f.data] for f in files]
    for f, r in zip(files[1:], rows[1:]):
        if [c[:keys] for c in r] != [c[:keys] for c in rows[0]]:
            what = "times" if keys == 1 else "times and distances r"
            raise InputError(f"cannot merge {names}: {files[0].path} and {f.path} have different {what}")
    total = sum(sizes)
    data = []
    for cells in zip(*rows):
        out = list(cells[0][:keys])
        vals = [[float(x) for x in c[keys:]] for c in cells]
        if keys == 1:
            # The driver writes count / R_b: R_b times it gives the count back.
            running = [round(v[0] * r) for v, r in zip(vals, sizes)]
            surviving = [round(v[1] * r) for v, r in zip(vals, sizes)]
            out += [_fmt(sum(running) / total), _fmt(sum(surviving) / total)]
            for j in range(2, len(vals[0]), 3):
                mean, err = _pool(sizes, [v[j] for v in vals], [v[j + 1] for v in vals])
                n = sum(running)
                run_sum = sum(k * v[j + 2] for k, v in zip(running, vals) if k > 0)
                out += [_fmt(mean), _fmt(err), _fmt(run_sum / n if n else math.nan)]
        else:
            mean, err = _pool(sizes, [v[0] for v in vals], [v[1] for v in vals])
            out += [_fmt(mean), _fmt(err)]
        data.append(",".join(out))
    return files[0].header, data


# ------------------------------------------------------------------ sweep rows

def _seq_sum(values) -> float:
    """Sum in order, as the driver accumulates (so the result is bit-equal to its own)."""
    s = 0.0
    for v in values:
        s += float(v)
    return s


def _mean_sd(values) -> tuple[float, float]:
    """The driver's mean and population sd: sum and sum of squares in order, max(0, .)."""
    n = len(values)
    s, s2 = 0.0, 0.0
    for v in values:
        v = float(v)
        s += v
        s2 += v * v
    mean = s / n
    return mean, math.sqrt(max(0.0, s2 / n - mean * mean))


def recompute_ensemble_row(replicas: Table, fixation: Table | None, columns: Sequence[str],
                           cap: float) -> dict[str, float]:
    """The driver's sweep-row statistics for one ensemble point, recomputed from its replicas
    file (and fixation file, for exit_prob_up) the way task_ensemble computes them. `cap` is
    the step cap (`steps`): mean_consensus_time_lower_bound counts unreached replicas there.
    A column that cannot be computed from these files is an InputError naming it."""
    R = len(replicas)
    if R == 0:
        raise InputError(f"{replicas.path}: no replicas")
    reached = _num(replicas, "reached_consensus") > 0.5
    count = int(reached.sum())
    cache: dict[str, object] = {}

    def times() -> np.ndarray:
        if "times" not in cache:
            t = replica_times(replicas)
            if t is None:
                raise InputError(f"{replicas.path}: first-passage replicas file without a "
                                 "consensus_time column: cannot recompute the consensus-time columns")
            cache["times"] = t
        return cache["times"]

    def consensus_mean_sd() -> tuple[float, float]:
        if "cm" not in cache:
            cache["cm"] = _mean_sd(times()[reached]) if count else (0.0, 0.0)
        return cache["cm"]

    def freeze() -> tuple[float, float]:
        if "fz" not in cache:
            v = _num(replicas, "freeze_time")
            v = v[~np.isnan(v)]
            cache["fz"] = _mean_sd(v) if v.size else (math.nan, math.nan)
        return cache["fz"]

    def threshold() -> np.ndarray:
        v = _num(replicas, "threshold_time")
        return v[v >= 0]

    def exit_prob_up() -> float:
        if count == 0:
            return 0.0
        if fixation is None or is_first_passage(replicas):
            raise InputError(f"{replicas.path}: cannot recompute exit_prob_up without a fixation file")
        won = [float(n) for op, n in zip(fixation["opinion"], fixation["replicas"]) if float(op) == 1.0]
        return (won[0] if won else 0.0) / count

    row: dict[str, float] = {}
    for c in columns:
        if c == "replicas":
            v = float(R)
        elif c == "consensus_fraction":
            v = count / R
        elif c == "absorbed_fraction":
            v = int((_num(replicas, "absorbed") > 0.5).sum()) / R
        elif c == "mean_consensus_time":
            v = consensus_mean_sd()[0]
        elif c == "sd_consensus_time":
            v = consensus_mean_sd()[1]
        elif c == "median_consensus_time":
            v = _driver_median(times(), reached)
        elif c == "mean_consensus_time_lower_bound":
            v = _seq_sum(np.where(reached, times(), cap)) / R
        elif c == "mean_freeze_time":
            v = freeze()[0]
        elif c == "sd_freeze_time":
            v = freeze()[1]
        elif c == "exit_prob_up":
            v = exit_prob_up()
        elif c == "threshold_fraction":
            v = threshold().size / R
        elif c == "mean_threshold_time":
            t = threshold()
            v = _seq_sum(t) / t.size if t.size else math.nan
        elif c.startswith(("mean_", "sd_")) and replicas.has(c.split("_", 1)[1]):
            mean, sd = _mean_sd(_num(replicas, c.split("_", 1)[1]))
            v = mean if c.startswith("mean_") else sd
        else:
            raise InputError(f"{replicas.path}: cannot recompute sweep column '{c}' from the "
                             "replicas file")
        row[c] = v
    return row


def _table(path: Path, kind: str, header: str, data: list[str]) -> Table:
    """A merged file, not yet written, as a Table."""
    names = header.split(",")
    rows = [line.split(",") for line in data]
    columns = {n: _to_column([r[j] for r in rows]) for j, n in enumerate(names)}
    return Table(path, kind, Provenance(), columns, names)


def _sweep_rows(sweep: list[_Csv], merged: dict[tuple[int | None, str], tuple[str, list[str]]],
                config: dict[str, str], out: str, names: str) -> tuple[str, list[str]]:
    """The sweep table of block-split points: each row recomputed from the point's merged
    replicas (and fixation) file, the swept values kept as written."""
    _check_headers(sweep, names)
    swept = [config[k].split("=", 1)[0] for k in ("sweep", "sweep2") if config.get(k)]
    cols = sweep[0].names
    if cols[:len(swept)] != swept:
        raise InputError(f"{sweep[0].path}: expected the swept keys {', '.join(swept)} first")
    points = sorted(p for p, suffix in merged if suffix == "replicas" and p is not None)
    if len(points) != len(sweep[0].data):
        raise InputError(f"cannot merge {names}: {sweep[0].path} has {len(sweep[0].data)} rows "
                         f"but there are {len(points)} point replicas files")
    data = []
    for point, line in zip(points, sweep[0].data):
        keys = line.split(",")[:len(swept)]
        steps = dict(zip(swept, keys)).get("steps", config.get("steps"))
        try:
            cap = float(steps)
        except (TypeError, ValueError):
            raise InputError(f"cannot merge {names}: no valid 'steps' in the configuration") from None
        base = Path(f"{out}_point{point}")
        reps = _table(Path(f"{base}_replicas.csv"), "replicas", *merged[(point, "replicas")])
        fix = merged.get((point, "fixation"))
        fix_t = _table(Path(f"{base}_fixation.csv"), "fixation", *fix) if fix else None
        row = recompute_ensemble_row(reps, fix_t, cols[len(swept):], cap)
        data.append(",".join(keys + [_fmt(v) for v in row.values()]))
    return sweep[0].header, data


# ------------------------------------------------------------------ merge

_Merged = dict[tuple[int | None, str], tuple[str, list[str]]]


def _merge_blocks(blocks: list[_Input], out: str, names: str) -> tuple[_Merged, dict]:
    """Merged (header, data lines) of each file kind of replica blocks of one run (sorted,
    checked), and the first block's preamble of each."""
    sizes = [b.replicas for b in blocks]
    keys = sorted({k for b in blocks for k in b.files},
                  key=lambda k: (k[0] is not None, k[0] or 0, k[1] == "sweep", k[1]))
    merged: _Merged = {}
    preambles: dict[tuple[int | None, str], list[str]] = {}
    for key in keys:
        point, suffix = key
        files = [_read_csv(b.files[key], suffix == "states") for b in blocks if key in b.files]
        preambles[key] = files[0].preamble
        if suffix == "sweep":
            continue   # after the points: rows are recomputed from their merged replicas
        if suffix in _PER_REPLICA:
            if suffix == "replicas":
                for f, b in zip(files, blocks):
                    _check_labels(f, b, names)
            merged[key] = _concat(files, names)
        elif suffix == "states":
            merged[key] = _join_states(files, names)
        elif suffix == "fixation":
            merged[key] = _sum_fixation(files, names)
        elif suffix in ("timeseries", "correlation") and files[0].names[0] == "time":
            merged[key] = _recombine(files, sizes, 1 if suffix == "timeseries" else 2, names)
        else:
            raise InputError(f"cannot merge {names}: {files[0].path} cannot be merged across "
                             "replica blocks")
    for key in keys:
        if key[1] == "sweep":
            files = [_read_csv(b.files[key], False) for b in blocks]
            if len(files) == 1:   # a whole point: its row is the driver's, kept as written
                merged[key] = (files[0].header, files[0].data)
            else:
                merged[key] = _sweep_rows(files, merged, blocks[0].config, out, names)
    return merged, preambles


def _merge_stationary_point(inp: _Input, names: str) -> tuple[_Merged, dict]:
    """A stationary sweep point: its one-row sweep table (a stationary sweep writes no
    per-point files)."""
    extra = sorted(k for k in inp.files if k != (None, "sweep"))
    if (None, "sweep") not in inp.files or extra:
        raise InputError(f"cannot merge {names}: {inp.name} (point {inp.point}) should hold only "
                         "its sweep table (a stationary sweep point writes no other file)")
    f = _read_csv(inp.files[(None, "sweep")], False)
    return {(None, "sweep"): (f.header, f.data)}, {(None, "sweep"): f.preamble}


def _copy_stationary_run(inputs: list[_Input], names: str) -> tuple[_Merged, dict]:
    """A whole stationary run (no --sweep-point): nothing to join, so exactly one input,
    whose files are taken as they are."""
    if len(inputs) != 1:
        raise InputError(f"cannot merge {names}: a stationary run without --sweep-point has "
                         "nothing to join (one input expected; split stationary runs by "
                         "--sweep-point)")
    files = {k: _read_csv(p, k[1] == "states") for k, p in inputs[0].files.items()}
    return ({k: (f.header, f.data) for k, f in files.items()},
            {k: f.preamble for k, f in files.items()})


def _group_points(loaded: list[_Input], names: str) -> dict[int, list[_Input]]:
    """The inputs of a sweep split by --sweep-point, checked to hold every point once."""
    if any(i.point is None for i in loaded):
        raise InputError(f"cannot merge {names}: some inputs have a --sweep-point and some have "
                         "not (a whole sweep is not merged with its points)")
    groups: dict[int, list[_Input]] = {}
    for inp in loaded:
        groups.setdefault(inp.point, []).append(inp)
    n = sweep_grid_size(loaded[0].config)
    if n is None:
        raise InputError(f"cannot merge {names}: the inputs have a --sweep-point but no --sweep")
    for p in sorted(groups):
        if p >= n:
            raise InputError(f"cannot merge {names}: point {p} is outside the sweep "
                             f"(it has {n} points, 0..{n - 1})")
        stationary = loaded[0].config.get("task") == "stationary"
        ranges = [(b.first, b.replicas) for b in groups[p]]
        if len(set(ranges)) < len(ranges) or (stationary and len(ranges) > 1):
            who = ", ".join(b.name for b in groups[p])
            raise InputError(f"cannot merge {names}: point {p} is duplicated ({who})")
    missing = [p for p in range(n) if p not in groups]
    if missing:
        what = (f"point {missing[0]}" if len(missing) == 1
                else "points " + ", ".join(str(p) for p in missing))
        raise InputError(f"cannot merge {names}: {what} of the {n} sweep points missing")
    return groups


def _clear_stale(out: str, keep: set[Path], protect: set[Path]) -> None:
    """Remove the ODSP outputs, configuration and merge report that an earlier merge into
    `out` left and this one does not write (only files named for this exact prefix)."""
    prefix = Path(out)
    if not prefix.parent.is_dir():
        return
    stem = prefix.name + "_"
    for f in prefix.parent.iterdir():
        rest = f.name[len(stem):]
        if (f.name.startswith(stem) and f not in keep and f not in protect
                and (_FILE_NAME_RE.fullmatch(rest) or rest in ("config.txt", "merge_report.txt"))):
            f.unlink()


def merge(out: str | Path, inputs: Sequence[str | Path]) -> Outputs:
    """Join the pieces of one run (replica blocks run with --first-replica, sweep points run
    with --sweep-point, or blocks of sweep points) into the prefix `out`, and return
    find_outputs(out).

    Inputs (output prefixes, in any order) must have equal configurations apart from csv,
    threads, first-replica, sweep-point and replicas, be written by the same ODSP version
    and commit, hold contiguous, non-overlapping replica ranges for each point, and together
    every point of the sweep once. Per-replica files (replicas, trajectories, clusters,
    domains) are concatenated in replica order, states files joined column-wise, fixation
    counts summed (a block without a fixation file adds none), ensemble time series and G(r)
    tables recombined from each block's replica count, means and errors (err = sd_pop /
    sqrt(R), as the driver writes it), and sweep rows recomputed from the merged replicas
    files; the one-row sweep tables of the points are stacked in point order and the
    `point<i>` files kept. Stationary sweep points need one input each; a whole stationary
    run (without --sweep-point) is one input, whose files are copied. Every merged file
    keeps the first input's preamble, with the configuration updated (replicas = the total,
    first-replica = the smallest start, dropped when 0, sweep-point dropped, csv = out) and
    a line '# merged by odspy from: <inputs>'. Files of an earlier merge into `out` that
    this one does not write are removed. Every refusal is an InputError naming the inputs.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    if not inputs:
        raise InputError("merge: no inputs given")
    out_s = _reduce_prefix(out)
    given = [_reduce_prefix(i) for i in inputs]
    names = ", ".join(given)
    for g in given:
        if Path(g).resolve() == Path(out_s).resolve():
            raise InputError(f"cannot merge {names}: the output prefix {out_s} is one of the inputs")
    loaded = [_load(g) for g in given]
    _check_inputs(loaded, names)
    stationary = loaded[0].config.get("task") == "stationary"
    # Inputs of one sweep point (or of a run without --sweep-point) are replica blocks.
    if stationary and all(i.point is None for i in loaded):
        groups = [loaded]
    elif any(i.point is not None for i in loaded):
        groups = [sorted(g, key=lambda b: b.first) for _, g in sorted(_group_points(loaded, names).items())]
    else:
        groups = [sorted(loaded, key=lambda b: b.first)]
    merged: _Merged = {}
    preambles: dict = {}
    parts: list[tuple[int, int]] = []
    for blocks in groups:
        if stationary and blocks[0].point is None:
            m, pre = _copy_stationary_run(blocks, names)
        elif stationary:
            m, pre = _merge_stationary_point(blocks[0], names)
        else:
            _check_blocks(blocks, names)
            m, pre = _merge_blocks(blocks, out_s, names)
            parts.append((blocks[0].first, sum(b.replicas for b in blocks)))
            stray = sorted(k[0] for k in m if k[0] is not None and k[0] != blocks[0].point
                           and blocks[0].point is not None)
            if stray:
                raise InputError(f"cannot merge {names}: {blocks[0].name} (point {blocks[0].point}) "
                                 f"holds files of point {stray[0]}")
        for key, value in m.items():
            if key[1] == "sweep" and key in merged:
                if value[0] != merged[key][0]:
                    raise InputError(f"cannot merge {names}: the sweep tables have different "
                                     f"columns ({merged[key][0]} / {value[0]})")
                merged[key] = (value[0], merged[key][1] + value[1])
            else:
                merged[key] = value
        for key, value in pre.items():
            preambles.setdefault(key, value)
    if len(set(parts)) > 1:
        raise InputError(f"cannot merge {names}: the sweep points hold different replica ranges "
                         f"({', '.join(f'{f}..{f + n - 1}' for f, n in parts)})")
    first, total = parts[0] if parts else (0, None)
    merged_from = ", ".join(b.name for blocks in groups for b in blocks)
    texts: dict[Path, str] = {}
    for (point, suffix), (header, data) in sorted(
            merged.items(), key=lambda e: (e[0][0] is not None, e[0][0] or 0, e[0][1])):
        path = Path(f"{out_s}_{'' if point is None else f'point{point}_'}{suffix}.csv")
        pre = _update_lines(preambles[(point, suffix)], out_s, total, first, merged_from)
        texts[path] = "\n".join(pre + [header] + data) + "\n"
    cfg = _update_lines(_read_lines(Path(groups[0][0].name + "_config.txt")), out_s, total, first,
                        merged_from)
    texts[Path(out_s + "_config.txt")] = "\n".join(cfg) + "\n"
    # everything checked and built before anything is touched
    protect = {f for i in loaded for f in (*i.files.values(), Path(i.name + "_config.txt"))}
    clash = sorted(str(p) for p in texts if p in protect)
    if clash:
        raise InputError(f"cannot merge {names}: it would overwrite the input file {clash[0]}")
    _clear_stale(out_s, set(texts), protect)
    for path, text in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return find_outputs(out_s)
