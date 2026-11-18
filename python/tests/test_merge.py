"""odspy.merge: replica blocks run with --first-replica, joined back into one run."""
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from odspy.errors import InputError
from odspy.io import find_outputs, read_config, read_table
from odspy.merge import merge, recompute_ensemble_row, sweep_grid_size

ENV = dict(os.environ, ODSP_MAX_THREADS="4")

VOTER = ("--model voter --dims 8x8 --task ensemble --log-times 4 --observables rho,length,domains "
         "--save-trajectories true --save-states true --steps 200 --seed 4")


def _run(odsp_bin, prefix, args):
    proc = subprocess.run([str(odsp_bin), *args.split(), "--csv", str(prefix)], env=ENV,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return str(prefix)


def _blocks(odsp_bin, tmp_path, base, sizes):
    """The full run (all replicas) and its split into blocks of the given sizes."""
    full = _run(odsp_bin, tmp_path / "full", f"{base} --replicas {sum(sizes)}")
    parts, first = [], 0
    for b, size in enumerate(sizes):
        extra = f" --first-replica {first}" if first else ""
        parts.append(_run(odsp_bin, tmp_path / f"block{b}", f"{base} --replicas {size}{extra}"))
        first += size
    return full, parts


def _body(path):
    """The lines after the preamble: the header (the `# node,...` line of a states file) and
    the data lines."""
    lines = Path(path).read_text().splitlines()
    return [ln for ln in lines if not ln.startswith("#") or ln.startswith("# node,")]


def _same_body(full, out, kind):
    assert _body(f"{out}_{kind}.csv") == _body(f"{full}_{kind}.csv"), kind


def _close_table(full, out, kind, **tol):
    a, b = read_table(f"{full}_{kind}.csv"), read_table(f"{out}_{kind}.csv")
    assert a.names == b.names
    assert len(a) == len(b)
    for name in a.names:
        assert np.asarray(b[name], float) == pytest.approx(np.asarray(a[name], float), nan_ok=True,
                                                           **tol), (kind, name)


def test_merge_replica_blocks_equals_single_run(odsp_bin, tmp_path):
    full, parts = _blocks(odsp_bin, tmp_path, VOTER, [4, 6])
    out = tmp_path / "merged"
    merged = merge(out, parts[::-1])   # any order: blocks are sorted by first-replica
    assert merged.prefix == str(out)
    for kind in ("replicas", "trajectories", "domains", "states", "fixation"):
        _same_body(full, out, kind)
    _close_table(full, out, "timeseries", rel=1e-12, abs=1e-15)


def test_merge_first_passage_and_majority_columns(odsp_bin, tmp_path):
    nv = ("--model noisy-voter --noise 0.001 --dims 6x6 --task ensemble --steps 2000 --log-times 4 "
          "--observables rho --seed 5")
    full, parts = _blocks(odsp_bin, tmp_path / "nv", nv, [3, 7])
    out = tmp_path / "nv" / "merged"
    merge(out, parts)
    assert read_table(f"{out}_replicas.csv").has("consensus_time")
    _same_body(full, out, "replicas")
    _close_table(full, out, "timeseries", rel=1e-12, abs=1e-15)

    mj = ("--model majority --dims 8x8 --task ensemble --steps 200 --log-times 4 "
          "--observables rho,domains --seed 7")
    full, parts = _blocks(odsp_bin, tmp_path / "mj", mj, [4, 3, 3])
    out = tmp_path / "mj" / "merged"
    merge(out, parts)
    assert read_table(f"{out}_replicas.csv").has("freeze_time")
    for kind in ("replicas", "domains", "fixation"):
        _same_body(full, out, kind)
    _close_table(full, out, "timeseries", rel=1e-12, abs=1e-15)


def test_merge_clusters(odsp_bin, tmp_path):
    base = "--model deffuant --graph complete --n 60 --epsilon 0.3 --task ensemble --steps 60 --seed 6"
    full, parts = _blocks(odsp_bin, tmp_path, base, [2, 2])
    out = tmp_path / "merged"
    merge(out, parts)
    for kind in ("replicas", "clusters"):
        _same_body(full, out, kind)


def test_merge_correlation_ensemble(odsp_bin, tmp_path):
    base = ("--model voter --dims 12x12 --task ensemble --steps 100 --log-times 4 --correlation true "
            "--seed 2")
    full, parts = _blocks(odsp_bin, tmp_path, base, [3, 5])
    out = tmp_path / "merged"
    merge(out, parts)
    a, b = read_table(f"{full}_correlation.csv"), read_table(f"{out}_correlation.csv")
    assert b.kind == "correlation_ensemble"
    assert np.array_equal(a["time"], b["time"]) and np.array_equal(a["r"], b["r"])
    for name in ("G_mean", "G_err"):
        assert b[name] == pytest.approx(a[name], rel=1e-12), name


SWEEPS = {
    "voter": ("--model voter --task ensemble --sweep dims=6x6,8x8 --replicas 21 --steps 60 "
              "--threshold 0.9 --observables rho,length --seed 3"),
    "majority": ("--model majority --task ensemble --sweep dims=6x6,8x8 --replicas 20 --steps 30 "
                 "--threshold 0.9 --observables rho --seed 3"),
    "noisy-voter": ("--model noisy-voter --noise 0.001 --task ensemble --sweep dims=6x6,8x8 "
                    "--replicas 21 --steps 150 --observables rho --seed 3"),
}


@pytest.mark.parametrize("name", SWEEPS)
def test_recompute_ensemble_row_matches_driver(odsp_bin, tmp_path, name):
    prefix = _run(odsp_bin, tmp_path / "sw", SWEEPS[name])
    out = find_outputs(prefix)
    sweep = out.get("sweep")
    columns = sweep.names[1:]
    cap = float(read_config(prefix + "_config.txt")["steps"])
    for i in range(len(sweep)):
        pt = out.points[i]
        row = recompute_ensemble_row(pt.get("replicas"), pt.get("fixation"), columns, cap)
        assert list(row) == columns
        for c in columns:
            want = float(sweep[c][i])
            if c == "median_consensus_time":
                assert row[c] == want or (math.isnan(want) and math.isnan(row[c])), c
            else:
                assert row[c] == pytest.approx(want, rel=1e-12, nan_ok=True), (i, c)


def test_recompute_ensemble_row_refuses_unknown_column(odsp_bin, tmp_path):
    prefix = _run(odsp_bin, tmp_path / "sw", SWEEPS["voter"])
    pt = find_outputs(prefix).points[0]
    with pytest.raises(InputError, match="mean_magic"):
        recompute_ensemble_row(pt.get("replicas"), pt.get("fixation"), ["replicas", "mean_magic"], 60)


def test_merge_block_split_sweep(odsp_bin, tmp_path):
    """A whole sweep split into replica blocks: every point's files merged, every sweep row
    recomputed from the merged replicas."""
    full, parts = _blocks(odsp_bin, tmp_path, SWEEPS["voter"].replace(" --replicas 21", ""), [9, 12])
    out = tmp_path / "merged"
    merged = merge(out, parts)
    assert sorted(merged.points) == [0, 1]
    for i in (0, 1):
        _same_body(f"{full}_point{i}", f"{out}_point{i}", "replicas")
        _same_body(f"{full}_point{i}", f"{out}_point{i}", "fixation")
    a, b = read_table(f"{full}_sweep.csv"), read_table(f"{out}_sweep.csv")
    assert a.names == b.names and list(b["dims"]) == list(a["dims"])
    for c in a.names[1:]:
        assert b[c] == pytest.approx(a[c], rel=1e-12, nan_ok=True), c


def _copy_with_version(src, dst, commit):
    """Copy the outputs of prefix `src` to prefix `dst`, with another commit in the version line."""
    src = Path(src)
    for f in src.parent.iterdir():
        if f.name.startswith(src.name + "_"):
            text = f.read_text().splitlines(keepends=True)
            text = [("# ODSP 0.1.0, git " + commit + ", written " + ln.split(", written ", 1)[1])
                    if ln.startswith("# ODSP ") else ln for ln in text]
            (Path(dst).parent / (Path(dst).name + f.name[len(src.name):])).write_text("".join(text))
    return str(dst)


SMALL = "--model voter --dims 6x6 --task ensemble --steps 50 --seed 4"


@pytest.mark.parametrize("case", ["seed", "steps", "gap", "overlap", "version", "out"])
def test_merge_refuses_mismatched_inputs(odsp_bin, tmp_path, case):
    a = _run(odsp_bin, tmp_path / "a", SMALL + " --replicas 4")
    b_args = {
        "seed": SMALL.replace("--seed 4", "--seed 5") + " --replicas 5 --first-replica 4",
        "steps": SMALL.replace("--steps 50", "--steps 60") + " --replicas 5 --first-replica 4",
        "gap": SMALL + " --replicas 4 --first-replica 5",
        "overlap": SMALL + " --replicas 4 --first-replica 3",
        "version": SMALL + " --replicas 5 --first-replica 4",
        "out": SMALL + " --replicas 5 --first-replica 4",
    }[case]
    b = _run(odsp_bin, tmp_path / "b", b_args)
    if case == "version":
        b = _copy_with_version(b, tmp_path / "bv", "0000000")
    expect = {"seed": "seed", "steps": "steps", "gap": "0..3", "overlap": "3..6",
              "version": "0000000", "out": "output"}[case]
    out = b if case == "out" else tmp_path / "merged"
    with pytest.raises(InputError) as err:
        merge(out, [a, b])
    msg = str(err.value)
    assert expect in msg
    assert a in msg and b in msg
    assert not list(tmp_path.glob("merged_*"))   # nothing written


def test_merged_provenance(odsp_bin, tmp_path):
    full, parts = _blocks(odsp_bin, tmp_path, VOTER, [4, 6])
    out = tmp_path / "merged"
    merged = merge(out, parts)
    cfg = read_config(f"{out}_config.txt")
    assert cfg["replicas"] == "10" and "first-replica" not in cfg
    assert cfg["csv"] == str(out)
    # the single run's configuration, apart from csv and first-replica (the driver writes
    # first-replica = 0; the merge drops the key when it is 0)
    single = {k: v for k, v in read_config(f"{full}_config.txt").items() if k != "first-replica"}
    assert cfg == {**single, "csv": str(out)}
    prov = merged.provenance
    assert prov.version and prov.commit and prov.written
    assert prov.config["replicas"] == "10" and "first-replica" not in prov.config
    for f in Path(tmp_path).glob("merged_*"):
        lines = f.read_text().splitlines()
        assert lines[0].startswith("# ODSP ")
        assert lines[1] == f"# merged by odspy from: {parts[0]}, {parts[1]}", f
    # a single input: a copy, with the merged provenance line
    one = merge(tmp_path / "one", [parts[1]])
    assert one.provenance.config["first-replica"] == "4" and one.provenance.config["replicas"] == "6"
    _same_body(parts[1], tmp_path / "one", "replicas")


# ------------------------------------------------------------------ sweep points

SWEEP3 = ("--model voter --task ensemble --sweep dims=6x6,8x8,10x10 --steps 60 --threshold 0.9 "
          "--observables rho,length --seed 3")


def _points(odsp_bin, tmp_path, base, n, extra=None):
    """The full sweep and one --sweep-point job per point (`extra[i]`: more options for point i)."""
    full = _run(odsp_bin, tmp_path / "full", base)
    jobs = [_run(odsp_bin, tmp_path / f"job{i}", f"{base} --sweep-point {i} {(extra or {}).get(i, '')}")
            for i in range(n)]
    return full, jobs


def test_sweep_grid_size():
    assert sweep_grid_size({"sweep": "dims=8x8,16x16"}) == 2
    assert sweep_grid_size({"sweep": "temperature=2.1:2.4:0.05"}) == 7
    assert sweep_grid_size({"sweep": "temperature=2.1:2.4:0.05", "sweep2": "noise=0.1,0.2,0.3"}) == 21
    assert sweep_grid_size({"sweep": "dims=8x8,,16x16,"}) == 2
    assert sweep_grid_size({}) is None


def test_merge_points_equals_single_run(odsp_bin, tmp_path):
    full, jobs = _points(odsp_bin, tmp_path, SWEEP3 + " --replicas 10", 3)
    out = tmp_path / "merged"
    merged = merge(out, jobs[::-1])
    assert sorted(merged.points) == [0, 1, 2]
    assert _body(f"{out}_sweep.csv") == _body(f"{full}_sweep.csv")
    for i in range(3):
        _same_body(f"{full}_point{i}", f"{out}_point{i}", "replicas")
        _same_body(f"{full}_point{i}", f"{out}_point{i}", "fixation")
    assert "sweep-point" not in read_config(f"{out}_config.txt")
    assert not [ln for ln in Path(f"{out}_sweep.csv").read_text().splitlines()
                if ln.startswith("#   sweep-point")]


def test_merge_points_with_blocks(odsp_bin, tmp_path):
    base = SWEEP3 + " --replicas 12"
    full = _run(odsp_bin, tmp_path / "full", base)
    jobs = [_run(odsp_bin, tmp_path / "j0", base + " --sweep-point 0"),
            _run(odsp_bin, tmp_path / "j1a", SWEEP3 + " --replicas 5 --sweep-point 1"),
            _run(odsp_bin, tmp_path / "j1b",
                 SWEEP3 + " --replicas 7 --first-replica 5 --sweep-point 1"),
            _run(odsp_bin, tmp_path / "j2", base + " --sweep-point 2")]
    out = tmp_path / "merged"
    merge(out, jobs)
    for i in range(3):
        _same_body(f"{full}_point{i}", f"{out}_point{i}", "replicas")
    a, b = read_table(f"{full}_sweep.csv"), read_table(f"{out}_sweep.csv")
    assert a.names == b.names and list(b["dims"]) == list(a["dims"])
    for c in a.names[1:]:
        assert b[c] == pytest.approx(a[c], rel=1e-12, nan_ok=True), c


def test_merge_stationary_points(odsp_bin, tmp_path):
    base = ("--model ising --task stationary --sweep temperature=2.2,2.3,2.4 --dims 6x6 "
            "--replicas 2 --samples 20 --seed 3")
    full, jobs = _points(odsp_bin, tmp_path, base, 3)
    out = tmp_path / "merged"
    merged = merge(out, jobs)
    assert _body(f"{out}_sweep.csv") == _body(f"{full}_sweep.csv")
    assert len(read_table(f"{out}_sweep.csv")) == 3
    assert "sweep-point" not in read_config(f"{out}_config.txt")
    assert merged.get("sweep") is not None


def test_merge_single_stationary_run(odsp_bin, tmp_path):
    """A stationary run without --sweep (one job: samples, histogram, states) has nothing to
    join: its files are copied with the merged preamble; two such runs are refused."""
    base = ("--model ising --task stationary --temperature 2.3 --dims 6x6 --replicas 2 "
            "--samples 20 --save-states true --seed 3")
    run = _run(odsp_bin, tmp_path / "run", base)
    out = tmp_path / "merged"
    merged = merge(out, [run])
    for kind in ("samples", "histogram", "states"):
        _same_body(run, out, kind)
    assert read_config(f"{out}_config.txt")["csv"] == str(out)
    assert merged.get("samples") is not None and merged.get("histogram") is not None
    other = _run(odsp_bin, tmp_path / "other", base)
    with pytest.raises(InputError, match="one input"):
        merge(tmp_path / "m2", [run, other])
    assert not list(tmp_path.glob("m2_*"))


def test_merge_refuses_stationary_point_with_whole_run(odsp_bin, tmp_path):
    """A --sweep-point job and a whole run of the same stationary sweep are refused in either
    order (never one silently dropped)."""
    base = ("--model ising --task stationary --sweep temperature=2.2,2.3 --dims 6x6 "
            "--replicas 2 --samples 20 --seed 3")
    whole = _run(odsp_bin, tmp_path / "whole", base)
    point = _run(odsp_bin, tmp_path / "pt1", base + " --sweep-point 1")
    for inputs in ([point, whole], [whole, point]):
        with pytest.raises(InputError):
            merge(tmp_path / "m", inputs)
        assert not list(tmp_path.glob("m_*"))


def test_merge_points_missing_or_duplicate(odsp_bin, tmp_path):
    _, jobs = _points(odsp_bin, tmp_path, SWEEP3 + " --replicas 6", 3)
    with pytest.raises(InputError, match="point 1") as err:
        merge(tmp_path / "m", [jobs[0], jobs[2]])
    assert "missing" in str(err.value)
    twin = _run(odsp_bin, tmp_path / "twin", SWEEP3 + " --replicas 6 --sweep-point 1")
    with pytest.raises(InputError, match="point 1") as err:
        merge(tmp_path / "m", [*jobs, twin])
    assert "duplicated" in str(err.value)
    assert not list(tmp_path.glob("m_*"))


def test_merge_removes_stale_files(odsp_bin, tmp_path):
    base = VOTER + " --correlation true"
    a = _run(odsp_bin, tmp_path / "a", base + " --replicas 3")
    b = _run(odsp_bin, tmp_path / "b", base + " --replicas 3 --first-replica 3")
    out = tmp_path / "merged"
    assert (Path(f"{a}_correlation.csv")).exists()
    merge(out, [a, b])
    assert Path(f"{out}_correlation.csv").exists()
    (tmp_path / "merged_merge_report.txt").write_text("old")
    (tmp_path / "merged_notes.txt").write_text("keep")
    (tmp_path / "merged_other_replicas.csv").write_text("keep")   # another prefix's file
    plain = VOTER
    c = _run(odsp_bin, tmp_path / "c", plain + " --replicas 3")
    d = _run(odsp_bin, tmp_path / "d", plain + " --replicas 3 --first-replica 3")
    merge(out, [c, d])
    assert not Path(f"{out}_correlation.csv").exists()
    assert not (tmp_path / "merged_merge_report.txt").exists()
    assert (tmp_path / "merged_notes.txt").exists()
    assert (tmp_path / "merged_other_replicas.csv").exists()
    assert Path(f"{out}_replicas.csv").exists() and Path(f"{out}_config.txt").exists()


def test_merge_blocks_with_different_threads(odsp_bin, tmp_path):
    base = SMALL + " --replicas 4"
    a = _run(odsp_bin, tmp_path / "a", base + " --threads 1")
    b = _run(odsp_bin, tmp_path / "b",
             SMALL + " --replicas 5 --first-replica 4 --threads 3")
    full = _run(odsp_bin, tmp_path / "full", SMALL + " --replicas 9")
    merge(tmp_path / "m", [a, b])
    _same_body(full, tmp_path / "m", "replicas")


def test_merge_block_without_fixation_file(odsp_bin, tmp_path):
    # replica 0 of this run does not reach consensus within the cap, so its block writes no
    # fixation file; replica 1 does
    base = "--model voter --dims 8x8 --task ensemble --steps 100 --seed 4"
    full, parts = _blocks(odsp_bin, tmp_path, base, [1, 1, 1])
    assert not Path(f"{parts[0]}_fixation.csv").exists()
    assert Path(f"{parts[1]}_fixation.csv").exists()
    out = tmp_path / "m"
    merge(out, parts)
    _same_body(full, out, "fixation")
    _same_body(full, out, "replicas")
