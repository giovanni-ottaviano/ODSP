"""Files from runs with --first-replica k: replica labels start at k."""
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from odspy.io import find_outputs
from odspy.plots import plot_clusters_vs_key, plot_domains_vs_key

ENV = dict(os.environ, ODSP_MAX_THREADS="4")


def _run(odsp_bin, prefix, args):
    subprocess.run([str(odsp_bin), *args.split(), "--csv", str(prefix)], env=ENV, check=True,
                   capture_output=True)
    return find_outputs(str(prefix))


def _ydata(figs):
    out = []
    for _, f in figs:
        for ax in f.axes:
            out += [np.asarray(line.get_ydata(), float) for line in ax.lines if line.get_marker() == "o"]
        plt.close(f)
    return out


def test_domains_vs_key_with_first_replica(odsp_bin, tmp_path):
    base = ("--model voter --dims 8x8 --task ensemble --sweep steps=10,30 --observables domains "
            "--seed 5")
    full = _run(odsp_bin, tmp_path / "full", base + " --replicas 5")
    part = _run(odsp_bin, tmp_path / "part", base + " --replicas 2 --first-replica 3")
    # the split run's per-point domains are the full run's rows of replicas 3 and 4
    for i in (0, 1):
        f, p = full.points[i].get("domains"), part.points[i].get("domains")
        sel = f["replica"].astype(int) >= 3
        assert np.array_equal(f["replica"][sel], p["replica"])
        assert np.array_equal(f["size"][sel], p["size"])
    ys = _ydata(plot_domains_vs_key(part))
    # domains per replica, from the full run's replicas 3 and 4
    expect = []
    for i in (0, 1):
        f = full.points[i].get("domains")
        reps = f["replica"].astype(int)
        expect.append(np.mean([np.sum(reps == r) for r in (3, 4)]))
    assert np.allclose(ys[0], expect)
    # largest-domain fraction
    big = []
    for i in (0, 1):
        f = full.points[i].get("domains")
        reps = f["replica"].astype(int)
        big.append(np.mean([f["size"][reps == r].max() / 64 for r in (3, 4)]))
    assert np.allclose(ys[1], big)


def test_clusters_vs_key_with_first_replica(odsp_bin, tmp_path):
    base = ("--model deffuant --graph complete --n 60 --task ensemble --sweep epsilon=0.2,0.4 "
            "--steps 60 --seed 6")
    full = _run(odsp_bin, tmp_path / "full", base + " --replicas 4")
    part = _run(odsp_bin, tmp_path / "part", base + " --replicas 2 --first-replica 2")
    ys = _ydata(plot_clusters_vs_key(part))
    expect = []
    for i in (0, 1):
        f = full.points[i].get("clusters")
        reps = f["replica"].astype(int)
        expect.append(np.mean([np.sum((reps == r) & (f["size"] >= 6)) for r in (2, 3)]))
    assert np.allclose(ys[0], expect)
