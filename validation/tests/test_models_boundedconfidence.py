"""Deffuant and Hegselmann-Krause (validation/models/boundedconfidence.py): consensus thresholds
and major-cluster counts on the complete graph."""

from __future__ import annotations

import numpy as np

from validation import theory
from validation.core import _value
from validation.models import MODELS

from .smoke import run_smoke

# R21: one row per BKR window (2, 4, 6 major clusters)
CHECKS = ["deffuant.threshold", "deffuant.clusters_n2", "deffuant.clusters_n4",
          "deffuant.clusters_n6", "hk.threshold", "hk.clusters"]


def test_boundedconfidence_smoke(tmp_path, odsp_bin):
    assert "boundedconfidence" in MODELS
    rows = run_smoke(tmp_path, odsp_bin, "boundedconfidence", CHECKS)
    by = {r["check"]: r for r in rows}

    d = by["deffuant.threshold"]
    eps, err = theory.deffuant_major_threshold()
    assert float(d["predicted"]) == eps and float(d["predicted_error"]) == err
    assert d["source"] == "deffuant_rate_equation"
    assert "one major cluster" in d["quantity"]                 # the consensus definition
    # R17: the largest N is the measured value, every N as drift, residual-shift caveat
    assert "threshold at every N" in d["warnings"]
    assert "residual finite-size shift" in d["warnings"]
    assert "complete consensus" in d["warnings"]                # the other definition, diagnostic
    assert "binomial" in d["statistics"]
    # R22: the gap in absolute and relative size, no finite-N form, the reproductions
    assert "gap at the largest N" in d["warnings"] and "% of eps*" in d["warnings"]
    assert "no published finite-N form; not extrapolated" in d["warnings"]
    assert "independent program" in d["warnings"]
    assert "not kept on disk" not in d["warnings"]                # unverifiable claim removed
    assert "monotone within errors" in d["warnings"] or "not monotone" in d["warnings"]
    assert "Delta mesh only" in d["warnings"]
    assert "0 of 400" in d["warnings"]                          # replay without the cap

    for n, eps in zip((2, 4, 6), sorted((e for e, _ in theory.deffuant_window_points()),
                                         reverse=True)):
        c = by[f"deffuant.clusters_n{n}"]
        assert float(c["predicted"]) == n and float(c["predicted_error"]) == 0.0
        assert c["source"] == "deffuant_rate_equation"
        assert f"{eps:.4f}" in c["sizes"]
        assert "replaces" in c["warnings"] and "1/(2 eps)" in c["warnings"]
        assert "1/(L eps)" in c["warnings"]                      # BKR's asymptotic form
        assert "count at every N" in c["warnings"]
        assert "decided after seeing the data" in c["warnings"]  # R21
        assert "exactly at every N" in c["warnings"]
        assert c["figure"] == "deffuant.clusters"                # one figure for the points

    h = by["hk.threshold"]
    c0, e0 = theory.hk_threshold()
    assert float(h["predicted"]) == c0 and float(h["predicted_error"]) == e0
    assert h["source"] == "hk_threshold"
    assert "complete consensus" in h["quantity"]
    assert "fixed order" in h["quantity"]                        # R23: Fortunato's scheme
    assert "threshold at every N" in h["warnings"]
    assert "residual finite-size shift" in h["warnings"]
    assert "synchronous" in h["warnings"]                        # the other schemes, diagnostics
    assert "random-order sweeps" in h["warnings"]
    assert "final number of clusters" in h["warnings"]           # what Fortunato says
    assert "modelling choice" in h["warnings"]                   # N -> infinity bracket as +-

    k = by["hk.clusters"]
    assert k["source"] == "hk_clusters"
    assert k["one_sided"] == "below"                            # "slightly fewer than 1/(2 eps)"
    assert "2 eps" in k["quantity"]
    assert "detects overcounting only" in k["quantity"]
    assert "synchronous" in k["quantity"]                         # BHT's scheme


def test_boundedconfidence_runs_match_the_cited_protocol():
    mod = MODELS["boundedconfidence"]
    for scale in ("smoke", "small", "full"):
        runs = mod.runs(scale)
        names = {r.name for r in runs}
        for model in ("deffuant", "hk"):
            sizes = mod.SCALE[scale][model]["sizes"]
            assert {f"{model}_threshold_N{N}" for N in sizes} <= names
        for r in runs:
            a = r.args
            # complete graph, opinions uniform on [0, 1], ensembles of independent replicas
            assert _value(a, "--graph") == "complete" and _value(a, "--task") == "ensemble"
            assert _value(a, "--init") == "uniform"
            assert "largest_cluster" in _value(a, "--observables").split(",")
            assert "clusters" in _value(a, "--observables").split(",")
            if _value(a, "--model") == "deffuant":
                assert _value(a, "--mu") == "0.5"                     # BKR: meet halfway
            else:
                assert _value(a, "--model") == "hk"
                # R23: Fortunato's threshold in fixed-order sweeps; diagnostics in random-order
                # sweeps and synchronous; BHT's counts synchronous
                expected = {"hk_threshold": "ordered", "hk_fresh": "sweep"}.get(
                    r.name.rsplit("_N", 1)[0], "synchronous")
                assert _value(a, "--update") == expected and _value(a, "--synchronous") is None
        hk_sizes = mod.SCALE[scale]["hk"]["sizes"]
        assert {f"hk_sync_N{N}" for N in hk_sizes} <= names         # synchronous diagnostic
        assert {f"hk_fresh_N{N}" for N in hk_sizes} <= names        # random-order diagnostic
        # the Deffuant count points are the centres of BKR's even windows
        cl = [r for r in runs if r.name.startswith("deffuant_clusters")]
        assert cl
        eps = sorted(float(v) for v in _value(cl[0].args, "--sweep")[len("epsilon="):].split(","))
        assert np.allclose(eps, sorted(e for e, _ in theory.deffuant_window_points()))
    for model in ("deffuant", "hk"):
        full = mod.SCALE["full"][model]["sizes"]
        assert len(full) >= 4 and max(full) >= 8 * min(full)
    assert max(mod.SCALE["full"]["hk"]["sizes"]) == 10000          # Fortunato's N


def test_half_crossing():
    mod = MODELS["boundedconfidence"]
    eps = np.array([0.1, 0.2, 0.3, 0.4])
    P = np.array([0.0, 0.25, 0.75, 1.0])
    assert abs(mod.half_crossing(eps, P) - 0.25) < 1e-12
    # the first upward crossing counts
    P2 = np.array([0.0, 0.6, 0.4, 1.0])
    assert abs(mod.half_crossing(eps, P2) - (0.1 + 0.1 * 0.5 / 0.6)) < 1e-12
    assert np.isnan(mod.half_crossing(eps, np.array([0.6, 0.7, 0.8, 0.9])))
