"""Files of models that can leave consensus (noisy voter, Ising/Potts at T > 0, ...): the
driver records the FIRST time consensus is reached, runs every replica to the cap and writes
stop_time = the cap; the first-passage time itself is not in the replicas file."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from odspy.cli import main
from odspy.errors import InputError
from odspy.helpers import consensus_time_vs_size, is_first_passage, summarise_replicas
from odspy.io import find_outputs, read_table
from odspy.plots.replicas import plot_replicas
from odspy.report import recovered

REASON = ("first-passage times: not recoverable from the files (the model can leave "
          "consensus; stop_time is the cap)")
HEADER = "# ODSP 0.1.0, git abc, written 2026-01-01T00:00:00+00:00\n"


def test_noisy_voter_files_are_first_passage(outputs):
    assert is_first_passage(find_outputs(outputs["nv"]).get("replicas"))
    sw = find_outputs(outputs["nv_sweep"])
    assert all(is_first_passage(pt.get("replicas")) for pt in sw.points.values())
    assert is_first_passage(sw.get("sweep"))


def test_absorbing_files_are_not_first_passage(outputs, tmp_path):
    # majority rule: frozen (absorbed without consensus) runs; mjs: frozen and capped runs
    assert not is_first_passage(find_outputs(outputs["maj"]).get("replicas"))
    mjs = find_outputs(outputs["mjs"])
    assert not is_first_passage(mjs.get("sweep"))
    assert not any(is_first_passage(pt.get("replicas")) for pt in mjs.points.values())
    # voter: finished runs absorbed, the rest hit the cap
    ens = find_outputs(outputs["ens_sweep"])
    assert not is_first_passage(ens.get("sweep"))
    assert not any(is_first_passage(pt.get("replicas")) for pt in ens.points.values())
    assert not is_first_passage(find_outputs(outputs["ens"]).get("replicas"))
    p = tmp_path / "v_replicas.csv"
    p.write_text(HEADER + "replica,reached_consensus,absorbed,stop_time\n"
                 "0,1,1,10\n1,0,0,100\n2,1,1,30\n")
    assert not is_first_passage(read_table(p))


def test_summarise_replicas_first_passage_times_not_recoverable(outputs):
    t = find_outputs(outputs["nv"]).get("replicas")
    assert (t["stop_time"] == 2000).all()        # the cap, not the first passage
    rec = summarise_replicas(t)
    for k in ("mean_consensus_time", "sd_consensus_time", "median_consensus_time",
              "mean_consensus_time_lower_bound"):
        assert k not in rec.values, k
    assert any(m.startswith("first-passage times (the model can leave consensus; stop_time "
                            "is the cap)") for m in rec.missing)
    assert rec.values["consensus_fraction"][0] == 1
    assert "mean_m" in rec.values and "mean_rho" in rec.values
    text = "\n".join(recovered(rec, "Ensemble summary"))
    assert "2000 ± 0" not in text
    assert ("not recoverable from the files: first-passage times (the model can leave "
            "consensus; stop_time is the cap)") in text


def test_plot_replicas_first_passage_skips_stop_times(outputs):
    t = find_outputs(outputs["nv"]).get("replicas")
    warns = []
    figs = dict(plot_replicas(t, warnings=warns))
    try:
        assert "replicas_stoptime" not in figs
        assert "replicas_final" in figs
        assert any(REASON in w and "survival" in w for w in warns)
    finally:
        for f in figs.values():
            plt.close(f)


def test_cli_plot_first_passage(outputs, tmp_path, capsys):
    code = main([outputs["nv"], "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert REASON in out
    assert "2000 ± 0" not in out


def test_median_refuses_first_passage(outputs):
    with pytest.raises(InputError, match="not recoverable"):
        consensus_time_vs_size([outputs["nv_sweep"]], censored="median", n_boot=20)


def test_exclude_uses_the_driver_first_passage_times(outputs, tmp_path, capsys, monkeypatch):
    sw = find_outputs(outputs["nv_sweep"]).get("sweep")
    r = consensus_time_vs_size([outputs["nv_sweep"]], censored="exclude")
    assert np.allclose(r.y, sw["mean_consensus_time"])
    assert (r.y < 2000).all()
    assert any("first-passage" in n for n in r.notes)
    from odspy import cli
    seen = {}
    real = cli.plot_fit

    def spy(result, xlabel, ylabel, prov=None):
        seen["ylabel"] = ylabel
        return real(result, xlabel, ylabel, prov)

    monkeypatch.setattr(cli, "plot_fit", spy)
    code = main(["fit", outputs["nv_sweep"], "--y", "mean_consensus_time", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "first-passage" in seen["ylabel"]
    assert "first-passage" in out


def test_sweep_figure_labels_first_passage_time(outputs):
    from odspy.plots import figures_for
    figs, warns = figures_for(find_outputs(outputs["nv_sweep"]))
    try:
        labels = [ax.get_ylabel() for _, f in figs for ax in f.axes]
        assert "first-passage time" in labels, labels      # short: a long label is clipped
    finally:
        for _, f in figs:
            plt.close(f)


def test_bounded_confidence_consensus_in_the_last_sweep_is_not_first_passage(outputs):
    t = find_outputs(outputs["dfe"]).get("replicas")
    late = (t["reached_consensus"] > 0.5) & (t["absorbed"] < 0.5)
    assert late.any() and (t["absorbed"] > 0.5).any()   # the fixture has the edge rows
    assert not is_first_passage(t)
    rec = summarise_replicas(t)
    assert "mean_consensus_time" in rec.values
    assert not any("first-passage" in m for m in rec.missing)


def test_sweep_with_late_consensus_rows_is_not_first_passage(tmp_path):
    p = tmp_path / "d_sweep.csv"
    p.write_text(HEADER + "n,replicas,consensus_fraction,absorbed_fraction\n"
                 "100,60,0.75,0.75\n150,60,0.65,0.62\n")
    assert not is_first_passage(read_table(p))
    p.write_text(HEADER + "n,replicas,consensus_fraction,absorbed_fraction\n"
                 "100,60,0,0\n150,60,0,0\n")
    assert not is_first_passage(read_table(p))     # nothing reached: cannot tell, not flagged


def test_first_passage_summary_is_not_doubled_and_has_no_fixation_line(outputs):
    rec = summarise_replicas(find_outputs(outputs["nv"]).get("replicas"))
    text = "\n".join(recovered(rec, "Ensemble summary"))
    assert text.count("not recoverable from the files") == 1
    assert "fixation" not in " ".join(rec.missing)
