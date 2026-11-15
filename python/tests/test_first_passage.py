"""Files of models that can leave consensus (noisy voter, Ising/Potts at T > 0, ...): the
driver records the FIRST time consensus is reached, runs every replica to the cap and writes
stop_time = the cap. Current drivers add the first-passage time as a `consensus_time` column;
files from older drivers lack it, and then the first-passage times are not recoverable."""

import shutil
from pathlib import Path

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


def legacy(prefix, tmp_path) -> str:
    """A copy of `prefix`'s files as an older driver wrote them: no consensus_time column."""
    src = Path(prefix)
    dst = tmp_path / ("old_" + src.name)
    for f in src.parent.glob(src.name + "_*"):
        target = Path(str(dst) + f.name[len(src.name):])
        if not f.name.endswith("_replicas.csv"):
            shutil.copy(f, target)
            continue
        lines = f.read_text().splitlines()
        body = [i for i, l in enumerate(lines) if not l.startswith("#")]
        cols = lines[body[0]].split(",")
        k = cols.index("consensus_time")
        for i in body:
            cells = lines[i].split(",")
            lines[i] = ",".join(cells[:k] + cells[k + 1:])
        target.write_text("\n".join(lines) + "\n")
    return str(dst)


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


def test_summarise_replicas_first_passage_times_not_recoverable(outputs, tmp_path):
    t = find_outputs(legacy(outputs["nv"], tmp_path)).get("replicas")
    assert not t.has("consensus_time")
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


def test_plot_replicas_first_passage_skips_stop_times(outputs, tmp_path):
    t = find_outputs(legacy(outputs["nv"], tmp_path)).get("replicas")
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
    code = main([legacy(outputs["nv"], tmp_path), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert code == 0
    assert REASON in out
    assert "2000 ± 0" not in out


def test_median_refuses_first_passage_without_consensus_time(outputs, tmp_path):
    with pytest.raises(InputError, match="not recoverable"):
        consensus_time_vs_size([legacy(outputs["nv_sweep"], tmp_path)], censored="median", n_boot=20)


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


def test_first_passage_summary_is_not_doubled_and_has_no_fixation_line(outputs, tmp_path):
    rec = summarise_replicas(find_outputs(legacy(outputs["nv"], tmp_path)).get("replicas"))
    text = "\n".join(recovered(rec, "Ensemble summary"))
    assert text.count("not recoverable from the files") == 1
    assert "fixation" not in " ".join(rec.missing)


# ------------------------------------------------- files with the consensus_time column

def test_first_passage_times_recovered_from_consensus_time(outputs):
    """With the column, summarise_replicas reproduces the driver's own sweep row."""
    sw = find_outputs(outputs["nv_sweep"])
    row = sw.get("sweep")
    for i, pt in sw.points.items():
        t = pt.get("replicas")
        assert t.has("consensus_time") and is_first_passage(t)
        rec = summarise_replicas(t)
        for k in ("mean_consensus_time", "sd_consensus_time", "median_consensus_time",
                  "mean_consensus_time_lower_bound"):
            got, want = rec.values[k][0], float(row[k][i])
            assert math.isclose(got, want, rel_tol=1e-9) or (math.isnan(got) and math.isnan(want)), (k, got, want)
        assert not any("first-passage" in m for m in rec.missing)
        assert any("first-passage" in n and "consensus_time" in n for n in rec.notes)
        assert "fixation" not in " ".join(rec.missing)


def test_plot_replicas_first_passage_draws_first_passage_times(outputs):
    t = find_outputs(outputs["nv"]).get("replicas")
    warns = []
    figs = dict(plot_replicas(t, warnings=warns))
    try:
        fig = figs["replicas_stoptime"]
        assert fig.axes[0].get_xlabel() == "first-passage time"
        line = fig.axes[1].get_lines()[0]
        x, y = line.get_xdata(), line.get_ydata()
        reached = t["reached_consensus"] > 0.5
        assert x[-1] < 2000                                  # events at first passages, not the cap
        assert math.isclose(y[-1], 1 - reached.mean())
        assert not any(REASON in w for w in warns)
    finally:
        for f in figs.values():
            plt.close(f)


def test_cli_plot_first_passage_recovers_times(outputs, tmp_path, capsys):
    code = main([outputs["nv"], "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert REASON not in out and "2000 ± 0" not in out
    assert "mean_consensus_time" in out


def test_median_uses_consensus_time(outputs):
    sw = find_outputs(outputs["nv_sweep"]).get("sweep")
    r = consensus_time_vs_size([outputs["nv_sweep"]], censored="median", n_boot=50)
    assert np.allclose(r.y, sw["median_consensus_time"])
    assert any("first-passage" in n for n in r.notes)


def test_censored_first_passage_matches_the_driver(outputs):
    """Some replicas never reach consensus before the cap: the recovered mean (over the
    replicas that did), sd, median and lower bound still equal the driver's sweep row."""
    sw = find_outputs(outputs["nv_cens"])
    row = sw.get("sweep")
    censored = 0
    for i, pt in sw.points.items():
        t = pt.get("replicas")
        censored += int((t["reached_consensus"] < 0.5).sum())
        rec = summarise_replicas(t)
        for k in ("mean_consensus_time", "sd_consensus_time", "median_consensus_time",
                  "mean_consensus_time_lower_bound"):
            got, want = rec.values[k][0], float(row[k][i])
            assert math.isclose(got, want, rel_tol=1e-9) or (math.isnan(got) and math.isnan(want)), \
                (i, k, got, want)
        assert "mean_consensus_time" in rec.values
    assert censored > 0                                   # the fixture really is censored


def test_censored_first_passage_figure(outputs):
    sw = find_outputs(outputs["nv_cens"])
    i, pt = max(sw.points.items(), key=lambda kv: int((kv[1].get("replicas")["reached_consensus"] < 0.5).sum()))
    t = pt.get("replicas")
    unreached = (t["reached_consensus"] < 0.5).mean()
    assert unreached > 0
    warns = []
    figs = dict(plot_replicas(t, warnings=warns))
    try:
        labels = [ax.get_xlabel() for f in figs.values() for ax in f.axes]
        assert "final consensus_time" not in labels               # bookkeeping, not an observable
        assert not any("consensus_time" in w and "non-finite" in w for w in warns)
        line = figs["replicas_stoptime"].axes[1].get_lines()[0]
        assert math.isclose(line.get_ydata()[-1], unreached)      # censored at the cap: tail = their share
    finally:
        for f in figs.values():
            plt.close(f)


def test_consensus_time_column_marks_first_passage_even_if_none_reached(tmp_path):
    p = tmp_path / "n_replicas.csv"
    p.write_text(HEADER + "replica,reached_consensus,absorbed,stop_time,consensus_time\n"
                 "0,0,0,100,nan\n1,0,0,100,nan\n")
    t = read_table(p)
    assert is_first_passage(t)
    rec = summarise_replicas(t, {"steps": "100"})
    assert rec.values["consensus_fraction"][0] == 0
    assert rec.values["mean_consensus_time_lower_bound"][0] == 100
