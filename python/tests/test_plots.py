import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from odspy.io import Provenance, find_outputs
from odspy.plots import (caption, figures_for, plot_domains, plot_sweep, plot_timeseries_ensemble,
                         save)
from odspy.plots.states import plot_states

HEADER = "# ODSP 0.1.0, git abc, written 2026-01-01T00:00:00+00:00\n"
CENSORED = "censored: mean biased low"


def cfg(**kw):
    return HEADER + "# configuration:\n" + "".join(f"#   {k} = {v}\n" for k, v in kw.items())


@pytest.fixture(autouse=True)
def no_leaks():
    plt.close("all")
    yield
    assert plt.get_fignums() == [], "a test left figures open"
    plt.close("all")


def test_every_fixture_renders(outputs, tmp_path):
    for name, prefix in outputs.items():
        out = find_outputs(prefix)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            figs, warns = figures_for(out)
        assert figs, name
        assert all(isinstance(w, str) for w in warns)
        names = [n for n, _ in figs]
        assert len(set(names)) == len(names), name
        for kind in out.tables:
            assert any(kind.split("_")[0] in n for n in names), (name, kind, names)
        for n, fig in figs:
            paths = save(fig, tmp_path / name, n)
            assert {p.suffix for p in paths} == {".pdf", ".png"}
            assert all(p.stat().st_size > 0 for p in paths)
    assert plt.get_fignums() == []


def test_per_point_files(outputs):
    out = find_outputs(outputs["ens_sweep"])
    figs, warns = figures_for(out)
    assert any("per-point" in w and "3" in w for w in warns)
    more, _ = figures_for(out, per_point=True)
    assert len(more) > len(figs)
    for _, f in figs + more:
        plt.close(f)


def test_cont_clusters_vs_key(outputs):
    figs, warns = figures_for(find_outputs(outputs["cont"]))
    names = [n for n, _ in figs]
    assert any("clusters" in n for n in names)
    assert any(CENSORED in w for w in warns)       # consensus_fraction < 1 at both points
    for _, f in figs:
        plt.close(f)


def test_single_point_sweep(tmp_path):
    (tmp_path / "e_sweep.csv").write_text(
        cfg(task="ensemble", sweep="dims=6x6", model="voter", dims="6x6", replicas="20")
        + "dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time,"
          "median_consensus_time,mean_consensus_time_lower_bound,mean_m,sd_m\n"
          "6x6,20,1,38.6,26.2,28,38.6,0.1,1\n")
    out = find_outputs(str(tmp_path / "e"))
    figs = plot_sweep(out.get("sweep"))
    assert figs
    for _, f in figs:
        plt.close(f)


def test_censored_points_faded(tmp_path):
    rows = ["dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time,"
            "median_consensus_time,mean_consensus_time_lower_bound"]
    for k, f in zip((6, 8, 10), (1.0, 0.8, 1.0)):
        rows.append(f"{k}x{k},20,{f},{k * k * 2.0},{k * k * 0.5},{k * k * 2.0},{k * k * 2.5}")
    (tmp_path / "e_sweep.csv").write_text(
        cfg(task="ensemble", sweep="dims=6x6,8x8,10x10", model="voter", dims="6x6",
            replicas="20") + "\n".join(rows) + "\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "e")))
    assert any(CENSORED in w and "8x8" in w for w in warns)
    texts = [t.get_text() for _, f in figs for ax in f.axes for t in ax.texts]
    labels = [lab for _, f in figs for f_ax in f.axes for lab in f_ax.get_legend_handles_labels()[1]]
    labels += [t.get_text() for _, f in figs for lg in f.legends for t in lg.get_texts()]
    assert any(CENSORED in s for s in labels) and any(CENSORED in s for s in texts)
    for _, f in figs:
        plt.close(f)


def test_nan_gaps(tmp_path):
    rows = ["time,running_fraction,survival,autocorrelation_tw10_mean,autocorrelation_tw10_err,"
            "autocorrelation_tw10_mean_running"]
    for t in (0, 1, 2, 5, 10, 20, 50):
        v = "nan" if t < 10 else "0.5"
        rows.append(f"{t},1,1,{v},{v},{v}")
    (tmp_path / "t_timeseries.csv").write_text(
        cfg(task="ensemble", model="voter", dims="8x8") + "\n".join(rows) + "\n")
    t = find_outputs(str(tmp_path / "t")).get("timeseries_ensemble")
    figs = plot_timeseries_ensemble(t)
    assert figs
    found = False
    for _, f in figs:
        for ax in f.axes:
            for line in ax.lines:
                y = line.get_ydata(orig=True)
                if np.ma.isMaskedArray(y) and np.ma.getmaskarray(y).any():
                    found = True
                    assert np.ma.getmaskarray(y)[:4].all()
    assert found
    for _, f in figs:
        plt.close(f)


def test_no_provenance_renders(outputs):
    figs, _ = figures_for(find_outputs(outputs["noprov"]))
    assert figs
    for _, f in figs:
        plt.close(f)
    fig = plt.figure()
    caption(fig, Provenance())
    assert fig.texts == []
    plt.close(fig)


def test_states_network_skipped(tmp_path):
    (tmp_path / "g_states.csv").write_text(
        cfg(task="stationary", model="ising", graph="regular", n="4")
        + "# node,chain0\n0,1\n1,-1\n2,1\n3,1\n")
    out = find_outputs(str(tmp_path / "g"))
    figs, warns = figures_for(out)
    assert figs == []
    assert any("snapshots are drawn for lattices only" in w for w in warns)


def test_states_panels_limited(tmp_path):
    ncol = 9
    head = "# node," + ",".join(f"chain{i}" for i in range(ncol)) + "\n"
    body = "".join(f"{i}," + ",".join(str((i + j) % 2 * 2 - 1) for j in range(ncol)) + "\n"
                   for i in range(16))
    (tmp_path / "s_states.csv").write_text(cfg(task="stationary", model="ising", dims="4x4")
                                           + head + body)
    out = find_outputs(str(tmp_path / "s"))
    warns: list[str] = []
    figs = plot_states(out.get("states"), out.provenance, max_panels=6, warnings=warns)
    assert len(figs) == 1 and len([a for a in figs[0][1].axes if a.images]) == 6
    assert any("3" in w and "skipped" in w for w in warns)
    plt.close(figs[0][1])


def test_stationary_flagged_points(tmp_path):
    rows = ["temperature,abs_mean,abs_mean_err,tau,tau_abs,chains"]
    rows += ["2.0,0.9,0.01,1.0,1.0,1", "2.3,0.5,0.05,40.0,40.0,1"]
    (tmp_path / "s_sweep.csv").write_text(
        cfg(task="stationary", model="ising", dims="8x8", sweep="temperature=2.0,2.3",
            samples="100", blocks="20") + "\n".join(rows) + "\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "s")))
    assert any("temperature=2.3" in w and "autocorrelation time" in w for w in warns)
    labels = [lab for _, f in figs for ax in f.axes for lab in ax.get_legend_handles_labels()[1]]
    assert any("jackknife error" in s for s in labels)
    for _, f in figs:
        plt.close(f)


def test_domains_log_binned(outputs):
    t = find_outputs(outputs["maj"]).get("domains")
    figs = plot_domains(t)
    assert figs
    for _, f in figs:
        plt.close(f)


# ---------------------------------------------------------------- fix round 1

def _zero_length_bars(fig):
    """Finite error-bar segments with identical ends (drawn as zero-length bars)."""
    bad = 0
    for ax in fig.axes:
        for cont in ax.containers:
            if not hasattr(cont, "has_yerr") or not cont.has_yerr:
                continue
            for coll in cont[2]:
                for seg in coll.get_segments():
                    seg = np.asarray(seg)
                    if seg.shape == (2, 2) and np.isfinite(seg).all() and seg[0, 1] == seg[1, 1]:
                        bad += 1
    return bad


def test_undefined_errors_not_drawn_as_zero(tmp_path):
    (tmp_path / "e_sweep.csv").write_text(
        cfg(task="ensemble", sweep="dims=6x6,8x8", model="voter", dims="6x6", replicas="1")
        + "dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time,"
          "median_consensus_time,mean_consensus_time_lower_bound,mean_m,sd_m\n"
          "6x6,1,1,38.6,0,38.6,38.6,0.1,0\n8x8,1,1,50,0,50,50,0.2,0\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "e")))
    assert figs
    assert sum(_zero_length_bars(f) for _, f in figs) == 0
    assert any("error undefined" in w for w in warns)
    for _, f in figs:
        plt.close(f)


def test_one_finished_run_consensus_point(tmp_path):
    (tmp_path / "e_sweep.csv").write_text(
        cfg(task="ensemble", sweep="dims=6x6,8x8", model="voter", dims="6x6", replicas="20")
        + "dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time,"
          "median_consensus_time,mean_consensus_time_lower_bound\n"
          "6x6,20,1,38.6,5,38.6,38.6\n8x8,20,0.05,50,0,nan,900\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "e")))
    assert sum(_zero_length_bars(f) for _, f in figs) == 0
    assert any("error undefined" in w for w in warns)
    for _, f in figs:
        plt.close(f)


def test_log_axes_mask_nonpositive(tmp_path):
    rows = ["time,running_fraction,survival,rho_mean,rho_err,rho_mean_running"]
    for t, v in ((0, 0.5), (1, 0.4), (10, 0.1), (100, 0.0), (1000, 0.0)):
        rows.append(f"{t},1,1,{v},0.01,{v}")
    (tmp_path / "t_timeseries.csv").write_text(
        cfg(task="ensemble", model="voter", dims="8x8") + "\n".join(rows) + "\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "t")))
    assert any("not shown on log axes" in w and "2" in w for w in warns)
    titles = [ax.get_title() for _, f in figs for ax in f.axes]
    assert any("not shown" in s for s in titles)
    for _, f in figs:
        plt.close(f)


def test_caption_does_not_name_a_swept_size(outputs):
    figs, _ = figures_for(find_outputs(outputs["stat2"]))
    for _, f in figs:
        txt = " ".join(t.get_text() for t in f.texts)
        assert "8x8" not in txt and "dims swept" in txt
        plt.close(f)


def test_unknown_kind_warns(tmp_path):
    (tmp_path / "e_timeseries.csv").write_text(cfg(task="run", model="voter", dims="4x4")
                                               + "sweep,m\n0,1\n1,0.5\n")
    out = find_outputs(str(tmp_path / "e"))
    out.tables["bogus"] = out.tables["timeseries_run"]
    figs, warns = figures_for(out)
    assert any("no standard figure for bogus" in w for w in warns)
    for _, f in figs:
        plt.close(f)


def test_nonfinite_drops_counted(tmp_path):
    (tmp_path / "r_replicas.csv").write_text(
        cfg(task="ensemble", model="majority", dims="4x4")
        + "replica,reached_consensus,absorbed,stop_time,freeze_time\n"
          "0,1,1,10,9\n1,1,1,20,nan\n2,0,1,nan,5\n3,1,1,30,25\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "r")))
    assert any("freeze" in w and "2 replicas" in w and "left out" in w for w in warns)
    assert any("stop_time" in w and "left out" in w for w in warns)
    for _, f in figs:
        plt.close(f)


def test_duplicate_heatmap_cells_warn(tmp_path):
    (tmp_path / "s_sweep.csv").write_text(
        cfg(task="stationary", model="ising", dims="8x8", sweep="dims=8x8,12x12",
            sweep2="temperature=2.0,2.3", samples="100", blocks="2")
        + "dims,temperature,abs_mean,abs_mean_err,chains\n"
          "8x8,2.0,0.9,0.01,2\n8x8,2.0,0.8,0.01,2\n12x12,2.0,0.7,0.01,2\n12x12,2.3,0.5,0.01,2\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "s")))
    assert any("duplicate" in w for w in warns)
    for _, f in figs:
        plt.close(f)


def test_missing_column_skips_with_warning(tmp_path):
    (tmp_path / "c_clusters.csv").write_text(
        cfg(task="ensemble", model="deffuant", n="100") + "replica,position\n0,0.5\n")
    (tmp_path / "c_timeseries.csv").write_text(
        cfg(task="run", model="voter", dims="4x4") + "sweep,m\n0,1\n1,0.5\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "c")))
    assert any("clusters" in w and "size" in w for w in warns)
    assert any(n.startswith("timeseries") for n, _ in figs)
    for _, f in figs:
        plt.close(f)


def test_log_panel_with_nothing_positive_falls_back_to_linear(tmp_path):
    (tmp_path / "c_correlation.csv").write_text(
        cfg(task="run", model="voter", dims="8x8") + "sweep,r,G\n10,0,0\n10,1,-0.1\n10,2,0\n")
    (tmp_path / "c_timeseries.csv").write_text(
        cfg(task="run", model="voter", dims="8x8") + "sweep,rho,length\n0,0,0\n1,0,0\n2,0,0\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "c")))
    assert any("no values > 0" in w and "linear" in w and "G" in w for w in warns)
    assert any("no values > 0" in w and "rho" in w for w in warns)
    for n, f in figs:
        save(f, tmp_path / "o", n)      # must not raise


def test_survival_frozen_runs_never_reach_consensus(outputs):
    from odspy.plots.replicas import plot_replicas, survival_curve
    t = find_outputs(outputs["maj"]).get("replicas")
    stop, done = t["stop_time"], t["reached_consensus"] > 0.5
    frozen = (t["absorbed"] > 0.5) & ~done
    assert frozen.any()                       # the fixture has frozen runs
    R = len(t)
    ts, ss = survival_curve(stop, t["reached_consensus"], t["absorbed"])
    # the driver's survival: S(t) = 1 - (runs that reached consensus by t) / R
    for tt, s in zip(ts, ss):
        assert s == pytest.approx(1 - np.sum(done & (stop <= tt)) / R)
    assert ss[-1] == pytest.approx(frozen.sum() / R)
    warns = []
    figs = dict(plot_replicas(t, warnings=warns))
    fig = figs["replicas_stoptime"]
    labels = [txt.get_text() for ax in fig.axes if ax.get_legend() for txt in ax.get_legend().get_texts()]
    assert any("frozen without consensus" in s and "never" in s for s in labels)
    assert not any("censored" in s for s in labels)     # no run hit the cap
    for f in figs.values():
        plt.close(f)


def test_survival_capped_runs_censored():
    from odspy.plots.replicas import survival_curve
    stop = np.array([1.0, 2.0, 3.0, 10.0, 10.0])
    reached = np.array([1, 1, 0, 0, 1])
    absorbed = np.array([1, 1, 1, 0, 1])    # run 2 froze at t=3, run 3 hit the cap at 10
    ts, ss = survival_curve(stop, reached, absorbed)
    # at risk: 5, 4 after t=1, 3 after t=2 (the frozen run stays at risk), event at 10 with
    # 3 at risk (the capped run is censored at 10, after the event)
    assert list(ts) == [0, 1, 2, 10]
    assert ss == pytest.approx([1, 0.8, 0.6, 0.6 * (1 - 1 / 3)])


def test_sweep_lower_bound_note_describes_the_driver(tmp_path):
    (tmp_path / "e_sweep.csv").write_text(
        cfg(task="ensemble", sweep="dims=6x6,8x8", model="voter", dims="6x6", replicas="20")
        + "dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time,"
          "median_consensus_time,mean_consensus_time_lower_bound\n"
          "6x6,20,1,38.6,5,38.6,38.6\n8x8,20,0.5,50,3,nan,900\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "e")))
    text = " ".join(w for w in warns if CENSORED in w)
    assert "at the cap" in text and "every run's stop time" not in text
    for _, f in figs:
        plt.close(f)


def _bar_lengths(fig):
    out = []
    for ax in fig.axes:
        for c in ax.collections:
            for seg in getattr(c, "get_segments", lambda: [])():
                if len(seg) == 2:
                    out.append(float(np.hypot(*(seg[1] - seg[0]))))
    return out


def test_binomial_fraction_at_one_drawn_with_wilson_interval(tmp_path):
    (tmp_path / "r_replicas.csv").write_text(
        cfg(task="ensemble", model="majority", dims="4x4")
        + "replica,reached_consensus,absorbed,stop_time,stripes\n"
        + "".join(f"{i},0,1,10,1\n" for i in range(10)))
    figs, warns = figures_for(find_outputs(str(tmp_path / "r")))
    fig = dict(figs)["replicas_stripes"]
    labels = [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    bars = _bar_lengths(fig)
    for _, f in figs:
        plt.close(f)
    assert any("Wilson" in s for s in labels)
    assert bars and max(bars) > 0.1              # 95% Wilson lower end ~0.72 for 10 of 10


def test_fixation_at_one_drawn_with_wilson_interval(tmp_path):
    (tmp_path / "f_fixation.csv").write_text(
        cfg(task="ensemble", model="voter", dims="4x4") + "opinion,replicas,probability\n1,20,1\n")
    figs, warns = figures_for(find_outputs(str(tmp_path / "f")))
    fig = dict(figs)["fixation"]
    labels = [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
    bars = _bar_lengths(fig)
    for _, f in figs:
        plt.close(f)
    assert any("Wilson" in s for s in labels) and bars and max(bars) > 0.1
