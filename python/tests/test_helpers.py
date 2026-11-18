import math

import numpy as np
import pytest

from odspy.errors import InputError
from odspy.helpers import (Recovered, consensus_time_vs_size, error_bar_uncertainty,
                           log_growth_vs_size, peak_vs_size, slope_at_crossing_vs_size,
                           stationary_caveats, stationary_flags, summarise_replicas,
                           summarise_samples)
from odspy.io import Provenance, Table, find_outputs, read_table
from odspy.sizes import SizeCurve

CENSORED = "censored: mean biased low"
HEADER = "# ODSP 0.1.0, git abc, written 2026-01-01T00:00:00+00:00\n"


def write_cfg(prefix, **cfg):
    lines = ["# configuration:"] + [f"#   {k} = {v}" for k, v in cfg.items()]
    return HEADER + "\n".join(lines) + "\n"


def make_ensemble_sweep(tmp_path, fractions=(1.0, 0.8, 1.0, 1.0)):
    prefix = tmp_path / "e"
    rows = ["dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time"]
    for k, f in zip((6, 8, 10, 12), fractions):
        rows.append(f"{k}x{k},20,{f},{k ** 2 * 2.0},{k ** 2 * 0.5}")
    (tmp_path / "e_sweep.csv").write_text(
        write_cfg(prefix, task="ensemble", sweep="dims=6x6,8x8,10x10,12x12", model="voter",
                  dims="6x6", replicas="20") + "\n".join(rows) + "\n")
    return str(prefix)


def test_consensus_time_censored_excluded(tmp_path):
    r = consensus_time_vs_size([make_ensemble_sweep(tmp_path)], model="line")
    assert list(r.used) == [True, False, True, True]
    text = " ".join(r.warnings)
    assert CENSORED in text and "size 8" in text


def test_consensus_time_all_finished_has_no_censoring_warning(tmp_path):
    r = consensus_time_vs_size([make_ensemble_sweep(tmp_path, (1, 1, 1, 1))], model="line")
    assert r.used.all() and not any(CENSORED in w for w in r.warnings)


def test_consensus_time_bad_options(tmp_path):
    with pytest.raises(InputError):
        consensus_time_vs_size([make_ensemble_sweep(tmp_path)], censored="drop")
    with pytest.raises(InputError):
        consensus_time_vs_size([])


def test_consensus_time_median_needs_point_files(tmp_path):
    with pytest.raises(InputError, match="replicas"):
        consensus_time_vs_size([make_ensemble_sweep(tmp_path)], censored="median")


def test_consensus_time_median(outputs):
    r = consensus_time_vs_size([outputs["ens_sweep"]], censored="median", n_boot=200)
    out = find_outputs(outputs["ens_sweep"])
    expect = []
    for i in range(3):
        rt = out.points[i].get("replicas")
        done = np.sort(rt["stop_time"][rt["reached_consensus"] > 0.5])
        expect.append(float(done[(len(rt["stop_time"]) + 1) // 2 - 1]))   # driver: ceil(R/2)-th

    assert np.allclose(r.y, expect)
    assert r.used.all() and np.all(r.yerr > 0)
    assert list(r.x) == [6, 8, 10]


def test_consensus_time_median_unfinished_half_excluded(tmp_path):
    prefix = make_ensemble_sweep(tmp_path, (1.0, 0.25, 1.0, 1.0))
    for i, k in enumerate((6, 8, 10, 12)):
        done = 4 if i == 1 else 20
        rows = ["replica,reached_consensus,absorbed,stop_time"]
        rows += [f"{j},{int(j < done)},{int(j < done)},{10 * (j + 1) + i}" for j in range(20)]
        (tmp_path / f"e_point{i}_replicas.csv").write_text(HEADER + "\n".join(rows) + "\n")
    r = consensus_time_vs_size([prefix], censored="median", model="line", n_boot=50)
    assert list(r.used) == [True, False, True, True]
    assert any("median undetermined" in w and "size 8" in w for w in r.warnings), r.warnings


def test_consensus_time_median_even_r_exactly_half_finished(tmp_path):
    """R = 4 with 2 finished: the driver's median (2nd smallest) exists; the point is kept."""
    rows = ["dims,replicas,consensus_fraction,mean_consensus_time,sd_consensus_time"]
    for k in (6, 8, 10, 12):
        rows.append(f"{k}x{k},4,{0.5 if k == 8 else 1.0},1,1")
    (tmp_path / "h_sweep.csv").write_text(
        write_cfg(None, task="ensemble", sweep="dims=6x6,8x8,10x10,12x12", model="voter",
                  dims="6x6", replicas="4") + "\n".join(rows) + "\n")
    expect = []
    for i, k in enumerate((6, 8, 10, 12)):
        stops = [10 * k, 20 * k, 30 * k, 40 * k]
        reached = [1, 1, 0, 0] if k == 8 else [1, 1, 1, 1]
        r = ["replica,reached_consensus,absorbed,stop_time"]
        r += [f"{j},{a},{a},{t}" for j, (a, t) in enumerate(zip(reached, stops))]
        (tmp_path / f"h_point{i}_replicas.csv").write_text(HEADER + "\n".join(r) + "\n")
        expect.append(float(sorted(t for a, t in zip(reached, stops) if a)[1]))
    r = consensus_time_vs_size([str(tmp_path / "h")], censored="median", model="line", n_boot=50)
    assert r.used.all() and np.all(np.isfinite(r.y))
    assert list(r.y) == expect


def curve(size, x, y, err):
    return SizeCurve(size, "L", np.asarray(x, float), np.asarray(y, float),
                     np.full(len(x), err), "temperature", "chi", "syn")


def test_peak_vs_size_exponent():
    rng = np.random.default_rng(1)
    x = np.linspace(2.0, 2.6, 13)
    curves = []
    for L in (8, 12, 16, 24, 32):
        peak = 0.5 * L ** 1.75
        y = peak * (1 - ((x - 2.3) / 0.4) ** 2) + rng.normal(0, 0.002 * peak, x.size)
        curves.append(curve(L, x, y, 0.002 * peak))
    r = peak_vs_size(curves, n_boot=200, seed=3)
    assert abs(r.params["b"] - 1.75) < max(3 * r.errors["b"], 0.02)
    assert r.errors["b"] < 0.05


def test_peak_vs_size_edge_peak_excluded():
    x = np.linspace(2.0, 2.6, 13)
    curves = [curve(L, x, 0.5 * L ** 1.75 * (1 - ((x - 2.3) / 0.4) ** 2), 0.01)
              for L in (8, 12, 16, 24)]
    curves.append(curve(32, x, 0.5 * 32 ** 1.75 * (1 - ((x - 2.7) / 0.4) ** 2), 0.01))
    r = peak_vs_size(curves, n_boot=50)
    assert list(r.used) == [True] * 4 + [False]
    assert any("peak at the edge of the scanned range for size L=32" in w
               and "lower bound" in w for w in r.warnings)
    assert abs(r.params["b"] - 1.75) < 0.05


def test_peak_vs_size_reports_partial_bracketing():
    x = np.linspace(2.0, 2.6, 13)
    # a flat top: the noise moves the maximum to the edge in some bootstrap draws
    curves = [curve(L, x, 0.5 * L * (1 - 0.02 * ((x - 2.3) / 0.3) ** 2), 0.05)
              for L in (8, 12, 16, 24)]
    r = peak_vs_size(curves, n_boot=200, seed=2)
    assert any("bootstrap draws" in w and "bracketed" in w for w in r.warnings)


def test_peak_vs_size_needs_errors_and_points():
    x = np.linspace(0, 1, 7)
    no_err = SizeCurve(4, "L", x, x, None, "t", "c", "s")
    with pytest.raises(InputError, match="error"):
        peak_vs_size([no_err, no_err, no_err])
    with pytest.raises(InputError, match="3"):
        peak_vs_size([curve(4, [0, 1], [1, 2], 0.1)] * 3)


def test_slope_at_crossing_exponent():
    rng = np.random.default_rng(2)
    x = np.linspace(2.1, 2.5, 9)
    curves = [curve(L, x, 0.5 - L ** 1.0 * 0.3 * (x - 2.3) + rng.normal(0, 0.002, x.size), 0.002)
              for L in (8, 16, 32, 64)]
    r = slope_at_crossing_vs_size(curves, 2.3, n_boot=200, seed=1)
    assert abs(r.params["b"] - 1.0) < max(3 * r.errors["b"], 0.05)
    with pytest.raises(InputError, match="x0"):
        slope_at_crossing_vs_size(curves, 9.0)


def test_slope_one_sided_points_excluded():
    x_ok = np.linspace(2.1, 2.5, 9)
    x_bad = np.array([2.0, 2.1, 2.2, 2.25, 2.9])   # nearest 3 to 2.28 all below it
    curves = [curve(L, x_ok, 0.5 - L * 0.3 * (x_ok - 2.3), 0.002) for L in (8, 16, 32, 64)]
    curves.append(curve(128, x_bad, 0.5 - 128 * 0.3 * (x_bad - 2.3), 0.002))
    r = slope_at_crossing_vs_size(curves, 2.28, n_boot=50)
    assert list(r.used) == [True] * 4 + [False]
    assert any("both sides" in w and "L=128" in w for w in r.warnings)


def test_log_growth_bad_x0():
    x = np.linspace(2.0, 2.6, 7)
    curves = [curve(L, x, np.log(L) + 0 * x, 0.01) for L in (8, 16, 32)]
    with pytest.raises(InputError, match="x0"):
        log_growth_vs_size(curves, x0="abc")


def test_log_growth():
    rng = np.random.default_rng(3)
    x = np.linspace(2.0, 2.6, 7)
    curves = [curve(L, x, 0.3 + 0.7 * math.log(L) + rng.normal(0, 0.01, x.size), 0.01)
              for L in (8, 16, 32, 64)]
    r = log_growth_vs_size(curves, x0=2.3, n_boot=200)
    assert abs(r.params["b"] - 0.7) < max(3 * r.errors["b"], 0.05)
    r2 = log_growth_vs_size(([8, 16, 32, 64], [0.3 + 0.7 * math.log(L) for L in (8, 16, 32, 64)],
                             [0.01] * 4))
    assert abs(r2.params["b"] - 0.7) < 1e-6
    with pytest.raises(InputError, match="x0"):
        log_growth_vs_size(curves)


def sweep_table(rows, **cfg):
    cols = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    prov = Provenance(config={k: str(v) for k, v in cfg.items()})
    from pathlib import Path
    return Table(Path("syn_sweep.csv"), "sweep", prov, cols, list(cols))


def test_error_bar_uncertainty():
    assert error_bar_uncertainty(10) == pytest.approx(1 / math.sqrt(18))
    assert math.isnan(error_bar_uncertainty(1)) and math.isnan(error_bar_uncertainty(0))


def test_stationary_caveats():
    t = sweep_table([{"temperature": 2.0, "tau_abs": 1.0, "chains": 1},
                     {"temperature": 2.3, "tau_abs": 40.0, "chains": 1}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.3")
    caveats = stationary_caveats(t)
    short = [c for c in caveats if "not much longer" in c]
    assert len(short) == 1 and "at 1 of 2 points (temperature=2.3)" in short[0]
    assert short[0].startswith("syn_sweep.csv: ") and "tau_abs up to 40" in short[0]
    notes = [c for c in caveats if "uncertain" in c]
    assert notes == ["syn_sweep.csv: error bars from 20 jackknife blocks are themselves "
                     "uncertain by about +-16%"]


def test_stationary_caveats_several_chains_no_block_caveats():
    # Several chains: errors are leave-one-chain-out, independent of the jackknife block
    # length, so the short-block caveat does not apply (blocks of 50 < 10 tau_abs = 400 at
    # T = 2.3). The sign note for several chains needs tau(x) > 2 tau(|x|): the T = 2.3 row
    # (tau = tau_abs = 40, chains of 1000 < 50 tau) is not flagged, since x there need not
    # change sign at all; the T = 2.0 row (tau = 15 tau_abs, chains of 1000 < 50 tau) is.
    t = sweep_table([{"temperature": 2.0, "tau": 30.0, "tau_abs": 2.0, "chains": 10},
                     {"temperature": 2.3, "tau": 40.0, "tau_abs": 40.0, "chains": 10}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.3")
    caveats = stationary_caveats(t)
    assert not any("not much longer" in c for c in caveats), caveats
    sign = [c for c in caveats if "changes sign" in c]
    assert len(sign) == 1 and "(temperature=2)" in sign[0], sign
    assert caveats[-1] == ("syn_sweep.csv: error bars from 10 chains are themselves uncertain "
                           "by about +-24% (more --replicas narrow them)")
    flags = stationary_flags(t)
    assert not flags["short"].any() and list(flags["sign"]) == [True, False]


def test_stationary_flags_sign_several_chains_needs_samples():
    # Without `samples` in the configuration the chain length is unknown: no several-chain
    # sign flag (rather than flagging every point against a length of 0).
    t = sweep_table([{"temperature": 2.0, "tau": 30.0, "tau_abs": 2.0, "chains": 8}],
                    blocks=20, sweep="temperature=2.0")
    assert not stationary_flags(t)["sign"].any()
    assert not any("changes sign" in c for c in stationary_caveats(t))


def test_stationary_flags_per_point_chains():
    # A merged table may mix one-chain and several-chain points: flag only the former.
    t = sweep_table([{"temperature": 2.0, "tau": 40.0, "tau_abs": 40.0, "chains": 1},
                     {"temperature": 2.3, "tau": 40.0, "tau_abs": 40.0, "chains": 4}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.3")
    assert list(stationary_flags(t)["short"]) == [True, False]
    short = [c for c in stationary_caveats(t) if "not much longer" in c]
    assert len(short) == 1 and "(temperature=2)" in short[0]


def test_stationary_caveats_one_line_per_kind():
    rows = [{"temperature": 2.0 + 0.1 * i, "tau_abs": 40.0, "chains": 1} for i in range(6)]
    t = sweep_table(rows, samples=1000, blocks=20, sweep="temperature=2.0:2.5:0.1")
    short = [c for c in stationary_caveats(t) if "not much longer" in c]
    assert len(short) == 1 and "at all 6 points" in short[0]


def test_stationary_caveats_rare_sign_changes():
    # blocks of 50 samples: long against tau_abs = 2, short against tau = 30
    t = sweep_table([{"temperature": 2.0, "tau": 30.0, "tau_abs": 2.0, "chains": 1},
                     {"temperature": 2.5, "tau": 2.0, "tau_abs": 1.0, "chains": 1}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.5")
    caveats = stationary_caveats(t)
    assert not any("not much longer" in c for c in caveats)
    sign = [c for c in caveats if "changes sign only rarely" in c]
    assert len(sign) == 1 and "(temperature=2)" in sign[0] and "plain mean" in sign[0]
    assert "|mean|, abs_susceptibility and Binder are fine" in sign[0]


def test_stationary_caveats_rare_sign_changes_several_chains():
    # Several chains started alike that rarely flip are biased the same way, which the
    # leave-one-chain-out errors do not show: flagged when each chain is shorter than
    # 50 tau(x) (the per-chain tau estimate saturates near chain length / 15 when too
    # short), whatever the block length; not flagged for chains long against tau(x).
    t = sweep_table([{"temperature": 2.0, "tau": 30.0, "tau_abs": 2.0, "chains": 8},
                     {"temperature": 2.2, "tau": 15.0, "tau_abs": 2.0, "chains": 8},
                     {"temperature": 2.5, "tau": 2.0, "tau_abs": 1.0, "chains": 8}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.2,2.5")
    assert list(stationary_flags(t)["sign"]) == [True, False, False]
    caveats = stationary_caveats(t)
    sign = [c for c in caveats if "changes sign only rarely" in c]
    assert len(sign) == 1 and "(temperature=2)" in sign[0], caveats
    assert "within a chain" in sign[0]
    assert "the plain mean and N var(x), and their errors, may be biased" in sign[0]
    assert "|mean|, abs_susceptibility and Binder are fine" in sign[0]
    assert not any("not much longer" in c for c in caveats)


def test_stationary_flags_sign_mixed_chains():
    # One-chain points keep the block rule (block 50 < 10 tau = 300), several-chain points
    # the chain-length rule (1000 >= 50 tau = 1000 at tau = 20: not flagged).
    t = sweep_table([{"temperature": 2.0, "tau": 30.0, "tau_abs": 2.0, "chains": 1},
                     {"temperature": 2.1, "tau": 20.0, "tau_abs": 2.0, "chains": 4},
                     {"temperature": 2.2, "tau": 21.0, "tau_abs": 2.0, "chains": 4}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.1,2.2")
    assert list(stationary_flags(t)["sign"]) == [True, False, True]
    sign = [c for c in stationary_caveats(t) if "changes sign only rarely" in c]
    assert len(sign) == 2, sign
    assert "(temperature=2)" in sign[0] and "within a chain" not in sign[0]
    assert "(temperature=2.2)" in sign[1] and "within a chain" in sign[1]


def test_stationary_caveats_chains_disagree():
    t = sweep_table([{"temperature": 2.0, "tau_abs": 1.0, "chains": 4, "abs_mean_err": 0.05,
                      "abs_mean_err_within": 0.01, "binder_err": 0.01, "binder_err_within": 0.01},
                     {"temperature": 2.5, "tau_abs": 1.0, "chains": 4, "abs_mean_err": 0.01,
                      "abs_mean_err_within": 0.01, "binder_err": 0.01, "binder_err_within": 0.01}],
                    samples=1000, blocks=20, sweep="temperature=2.0,2.5")
    bad = [c for c in stationary_caveats(t) if "disagree" in c]
    assert len(bad) == 1 and "(temperature=2)" in bad[0]


def test_stationary_caveats_single_chain_uses_blocks():
    t = sweep_table([{"temperature": 2.0, "tau_abs": 1.0, "chains": 1}], samples=1000,
                    blocks=20, sweep="temperature=2.0")
    assert stationary_caveats(t) == ["syn_sweep.csv: error bars from 20 jackknife blocks are "
                                     "themselves uncertain by about +-16%"]


@pytest.mark.parametrize("variant", ["one", "chains"])
def test_summarise_samples_matches_driver(stat_point, variant):
    sweep = find_outputs(stat_point[variant]["sweep"]).get("sweep")
    samples = find_outputs(stat_point[variant]["samples"]).get("samples")
    config = sweep.provenance.config
    assert (variant == "chains") == (int(sweep["chains"][0]) > 1)
    rec = summarise_samples(samples, config)
    assert isinstance(rec, Recovered)
    for name in ("mean", "abs_mean", "susceptibility", "abs_susceptibility", "binder"):
        assert rec.values[name][0] == pytest.approx(float(sweep[name][0]), abs=1e-9, rel=0)
        assert rec.values[name][1] == pytest.approx(float(sweep[name + "_err"][0]),
                                                    abs=1e-9, rel=0)
        assert rec.values[name][1] > 0
    for name in ("tau", "tau_abs"):
        assert rec.values[name][0] == pytest.approx(float(sweep[name][0]), abs=1e-9, rel=0)
    assert "tau" not in rec.missing and "xi" in rec.missing and "specific_heat" in rec.missing
    assert ("chains" in rec.method) == (variant == "chains")


def test_summarise_samples_bad_input(tmp_path):
    t = sweep_table([{"temperature": 2.0}])
    with pytest.raises(InputError, match="sample"):
        summarise_samples(t, {"dims": "4x4"})


def replicas_table(tmp_path, reached, stop, extra=None):
    cols = ["replica,reached_consensus,absorbed,stop_time,rho"]
    for j, (r, s) in enumerate(zip(reached, stop)):
        cols.append(f"{j},{r},{r},{s},{0.0 if r else 0.1}")
    p = tmp_path / "x_replicas.csv"
    p.write_text(HEADER + "\n".join(cols) + "\n")
    return read_table(p)


def test_summarise_replicas_censored(tmp_path):
    t = replicas_table(tmp_path, [1, 1, 1, 0], [10, 20, 30, 100])
    rec = summarise_replicas(t)
    mean, err = rec.values["mean_consensus_time"]
    assert mean == pytest.approx(20.0)
    assert err == pytest.approx(np.std([10, 20, 30]) / math.sqrt(3))
    assert any(CENSORED in m for m in rec.missing)
    assert rec.values["consensus_fraction"][0] == 0.75
    assert rec.values["consensus_fraction"][1] == pytest.approx(math.sqrt(0.75 * 0.25 / 4))
    assert rec.values["mean_rho"][0] == pytest.approx(0.025)
    assert any("fixation" in m for m in rec.missing)


def test_summarise_replicas_matches_driver(outputs):
    out = find_outputs(outputs["ens"])
    rec = summarise_replicas(out.get("replicas"))
    sw = find_outputs(outputs["ens_sweep"])
    rec2 = summarise_replicas(sw.points[1].get("replicas"))
    row = sw.get("sweep")
    assert rec2.values["mean_consensus_time"][0] == pytest.approx(float(row["mean_consensus_time"][1]))
    assert rec2.values["sd_consensus_time"][0] == pytest.approx(float(row["sd_consensus_time"][1]))
    assert rec2.values["median_consensus_time"][0] == pytest.approx(float(row["median_consensus_time"][1]))
    assert rec2.values["mean_consensus_time_lower_bound"][0] == pytest.approx(
        float(row["mean_consensus_time_lower_bound"][1]))
    assert not any(CENSORED in m for m in rec2.missing)
    assert rec.method


def test_summarise_replicas_lower_bound_matches_driver_frozen_and_capped(outputs):
    sw = find_outputs(outputs["mjs"])
    row = sw.get("sweep")
    seen = set()
    for i, pt in sw.points.items():
        t = pt.get("replicas")
        done = t["reached_consensus"] > 0.5
        frozen = ~done & (t["absorbed"] > 0.5)
        capped = ~done & ~(t["absorbed"] > 0.5)
        rec = summarise_replicas(t)
        assert rec.values["mean_consensus_time_lower_bound"][0] == pytest.approx(
            float(row["mean_consensus_time_lower_bound"][i]))
        text = " ".join(rec.missing)
        assert ("froze without consensus" in text) == bool(frozen.any())
        assert ("hit the cap" in text) == bool(capped.any())
        seen |= {"frozen"} if frozen.any() else set()
        seen |= {"capped"} if capped.any() else set()
    assert seen == {"frozen", "capped"}      # the fixture exercises both


def test_summarise_replicas_frozen_only_matches_driver(outputs):
    t = find_outputs(outputs["maj"]).get("replicas")
    rec = summarise_replicas(t)
    done = t["reached_consensus"] > 0.5
    # the driver: unfinished runs count as the cap (--steps 100000), 60017.1 printed
    expect = (t["stop_time"][done].sum() + 100000 * (~done).sum()) / len(t)
    assert rec.values["mean_consensus_time_lower_bound"][0] == pytest.approx(expect)
    text = " ".join(rec.missing)
    assert "froze without consensus" in text and "infinite" in text
    assert "hit the cap" not in text


def test_summarise_replicas_lower_bound_needs_the_cap(tmp_path):
    t = replicas_table(tmp_path, [1, 1, 1, 0], [10, 20, 30, 100])   # no configuration
    rec = summarise_replicas(t)
    assert math.isnan(rec.values["mean_consensus_time_lower_bound"][0])
    assert any("--steps" in m for m in rec.missing)


def test_summarise_replicas_wrong_table(tmp_path):
    with pytest.raises(InputError, match="reached_consensus"):
        summarise_replicas(sweep_table([{"temperature": 2.0}]))


def test_coarsening_fit_uses_trajectories(outputs):
    from odspy.helpers import coarsening_fit
    r = coarsening_fit(outputs["ens"], "length", xrange=(2, 100))
    assert r.errors["b"] > 0
    assert not any("not trustworthy" in w for w in r.warnings)
    with pytest.raises(InputError, match="time series"):
        coarsening_fit(outputs["stat"], "length")


def test_refusal_names_the_censored_exclusions(tmp_path):
    with pytest.raises(InputError) as e:
        consensus_time_vs_size([make_ensemble_sweep(tmp_path, (1.0, 0.8, 0.9, 1.0))])
    msg = str(e.value)
    assert "have 2" in msg and CENSORED in msg and "size 8" in msg and "size 10" in msg


def test_refusal_names_the_undetermined_medians(outputs):
    # mjs: the 12x12 point finished in 45% of the runs, so its median is undetermined
    with pytest.raises(InputError) as e:
        consensus_time_vs_size([outputs["mjs"]], censored="median", model="line", n_boot=20)
    msg = str(e.value)
    assert "have 1" in msg and "median undetermined" in msg


def test_fraction_at_zero_or_one_has_no_zero_error(outputs):
    from odspy.report import recovered
    rec = summarise_replicas(find_outputs(outputs["maj"]).get("replicas"))   # all absorbed
    p, e = rec.values["absorbed_fraction"]
    assert p == 1 and math.isnan(e)
    text = "\n".join(recovered(rec, "Ensemble summary"))
    assert "absorbed_fraction = 1\n" in text + "\n" and "± 0\n" not in text + "\n"
    assert any("Wilson" in n and "absorbed_fraction" in n for n in rec.notes)
    assert "Wilson interval [0.839, 1]" in text            # R = 20


def test_wilson_interval():
    from odspy.helpers import wilson_interval
    lo, hi = wilson_interval(20, 20)
    assert hi == 1 and lo == pytest.approx(0.8389, abs=1e-3)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0 and hi == pytest.approx(0.1611, abs=1e-3)


def test_summarise_samples_picks_the_named_observable():
    from pathlib import Path
    rng = np.random.default_rng(1)
    n = 200
    cols = {"sample": np.arange(n, dtype=float), "chain": np.zeros(n), "sweep": np.arange(n) + 1.0,
            "m": rng.normal(0.5, 0.1, n), "energy": rng.normal(-1.5, 0.1, n)}
    t = Table(Path("two_samples.csv"), "samples", Provenance(), cols, list(cols))
    with pytest.raises(InputError, match="several observable columns"):
        summarise_samples(t, {"dims": "10x10"})
    by_config = summarise_samples(t, {"dims": "10x10", "observable": "m"})
    assert by_config.values["mean"][0] == pytest.approx(cols["m"].mean())
    by_arg = summarise_samples(t, {"dims": "10x10", "observable": "m"}, observable="energy")
    assert by_arg.values["mean"][0] == pytest.approx(cols["energy"].mean())
    with pytest.raises(InputError, match="no observable column 'rho'"):
        summarise_samples(t, {"dims": "10x10"}, observable="rho")
