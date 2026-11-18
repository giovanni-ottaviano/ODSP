import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from odspy import cli
from odspy.cli import main


def run(argv, capsys):
    code = main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_plot_all(outputs, tmp_path, capsys):
    code, out, err = run([outputs["ens"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert list(tmp_path.glob("*.pdf")) and list(tmp_path.glob("*.png"))
    report = (tmp_path / "ens_report.txt").read_text()
    assert report == out
    assert "ens_replicas.csv" in report
    assert "trajectories" in report
    assert "mean_consensus_time" in report
    for p in tmp_path.glob("*.pdf"):
        assert str(p) in report


def test_plot_is_explicit_subcommand(outputs, tmp_path, capsys):
    code, out, err = run(["plot", outputs["run"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert (tmp_path / "run_report.txt").exists()


def test_plot_without_trajectories_says_so(outputs, tmp_path, capsys):
    code, out, err = run([outputs["ens_sweep"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "Warnings:" in out


def test_plot_stationary_run_reports_samples(outputs, tmp_path, capsys):
    code, out, err = run([outputs["stat"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "not recoverable from the files" in out or "binder" in out


def test_crossings_command(outputs, tmp_path, capsys):
    code, out, err = run(["crossings", outputs["stat2"], "--column", "binder", "--boot", "50",
                          "--out", str(tmp_path)], capsys)
    assert code == 0, err
    pairs = [ln for ln in out.splitlines() if ln.startswith("  L=") and " vs " in ln]
    assert len(pairs) == 2                                    # 8 vs 12, 12 vs 16
    assert sum("x = " in ln and "±" in ln for ln in out.splitlines()) >= 2
    assert (tmp_path / "stat2_crossings_binder.pdf").exists()
    assert (tmp_path / "stat2_crossings_binder_report.txt").exists()


def test_fit_command_timeseries(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["ens"], "--x", "time", "--y", "length", "--range", "2:40",
                          "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "bootstrap over replicas" in out
    assert list(tmp_path.glob("*fit*.pdf"))


def test_fit_consensus_time_censored_median(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["ens_sweep"], "--y", "mean_consensus_time",
                          "--censored", "median", "--boot", "50", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "power fit" in out
    code, out, err = run(["fit", outputs["ens_sweep"], "--y", "mean_consensus_time",
                          "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "power fit" in out


def test_fit_sweep_table_column(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["cont"], "--x", "epsilon", "--y", "mean_consensus_time",
                          "--model", "line", "--unweighted", "--out", str(tmp_path)], capsys)
    # two points only: reported either as a fit or as an input error, never a traceback
    assert code in (0, 2)


def test_collapse_command(outputs, tmp_path, capsys):
    code, out, err = run(["collapse", outputs["stat2"], "--column", "binder", "--xc", "2.269",
                          "--nu", "1", "--a", "0", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "S = " in out
    assert (tmp_path / "stat2_collapse_binder.pdf").exists()


def test_collapse_search(outputs, tmp_path, capsys):
    code, out, err = run(["collapse", outputs["stat2"], "--column", "binder", "--xc", "2.269",
                          "--nu", "1", "--search", "--fix-a", "--boot", "20",
                          "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "S = " in out


def test_unknown_prefix_exit_2(tmp_path, capsys):
    code, out, err = run([str(tmp_path / "nothing"), "--out", str(tmp_path)], capsys)
    assert code == 2
    assert "odspy: error: no ODSP outputs found for prefix" in err
    assert "Traceback" not in err


def test_bad_column_exit_2(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["ens"], "--x", "time", "--y", "nope",
                          "--out", str(tmp_path)], capsys)
    assert code == 2
    assert "odspy: error:" in err and "nope" in err and "ens_" in err
    code, out, err = run(["fit", outputs["ens_sweep"], "--x", "replicas", "--y", "nope",
                          "--out", str(tmp_path)], capsys)
    assert code == 2 and "nope" in err and "sweep" in err


def test_crossings_bad_column_exit_2(outputs, tmp_path, capsys):
    code, out, err = run(["crossings", outputs["stat2"], "--column", "nope", "--out", str(tmp_path)], capsys)
    assert code == 2 and "nope" in err


def test_usage_error_exit_2(capsys):
    assert main(["fit"]) == 2
    assert main(["fit", "x", "--y", "a"]) == 2   # --x missing, not a consensus fit
    capsys.readouterr()


def odsp_run(odsp_bin, tmp_path, name, args):
    """Run bin/odsp with `args` into tmp_path/name; returns the prefix."""
    prefix = tmp_path / name
    env = dict(os.environ, ODSP_MAX_THREADS="8")
    proc = subprocess.run([str(odsp_bin), *args.split(), "--csv", str(prefix)], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return str(prefix)


@pytest.mark.parametrize("name,args", [
    ("t0", "--model voter --dims 8x8 --init up --task ensemble --replicas 10 --steps 100 "
           "--log-times 3 --observables rho,abs_m --save-trajectories true"),
    ("r0", "--model voter --init up --steps 20 --observables rho"),
])
def test_plot_log_observable_all_nonpositive(odsp_bin, tmp_path, capsys, name, args):
    prefix = odsp_run(odsp_bin, tmp_path, name, args)
    code, out, err = run([prefix, "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert "Traceback" not in err
    assert "no values > 0" in out and "linear" in out
    assert list((tmp_path / "o").glob("*.pdf"))


def _bad_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    def broken(path, **kw):
        if str(path).endswith(".png"):      # the pdf is written first, then must be removed
            raise ValueError("Data cannot be log-scaled because all values are <= 0.")
        type(fig).savefig(fig, path, **kw)
    fig.savefig = broken
    return fig


def test_plot_one_bad_figure_is_skipped_and_reported(outputs, tmp_path, capsys, monkeypatch):
    def fake(o, per_point=False, warnings=None):
        yield "bad", _bad_figure()
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        yield "good", fig
    monkeypatch.setattr(cli, "iter_figures", fake)
    code, out, err = run([outputs["run"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert (tmp_path / "run_good.pdf").exists()
    assert not (tmp_path / "run_bad.png").exists() and not (tmp_path / "run_bad.pdf").exists()
    assert "run_bad" in out and "not saved" in out and "ValueError" in out
    assert plt.get_fignums() == []


def test_internal_error_exit_3(outputs, tmp_path, capsys, monkeypatch):
    def boom(args):
        raise RuntimeError("boom")
    monkeypatch.setitem(cli.HANDLERS, "plot", boom)
    code, out, err = run([outputs["run"], "--out", str(tmp_path)], capsys)
    assert code == 3
    assert "odspy: internal error: RuntimeError: boom (please report)" in err
    assert "Traceback" not in err


def test_plot_per_point_keeps_few_figures_open(odsp_bin, tmp_path, capsys, monkeypatch):
    import warnings
    prefix = odsp_run(odsp_bin, tmp_path, "pp",
                      "--model voter --task ensemble --sweep dims=4x4,5x5,6x6,7x7,8x8,9x9 "
                      "--replicas 6 --log-times 3 --seed 9")
    most = []
    real = cli._save

    def counting(rep, fig, out, stem):
        most.append(len(plt.get_fignums()))
        real(rep, fig, out, stem)
    monkeypatch.setattr(cli, "_save", counting)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # matplotlib's "More than 20 figures" warning
        code, out, err = run([prefix, "--per-point", "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert len(most) > 20 and max(most) <= 5
    assert plt.get_fignums() == []


def test_collapse_poor_search_reports_no_errors(outputs, tmp_path, capsys):
    code, out, err = run(["collapse", outputs["stat2"], "--xc", "2.269", "--nu", "1", "--search",
                          "--fix-a", "--a", "1", "--boot", "20", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "POOR COLLAPSE" in out
    assert "xc = " in out and "± " not in out.split("POOR")[0].split("collapse: S")[1]
    assert "errors withheld" in out


def test_each_command_keeps_its_own_report(outputs, tmp_path, capsys):
    code, out, err = run([outputs["ens"], "--out", str(tmp_path)], capsys)
    assert code == 0, err
    code, fit_out, err = run(["fit", outputs["ens"], "--x", "time", "--y", "length",
                              "--range", "2:40", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert (tmp_path / "ens_report.txt").read_text() == out
    assert (tmp_path / "ens_fit_length_report.txt").read_text() == fit_out
    code, c_out, err = run(["crossings", outputs["stat2"], "--column", "binder", "--boot", "20",
                            "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert (tmp_path / "stat2_crossings_binder_report.txt").read_text() == c_out
    code, k_out, err = run(["collapse", outputs["stat2"], "--column", "binder", "--xc", "2.269",
                            "--nu", "1", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert (tmp_path / "stat2_collapse_binder_report.txt").read_text() == k_out
    assert (tmp_path / "stat2_crossings_binder_report.txt").read_text() == c_out


def test_report_names_are_sanitised():
    assert cli._safe("a/b c") == "a_b_c" and cli._safe("abs_m") == "abs_m"


G_LABEL = ("errors not trustworthy: correlated points (G(r) at different r share the same "
           "configurations)")
RUN_LABEL = "errors not trustworthy: one autocorrelated trajectory"


@pytest.mark.parametrize("extra", [[], ["--unweighted"]])
def test_fit_correlation_uses_g_err_and_is_labelled(outputs, tmp_path, capsys, extra):
    code, out, err = run(["fit", outputs["stat"], "--x", "r", "--y", "G_mean", "--range", "1:3",
                          "--model", "line", *extra, "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert G_LABEL in out
    assert "chi2/dof = " not in out.replace("sqrt(chi2/dof) = ", "") and "p = " not in out
    if not extra:
        assert "weighted, correlated points" in out      # G_err was found and used


def test_fit_single_run_series_is_labelled(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["run"], "--x", "sweep", "--y", "rho", "--range", "1:50",
                          "--model", "power", "--unweighted", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert RUN_LABEL in out
    assert "chi2/dof = " not in out.replace("sqrt(chi2/dof) = ", "") and "p = " not in out


def _two_curve_correlation(tmp_path):
    """A G(r) file with two curves (t = 10: G = 1 - 0.2 r; t = 100: G = 1 - 0.1 r)."""
    rows = ["time,r,G_mean,G_err"]
    for t, slope in ((10, 0.2), (100, 0.1)):
        for r in range(5):
            rows.append(f"{t},{r},{1 - slope * r},0.01")
    (tmp_path / "gc_correlation.csv").write_text("\n".join(rows) + "\n")
    return str(tmp_path / "gc")


def test_fit_correlation_fits_the_last_curve_by_default(tmp_path, capsys):
    p = _two_curve_correlation(tmp_path)
    code, out, err = run(["fit", p, "--x", "r", "--y", "G_mean", "--model", "line",
                          "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert "time = 100" in out and "2 curves" in out
    assert "fitted together" not in out
    assert "b = -0.1" in out                     # the t = 100 slope, not a pooled one


def test_fit_correlation_curve_option_picks_another(tmp_path, capsys):
    p = _two_curve_correlation(tmp_path)
    code, out, err = run(["fit", p, "--x", "r", "--y", "G_mean", "--model", "line",
                          "--curve", "10", "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert "time = 10" in out and "b = -0.2" in out


def test_fit_correlation_unknown_curve_is_an_input_error(tmp_path, capsys):
    p = _two_curve_correlation(tmp_path)
    code, out, err = run(["fit", p, "--x", "r", "--y", "G_mean", "--curve", "7",
                          "--out", str(tmp_path / "o")], capsys)
    assert code == 2
    assert "no curve with time = 7" in err and "10" in err and "100" in err


def test_curve_option_needs_a_multi_curve_table(outputs, tmp_path, capsys):
    code, out, err = run(["fit", outputs["stat2"], "--x", "temperature", "--y", "binder",
                          "--curve", "3", "--out", str(tmp_path)], capsys)
    assert code == 2 and "--curve is only for correlation files with several curves" in err


def test_collapse_starts_option(outputs, tmp_path, capsys):
    base = ["collapse", outputs["stat2"], "--column", "binder", "--xc", "2.269", "--nu", "1",
            "--search", "--boot", "4", "--out", str(tmp_path)]
    code, out, err = run(base + ["--starts", "1"], capsys)
    assert code == 0, err
    assert ", 1 start" in out
    code, out, err = run(base, capsys)
    assert code == 0, err
    assert "best of 15 starts" in out
    code, out, err = run(base + ["--starts", "0"], capsys)
    assert code == 2 and "starts" in err


def test_collapse_starts_without_search_is_ignored_with_a_warning(outputs, tmp_path, capsys):
    code, out, err = run(["collapse", outputs["stat2"], "--column", "binder", "--xc", "2.269",
                          "--nu", "1", "--starts", "3", "--out", str(tmp_path)], capsys)
    assert code == 0, err
    assert "--starts" in out and "ignored" in out


def test_fit_correlation_close_large_times_are_not_pooled(tmp_path, capsys):
    # times 99999800 .. 100000000, 50 apart: a relative tolerance would merge them
    rows = ["time,r,G_mean,G_err"]
    times = list(range(99999800, 100000001, 50))
    for k, t in enumerate(times):
        for r in range(6):
            rows.append(f"{t},{r},{1 - 0.01 * (k + 1) * r},0.01")
    (tmp_path / "big_correlation.csv").write_text("\n".join(rows) + "\n")
    p = str(tmp_path / "big")
    code, out, err = run(["fit", p, "--x", "r", "--y", "G_mean", "--model", "line",
                          "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert "time = 100000000" in out and "b = -0.05" in out     # the last curve alone (k = 4)
    code, out, err = run(["fit", p, "--x", "r", "--y", "G_mean", "--model", "line",
                          "--curve", "99999900", "--out", str(tmp_path / "o")], capsys)
    assert code == 0, err
    assert "time = 99999900" in out and "b = -0.03" in out      # k = 2, not its neighbour


def test_prefix_named_like_a_command_gets_a_hint(tmp_path, capsys, monkeypatch):
    (tmp_path / "fit_sweep.csv").write_text("x,y\n1,2\n")
    monkeypatch.chdir(tmp_path)
    code, out, err = run(["fit"], capsys)
    assert code == 2
    assert "for the outputs with prefix fit write ./fit" in err
    code, out, err = run(["fit", "--help"], capsys)
    assert code == 0 and "write ./fit" not in err


def test_different_models_names_a_file_without_provenance(outputs, tmp_path):
    from odspy import InputError
    from odspy.sizes import families
    src = Path(outputs["ens_sweep"] + "_sweep.csv")
    bare = tmp_path / "bare_sweep.csv"
    bare.write_text("".join(ln for ln in src.read_text().splitlines(True) if not ln.startswith("#")))
    with pytest.raises(InputError) as e:
        families([outputs["ens_sweep"], str(tmp_path / "bare")], "mean_consensus_time")
    assert "model unknown: no provenance" in str(e.value) and "None" not in str(e.value)


def test_cli_merge(odsp_bin, tmp_path, capsys):
    base = "--model voter --dims 6x6 --task ensemble --steps 50 --seed 4"
    a = odsp_run(odsp_bin, tmp_path, "a", base + " --replicas 4")
    b = odsp_run(odsp_bin, tmp_path, "b", base + " --replicas 5 --first-replica 4")
    out = str(tmp_path / "merged")
    code, stdout, err = run(["merge", out, a, b], capsys)
    assert code == 0, err
    assert Path(out + "_replicas.csv").exists()
    report = Path(out + "_merge_report.txt")
    assert report.exists() and str(Path(out + "_replicas.csv")) in report.read_text()
    assert a in stdout and b in stdout
    c = odsp_run(odsp_bin, tmp_path, "c", base.replace("--seed 4", "--seed 5") + " --replicas 5 --first-replica 4")
    code, stdout, err = run(["merge", str(tmp_path / "bad"), a, c], capsys)
    assert code == 2 and "seed" in err and "odspy: error" in err
