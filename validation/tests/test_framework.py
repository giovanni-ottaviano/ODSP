"""Tests of the validation-campaign framework (split, local runner, merge, analyse, SLURM
arrays, summary) on the fake voter model of fake_model.py."""

from __future__ import annotations

import csv
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from odspy import find_outputs

import validation.models
from validation import campaign
from validation.core import Check, Run, merged_prefix, split
from validation.report import write_summary
from validation.runner import is_done, run_jobs
from validation.slurm import read_conf, write_arrays

from . import fake_model


def opt(args: tuple[str, ...], key: str) -> str | None:
    """The value after `key` in an argument list (None when absent)."""
    return args[args.index(key) + 1] if key in args else None


def summary_rows(scale_root: Path) -> list[dict[str, str]]:
    with open(scale_root / "summary.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def models(monkeypatch):
    """Replace the model registry by the given models for one test."""
    def use(*mods) -> None:
        monkeypatch.setattr(validation.models, "MODELS", {m.NAME: m for m in mods})
    return use


def run_campaign(stage: str, root: Path, odsp: Path, *extra: str) -> int:
    return campaign.main([stage, "--scale", "smoke", "--root", str(root), "--odsp", str(odsp),
                          *extra])


def test_split_points_and_blocks(tmp_path):
    runs = {r.name: r for r in fake_model.runs("smoke")}
    points = split("fake", runs["sweep"], tmp_path)
    assert [opt(j.args, "--sweep-point") for j in points] == ["0", "1", "2"]
    for i, j in enumerate(points):
        assert j.model == "fake" and j.run == "sweep" and j.threads == 1
        assert j.prefix == tmp_path / "fake" / "parts" / "sweep" / f"p{i}_b0"
        assert opt(j.args, "--csv") == str(j.prefix)
        assert opt(j.args, "--seed") == "1" and opt(j.args, "--threads") == "1"
        assert "--first-replica" not in j.args
    blocks = split("fake", runs["ens"], tmp_path)
    assert [(opt(j.args, "--replicas"), opt(j.args, "--first-replica")) for j in blocks] == [
        ("3", "0"), ("3", "3"), ("1", "6")]
    for j in blocks:
        assert j.args.count("--replicas") == 1 and "--sweep-point" not in j.args
        assert opt(j.args, "--seed") == "2"
    assert [j.prefix.name for j in blocks] == ["p0_b0", "p0_b3", "p0_b6"]
    assert merged_prefix(tmp_path, "fake", "ens") == tmp_path / "fake" / "ens"
    # models never pass what split sets, and block needs an ensemble
    with pytest.raises(ValueError, match="--seed"):
        split("fake", Run("x", ("--model", "voter", "--seed", "3"), seed=1), tmp_path)
    with pytest.raises(ValueError, match="ensemble"):
        split("fake", Run("x", ("--model", "voter", "--replicas", "4"), seed=1, block=2), tmp_path)


def test_run_merge_analyse_end_to_end(tmp_path, odsp_bin, models):
    models(fake_model)
    assert run_campaign("run", tmp_path, odsp_bin) == 0
    scale_root = tmp_path / "smoke"
    rows = summary_rows(scale_root)
    assert len(rows) == 1 and rows[0]["model"] == "fake" and rows[0]["check"] == "fake.exit"
    sweep = find_outputs(merged_prefix(scale_root, "fake", "sweep"))
    assert len(sweep.get("sweep")) == 3 and sorted(sweep.points) == [0, 1, 2]
    ens = find_outputs(merged_prefix(scale_root, "fake", "ens"))
    assert list(ens.get("replicas")["replica"]) == list(range(7))
    assert (scale_root / "summary.md").exists() and (scale_root / "log.txt").exists()


def test_rerun_skips_done_and_redoes_partial(tmp_path, odsp_bin, models):
    models(fake_model)
    assert run_campaign("run", tmp_path, odsp_bin) == 0
    scale_root = tmp_path / "smoke"
    jobs = [j for r in fake_model.runs("smoke") for j in split("fake", r, scale_root)]
    redo = jobs[4]                     # the second replica block of the ensemble
    others = [p for j in jobs if j is not redo
              for p in j.prefix.parent.glob(j.prefix.name + "[._]*")]
    before = {p: p.stat().st_mtime_ns for p in others}
    Path(f"{redo.prefix}.done").unlink()
    Path(f"{redo.prefix}_replicas.csv").write_text("garbage\n1,2\n")
    assert not is_done(redo)
    assert run_campaign("run", tmp_path, odsp_bin) == 0
    assert is_done(redo)
    assert {p: p.stat().st_mtime_ns for p in others} == before   # nothing else was rerun
    log = (scale_root / "log.txt").read_text()
    assert "1 job(s) to run" in log
    ens = find_outputs(merged_prefix(scale_root, "fake", "ens"))
    assert list(ens.get("replicas")["replica"]) == list(range(7))
    assert len(summary_rows(scale_root)) == 1


def test_failed_job_is_reported_and_skipped(tmp_path, odsp_bin, models):
    broken = fake_model.make_model("broken", bad=True)
    models(fake_model, broken)
    assert run_campaign("run", tmp_path, odsp_bin) != 0
    scale_root = tmp_path / "smoke"
    log = (scale_root / "log.txt").read_text()
    bad_job = split("broken", broken.runs("smoke")[0], scale_root)[0]
    assert str(bad_job.prefix) in log and "FAILED" in log
    assert not is_done(bad_job)
    # the other runs finished and merged, the broken model's analysis was skipped
    assert len(find_outputs(merged_prefix(scale_root, "broken", "ens")).get("replicas")) == 7
    assert len(find_outputs(merged_prefix(scale_root, "fake", "sweep")).get("sweep")) == 3
    rows = summary_rows(scale_root)
    assert [r["model"] for r in rows] == ["fake"]
    assert "skipped broken" in log


def test_analyse_with_missing_runs(tmp_path, odsp_bin, models):
    models(fake_model, fake_model.make_model("fake2"))
    assert run_campaign("run", tmp_path, odsp_bin) == 0
    scale_root = tmp_path / "smoke"
    assert len(summary_rows(scale_root)) == 2
    part = scale_root / "fake2" / "parts" / "ens"
    for f in part.glob("p0_b3[._]*"):
        f.unlink()
    assert run_campaign("analyse", tmp_path, odsp_bin) != 0
    rows = summary_rows(scale_root)
    assert [r["check"] for r in rows] == ["fake.exit"]
    log = (scale_root / "log.txt").read_text()
    assert "skipped fake2" in log and "ens" in log.split("skipped fake2", 1)[1].splitlines()[0]


CONF = {"partition": "short", "account": "proj42", "time": "01:00:00", "mem": "2G",
        "max_concurrent": "50", "setup": "module load gcc/12"}


def test_slurm_arrays(tmp_path, odsp_bin, models):
    sweep = Run("a", ("--model", "voter", "--graph", "complete", "--task", "ensemble",
                      "--replicas", "2", "--sweep", "n=10,20,30"), seed=1)
    ens = Run("b", ("--model", "voter", "--graph", "complete", "--n", "20", "--task", "ensemble",
                    "--replicas", "7"), seed=2, threads=4, block=3)
    jobs = split("m", sweep, tmp_path) + split("m", ens, tmp_path)
    out = tmp_path / "slurm"
    scripts = write_arrays(jobs, CONF, out)
    assert sorted(p.name for p in scripts) == ["array_t1.sbatch", "array_t4.sbatch"]
    text = (out / "array_t4.sbatch").read_text()
    for needle in ("--cpus-per-task=4", "--partition=short", "--account=proj42", "--time=01:00:00",
                   "--mem=2G", "--array=1-3%50", "module load gcc/12",
                   'sed -n "${SLURM_ARRAY_TASK_ID}p"'):
        assert needle in text, needle
    assert "jobs_t4.txt" not in text    # self-contained: the list file is only a copy
    assert "--cpus-per-task=1" in (out / "array_t1.sbatch").read_text()
    for t, js in ((1, jobs[:3]), (4, jobs[3:])):
        lines = (out / f"jobs_t{t}.txt").read_text().splitlines()
        assert len(lines) == len(js)
        script = (out / f"array_t{t}.sbatch").read_text()
        for line, j in zip(lines, js):
            # done-guard, then the job from the repository, marked done on success
            assert line.startswith(f"[ -e {j.prefix}.done ] || {{ cd ") and " && bin/odsp " in line
            assert line.endswith(f"&& touch {j.prefix}.done; }}")
            assert line in script.splitlines()      # the script carries its own job list
    with pytest.raises(FileNotFoundError, match="slurm.conf.example"):
        read_conf(tmp_path / "missing.conf")
    # the command: reads the conf, writes the arrays of the jobs not done
    conf = tmp_path / "slurm.conf"
    conf.write_text("# test\n" + "".join(f"{k} = {v}\n" for k, v in CONF.items()))
    models(fake_model)
    assert run_campaign("slurm", tmp_path, odsp_bin, "--conf", str(conf)) == 0
    lines = (tmp_path / "smoke" / "slurm" / "jobs_t1.txt").read_text().splitlines()
    assert len(lines) == 6
    assert run_campaign("slurm", tmp_path, odsp_bin, "--conf", str(tmp_path / "nope")) != 0


def resolve(script: Path, task: int | None) -> tuple[int, str]:
    """Run an array script as SLURM would for one task id, with every job already done (so
    the done-guard makes the job a no-op): the exit status and the job line it picked."""
    env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(task)
    p = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def test_slurm_chunks_and_stable_scripts(tmp_path):
    run = Run("s", ("--model", "voter", "--graph", "complete", "--task", "ensemble",
                    "--replicas", "2", "--sweep", "n=10,11,12,13,14,15,16"), seed=1)
    jobs = split("m", run, tmp_path / "res")
    for j in jobs:                      # done: the scripts only print what they would run
        j.prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{j.prefix}.done").touch()
    conf = dict(CONF, max_array_size="3", setup="")
    out = tmp_path / "slurm"
    scripts = write_arrays(jobs, conf, out)
    assert [p.name for p in scripts] == ["array_t1_1.sbatch", "array_t1_2.sbatch",
                                         "array_t1_3.sbatch"]
    sizes = [3, 3, 1]
    resolved: dict[Path, list[str]] = {}
    for s, n in zip(scripts, sizes):
        assert f"--array=1-{n}%50" in s.read_text()
        resolved[s] = []
        for task in range(1, n + 1):
            rc, line = resolve(s, task)
            assert rc == 0, line
            resolved[s].append(line)
        assert resolve(s, n + 1)[0] != 0        # out of range fails loudly
        assert resolve(s, 0)[0] != 0
        assert resolve(s, None)[0] != 0         # not an array task
    lines = [l for ls in resolved.values() for l in ls]
    expected = [f"[ -e {j.prefix}.done ] || {{ cd " for j in jobs]
    assert len(lines) == len(jobs) and all(l.startswith(e) for l, e in zip(lines, expected))
    # a later call (fewer jobs left) must not change what the submitted scripts run: sbatch
    # keeps a copy of the script, so the copy taken now must still resolve the same way
    kept = tmp_path / "submitted"
    kept.mkdir()
    copies = {s: kept / s.name for s in scripts}
    for s, c in copies.items():
        c.write_text(s.read_text())
    later = write_arrays(jobs[4:], conf, out)
    assert [p.name for p in later] == ["array_t1.sbatch"]
    for s, c in copies.items():
        assert [resolve(c, t)[1] for t in range(1, len(resolved[s]) + 1)] == resolved[s]
    # max_array_size is optional (default 1000) and must be a positive integer
    path = tmp_path / "c.conf"
    path.write_text("".join(f"{k} = {v}\n" for k, v in CONF.items()))
    assert read_conf(path).get("max_array_size", "1000") == "1000"
    path.write_text(path.read_text() + "max_array_size = 0\n")
    with pytest.raises(ValueError, match="max_array_size"):
        read_conf(path)


def test_threads_budget(tmp_path, odsp_bin):
    times = tmp_path / "times"
    times.mkdir()
    wrapper = tmp_path / "odsp_wrapper"
    wrapper.write_text(f"""#!{sys.executable}
import os, subprocess, sys, time
start = time.monotonic()
rc = subprocess.call([{str(odsp_bin)!r}] + sys.argv[1:], stdout=subprocess.DEVNULL)
time.sleep(0.4)
with open(os.path.join({str(times)!r}, str(os.getpid())), "w") as f:
    f.write(f"{{start}} {{time.monotonic()}}")
sys.exit(rc)
""")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    run = Run("s", ("--model", "voter", "--graph", "complete", "--task", "ensemble",
                    "--replicas", "2", "--sweep", "n=10,12,14,16,18"), seed=1, threads=2)
    jobs = split("m", run, tmp_path)
    assert run_jobs(jobs, 4, wrapper) == []
    assert all(is_done(j) for j in jobs)
    spans = [tuple(map(float, f.read_text().split())) for f in times.iterdir()]
    assert len(spans) == 5
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    running = peak = 0
    for _, d in events:
        running += d
        peak = max(peak, running)
    assert peak == 2


def test_summary_sigma(tmp_path):
    c = Check(model="m", check="m.x", quantity="x", measured=1.1, error=0.05, predicted=1.0,
              predicted_error=0.0, source="src", sizes="N=10", statistics="100 replicas",
              warnings=["a | b"], figure=None)
    assert c.sigma == pytest.approx(2.0)
    write_summary([c], tmp_path)
    rows = summary_rows(tmp_path)
    assert len(rows) == 1 and float(rows[0]["sigma"]) == pytest.approx(2.0)
    table = [l for l in (tmp_path / "summary.md").read_text().splitlines() if l.startswith("|")]
    assert len(table) == 3             # header, separator, one row
