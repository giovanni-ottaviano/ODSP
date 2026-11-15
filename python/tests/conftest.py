import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def odsp_bin():
    path = ROOT / "bin" / "odsp"
    if not path.exists():
        pytest.skip("bin/odsp not built: run make")
    return path


COMMANDS = {
    "run": "--model voter --dims 16x16 --steps 50 --observables m,rho,length --seed 1",
    "ens": ("--model voter --dims 16x16 --task ensemble --log-times 5 --replicas 24 "
            "--observables rho,length,domains --save-trajectories true --seed 2"),
    "ens_sweep": ("--model voter --task ensemble --sweep dims=6x6,8x8,10x10 "
                  "--replicas 20 --seed 3"),
    "stat2": ("--model ising --task stationary --sweep dims=8x8,12x12,16x16 "
              "--sweep2 temperature=2.1:2.4:0.05 --replicas 4 --samples 200 "
              "--burn-in 100 --seed 4"),
    "stat": ("--model ising --task stationary --dims 8x8 --temperature 2.3 "
             "--samples 100 --burn-in 50 --save-states true --correlation true --seed 5"),
    "cont": ("--model deffuant --graph complete --n 200 --task ensemble "
             "--sweep epsilon=0.2,0.4 --observables clusters --replicas 4 --seed 6"),
    "maj": ("--model majority --dims 16x16 --task ensemble "
            "--observables stripes,domains --replicas 20 --steps 100000 --seed 7"),
    "mjs": ("--model majority --task ensemble --sweep dims=8x8,12x12 --replicas 20 --steps 30 "
            "--seed 3"),
    "noprov": "--model voter --dims 16x16 --steps 50 --observables m,rho,length --seed 1 --provenance false",
    # consensus is not absorbing: first-passage times, every replica runs to the cap
    "nv": ("--model noisy-voter --noise 0.001 --dims 6x6 --task ensemble --replicas 30 "
           "--steps 2000 --seed 5"),
    # bounded confidence: absorbing, but `absorbed` holds one sweep after the last big change,
    # so 2 replicas that reach consensus in the last sweep before the cap have absorbed = 0
    "dfe": ("--model deffuant --epsilon 0.5 --graph complete --n 100 --task ensemble "
            "--replicas 60 --steps 30 --seed 1"),
    "nv_sweep": ("--model noisy-voter --noise 0.001 --task ensemble --sweep dims=6x6,8x8,10x10 "
                 "--replicas 20 --steps 2000 --seed 3"),
}


@pytest.fixture(scope="session")
def outputs(odsp_bin, tmp_path_factory):
    """Dict name -> output prefix, produced once with small sizes and fixed seeds."""
    env = dict(os.environ, ODSP_MAX_THREADS="8")
    result = {}
    for name, args in COMMANDS.items():
        d = tmp_path_factory.mktemp(name)
        prefix = d / name
        cmd = [str(odsp_bin), *args.split(), "--csv", str(prefix)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"fixture '{name}' failed: {' '.join(cmd)}\n{proc.stderr}")
        result[name] = str(prefix)
    return result


@pytest.fixture(scope="session")
def stat_point(odsp_bin, tmp_path_factory):
    """The `stat` command as a one-point sweep (`--sweep temperature=2.3`: the driver's own
    summary, in _sweep.csv) and as a plain run (the samples, in _samples.csv), same seed.
    Sweep points write no sample files, and point 0 uses the run's seed, so the two agree.
    Returns {"one": {...}, "chains": {...}} (the second with --replicas 4), each with the
    keys "sweep" and "samples" (output prefixes)."""
    env = dict(os.environ, ODSP_MAX_THREADS="8")
    base = COMMANDS["stat"].replace(" --save-states true --correlation true", "")
    result = {}
    for name, extra in (("one", ""), ("chains", " --replicas 4")):
        paths = {}
        for kind, args in (("sweep", base.replace("--temperature 2.3", "--sweep temperature=2.3")),
                           ("samples", base)):
            assert (kind == "sweep") == ("--sweep temperature=2.3" in args)
            prefix = tmp_path_factory.mktemp(f"stat_{name}_{kind}") / "s"
            cmd = [str(odsp_bin), *(args + extra).split(), "--csv", str(prefix)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                pytest.fail(f"fixture 'stat_point' failed: {' '.join(cmd)}\n{proc.stderr}")
            paths[kind] = str(prefix)
        result[name] = paths
    return result


import matplotlib  # noqa: E402

matplotlib.use("Agg")
