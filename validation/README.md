# Validation campaign

Every ODSP model run at production scale (on a SLURM cluster) or at small scale (on one
machine) and compared with theory. The result is a table of measured value, prediction (with
its citation) and deviation in σ for every check, plus one figure per check.

Run everything from the repository root, with the analysis package's Python:

```
make -j10 bin/odsp
python/.venv/bin/python -m validation.campaign <stage> --scale <scale> [options]
```

## Scales

| scale   | what for                                    | where           |
|---------|---------------------------------------------|-----------------|
| `smoke` | seconds; the pipeline under pytest          | any machine     |
| `small` | about an hour; a regression check           | one machine     |
| `full`  | production statistics and sizes             | a SLURM cluster |

All three come from one definition per model (only sizes and statistics differ), so the
small scale exercises the same pipeline as the full one.

## Stages

| stage     | what it does |
|-----------|--------------|
| `list`    | the runs of each model and the jobs they split into |
| `run`     | run the jobs not done yet on this machine, then `merge`, then `analyse` |
| `slurm`   | write SLURM job arrays for the jobs not done yet |
| `status`  | per run: jobs done, merged or not; per model: analysed or not |
| `merge`   | join the jobs of every run whose jobs are all done into one output prefix |
| `analyse` | analyse every model whose runs are all merged; write `summary.csv` and `summary.md` |

Options (every stage):

- `--scale smoke|small|full` (required);
- `--models a,b` to restrict the stage to some models (default: all);
- `--threads N`: the local thread budget of `run` (default 10). A job starts when the
  threads of the running jobs plus its own fit in the budget;
- `--force`: `run` reruns jobs already done, `slurm` lists them again (and removes their
  done markers), `merge` and `run` re-merge runs that are up to date;
- `--root DIR`: the results root (default `validation/results`);
- `--odsp PATH`: the driver binary used by `run` (default `bin/odsp`);
- `--conf FILE`: the SLURM configuration of `slurm` (default `validation/slurm.conf`).

The exit status is 0 when everything the stage attempted succeeded and 1 otherwise (a failed
job, merge or analysis, or a model skipped because its runs are not ready).

## How a run becomes jobs

A model (`validation/models/<name>.py`) declares, per scale, its logical **runs**: odsp
arguments, a seed, the threads per job and, for a large ensemble, a replica block size.
Each run is split into jobs:

- a sweep (`--sweep`, optionally `--sweep2`) gives one job per point, `--sweep-point i`:
  the job runs exactly what point i gets in the full sweep (same seed, same files);
- an ensemble with a `block` gives one job per block of replicas,
  `--replicas b --first-replica k` (same seed, disjoint replica ranges);
- both together give one job per (point, block); anything else is one job (e.g. a
  stationary run without `--sweep`, whose samples and histogram `merge` copies as they are).

Every job gets `--seed`, `--threads` and `--csv results/<scale>/<model>/parts/<run>/p<point>_b<first>`.
A job is **done** when `<prefix>.done` exists: it is written only when odsp exits with 0, so
an interrupted or failed job is simply run again by the next `run` (its partial outputs are
removed first). Locally, each job's output goes to `<prefix>.log`.

`merge` joins a run's jobs with `odspy.merge` into `results/<scale>/<model>/<run>`, which
odspy then reads as if it came from one run. A merge is redone only when a job finished after
it (or with `--force`); a failed merge leaves no merged outputs behind.

`analyse` calls each model's analysis only when all its runs are merged and up to date;
otherwise the model is skipped and `log.txt` names the runs it waits for. A model whose
analysis raises is reported (with the traceback) and the others still go into the table.
Each model's checks are kept in `results/<scale>/<model>/checks.json`, so analysing one
model (`--models`) leaves the other models' rows in the summary.

## Locally

```
python/.venv/bin/python -m validation.campaign list --scale small
python/.venv/bin/python -m validation.campaign run --scale small --threads 10
python/.venv/bin/python -m validation.campaign status --scale small
```

`run` can be interrupted and started again: it runs only the jobs not done, merges only
what changed and re-analyses.

## On a SLURM cluster

1. Build `bin/odsp` on the cluster, and create the Python environment (`python/.venv`) as in
   the main README.
2. Copy `validation/slurm.conf.example` to `validation/slurm.conf` and edit it (the file is
   not tracked by git). Keys (all but `max_array_size` required; all but `time` may be
   empty, which leaves that line out):

   | key              | meaning |
   |------------------|---------|
   | `partition`      | `#SBATCH --partition` |
   | `account`        | `#SBATCH --account` |
   | `time`           | wall-time limit of one array task (one job), `#SBATCH --time` |
   | `mem`            | memory per task, `#SBATCH --mem` |
   | `max_concurrent` | at most this many tasks of one array at once, `--array=1-n%max` |
   | `max_array_size` | optional, default 1000: at most this many jobs per array script; keep it below the cluster's `MaxArraySize` (`scontrol show config \| grep MaxArraySize`, 1001 by default) |
   | `setup`          | one shell line run before the job, e.g. `module load gcc` |

   `time` is one limit for every array, so it must cover the longest job of the scale. A
   job that hits it is killed and run again from the start by the next `slurm` call (jobs
   do not checkpoint). At full scale the longest jobs are the finite-size-scaling points at
   the largest size, 16 chains on 8 threads (two rounds of chains), estimated at 2e7
   single-site updates per second per thread: majority vote L = 256 about 11 h, Ising
   L = 256 about 5.5 h, Potts L = 192 about 4.2 h. The zero-temperature majority-rule stripe
   blocks at L = 256 (20 replicas on 2 threads) usually take 20-60 min, but a replica caught in
   a long-lived diagonal stripe (decay time ~ 0.06 L^3 sweeps, about 55 min at L = 256) runs
   alone: the slowest of the ~400 such replicas is expected near 5.5 h, and a replica that runs
   to the step cap (1.23e7 sweeps, chosen so that fewer than 0.01 replicas of the whole run are
   expected to reach it) takes about 11.2 h, the worst case of that job. If such a job is killed
   by the time limit, it would time out again with the same seed: run its line from
   `jobs_t2.txt` by hand without a limit, or lower the cap (`TAU_DIAG` or `CAPPED_EXPECTED` in
   `validation/models/majority.py`) and run `slurm` again, which reruns that whole size (a capped replica counts
   as not frozen in stripes and is reported in the row). Every other job takes under 1 h. Ask for
   at least 24 h (twice the longest) on dedicated nodes, and up to 55 h (five times) on
   shared or virtualised nodes: slowdowns of up to 5x were seen on a local virtual machine.

3. Write the arrays and submit every script it lists:

   ```
   python/.venv/bin/python -m validation.campaign slurm --scale full
   sbatch validation/results/full/slurm/array_t1.sbatch
   sbatch validation/results/full/slurm/array_t4.sbatch     # one array per thread count
   ```

   Jobs are grouped by thread count, and a group with more than `max_array_size` jobs is
   cut into several scripts, `array_t<n>_1.sbatch`, `array_t<n>_2.sbatch`, ... Each script
   requests `--cpus-per-task=<n>` and carries its own job list: task i runs line i,

   ```
   [ -e <prefix>.done ] || { cd <repo> && bin/odsp <args> && touch <prefix>.done; }
   ```

   so a job already done is skipped, and a task id outside the list exits with an error.
   Because sbatch keeps a copy of the script, a later `slurm` call (which rewrites the
   scripts in `results/full/slurm/`) cannot change what already submitted arrays run.
   `jobs_t<n>[_<k>].txt` is a readable copy of each script's list. Task output goes to
   `results/full/slurm/logs/`.
4. When the arrays have finished (or been cancelled; a job still running in an old array
   is not done yet and would be listed again), `status` shows what is done. Run `slurm`
   again to write arrays for the jobs that failed or timed out (only those), then

   ```
   python/.venv/bin/python -m validation.campaign merge --scale full
   python/.venv/bin/python -m validation.campaign analyse --scale full
   ```

## Outputs

Under `results/<scale>/` (not tracked by git):

- `<model>/parts/<run>/`: the jobs' outputs, `.done` markers and logs;
- `<model>/<run>_*.csv`, `<model>/<run>_config.txt`: the merged runs;
- `figures/`: one figure per check;
- `summary.csv` (full precision: model, check, quantity, measured, error, predicted,
  predicted_error, source, sigma, tolerance_sigma, sizes, statistics, warnings, figure) and
  `summary.md` (the same as a table; `!` marks |σ| above the check's tolerance);
- `log.txt`: what every `run`, `slurm`, `merge` and `analyse` did, skipped or failed.

σ = (measured − predicted) / hypot(error, predicted error). At full scale the table reports
σ; at smoke and small scale it doubles as a regression check against each check's stated
tolerance.

## Adding a model

A module `validation/models/<name>.py` defines

```python
NAME = "<name>"
def runs(scale: str) -> list[Run]: ...                              # validation.core.Run
def analyse(scale: str, data: Path, figures: Path) -> list[Check]: ...  # validation.core.Check
```

where `data` is `results/<scale>/<name>/` (the merged prefix of run `r` is `data / r`).
Run arguments never contain `--csv`, `--seed`, `--threads`, `--sweep-point` or
`--first-replica`: the campaign sets them. Every `Check.source` is a key of
`theory.CITATIONS`. The package registers every such module by `NAME`.

`validation/analysis.py` holds the helpers the model analyses share: `weighted_ratio` and
`ratio_check` (simulation / theory over several points, predicted 1), `exponent_check` (a fit
parameter from an odspy `FitResult`, with the fit's warnings and quality in `Check.warnings`),
`plot_ratio`, `plot_exponent` and `save_figure` (`figures/<check>.pdf/.png`). Each model has a
smoke test `validation/tests/test_models_<name>.py` built on `tests/smoke.py`'s `run_smoke`.

## Tests

```
python/.venv/bin/pytest validation/tests -q
```
