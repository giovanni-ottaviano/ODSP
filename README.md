# ODSP — Opinion Dynamics Simulation Package

A header-only C++17 library for Monte Carlo simulations of opinion-dynamics
and spin models (voter and its variants, q-voter, majority rule, majority vote
with noise, Sznajd, Ising, Potts) with two or many opinions, and
continuous-opinion bounded-confidence models (Deffuant-Weisbuch,
Hegselmann-Krause), on lattices, complete graphs and complex networks
(generated or read from file), with zealots, demo programs and a test suite.

## Layout

```
include/                    the library (header-only)
  BaseGraph.hpp               abstract graph: node-indexed topology + per-node state; NeighbourView
  RegularLattice.hpp          D-dimensional lattice (any D >= 1), periodic or open boundaries
  CompleteGraph.hpp           fully connected graph (mean field), O(N) memory, O(1) neighbourhood queries
  AdjacencyGraph.hpp          any simple undirected graph, built from an edge list
  EdgeList.hpp                edge lists: simplify, validate, degrees, connected components
  Generators.hpp              Erdos-Renyi, random-regular, Watts-Strogatz, Barabasi-Albert, star, complete
  EdgeListIO.hpp              read/write edge-list files (SNAP, KONECT, CSV-like formats)
  Model.hpp                   base class: Monte Carlo driver, observables, stationary sampling
  VoterModel.hpp              copy one neighbour; node / link / reverse update schemes
  RejectionFreeVoterModel.hpp same dynamics as node-update voter, skipping wasted updates
  NoisyVoterModel.hpp         voter + spontaneous random opinion with probability `noise`
  MajorityModel.hpp           adopt the majority opinion of all neighbours; configurable tie rule
  QVoterModel.hpp             q-voter with independence or anticonformity
  NonlinearVoterModel.hpp     flip with probability (fraction disagreeing)^alpha
  MajorityVoteModel.hpp       majority-vote model with noise (Oliveira)
  IsingModel.hpp              Ising model, Metropolis or Glauber dynamics
  PottsModel.hpp              q-state Potts model, heat bath or Metropolis dynamics
  OpinionCounts.hpp           per-opinion node counts (used by Model)
  BoundedConfidenceModel.hpp  base of the continuous-opinion models: consensus/convergence tests
  DeffuantModel.hpp           Deffuant-Weisbuch: random pairs within epsilon move toward each other
  HegselmannKrauseModel.hpp   Hegselmann-Krause: move to the mean of the opinions within epsilon
  OpinionClusters.hpp         clusters of continuous opinions (number, sizes, positions)
  SznajdModel.hpp             Sznajd outflow dynamics
  Ensemble.hpp                many independent replicas (multi-threaded) -> ensemble statistics
  Statistics.hpp              jackknife errors, integrated autocorrelation time
  Utils.hpp                   initial conditions: random +/-1, q opinions in given shares, distinct
                              opinions, uniform continuous opinions
  Cli.hpp, CsvWriter.hpp      command line and CSV output for the demos
src/odsp.cpp, src/driver/    the command-line driver (bin/odsp)
src/*_demo.cpp              demo programs
tests/                      test suite (self-contained, no external dependency)
```

## Header-only design

The library in `include/` is **header-only**: every `.hpp` contains both the
declarations and the definitions, and there are no library `.cpp` files to
compile or link. To use it in your own program, include the headers and add
the include path:

```sh
g++ -std=c++17 -O2 -pthread -Iinclude my_study.cpp -o my_study
```

Why:

- Almost everything is a template on the opinion type `T` (`Model<T>`,
  `VoterModel<T>`, `Ensemble<T>`, ...). The compiler generates the code for
  each `T` where it is used, so it needs the full definitions there; template
  definitions in a separate `.cpp` would not link from other files.
- Non-template functions defined in headers are marked `inline` (and member
  functions defined inside a class are implicitly inline). Without it, every
  file including the header would define the function again, and linking
  would fail.

The cost is compile time: every program recompiles all the library code it
uses, and changing a header rebuilds everything that includes it. That is why
the test files compile to separate objects in parallel (`make -j8 test`). If
build times become a problem, the non-template code (edge lists, generators,
file I/O, statistics, driver options) can move to `.cpp` files built into a
static library, without changing any code that uses it.

## Build and run

Requires `g++` with C++17 support and GNU make.

```sh
make                     # build the driver and the demos into bin/
make test                # build and run the test suite
make test FILTER=voter   # run only test cases whose name contains "voter"
make debug               # build everything with AddressSanitizer + UBSan into bin/debug/, run the tests
make clean
```

## The `odsp` driver

`bin/odsp` runs any model on any topology without writing C++:

```sh
odsp --model voter --graph lattice --dims 64x64 --steps 2000 --csv out/voter
odsp --model qvoter --q 3 --p 0.2 --graph ba --n 2000 --m 3 --task stationary --samples 2000 --interval 10
odsp --model ising --temperature 2.3 --init up --task stationary --samples 2000 --interval 50
odsp --model majority-vote --init up --dims 16x16 --task stationary --samples 4000 --interval 10 --sweep noise=0.06:0.09:0.005 --csv mv
odsp --model deffuant --epsilon 0.2 --graph complete --n 1000 --task ensemble --observables clusters
```

- **Tasks**: `run` (one trajectory, time series of `--observables`),
  `ensemble` (`--replicas` runs: consensus times, fixation probabilities,
  final observables, per-replica table) and `stationary` (mean, susceptibility,
  Binder cumulant with jackknife errors, autocorrelation time, a histogram of
  the samples -- printed and in `prefix_histogram.csv` -- and, for Ising and
  Potts, energy and specific heat N·var(e)/T²). `ensemble --threshold x` also
  records when the largest opinion share first reaches x; continuous-opinion
  ensembles write every replica's final clusters to `prefix_clusters.csv`.
- **Time series**: `--log-times [k]` (k points per decade, default 10) or
  `--every n` sets when `run` records. Given to `ensemble`, it also writes
  `prefix_timeseries.csv`: at each time, every observable's mean over all
  replicas with its standard error (absorbed runs keep their final value, which
  is exact since an absorbed state never changes), its mean over the runs still
  active, the fraction of runs still active, and the survival probability S(t)
  (fraction not yet in consensus; first passage for models that can leave it).
  `--sweep key=v1,v2,...` or `key=start:stop:step` repeats `ensemble` or
  `stationary` for each value of any option and writes one table. Sweep points
  run in parallel within `--threads`, and `stationary --replicas R` runs R
  independent chains in parallel, pooling their samples (errors from the spread
  between chains, with a warning when chains disagree beyond their own errors).
  Results never depend on the number of threads.
- **Everything is an option**: model and its parameters, graph (`lattice`,
  `complete`, `gnp`, `gnm`, `regular`, `ws`, `ba`, `star`, `file`), initial
  condition, zealots, observables. `odsp --help models | graphs | tasks | init |
  observables` lists them all with defaults.
- **Mistakes are caught**: an option that is misspelled or does not apply to
  the chosen model is an error ("--temprature ... a typo?", "--q does not
  apply here"), not silently ignored.
- **Reproducible**: the seed is printed, and with `--csv prefix` the complete
  effective configuration (defaults included) is saved to
  `prefix_config.txt`; `odsp --config prefix_config.txt` reruns it exactly.
  Options can also come from any `key = value` file via `--config`, with the
  command line overriding it.
- **Provenance**: every CSV starts with `#` comment lines recording the ODSP
  version and git commit (`-dirty` = built from uncommitted changes), the date,
  the command line and the full configuration; `odsp --version` prints the
  version. Read the files with `pandas.read_csv(f, comment="#")` or
  `numpy.genfromtxt(f, delimiter=",", comments="#", names=True)`;
  `--provenance false` writes plain CSV for tools that cannot skip comments.
- Random networks are generated once from the seed and shared by all replicas
  and sweep points (`--regenerate-graph` draws a new one per replica).

## Demo programs

| Program        | What it shows |
|----------------|---------------|
| `bin/basics`   | voter and majority models on a 24x24 lattice and a complete graph; coarsening curve rho(t); plain vs rejection-free voter timing on a 1D ring |
| `bin/ensemble` | consensus-time distribution, exit probability P(+1) = f0, freezing of the majority model |
| `bin/noisy`    | noisy voter order/disorder crossover: <\|m\|>, susceptibility, Binder cumulant vs noise, with error bars |
| `bin/models`   | Ising vs Onsager, majority-vote Binder crossing, q-voter phase diagram vs mean field, zealots vs the exact Beta-Binomial law, nonlinear voter, Sznajd vs voter exit probability |
| `bin/opinions` | voter model with q opinions vs exact fixation and consensus-time results; decay of the number of opinions on a lattice vs a complete graph; Potts transitions for q = 3 (continuous) and q = 8 (first order) |
| `bin/continuous` | Deffuant-Weisbuch and Hegselmann-Krause: number of opinion clusters vs the confidence bound (the 1/(2ε) rule), a fragmented final state, and the effect of the network |
| `bin/networks` | voter model on random-regular, Erdos-Renyi, Barabasi-Albert and Watts-Strogatz networks: active-link plateau vs the pair approximation, consensus times, exit probability for each update scheme; `--edges <file>` adds a network from file |

Every demo accepts:

```
--seed <n>       seed the RNG (default: random; `ensemble` defaults to 2024)
--csv <prefix>   also write results to <prefix>_*.csv (directories are created)
--threads <n>    threads for ensemble runs (default 0 = all cores); results do not depend on it
--help
```

The seed in use is always printed first, so any run can be repeated exactly.
"All cores" (`--threads 0`, or `threads = 0` in `Ensemble::run`) is capped by
the environment variable `ODSP_MAX_THREADS` when set -- e.g.
`export ODSP_MAX_THREADS=10` keeps two cores free on a 12-core machine. The
test suite caps itself at 8.
`bin/networks` also takes `--edges <file>`.
Via make: `make run ARGS="--model voter --steps 100"` runs the driver; `make run-basics ARGS="--seed 42"`
(likewise `run-ensemble`, `run-noisy`, `run-networks`, `run-models`, `run-opinions`, `run-continuous`) runs a demo.

## Topologies

| Graph | Construction | Notes |
|---|---|---|
| `RegularLattice<T>(states, dims, boundary)` | hypercubic, nearest neighbours | `Boundary::Periodic` (default, sides >= 3) or `Boundary::Open` (boundary sites have fewer neighbours) |
| `CompleteGraph<T>(states)` | all pairs | stores no edges; O(1) neighbourhood queries |
| `AdjacencyGraph<T>(states, edges)` | any simple undirected graph | compressed storage; rejects self-loops and repeated edges |

Networks are built from an edge list, from a generator or a file:

```cpp
std::mt19937 rng(42);
const std::size_t N = 1000;
EdgeList edges = barabasi_albert_edges(N, 3, rng);   // or gnp_edges, gnm_edges, random_regular_edges,
                                                     //    watts_strogatz_edges, star_edges, complete_edges
auto graph = std::make_unique<AdjacencyGraph<int>>(create_random_lattice(N, 0.5, rng), edges);
VoterModel<int> model(std::move(graph), &rng);

EdgeListFile file = read_edge_list("network.txt");   // labels -> 0..n-1, drops self-loops and repeats
Subgraph lcc = largest_connected_component(file.nodes, file.edges);
```

Voter-like dynamics only reach consensus on a connected graph: sparse
Erdos-Renyi graphs and empirical networks often are not, so take the largest
connected component. Generators are reproducible from the RNG seed;
`write_edge_list` saves a network for reuse.

## Models

| Model | Rule | Notes |
|---|---|---|
| `VoterModel` | copy the opinion across one random edge | `VoterUpdate::Node` (default), `Link`, `Reverse`; they differ on graphs with heterogeneous degrees (see below) |
| `RejectionFreeVoterModel` | node-update voter | exact same statistics; ~10x faster in 1D, about plain speed in 2D/3D; not for complete graphs |
| `NoisyVoterModel` | voter, or with probability `noise` a random opinion | no absorbing state: use `sample_stationary` |
| `MajorityModel` | adopt the neighbourhood majority (plurality for >2 opinions) | `TieRule::Random` (default) or `KeepCurrent` (= zero-temperature Glauber for ±1) |
| `QVoterModel` | adopt the opinion of q neighbours if they are unanimous; with probability p act as a nonconformist | `Nonconformity::Independence` or `Anticonformity`; panel with or without repetition; mean-field p_c = (q−1)/(q−1+2^(q−1)) and (q−1)/(2q) |
| `NonlinearVoterModel` | flip with probability f^α, f = fraction of disagreeing neighbours | α = 1 is the voter model; α > 1 ordering, α < 1 disordering |
| `MajorityVoteModel` | adopt the neighbourhood majority with probability 1 − noise, the minority otherwise | order-disorder transition at noise_c = 0.075 on the square lattice; \|m\| = 1 − 2·noise on a complete graph |
| `IsingModel` | single spin flips at temperature T, H = −J Σ sᵢsⱼ − h Σ sᵢ | `IsingDynamics::Metropolis` or `Glauber`; `energy()`; J = 1/(N−1) on a complete graph gives Curie–Weiss |
| `SznajdModel` | an agreeing pair of neighbours converts all of its neighbours | exit probability much steeper than the voter model's |
| `PottsModel` | q-state Potts model at temperature T, H = −J Σ δ(sᵢ,sⱼ), states 0..q−1 | `PottsDynamics::HeatBath` or `Metropolis`; exact T_c = 1/ln(1+√q) on the square lattice, continuous for q ≤ 4, first order above; `order_parameter()`, `energy()` |

**Continuous opinions** (`T = double`): `DeffuantModel<>(graph, &rng, epsilon, mu = 0.5)` and
`HegselmannKrauseModel<>(graph, &rng, epsilon, synchronous = true)`. Agents only
interact with opinions within ε of their own. Such dynamics never reach exact
consensus, so with a `tolerance`: consensus = all opinions within it;
converged = every pair of neighbours that can still interact agrees within it
(consensus, or a frozen set of separated clusters) -- `advance()` stops there.
Analyse final states with `model.clusters()` / `count_clusters(states, gap,
min_fraction)`. On complete graphs Hegselmann-Krause rounds and the
convergence test cost O(N log N) (sorted opinions), not O(N²).

**Number of opinions.** Opinions are values of the template type `T`. The
voter models (`VoterModel`, `RejectionFreeVoterModel`), `MajorityModel`
(plurality), `NoisyVoterModel` (pass the opinion set, e.g. `{0, 1, 2, 3}`) and
`PottsModel` work with any number of opinions; the other models assume ±1.
Initial conditions: `create_opinions_with_fractions(N, {0.2, 0.3, 0.5}, rng)`,
`create_balanced_opinions(N, q, rng)`, `create_distinct_opinions(N)`.

**Zealots.** Any node of any model can be frozen with `model.set_frozen(node)`
or `set_frozen(list)`: it never changes but still influences its neighbours.
Zealots of a single opinion make that consensus inevitable; zealots of both
opinions prevent consensus, and the stationary state can be measured with
`sample_stationary` (on a complete graph its distribution is exactly
Beta-Binomial).

**Voter update schemes.** Each conserves, on average, a different weighted
magnetization M, and the probability of ending at +1 consensus is (1 + M₀)/2:
node update conserves Σkᵢsᵢ / Σkᵢ (`get_degree_weighted_magnetization()`), link
update conserves the plain m, and reverse update (invasion) conserves
Σ(sᵢ/kᵢ) / Σ(1/kᵢ). On lattices and complete graphs all degrees are equal and
the three coincide.

**Time units.** One update is one attempted application of the rule; one
sweep (Monte Carlo step) is N updates.

## Observables

- **Opinion counts** (integer opinion types): `number_of_opinions()`,
  `count_of(op)`, `opinion_counts()`, `largest_opinion_fraction()`, all
  maintained incrementally. Consensus means one opinion left, for any number
  of opinions.
- **Magnetization** `m = sum of spins` (and `m / N`), and the
  **degree-weighted magnetization**: the order parameters of ±1 models
  (meaningless for other labelings).
- **Persistence** (fraction of nodes that never changed opinion; coarsening
  exponent θ) and **activity** (`get_changes() / get_updates()`: fraction of
  attempted updates that changed something).
- **Diversity and mixing** (integer opinions): `opinion_entropy()`,
  `effective_number_of_opinions()` = exp(entropy), and `segregation()` =
  1 − ρ / ρ_random (0 for random mixing, exactly 0 on a complete graph; → 1 for
  segregated opinions; 1 at consensus). Continuous opinions:
  `opinion_variance()` and `bimodality_coefficient()` (Sarle; > 5/9 suggests
  two peaks).
- **Spatial and temporal correlations** (`Correlations.hpp`, lattices):
  G(r) along the lattice axes, one definition for all opinion types --
  (P_same(r) − Σ p²)/(1 − Σ p²) for discrete opinions (the connected
  correlation for ±1 spins), the Pearson correlation for continuous ones; the
  coarsening length L(t) where G drops to ½ (observable `length`); the
  second-moment correlation length ξ from the structure factor, reported with
  ξ/L by `stationary` on periodic hypercubic lattices (from averaged S(0) and
  S(k_min), with jackknife errors); and the two-time autocorrelation C(t, t_w)
  and overlap (observables `autocorrelation`, `overlap` with `--tw t1,t2,...`).
  `--correlation` writes the G(r) curves: per recording time (`run`),
  averaged over replicas (`ensemble` time series) or over samples
  (`stationary`).
- **Active-link density** `rho`: fraction of edges joining disagreeing nodes.
  It tracks coarsening (for the voter model `m` is conserved on average while
  `rho` decays) and `rho = 0` exactly at consensus.
- **Stationary statistics** (`Model::sample_stationary`): time averages of
  `m`, `|m|`, `m^2`, `m^4`, the susceptibilities N(⟨m²⟩−⟨m⟩²) and
  N(⟨m²⟩−⟨|m|⟩²), and the Binder cumulant, each with a block-jackknife error,
  plus the integrated autocorrelation times of `m` and `|m|`. The errors are
  reliable when the autocorrelation time is much shorter than a jackknife
  block (n_samples / 20 samples by default). An overload takes any observable,
  e.g. `model.sample_stationary([](const Model<int>& m) { return m.largest_opinion_fraction(); }, burn_in, n)`.
- **Ensemble statistics** (`Ensemble::run`; an optional last argument
  measures any quantity on each replica's final state, e.g. the number of
  opinion clusters): consensus-time distribution,
  fixation probability of each opinion (`fixation_probability(op)`;
  `exit_prob_up` is that of +1), mean number of surviving opinions, moments of
  the final `m`, and censoring-aware summaries. For models that can leave
  consensus (noisy voter, Ising/Potts at T > 0, q-voter with p > 0, majority
  vote with noise), consensus times are *first-passage* times, every replica
  runs to the cap, and no fixation is reported (`Result::first_passage`).
  Summaries:
  survival curve `survival(t)`, median consensus time, and a lower bound on the
  mean. The plain mean consensus time only covers runs that finished within the
  cap, so it is biased low whenever `survival_fraction > 0`.

All observables are updated incrementally on every state change, so reading
them is O(1). `Model::advance` takes an optional observer callback for
recording time series.

## Extending

**A new model** derives from `Model<T>` and implements one method,
`_single_update()`, which picks a node (`random_node()`, `random_neighbour(i)`),
reads the graph through `graph()` and changes a state **only** through
`_apply(node, value)`, which keeps the observables consistent. See
`VoterModel.hpp`. Override `consensus_is_absorbing()` if the model can leave
consensus. Zealots work automatically through `_apply`; a model that caches
per-node data must also override `_on_frozen_changed()` (see
`RejectionFreeVoterModel.hpp`). Other helpers: `random_directed_edge()` for a
uniform random edge, `neighbour_spin_sum(i)` for the local field of ±1 spins.
Models whose consensus or end state needs a different definition (e.g.
continuous opinions) override `is_consensus_reached()` and
`is_absorbing_state()`; see `BoundedConfidenceModel.hpp`.

**A new topology** derives from `BaseGraph<T>` and implements `size()`,
`neighbours(node)` (return `NeighbourView(list)` over stored ids),
`get_state`, `set_state` and `states()`. If the graph can count neighbour
states faster than by scanning, override `count_neighbours_in_state` and
`neighbour_state_counts` as well (see `CompleteGraph.hpp`). The active-link
bookkeeping assumes an undirected graph without self-loops or repeated
neighbours.

## Tests

`make test` runs the driver's tests (option parsing, every model and graph
built from options, error reporting, end-to-end runs, exact config replay) and
unit tests (lattice geometry, neighbourhood queries, input
validation, CSV output, incremental bookkeeping against full recounts,
statistics helpers) and statistical tests against exact results:

- voter model on a complete graph: exit probability equals the initial
  up-fraction; mean consensus time matches the exact birth-death-chain value
  (for both the plain and the rejection-free algorithm);
- voter update schemes on a star graph: exit probabilities 1/2, 1/(n+1) and
  1/(1+n²) for node, link and reverse update; on a Barabasi-Albert network
  with the hubs up, P(+1) = (1 + ω₀)/2 for node update and (1 + m₀)/2 for link
  update;
- network generators: edge counts, degree sequences (exact for random-regular,
  the k⁻³ tail for Barabasi-Albert), the ring limit of Watts-Strogatz,
  reproducibility; edge-list parsing of messy input and round trips;
- voter active-link plateau on a random-regular graph against the pair
  approximation (an approximation, accurate to < 1% there);
- rejection-free vs plain voter on a lattice; noisy voter at full noise
  (independent spins); jackknife and autocorrelation estimates on series with
  known answers;
- zealots: frozen nodes never change in any model, and the voter model with
  zealots on a complete graph has the exact Beta-Binomial stationary mean and
  variance (plain and rejection-free algorithms);
- q-voter (q = 1) and nonlinear voter (α = 1) reproduce the voter model's exact
  consensus time; q-voter stationary magnetization against mean field for
  independence and anticonformity; majority vote against |m| = 1 − 2·noise;
- Ising: exact enumeration of the 3×3 lattice (energy and |m|, both
  dynamics), Onsager's magnetization at T = 2, Curie–Weiss self-consistency;
- many opinions: opinion counts against recounts in every model; voter
  fixation probability = initial share with 3 opinions; mean consensus time
  (N−1)² from N distinct opinions on a complete graph (coalescing random walk
  duality), for both voter algorithms; multinomial stationary state of the
  noisy voter at full noise; Potts exact enumeration on 3×3 (q = 3, both
  dynamics), q = 2 Potts = Ising at half coupling (Onsager), q = 3 transition;
- continuous opinions: Deffuant conserves the mean opinion and never leaves
  the initial range; ε above the range gives consensus at the initial mean;
  strict (Deffuant) and inclusive (Hegselmann-Krause) confidence bounds;
  Hegselmann-Krause reaches the mean in one round when everyone is in range and
  preserves the order of opinions on a complete graph; the O(N log N) complete-
  graph paths give the same results as the general code; converged states are
  frozen on sparse graphs; ~1/(2ε) clusters.

Statistical tests use fixed seeds and tolerances of about 4 standard errors.

## License

GPL-3.0 — see `LICENSE`.
