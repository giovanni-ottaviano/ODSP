# TODO

## Adaptive (coevolving) networks

Networks whose links rewire as opinions change, so that topology and opinions
evolve together.

**Reference model — coevolving voter model** (Holme & Newman, Phys. Rev. E 74,
056108, 2006; Vazquez, Eguíluz & San Miguel, Phys. Rev. Lett. 100, 108702, 2008).
Pick a random node i and a random neighbour j that disagrees with it. With
probability p, i cuts the link to j and rewires it to a random node that holds
its own opinion. Otherwise (probability 1 − p), i copies j's opinion. Expected
behaviour: a fragmentation transition at p_c ≈ (μ − 2)/(μ − 1) in the pair
approximation (μ = mean degree). Below p_c the network stays connected and
reaches consensus. Above p_c it splits into disconnected components, each in
consensus on a different opinion.

**What the current code assumes and would have to change:**

- **Mutable topology.** `BaseGraph` exposes topology as read-only
  (`neighbours()` returns a view into fixed storage). A rewirable graph needs
  `add_edge` / `remove_edge`. The CSR layout of `AdjacencyGraph` can't do that
  cheaply, so this needs a new graph type, e.g. per-node neighbour vectors with
  swap-and-pop removal. Rewiring invalidates any `NeighbourView` held across it.
  `RegularLattice` and `CompleteGraph` stay fixed.
- **`Model` bookkeeping.** `_total_half_edges` is computed once and treated as
  constant. A rewiring step must also update the active-link count (the link
  moves from a disagreeing pair to an agreeing one) and the degree-weighted
  magnetization (two nodes change degree). The cleanest way is a single
  `_rewire(i, old_j, new_j)` in `Model`, the topology counterpart of `_apply`,
  so models can't desynchronise the observables.
- **Caches that assume fixed topology:**
  - the cumulative degree table behind `Model::random_directed_edge()`
  - the per-node rates in `RejectionFreeVoterModel`
  - `IsingModel`'s / `PottsModel`'s tables sized by the maximum degree

  Each needs invalidation or a "topology changed" hook, or should be declared
  incompatible with rewiring.
- **Observables.** Number and sizes of connected components over time
  (incremental, or `connected_components()` on demand). Also active-link density
  at the end, and the fraction of nodes in the largest component.

**Tests to write:**

- Rewiring preserves the number of edges.
- The graph stays simple (no self-loops or repeated edges).
- The incremental observables match a recount after every sweep.
- At p = 0 the model is exactly the voter model.
- At large p the network fragments into components that are each in consensus.
- The transition point is near (μ − 2)/(μ − 1) on random regular graphs.

## Axelrod model of cultural dissemination

Each agent carries a *culture*: a vector of F features, each taking one of q
traits (Axelrod, J. Conflict Resolution 41, 203, 1997).

**Update rule.** Pick a random agent i and a random neighbour j. Their overlap
is the fraction of features on which they share the same trait. With
probability equal to the overlap, i copies j's trait on one randomly chosen
feature where they still differ. If the overlap is 0 or 1, nothing happens.

**Absorbing states.** Every pair of neighbours is either identical or shares
nothing. These states are often fragmented into regions with distinct cultures.

**Reference results:**
- **Monocultural/multicultural transition in q** (Castellano, Marsili &
  Vespignani, Phys. Rev. Lett. 85, 3536, 2000). On a 2D lattice a giant
  cultural region exists for small q and fragments for large q. For F = 10,
  q_c ≈ 55; the transition is discontinuous for F > 2 and continuous for F = 2.
- **Cultural drift** (random trait changes at a small rate; Klemm, Eguíluz,
  Toral & San Miguel, 2003) destroys the fragmented states.

**What the current code assumes and would have to change:**

- **State type.** A culture is a vector, but `Model` converts states to
  `double` (magnetization) and has opinion counts only for integer types.
- **Option A (recommended first):** encode a culture as one integer, with base-q
  digits, one per feature. F = 10 with q up to ~79 fits in 64 bits, enough for
  the region around q_c. Opinion counts and consensus detection then work
  unchanged. Feature access is digit arithmetic, which needs a small encoding
  helper with tests.
- **Option B:** make `Model` work with non-arithmetic states (magnetization and
  degree-weighted magnetization only for arithmetic `T`, opinion counts via
  hashing or ordering). No limit on F and q, but a larger change to the core.
- **New observable.** The largest cultural region: the largest connected
  component of the graph restricted to edges between identical agents.
  Normalised by N, it is the order parameter.
- **Absorbing-state test** (`is_absorbing_state()`). Every edge has overlap 0
  or 1. That is an O(E·F) check, so gate it the way `BoundedConfidenceModel`
  does, by checking only after a sweep without changes.
- **Optional.** A drift rate r (random trait change per update) as a
  constructor parameter.

**Tests to write:**

- Encoding round-trips (pack/unpack features).
- q = 1: the initial state is already consensus.
- Overlap and copy rule on hand-made pairs (overlap 0, partial, 1).
- The final state satisfies the absorbing criterion on lattices and networks.
- Small q gives a monocultural final state on a 2D lattice; large q gives a
  fragmented one.
- With drift, the system no longer freezes.
- Bookkeeping (opinion counts = number of distinct cultures) matches a recount.

## Research-workflow improvements (from the 2026-10-04 scan)

### ~~Parallel sweeps and parallel stationary chains~~ (done 2026-10-04)
Implemented: sweep points run in parallel within `--threads`, and
`stationary --replicas R` pools R parallel chains. Measured at 10 threads:
an 11-point stationary sweep 7.7 s -> 1.6 s; 10 chains 6.9 s -> 1.4 s.

### ~~Provenance in every output file~~ (done 2026-10-04)
Implemented: `#` preamble (version, git commit, date, command, configuration)
in every CSV of the driver and the demos, `--provenance false`, `--version`.

### ~~Ensemble-averaged time series, log-spaced recording~~ (done 2026-10-04)
Implemented: `--log-times [k]` / `--every n` for `run` and `ensemble`;
`prefix_timeseries.csv` with means, errors, survivor means, running fraction
and S(t). See the README.

## Additional metrics (from the 2026-10-04 metrics review)

Effort: S = under an hour, M = a few hours. Time-dependent metrics (persistence,
domains, activity, correlations) gain most once the ensemble-averaged time
series above exists.

### Coarsening and ordering
- **Domain statistics (M).** Number of domains, largest domain, mean size and
  size distribution, where a domain is a connected group of same-opinion
  nodes (union-find). Real spatial ordering beyond the global m and ρ; key for
  majority-rule freezing and for Axelrod.
- ~~**Persistence (S).**~~ Done 2026-10-04: Model::persistence(), observable `persistence`.
- ~~**Two-time autocorrelation C(t, t_w) (M).**~~ Done 2026-10-05: observables `autocorrelation`, `overlap` with `--tw`.
- ~~**Spatial correlation function and correlation length (M, lattices).**~~ Done 2026-10-05: `Correlations.hpp`, observable `length`, `stationary` ξ and ξ/L, `--correlation` curves.
  Later: an FFT version for lattices beyond ~512² (the direct G(r) is O(N·D·L/2)).
- ~~**Activity (S).**~~ Done 2026-10-04: Model::get_changes(), observable `activity`.

### Stationary states and transitions
- ~~**Histogram of m or of any sampled observable (S).**~~ Done 2026-10-04: `stationary` text histogram + `prefix_histogram.csv`, `--bins`.
- ~~**Specific heat N·var(e)/T² for Ising and Potts (S).**~~ Done 2026-10-04: `stationary` energy and specific heat with errors.
- **Switching and residence times (M).** How often the majority flips in the
  noisy voter (Kirman herding), and the time spent near each consensus.
- ~~**Time to reach a threshold (S).**~~ Done 2026-10-04: `ensemble --threshold x`.

### Many opinions, networks, continuous opinions
- ~~**Effective number of opinions (S).**~~ Done 2026-10-04: Model::opinion_entropy(), effective_number_of_opinions().
- ~~**Segregation (S).**~~ Done 2026-10-04: Model::segregation().
- **Degree-resolved magnetization and activity (M).** Hubs vs leaves; direct
  comparison with heterogeneous mean-field theory.
- ~~**Continuous opinions (S).**~~ Done 2026-10-04: opinion_variance(), bimodality_coefficient(), `prefix_clusters.csv`.

## Further ideas (from the 2026-10-04 scan)

- ~~**Two-parameter sweeps (S–M).**~~ Done 2026-10-17: `--sweep2 key=...`, the
  grid of all combinations in one table; 1D sweeps unchanged byte for byte.
  Original note: A grid over two options (e.g. `--dims` ×
  `--noise`) for finite-size scaling; today a sweep varies one option.
- **Replica offset (S).** `--first-replica k`, so the replicas of one ensemble
  can be split across cluster job arrays and merged reproducibly.
- **Plotting and analysis scripts (M).** Time series, sweep tables with error
  bars, Binder crossings, data collapse. Needs the analysis environment you use
  (the system Python here has no numpy/matplotlib).
- ~~**Save final states (S).**~~ Done 2026-10-17: `--save-states true` writes
  `prefix_states.csv` (per replica / chain / sweep point), which loads back
  with `--init file`. A first step towards checkpointing (no RNG state).
- **Checkpoint/resume (M).** Save and restore the full state (opinions, RNG,
  counters) for runs lasting hours.
- **Faster very large lattices (M).** About 10 sweeps/s at 10^6 nodes, limited
  by memory traffic: try 32-bit neighbour indices or arithmetic neighbours.
  Measure first; compiler flags gave nothing.
- **Wolff cluster algorithm (M).** For Ising/Potts near T_c, only if equilibrium
  criticality is part of the research.
- **Hysteresis / annealing protocols (M).** Slowly varying a parameter, for
  first-order transitions (q-voter q > 5, Potts q > 4).
- ~~**Initial opinions from a file (S).**~~ Done 2026-10-16: `--init file
  --init-path F [--init-fill random|x]`, nodes by edge-list label or index.
- **Static library for non-template code (M).** Edge lists, generators, I/O,
  statistics, driver options into `libodsp.a` if build times grow (see the
  README's header-only section).
- **CI (S).** `.github/workflows/ci.yml` exists locally but is excluded
  (`.git/info/exclude`); enable by removing that line and committing it.

## From the 2026-10-16 benchmark campaign

The benchmark suite (`benchmarks/`) reproduced every known result it checks;
these two weaknesses showed up along the way.

- ~~**Rebalance threads across sweep points (S–M).**~~ Done 2026-10-16:
  `include/Parallel.hpp`, a shared pool where nested loops (a point's replicas
  or chains) borrow idle threads. 3D voter sweep 94.7 s -> 38.3 s, Ising
  stationary sweep 17.3 s -> 7.9 s at 10 threads, identical output.
  Original note: `execute()` in
  `src/driver/Driver.hpp` splits `--threads` across the sweep points up front:
  `concurrent = min(threads, points)` points at a time, each with
  `inner = threads / concurrent` threads. The remainder is never used: 4
  points at 10 threads use 8. When the small points finish, their threads
  stay idle instead of helping the largest one. The 3D voter sweep
  (`dims=8³..20³`) spent its last minutes at 2 of 10 threads. Fix: a shared
  pool of replica (or chain) tasks across all points, e.g. one atomic work
  counter over (point, replica) pairs, keeping results stored by index so the
  output still does not depend on the thread count.
- ~~**Stricter error-bar warning in `stationary` (S).**~~ Dropped
  2026-10-17: the diagnosis was wrong. A calibration (8 independent repeats
  of the L = 64 Ising run at T_c, 10 chains each) showed no bias (U 0.6098 ±
  0.0014, ξ/L 0.899 ± 0.010), and the within-chain error from blocks of
  ~14 τ equals the between-chain error (0.0030 both), so short blocks
  were not shrinking the errors. The scatter across repeats was ~1.3× the
  mean reported error: not significant with 8 repeats. The benchmark's 3σ
  came from a run whose error estimate happened to be low (0.0018). With
  R chains an error bar is itself uncertain by ~1/sqrt(2(R−1)) (±24% for 10),
  so the output now states that ("error bars from 10 chains are themselves
  uncertain by about ±24%"), in single runs and once per sweep.