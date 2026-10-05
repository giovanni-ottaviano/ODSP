#ifndef ODSP_ENSEMBLE_HPP
#define ODSP_ENSEMBLE_HPP

#include <vector>
#include <functional>
#include <memory>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#include "Model.hpp"

// Runs many independent replicas of a model and aggregates the ENSEMBLE-level
// observables -- the ones that are only defined across a collection of runs, not
// within a single trajectory: consensus-time distribution, exit (fixation)
// probability, survival fraction, and the moments of the order parameter.
//
// It sits strictly ABOVE Model and never reaches into a model's internals. The
// only thing it needs is a way to manufacture a fresh replica, because each run
// requires its own graph, its own random initial condition, and its own RNG.
// That is the ModelFactory: given a per-replica RNG, return a ready-to-run
// Model. Example:
//
//   auto factory = [dims, N](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
//       auto g = std::make_unique<RegularLattice<int>>(
//                    create_random_lattice(N, 0.5, rng), dims);
//       return std::make_unique<VoterModel<int>>(std::move(g), &rng);
//   };
//   Ensemble<int> ens(factory, /*seed=*/12345);
//   auto res = ens.run(/*replicas=*/500, /*max_steps=*/30000, true, /*threads=*/0);
//
// Parallelism: replicas run on `threads` worker threads. Every replica's seed
// is drawn up front from the ensemble seed and results are stored by replica
// index, so the Result is IDENTICAL for any number of threads. With threads > 1
// the factory is called concurrently, so it must not modify shared state
// (capturing parameters by value, as above, is safe).
// Number of worker threads for a requested count. threads <= 0 means "all
// cores": std::thread::hardware_concurrency(), capped by the environment
// variable ODSP_MAX_THREADS when it holds a positive integer (e.g. to keep some
// cores free on a shared machine: export ODSP_MAX_THREADS=10). An explicit
// positive count is returned unchanged.
inline int resolve_threads(int requested) {
    if (requested > 0) return requested;
    int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (const char* cap = std::getenv("ODSP_MAX_THREADS")) {
        char* end = nullptr;
        const long v = std::strtol(cap, &end, 10);
        if (end != cap && *end == '\0' && v > 0) n = static_cast<int>(std::min<long>(n, v));
    }
    return n;
}

template <typename T>
class Ensemble {
public:
    using ModelFactory = std::function<std::unique_ptr<Model<T>>(std::mt19937&)>;

    struct Result {
        int replicas = 0;
        long long max_steps = 0;         // the cap each replica ran for
        int consensus_count = 0;         // replicas that reached consensus within the cap
        double survival_fraction = 0.0;  // fraction that did NOT (censored / frozen)

        // Consensus time over the replicas that reached consensus (in replica
        // order), in the unit the run advanced by: sweeps if use_mc_steps,
        // single updates otherwise.
        std::vector<long long> consensus_times;

        // Mean and stddev over the replicas that reached consensus. CAUTION:
        // if some replicas were censored by the cap, this mean is biased LOW,
        // because the slowest runs are exactly the ones left out. Prefer the
        // median (robust to censoring as long as most runs finish) and the
        // survival curve, or raise the cap until survival_fraction is ~0.
        double mean_consensus_time = 0.0;
        double stddev_consensus_time = 0.0;

        // Time by which half of ALL replicas reached consensus; NaN if fewer
        // than half did within the cap. Unaffected by censoring otherwise.
        double median_consensus_time = std::numeric_limits<double>::quiet_NaN();

        // Mean over ALL replicas, counting censored ones as max_steps: a lower
        // bound on the true mean consensus time.
        double mean_consensus_time_lower_bound = 0.0;

        // Survival function S(t): fraction of replicas still without consensus
        // at time t (same unit as consensus_times). Defined for t <= max_steps.
        double survival(long long t) const {
            if (replicas == 0) return 0.0;
            const auto done = std::count_if(consensus_times.begin(), consensus_times.end(),
                                            [t](long long c) { return c <= t; });
            return 1.0 - static_cast<double>(done) / replicas;
        }

        // Fixation: how many replicas reached consensus on each opinion, as
        // (opinion, replicas) in increasing opinion order. For the voter model
        // on a regular graph, P(opinion k wins) equals its initial share (with
        // any number of opinions) -- a clean analytic check.
        std::vector<std::pair<T, int>> fixation_counts;
        // Among replicas that reached consensus, the fraction won by `opinion`.
        double fixation_probability(T opinion) const {
            if (consensus_count == 0) return 0.0;
            for (const auto& entry : fixation_counts)
                if (entry.first == opinion) return static_cast<double>(entry.second) / consensus_count;
            return 0.0;
        }
        // fixation_probability(+1): the exit probability of binary models.
        double exit_prob_up = 0.0;

        // Mean number of opinions still held when the replicas stopped
        // (integer opinion types; 1 for a replica at consensus).
        double mean_final_opinions = 0.0;

        // Moments of the final order parameter m/site across ALL replicas, plus
        // the two quantities usually built from them. NOTE: for purely absorbing
        // models measured at consensus, |m| = 1, so these are trivial; for models
        // with a stationary state use Model::sample_stationary() instead.
        double m1 = 0.0, m2 = 0.0, m4 = 0.0;     // <m>, <m^2>, <m^4>
        double susceptibility = 0.0;             // N (<m^2> - <m>^2)
        double binder = 0.0;                     // 1 - <m^4> / (3 <m^2>^2)

        double mean_final_rho = 0.0;             // mean active-link density at stop

        // The optional final observable of run(), evaluated on every replica
        // when it stops (replica order), with its mean and stddev. E.g. the
        // number of opinion clusters, or the time to convergence.
        std::vector<double> final_values;
        double mean_final_value = 0.0;
        double stddev_final_value = 0.0;

        // With several final observables (the vector overload of run()):
        // final_observables[k][r] is observable k on replica r. final_values
        // and its mean/stddev then refer to the first observable.
        std::vector<std::vector<double>> final_observables;

        // Per replica, in replica order: when it stopped (same unit as
        // consensus_times) and whether it had reached consensus.
        std::vector<long long> stop_times;
        std::vector<char>      reached;

        // True for models that can leave consensus (consensus_is_absorbing()
        // false): consensus_count and consensus_times then count the FIRST
        // passage to consensus, every replica runs to the cap, and there is no
        // fixation (fixation_counts empty, exit_prob_up NaN).
        bool first_passage = false;
    };

    // Measured on each replica's final state (see Result::final_values).
    using FinalObservable = std::function<double(const Model<T>&)>;

    // Optional per-replica callbacks, e.g. to record first-passage times of
    // custom conditions or to keep per-replica data beyond scalars:
    //   step(r)   returns an observer for replica r, called at the start and
    //             after every step (sweep or update) of that replica;
    //   finish    is called with (r, final state) when replica r stops.
    // Each replica runs on one thread, so writing to per-replica slots indexed
    // by r is safe.
    struct Hooks {
        std::function<typename Model<T>::Observer(int replica)> step;
        std::function<void(int replica, const Model<T>&)>       finish;
    };

    explicit Ensemble(ModelFactory factory, unsigned seed = std::random_device{}())
        : _factory(std::move(factory)), _seed_rng(seed) {}

    // Each replica advances for at most max_steps iterations (sweeps if
    // use_mc_steps, single updates otherwise), stopping early in an absorbing
    // state (Model::is_absorbing_state). threads <= 0 uses all hardware
    // threads; the result does not depend on the thread count.
    Result run(int replicas, long long max_steps, bool use_mc_steps = true, int threads = 1,
               const FinalObservable& final_observable = nullptr);

    // Same, measuring several observables on each replica's final state, with
    // optional per-replica hooks.
    Result run(int replicas, long long max_steps, bool use_mc_steps, int threads,
               const std::vector<FinalObservable>& final_observables, const Hooks& hooks = Hooks());

private:
    struct ReplicaOutcome {
        bool        reached = false;       // reached consensus (first passage for non-absorbing models)
        bool        first_passage = false; // the model can leave consensus
        long long   consensus_time = 0;    // when consensus was (first) reached
        long long   stop_time = 0;         // when the replica stopped
        double      m = 0.0;
        double      rho = 0.0;
        std::size_t nodes = 0;
        T           winner{};          // the consensus opinion, if reached
        std::size_t opinions = 0;      // opinions held at the stop (integer opinion types)
        std::vector<double> values;    // final observables
    };

    ModelFactory _factory;
    std::mt19937 _seed_rng;   // spawns one independent seed per replica
};

// ---------------------------------------------------------------------------
// Definition
// ---------------------------------------------------------------------------

template <typename T>
typename Ensemble<T>::Result
Ensemble<T>::run(int replicas, long long max_steps, bool use_mc_steps, int threads,
                 const FinalObservable& final_observable) {
    std::vector<FinalObservable> list;
    if (final_observable) list.push_back(final_observable);
    return run(replicas, max_steps, use_mc_steps, threads, list);
}

template <typename T>
typename Ensemble<T>::Result
Ensemble<T>::run(int replicas, long long max_steps, bool use_mc_steps, int threads,
                 const std::vector<FinalObservable>& final_observables, const Hooks& hooks) {
    Result res;
    res.max_steps = max_steps;
    if (replicas <= 0) return res;
    res.replicas = replicas;

    // Seeds first, in replica order, so the outcome is thread-count independent.
    std::vector<unsigned> seeds(replicas);
    for (unsigned& s : seeds) s = _seed_rng();

    std::vector<ReplicaOutcome> outcomes(replicas);
    auto run_replica = [&](int r) {
        // Distinct, reproducible RNG per replica; it outlives the model it backs.
        std::mt19937 rng(seeds[r]);
        std::unique_ptr<Model<T>> model = _factory(rng);
        ReplicaOutcome& out = outcomes[r];
        out.nodes   = model->get_graph().size();
        // get_mc_steps() stays 0 when advancing by single updates.
        auto now = [use_mc_steps](const Model<T>& m) { return use_mc_steps ? m.get_mc_steps() : m.get_updates(); };
        const typename Model<T>::Observer user = hooks.step ? hooks.step(r) : typename Model<T>::Observer();
        if (user) user(*model);   // the start
        if (model->consensus_is_absorbing()) {
            out.reached = model->advance(max_steps, use_mc_steps, /*stop_when_absorbed=*/true, user);
            out.consensus_time = now(*model);
        } else {
            // Consensus can be left again: record the FIRST time it is
            // reached (checked at the start and after every step) and run to
            // the cap, so the final observables are measured there.
            out.first_passage = true;
            if (model->is_consensus_reached()) { out.reached = true; out.consensus_time = now(*model); }
            model->advance(max_steps, use_mc_steps, /*stop_when_absorbed=*/false,
                           [&out, &now, &user](const Model<T>& m) {
                               if (!out.reached && m.is_consensus_reached()) {
                                   out.reached = true;
                                   out.consensus_time = now(m);
                               }
                               if (user) user(m);
                           });
        }
        out.stop_time = now(*model);
        out.m       = model->get_magnetization_per_site();
        out.rho     = model->get_active_link_density();
        if (out.reached && !out.first_passage && out.nodes > 0) out.winner = model->get_graph().get_state(0);
        if constexpr (Model<T>::discrete_opinions) out.opinions = model->number_of_opinions();
        for (const FinalObservable& f : final_observables) out.values.push_back(f(*model));
        if (hooks.finish) hooks.finish(r, *model);
    };

    threads = resolve_threads(threads);
    threads = std::min(threads, replicas);

    if (threads == 1) {
        for (int r = 0; r < replicas; ++r) run_replica(r);
    } else {
        std::atomic<int>   next{0};
        std::exception_ptr error;
        std::mutex         error_mutex;
        auto worker = [&]() {
            for (int r = next++; r < replicas; r = next++) {
                try {
                    run_replica(r);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                    next = replicas;   // stop handing out work
                }
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (std::thread& t : pool) t.join();
        if (error) std::rethrow_exception(error);
    }

    // Aggregate in replica order.
    double node_count = 0.0, sum_rho = 0.0, sum_capped_time = 0.0, sum_opinions = 0.0;
    for (const ReplicaOutcome& o : outcomes) {
        node_count = static_cast<double>(o.nodes);
        sum_rho += o.rho;
        sum_opinions += static_cast<double>(o.opinions);
        const double m = o.m, m2 = m * m;
        res.m1 += m; res.m2 += m2; res.m4 += m2 * m2;
        if (o.reached) {
            ++res.consensus_count;
            res.consensus_times.push_back(o.consensus_time);
            sum_capped_time += static_cast<double>(o.consensus_time);
            if (!o.first_passage) {   // a fixed outcome only where consensus is absorbing
                auto it = std::find_if(res.fixation_counts.begin(), res.fixation_counts.end(),
                                       [&o](const std::pair<T, int>& e) { return e.first == o.winner; });
                if (it == res.fixation_counts.end()) res.fixation_counts.emplace_back(o.winner, 1);
                else ++it->second;
            }
        } else {
            sum_capped_time += static_cast<double>(max_steps);
        }
    }

    res.survival_fraction = 1.0 - static_cast<double>(res.consensus_count) / replicas;
    res.mean_final_rho    = sum_rho / replicas;
    res.mean_consensus_time_lower_bound = sum_capped_time / replicas;
    res.mean_final_opinions = sum_opinions / replicas;
    std::sort(res.fixation_counts.begin(), res.fixation_counts.end());
    res.first_passage = outcomes.front().first_passage;
    res.exit_prob_up = res.first_passage ? std::numeric_limits<double>::quiet_NaN() : res.fixation_probability(T(1));

    // Consensus-time mean / stddev (over the runs that reached consensus).
    if (!res.consensus_times.empty()) {
        const double n = static_cast<double>(res.consensus_times.size());
        double s = 0.0, s2 = 0.0;
        for (long long t : res.consensus_times) { s += t; s2 += static_cast<double>(t) * t; }
        res.mean_consensus_time = s / n;
        const double var = std::max(0.0, s2 / n - res.mean_consensus_time * res.mean_consensus_time);
        res.stddev_consensus_time = std::sqrt(var);
    }

    // Median over ALL replicas (censored ones count as +infinity): the time by
    // which ceil(R/2) replicas have reached consensus.
    const int half = (replicas + 1) / 2;
    if (res.consensus_count >= half) {
        std::vector<long long> sorted = res.consensus_times;
        std::nth_element(sorted.begin(), sorted.begin() + (half - 1), sorted.end());
        res.median_consensus_time = static_cast<double>(sorted[half - 1]);
    }

    res.final_observables.assign(final_observables.size(), std::vector<double>());
    for (std::size_t k = 0; k < final_observables.size(); ++k) {
        res.final_observables[k].reserve(replicas);
        for (const ReplicaOutcome& o : outcomes) res.final_observables[k].push_back(o.values[k]);
    }
    if (!final_observables.empty()) {
        res.final_values = res.final_observables[0];
        double sv = 0.0, sv2 = 0.0;
        for (double v : res.final_values) { sv += v; sv2 += v * v; }
        res.mean_final_value = sv / replicas;
        res.stddev_final_value = std::sqrt(std::max(0.0, sv2 / replicas - res.mean_final_value * res.mean_final_value));
    }
    for (const ReplicaOutcome& o : outcomes) {
        res.stop_times.push_back(o.stop_time);
        res.reached.push_back(o.reached ? 1 : 0);
    }

    // Order-parameter moments across replicas.
    res.m1 /= replicas; res.m2 /= replicas; res.m4 /= replicas;
    res.susceptibility = node_count * (res.m2 - res.m1 * res.m1);
    if (res.m2 > 0.0) res.binder = 1.0 - res.m4 / (3.0 * res.m2 * res.m2);

    return res;
}

#endif // ODSP_ENSEMBLE_HPP
