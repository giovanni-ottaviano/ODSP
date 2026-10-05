#ifndef ODSP_MODEL_HPP
#define ODSP_MODEL_HPP

#include <vector>
#include <random>
#include <memory>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "BaseGraph.hpp"
#include "Statistics.hpp"
#include "OpinionCounts.hpp"

// Common machinery shared by every opinion-dynamics model. It owns the
// population, the RNG, the running observables and the step counters, and
// implements the Monte Carlo driver. A concrete model implements exactly one
// method, _single_update(), describing its local rule.
//
// Template-method pattern: advance() calls the virtual _advance_updates(n),
// whose default runs _single_update() n times. Subclasses read the graph
// through the const graph() accessor and mutate it ONLY through _apply(),
// which keeps every observable in sync incrementally -- so no model can
// silently desynchronise the bookkeeping. A model with a smarter algorithm
// (e.g. rejection-free, which skips updates that would change nothing) can
// override _advance_updates() instead.
//
// Time: one "update" is one attempted application of the local rule; one
// Monte Carlo step ("sweep") is N updates, so that each node is updated once
// per sweep on average.
//
// Observables
// -----------
//   opinion counts     : for integer opinion types, the number of nodes holding
//       each opinion, the number of surviving opinions and the largest group.
//       Consensus means "one opinion left", for any number of opinions.
//   magnetization      : sum of the states -- the order parameter of +/-1
//       models (for other labelings, e.g. 0..q-1, it has no meaning).
//   degree-weighted magnetization:
//       sum_i k_i s_i / sum_i k_i. On graphs whose nodes have different
//       degrees this, not m, is the quantity conserved on average by the
//       node-update voter model, so it predicts its exit probability.
//   active-link density: fraction of edges whose endpoints DISAGREE. This is
//       the quantity that actually tracks coarsening: for the voter model the
//       magnetization is conserved in the mean and says nothing about domain
//       growth, whereas rho decays from ~0.5 toward 0 as domains merge, and
//       stays > 0 in the frozen states of the majority model on a lattice.
//
//       rho is maintained incrementally in _apply() using the graph's
//       count_neighbours_in_state(): O(degree) per accepted flip on a sparse
//       graph, O(1) on a complete graph. Pass track_active_links=false to skip
//       it; the accessor then recounts on demand.
//
// Assumption: the incremental active-link update assumes an UNDIRECTED graph
// with no self-loops or repeated neighbours (both hold for RegularLattice with
// its side>=3 guard, and for CompleteGraph). For directed graphs the delta
// bookkeeping would need revisiting.
template <typename T>
class Model {
public:
    // Called by advance() during a run; see advance().
    using Observer = std::function<void(const Model<T>&)>;
    // A scalar measured on the current state, e.g. for sample_stationary().
    using Observable = std::function<double(const Model<T>&)>;
    // Whether opinions are discrete labels with per-opinion counts.
    static constexpr bool discrete_opinions = std::is_integral<T>::value;

    Model(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
          bool track_active_links = true);
    virtual ~Model() = default;

    const BaseGraph<T>& get_graph() const { return *_graph; }

    double get_magnetization() const { return _magnetization; }
    double get_magnetization_per_site() const {
        return _magnetization / static_cast<double>(_graph->size());
    }
    // sum_i k_i s_i / sum_i k_i, in [-1, 1]. Equals m/site on regular graphs.
    double get_degree_weighted_magnetization() const {
        return _total_half_edges == 0 ? 0.0
             : _weighted_magnetization / static_cast<double>(_total_half_edges);
    }

    // Number of disagreeing (undirected) edges, and that count normalised by
    // the total number of edges. rho in [0, 1]; rho == 0 iff consensus.
    long long get_active_links() const {
        return (_track_active_links ? _active_half_edges : _scan_active_half_edges()) / 2;
    }
    double get_active_link_density() const {
        if (_total_half_edges == 0) return 0.0;
        const long long active = _track_active_links ? _active_half_edges : _scan_active_half_edges();
        return static_cast<double>(active) / static_cast<double>(_total_half_edges);
    }

    long long get_edge_count() const { return _total_half_edges / 2; }   // undirected edges

    long long get_mc_steps() const { return _executed_mc_steps; }
    long long get_updates()  const { return _executed_updates; }   // single updates attempted
    // All nodes hold the same opinion. Models with continuous opinions
    // override this with "all opinions within a tolerance".
    virtual bool is_consensus_reached() const;

    // No further (significant) change can happen: consensus for models where
    // it is absorbing; convergence for continuous-opinion models (which may
    // end fragmented). advance() stops here when asked to.
    virtual bool is_absorbing_state() const { return consensus_is_absorbing() && is_consensus_reached(); }

    // --- persistence and activity ---
    // Fraction of nodes whose opinion has not changed since the start (or the
    // last reset_persistence()). Frozen nodes count as persistent.
    double persistence() const {
        const std::size_t n = _graph->size();
        return n == 0 ? 1.0 : 1.0 - static_cast<double>(_changed_count) / static_cast<double>(n);
    }
    void reset_persistence() {
        std::fill(_changed.begin(), _changed.end(), 0);
        _changed_count = 0;
    }
    // Total number of opinion changes so far; divided by get_updates() it is the
    // activity (fraction of attempted updates that changed something).
    long long get_changes() const { return _changes; }

    // --- spread of opinions ---
    // Population variance of the states, and Sarle's bimodality coefficient of
    // their distribution (> 5/9 suggests two peaks); O(N) per call.
    double opinion_variance() const;
    double bimodality_coefficient() const;

    // --- diversity and segregation (integer opinion types) ---
    // Shannon entropy H = -sum p_k ln p_k of the opinion shares, and exp(H): the
    // effective number of opinions (equal shares of q opinions give q; tiny
    // minorities barely count). O(number of labels) per call.
    double opinion_entropy() const;
    double effective_number_of_opinions() const { return std::exp(opinion_entropy()); }
    // Segregation index S = 1 - rho / rho_random, with rho_random = 1 - sum_k
    // n_k (n_k - 1) / (N (N - 1)) the expected rho if the same opinions were
    // placed at random. S = 0: random mixing (exactly 0 on a complete graph);
    // S -> 1: opinions confined to separate regions. At consensus, where
    // rho_random = 0, S is reported as 1 (no mixing at all).
    double segregation() const;

    // --- opinion counts (integer opinion types) ---
    std::size_t number_of_opinions() const { return _opinions.distinct(); }   // opinions still held
    std::size_t count_of(T opinion)  const { return _opinions.count(opinion); }
    // (opinion, number of holders) for every opinion held, in increasing order.
    std::vector<std::pair<T, std::size_t>> opinion_counts() const { return _opinions.sorted(); }
    // Size of the largest opinion group over N (O(number of labels) per call).
    double largest_opinion_fraction() const {
        return _graph->size() == 0 ? 0.0
             : static_cast<double>(_opinions.max_count()) / static_cast<double>(_graph->size());
    }

    // Whether reaching consensus is an absorbing (final) state for this model.
    // True for voter/majority; a noisy model overrides this to false, since a
    // spontaneous event can always leave consensus. The drivers only early-stop
    // on consensus when this is true.
    virtual bool consensus_is_absorbing() const { return true; }

    void set_rng(std::mt19937* rng) { _rng = rng; }

    // --- zealots (frozen nodes) ---
    // A frozen node never changes its opinion: _apply() leaves it untouched, so
    // every model supports zealots / stubborn agents with no model-specific
    // code. Zealots of a single opinion make that consensus the only absorbing
    // state; zealots of BOTH opinions make consensus impossible, and the
    // dynamics has a stationary state instead (measure it with
    // sample_stationary()).
    void set_frozen(std::size_t node, bool frozen = true);
    void set_frozen(const std::vector<std::size_t>& nodes, bool frozen = true);
    bool is_frozen(std::size_t node) const { return !_frozen.empty() && _frozen[node]; }
    std::size_t frozen_count() const { return _frozen_count; }

    // Time-averaged (stationary) order-parameter statistics, for models with a
    // stationary distribution rather than an absorbing state. Runs a burn-in,
    // then records m/site every `interval` sweeps for `n_samples` samples, and
    // returns the time-averaged moments plus susceptibility and Binder cumulant.
    // This is what makes chi/Binder meaningful: they are computed over the
    // fluctuating stationary state, not at a frozen |m| = 1.
    //
    // Errors are block-jackknife estimates over `error_blocks` blocks; they are
    // trustworthy when each block (n_samples / error_blocks samples) is much
    // longer than the autocorrelation times reported alongside them.
    struct StationarySample {
        std::vector<double> m;                 // the m/site samples
        double mean = 0.0, abs_mean = 0.0;     // <m>, <|m|>
        double m2 = 0.0, m4 = 0.0;             // <m^2>, <m^4>
        double susceptibility = 0.0;           // N (<m^2> - <m>^2)
        double abs_susceptibility = 0.0;       // N (<m^2> - <|m|>^2): the usual finite-size choice,
                                               // since <m> -> 0 by symmetry in a long enough run
        double binder = 0.0;                   // 1 - <m^4> / (3 <m^2>^2)

        // Standard errors (block jackknife).
        double mean_err = 0.0, abs_mean_err = 0.0;
        double susceptibility_err = 0.0, abs_susceptibility_err = 0.0, binder_err = 0.0;

        // Integrated autocorrelation times of m and |m|, in units of the
        // sampling interval (0.5 = uncorrelated). tau(m) is usually much longer
        // than tau(|m|), because m flips sign only rarely in the ordered phase.
        double tau_m = 0.5, tau_abs_m = 0.5;
    };
    StationarySample sample_stationary(long long burn_in_sweeps, int n_samples,
                                       int interval_sweeps = 1, bool use_mc_steps = true,
                                       int error_blocks = 20);

    // Same, for any scalar observable instead of m/site (e.g. the Potts order
    // parameter, or the share of one opinion). The samples land in
    // StationarySample::m and all statistics refer to them; "susceptibility"
    // is then N times the variance of the observable.
    StationarySample sample_stationary(const Observable& observable, long long burn_in_sweeps,
                                       int n_samples, int interval_sweeps = 1,
                                       bool use_mc_steps = true, int error_blocks = 20);

    // Advance the dynamics for `steps` iterations (a sweep of N updates if
    // use_mc_steps, else a single update). If stop_when_absorbed, it stops
    // early once is_absorbing_state() holds (consensus, for models where
    // consensus is absorbing). If an observer is
    // given, it is called after every `observe_every` iterations (and not at
    // the start). Returns whether consensus currently holds. run_montecarlo()
    // and sample_stationary() are built on this.
    bool advance(long long steps, bool use_mc_steps = true, bool stop_when_absorbed = true,
                 const Observer& observer = nullptr, long long observe_every = 1);

    // Run for `steps` iterations. Stops early on consensus; logs every
    // log_percentage %. Set log_percentage > 100 to suppress the periodic
    // lines (only the final summary prints).
    void run_montecarlo(long long steps, int log_percentage = 10, bool use_mc_steps = true);

protected:
    // Helpers for subclasses' _single_update():
    std::mt19937&       rng()         { return *_rng; }
    const BaseGraph<T>& graph() const { return *_graph; }
    std::size_t         size()  const { return _graph->size(); }
    std::size_t         random_node();
    std::size_t         random_neighbour(std::size_t node);   // node must have degree > 0
    // Uniformly random directed edge (i, j) -- equivalently a random edge with a
    // random orientation. Node i is chosen with probability k_i / sum k. The
    // graph must have at least one edge.
    std::pair<std::size_t, std::size_t> random_directed_edge();
    // Sum of the neighbours' spins, for +/-1 states: 2 (# up neighbours) - degree.
    // O(1) on a complete graph, O(degree) otherwise.
    long long           neighbour_spin_sum(std::size_t node) const;
    void                _apply(std::size_t node, T new_value);   // the only way to mutate state;
                                                                 // a no-op on frozen nodes

    // Called after the set of frozen nodes changes, for models that cache
    // per-node data depending on it.
    virtual void _on_frozen_changed() {}

    // For models defined only for +/-1 opinions: throws std::invalid_argument
    // naming the first node holding anything else.
    void _require_binary_states(const char* model_name) const {
        for (std::size_t i = 0; i < _graph->size(); ++i) {
            const T s = _graph->get_state(i);
            if (s != T(1) && s != T(-1))
                throw std::invalid_argument(std::string(model_name) + ": opinions must be +1 or -1, but node " +
                                            std::to_string(i) + " holds " + std::to_string(s) + ".");
        }
    }

    // The model-specific local rule: perform one attempted update.
    virtual void _single_update() = 0;

    // Perform n attempted updates. The default runs _single_update() n times;
    // override only to simulate the same dynamics faster. The step counters
    // are maintained by the caller.
    virtual void _advance_updates(long long n) {
        for (long long i = 0; i < n; ++i) _single_update();
    }

private:
    void      _compute_magnetizations();
    long long _count_half_edges() const;
    long long _scan_active_half_edges() const;
    void      _step(bool use_mc_steps);

    std::unique_ptr<BaseGraph<T>> _graph;
    double    _magnetization;
    double    _weighted_magnetization;   // sum_i k_i s_i
    bool      _track_active_links;
    long long _active_half_edges;   // directed half-edges with disagreeing ends (2x undirected)
    long long _total_half_edges;    // sum of degrees (constant while topology is fixed)
    long long _executed_mc_steps;
    long long _executed_updates;
    std::mt19937* _rng;

    struct NoCounts {};   // non-integer opinions: no per-opinion counts
    std::conditional_t<discrete_opinions, OpinionCounts<T>, NoCounts> _opinions;

    std::vector<char> _changed;         // per node: changed since the start / last reset
    std::size_t       _changed_count = 0;
    long long         _changes = 0;     // total opinion changes

    std::vector<char> _frozen;          // empty => no frozen nodes
    std::size_t       _frozen_count = 0;

    // For random_directed_edge(): cumulative degrees, built on first use (the
    // topology is fixed); left empty for regular graphs, where a uniform node
    // then a uniform neighbour already gives a uniform directed edge.
    bool                            _degree_table_built = false;
    std::vector<unsigned long long> _cumulative_degree;
};

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

template <typename T>
Model<T>::Model(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, bool track_active_links)
    : _graph(std::move(graph)), _magnetization(0.0), _weighted_magnetization(0.0),
      _track_active_links(track_active_links), _active_half_edges(0), _total_half_edges(0),
      _executed_mc_steps(0), _executed_updates(0), _rng(rng)
{
    _compute_magnetizations();
    if constexpr (discrete_opinions) _opinions.reset(_graph->states());
    _changed.assign(_graph->size(), 0);
    _total_half_edges = _count_half_edges();
    if (_track_active_links) _active_half_edges = _scan_active_half_edges();
}

template <typename T>
void Model<T>::_compute_magnetizations() {
    _magnetization = 0.0;
    _weighted_magnetization = 0.0;
    for (std::size_t i = 0; i < _graph->size(); ++i) {
        const double s = static_cast<double>(_graph->get_state(i));
        _magnetization += s;
        _weighted_magnetization += static_cast<double>(_graph->degree(i)) * s;
    }
}

template <typename T>
long long Model<T>::_count_half_edges() const {
    long long total = 0;
    for (std::size_t i = 0; i < _graph->size(); ++i)
        total += static_cast<long long>(_graph->degree(i));
    return total;
}

template <typename T>
long long Model<T>::_scan_active_half_edges() const {
    long long active = 0;
    for (std::size_t i = 0; i < _graph->size(); ++i)
        active += static_cast<long long>(_graph->degree(i))
                - static_cast<long long>(_graph->count_neighbours_in_state(i, _graph->get_state(i)));
    return active;
}

template <typename T>
double Model<T>::opinion_variance() const {
    const std::vector<T>& s = _graph->states();
    if (s.empty()) return 0.0;
    const double mean = _magnetization / static_cast<double>(s.size());
    double var = 0.0;
    for (const T& v : s) var += (static_cast<double>(v) - mean) * (static_cast<double>(v) - mean);
    return var / static_cast<double>(s.size());
}

template <typename T>
double Model<T>::bimodality_coefficient() const {
    const std::vector<T>& s = _graph->states();
    return stats::bimodality_coefficient(std::vector<double>(s.begin(), s.end()));
}

template <typename T>
double Model<T>::opinion_entropy() const {
    const double n = static_cast<double>(_graph->size());
    double h = 0.0;
    for (const auto& entry : _opinions.sorted()) {
        const double p = static_cast<double>(entry.second) / n;
        h -= p * std::log(p);
    }
    return h;
}

template <typename T>
double Model<T>::segregation() const {
    const double n = static_cast<double>(_graph->size());
    if (n < 2.0) return 1.0;
    double same = 0.0;   // ordered pairs of distinct nodes with the same opinion
    for (const auto& entry : _opinions.sorted()) {
        const double c = static_cast<double>(entry.second);
        same += c * (c - 1.0);
    }
    const double rho_random = 1.0 - same / (n * (n - 1.0));
    if (rho_random <= 0.0) return 1.0;
    return 1.0 - get_active_link_density() / rho_random;
}

template <typename T>
bool Model<T>::is_consensus_reached() const {
    if (_graph->size() == 0) return false;
    if constexpr (discrete_opinions) {
        return _opinions.distinct() == 1;
    } else {
        const std::vector<T>& s = _graph->states();
        return std::all_of(s.begin(), s.end(), [&s](const T& v) { return v == s.front(); });
    }
}

template <typename T>
std::size_t Model<T>::random_node() {
    std::uniform_int_distribution<std::size_t> d(0, _graph->size() - 1);
    return d(*_rng);
}

template <typename T>
std::size_t Model<T>::random_neighbour(std::size_t node) {
    const NeighbourView nb = _graph->neighbours(node);
    std::uniform_int_distribution<std::size_t> d(0, nb.size() - 1);
    return nb[d(*_rng)];
}

template <typename T>
void Model<T>::set_frozen(std::size_t node, bool frozen) {
    if (_frozen.empty()) {
        if (!frozen) return;
        _frozen.assign(_graph->size(), 0);
    }
    if (static_cast<bool>(_frozen[node]) != frozen) {
        _frozen[node] = frozen ? 1 : 0;
        if (frozen) ++_frozen_count;
        else        --_frozen_count;
    }
    _on_frozen_changed();
}

template <typename T>
void Model<T>::set_frozen(const std::vector<std::size_t>& nodes, bool frozen) {
    if (nodes.empty()) return;
    if (_frozen.empty() && frozen) _frozen.assign(_graph->size(), 0);
    if (_frozen.empty()) return;
    for (std::size_t node : nodes) {
        if (static_cast<bool>(_frozen[node]) == frozen) continue;
        _frozen[node] = frozen ? 1 : 0;
        if (frozen) ++_frozen_count;
        else        --_frozen_count;
    }
    _on_frozen_changed();
}

template <typename T>
std::pair<std::size_t, std::size_t> Model<T>::random_directed_edge() {
    if (!_degree_table_built) {
        _degree_table_built = true;
        const std::size_t n = _graph->size();
        bool regular = true;
        unsigned long long total = 0;
        _cumulative_degree.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            total += _graph->degree(i);
            _cumulative_degree[i] = total;
            if (_graph->degree(i) != _graph->degree(0)) regular = false;
        }
        if (regular) _cumulative_degree.clear();
    }

    std::size_t i;
    if (_cumulative_degree.empty()) {
        i = random_node();
    } else {
        std::uniform_int_distribution<unsigned long long> d(0, _cumulative_degree.back() - 1);
        const unsigned long long r = d(*_rng);
        i = static_cast<std::size_t>(
            std::upper_bound(_cumulative_degree.begin(), _cumulative_degree.end(), r) - _cumulative_degree.begin());
    }
    return {i, random_neighbour(i)};
}

template <typename T>
long long Model<T>::neighbour_spin_sum(std::size_t node) const {
    return 2 * static_cast<long long>(_graph->count_neighbours_in_state(node, T(1)))
         - static_cast<long long>(_graph->degree(node));
}

template <typename T>
void Model<T>::_apply(std::size_t node, T new_value) {
    const T old = _graph->get_state(node);
    if (new_value == old || is_frozen(node)) return;

    if (_track_active_links) {
        // Edge (node, j) is active iff state(j) != state(node). Moving node from
        // `old` to `new_value` activates its edges to neighbours holding `old`
        // and deactivates those to neighbours holding `new_value`; each edge
        // counts as two half-edges.
        const long long c_old = static_cast<long long>(_graph->count_neighbours_in_state(node, old));
        const long long c_new = static_cast<long long>(_graph->count_neighbours_in_state(node, new_value));
        _active_half_edges += 2 * (c_old - c_new);
    }

    const double ds = static_cast<double>(new_value) - static_cast<double>(old);
    _magnetization += ds;
    if constexpr (discrete_opinions) _opinions.move(old, new_value);
    ++_changes;
    if (!_changed[node]) { _changed[node] = 1; ++_changed_count; }
    _weighted_magnetization += static_cast<double>(_graph->degree(node)) * ds;
    _graph->set_state(node, new_value);
}

template <typename T>
void Model<T>::_step(bool use_mc_steps) {
    const long long n = use_mc_steps ? static_cast<long long>(_graph->size()) : 1;
    _advance_updates(n);
    _executed_updates += n;
    if (use_mc_steps) ++_executed_mc_steps;
}

template <typename T>
bool Model<T>::advance(long long steps, bool use_mc_steps, bool stop_when_absorbed,
                       const Observer& observer, long long observe_every) {
    if (observe_every < 1) observe_every = 1;
    for (long long i = 1; i <= steps; ++i) {
        _step(use_mc_steps);
        if (observer && i % observe_every == 0) observer(*this);
        if (stop_when_absorbed && is_absorbing_state()) break;
    }
    return is_consensus_reached();
}

template <typename T>
typename Model<T>::StationarySample
Model<T>::sample_stationary(long long burn_in_sweeps, int n_samples,
                            int interval_sweeps, bool use_mc_steps, int error_blocks) {
    return sample_stationary([](const Model<T>& model) { return model.get_magnetization_per_site(); },
                             burn_in_sweeps, n_samples, interval_sweeps, use_mc_steps, error_blocks);
}

template <typename T>
typename Model<T>::StationarySample
Model<T>::sample_stationary(const Observable& observable, long long burn_in_sweeps, int n_samples,
                            int interval_sweeps, bool use_mc_steps, int error_blocks) {
    StationarySample s;
    if (n_samples <= 0) return s;
    if (interval_sweeps < 1) interval_sweeps = 1;

    advance(burn_in_sweeps, use_mc_steps, /*stop_when_absorbed=*/false);
    s.m.reserve(n_samples);
    advance(static_cast<long long>(n_samples) * interval_sweeps, use_mc_steps, /*stop_when_absorbed=*/false,
            [&s, &observable](const Model<T>& model) { s.m.push_back(observable(model)); },
            interval_sweeps);

    const double N = static_cast<double>(get_graph().size());
    auto chi      = [N](const stats::Moments& mo) { return N * (mo.m2 - mo.m1 * mo.m1); };
    auto chi_abs  = [N](const stats::Moments& mo) { return N * (mo.m2 - mo.abs_m1 * mo.abs_m1); };
    auto binder   = [](const stats::Moments& mo) {
        return mo.m2 > 0.0 ? 1.0 - mo.m4 / (3.0 * mo.m2 * mo.m2) : 0.0;
    };

    const stats::Moments mo = stats::moments(s.m);
    s.mean = mo.m1; s.abs_mean = mo.abs_m1; s.m2 = mo.m2; s.m4 = mo.m4;
    s.susceptibility     = chi(mo);
    s.abs_susceptibility = chi_abs(mo);
    s.binder             = binder(mo);

    s.mean_err     = stats::jackknife_error(s.m, error_blocks, [](const stats::Moments& m) { return m.m1; });
    s.abs_mean_err = stats::jackknife_error(s.m, error_blocks, [](const stats::Moments& m) { return m.abs_m1; });
    s.susceptibility_err     = stats::jackknife_error(s.m, error_blocks, chi);
    s.abs_susceptibility_err = stats::jackknife_error(s.m, error_blocks, chi_abs);
    s.binder_err             = stats::jackknife_error(s.m, error_blocks, binder);

    std::vector<double> abs_m(s.m.size());
    std::transform(s.m.begin(), s.m.end(), abs_m.begin(), [](double v) { return std::fabs(v); });
    s.tau_m     = stats::integrated_autocorrelation_time(s.m);
    s.tau_abs_m = stats::integrated_autocorrelation_time(abs_m);
    return s;
}

template <typename T>
void Model<T>::run_montecarlo(long long steps, int log_percentage, bool use_mc_steps) {
    if (steps <= 0) return;
    if (log_percentage <= 0) log_percentage = 100;

    const long long start = use_mc_steps ? _executed_mc_steps : _executed_updates;
    const long long log_interval = std::max<long long>(1, steps * log_percentage / 100);
    auto log = [&](const Model<T>& m) {
        const long long done = (use_mc_steps ? m.get_mc_steps() : m.get_updates()) - start;
        std::cout << "  [" << static_cast<int>(100 * done / steps) << "%] m/site = "
                  << m.get_magnetization_per_site() << "   rho = " << m.get_active_link_density() << '\n';
    };
    const Observer observer = log_percentage <= 100 ? Observer(log) : Observer();

    const bool reached = advance(steps, use_mc_steps, /*stop_when_absorbed=*/true, observer, log_interval);
    if (reached && consensus_is_absorbing()) {
        std::cout << "Consensus reached after "
                  << (use_mc_steps ? _executed_mc_steps : _executed_updates)
                  << (use_mc_steps ? " sweeps" : " updates")
                  << " (rho = " << get_active_link_density() << ").\n";
    } else {
        std::cout << "Stopped without consensus. Final m/site = " << get_magnetization_per_site()
                  << ", rho = " << get_active_link_density() << ".\n";
    }
}

#endif // ODSP_MODEL_HPP
