#ifndef ODSP_HEGSELMANNKRAUSEMODEL_HPP
#define ODSP_HEGSELMANNKRAUSEMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "BoundedConfidenceModel.hpp"

// Hegselmann-Krause bounded-confidence model (Hegselmann & Krause, J. Artif.
// Soc. Soc. Simul. 5(3), 2002). Each agent moves to the AVERAGE opinion of its
// "confidants": itself and the neighbours whose opinion is within epsilon,
//     x_i <- mean{ x_j : j = i or j neighbour of i, |x_j - x_i| <= epsilon }.
//
// Update modes (HKUpdate):
//   Synchronous (default, the original model): all agents update together
//       from the same old opinions; one round = one sweep. Deterministic given
//       the initial opinions, and on a complete graph it converges in finitely
//       many rounds, preserving the order of opinions.
//   Random: one uniformly random agent updates at a time (with replacement),
//       using current opinions; in a sweep of N updates about N/e agents are
//       not updated at all, so some can be left behind by clusters that form
//       and move away within the first sweeps.
//   Sweep: sequential updates in which every sweep visits every agent exactly
//       once, in a fresh uniformly random order (a random permutation per
//       sweep), using current opinions.
//   Ordered: the same with a fixed order, agents 0, 1, ..., N-1 in every sweep
//       (no randomness after the start). With i.i.d. initial opinions the index
//       order is a random order fixed for the whole run: the ordered sweeps of
//       Fortunato (2005). The order matters for the probability of consensus
//       on the complete graph, not only the mode: fixed and fresh orders differ.
//   In Sweep and Ordered modes N updates = one sweep; a partial sweep
//   continues where the last one stopped.
//
// On a complete graph with uniform opinions in [0, 1] it ends in about
// 1/(2 epsilon) clusters, with consensus for epsilon above ~0.2 (for large N).
// Complete graphs use an O(N log N) round (sorted opinions + prefix sums: the
// confidants of an opinion form a contiguous block) instead of O(N^2), and in
// Sweep and Ordered modes a sorted copy of the opinions (the confidants are
// again a block).
enum class HKUpdate { Synchronous, Random, Sweep, Ordered };

template <typename T = double>
class HegselmannKrauseModel : public BoundedConfidenceModel<T> {
public:
    HegselmannKrauseModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double epsilon,
                          HKUpdate update, double tolerance = 1e-9, bool track_active_links = false)
        : BoundedConfidenceModel<T>(std::move(graph), rng, epsilon, tolerance, track_active_links),
          _update(update) {}

    // synchronous = true / false: HKUpdate::Synchronous / HKUpdate::Random.
    HegselmannKrauseModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double epsilon,
                          bool synchronous = true, double tolerance = 1e-9, bool track_active_links = false)
        : HegselmannKrauseModel(std::move(graph), rng, epsilon,
                                synchronous ? HKUpdate::Synchronous : HKUpdate::Random, tolerance,
                                track_active_links) {}

    HKUpdate update() const { return _update; }
    bool synchronous() const { return _update == HKUpdate::Synchronous; }
    long long rounds() const { return _rounds; }   // synchronous rounds performed

protected:
    bool _within_confidence(double d) const override { return std::fabs(d) <= this->confidence(); }

    // Sequential update of one agent: a random one (Random), or the next one of
    // this sweep's order (Sweep: random; Ordered: by index).
    void _single_update() override {
        this->_count_attempts(1);
        const bool sweeps = _update == HKUpdate::Sweep || _update == HKUpdate::Ordered;
        const std::size_t i = sweeps ? _next_in_sweep() : this->random_node();
        const T old = this->graph().get_state(i);
        const bool fast = sweeps && this->graph().is_complete();
        const T value = fast ? _sorted_mean(old) : _confidant_mean(i);
        this->_apply(i, value);
        if (fast) _move_sorted(old, this->graph().get_state(i));   // frozen nodes do not move
        this->_record_change(static_cast<double>(value - old));
    }

    // Synchronous mode: N attempted updates make one round (so a sweep is a
    // round; single updates accumulate until a round is due).
    void _advance_updates(long long n) override {
        if (_update != HKUpdate::Synchronous) { BoundedConfidenceModel<T>::_advance_updates(n); return; }
        const long long N = static_cast<long long>(this->size());
        if (N == 0) return;
        _pending += n;
        while (_pending >= N) {
            _round();
            _pending -= N;
        }
    }

private:
    // Sweep mode: the agents in a random order, reshuffled at the start of each
    // sweep; Ordered mode: in index order. On a complete graph the sorted copy
    // of the opinions is rebuilt there too (and kept sorted by _move_sorted
    // within the sweep).
    std::size_t _next_in_sweep() {
        const std::size_t n = this->size();
        if (_sweep_pos == 0 || _sweep_order.size() != n) {
            _sweep_order.resize(n);
            std::iota(_sweep_order.begin(), _sweep_order.end(), std::size_t{0});
            if (_update == HKUpdate::Sweep) std::shuffle(_sweep_order.begin(), _sweep_order.end(), this->rng());
            _sweep_pos = 0;
            if (this->graph().is_complete()) {
                _sorted = this->graph().states();
                std::sort(_sorted.begin(), _sorted.end());
            }
        }
        const std::size_t i = _sweep_order[_sweep_pos];
        if (++_sweep_pos == n) _sweep_pos = 0;
        return i;
    }

    // Confidant mean of opinion xi on a complete graph: its confidants (itself
    // included) are a contiguous block of the sorted opinions, found by binary
    // search and then adjusted with _within_confidence itself, so the block is
    // exactly the set the definition gives. O(log N + block) per update instead
    // of O(N) through the neighbour view.
    T _sorted_mean(T xi) const {
        const T eps = static_cast<T>(this->confidence());
        auto lo = std::lower_bound(_sorted.begin(), _sorted.end(), xi - eps);
        auto hi = std::upper_bound(_sorted.begin(), _sorted.end(), xi + eps);
        while (lo != _sorted.begin() && this->_within_confidence(static_cast<double>(*(lo - 1) - xi))) --lo;
        while (lo != _sorted.end() && !this->_within_confidence(static_cast<double>(*lo - xi))) ++lo;
        while (hi != _sorted.end() && this->_within_confidence(static_cast<double>(*hi - xi))) ++hi;
        while (hi != lo && !this->_within_confidence(static_cast<double>(*(hi - 1) - xi))) --hi;
        T sum = 0;
        for (auto it = lo; it != hi; ++it) sum += *it;
        return sum / static_cast<T>(hi - lo);
    }

    // Keeps _sorted sorted after one opinion moved from `from` to `to`.
    void _move_sorted(T from, T to) {
        if (from == to) return;
        _sorted.erase(std::lower_bound(_sorted.begin(), _sorted.end(), from));
        _sorted.insert(std::upper_bound(_sorted.begin(), _sorted.end(), to), to);
    }

    T _confidant_mean(std::size_t i) const {
        const BaseGraph<T>& g = this->graph();
        const T xi = g.get_state(i);
        T sum = xi;
        std::size_t count = 1;
        for (std::size_t j : g.neighbours(i)) {
            const T xj = g.get_state(j);
            if (_within_confidence(static_cast<double>(xj - xi))) { sum += xj; ++count; }
        }
        return sum / static_cast<T>(count);
    }

    void _round() {
        const BaseGraph<T>& g = this->graph();
        const std::size_t n = g.size();
        _next.resize(n);
        if (g.is_complete()) _complete_round();
        else for (std::size_t i = 0; i < n; ++i) _next[i] = _confidant_mean(i);

        double max_change = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            max_change = std::max(max_change, static_cast<double>(std::fabs(_next[i] - g.get_state(i))));
            this->_apply(i, _next[i]);
        }
        this->_count_attempts(static_cast<long long>(n));
        this->_record_change(max_change);
        ++_rounds;
    }

    // Complete graph: with opinions sorted, the confidants of the k-th smallest
    // opinion are a contiguous block [lo, hi] of the sorted order (including
    // itself), whose ends only move forward as k grows.
    void _complete_round() {
        const std::vector<T>& x = this->graph().states();
        const std::size_t n = x.size();
        _order.resize(n);
        std::iota(_order.begin(), _order.end(), std::size_t{0});
        std::sort(_order.begin(), _order.end(), [&x](std::size_t a, std::size_t b) { return x[a] < x[b]; });
        _prefix.assign(n + 1, T(0));
        for (std::size_t k = 0; k < n; ++k) _prefix[k + 1] = _prefix[k] + x[_order[k]];

        std::size_t lo = 0, hi = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const T v = x[_order[k]];
            while (!_within_confidence(static_cast<double>(v - x[_order[lo]]))) ++lo;
            if (hi < k) hi = k;
            while (hi + 1 < n && _within_confidence(static_cast<double>(x[_order[hi + 1]] - v))) ++hi;
            _next[_order[k]] = (_prefix[hi + 1] - _prefix[lo]) / static_cast<T>(hi - lo + 1);
        }
    }

    HKUpdate                 _update;
    std::vector<std::size_t> _sweep_order;   // Sweep/Ordered: this sweep's order of the agents
    std::size_t              _sweep_pos = 0; // Sweep/Ordered: next position in _sweep_order
    std::vector<T>           _sorted;        // Sweep/Ordered, complete graph: the opinions, sorted
    long long                _pending = 0;   // attempted updates not yet turned into a round
    long long                _rounds = 0;
    std::vector<T>           _next;          // scratch: opinions after the round
    std::vector<std::size_t> _order;         // scratch: indices sorted by opinion
    std::vector<T>           _prefix;        // scratch: prefix sums of sorted opinions
};

#endif // ODSP_HEGSELMANNKRAUSEMODEL_HPP
