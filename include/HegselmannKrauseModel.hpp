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
//   synchronous (default, the original model): all agents update together
//       from the same old opinions; one round = one sweep. Deterministic given
//       the initial opinions, and on a complete graph it converges in finitely
//       many rounds, preserving the order of opinions.
//   asynchronous: one random agent updates at a time, using current opinions.
//
// On a complete graph with uniform opinions in [0, 1] it ends in about
// 1/(2 epsilon) clusters, with consensus for epsilon above ~0.2 (for large N).
// Complete graphs use an O(N log N) round (sorted opinions + prefix sums: the
// confidants of an opinion form a contiguous block) instead of O(N^2).
template <typename T = double>
class HegselmannKrauseModel : public BoundedConfidenceModel<T> {
public:
    HegselmannKrauseModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double epsilon,
                          bool synchronous = true, double tolerance = 1e-9, bool track_active_links = false)
        : BoundedConfidenceModel<T>(std::move(graph), rng, epsilon, tolerance, track_active_links),
          _synchronous(synchronous) {}

    bool synchronous() const { return _synchronous; }
    long long rounds() const { return _rounds; }   // synchronous rounds performed

protected:
    bool _within_confidence(double d) const override { return std::fabs(d) <= this->confidence(); }

    // Asynchronous update of one random agent.
    void _single_update() override {
        this->_count_attempts(1);
        const std::size_t i = this->random_node();
        const T old = this->graph().get_state(i);
        const T value = _confidant_mean(i);
        this->_apply(i, value);
        this->_record_change(static_cast<double>(value - old));
    }

    // Synchronous mode: N attempted updates make one round (so a sweep is a
    // round; single updates accumulate until a round is due).
    void _advance_updates(long long n) override {
        if (!_synchronous) { BoundedConfidenceModel<T>::_advance_updates(n); return; }
        const long long N = static_cast<long long>(this->size());
        if (N == 0) return;
        _pending += n;
        while (_pending >= N) {
            _round();
            _pending -= N;
        }
    }

private:
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

    bool                     _synchronous;
    long long                _pending = 0;   // attempted updates not yet turned into a round
    long long                _rounds = 0;
    std::vector<T>           _next;          // scratch: opinions after the round
    std::vector<std::size_t> _order;         // scratch: indices sorted by opinion
    std::vector<T>           _prefix;        // scratch: prefix sums of sorted opinions
};

#endif // ODSP_HEGSELMANNKRAUSEMODEL_HPP
