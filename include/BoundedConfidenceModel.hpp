#ifndef ODSP_BOUNDEDCONFIDENCEMODEL_HPP
#define ODSP_BOUNDEDCONFIDENCEMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Model.hpp"
#include "OpinionClusters.hpp"

// Common base of the continuous-opinion, bounded-confidence models
// (DeffuantModel, HegselmannKrauseModel): opinions are real numbers, and two
// agents influence each other only if their opinions differ by less than the
// confidence bound epsilon.
//
// Such dynamics never reach consensus exactly -- opinions converge
// asymptotically, and may split into several clusters too far apart to
// interact. So, with a `tolerance`:
//   * consensus      : every opinion within `tolerance` of every other;
//   * converged      : every pair of neighbours that can still interact agrees
//                      within `tolerance` -- consensus or a frozen set of
//                      separated clusters. This is is_absorbing_state(), where
//                      advance() stops.
// The convergence test costs O(edges), or O(N log N) on a complete graph (it
// sorts the opinions), so it only runs after a full sweep's worth of attempts
// has changed no opinion by more than `tolerance`.
//
// Observables: opinion_range(), clusters() (see OpinionClusters.hpp), and the
// inherited get_magnetization_per_site(), which here is the MEAN OPINION.
// Active-link tracking is off by default (with real-valued opinions almost
// every edge "disagrees", so it carries no information).
template <typename T = double>
class BoundedConfidenceModel : public Model<T> {
    static_assert(std::is_floating_point<T>::value,
                  "BoundedConfidenceModel: opinions must be a floating-point type.");

public:
    BoundedConfidenceModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double epsilon,
                           double tolerance, bool track_active_links)
        : Model<T>(std::move(graph), rng, track_active_links), _epsilon(epsilon), _tolerance(tolerance) {
        if (!(epsilon > 0.0)) throw std::invalid_argument("bounded confidence: epsilon must be > 0.");
        if (!(tolerance > 0.0)) throw std::invalid_argument("bounded confidence: tolerance must be > 0.");
    }

    double confidence() const { return _epsilon; }
    double tolerance()  const { return _tolerance; }

    double opinion_range() const {
        const std::vector<T>& x = this->graph().states();
        if (x.empty()) return 0.0;
        const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
        return static_cast<double>(*hi - *lo);
    }

    std::vector<OpinionCluster> clusters(double gap = 1e-3) const {
        const std::vector<T>& x = this->graph().states();
        return opinion_clusters(std::vector<double>(x.begin(), x.end()), gap);
    }

    bool is_consensus_reached() const override {
        return this->size() > 0 && opinion_range() <= _tolerance;
    }

    bool is_absorbing_state() const override {
        if (_attempts - _last_big_change < static_cast<long long>(this->size())) return false;   // still moving
        return this->graph().is_complete() ? _converged_sorted() : _converged_edges();
    }

protected:
    // Whether opinions differing by d influence each other (strict for
    // Deffuant, inclusive for Hegselmann-Krause, following the original papers).
    virtual bool _within_confidence(double d) const = 0;

    // Derived classes count every attempted update and report each opinion
    // change, which drives the cheap "still moving" gate above.
    void _count_attempts(long long n) { _attempts += n; }
    void _record_change(double change) {
        if (std::fabs(change) > _tolerance) _last_big_change = _attempts;
    }

private:
    // Every interacting pair of neighbours agrees within the tolerance.
    bool _converged_edges() const {
        const BaseGraph<T>& g = this->graph();
        for (std::size_t i = 0; i < g.size(); ++i) {
            const double xi = static_cast<double>(g.get_state(i));
            for (std::size_t j : g.neighbours(i)) {
                if (j < i) continue;
                const double d = std::fabs(static_cast<double>(g.get_state(j)) - xi);
                if (d > _tolerance && _within_confidence(d)) return false;
            }
        }
        return true;
    }

    // Complete graph: every pair interacts if close enough. Sorted opinions:
    // for each one, the farthest opinion above it still within confidence
    // (two pointers) must be within the tolerance.
    bool _converged_sorted() const {
        const std::vector<T>& s = this->graph().states();
        std::vector<double> x(s.begin(), s.end());
        std::sort(x.begin(), x.end());
        std::size_t j = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (j < i) j = i;
            while (j + 1 < x.size() && _within_confidence(x[j + 1] - x[i])) ++j;
            if (x[j] - x[i] > _tolerance) return false;
        }
        return true;
    }

    double    _epsilon, _tolerance;
    long long _attempts = 0;           // attempted updates so far
    long long _last_big_change = 0;    // attempt count at the last change > tolerance
};

#endif // ODSP_BOUNDEDCONFIDENCEMODEL_HPP
