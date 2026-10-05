#ifndef ODSP_NONLINEARVOTERMODEL_HPP
#define ODSP_NONLINEARVOTERMODEL_HPP

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

#include "Model.hpp"

// Nonlinear voter model: a random node flips with probability f^alpha, where f
// is the fraction of its neighbours that disagree with it (see e.g.
// Schweitzer & Behera, Eur. Phys. J. B 67, 301, 2009; Castellano et al. 2009).
//
//   alpha = 1 : exactly the voter model (copying a random neighbour flips the
//               node with probability f).
//   alpha > 1 : minorities are under-weighted -- an ordering, majority-like
//               bias: on a complete graph consensus comes in a time ~ ln N
//               instead of ~ N.
//   alpha < 1 : minorities are over-weighted -- a disordering bias: the
//               mixed state x = 1/2 is stable in mean field, and consensus on a
//               complete graph takes a time exponential in N.
//
// Mean field: dx/dt = (1-x) x^alpha - x (1-x)^alpha.
// Binary +/-1 opinions.
template <typename T>
class NonlinearVoterModel : public Model<T> {
public:
    NonlinearVoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double alpha,
                        bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _alpha(alpha) {
        if (!(alpha > 0.0)) throw std::invalid_argument("NonlinearVoterModel: alpha must be > 0.");
        this->_require_binary_states("NonlinearVoterModel");
    }

    double alpha() const { return _alpha; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        const std::size_t k = this->graph().degree(node);
        if (k == 0) return;
        const T s = this->graph().get_state(node);
        const std::size_t disagree = k - this->graph().count_neighbours_in_state(node, s);
        if (disagree == 0) return;

        const double f = static_cast<double>(disagree) / static_cast<double>(k);
        const double p_flip = _alpha == 1.0 ? f : std::pow(f, _alpha);
        if (std::bernoulli_distribution(p_flip)(this->rng())) this->_apply(node, T(-s));
    }

private:
    double _alpha;
};

#endif // ODSP_NONLINEARVOTERMODEL_HPP
