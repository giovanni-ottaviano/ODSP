#ifndef ODSP_MAJORITYVOTEMODEL_HPP
#define ODSP_MAJORITYVOTEMODEL_HPP

#include <cstddef>
#include <random>
#include <stdexcept>

#include "Model.hpp"

// Majority-vote model with noise (M. J. de Oliveira, J. Stat. Phys. 66, 273,
// 1992). A random node adopts the majority opinion of its neighbours with
// probability 1 - noise and the minority opinion with probability `noise`; on a
// tie it takes either opinion with probability 1/2.
//
// A non-equilibrium model with up-down symmetry and an order-disorder phase
// transition in the 2D Ising universality class: on the square lattice
// noise_c = 0.075 (critical value 0.0750 +- 0.0001 from finite-size scaling).
// On a complete graph the majority is the global one, so the stationary state
// is |m| = 1 - 2 noise for every noise < 1/2.
//
// At noise = 0 this is majority rule with random tie-breaking (MajorityModel
// with TieRule::Random). Binary +/-1 opinions.
template <typename T>
class MajorityVoteModel : public Model<T> {
public:
    MajorityVoteModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double noise,
                      bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _noise(noise) {
        if (noise < 0.0 || noise > 1.0) throw std::invalid_argument("MajorityVoteModel: noise must be in [0, 1].");
        this->_require_binary_states("MajorityVoteModel");
    }

    double noise() const { return _noise; }
    bool consensus_is_absorbing() const override { return _noise == 0.0; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        if (this->graph().degree(node) == 0) return;

        const long long field = this->neighbour_spin_sum(node);
        T majority;
        if (field > 0)      majority = T(1);
        else if (field < 0) majority = T(-1);
        else                majority = std::bernoulli_distribution(0.5)(this->rng()) ? T(1) : T(-1);

        const bool minority = _noise > 0.0 && std::bernoulli_distribution(_noise)(this->rng());
        this->_apply(node, minority ? T(-majority) : majority);
    }

private:
    double _noise;
};

#endif // ODSP_MAJORITYVOTEMODEL_HPP
