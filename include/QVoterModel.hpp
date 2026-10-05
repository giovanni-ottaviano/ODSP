#ifndef ODSP_QVOTERMODEL_HPP
#define ODSP_QVOTERMODEL_HPP

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "Model.hpp"

// What a node does when it does NOT conform (with probability p per update).
//   None           : always conform (plain nonlinear q-voter).
//   Independence   : ignore the neighbours and adopt a random opinion
//                    (Nyczka, Sznajd-Weron & Cislo, Phys. Rev. E 86, 011105, 2012).
//   Anticonformity : if the panel is unanimous, adopt the OPPOSITE of its opinion.
enum class Nonconformity { None, Independence, Anticonformity };

// q-voter model (Castellano, Munoz & Pastor-Satorras, Phys. Rev. E 80, 041129,
// 2009) with nonconformity. At each update a random node i draws a panel of q
// of its neighbours (with repetition by default, as in the original model, or
// q distinct ones). If the panel is unanimous, i adopts its opinion
// (conformity); otherwise nothing happens. With probability p the node is a
// nonconformist for this update instead (see Nonconformity).
//
// q = 1 with p = 0 is exactly the voter model. For q > 1 the influence is
// nonlinear: a unanimous group is needed. Nonconformity drives an
// order-disorder transition; in mean field (complete graph, N -> infinity,
// panel with repetition) it occurs at
//     independence:    p_c = (q - 1) / (q - 1 + 2^(q-1))
//                      continuous for q <= 5, discontinuous for q > 5;
//     anticonformity:  p_c = (q - 1) / (2q),  always continuous.
// and the stationary up-fraction x solves dx/dt = 0 with
//     independence:    dx/dt = (1-p) [(1-x) x^q - x (1-x)^q] + p (1/2 - x)
//     anticonformity:  dx/dt = (1-p) [(1-x) x^q - x (1-x)^q] + p [(1-x)^(q+1) - x^(q+1)]
//
// Binary +/-1 opinions. Without repetition, a node with fewer than q
// neighbours is never influenced.
template <typename T>
class QVoterModel : public Model<T> {
public:
    QVoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, int q,
                double p = 0.0, Nonconformity nonconformity = Nonconformity::Independence,
                bool with_repetition = true, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links),
          _q(q), _p(nonconformity == Nonconformity::None ? 0.0 : p),
          _nonconformity(nonconformity), _with_repetition(with_repetition) {
        if (q < 1) throw std::invalid_argument("QVoterModel: q must be >= 1.");
        if (p < 0.0 || p > 1.0) throw std::invalid_argument("QVoterModel: p must be in [0, 1].");
        this->_require_binary_states("QVoterModel");
    }

    int           q()                const { return _q; }
    double        p()                const { return _p; }
    Nonconformity nonconformity()    const { return _nonconformity; }
    bool          with_repetition()  const { return _with_repetition; }

    // Nonconformists can always leave consensus.
    bool consensus_is_absorbing() const override { return _p == 0.0; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        const std::size_t k = this->graph().degree(node);
        if (k == 0) return;

        if (_nonconformity == Nonconformity::Independence && _p > 0.0 &&
            std::bernoulli_distribution(_p)(this->rng())) {
            this->_apply(node, std::bernoulli_distribution(0.5)(this->rng()) ? T(1) : T(-1));
            return;
        }

        T panel_opinion;
        if (!_unanimous_panel(node, k, panel_opinion)) return;

        if (_nonconformity == Nonconformity::Anticonformity && _p > 0.0 &&
            std::bernoulli_distribution(_p)(this->rng()))
            this->_apply(node, T(-panel_opinion));
        else
            this->_apply(node, panel_opinion);
    }

private:
    // Draws the panel; returns true (and its opinion) iff it is unanimous.
    bool _unanimous_panel(std::size_t node, std::size_t k, T& opinion) {
        const NeighbourView nb = this->graph().neighbours(node);
        const std::size_t q = static_cast<std::size_t>(_q);

        if (_with_repetition) {
            std::uniform_int_distribution<std::size_t> pick(0, k - 1);
            opinion = this->graph().get_state(nb[pick(this->rng())]);
            for (std::size_t m = 1; m < q; ++m)
                if (this->graph().get_state(nb[pick(this->rng())]) != opinion) return false;
            return true;
        }

        if (k < q) return false;
        // q distinct neighbour indices by Floyd's algorithm: O(q) draws.
        _chosen.clear();
        for (std::size_t j = k - q; j < k; ++j) {
            const std::size_t t = std::uniform_int_distribution<std::size_t>(0, j)(this->rng());
            const bool seen = std::find(_chosen.begin(), _chosen.end(), t) != _chosen.end();
            _chosen.push_back(seen ? j : t);
        }
        opinion = this->graph().get_state(nb[_chosen[0]]);
        for (std::size_t m = 1; m < q; ++m)
            if (this->graph().get_state(nb[_chosen[m]]) != opinion) return false;
        return true;
    }

    int                      _q;
    double                   _p;
    Nonconformity            _nonconformity;
    bool                     _with_repetition;
    std::vector<std::size_t> _chosen;   // scratch for distinct panels
};

#endif // ODSP_QVOTERMODEL_HPP
