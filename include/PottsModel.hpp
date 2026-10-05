#ifndef ODSP_POTTSMODEL_HPP
#define ODSP_POTTSMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Model.hpp"

// Single-site dynamics for the Potts model.
//   Metropolis : propose one of the other q-1 states uniformly, accept with
//                probability min(1, exp(-dE / T)).
//   HeatBath   : draw the new state from its conditional Boltzmann
//                distribution, P(s) ~ exp(J n_s / T), n_s = neighbours in state s.
// Both satisfy detailed balance with the same equilibrium.
enum class PottsDynamics { Metropolis, HeatBath };

// q-state Potts model, H = -J sum_<ij> delta(s_i, s_j), states 0..q-1, at
// temperature T: the equilibrium multi-opinion counterpart of the Ising model
// (q = 2 is the Ising model with coupling J/2).
//
// Exact results to compare with, for J = 1 on the square lattice:
//   T_c = 1 / ln(1 + sqrt q)   (Baxter); the transition is continuous for
//   q <= 4 and first order (latent heat, coexistence) for q > 4.
// Order parameter: (q * f_max - 1) / (q - 1), f_max the largest opinion share:
// 1 in a fully ordered state, ~0 in the disordered one (order_parameter()).
template <typename T>
class PottsModel : public Model<T> {
public:
    PottsModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, int q, double temperature,
               double J = 1.0, PottsDynamics dynamics = PottsDynamics::HeatBath, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links),
          _q(q), _temperature(temperature), _J(J), _dynamics(dynamics) {
        if (q < 2) throw std::invalid_argument("PottsModel: q must be >= 2.");
        if (!(temperature > 0.0)) throw std::invalid_argument("PottsModel: temperature must be > 0.");
        for (std::size_t i = 0; i < this->size(); ++i) {
            const T s = this->graph().get_state(i);
            if (s < T(0) || s >= T(q))
                throw std::invalid_argument("PottsModel: node " + std::to_string(i) + " has state " +
                                            std::to_string(s) + ", outside 0.." + std::to_string(q - 1) + ".");
            _max_degree = std::max(_max_degree, this->graph().degree(i));
        }
        // Boltzmann factors exp(J n / T) for n = 0..max_degree.
        _boltzmann.resize(_max_degree + 1);
        for (std::size_t n = 0; n <= _max_degree; ++n)
            _boltzmann[n] = std::exp(_J * static_cast<double>(n) / _temperature);
    }

    int           q()           const { return _q; }
    double        temperature() const { return _temperature; }
    double        coupling()    const { return _J; }
    PottsDynamics dynamics()    const { return _dynamics; }

    // H = -J (#agreeing edges) = -J (#edges - #disagreeing edges).
    double energy() const {
        return -_J * static_cast<double>(this->get_edge_count() - this->get_active_links());
    }
    double energy_per_site() const { return energy() / static_cast<double>(this->size()); }

    double order_parameter() const {
        return (static_cast<double>(_q) * this->largest_opinion_fraction() - 1.0) / static_cast<double>(_q - 1);
    }

    // Thermal fluctuations always leave consensus at T > 0.
    bool consensus_is_absorbing() const override { return false; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        const T old = this->graph().get_state(node);

        if (_dynamics == PottsDynamics::Metropolis) {
            // Uniform among the q-1 other states.
            T proposal = static_cast<T>(std::uniform_int_distribution<int>(0, _q - 2)(this->rng()));
            if (proposal >= old) ++proposal;
            const std::size_t n_old = this->graph().count_neighbours_in_state(node, old);
            const std::size_t n_new = this->graph().count_neighbours_in_state(node, proposal);
            // dE = -J (n_new - n_old); accept with min(1, exp(-dE/T)).
            if (n_new >= n_old && _J >= 0.0) { this->_apply(node, proposal); return; }
            const double p = std::exp(_J * (static_cast<double>(n_new) - static_cast<double>(n_old)) / _temperature);
            if (_uniform(this->rng()) < p) this->_apply(node, proposal);
            return;
        }

        // Heat bath: states absent from the neighbourhood all have weight 1.
        this->graph().neighbour_state_counts(node, _tally);
        double total = static_cast<double>(_q) - static_cast<double>(_tally.size());
        for (const auto& entry : _tally) total += _boltzmann[entry.second];
        double u = _uniform(this->rng()) * total;
        for (const auto& entry : _tally) {
            u -= _boltzmann[entry.second];
            if (u < 0.0) { this->_apply(node, entry.first); return; }
        }
        // One of the absent states, uniformly (rejection: the tally is small).
        if (_tally.size() == static_cast<std::size_t>(_q)) {   // rounding left u >= 0 at the very end
            this->_apply(node, _tally.back().first);
            return;
        }
        std::uniform_int_distribution<int> pick(0, _q - 1);
        while (true) {
            const T s = static_cast<T>(pick(this->rng()));
            bool present = false;
            for (const auto& entry : _tally) if (entry.first == s) { present = true; break; }
            if (!present) { this->_apply(node, s); return; }
        }
    }

private:
    int           _q;
    double        _temperature, _J;
    PottsDynamics _dynamics;
    std::size_t   _max_degree = 0;
    std::vector<double> _boltzmann;                      // exp(J n / T)
    std::vector<std::pair<T, std::size_t>> _tally;      // scratch: neighbour histogram
    std::uniform_real_distribution<double> _uniform{0.0, 1.0};
};

#endif // ODSP_POTTSMODEL_HPP
