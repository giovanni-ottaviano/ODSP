#ifndef ODSP_ISINGMODEL_HPP
#define ODSP_ISINGMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "Model.hpp"

// Single-spin-flip dynamics for the Ising model.
//   Metropolis : accept a flip with probability min(1, exp(-dE / T)).
//   Glauber    : heat bath, accept with probability 1 / (1 + exp(dE / T)).
// Both satisfy detailed balance with the Boltzmann distribution, so they share
// the same equilibrium; they differ in the kinetics.
enum class IsingDynamics { Metropolis, Glauber };

// Ising model, H = -J sum_<ij> s_i s_j - h sum_i s_i, at temperature T (k_B = 1),
// simulated by random sequential single-spin flips. It is the equilibrium
// reference point of the family: the noisy and majority-vote models share its
// up-down symmetry, but only this one has a Boltzmann stationary state, so its
// results can be checked against exact thermodynamics:
//   * 2D square lattice (J = 1): T_c = 2 / ln(1 + sqrt 2) = 2.269...,
//     spontaneous magnetization m = (1 - sinh(2/T)^-4)^(1/8) for T < T_c (Onsager);
//   * complete graph with J = 1/(N-1) (Curie-Weiss): T_c = 1, m = tanh(m / T).
//
// The flip probabilities depend only on the spin and the integer sum of its
// neighbours' spins, so they are tabulated once: no exp() in the hot loop.
// Binary +/-1 states.
template <typename T>
class IsingModel : public Model<T> {
public:
    IsingModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double temperature,
               double J = 1.0, double h = 0.0, IsingDynamics dynamics = IsingDynamics::Metropolis,
               bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links),
          _temperature(temperature), _J(J), _h(h), _dynamics(dynamics) {
        if (!(temperature >= 0.0)) throw std::invalid_argument("IsingModel: temperature must be >= 0.");
        this->_require_binary_states("IsingModel");
        for (std::size_t i = 0; i < this->size(); ++i) {
            _max_degree = std::max(_max_degree, this->graph().degree(i));
            if (this->graph().degree(i) == 0) _has_isolated_nodes = true;
        }
        _build_table();
    }

    double        temperature() const { return _temperature; }
    double        coupling()    const { return _J; }
    double        field()       const { return _h; }
    IsingDynamics dynamics()    const { return _dynamics; }

    // Energy H, from the observables Model already maintains:
    // sum over edges of s_i s_j = (#edges) - 2 (#disagreeing edges).
    double energy() const {
        const double bond_sum = static_cast<double>(this->get_edge_count() - 2 * this->get_active_links());
        return -_J * bond_sum - _h * this->get_magnetization();
    }
    double energy_per_site() const { return energy() / static_cast<double>(this->size()); }

    // At T = 0 with a ferromagnetic coupling and no field, no flip out of
    // consensus is ever accepted -- except for an isolated node, whose flip
    // costs nothing (dE = 0) and is accepted.
    bool consensus_is_absorbing() const override {
        return _temperature == 0.0 && _J > 0.0 && _h == 0.0 && !_has_isolated_nodes;
    }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        const T s = this->graph().get_state(node);
        const long long n = this->neighbour_spin_sum(node);
        const std::size_t idx = static_cast<std::size_t>(n + static_cast<long long>(_max_degree));
        const double p = (s > 0 ? _accept_up : _accept_down)[idx];
        if (p >= 1.0 || (p > 0.0 && _uniform(this->rng()) < p)) this->_apply(node, T(-s));
    }

private:
    // Acceptance probability of flipping spin s with neighbour sum n, for every
    // n in [-max_degree, max_degree]: dE = 2 s (J n + h).
    void _build_table() {
        const std::size_t size = 2 * _max_degree + 1;
        _accept_up.resize(size);
        _accept_down.resize(size);
        for (std::size_t idx = 0; idx < size; ++idx) {
            const double n = static_cast<double>(idx) - static_cast<double>(_max_degree);
            _accept_up[idx]   = _acceptance( 2.0 * (_J * n + _h));
            _accept_down[idx] = _acceptance(-2.0 * (_J * n + _h));
        }
    }

    double _acceptance(double dE) const {
        if (_temperature == 0.0) {
            if (_dynamics == IsingDynamics::Metropolis) return dE <= 0.0 ? 1.0 : 0.0;
            return dE < 0.0 ? 1.0 : (dE > 0.0 ? 0.0 : 0.5);
        }
        if (_dynamics == IsingDynamics::Metropolis) return dE <= 0.0 ? 1.0 : std::exp(-dE / _temperature);
        return 1.0 / (1.0 + std::exp(dE / _temperature));
    }

    double        _temperature, _J, _h;
    IsingDynamics _dynamics;
    std::size_t   _max_degree = 0;
    bool          _has_isolated_nodes = false;
    std::vector<double> _accept_up, _accept_down;   // indexed by neighbour sum + max_degree
    std::uniform_real_distribution<double> _uniform{0.0, 1.0};
};

#endif // ODSP_ISINGMODEL_HPP
