#ifndef ODSP_NOISYVOTERMODEL_HPP
#define ODSP_NOISYVOTERMODEL_HPP

#include <random>
#include <stdexcept>
#include <cstddef>
#include <vector>

#include "Model.hpp"

// Noisy voter model (symmetric Kirman model). When a node is updated:
//   * with probability (1 - noise): it copies a random neighbour   (imitation),
//   * with probability  noise:      it adopts an opinion drawn uniformly from the
//                                    opinion set (spontaneous, independent of its
//                                    neighbours); +/-1 by default.
//
// The spontaneous events destroy the absorbing consensus of the plain voter
// model, so the system settles into a STATIONARY distribution instead. Sweeping
// `noise` drives a finite-size transition between an ordered, bimodal state
// (m near +/-1, Binder ~ 2/3) at low noise and a disordered, unimodal state
// (m near 0, Gaussian, Binder ~ 0) at high noise. Use Model::sample_stationary()
// to measure the order parameter, susceptibility and Binder cumulant across that
// crossover. At noise = 0 this reduces exactly to the voter model (and is again
// absorbing).
//
// Multi-opinion version: pass the opinion set, e.g. {0, 1, ..., q-1}. The
// stationary state then has q symmetric ordered phases at low noise; use
// largest_opinion_fraction() (or the share of one opinion) as the order
// parameter, via the observable overload of sample_stationary().
template <typename T>
class NoisyVoterModel : public Model<T> {
public:
    NoisyVoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
                    double noise, bool track_active_links = true)
        : NoisyVoterModel(std::move(graph), rng, noise, std::vector<T>{T(1), T(-1)}, track_active_links) {}

    NoisyVoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
                    double noise, std::vector<T> opinions, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _noise(noise), _opinions(std::move(opinions)) {
        if (noise < 0.0 || noise > 1.0)
            throw std::invalid_argument("NoisyVoterModel: noise must be in [0, 1].");
        if (_opinions.empty())
            throw std::invalid_argument("NoisyVoterModel: the opinion set must not be empty.");
    }

    double noise() const { return _noise; }
    const std::vector<T>& opinions() const { return _opinions; }

    // Absorbing only in the noiseless limit.
    bool consensus_is_absorbing() const override { return _noise == 0.0; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();

        std::bernoulli_distribution spontaneous(_noise);
        if (spontaneous(this->rng())) {
            std::uniform_int_distribution<std::size_t> pick(0, _opinions.size() - 1);
            this->_apply(node, _opinions[pick(this->rng())]);
            return;
        }

        if (this->graph().degree(node) == 0) return;
        this->_apply(node, this->graph().get_state(this->random_neighbour(node)));
    }

private:
    double         _noise;
    std::vector<T> _opinions;   // spontaneous changes draw uniformly from these
};

#endif // ODSP_NOISYVOTERMODEL_HPP
