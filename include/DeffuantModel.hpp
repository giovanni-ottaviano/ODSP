#ifndef ODSP_DEFFUANTMODEL_HPP
#define ODSP_DEFFUANTMODEL_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "BoundedConfidenceModel.hpp"

// Deffuant-Weisbuch bounded-confidence model (Deffuant, Neau, Amblard &
// Weisbuch, Adv. Complex Syst. 3, 87, 2000). At each update a random pair of
// neighbours (a uniformly random edge) with opinions x, y interacts if
// |x - y| < epsilon: both move toward each other by a fraction mu of their
// difference,
//     x <- x + mu (y - x),    y <- y + mu (x - y).
// Otherwise nothing happens. mu in (0, 1/2]; mu = 1/2 makes them meet halfway.
//
// Properties worth knowing (and tested):
//   * the sum of opinions is conserved exactly by every interaction, so the
//     mean opinion is constant (without zealots);
//   * opinions never leave the initial range;
//   * on a complete graph with opinions uniform in [0, 1], the dynamics ends
//     in about 1/(2 epsilon) major clusters, and in consensus for epsilon above
//     ~0.27; epsilon >= the initial range always gives consensus at the mean.
//
// Time: one update is one pair selection; a sweep is N of them.
template <typename T = double>
class DeffuantModel : public BoundedConfidenceModel<T> {
public:
    DeffuantModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, double epsilon,
                  double mu = 0.5, double tolerance = 1e-6, bool track_active_links = false)
        : BoundedConfidenceModel<T>(std::move(graph), rng, epsilon, tolerance, track_active_links), _mu(mu) {
        if (!(mu > 0.0 && mu <= 0.5)) throw std::invalid_argument("DeffuantModel: mu must be in (0, 1/2].");
    }

    double convergence_parameter() const { return _mu; }

protected:
    bool _within_confidence(double d) const override { return std::fabs(d) < this->confidence(); }

    void _single_update() override {
        this->_count_attempts(1);
        if (this->get_edge_count() == 0) return;
        const auto [i, j] = this->random_directed_edge();
        const T xi = this->graph().get_state(i), xj = this->graph().get_state(j);
        const T d = xj - xi;
        if (!_within_confidence(static_cast<double>(d))) return;
        const T shift = static_cast<T>(_mu) * d;
        this->_apply(i, xi + shift);
        this->_apply(j, xj - shift);
        this->_record_change(static_cast<double>(shift));
    }

private:
    double _mu;
};

#endif // ODSP_DEFFUANTMODEL_HPP
