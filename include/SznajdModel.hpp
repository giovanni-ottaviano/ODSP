#ifndef ODSP_SZNAJDMODEL_HPP
#define ODSP_SZNAJDMODEL_HPP

#include <cstddef>
#include <utility>

#include "Model.hpp"

// Sznajd model, "united we stand, divided we fall" (K. Sznajd-Weron & J.
// Sznajd, Int. J. Mod. Phys. C 11, 1157, 2000), in its common outflow form: a
// random pair of neighbours (a uniformly random edge) that AGREES persuades all
// of its other neighbours to adopt its opinion; a disagreeing pair does nothing.
//
// Unlike the voter model, influence flows OUT of a group and can change many
// nodes in one update, and agreeing pairs are needed. On regular lattices the
// dynamics always reaches consensus, and the exit probability is much steeper
// than the voter model's P(+1) = f0 (it tends to a step at f0 = 1/2 in large 2D
// systems).
//
// Time unit: one update is one pair selection, a sweep is N of them.
template <typename T>
class SznajdModel : public Model<T> {
public:
    using Model<T>::Model;

protected:
    void _single_update() override {
        if (this->get_edge_count() == 0) return;
        const auto [i, j] = this->random_directed_edge();
        const T opinion = this->graph().get_state(i);
        if (this->graph().get_state(j) != opinion) return;

        for (std::size_t l : this->graph().neighbours(i))
            if (l != j) this->_apply(l, opinion);
        for (std::size_t l : this->graph().neighbours(j))
            if (l != i) this->_apply(l, opinion);
    }
};

#endif // ODSP_SZNAJDMODEL_HPP
