#ifndef ODSP_VOTERMODEL_HPP
#define ODSP_VOTERMODEL_HPP

#include <cstddef>
#include <random>
#include <vector>

#include "Model.hpp"

// How a voter update picks who copies whom. On graphs where every node has the
// same degree (lattices, complete graph) all three give the same dynamics up
// to time rescaling; on graphs with heterogeneous degrees they differ, and
// each conserves (in the mean) a different weighted magnetization, which then
// equals 2 P(+1 consensus) - 1:
//
//   Node    : random node i copies a random neighbour.   conserves sum k_i s_i / sum k_i
//             (the classic voter model)                   (get_degree_weighted_magnetization)
//   Link    : random edge, a random endpoint copies the   conserves sum s_i / N  (plain m)
//             other (= uniform random directed edge).
//   Reverse : random node i imposes its opinion on a      conserves sum (s_i/k_i) / sum (1/k_i)
//             random neighbour (invasion process).
enum class VoterUpdate { Node, Link, Reverse };

// Voter model: an opinion is copied across one random edge per update.
template <typename T>
class VoterModel : public Model<T> {
public:
    VoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
               VoterUpdate update = VoterUpdate::Node, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _update(update) {}

    VoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, bool track_active_links)
        : VoterModel(std::move(graph), rng, VoterUpdate::Node, track_active_links) {}

    VoterUpdate update_scheme() const { return _update; }

protected:
    void _single_update() override {
        switch (_update) {
        case VoterUpdate::Node: {
            const std::size_t node = this->random_node();
            if (this->graph().degree(node) == 0) return;
            this->_apply(node, this->graph().get_state(this->random_neighbour(node)));
            return;
        }
        case VoterUpdate::Link: {
            if (this->get_edge_count() == 0) return;
            const auto [node, source] = this->random_directed_edge();
            this->_apply(node, this->graph().get_state(source));
            return;
        }
        case VoterUpdate::Reverse: {
            const std::size_t source = this->random_node();
            if (this->graph().degree(source) == 0) return;
            this->_apply(this->random_neighbour(source), this->graph().get_state(source));
            return;
        }
        }
    }

private:
    VoterUpdate _update;
};

#endif // ODSP_VOTERMODEL_HPP
