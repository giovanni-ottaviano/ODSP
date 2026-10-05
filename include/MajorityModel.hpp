#ifndef ODSP_MAJORITYMODEL_HPP
#define ODSP_MAJORITYMODEL_HPP

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "Model.hpp"

// What a node does when several opinions tie for the majority among its
// neighbours.
//   Random      : adopt one of the tied opinions uniformly at random.
//   KeepCurrent : keep its own opinion if it is among the tied ones (otherwise
//                 pick at random among them). For +/-1 spins this is exactly
//                 zero-temperature Glauber dynamics of the Ising model.
enum class TieRule { Random, KeepCurrent };

// Majority rule: a random node adopts the MAJORITY opinion among ALL of its
// neighbours (plurality when there are more than two opinions).
//
// Unlike the voter model this consumes the whole neighbourhood at once, which
// is precisely why it's a useful check that BaseGraph isn't secretly shaped
// around the voter model's "one random neighbour" access pattern. The
// neighbourhood histogram comes from BaseGraph::neighbour_state_counts(), so a
// graph that can answer it quickly (the complete graph, in O(1)) makes this
// model fast there too.
//
// Note: on a regular lattice, majority dynamics can freeze into stable domain
// configurations (e.g. flat interfaces where every boundary node already sees a
// local majority of its own kind), so full consensus is NOT guaranteed -- a run
// may terminate at the step cap in a partially ordered state. On a complete
// graph it converges to consensus quickly. This qualitative difference from the
// voter model is a feature worth observing, not a bug.
template <typename T>
class MajorityModel : public Model<T> {
public:
    MajorityModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
                  TieRule tie_rule = TieRule::Random, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _tie_rule(tie_rule) {}

    MajorityModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, bool track_active_links)
        : MajorityModel(std::move(graph), rng, TieRule::Random, track_active_links) {}

    TieRule tie_rule() const { return _tie_rule; }

protected:
    void _single_update() override {
        const std::size_t node = this->random_node();
        this->graph().neighbour_state_counts(node, _tally);
        if (_tally.empty()) return;   // isolated node

        // Leading opinion(s).
        std::size_t best = 0;
        for (const auto& entry : _tally) best = std::max(best, entry.second);
        _winners.clear();
        for (const auto& entry : _tally)
            if (entry.second == best) _winners.push_back(entry.first);

        if (_winners.size() == 1) {
            this->_apply(node, _winners.front());
            return;
        }
        if (_tie_rule == TieRule::KeepCurrent) {
            const T own = this->graph().get_state(node);
            for (const T& w : _winners)
                if (w == own) return;   // own opinion is among the tied: keep it
        }
        std::uniform_int_distribution<std::size_t> d(0, _winners.size() - 1);
        this->_apply(node, _winners[d(this->rng())]);
    }

private:
    TieRule _tie_rule;
    // Scratch buffers reused across updates, so the hot loop never allocates.
    std::vector<std::pair<T, std::size_t>> _tally;
    std::vector<T>                         _winners;
};

#endif // ODSP_MAJORITYMODEL_HPP
