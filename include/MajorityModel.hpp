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
//                 pick at random among them).
// For +/-1 spins, Random is exactly zero-temperature Glauber dynamics of the
// Ising model (a spin with zero local field flips with probability 1/2).
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
// Frozen states: on a regular lattice, majority dynamics can freeze into stable
// domain configurations (e.g. flat stripes where every boundary node already
// sees a local majority of its own kind), so full consensus is NOT guaranteed.
// is_absorbing_state() recognises them exactly: consensus, or a state where no
// unfrozen node can change (Random: every node's opinion is the unique plurality
// among its neighbours; KeepCurrent: it is among the tied leaders). That check
// costs O(N * degree), so it runs only once a full sweep's worth of attempted
// updates (N) has passed without any change. last_change_update() gives the
// exact time of the last change, which the detection follows by exactly N
// attempts (one sweep). A failed check is not repeated until the state changes.
// The const checks use mutable scratch, so do not query one model from several
// threads at once. On a complete graph it converges to consensus quickly.
template <typename T>
class MajorityModel : public Model<T> {
public:
    MajorityModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
                  TieRule tie_rule = TieRule::Random, bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links), _tie_rule(tie_rule) {}

    MajorityModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng, bool track_active_links)
        : MajorityModel(std::move(graph), rng, TieRule::Random, track_active_links) {}

    TieRule tie_rule() const { return _tie_rule; }

    // Consensus (checked at once), or frozen (checked only after N attempted
    // updates without a change).
    bool is_absorbing_state() const override {
        if (this->is_consensus_reached()) return true;
        if (_attempts - _last_change < static_cast<long long>(this->size())) return false;   // still moving
        // A failed check stays valid until the state changes (an opinion, or the
        // set of zealots): in single-update runs this check is asked after every
        // update, and must not cost O(N * degree) each time.
        if (_failed_at == _last_change) return false;
        ++_frozen_checks;
        if (is_frozen_state()) return true;
        _failed_at = _last_change;
        return false;
    }

    // How many full frozen checks is_absorbing_state() has run (diagnostic).
    long long frozen_checks() const { return _frozen_checks; }

    // Exact and ungated: no unfrozen node can change its opinion. Zealots and
    // isolated nodes never block it. O(N * degree).
    bool is_frozen_state() const {
        const BaseGraph<T>& g = this->graph();
        for (std::size_t i = 0; i < g.size(); ++i) {
            if (this->is_frozen(i)) continue;
            g.neighbour_state_counts(i, _check_tally);
            if (_check_tally.empty()) continue;   // isolated node
            const T own = g.get_state(i);
            std::size_t best = 0, own_count = 0, leaders = 0;
            for (const auto& entry : _check_tally) {
                if (entry.second > best) { best = entry.second; leaders = 1; }
                else if (entry.second == best) ++leaders;
                if (entry.first == own) own_count = entry.second;
            }
            if (own_count != best) return false;   // a strict majority elsewhere
            if (leaders > 1 && _tie_rule == TieRule::Random) return false;   // a tie it may leave
        }
        return true;
    }

    // 1-based index of the attempted update that last changed an opinion, counted
    // by this model (equal to get_updates() between steps); 0 if none yet.
    long long last_change_update() const { return _last_change; }

protected:
    void _single_update() override {
        ++_attempts;
        const long long before = this->get_changes();
        _update();
        if (this->get_changes() != before) _last_change = _attempts;
    }

    void _on_frozen_changed() override { _failed_at = -1; }   // a new zealot can freeze the state

private:
    void _update() {
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

    TieRule _tie_rule;
    long long _attempts = 0;      // attempted updates (calls of _single_update)
    long long _last_change = 0;   // _attempts at the last opinion change
    mutable long long _frozen_checks = 0;
    mutable long long _failed_at = -1;   // _last_change when a frozen check last failed; -1: none
    mutable std::vector<std::pair<T, std::size_t>> _check_tally;   // scratch for is_frozen_state()
    // Scratch buffers reused across updates, so the hot loop never allocates.
    std::vector<std::pair<T, std::size_t>> _tally;
    std::vector<T>                         _winners;
};

#endif // ODSP_MAJORITYMODEL_HPP
