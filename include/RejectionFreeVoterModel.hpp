#ifndef ODSP_REJECTIONFREEVOTERMODEL_HPP
#define ODSP_REJECTIONFREEVOTERMODEL_HPP

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "Model.hpp"

// Voter model (node update) that skips updates which would change nothing.
//
// In the plain VoterModel an update is wasted whenever the chosen node copies a
// neighbour that already agrees with it. The fraction of useful ("effective")
// updates is
//     p = W / N,   W = sum_i a_i / k_i,
// where a_i is the number of neighbours disagreeing with node i. p shrinks as
// domains coarsen: in 1D it decays like t^(-1/2), so the plain model ends up
// wasting almost every update.
//
// This class simulates the SAME Markov chain in two regimes, chosen step by step
// from the current p (each regime is an exact sampler of the chain, so
// switching between them is exact too):
//
//   * direct (p high): plain attempts, as in VoterModel. Only the integer counts
//     a_i and the running total W are maintained on each flip -- nearly the
//     cost of the plain model.
//   * jump (p low): the number of attempts up to and including the next
//     effective one is 1 + Geometric(p), so the update/sweep counters advance
//     exactly as if every attempt had been simulated; the event itself picks
//     node i with probability (a_i/k_i) / W from a sum tree (O(log N)) and then
//     one of its a_i disagreeing neighbours uniformly. The tree is rebuilt in
//     O(N) when entering this regime and kept up to date while in it.
//
// Trajectories have exactly the statistics of VoterModel with VoterUpdate::Node,
// in the same time units (the tests check consensus times and exit
// probabilities against exact results and against the plain model).
//
// When it pays off: wherever p gets small -- 1D and quasi-1D systems, and the
// late, nearly ordered stage of any run. On 2D/3D lattices p stays around
// 0.1-0.3 for most of the run (rho decays only logarithmically in 2D and
// plateaus in 3D), so this class runs at about the plain model's speed there.
// On a complete graph (k = N - 1) each flip costs O(N): use VoterModel, which is
// O(1) there.
//
// Zealots (Model::set_frozen) are supported: a frozen node has weight 0, so it
// is never chosen to change, but it still influences its neighbours.
template <typename T>
class RejectionFreeVoterModel : public Model<T> {
public:
    RejectionFreeVoterModel(std::unique_ptr<BaseGraph<T>> graph, std::mt19937* rng,
                            bool track_active_links = true)
        : Model<T>(std::move(graph), rng, track_active_links) {
        const BaseGraph<T>& g = this->graph();
        const std::size_t n = g.size();
        _active.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            _active[i] = g.degree(i) - g.count_neighbours_in_state(i, g.get_state(i));
        _capacity = 1;
        while (_capacity < n) _capacity *= 2;
        _tree.assign(2 * _capacity, 0.0);
        _rebuild_tree();
    }

    // Probability that an attempted update changes the state: W / N.
    double effective_update_probability() const {
        return this->size() == 0 ? 0.0 : _W / static_cast<double>(this->size());
    }

protected:
    void _single_update() override { _advance_updates(1); }

    void _on_frozen_changed() override { _rebuild_tree(); }

    void _advance_updates(long long n) override {
        const double N = static_cast<double>(this->size());
        long long remaining = n;
        while (remaining > 0) {
            const double p = std::min(1.0, _W / N);

            // Regime choice, with hysteresis so the O(N) tree rebuild on
            // entering the jump regime happens rarely.
            if (_jumping && p > 2.0 * JUMP_BELOW) {
                _jumping = false;
            } else if (!_jumping && p < JUMP_BELOW) {
                _rebuild_tree();   // also resets _W exactly, clearing rounding drift
                _jumping = true;
                continue;
            }

            if (!_jumping) {
                --remaining;
                _direct_attempt();
                continue;
            }

            if (_W <= 0.0) return;   // no disagreeing pair left: nothing can ever change
            long long wait = 1;      // attempts up to and including the next effective one
            if (p < 1.0) wait += std::geometric_distribution<long long>(p)(this->rng());
            // If the event falls after this window, drop it: attempts are
            // independent given the state, so redrawing later is exact
            // (memorylessness of the geometric distribution).
            if (wait > remaining) return;
            remaining -= wait;
            _jump_to_effective_update();
        }
    }

private:
    // Jump regime below this effective-update probability (leave it above
    // twice this). Benchmarks on 1D-3D lattices are insensitive to the exact
    // value between 0.02 and 0.2.
    static constexpr double JUMP_BELOW = 0.1;

    double _weight(std::size_t i) const {
        if (this->is_frozen(i)) return 0.0;
        const std::size_t k = this->graph().degree(i);
        return k == 0 ? 0.0 : static_cast<double>(_active[i]) / static_cast<double>(k);
    }

    // Rebuild the sum tree from the counts a_i, and W from the tree.
    void _rebuild_tree() {
        const std::size_t n = this->size();
        for (std::size_t i = 0; i < n; ++i) _tree[_capacity + i] = _weight(i);
        for (std::size_t v = _capacity - 1; v >= 1; --v) _tree[v] = _tree[2 * v] + _tree[2 * v + 1];
        _W = _tree[1];
    }

    void _set_tree_weight(std::size_t i) {
        std::size_t v = _capacity + i;
        _tree[v] = _weight(i);
        // Recompute ancestors from their children (rather than adding a delta)
        // so rounding errors cannot accumulate: the total is exactly 0 when all
        // weights are 0.
        for (v /= 2; v >= 1; v /= 2) _tree[v] = _tree[2 * v] + _tree[2 * v + 1];
    }

    // Node with probability weight / total. Never returns a zero-weight leaf:
    // a child with zero sum is never entered, even at rounding boundaries.
    std::size_t _sample_node() {
        std::uniform_real_distribution<double> d(0.0, _tree[1]);
        double u = d(this->rng());
        std::size_t v = 1;
        while (v < _capacity) {
            const double left = _tree[2 * v];
            if ((u < left && left > 0.0) || _tree[2 * v + 1] <= 0.0) {
                v = 2 * v;
            } else {
                u -= left;
                v = 2 * v + 1;
            }
        }
        return v - _capacity;
    }

    // One attempt of the plain voter rule.
    void _direct_attempt() {
        const BaseGraph<T>& g = this->graph();
        const std::size_t i = this->random_node();
        if (g.degree(i) == 0 || this->is_frozen(i)) return;
        const T sj = g.get_state(this->random_neighbour(i));
        if (sj != g.get_state(i)) _flip(i, sj);
    }

    // The next effective update: node with probability (a_i/k_i) / W, then a
    // uniform choice among its disagreeing neighbours.
    void _jump_to_effective_update() {
        const BaseGraph<T>& g = this->graph();
        const std::size_t i = _sample_node();
        const T old = g.get_state(i);

        std::uniform_int_distribution<std::size_t> pick(0, _active[i] - 1);
        std::size_t r = pick(this->rng());
        for (std::size_t j : g.neighbours(i)) {
            const T sj = g.get_state(j);
            if (sj != old && r-- == 0) { _flip(i, sj); return; }
        }
    }

    // Node i adopts new_value != its state: refresh the counts of i and its
    // neighbours, W, and (in the jump regime) the tree.
    void _flip(std::size_t i, T new_value) {
        const BaseGraph<T>& g = this->graph();
        const T old = g.get_state(i);
        this->_apply(i, new_value);

        std::size_t a_i = 0;
        for (std::size_t j : g.neighbours(i)) {
            const T sj = g.get_state(j);
            const bool before = (sj != old), after = (sj != new_value);
            a_i += after ? 1 : 0;
            if (before == after) continue;
            const double w = this->is_frozen(j) ? 0.0 : 1.0 / static_cast<double>(g.degree(j));
            if (after) { ++_active[j]; _W += w; }
            else       { --_active[j]; _W -= w; }
            if (_jumping) _set_tree_weight(j);
        }
        _W += (static_cast<double>(a_i) - static_cast<double>(_active[i])) / static_cast<double>(g.degree(i));
        _active[i] = a_i;
        if (_jumping) {
            _set_tree_weight(i);
            _W = _tree[1];   // exact while the tree is maintained
        }
    }

    std::vector<std::size_t> _active;          // a_i: neighbours disagreeing with i
    double                   _W = 0.0;         // sum_i a_i / k_i (exact in the jump regime)
    bool                     _jumping = false; // current regime
    std::size_t              _capacity = 1;    // leaves in the sum tree (power of two >= N)
    std::vector<double>      _tree;            // sum tree: leaves at [_capacity, 2 _capacity)
};

#endif // ODSP_REJECTIONFREEVOTERMODEL_HPP
