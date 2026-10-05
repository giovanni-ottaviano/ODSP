#ifndef ODSP_COMPLETEGRAPH_HPP
#define ODSP_COMPLETEGRAPH_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "BaseGraph.hpp"
#include "OpinionCounts.hpp"

// Complete graph: every node is connected to every other node. Running the
// voter model here recovers the mean-field / fully-mixed case.
//
// Nothing about the topology is stored: neighbours() returns an implicit view
// ("all nodes except me"), so memory is O(N) instead of O(N^2). The graph also
// keeps a running count of nodes per opinion, which answers the neighbourhood
// queries in O(1) -- the neighbours of i holding s are all holders of s except
// possibly i itself. Counts and updates are O(1) whatever the number of
// opinions (OpinionCounts), so the active-link bookkeeping is O(1) per update
// here instead of O(N); neighbour_state_counts() lists every opinion held, so
// the majority rule costs O(number of opinions). The histogram is kept only for
// integer opinion types: with continuous opinions every value is distinct, and
// the queries fall back to a scan.
template <typename T>
class CompleteGraph : public BaseGraph<T> {
public:
    explicit CompleteGraph(std::vector<T> states) : _states(std::move(states)) {
        if constexpr (counted) _counts.reset(_states);
    }

    std::size_t   size() const override { return _states.size(); }
    bool          is_complete() const override { return true; }
    NeighbourView neighbours(std::size_t node) const override {
        return NeighbourView::all_except(_states.size(), node);
    }

    T    get_state(std::size_t node) const override { return _states[node]; }
    void set_state(std::size_t node, T value) override {
        if constexpr (counted) _counts.move(_states[node], value);
        _states[node] = value;
    }
    const std::vector<T>& states() const override { return _states; }

    std::size_t count_neighbours_in_state(std::size_t node, T s) const override {
        return count_in_state(s) - (_states[node] == s ? 1 : 0);
    }

    void neighbour_state_counts(std::size_t node,
                                std::vector<std::pair<T, std::size_t>>& out) const override {
        if constexpr (!counted) BaseGraph<T>::neighbour_state_counts(node, out);
        else {
            out.clear();
            const T own = _states[node];
            for (const auto& entry : _counts.sorted()) {
                const std::size_t c = entry.second - (entry.first == own ? 1 : 0);
                if (c > 0) out.emplace_back(entry.first, c);
            }
        }
    }

    // Number of nodes (of the whole graph) currently holding opinion s.
    std::size_t count_in_state(T s) const {
        if constexpr (!counted) return static_cast<std::size_t>(std::count(_states.begin(), _states.end(), s));
        else                    return _counts.count(s);
    }

private:
    static constexpr bool counted = std::is_integral<T>::value;

    struct NoCounts {};
    std::vector<T> _states;
    std::conditional_t<counted, OpinionCounts<T>, NoCounts> _counts;   // nodes per opinion
};

#endif // ODSP_COMPLETEGRAPH_HPP
