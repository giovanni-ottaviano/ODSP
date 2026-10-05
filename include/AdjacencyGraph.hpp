#ifndef ODSP_ADJACENCYGRAPH_HPP
#define ODSP_ADJACENCYGRAPH_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BaseGraph.hpp"
#include "EdgeList.hpp"

// General undirected graph built from an edge list: random networks from
// Generators.hpp, empirical networks from EdgeListIO.hpp, or any hand-made
// topology. Combine a topology with initial opinions:
//
//   const EdgeList edges = barabasi_albert_edges(N, 3, rng);
//   auto g = std::make_unique<AdjacencyGraph<int>>(create_random_lattice(N, 0.5, rng), edges);
//
// Storage is compressed sparse row: one offsets array and one flat neighbour
// array (each edge stored in both directions), so neighbours(i) is a
// contiguous slice -- compact and cache-friendly for the models' hot loops.
// Neighbours of a node appear in the order their edges were listed.
//
// The edge list must describe a SIMPLE graph on states.size() nodes (no
// self-loops or repeated edges, as the models' bookkeeping requires);
// otherwise the constructor throws std::invalid_argument. Clean external data
// with simplify_edges() first.
template <typename T>
class AdjacencyGraph : public BaseGraph<T> {
public:
    AdjacencyGraph(std::vector<T> states, const EdgeList& edges) : _states(std::move(states)) {
        const std::size_t n = _states.size();
        try {
            validate_simple_graph(n, edges);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("AdjacencyGraph: ") + e.what());
        }

        _offsets.assign(n + 1, 0);
        for (const Edge& e : edges) { ++_offsets[e.first + 1]; ++_offsets[e.second + 1]; }
        for (std::size_t i = 0; i < n; ++i) _offsets[i + 1] += _offsets[i];

        _neighbours.resize(2 * edges.size());
        std::vector<std::size_t> fill(_offsets.begin(), _offsets.end() - 1);
        for (const Edge& e : edges) {
            _neighbours[fill[e.first]++]  = e.second;
            _neighbours[fill[e.second]++] = e.first;
        }
    }

    std::size_t   size() const override { return _states.size(); }
    NeighbourView neighbours(std::size_t node) const override {
        return NeighbourView(_neighbours.data() + _offsets[node], _offsets[node + 1] - _offsets[node]);
    }
    T    get_state(std::size_t node) const override { return _states[node]; }
    void set_state(std::size_t node, T value) override { _states[node] = value; }
    const std::vector<T>& states() const override { return _states; }

    std::size_t num_edges() const { return _neighbours.size() / 2; }

private:
    std::vector<T>           _states;
    std::vector<std::size_t> _offsets;      // node i's neighbours at [_offsets[i], _offsets[i+1])
    std::vector<std::size_t> _neighbours;
};

#endif // ODSP_ADJACENCYGRAPH_HPP
