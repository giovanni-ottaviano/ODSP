#ifndef ODSP_EDGELIST_HPP
#define ODSP_EDGELIST_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Undirected edge lists: the common currency between network generators, file
// I/O and AdjacencyGraph. Nodes are indices in [0, n); an edge (u, v) and
// (v, u) are the same edge.
//
// The models require SIMPLE graphs (no self-loops, no repeated edges): their
// incremental bookkeeping counts each neighbour once. Generators produce simple
// graphs; for external data use simplify_edges(). Voter-like dynamics only reach
// global consensus on a CONNECTED graph; use largest_connected_component() when
// in doubt.

using Edge     = std::pair<std::size_t, std::size_t>;
using EdgeList = std::vector<Edge>;

struct SimplifyReport {
    std::size_t self_loops = 0;   // edges (u, u) removed
    std::size_t duplicates = 0;   // repeated edges removed (either orientation)
};

// Make `edges` simple: drop self-loops and repeated edges. On return every edge
// is stored as (min, max) and the list is sorted.
inline SimplifyReport simplify_edges(EdgeList& edges) {
    SimplifyReport report;
    EdgeList kept;
    kept.reserve(edges.size());
    for (const Edge& e : edges) {
        if (e.first == e.second) { ++report.self_loops; continue; }
        kept.emplace_back(std::min(e.first, e.second), std::max(e.first, e.second));
    }
    std::sort(kept.begin(), kept.end());
    const auto last = std::unique(kept.begin(), kept.end());
    report.duplicates = static_cast<std::size_t>(kept.end() - last);
    kept.erase(last, kept.end());
    edges.swap(kept);
    return report;
}

// Throws std::invalid_argument naming the first problem if `edges` is not a
// simple graph on n nodes.
inline void validate_simple_graph(std::size_t n, const EdgeList& edges) {
    EdgeList sorted;
    sorted.reserve(edges.size());
    for (const Edge& e : edges) {
        if (e.first >= n || e.second >= n)
            throw std::invalid_argument("edge (" + std::to_string(e.first) + ", " + std::to_string(e.second) +
                                        ") refers to a node >= " + std::to_string(n) + ".");
        if (e.first == e.second)
            throw std::invalid_argument("self-loop at node " + std::to_string(e.first) +
                                        " (use simplify_edges() to drop self-loops).");
        sorted.emplace_back(std::min(e.first, e.second), std::max(e.first, e.second));
    }
    std::sort(sorted.begin(), sorted.end());
    const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
    if (dup != sorted.end())
        throw std::invalid_argument("repeated edge (" + std::to_string(dup->first) + ", " +
                                    std::to_string(dup->second) + ") (use simplify_edges() to drop repeats).");
}

inline std::vector<std::size_t> degree_sequence(std::size_t n, const EdgeList& edges) {
    std::vector<std::size_t> degree(n, 0);
    for (const Edge& e : edges) { ++degree[e.first]; ++degree[e.second]; }
    return degree;
}

struct Components {
    std::vector<std::size_t> label;   // label[i] = component of node i, in [0, count)
    std::vector<std::size_t> sizes;   // sizes[c] = number of nodes in component c
    std::size_t count() const { return sizes.size(); }
};

// Connected components (union-find). Isolated nodes are components of size 1.
// Components are numbered in order of their smallest node.
inline Components connected_components(std::size_t n, const EdgeList& edges) {
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    auto find = [&parent](std::size_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (const Edge& e : edges) {
        const std::size_t a = find(e.first), b = find(e.second);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }

    Components c;
    c.label.assign(n, 0);
    std::vector<std::size_t> root_label(n, n);   // n = not yet labelled
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = find(i);
        if (root_label[r] == n) { root_label[r] = c.sizes.size(); c.sizes.push_back(0); }
        c.label[i] = root_label[r];
        ++c.sizes[c.label[i]];
    }
    return c;
}

// A subgraph with nodes renumbered 0..nodes-1; original[i] is the index node i
// had in the full graph.
struct Subgraph {
    std::size_t              nodes = 0;
    EdgeList                 edges;
    std::vector<std::size_t> original;
};

// The largest connected component (the first one found on ties).
inline Subgraph largest_connected_component(std::size_t n, const EdgeList& edges) {
    Subgraph sub;
    if (n == 0) return sub;
    const Components c = connected_components(n, edges);
    const std::size_t best = static_cast<std::size_t>(
        std::max_element(c.sizes.begin(), c.sizes.end()) - c.sizes.begin());

    std::vector<std::size_t> new_index(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        if (c.label[i] != best) continue;
        new_index[i] = sub.original.size();
        sub.original.push_back(i);
    }
    sub.nodes = sub.original.size();
    for (const Edge& e : edges)
        if (c.label[e.first] == best)   // both endpoints share the component
            sub.edges.emplace_back(new_index[e.first], new_index[e.second]);
    return sub;
}

#endif // ODSP_EDGELIST_HPP
