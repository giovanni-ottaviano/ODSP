#ifndef ODSP_GENERATORS_HPP
#define ODSP_GENERATORS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "EdgeList.hpp"

// Random and deterministic network generators. Each returns the EDGE LIST of a
// simple undirected graph on n nodes; pair it with initial opinions in an
// AdjacencyGraph. All randomness comes from the rng passed in, so a fixed seed
// reproduces the same network. Invalid parameters throw std::invalid_argument.
//
//   complete_edges(n)                 every pair
//   star_edges(leaves)                hub 0 joined to nodes 1..leaves
//   gnp_edges(n, p, rng)              Erdos-Renyi G(n, p): each pair independently with prob p
//   gnm_edges(n, m, rng)              Erdos-Renyi G(n, m): m edges uniformly among all pairs
//   random_regular_edges(n, k, rng)   every node has degree exactly k
//   watts_strogatz_edges(n, k, beta, rng)  ring of k nearest neighbours, edges rewired with prob beta
//   barabasi_albert_edges(n, m, rng)  preferential attachment, P(k) ~ k^-3
//
// Erdos-Renyi graphs are disconnected when the mean degree is small (isolated
// nodes appear below ~ln n); the voter model then cannot reach global
// consensus. Use largest_connected_component() from EdgeList.hpp if needed.

namespace generators_detail {

inline void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

// Set of undirected edges for duplicate checks.
class EdgeSet {
public:
    explicit EdgeSet(std::size_t n) : _n(n) {}
    bool contains(std::size_t u, std::size_t v) const { return _set.count(_key(u, v)) > 0; }
    bool insert(std::size_t u, std::size_t v) { return _set.insert(_key(u, v)).second; }
    void erase(std::size_t u, std::size_t v) { _set.erase(_key(u, v)); }
    void reserve(std::size_t m) { _set.reserve(m); }

private:
    std::uint64_t _key(std::size_t u, std::size_t v) const {
        if (u > v) std::swap(u, v);
        return static_cast<std::uint64_t>(u) * _n + v;
    }
    std::size_t                     _n;
    std::unordered_set<std::uint64_t> _set;
};

} // namespace generators_detail

inline EdgeList complete_edges(std::size_t n) {
    EdgeList edges;
    edges.reserve(n * (n > 0 ? n - 1 : 0) / 2);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) edges.emplace_back(i, j);
    return edges;
}

inline EdgeList star_edges(std::size_t leaves) {
    EdgeList edges;
    edges.reserve(leaves);
    for (std::size_t i = 1; i <= leaves; ++i) edges.emplace_back(0, i);
    return edges;
}

// G(n, p) in O(n + m) time by skipping over absent pairs with geometric jumps
// (Batagelj & Brandes, Phys. Rev. E 71, 036113, 2005).
inline EdgeList gnp_edges(std::size_t n, double p, std::mt19937& rng) {
    generators_detail::require(p >= 0.0 && p <= 1.0, "gnp_edges: p must be in [0, 1].");
    EdgeList edges;
    if (p == 0.0 || n < 2) return edges;
    edges.reserve(static_cast<std::size_t>(p * static_cast<double>(n) * (n - 1) / 2.0 * 1.1) + 16);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double log_q = std::log1p(-p);   // -inf for p = 1: every jump is then 0
    long long v = 1, w = -1;
    const long long N = static_cast<long long>(n);
    while (v < N) {
        const double r = uniform(rng);
        w += 1 + (p == 1.0 ? 0 : static_cast<long long>(std::floor(std::log1p(-r) / log_q)));
        while (w >= v && v < N) { w -= v; ++v; }
        if (v < N) edges.emplace_back(static_cast<std::size_t>(w), static_cast<std::size_t>(v));
    }
    return edges;
}

// G(n, m) by rejection sampling of pairs: efficient for sparse graphs (m well
// below n(n-1)/2); it still terminates for dense ones, just more slowly.
inline EdgeList gnm_edges(std::size_t n, std::size_t m, std::mt19937& rng) {
    const double max_edges = static_cast<double>(n) * (n > 0 ? n - 1 : 0) / 2.0;
    generators_detail::require(static_cast<double>(m) <= max_edges,
                               "gnm_edges: m exceeds n(n-1)/2, the number of node pairs.");
    EdgeList edges;
    edges.reserve(m);
    generators_detail::EdgeSet seen(n);
    seen.reserve(m);
    std::uniform_int_distribution<std::size_t> node(0, n > 0 ? n - 1 : 0);
    while (edges.size() < m) {
        const std::size_t u = node(rng), v = node(rng);
        if (u != v && seen.insert(u, v)) edges.emplace_back(std::min(u, v), std::max(u, v));
    }
    return edges;
}

// Random k-regular graph: points are paired at random, rejecting pairs that
// would create a self-loop or a repeated edge (Steger & Wormald, Combin. Probab.
// Comput. 8, 377, 1999). The result is asymptotically uniform over k-regular
// graphs for k small compared with n; on the rare dead end the attempt restarts.
inline EdgeList random_regular_edges(std::size_t n, std::size_t k, std::mt19937& rng) {
    generators_detail::require(k < n || (k == 0 && n == 0),
                               "random_regular_edges: k must be smaller than n.");
    generators_detail::require((n * k) % 2 == 0, "random_regular_edges: n * k must be even.");

    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::vector<std::size_t> points;   // node i appears k times: its free "stubs"
        points.reserve(n * k);
        for (std::size_t i = 0; i < n; ++i) points.insert(points.end(), k, i);

        EdgeList edges;
        edges.reserve(n * k / 2);
        generators_detail::EdgeSet seen(n);
        seen.reserve(n * k / 2);

        auto take = [&points](std::size_t a, std::size_t b) {   // remove points a and b
            if (a < b) std::swap(a, b);                         // a > b: pop the later one first
            points[a] = points.back(); points.pop_back();
            points[b] = points.back(); points.pop_back();
        };

        bool dead_end = false;
        while (!points.empty() && !dead_end) {
            bool paired = false;
            std::uniform_int_distribution<std::size_t> pick(0, points.size() - 1);
            for (int tries = 0; tries < 64 && !paired; ++tries) {
                const std::size_t a = pick(rng), b = pick(rng);
                const std::size_t u = points[a], v = points[b];
                if (a == b || u == v || seen.contains(u, v)) continue;
                seen.insert(u, v);
                edges.emplace_back(std::min(u, v), std::max(u, v));
                take(a, b);
                paired = true;
            }
            if (paired) continue;

            // Many failures in a row: few points left. Look for any valid pair
            // exhaustively; none means a dead end.
            std::vector<std::pair<std::size_t, std::size_t>> valid;
            for (std::size_t a = 0; a < points.size(); ++a)
                for (std::size_t b = a + 1; b < points.size(); ++b)
                    if (points[a] != points[b] && !seen.contains(points[a], points[b]))
                        valid.emplace_back(a, b);
            if (valid.empty()) { dead_end = true; break; }
            const auto [a, b] = valid[std::uniform_int_distribution<std::size_t>(0, valid.size() - 1)(rng)];
            const std::size_t u = points[a], v = points[b];
            seen.insert(u, v);
            edges.emplace_back(std::min(u, v), std::max(u, v));
            take(a, b);
        }
        if (!dead_end) return edges;
    }
    throw std::runtime_error("random_regular_edges: no valid pairing found after 1000 attempts.");
}

// Watts-Strogatz small world: a ring where each node links to its k nearest
// neighbours (k/2 on each side), then each edge (i, i+j) is rewired with
// probability beta to (i, w), w uniform among nodes not already linked to i.
// beta = 0 gives the ring lattice, beta = 1 a random graph; the number of edges
// n k / 2 is preserved. (Same procedure as NetworkX's watts_strogatz_graph.)
inline EdgeList watts_strogatz_edges(std::size_t n, std::size_t k, double beta, std::mt19937& rng) {
    generators_detail::require(k % 2 == 0, "watts_strogatz_edges: k must be even.");
    generators_detail::require(k < n, "watts_strogatz_edges: k must be smaller than n.");
    generators_detail::require(beta >= 0.0 && beta <= 1.0, "watts_strogatz_edges: beta must be in [0, 1].");

    // Ordered neighbour sets: O(log k) membership tests and a deterministic,
    // platform-independent final edge order.
    std::vector<std::set<std::size_t>> adj(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 1; j <= k / 2; ++j) {
            adj[i].insert((i + j) % n);
            adj[(i + j) % n].insert(i);
        }

    std::bernoulli_distribution rewire(beta);
    std::uniform_int_distribution<std::size_t> node(0, n - 1);
    for (std::size_t j = 1; j <= k / 2; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t old = (i + j) % n;
            if (!rewire(rng)) continue;
            if (adj[i].size() >= n - 1) continue;      // i is already linked to everyone
            if (adj[i].count(old) == 0) continue;      // this edge was removed by an earlier rewiring of `old`
            std::size_t w;
            do { w = node(rng); } while (w == i || adj[i].count(w));
            adj[i].erase(old);   adj[old].erase(i);
            adj[i].insert(w);    adj[w].insert(i);
        }
    }

    EdgeList edges;
    edges.reserve(n * k / 2);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j : adj[i])
            if (j > i) edges.emplace_back(i, j);
    return edges;
}

// Barabasi-Albert preferential attachment: start from a complete graph on m + 1
// nodes, then add nodes one at a time, each linking to m distinct existing
// nodes chosen with probability proportional to their degree. Every node has
// degree >= m, the mean degree tends to 2m, and the degree distribution decays
// as P(k) = 2m(m+1) / (k(k+1)(k+2)) ~ k^-3.
inline EdgeList barabasi_albert_edges(std::size_t n, std::size_t m, std::mt19937& rng) {
    generators_detail::require(m >= 1, "barabasi_albert_edges: m must be >= 1.");
    generators_detail::require(m < n, "barabasi_albert_edges: m must be smaller than n.");

    EdgeList edges = complete_edges(m + 1);
    edges.reserve(edges.size() + (n - m - 1) * m);

    // Every edge endpoint, once per incident edge: a uniform pick from this list
    // is a degree-proportional pick of a node.
    std::vector<std::size_t> endpoints;
    endpoints.reserve(2 * (edges.size() + (n - m - 1) * m));
    for (const Edge& e : edges) { endpoints.push_back(e.first); endpoints.push_back(e.second); }

    std::vector<std::size_t> targets;
    targets.reserve(m);
    for (std::size_t v = m + 1; v < n; ++v) {
        targets.clear();
        std::uniform_int_distribution<std::size_t> pick(0, endpoints.size() - 1);
        while (targets.size() < m) {   // m distinct targets
            const std::size_t t = endpoints[pick(rng)];
            if (std::find(targets.begin(), targets.end(), t) == targets.end()) targets.push_back(t);
        }
        for (std::size_t t : targets) {
            edges.emplace_back(t, v);
            endpoints.push_back(t);
            endpoints.push_back(v);
        }
    }
    return edges;
}

#endif // ODSP_GENERATORS_HPP
