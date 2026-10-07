#ifndef ODSP_TESTHELPERS_HPP
#define ODSP_TESTHELPERS_HPP

#include <fstream>
#include <map>
#include <string>
#include <numeric>
#include <utility>
#include <vector>

#include "TestFramework.hpp"
#include "Model.hpp"

// Helpers shared by several test files.

// A file's contents without its '#' comment lines (the provenance preamble).
inline std::string read_without_comments(const std::string& path) {
    std::ifstream in(path);
    std::string line, out;
    while (std::getline(in, line))
        if (line.empty() || line[0] != '#') out += line + "\n";
    return out;
}

// Reference values recomputed from scratch on the model's current state.
inline long long count_active_links(const BaseGraph<int>& g) {
    long long half_edges = 0;
    for (std::size_t i = 0; i < g.size(); ++i)
        for (std::size_t j : g.neighbours(i))
            if (g.get_state(i) != g.get_state(j)) ++half_edges;
    return half_edges / 2;
}

inline long long count_edges(const BaseGraph<int>& g) {
    long long half_edges = 0;
    for (std::size_t i = 0; i < g.size(); ++i) half_edges += static_cast<long long>(g.neighbours(i).size());
    return half_edges / 2;
}

inline double sum_states(const BaseGraph<int>& g) {
    return static_cast<double>(std::accumulate(g.states().begin(), g.states().end(), 0LL));
}

inline double degree_weighted_magnetization(const BaseGraph<int>& g) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        num += static_cast<double>(g.degree(i)) * g.get_state(i);
        den += static_cast<double>(g.degree(i));
    }
    return num / den;
}

// After every sweep, the incrementally maintained observables must equal a
// full recount. Runs without early stopping so the check also covers states
// close to (and at) consensus.
inline void check_bookkeeping(Model<int>& m, int sweeps) {
    for (int s = 0; s < sweeps; ++s) {
        m.advance(1, /*use_mc_steps=*/true, /*stop_when_absorbed=*/false);
        const BaseGraph<int>& g = m.get_graph();
        CHECK(m.get_magnetization() == sum_states(g));
        CHECK_NEAR(m.get_degree_weighted_magnetization(), degree_weighted_magnetization(g), 1e-12);
        CHECK(m.get_active_links() == count_active_links(g));
        CHECK_NEAR(m.get_active_link_density(),
                   static_cast<double>(count_active_links(g)) / static_cast<double>(count_edges(g)), 1e-12);
        CHECK(m.is_consensus_reached() == (m.get_active_links() == 0));   // connected graphs

        // Opinion counts against a recount.
        std::map<int, std::size_t> ref;
        for (int v : g.states()) ++ref[v];
        const std::vector<std::pair<int, std::size_t>> expected(ref.begin(), ref.end());
        CHECK(m.opinion_counts() == expected);
        CHECK(m.number_of_opinions() == ref.size());
        CHECK(m.is_consensus_reached() == (ref.size() == 1));
    }
}

// Exact mean number of single updates to consensus for the voter model on a
// complete graph, starting from n up spins. The up-count is a birth-death
// chain with equal up/down probabilities p_k = k (N-k) / (N (N-1)) per update,
// so T_k satisfies  p_k (2 T_k - T_{k+1} - T_{k-1}) = 1,  T_0 = T_N = 0.
// With d_k = T_k - T_{k-1}:  d_{k+1} = d_k - 1/p_k, and T_N = 0 fixes d_1.
inline double exact_mean_updates_to_consensus(int N, int n) {
    auto p = [N](int k) { return static_cast<double>(k) * (N - k) / (static_cast<double>(N) * (N - 1)); };

    // T_N = sum_{k=1..N} d_k = N d_1 - sum_{k=1..N} sum_{j=1..k-1} 1/p_j = 0.
    double nested = 0.0, inner = 0.0;
    for (int k = 1; k <= N; ++k) {
        nested += inner;              // inner = sum_{j<k} 1/p_j
        if (k < N) inner += 1.0 / p(k);
    }
    const double d1 = nested / N;

    double T = 0.0, d = d1;
    for (int k = 1; k <= n; ++k) {
        T += d;
        d -= 1.0 / p(k);
    }
    return T;
}

#endif // ODSP_TESTHELPERS_HPP
