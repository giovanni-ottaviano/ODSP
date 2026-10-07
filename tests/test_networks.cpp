#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "TestFramework.hpp"
#include "AdjacencyGraph.hpp"
#include "EdgeList.hpp"
#include "EdgeListIO.hpp"
#include "Generators.hpp"
#include "RegularLattice.hpp"
#include "VoterModel.hpp"
#include "Ensemble.hpp"
#include "Utils.hpp"

namespace {

bool is_simple(std::size_t n, const EdgeList& edges) {
    try { validate_simple_graph(n, edges); return true; }
    catch (const std::invalid_argument&) { return false; }
}

std::set<Edge> normalized(const EdgeList& edges) {
    std::set<Edge> s;
    for (const Edge& e : edges) s.emplace(std::min(e.first, e.second), std::max(e.first, e.second));
    return s;
}

} // namespace

// ------------------------------------------------------------- edge-list tools

TEST_CASE(simplify_edges_drops_loops_and_repeats) {
    EdgeList edges = {{0, 1}, {1, 0}, {2, 2}, {1, 2}, {0, 1}, {3, 3}, {2, 1}};
    const SimplifyReport r = simplify_edges(edges);
    CHECK(r.self_loops == 2);
    CHECK(r.duplicates == 3);
    CHECK(edges == (EdgeList{{0, 1}, {1, 2}}));
}

TEST_CASE(validate_simple_graph_reports_problems) {
    CHECK(is_simple(3, {{0, 1}, {1, 2}}));
    CHECK_THROWS_AS(validate_simple_graph(3, {{0, 3}}), std::invalid_argument);
    CHECK_THROWS_AS(validate_simple_graph(3, {{1, 1}}), std::invalid_argument);
    CHECK_THROWS_AS(validate_simple_graph(3, {{0, 1}, {1, 0}}), std::invalid_argument);
}

TEST_CASE(connected_components_and_largest_component) {
    // Components: {0,1,2} (path), {3,4}, {5} (isolated), {6,7,8,9} (cycle).
    const EdgeList edges = {{0, 1}, {1, 2}, {3, 4}, {6, 7}, {7, 8}, {8, 9}, {9, 6}};
    const Components c = connected_components(10, edges);
    CHECK(c.count() == 4);
    CHECK(c.sizes == (std::vector<std::size_t>{3, 2, 1, 4}));
    CHECK(c.label[0] == c.label[2] && c.label[3] == c.label[4] && c.label[6] == c.label[9]);
    CHECK(c.label[0] != c.label[3] && c.label[5] != c.label[6]);

    const Subgraph lcc = largest_connected_component(10, edges);
    CHECK(lcc.nodes == 4);
    CHECK(lcc.original == (std::vector<std::size_t>{6, 7, 8, 9}));
    CHECK(normalized(lcc.edges) == (std::set<Edge>{{0, 1}, {1, 2}, {2, 3}, {0, 3}}));
    CHECK(degree_sequence(10, edges) == (std::vector<std::size_t>{1, 2, 1, 1, 1, 0, 2, 2, 2, 2}));
}

// -------------------------------------------------------------- AdjacencyGraph

TEST_CASE(adjacency_graph_stores_undirected_edges) {
    // Triangle 0-1-2 plus pendant 3 attached to 2, plus isolated node 4.
    AdjacencyGraph<int> g({1, -1, 1, -1, 1}, {{0, 1}, {1, 2}, {2, 0}, {2, 3}});
    CHECK(g.size() == 5);
    CHECK(g.num_edges() == 4);
    CHECK(g.neighbours(0).to_vector() == (std::vector<std::size_t>{1, 2}));   // in listing order
    CHECK(g.neighbours(2).to_vector() == (std::vector<std::size_t>{1, 0, 3}));
    CHECK(g.neighbours(3).to_vector() == (std::vector<std::size_t>{2}));
    CHECK(g.neighbours(4).empty());
    CHECK(g.degree(2) == 3);
    CHECK(g.count_neighbours_in_state(2, 1) == 1);
    CHECK(g.count_neighbours_in_state(2, -1) == 2);
    g.set_state(3, 1);
    CHECK(g.states() == (std::vector<int>{1, -1, 1, 1, 1}));
}

TEST_CASE(adjacency_graph_rejects_non_simple_input) {
    CHECK_THROWS_AS(AdjacencyGraph<int>({1, 1}, {{0, 0}}), std::invalid_argument);
    CHECK_THROWS_AS(AdjacencyGraph<int>({1, 1}, {{0, 1}, {1, 0}}), std::invalid_argument);
    CHECK_THROWS_AS(AdjacencyGraph<int>({1, 1}, {{0, 2}}), std::invalid_argument);   // node out of range
}

// ------------------------------------------------------------------ generators

TEST_CASE(complete_and_star_generators) {
    CHECK(complete_edges(5).size() == 10);
    CHECK(is_simple(5, complete_edges(5)));
    CHECK(complete_edges(1).empty());
    CHECK(star_edges(4) == (EdgeList{{0, 1}, {0, 2}, {0, 3}, {0, 4}}));
}

TEST_CASE(gnp_has_binomial_edge_count) {
    std::mt19937 rng(1);
    const std::size_t n = 2000;
    const double p = 0.003, pairs = n * (n - 1) / 2.0;
    const EdgeList edges = gnp_edges(n, p, rng);
    CHECK(is_simple(n, edges));
    CHECK_NEAR(static_cast<double>(edges.size()), p * pairs, 4.0 * std::sqrt(p * (1 - p) * pairs));

    CHECK(gnp_edges(50, 0.0, rng).empty());
    CHECK(normalized(gnp_edges(30, 1.0, rng)) == normalized(complete_edges(30)));
    CHECK_THROWS_AS(gnp_edges(10, 1.5, rng), std::invalid_argument);
}

TEST_CASE(gnp_covers_all_pairs_uniformly) {
    // Each of the 6 pairs of 4 nodes should appear with frequency p.
    std::mt19937 rng(2);
    std::map<Edge, int> freq;
    const int trials = 20000;
    for (int t = 0; t < trials; ++t)
        for (const Edge& e : gnp_edges(4, 0.3, rng)) ++freq[e];
    CHECK(freq.size() == 6);
    for (const auto& kv : freq)
        CHECK_NEAR(kv.second / static_cast<double>(trials), 0.3, 4.0 * std::sqrt(0.21 / trials));
}

TEST_CASE(gnm_has_exactly_m_edges) {
    std::mt19937 rng(3);
    const EdgeList edges = gnm_edges(500, 1500, rng);
    CHECK(edges.size() == 1500);
    CHECK(is_simple(500, edges));
    CHECK(normalized(gnm_edges(6, 15, rng)) == normalized(complete_edges(6)));   // m = all pairs
    CHECK_THROWS_AS(gnm_edges(5, 11, rng), std::invalid_argument);
}

TEST_CASE(random_regular_graph_has_uniform_degree) {
    std::mt19937 rng(4);
    for (std::size_t k : {0, 1, 3, 4, 10}) {
        const std::size_t n = 200;
        const EdgeList edges = random_regular_edges(n, k, rng);
        CHECK(is_simple(n, edges));
        const auto deg = degree_sequence(n, edges);
        CHECK(std::all_of(deg.begin(), deg.end(), [k](std::size_t d) { return d == k; }));
    }
    // Dense case: k = n - 1 forces the complete graph.
    CHECK(normalized(random_regular_edges(8, 7, rng)) == normalized(complete_edges(8)));
    CHECK_THROWS_AS(random_regular_edges(5, 3, rng), std::invalid_argument);   // n k odd
    CHECK_THROWS_AS(random_regular_edges(5, 5, rng), std::invalid_argument);   // k >= n
}

TEST_CASE(watts_strogatz_ring_and_rewiring) {
    std::mt19937 rng(5);
    const std::size_t n = 100, k = 6;

    // beta = 0: the ring lattice, node i linked to i +/- 1..3.
    const EdgeList ring = watts_strogatz_edges(n, k, 0.0, rng);
    std::set<Edge> expected;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 1; j <= k / 2; ++j)
            expected.emplace(std::min(i, (i + j) % n), std::max(i, (i + j) % n));
    CHECK(normalized(ring) == expected);

    for (double beta : {0.1, 0.5, 1.0}) {
        const EdgeList ws = watts_strogatz_edges(n, k, beta, rng);
        CHECK(ws.size() == n * k / 2);   // rewiring preserves the edge count
        CHECK(is_simple(n, ws));
        std::size_t kept = 0;
        for (const Edge& e : normalized(ws)) kept += expected.count(e);
        // A fraction ~beta of the ring edges is rewired (loosely: rewired edges
        // can land back on ring positions).
        CHECK_NEAR(1.0 - static_cast<double>(kept) / ring.size(), beta, 0.12);
    }
    CHECK_THROWS_AS(watts_strogatz_edges(10, 3, 0.1, rng), std::invalid_argument);    // odd k
    CHECK_THROWS_AS(watts_strogatz_edges(10, 10, 0.1, rng), std::invalid_argument);   // k >= n
}

TEST_CASE(barabasi_albert_degree_structure) {
    std::mt19937 rng(6);
    const std::size_t n = 20000, m = 3;
    const EdgeList edges = barabasi_albert_edges(n, m, rng);
    CHECK(edges.size() == m * (m + 1) / 2 + (n - m - 1) * m);
    CHECK(is_simple(n, edges));
    CHECK(connected_components(n, edges).count() == 1);

    const auto deg = degree_sequence(n, edges);
    CHECK(*std::min_element(deg.begin(), deg.end()) == m);
    // Scale-free tail: P(k >= K) = m(m+1) / (K(K+1)) for the BA model, so the
    // fraction with degree >= 30 should be ~ 12/930 ~ 1.3%, far above anything
    // an Erdos-Renyi graph with the same mean degree (6) would produce.
    const double frac30 = std::count_if(deg.begin(), deg.end(), [](std::size_t d) { return d >= 30; })
                        / static_cast<double>(n);
    CHECK_NEAR(frac30, 12.0 / 930.0, 0.004);
    CHECK(*std::max_element(deg.begin(), deg.end()) > 150);   // hubs ~ m sqrt(n) ~ 400

    CHECK_THROWS_AS(barabasi_albert_edges(10, 0, rng), std::invalid_argument);
    CHECK_THROWS_AS(barabasi_albert_edges(3, 3, rng), std::invalid_argument);
}

TEST_CASE(generators_are_reproducible) {
    std::mt19937 a(7), b(7);
    CHECK(barabasi_albert_edges(300, 2, a) == barabasi_albert_edges(300, 2, b));
    CHECK(watts_strogatz_edges(300, 4, 0.2, a) == watts_strogatz_edges(300, 4, 0.2, b));
    CHECK(random_regular_edges(300, 3, a) == random_regular_edges(300, 3, b));
    CHECK(gnp_edges(300, 0.02, a) == gnp_edges(300, 0.02, b));
    CHECK(gnm_edges(300, 400, a) == gnm_edges(300, 400, b));
}

// --------------------------------------------------------------- edge-list I/O

TEST_CASE(read_edge_list_handles_common_formats) {
    std::istringstream in(
        "% KONECT-style comment\n"
        "# SNAP-style comment\n"
        "\n"
        "alice bob\n"
        "bob\tcarol 0.5 1286600000\n"   // tab separator, extra columns ignored
        "carol,alice\n"                 // comma separator
        "bob alice\n"                   // repeat in the other orientation
        "dave dave\n"                   // self-loop
        "erin;alice\r\n");              // semicolon, Windows line ending
    const EdgeListFile f = read_edge_list(in);
    CHECK(f.nodes == 5);
    CHECK(f.labels == (std::vector<std::string>{"alice", "bob", "carol", "dave", "erin"}));
    CHECK(f.self_loops_dropped == 1);
    CHECK(f.duplicates_dropped == 1);
    CHECK(normalized(f.edges) == (std::set<Edge>{{0, 1}, {1, 2}, {0, 2}, {0, 4}}));
}

TEST_CASE(read_edge_list_reports_bad_lines) {
    std::istringstream in("1 2\n3\n");
    try {
        read_edge_list(in, "net.txt");
        CHECK(false);   // must throw
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("net.txt:2") != std::string::npos);
    }
    CHECK_THROWS_AS(read_edge_list("/nonexistent/file.txt"), std::runtime_error);
}

TEST_CASE(edge_list_round_trip) {
    std::mt19937 rng(8);
    const EdgeList edges = barabasi_albert_edges(100, 2, rng);
    std::stringstream buf;
    write_edge_list(buf, 100, edges);
    const EdgeListFile f = read_edge_list(buf);
    CHECK(f.nodes == 100);
    CHECK(f.self_loops_dropped == 0 && f.duplicates_dropped == 0);
    // Same graph up to relabelling: map each edge back through the labels.
    std::set<Edge> back;
    for (const Edge& e : f.edges) {
        const std::size_t u = std::stoul(f.labels[e.first]), v = std::stoul(f.labels[e.second]);
        back.emplace(std::min(u, v), std::max(u, v));
    }
    CHECK(back == normalized(edges));
}

// ----------------------------------------------------- open-boundary lattice

TEST_CASE(open_lattice_has_boundary_degrees) {
    std::vector<int> states(4 * 5, 1);
    RegularLattice<int> lat(states, {4, 5}, Boundary::Open);
    CHECK(lat.get_boundary() == Boundary::Open);
    CHECK(lat.degree(lat.linear_index({0, 0})) == 2);   // corner
    CHECK(lat.degree(lat.linear_index({0, 2})) == 3);   // edge
    CHECK(lat.degree(lat.linear_index({2, 2})) == 4);   // bulk
    std::vector<std::size_t> nb = lat.neighbours(lat.linear_index({0, 0})).to_vector();
    std::sort(nb.begin(), nb.end());
    CHECK(nb == (std::vector<std::size_t>{1, 5}));      // (0,1) and (1,0), no wrap-around

    // Total edges of an open L1 x L2 grid: L1 (L2-1) + L2 (L1-1).
    std::size_t half_edges = 0;
    for (std::size_t i = 0; i < lat.size(); ++i) half_edges += lat.degree(i);
    CHECK(half_edges / 2 == 4 * 4 + 5 * 3);
}

TEST_CASE(open_lattice_allows_short_sides) {
    RegularLattice<int> chain(std::vector<int>{1, 1}, {2}, Boundary::Open);   // 2-node chain
    CHECK(chain.neighbours(0).to_vector() == (std::vector<std::size_t>{1}));
    RegularLattice<int> single(std::vector<int>{1}, {1}, Boundary::Open);
    CHECK(single.neighbours(0).empty());
    RegularLattice<int> slab(std::vector<int>(10, 1), {1, 10}, Boundary::Open);   // axis of length 1
    CHECK(slab.degree(slab.linear_index({0, 5})) == 2);
    CHECK_THROWS_AS(RegularLattice<int>(std::vector<int>{1, 1}, {2}), std::invalid_argument);   // periodic
    CHECK_THROWS_AS(RegularLattice<int>(std::vector<int>{}, {0}, Boundary::Open), std::invalid_argument);
}

// ------------------------------------------------- dynamics on generated graphs

TEST_CASE(voter_exit_probability_on_scale_free_network) {
    // On a BA network, start with the hubs up so that the degree-weighted
    // magnetization omega_0 differs strongly from m_0. Node update must give
    // P(+1) = (1 + omega_0) / 2, link update (1 + m_0) / 2.
    std::mt19937 build(9);
    const std::size_t n = 300;
    const EdgeList edges = barabasi_albert_edges(n, 2, build);
    const auto deg = degree_sequence(n, edges);

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&deg](std::size_t a, std::size_t b) { return deg[a] > deg[b]; });
    std::vector<int> states(n, -1);
    for (std::size_t r = 0; r < n / 10; ++r) states[order[r]] = 1;   // top 10% by degree

    double sum_k = 0.0, sum_ks = 0.0, sum_s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double k = static_cast<double>(deg[i]);   // not size_t * int: -1 would wrap around
        sum_k += k; sum_ks += k * states[i]; sum_s += states[i];
    }
    const double omega0 = sum_ks / sum_k, m0 = sum_s / n;
    CHECK(omega0 - m0 > 0.3);   // the test is only meaningful if they differ a lot

    const int replicas = 2000;   // node and link predictions differ by > 10 standard errors
    for (VoterUpdate u : {VoterUpdate::Node, VoterUpdate::Link}) {
        auto factory = [&edges, &states, u](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<VoterModel<int>>(std::make_unique<AdjacencyGraph<int>>(states, edges), &rng, u);
        };
        Ensemble<int> ens(factory, /*seed=*/10);
        const auto res = ens.run(replicas, 10000000, true, /*threads=*/0);
        CHECK(res.consensus_count == replicas);
        const double expected = (1.0 + (u == VoterUpdate::Node ? omega0 : m0)) / 2.0;
        CHECK_NEAR(res.exit_prob_up, expected, 4.0 * std::sqrt(expected * (1.0 - expected) / replicas));
    }
}

TEST_CASE(voter_active_link_plateau_on_random_regular_graph) {
    // Pair approximation for uncorrelated random graphs: rho = rho* (1 - w^2)
    // with rho* = (k-2) / (2(k-1)) = 1/3 for k = 4. Not an exact result, but
    // accurate to well under 1% on random-regular graphs, while a clustered
    // graph of the same degree (a k = 4 ring) is far below it -- a regression
    // test for the dynamics on generated networks.
    std::mt19937 rng(11);
    auto plateau = [&rng](std::size_t n, const EdgeList& edges) {
        double sum = 0.0;
        long long count = 0;
        for (int r = 0; r < 4; ++r) {
            VoterModel<int> m(std::make_unique<AdjacencyGraph<int>>(create_random_lattice(n, 0.5, rng), edges), &rng);
            m.advance(80, true, false, [&](const Model<int>& model) {
                if (model.get_mc_steps() <= 20) return;
                const double w = model.get_degree_weighted_magnetization();
                sum += model.get_active_link_density() / (1.0 - w * w);
                ++count;
            });
        }
        return sum / static_cast<double>(count);
    };
    const std::size_t n = 4000;
    CHECK_NEAR(plateau(n, random_regular_edges(n, 4, rng)), 1.0 / 3.0, 0.01);
    CHECK(plateau(n, watts_strogatz_edges(n, 4, 0.0, rng)) < 0.2);
}
