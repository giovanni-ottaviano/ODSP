#include <cstddef>
#include <random>
#include <vector>

#include "TestFramework.hpp"
#include "AdjacencyGraph.hpp"
#include "CompleteGraph.hpp"
#include "Domains.hpp"
#include "EdgeList.hpp"
#include "RegularLattice.hpp"

// Domains: maximal groups of same-opinion nodes connected through same-opinion
// edges (union-find), and their wrap counts on periodic lattices.

namespace {

// A periodic (or open) L x L lattice with s(x, y) from f(x, y); x is the row.
template <typename F>
RegularLattice<int> square(int L, F f, Boundary b = Boundary::Periodic) {
    std::vector<int> s(static_cast<std::size_t>(L * L));
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) s[static_cast<std::size_t>(x * L + y)] = f(x, y);
    return RegularLattice<int>(s, {L, L}, b);
}

std::vector<std::size_t> sizes(const DomainStats& d) {
    std::vector<std::size_t> out;
    for (const Domain& x : d.domains) out.push_back(x.size);
    return out;
}

bool wraps_unknown(const DomainStats& d) {
    if (d.dimensions != 0) return false;
    for (const Domain& x : d.domains)
        if (x.wraps != -1) return false;
    return true;
}

} // namespace

// ------------------------------------------------------------- sizes

TEST_CASE(domains_consensus_is_one_domain) {
    const auto g = square(8, [](int, int) { return 1; });
    const DomainStats d = find_domains(g);
    CHECK(d.count() == 1);
    CHECK(d.domains[0].size == 64);
    CHECK(d.nodes == 64);
    CHECK_NEAR(d.density(), 1.0 / 64, 1e-15);
    CHECK_NEAR(d.largest_fraction(), 1.0, 1e-15);
    CHECK_NEAR(d.mean_size(), 64.0, 1e-12);
    CHECK_NEAR(d.weighted_mean_size(), 64.0, 1e-12);
}

TEST_CASE(domains_checkerboard_all_singletons) {
    const auto g = square(4, [](int x, int y) { return (x + y) % 2 ? 1 : -1; });
    const DomainStats d = find_domains(g);
    CHECK(d.count() == 16);
    for (const Domain& x : d.domains) CHECK(x.size == 1);
    CHECK_NEAR(d.weighted_mean_size(), 1.0, 1e-15);
}

TEST_CASE(domains_hand_counted_lattice) {
    const auto g = square(4, [](int x, int y) { return x < 2 && y < 2 ? 1 : -1; }, Boundary::Open);
    const DomainStats d = find_domains(g);
    CHECK(sizes(d) == (std::vector<std::size_t>{12, 4}));
    CHECK_NEAR(d.largest_fraction(), 0.75, 1e-15);
    CHECK_NEAR(d.mean_size(), 8.0, 1e-12);
    CHECK_NEAR(d.weighted_mean_size(), (144.0 + 16.0) / 16.0, 1e-12);
    CHECK(wraps_unknown(d));
}

TEST_CASE(domains_adjacency_graph_with_isolated_node) {
    const EdgeList edges = {{0, 1}, {1, 2}, {3, 4}};
    const AdjacencyGraph<int> g({1, 1, -1, 1, 1, 1}, edges);
    CHECK(g.size() == 6);
    const DomainStats d = find_domains(g);
    CHECK(sizes(d) == (std::vector<std::size_t>{2, 2, 1, 1}));
    CHECK(wraps_unknown(d));
}

TEST_CASE(domains_multi_opinion) {
    const RegularLattice<int> g({0, 0, 1, 2, 2, 2}, {6}, Boundary::Open);
    const DomainStats d = find_domains(g);
    CHECK(sizes(d) == (std::vector<std::size_t>{3, 2, 1}));
    CHECK(wraps_unknown(d));
}

TEST_CASE(domains_complete_graph) {
    const CompleteGraph<int> g({0, 1, 0, 2, 0, 1, 0, 1, 0, 0});
    const DomainStats d = find_domains(g);
    CHECK(sizes(d) == (std::vector<std::size_t>{6, 3, 1}));
    CHECK(wraps_unknown(d));
}

TEST_CASE(domains_empty_graph_is_zero) {
    const AdjacencyGraph<int> g(std::vector<int>{}, EdgeList{});
    const DomainStats d = find_domains(g);
    CHECK(d.count() == 0);
    CHECK(d.density() == 0.0 && d.largest_fraction() == 0.0 && d.mean_size() == 0.0 && d.weighted_mean_size() == 0.0);
    CHECK(!d.stripes());
}

// ------------------------------------------------------------- wrapping

namespace {
std::vector<int> wraps_of(const DomainStats& d) {
    std::vector<int> out;
    for (const Domain& x : d.domains) out.push_back(x.wraps);
    return out;
}
} // namespace

TEST_CASE(domains_wrap_horizontal_stripes) {
    const DomainStats d = find_domains(square(8, [](int x, int) { return x < 4 ? 1 : -1; }));
    CHECK(d.dimensions == 2);
    CHECK(sizes(d) == (std::vector<std::size_t>{32, 32}));
    CHECK(wraps_of(d) == (std::vector<int>{1, 1}));
    CHECK(d.stripes());
}

TEST_CASE(domains_wrap_vertical_stripes) {
    const DomainStats d = find_domains(square(8, [](int, int y) { return y < 4 ? 1 : -1; }));
    CHECK(wraps_of(d) == (std::vector<int>{1, 1}));
    CHECK(d.stripes());
}

TEST_CASE(domains_wrap_diagonal_stripes) {
    const DomainStats d = find_domains(square(8, [](int x, int y) { return (x + y) % 8 < 4 ? 1 : -1; }));
    CHECK(sizes(d) == (std::vector<std::size_t>{32, 32}));
    CHECK(wraps_of(d) == (std::vector<int>{1, 1}));
    CHECK(d.stripes());
}

TEST_CASE(domains_wrap_droplet_in_sea) {
    const DomainStats d = find_domains(square(8, [](int x, int y) { return x >= 2 && x < 5 && y >= 2 && y < 5 ? 1 : -1; }));
    CHECK(sizes(d) == (std::vector<std::size_t>{55, 9}));
    CHECK(wraps_of(d) == (std::vector<int>{2, 0}));
    CHECK(!d.stripes());
}

TEST_CASE(domains_wrap_droplet_across_the_boundary) {
    // The same droplet split by the periodic boundary: still a droplet.
    const DomainStats d = find_domains(square(8, [](int x, int y) { return (x >= 7 || x < 2) && (y >= 7 || y < 2) ? 1 : -1; }));
    CHECK(sizes(d) == (std::vector<std::size_t>{55, 9}));
    CHECK(wraps_of(d) == (std::vector<int>{2, 0}));
    CHECK(!d.stripes());
}

TEST_CASE(domains_wrap_consensus) {
    const DomainStats d = find_domains(square(8, [](int, int) { return 1; }));
    CHECK(wraps_of(d) == (std::vector<int>{2}));
    CHECK(!d.stripes());
}

TEST_CASE(domains_wrap_side_three) {
    const DomainStats d = find_domains(square(3, [](int x, int) { return x == 0 ? 1 : -1; }));
    CHECK(sizes(d) == (std::vector<std::size_t>{6, 3}));
    CHECK(wraps_of(d) == (std::vector<int>{1, 1}));
    CHECK(d.stripes());
    const DomainStats c = find_domains(square(3, [](int, int) { return 1; }));
    CHECK(wraps_of(c) == (std::vector<int>{2}));
    CHECK(!c.stripes());
}

TEST_CASE(domains_wrap_3d_slab) {
    std::vector<int> s(64);
    for (int i = 0; i < 64; ++i) s[static_cast<std::size_t>(i)] = i / 16 < 2 ? 1 : -1;   // first coordinate < 2
    const DomainStats d = find_domains(RegularLattice<int>(s, {4, 4, 4}));
    CHECK(d.dimensions == 3);
    CHECK(wraps_of(d) == (std::vector<int>{2, 2}));
    CHECK(d.stripes());
}

TEST_CASE(domains_wrap_ring) {
    const DomainStats d = find_domains(RegularLattice<int>({1, 1, 1, 1, 1, -1, -1, -1, -1, -1}, {10}));
    CHECK(d.dimensions == 1);
    CHECK(wraps_of(d) == (std::vector<int>{0, 0}));
    CHECK(!d.stripes());
    const DomainStats c = find_domains(RegularLattice<int>(std::vector<int>(10, 1), {10}));
    CHECK(wraps_of(c) == (std::vector<int>{1}));
    CHECK(!c.stripes());
}

TEST_CASE(domains_open_lattice_wraps_unknown) {
    const DomainStats d = find_domains(square(8, [](int x, int) { return x < 4 ? 1 : -1; }, Boundary::Open));
    CHECK(wraps_unknown(d));
    CHECK(!d.stripes());
}

TEST_CASE(domains_periodic_sizes_match_plain_union_find) {
    // The winding union-find must find the same domains as the plain one on the
    // same edges (an AdjacencyGraph copy of the periodic lattice).
    std::mt19937 rng(11);
    for (int q : {2, 3}) {
        std::uniform_int_distribution<int> pick(0, q - 1);
        std::vector<int> s(16 * 12);
        for (int& x : s) x = pick(rng);
        const RegularLattice<int> lat(s, {16, 12});
        EdgeList edges;
        for (std::size_t i = 0; i < lat.size(); ++i)
            for (std::size_t j : lat.neighbours(i))
                if (j > i) edges.push_back({i, j});
        const DomainStats a = find_domains(lat), b = find_domains(AdjacencyGraph<int>(s, edges));
        CHECK(sizes(a) == sizes(b));
        CHECK(a.dimensions == 2 && b.dimensions == 0);
    }
}
