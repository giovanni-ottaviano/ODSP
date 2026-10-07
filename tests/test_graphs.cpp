#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "TestFramework.hpp"
#include "RegularLattice.hpp"
#include "CompleteGraph.hpp"
#include "Utils.hpp"

namespace {

std::vector<int> iota_states(std::size_t n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

std::vector<std::size_t> sorted(const NeighbourView& nb) {
    std::vector<std::size_t> v = nb.to_vector();
    std::sort(v.begin(), v.end());
    return v;
}

// Row-major coordinates of a linear index.
std::vector<int> coords_of(std::size_t i, const std::vector<int>& dims) {
    std::vector<int> c(dims.size());
    for (std::size_t k = dims.size(); k-- > 0; ) {
        c[k] = static_cast<int>(i % static_cast<std::size_t>(dims[k]));
        i /= static_cast<std::size_t>(dims[k]);
    }
    return c;
}

} // namespace

// ---------------------------------------------------------------- RegularLattice

TEST_CASE(lattice_1d_ring_neighbours) {
    RegularLattice<int> lat(iota_states(5), {5});
    CHECK(lat.size() == 5);
    CHECK(sorted(lat.neighbours(0)) == (std::vector<std::size_t>{1, 4}));
    CHECK(sorted(lat.neighbours(2)) == (std::vector<std::size_t>{1, 3}));
    CHECK(sorted(lat.neighbours(4)) == (std::vector<std::size_t>{0, 3}));
}

TEST_CASE(lattice_2d_corner_wraps_on_both_axes) {
    // 3 rows x 4 columns, row-major: (r, c) -> 4r + c.
    RegularLattice<int> lat(iota_states(12), {3, 4});
    // (0,0): up (2,0)=8, down (1,0)=4, left (0,3)=3, right (0,1)=1.
    CHECK(sorted(lat.neighbours(0)) == (std::vector<std::size_t>{1, 3, 4, 8}));
    // (1,2)=6: (0,2)=2, (2,2)=10, (1,1)=5, (1,3)=7.
    CHECK(sorted(lat.neighbours(6)) == (std::vector<std::size_t>{2, 5, 7, 10}));
}

TEST_CASE(lattice_3d_adjacency_properties) {
    const std::vector<int> dims = {3, 4, 5};
    RegularLattice<int> lat(iota_states(60), dims);

    for (std::size_t i = 0; i < lat.size(); ++i) {
        const auto& nb = lat.neighbours(i);
        CHECK(nb.size() == 6);                                       // 2 per dimension
        CHECK(std::set<std::size_t>(nb.begin(), nb.end()).size() == 6); // all distinct
        const std::vector<int> ci = coords_of(i, dims);
        for (std::size_t j : nb) {
            CHECK(j != i);                                           // no self-loop
            const auto& back = lat.neighbours(j);
            CHECK(std::find(back.begin(), back.end(), i) != back.end()); // undirected

            // Exactly one coordinate differs, by +/-1 modulo its side.
            const std::vector<int> cj = coords_of(j, dims);
            int differing = 0;
            for (std::size_t k = 0; k < dims.size(); ++k) {
                if (ci[k] == cj[k]) continue;
                ++differing;
                const int d = ((cj[k] - ci[k]) % dims[k] + dims[k]) % dims[k];
                CHECK(d == 1 || d == dims[k] - 1);
            }
            CHECK(differing == 1);
        }
    }
}

TEST_CASE(lattice_coordinate_and_node_apis_agree) {
    const std::vector<int> dims = {3, 4, 5};
    RegularLattice<int> lat(iota_states(60), dims);

    CHECK(lat.linear_index({0, 0, 0}) == 0);
    CHECK(lat.linear_index({0, 0, 4}) == 4);
    CHECK(lat.linear_index({0, 1, 0}) == 5);
    CHECK(lat.linear_index({1, 0, 0}) == 20);
    CHECK(lat.linear_index({2, 3, 4}) == 59);
    CHECK(lat.get_site({1, 2, 3}) == 20 + 10 + 3);

    lat.set_site({2, 1, 0}, -7);
    CHECK(lat.get_state(lat.linear_index({2, 1, 0})) == -7);
    lat.set_state(0, 42);
    CHECK(lat.get_site({0, 0, 0}) == 42);
    CHECK(lat.states()[0] == 42);
    CHECK(lat.get_dimensions() == dims);
}

TEST_CASE(lattice_rejects_invalid_input) {
    CHECK_THROWS_AS(RegularLattice<int>(std::vector<int>{}, std::vector<int>{}), std::invalid_argument);
    CHECK_THROWS_AS(RegularLattice<int>(iota_states(4), {2, 2}), std::invalid_argument);   // side < 3
    CHECK_THROWS_AS(RegularLattice<int>(iota_states(6), {3, 2}), std::invalid_argument);   // one side < 3
    CHECK_THROWS_AS(RegularLattice<int>(iota_states(10), {3, 3}), std::invalid_argument);  // size mismatch

    RegularLattice<int> lat(iota_states(9), {3, 3});
    CHECK_THROWS_AS(lat.get_site({3, 0}), std::out_of_range);
    CHECK_THROWS_AS(lat.get_site({0, -1}), std::out_of_range);
    CHECK_THROWS_AS(lat.get_site({0}), std::out_of_range);        // wrong number of coordinates
    CHECK_THROWS_AS(lat.set_site({0, 0, 0}, 1), std::out_of_range);
}

TEST_CASE(lattice_move_constructor_matches_copy_constructor) {
    std::vector<int> states = iota_states(27);
    std::vector<int> dims = {3, 3, 3};
    RegularLattice<int> copied(states, dims);
    RegularLattice<int> moved(std::move(states), std::move(dims));
    CHECK(copied.states() == moved.states());
    for (std::size_t i = 0; i < copied.size(); ++i)
        CHECK(copied.neighbours(i).to_vector() == moved.neighbours(i).to_vector());
}

// ----------------------------------------------------------------- CompleteGraph

TEST_CASE(complete_graph_connects_every_other_node) {
    CompleteGraph<int> g(iota_states(7));
    CHECK(g.size() == 7);
    for (std::size_t i = 0; i < g.size(); ++i) {
        const auto& nb = g.neighbours(i);
        CHECK(nb.size() == 6);
        CHECK(std::find(nb.begin(), nb.end(), i) == nb.end());
        CHECK(std::set<std::size_t>(nb.begin(), nb.end()).size() == 6);
    }
    g.set_state(3, -1);
    CHECK(g.get_state(3) == -1);
    CHECK(g.states()[3] == -1);
}

// ------------------------------------------------------------------------- Utils

TEST_CASE(random_lattice_has_requested_up_fraction) {
    std::mt19937 rng(1);
    for (double f : {0.0, 0.25, 0.5, 0.7, 1.0}) {
        const std::vector<int> v = create_random_lattice(200, f, rng);
        CHECK(v.size() == 200);
        CHECK(std::all_of(v.begin(), v.end(), [](int s) { return s == 1 || s == -1; }));
        const long long up = std::count(v.begin(), v.end(), 1);
        CHECK(up == static_cast<long long>(f * 200));
    }
}

TEST_CASE(random_lattice_is_shuffled) {
    std::mt19937 rng(1);
    const std::vector<int> v = create_random_lattice(1000, 0.5, rng);
    // Unshuffled it would be 500 x (+1) then 500 x (-1); a shuffle mixes the halves.
    CHECK(std::count(v.begin(), v.begin() + 500, 1) < 400);
}

TEST_CASE(random_lattice_rejects_bad_fraction) {
    std::mt19937 rng(1);
    CHECK_THROWS_AS(create_random_lattice(10, -0.1, rng), std::invalid_argument);
    CHECK_THROWS_AS(create_random_lattice(10, 1.1, rng), std::invalid_argument);
}

// ----------------------------------------------------------------- NeighbourView

TEST_CASE(implicit_neighbour_view_skips_self) {
    const NeighbourView nb = NeighbourView::all_except(5, 2);
    CHECK(nb.size() == 4);
    CHECK(nb.to_vector() == (std::vector<std::size_t>{0, 1, 3, 4}));
    CHECK(nb[0] == 0 && nb[1] == 1 && nb[2] == 3 && nb[3] == 4);
    CHECK(NeighbourView::all_except(5, 0).to_vector() == (std::vector<std::size_t>{1, 2, 3, 4}));
    CHECK(NeighbourView::all_except(5, 4).to_vector() == (std::vector<std::size_t>{0, 1, 2, 3}));
    CHECK(NeighbourView::all_except(1, 0).empty());
    CHECK(NeighbourView::all_except(0, 0).empty());
}

TEST_CASE(explicit_neighbour_view_wraps_storage) {
    const std::vector<std::size_t> list = {7, 3, 9};
    const NeighbourView nb(list);
    CHECK(nb.size() == 3);
    CHECK(nb[1] == 3);
    CHECK(nb.to_vector() == list);
    CHECK(NeighbourView(list.data() + 1, 2).to_vector() == (std::vector<std::size_t>{3, 9}));
}

// ------------------------------------------------- neighbourhood state queries

namespace {

// Reference answers computed by brute force from neighbours().
template <typename G>
void check_neighbourhood_queries(const G& g, const std::vector<int>& alphabet) {
    std::vector<std::pair<int, std::size_t>> fast;
    for (std::size_t i = 0; i < g.size(); ++i) {
        std::map<int, std::size_t> ref;
        for (std::size_t j : g.neighbours(i)) ++ref[g.get_state(j)];
        for (int s : alphabet)
            CHECK(g.count_neighbours_in_state(i, s) == (ref.count(s) ? ref[s] : 0));

        g.neighbour_state_counts(i, fast);
        std::map<int, std::size_t> got(fast.begin(), fast.end());
        CHECK(got.size() == fast.size());   // no duplicate states
        CHECK(got == ref);                   // same histogram, no zero entries
    }
}

} // namespace

TEST_CASE(complete_graph_fast_queries_match_brute_force) {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> opinion(0, 2);
    std::vector<int> states(12);
    for (int& s : states) s = opinion(rng);
    CompleteGraph<int> g(states);
    check_neighbourhood_queries(g, {0, 1, 2, 3});

    // Counts must follow set_state, including an opinion vanishing entirely.
    for (std::size_t i = 0; i < g.size(); ++i) g.set_state(i, i < 4 ? 1 : 2);
    CHECK(g.count_in_state(0) == 0);
    CHECK(g.count_in_state(1) == 4);
    CHECK(g.count_in_state(2) == 8);
    check_neighbourhood_queries(g, {0, 1, 2});
}

TEST_CASE(lattice_default_queries_match_brute_force) {
    std::mt19937 rng(6);
    std::uniform_int_distribution<int> opinion(-1, 1);
    std::vector<int> states(4 * 5);
    for (int& s : states) s = opinion(rng);
    RegularLattice<int> lat(states, {4, 5});
    check_neighbourhood_queries(lat, {-1, 0, 1});
}
