#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "TestFramework.hpp"
#include "AdjacencyGraph.hpp"
#include "CompleteGraph.hpp"
#include "Generators.hpp"
#include "RegularLattice.hpp"
#include "Ensemble.hpp"
#include "OpinionClusters.hpp"
#include "DeffuantModel.hpp"
#include "HegselmannKrauseModel.hpp"
#include "Utils.hpp"

namespace {

using Graph = std::unique_ptr<BaseGraph<double>>;

Graph complete_uniform(std::size_t n, std::mt19937& rng) {
    return std::make_unique<CompleteGraph<double>>(create_uniform_opinions(n, rng));
}

double sum_of(const std::vector<double>& x) { return std::accumulate(x.begin(), x.end(), 0.0); }

// Every neighbouring pair either cannot interact or agrees within tol.
bool frozen_state(const BaseGraph<double>& g, double eps, double tol, bool strict) {
    for (std::size_t i = 0; i < g.size(); ++i)
        for (std::size_t j : g.neighbours(i)) {
            const double d = std::fabs(g.get_state(i) - g.get_state(j));
            const bool interacts = strict ? d < eps : d <= eps;
            if (interacts && d > tol) return false;
        }
    return true;
}

} // namespace

// =========================================================== cluster tools

TEST_CASE(opinion_clusters_split_at_gaps) {
    const std::vector<double> x = {0.10, 0.9, 0.1001, 0.5, 0.0999, 0.9002, 0.95};
    const auto c = opinion_clusters(x, 0.01);
    CHECK(c.size() == 4);   // {~0.1 x3}, {0.5}, {0.9, 0.9002}, {0.95}
    CHECK(c[0].size == 3 && c[1].size == 1 && c[2].size == 2 && c[3].size == 1);
    CHECK_NEAR(c[0].position, 0.1, 1e-12);
    CHECK(c[0].min == 0.0999 && c[0].max == 0.1001);
    CHECK(count_clusters(x, 0.01) == 4);
    CHECK(count_clusters(x, 0.01, 0.25) == 2);          // clusters with >= 1.75 nodes
    CHECK(count_clusters(x, 0.1) == 3);                  // 0.9.. and 0.95 merge
    CHECK_NEAR(largest_cluster_fraction(x, 0.01), 3.0 / 7.0, 1e-12);
    CHECK(opinion_clusters({}, 0.1).empty());
    CHECK_THROWS_AS(opinion_clusters(x, -1.0), std::invalid_argument);
}

TEST_CASE(uniform_opinions_lie_in_range) {
    std::mt19937 rng(1);
    const auto x = create_uniform_opinions(1000, rng, -2.0, 3.0);
    CHECK(std::all_of(x.begin(), x.end(), [](double v) { return v >= -2.0 && v < 3.0; }));
    CHECK_NEAR(sum_of(x) / 1000.0, 0.5, 0.2);
    CHECK_THROWS_AS(create_uniform_opinions(10, rng, 1.0, 1.0), std::invalid_argument);
}

// ================================================================= Deffuant

TEST_CASE(deffuant_conserves_mean_and_range) {
    std::mt19937 rng(2);
    std::vector<Graph> graphs;
    graphs.push_back(complete_uniform(200, rng));
    graphs.push_back(std::make_unique<RegularLattice<double>>(create_uniform_opinions(400, rng), std::vector<int>{20, 20}));
    graphs.push_back(std::make_unique<AdjacencyGraph<double>>(create_uniform_opinions(300, rng),
                                                              barabasi_albert_edges(300, 2, rng)));
    for (Graph& g : graphs) {
        const std::vector<double> x0 = g->states();
        const double total0 = sum_of(x0);
        const auto [lo, hi] = std::minmax_element(x0.begin(), x0.end());
        const double min0 = *lo, max0 = *hi;
        DeffuantModel<> m(std::move(g), &rng, 0.3, 0.3);
        for (int s = 0; s < 100; ++s) {
            m.advance(1, true, false);
            const auto& x = m.get_graph().states();
            CHECK_NEAR(sum_of(x), total0, 1e-9);
            CHECK_NEAR(m.get_magnetization(), total0, 1e-9);   // incremental bookkeeping
            const auto [l, h] = std::minmax_element(x.begin(), x.end());
            CHECK(*l >= min0 && *h <= max0);
        }
    }
}

TEST_CASE(deffuant_wide_confidence_reaches_consensus_at_the_mean) {
    std::mt19937 rng(3);
    for (int r = 0; r < 10; ++r) {
        DeffuantModel<> m(complete_uniform(300, rng), &rng, /*epsilon=*/1.0);
        const double mean0 = m.get_magnetization_per_site();
        CHECK(m.advance(1000000));                  // stops at convergence, in consensus
        CHECK(m.is_consensus_reached() && m.opinion_range() <= m.tolerance());
        for (double x : m.get_graph().states()) CHECK_NEAR(x, mean0, 1e-6);
    }
}

TEST_CASE(deffuant_pair_interaction_rules) {
    std::mt19937 rng(4);
    // Close enough: with mu = 1/2 they meet halfway in one interaction.
    DeffuantModel<> meet(std::make_unique<CompleteGraph<double>>(std::vector<double>{0.2, 0.4}), &rng, 0.5);
    meet.advance(1, false, false);
    CHECK_NEAR(meet.get_graph().get_state(0), 0.3, 1e-15);
    CHECK_NEAR(meet.get_graph().get_state(1), 0.3, 1e-15);
    // Exactly at the bound: no interaction (strict |d| < epsilon); the run
    // stops as converged after one sweep with nothing changed.
    DeffuantModel<> apart(std::make_unique<CompleteGraph<double>>(std::vector<double>{0.0, 0.25}), &rng, 0.25);
    CHECK(!apart.advance(1000));
    CHECK(apart.get_mc_steps() == 1);
    CHECK(apart.get_graph().states() == (std::vector<double>{0.0, 0.25}));
}

TEST_CASE(deffuant_converged_state_is_frozen_on_sparse_graphs) {
    std::mt19937 rng(5);
    DeffuantModel<> lat(std::make_unique<RegularLattice<double>>(create_uniform_opinions(400, rng),
                                                                 std::vector<int>{20, 20}), &rng, 0.3);
    lat.advance(10000000);
    CHECK(lat.get_mc_steps() < 10000000);
    CHECK(frozen_state(lat.get_graph(), 0.3, lat.tolerance(), /*strict=*/true));
}

TEST_CASE(deffuant_complete_graph_fast_check_matches_edge_check) {
    // CompleteGraph and an explicit complete AdjacencyGraph give the same
    // random sequence, so identical trajectories; the sorted convergence check
    // (complete graphs) and the per-edge one must stop at the same sweep.
    for (double eps : {0.15, 0.4}) {
        std::mt19937 init(6);
        const std::vector<double> x0 = create_uniform_opinions(120, init);
        std::mt19937 rng_a(7), rng_b(7);
        DeffuantModel<> fast(std::make_unique<CompleteGraph<double>>(x0), &rng_a, eps);
        DeffuantModel<> slow(std::make_unique<AdjacencyGraph<double>>(x0, complete_edges(120)), &rng_b, eps);
        fast.advance(1000000);
        slow.advance(1000000);
        CHECK(fast.get_mc_steps() == slow.get_mc_steps());
        CHECK(fast.get_graph().states() == slow.get_graph().states());
    }
}

TEST_CASE(deffuant_cluster_count_follows_one_over_two_epsilon) {
    auto clusters = [](double eps, unsigned seed) {
        Ensemble<double> ens([eps](std::mt19937& rng) -> std::unique_ptr<Model<double>> {
            return std::make_unique<DeffuantModel<>>(complete_uniform(1000, rng), &rng, eps);
        }, seed);
        return ens.run(20, 1000000, true, 0, [](const Model<double>& m) {
            return static_cast<double>(count_clusters(m.get_graph().states(), 1e-3, 0.01));
        });
    };
    const auto narrow = clusters(0.1, 8);
    CHECK(narrow.final_values.size() == 20);
    CHECK(narrow.mean_final_value >= 4.0 && narrow.mean_final_value <= 6.0);   // 1/(2 eps) = 5
    const auto wide = clusters(0.5, 9);
    CHECK(wide.mean_final_value == 1.0);
}

TEST_CASE(deffuant_respects_zealots) {
    std::mt19937 rng(10);
    DeffuantModel<> m(complete_uniform(100, rng), &rng, 0.3);
    m.set_frozen(0);
    const double z = m.get_graph().get_state(0);
    m.advance(500, true, false);
    CHECK(m.get_graph().get_state(0) == z);
}

TEST_CASE(deffuant_validates_parameters) {
    std::mt19937 rng(11);
    CHECK_THROWS_AS(DeffuantModel<>(complete_uniform(5, rng), &rng, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(DeffuantModel<>(complete_uniform(5, rng), &rng, 0.2, 0.6), std::invalid_argument);
    CHECK_THROWS_AS(DeffuantModel<>(complete_uniform(5, rng), &rng, 0.2, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(DeffuantModel<>(complete_uniform(5, rng), &rng, 0.2, 0.5, 0.0), std::invalid_argument);
}

// ======================================================= Hegselmann-Krause

TEST_CASE(hk_wide_confidence_converges_in_one_round) {
    std::mt19937 rng(12);
    HegselmannKrauseModel<> m(complete_uniform(500, rng), &rng, /*epsilon=*/1.0);
    const double mean0 = m.get_magnetization_per_site();
    m.advance(1, true, false);   // one round: everyone averages everyone
    CHECK(m.rounds() == 1);
    CHECK(m.opinion_range() < 1e-12);
    CHECK_NEAR(m.get_graph().get_state(0), mean0, 1e-12);
    CHECK(m.advance(100));       // converged right away
    CHECK(m.rounds() <= 3);
}

TEST_CASE(hk_inclusive_confidence_bound) {
    std::mt19937 rng(13);
    HegselmannKrauseModel<> m(std::make_unique<CompleteGraph<double>>(std::vector<double>{0.0, 0.25}), &rng, 0.25);
    m.advance(1, true, false);
    CHECK_NEAR(m.get_graph().get_state(0), 0.125, 1e-15);
    CHECK_NEAR(m.get_graph().get_state(1), 0.125, 1e-15);
}

TEST_CASE(hk_preserves_opinion_order_on_complete_graph) {
    std::mt19937 rng(14);
    HegselmannKrauseModel<> m(complete_uniform(400, rng), &rng, 0.12);
    std::vector<std::size_t> order(400);
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto& x = m.get_graph().states();
    std::sort(order.begin(), order.end(), [&x](std::size_t a, std::size_t b) { return x[a] < x[b]; });
    for (int round = 0; round < 30; ++round) {
        m.advance(1, true, false);
        for (std::size_t k = 1; k < order.size(); ++k) CHECK(x[order[k - 1]] <= x[order[k]]);
    }
}

TEST_CASE(hk_fast_complete_round_matches_general_round) {
    std::mt19937 init(15);
    const std::vector<double> x0 = create_uniform_opinions(300, init);
    std::mt19937 rng_a(16), rng_b(16);
    HegselmannKrauseModel<> fast(std::make_unique<CompleteGraph<double>>(x0), &rng_a, 0.1);
    HegselmannKrauseModel<> slow(std::make_unique<AdjacencyGraph<double>>(x0, complete_edges(300)), &rng_b, 0.1);
    for (int round = 0; round < 25; ++round) {
        fast.advance(1, true, false);
        slow.advance(1, true, false);
        for (std::size_t i = 0; i < 300; ++i)
            CHECK_NEAR(fast.get_graph().get_state(i), slow.get_graph().get_state(i), 1e-9);
    }
}

TEST_CASE(hk_converges_to_separated_clusters) {
    // A converged state on a complete graph: clusters further apart than epsilon.
    std::mt19937 rng(17);
    for (double eps : {0.05, 0.1, 0.3}) {
        HegselmannKrauseModel<> m(complete_uniform(1000, rng), &rng, eps);
        m.advance(100000);
        CHECK(m.rounds() < 100);   // finite-time convergence, in practice a few dozen rounds at most
        const auto c = m.clusters(1e-6);
        for (std::size_t k = 1; k < c.size(); ++k) CHECK(c[k].position - c[k - 1].position > eps);
        if (eps == 0.3)  CHECK(c.size() == 1);
        if (eps == 0.05) CHECK(c.size() >= 5);
    }
}

TEST_CASE(hk_asynchronous_and_sparse_graphs_converge) {
    std::mt19937 rng(18);
    HegselmannKrauseModel<> async(complete_uniform(300, rng), &rng, 0.15, /*synchronous=*/false);
    async.advance(1000000);
    CHECK(async.get_mc_steps() < 1000000 && async.rounds() == 0);
    CHECK(frozen_state(async.get_graph(), 0.15, async.tolerance(), /*strict=*/false));

    HegselmannKrauseModel<> lattice(std::make_unique<RegularLattice<double>>(create_uniform_opinions(400, rng),
                                                                             std::vector<int>{20, 20}), &rng, 0.2);
    lattice.advance(1000000);
    CHECK(lattice.get_mc_steps() < 1000000);
    CHECK(frozen_state(lattice.get_graph(), 0.2, lattice.tolerance(), false));

    // Single updates accumulate into rounds in synchronous mode.
    HegselmannKrauseModel<> sync(complete_uniform(10, rng), &rng, 0.2);
    sync.advance(25, /*use_mc_steps=*/false, false);
    CHECK(sync.rounds() == 2);
}
