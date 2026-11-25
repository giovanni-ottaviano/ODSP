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

// ------------------------------------------------ HK update modes (fix round R20)

TEST_CASE(hk_existing_update_modes_unchanged) {
    // Golden values recorded with the code before the random-order sweep mode was added: the
    // synchronous and random-with-replacement modes must give the same trajectories, through
    // the bool constructor and through HKUpdate.
    for (int api = 0; api < 2; ++api)
        for (int mode = 0; mode < 2; ++mode) {
            const HKUpdate u = mode == 0 ? HKUpdate::Synchronous : HKUpdate::Random;
            std::mt19937 rng(21 + mode);
            auto g = complete_uniform(200, rng);
            std::unique_ptr<HegselmannKrauseModel<>> m =
                api == 0 ? std::make_unique<HegselmannKrauseModel<>>(std::move(g), &rng, 0.15, mode == 0)
                         : std::make_unique<HegselmannKrauseModel<>>(std::move(g), &rng, 0.15, u);
            CHECK(m->update() == u && m->synchronous() == (mode == 0));
            m->advance(mode == 0 ? 5 : 3, true, false);
            const auto& x = m->get_graph().states();
            CHECK_NEAR(x[0], mode == 0 ? 0.75385546431205686 : 0.79577427969661874, 1e-13);
            CHECK_NEAR(x[57], mode == 0 ? 0.31227517749784495 : 0.82176843597863869, 1e-13);
            CHECK_NEAR(sum_of(x), mode == 0 ? 100.92329283839362 : 101.85777086473013, 1e-11);

            std::mt19937 rng2(31 + mode);
            auto lat = std::make_unique<RegularLattice<double>>(create_uniform_opinions(100, rng2), std::vector<int>{10, 10});
            std::unique_ptr<HegselmannKrauseModel<>> l =
                api == 0 ? std::make_unique<HegselmannKrauseModel<>>(std::move(lat), &rng2, 0.3, mode == 0)
                         : std::make_unique<HegselmannKrauseModel<>>(std::move(lat), &rng2, 0.3, u);
            l->advance(4, true, false);
            const auto& y = l->get_graph().states();
            CHECK_NEAR(y[0], mode == 0 ? 0.35129543564337529 : 0.20101225810197995, 1e-13);
            CHECK_NEAR(y[57], mode == 0 ? 0.55722869949364118 : 0.15659462249014081, 1e-13);
            CHECK_NEAR(sum_of(y), mode == 0 ? 49.619145311264511 : 52.483885254337373, 1e-11);
        }
}

TEST_CASE(hk_sweep_mode_updates_every_agent_once_per_sweep) {
    // With distinct opinions every update moves the agent, so the number of opinion changes
    // counts the updates and persistence tells which agents were updated.
    std::mt19937 rng(41);
    HegselmannKrauseModel<> m(complete_uniform(40, rng), &rng, 0.3, HKUpdate::Sweep);
    CHECK(m.update() == HKUpdate::Sweep && !m.synchronous());
    m.advance(20, /*use_mc_steps=*/false, false);              // half a sweep: 20 distinct agents
    CHECK(m.get_changes() == 20);
    CHECK_NEAR(m.persistence(), 0.5, 1e-12);
    m.advance(20, false, false);                               // the other 20: the sweep is complete
    CHECK(m.get_changes() == 40);
    CHECK_NEAR(m.persistence(), 0.0, 1e-12);
    m.reset_persistence();
    m.advance(1, true, false);                                 // a new order: again each agent once
    CHECK(m.get_changes() == 80);
    CHECK_NEAR(m.persistence(), 0.0, 1e-12);
    CHECK(m.rounds() == 0);

    // Sparse graph too (eps = 1: every neighbour is a confidant, so every update moves the agent).
    HegselmannKrauseModel<> lat(std::make_unique<RegularLattice<double>>(create_uniform_opinions(100, rng),
                                                                         std::vector<int>{10, 10}), &rng, 1.0, HKUpdate::Sweep);
    lat.advance(1, true, false);
    CHECK(lat.get_changes() == 100);
    CHECK_NEAR(lat.persistence(), 0.0, 1e-12);

    // Random with replacement leaves agents out of a sweep (about e^-1 of them).
    HegselmannKrauseModel<> rnd(complete_uniform(400, rng), &rng, 0.3, HKUpdate::Random);
    rnd.advance(1, true, false);
    CHECK(rnd.persistence() > 0.25 && rnd.persistence() < 0.5);
}

TEST_CASE(hk_sweep_mode_is_reproducible) {
    std::mt19937 init(42);
    const std::vector<double> x0 = create_uniform_opinions(300, init);
    auto run = [&x0](unsigned seed) {
        std::mt19937 rng(seed);
        HegselmannKrauseModel<> m(std::make_unique<CompleteGraph<double>>(x0), &rng, 0.12, HKUpdate::Sweep);
        m.advance(10, true, false);
        return m.get_graph().states();
    };
    CHECK(run(7) == run(7));
    CHECK(run(7) != run(8));                                  // the order is random
}

TEST_CASE(hk_sweep_mode_converges_exactly) {
    std::mt19937 rng(43);
    for (double eps : {0.1, 0.25}) {
        HegselmannKrauseModel<> m(complete_uniform(300, rng), &rng, eps, HKUpdate::Sweep);
        m.advance(1000000);
        CHECK(m.get_mc_steps() < 1000000 && m.is_absorbing_state());
        CHECK(frozen_state(m.get_graph(), eps, m.tolerance(), /*strict=*/false));
        const std::vector<double> before = m.get_graph().states();
        m.advance(5, true, false);                           // nothing moves by more than the tolerance
        for (std::size_t i = 0; i < before.size(); ++i)
            CHECK(std::fabs(m.get_graph().get_state(i) - before[i]) <= m.tolerance());
        // the mean is not conserved by sequential averaging, but opinions stay in the range
        for (double v : m.get_graph().states()) CHECK(v >= 0.0 && v <= 1.0);
    }
    HegselmannKrauseModel<> lat(std::make_unique<RegularLattice<double>>(create_uniform_opinions(400, rng),
                                                                         std::vector<int>{20, 20}), &rng, 0.2, HKUpdate::Sweep);
    lat.advance(1000000);
    CHECK(lat.get_mc_steps() < 1000000);
    CHECK(frozen_state(lat.get_graph(), 0.2, lat.tolerance(), false));
}

TEST_CASE(hk_sweep_mode_fast_complete_path_matches_general_path) {
    // Same seed -> same random orders; the complete graph's sorted-block mean must give the
    // trajectory of the general neighbour loop (up to rounding).
    std::mt19937 init(44);
    const std::vector<double> x0 = create_uniform_opinions(300, init);
    std::mt19937 rng_a(45), rng_b(45);
    HegselmannKrauseModel<> fast(std::make_unique<CompleteGraph<double>>(x0), &rng_a, 0.1, HKUpdate::Sweep);
    HegselmannKrauseModel<> slow(std::make_unique<AdjacencyGraph<double>>(x0, complete_edges(300)), &rng_b, 0.1,
                                 HKUpdate::Sweep);
    for (int sweep = 0; sweep < 25; ++sweep) {
        fast.advance(1, true, false);
        slow.advance(1, true, false);
        for (std::size_t i = 0; i < 300; ++i)
            CHECK_NEAR(fast.get_graph().get_state(i), slow.get_graph().get_state(i), 1e-9);
    }
    fast.advance(37, false, false);                            // partial sweeps continue the order
    slow.advance(37, false, false);
    for (std::size_t i = 0; i < 300; ++i) CHECK_NEAR(fast.get_graph().get_state(i), slow.get_graph().get_state(i), 1e-9);
}

// ------------------------------------------------ HK ordered sweeps (fix round R23)

namespace {
// The order in which a sequential HK model updates its agents over `updates` single updates:
// with distinct opinions every update moves the agent, so the one that changed is the one
// updated.
std::vector<std::size_t> update_order(HegselmannKrauseModel<>& m, long long updates) {
    std::vector<std::size_t> order;
    for (long long k = 0; k < updates; ++k) {
        const std::vector<double> before = m.get_graph().states();
        m.advance(1, /*use_mc_steps=*/false, false);
        for (std::size_t i = 0; i < before.size(); ++i)
            if (m.get_graph().get_state(i) != before[i]) order.push_back(i);
    }
    return order;
}
}  // namespace

TEST_CASE(hk_ordered_mode_visits_agents_in_index_order) {
    std::mt19937 rng(51);
    const std::size_t n = 30;
    HegselmannKrauseModel<> m(complete_uniform(n, rng), &rng, 0.3, HKUpdate::Ordered);
    CHECK(m.update() == HKUpdate::Ordered && !m.synchronous());
    std::vector<std::size_t> expected(n);
    std::iota(expected.begin(), expected.end(), std::size_t{0});
    const auto first = update_order(m, static_cast<long long>(n));
    const auto second = update_order(m, static_cast<long long>(n));
    CHECK(first == expected);
    CHECK(second == expected);
    CHECK(m.rounds() == 0);
}

TEST_CASE(hk_ordered_mode_uses_current_opinions) {
    // Brute force: agents 0..N-1 in turn move to the mean of the CURRENT opinions within eps of
    // their own (themselves included), for three sweeps.
    std::mt19937 init(52);
    std::vector<double> x = create_uniform_opinions(60, init);
    std::mt19937 rng(53);
    HegselmannKrauseModel<> m(std::make_unique<CompleteGraph<double>>(x), &rng, 0.15, HKUpdate::Ordered);
    m.advance(3, true, false);
    for (int sweep = 0; sweep < 3; ++sweep)
        for (std::size_t i = 0; i < x.size(); ++i) {
            double sum = 0.0;
            int count = 0;
            for (double y : x)
                if (std::fabs(y - x[i]) <= 0.15) { sum += y; ++count; }
            x[i] = sum / count;
        }
    for (std::size_t i = 0; i < x.size(); ++i) CHECK_NEAR(m.get_graph().get_state(i), x[i], 1e-12);
}

TEST_CASE(hk_ordered_mode_is_deterministic_and_matches_general_path) {
    std::mt19937 init(54);
    const std::vector<double> x0 = create_uniform_opinions(300, init);
    auto run = [&x0](unsigned seed, bool complete) {
        std::mt19937 rng(seed);
        std::unique_ptr<BaseGraph<double>> g;
        if (complete) g = std::make_unique<CompleteGraph<double>>(x0);
        else          g = std::make_unique<AdjacencyGraph<double>>(x0, complete_edges(300));
        HegselmannKrauseModel<> m(std::move(g), &rng, 0.12, HKUpdate::Ordered);
        m.advance(10, true, false);
        m.advance(37, false, false);                         // a partial sweep continues the order
        return m.get_graph().states();
    };
    CHECK(run(7, true) == run(8, true));                    // no randomness after the start
    const auto a = run(7, true), b = run(7, false);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-9);
}

TEST_CASE(hk_sweep_mode_reshuffles_every_sweep) {
    std::mt19937 rng(55);
    const std::size_t n = 30;
    HegselmannKrauseModel<> m(complete_uniform(n, rng), &rng, 0.3, HKUpdate::Sweep);
    const auto first = update_order(m, static_cast<long long>(n));
    const auto second = update_order(m, static_cast<long long>(n));
    auto sorted = [](std::vector<std::size_t> v) { std::sort(v.begin(), v.end()); return v; };
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), std::size_t{0});
    CHECK(sorted(first) == all && sorted(second) == all);   // each a permutation
    CHECK(first != second);                                  // a fresh order each sweep
    CHECK(first != all);
}

TEST_CASE(hk_ordered_mode_converges_exactly) {
    std::mt19937 rng(56);
    HegselmannKrauseModel<> m(complete_uniform(300, rng), &rng, 0.2, HKUpdate::Ordered);
    m.advance(1000000);
    CHECK(m.get_mc_steps() < 1000000 && m.is_absorbing_state());
    CHECK(frozen_state(m.get_graph(), 0.2, m.tolerance(), /*strict=*/false));
}
