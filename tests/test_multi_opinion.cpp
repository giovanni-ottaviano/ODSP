#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "AdjacencyGraph.hpp"
#include "CompleteGraph.hpp"
#include "Generators.hpp"
#include "RegularLattice.hpp"
#include "Ensemble.hpp"
#include "OpinionCounts.hpp"
#include "VoterModel.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "NoisyVoterModel.hpp"
#include "MajorityModel.hpp"
#include "PottsModel.hpp"
#include "Statistics.hpp"
#include "Utils.hpp"

namespace {

std::unique_ptr<BaseGraph<int>> lattice_of(std::vector<int> states, std::vector<int> dims) {
    return std::make_unique<RegularLattice<int>>(std::move(states), std::move(dims));
}

std::unique_ptr<BaseGraph<int>> uniform_lattice(int L, int value) {
    return lattice_of(std::vector<int>(static_cast<std::size_t>(L * L), value), {L, L});
}

} // namespace

// ============================================================ OpinionCounts

TEST_CASE(opinion_counts_track_moves) {
    OpinionCounts<int> c;
    c.reset({1, -1, 1, 1, -1});
    CHECK(c.distinct() == 2);
    CHECK(c.count(1) == 3 && c.count(-1) == 2 && c.count(0) == 0 && c.count(7) == 0);
    CHECK(c.max_count() == 3);
    c.move(-1, 1);
    c.move(-1, 1);
    CHECK(c.distinct() == 1 && c.count(1) == 5 && c.count(-1) == 0);
    CHECK(c.sorted() == (std::vector<std::pair<int, std::size_t>>{{1, 5}}));
}

TEST_CASE(opinion_counts_grow_their_range) {
    OpinionCounts<int> c;
    c.reset({3, 3, 4});
    c.move(3, -5);    // below the initial range
    c.move(4, 40);    // above it
    CHECK(c.sorted() == (std::vector<std::pair<int, std::size_t>>{{-5, 1}, {3, 1}, {40, 1}}));
    CHECK(c.distinct() == 3);
}

TEST_CASE(opinion_counts_switch_to_map_for_sparse_labels) {
    OpinionCounts<long long> c;
    c.reset({0, 1000000000000LL, 0});   // range far beyond the dense limit
    CHECK(c.distinct() == 2);
    CHECK(c.count(1000000000000LL) == 1 && c.count(0) == 2);
    c.move(0, -7);
    CHECK(c.sorted() == (std::vector<std::pair<long long, std::size_t>>{{-7, 1}, {0, 1}, {1000000000000LL, 1}}));

    OpinionCounts<int> d;            // dense at first, then a far label forces the switch
    d.reset({0, 1, 2});
    d.move(2, 2000000000);
    CHECK(d.count(2000000000) == 1 && d.count(2) == 0 && d.distinct() == 3);
    CHECK(d.max_count() == 1);
}

// ===================================================== initial conditions

TEST_CASE(opinion_initial_conditions) {
    std::mt19937 rng(1);
    const std::vector<int> v = create_opinions_with_fractions(10, {0.25, 0.25, 0.5}, rng);
    std::map<int, int> c;
    for (int x : v) ++c[x];
    // floor gives 2, 2, 5 (sum 9); the leftover node goes to a largest remainder (0.5 each for 0 and 1).
    CHECK(v.size() == 10 && c[2] == 5 && c[0] + c[1] == 5 && std::abs(c[0] - c[1]) == 1);

    const std::vector<int> b = create_balanced_opinions(11, 3, rng);
    std::map<int, int> cb;
    for (int x : b) ++cb[x];
    CHECK(cb.size() == 3 && cb[0] == 4 && cb[1] == 4 && cb[2] == 3);

    CHECK(create_distinct_opinions(4) == (std::vector<int>{0, 1, 2, 3}));
    CHECK_THROWS_AS(create_opinions_with_fractions(10, {0.5, 0.4}, rng), std::invalid_argument);
    CHECK_THROWS_AS(create_opinions_with_fractions(10, {1.2, -0.2}, rng), std::invalid_argument);
    CHECK_THROWS_AS(create_balanced_opinions(10, 0, rng), std::invalid_argument);
}

// ================================================ bookkeeping with q opinions

TEST_CASE(multi_opinion_bookkeeping_in_all_models) {
    std::mt19937 rng(2);
    const EdgeList net = barabasi_albert_edges(150, 2, rng);
    VoterModel<int> voter(lattice_of(create_balanced_opinions(100, 4, rng), {10, 10}), &rng);
    check_bookkeeping(voter, 100);
    RejectionFreeVoterModel<int> rf(lattice_of(create_balanced_opinions(100, 4, rng), {10, 10}), &rng);
    check_bookkeeping(rf, 100);
    VoterModel<int> distinct(lattice_of(create_distinct_opinions(64), {8, 8}), &rng);
    check_bookkeeping(distinct, 100);
    MajorityModel<int> majority(std::make_unique<AdjacencyGraph<int>>(create_balanced_opinions(150, 3, rng), net), &rng);
    check_bookkeeping(majority, 50);
    NoisyVoterModel<int> noisy(lattice_of(create_balanced_opinions(100, 5, rng), {10, 10}), &rng, 0.1,
                               std::vector<int>{0, 1, 2, 3, 4});
    check_bookkeeping(noisy, 100);
    PottsModel<int> potts_hb(lattice_of(create_balanced_opinions(100, 3, rng), {10, 10}), &rng, 3, 1.0);
    check_bookkeeping(potts_hb, 50);
    PottsModel<int> potts_m(std::make_unique<CompleteGraph<int>>(create_balanced_opinions(60, 4, rng)), &rng, 4, 20.0,
                            1.0, PottsDynamics::Metropolis);
    check_bookkeeping(potts_m, 50);
}

TEST_CASE(voter_never_creates_opinions) {
    std::mt19937 rng(3);
    VoterModel<int> m(lattice_of(create_distinct_opinions(100), {10, 10}), &rng);
    std::size_t previous = m.number_of_opinions();
    CHECK(previous == 100);
    for (int s = 0; s < 300; ++s) {
        m.advance(1, true, false);
        CHECK(m.number_of_opinions() <= previous);
        previous = m.number_of_opinions();
    }
    CHECK(previous < 30);
}

// ===================================================== exact voter results

TEST_CASE(multi_opinion_voter_fixation_equals_initial_share) {
    // Each opinion's share is a martingale on a regular graph, so opinion k
    // wins with probability equal to its initial share, for any number of opinions.
    const std::vector<double> shares = {0.2, 0.3, 0.5};
    const int replicas = 3000;
    Ensemble<int> ens([&shares](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<VoterModel<int>>(
            std::make_unique<CompleteGraph<int>>(create_opinions_with_fractions(50, shares, rng)), &rng);
    }, /*seed=*/4);
    const auto res = ens.run(replicas, 1000000, true, /*threads=*/0);
    CHECK(res.consensus_count == replicas);
    CHECK(res.fixation_counts.size() == 3);
    CHECK(res.mean_final_opinions == 1.0);
    for (int k = 0; k < 3; ++k) {
        const double p = shares[k];
        CHECK_NEAR(res.fixation_probability(k), p, 4.0 * std::sqrt(p * (1 - p) / replicas));
    }
    CHECK(res.fixation_probability(7) == 0.0);
}

TEST_CASE(voter_consensus_time_from_distinct_opinions) {
    // Starting from N distinct opinions on a complete graph, the voter model is
    // dual to N coalescing random walkers: with k lineages left an update
    // merges two of them with probability k (k-1) / (N (N-1)), so the mean
    // number of updates to consensus is N (N-1) sum_{k=2..N} 1/(k(k-1)) = (N-1)^2.
    const int N = 30, replicas = 3000;
    for (int algorithm = 0; algorithm < 2; ++algorithm) {
        Ensemble<int> ens([algorithm](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            auto g = std::make_unique<CompleteGraph<int>>(create_distinct_opinions(N));
            if (algorithm == 0) return std::make_unique<VoterModel<int>>(std::move(g), &rng);
            return std::make_unique<RejectionFreeVoterModel<int>>(std::move(g), &rng);
        }, /*seed=*/5 + algorithm);
        const auto res = ens.run(replicas, 100000000LL, /*use_mc_steps=*/false, /*threads=*/0);
        CHECK(res.consensus_count == replicas);
        const double se = res.stddev_consensus_time / std::sqrt(static_cast<double>(replicas));
        CHECK_NEAR(res.mean_consensus_time, static_cast<double>((N - 1) * (N - 1)), 4.0 * se);
        // By symmetry every initial opinion is equally likely to win.
        CHECK(res.fixation_counts.size() > 25);
    }
}

TEST_CASE(multi_opinion_noisy_voter_at_full_noise) {
    // noise = 1: every update draws a fresh opinion uniformly from q, so the
    // stationary counts are multinomial: share of opinion 0 has mean 1/q and
    // N * variance = (1/q)(1 - 1/q).
    const int q = 4;
    std::mt19937 rng(6);
    NoisyVoterModel<int> m(std::make_unique<CompleteGraph<int>>(create_balanced_opinions(200, q, rng)), &rng, 1.0,
                           std::vector<int>{0, 1, 2, 3});
    const auto s = m.sample_stationary([](const Model<int>& model) {
        return static_cast<double>(model.count_of(0)) / static_cast<double>(model.get_graph().size());
    }, /*burn_in=*/20, /*n_samples=*/8000, /*interval=*/1);
    CHECK_NEAR(s.mean, 1.0 / q, 4.0 * s.mean_err);
    CHECK_NEAR(s.susceptibility, (1.0 / q) * (1.0 - 1.0 / q), 4.0 * s.susceptibility_err);
    CHECK_THROWS_AS(NoisyVoterModel<int>(uniform_lattice(3, 0), &rng, 0.1, std::vector<int>{}), std::invalid_argument);
}

TEST_CASE(sample_stationary_observable_overload) {
    // The default overload samples m/site; passing that observable explicitly
    // must give the identical series from an identical model and RNG.
    std::mt19937 rng_a(7), rng_b(7);
    NoisyVoterModel<int> a(uniform_lattice(6, 1), &rng_a, 0.2);
    NoisyVoterModel<int> b(uniform_lattice(6, 1), &rng_b, 0.2);
    const auto sa = a.sample_stationary(10, 100, 2);
    const auto sb = b.sample_stationary([](const Model<int>& m) { return m.get_magnetization_per_site(); }, 10, 100, 2);
    CHECK(sa.m == sb.m);
    CHECK(sa.binder == sb.binder);
}

// ===================================================================== Potts

TEST_CASE(potts_matches_exact_enumeration_on_3x3) {
    // All 3^9 states of a periodic 3x3 lattice, q = 3, T = 1.
    const int q = 3;
    const double T = 1.0;
    RegularLattice<int> geometry(std::vector<int>(9, 0), {3, 3});
    double Z = 0.0, sumE = 0.0, sumOrder = 0.0;
    std::vector<int> spin(9);
    for (int config = 0; config < 19683; ++config) {
        int c = config;
        int count[3] = {0, 0, 0};
        for (int i = 0; i < 9; ++i) { spin[i] = c % 3; c /= 3; ++count[spin[i]]; }
        double E = 0.0;
        for (std::size_t i = 0; i < 9; ++i)
            for (std::size_t j : geometry.neighbours(i))
                if (j > i && spin[i] == spin[j]) E -= 1.0;
        const double fmax = *std::max_element(count, count + 3) / 9.0;
        const double w = std::exp(-E / T);
        Z += w; sumE += w * E / 9.0; sumOrder += w * (q * fmax - 1.0) / (q - 1.0);
    }
    const double exact_e = sumE / Z, exact_order = sumOrder / Z;

    for (PottsDynamics dyn : {PottsDynamics::HeatBath, PottsDynamics::Metropolis}) {
        std::mt19937 rng(8);
        PottsModel<int> m(lattice_of(create_balanced_opinions(9, q, rng), {3, 3}), &rng, q, T, 1.0, dyn);
        m.advance(1000, true, false);
        std::vector<double> e, order;
        m.advance(200000, true, false, [&](const Model<int>&) {
            e.push_back(m.energy_per_site());
            order.push_back(m.order_parameter());
        });
        auto mean = [](const stats::Moments& mo) { return mo.m1; };
        CHECK_NEAR(stats::moments(e).m1, exact_e, 4.0 * stats::jackknife_error(e, 20, mean));
        CHECK_NEAR(stats::moments(order).m1, exact_order, 4.0 * stats::jackknife_error(order, 20, mean));
    }
}

TEST_CASE(potts_q2_is_ising_with_half_coupling) {
    // q = 2 Potts with J = 1 at T = 1 is the Ising model with J = 1/2, i.e.
    // Ising J = 1 at T = 2: Onsager's m = 0.9113, and the Potts order
    // parameter 2 f_max - 1 is |m|.
    const double onsager = std::pow(1.0 - std::pow(std::sinh(1.0), -4.0), 0.125);
    std::mt19937 rng(9);
    PottsModel<int> m(uniform_lattice(32, 0), &rng, 2, 1.0);
    const auto s = m.sample_stationary([](const Model<int>& model) {
        return static_cast<const PottsModel<int>&>(model).order_parameter();
    }, 300, 6000, 1);
    // Statistical tolerance: the finite-size shift at L = 32, T = 0.88 T_c is
    // far below the error (checked: 0.910 +- 0.001 from much longer runs).
    CHECK_NEAR(s.mean, onsager, 4.0 * s.mean_err);
}

TEST_CASE(potts_q3_transition_on_square_lattice) {
    // T_c = 1 / ln(1 + sqrt 3) = 0.995: ordered below, disordered above.
    auto order = [](double T, unsigned seed) {
        std::mt19937 rng(seed);
        PottsModel<int> m(uniform_lattice(24, 0), &rng, 3, T);
        return m.sample_stationary([](const Model<int>& model) {
            return static_cast<const PottsModel<int>&>(model).order_parameter();
        }, 300, 500, 1).mean;
    };
    CHECK(order(0.85, 10) > 0.8);
    CHECK(order(1.2, 11) < 0.15);
}

TEST_CASE(potts_validates_parameters) {
    std::mt19937 rng(12);
    CHECK_THROWS_AS(PottsModel<int>(uniform_lattice(3, 0), &rng, 1, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(PottsModel<int>(uniform_lattice(3, 0), &rng, 3, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(PottsModel<int>(uniform_lattice(3, 3), &rng, 3, 1.0), std::invalid_argument);    // state >= q
    CHECK_THROWS_AS(PottsModel<int>(uniform_lattice(3, -1), &rng, 3, 1.0), std::invalid_argument);   // state < 0
    PottsModel<int> m(uniform_lattice(4, 2), &rng, 3, 1.0);
    CHECK(m.order_parameter() == 1.0);
    CHECK(m.energy() == -32.0);   // 4x4 periodic: 32 edges, all agreeing
    CHECK(!m.consensus_is_absorbing());
}

// ============================================================== Ensemble

TEST_CASE(ensemble_reports_surviving_opinions_when_censored) {
    Ensemble<int> ens([](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<VoterModel<int>>(lattice_of(create_distinct_opinions(100), {10, 10}), &rng);
    }, /*seed=*/13);
    const auto res = ens.run(20, /*max_steps=*/5);
    CHECK(res.consensus_count == 0);
    CHECK(res.fixation_counts.empty());
    CHECK(res.mean_final_opinions > 10.0);
}

TEST_CASE(non_integer_opinions_compile_and_reach_consensus) {
    // T = double: no per-opinion counts, consensus detected by comparing states.
    Ensemble<double> ens([](std::mt19937& rng) -> std::unique_ptr<Model<double>> {
        std::vector<double> states = {0.25, 0.5, 0.75, 1.0, 0.25, 0.5};
        return std::make_unique<VoterModel<double>>(std::make_unique<CompleteGraph<double>>(states), &rng);
    }, /*seed=*/14);
    const auto res = ens.run(200, 100000);
    CHECK(res.consensus_count == 200);
    double total = 0.0;
    for (const auto& entry : res.fixation_counts) total += res.fixation_probability(entry.first);
    CHECK_NEAR(total, 1.0, 1e-12);
    CHECK_NEAR(res.fixation_probability(0.25), 2.0 / 6.0, 4.0 * std::sqrt((2.0 / 6.0) * (4.0 / 6.0) / 200));
}
