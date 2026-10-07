#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <random>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "CompleteGraph.hpp"
#include "RegularLattice.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "VoterModel.hpp"
#include "Ensemble.hpp"
#include "Utils.hpp"
#include "AdjacencyGraph.hpp"
#include "Generators.hpp"

namespace {

// Voter model on a complete graph of N nodes, started from round(f0 * N) up
// spins. Active-link tracking is off: it costs O(N) per flip on a dense graph
// and these tests do not need rho.
Ensemble<int>::ModelFactory complete_voter(std::size_t N, double f0) {
    return [N, f0](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        auto g = std::make_unique<CompleteGraph<int>>(create_random_lattice(N, f0, rng));
        return std::make_unique<VoterModel<int>>(std::move(g), &rng, /*track_active_links=*/false);
    };
}

Ensemble<int>::ModelFactory complete_rejection_free_voter(std::size_t N, double f0) {
    return [N, f0](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        auto g = std::make_unique<CompleteGraph<int>>(create_random_lattice(N, f0, rng));
        return std::make_unique<RejectionFreeVoterModel<int>>(std::move(g), &rng, /*track_active_links=*/false);
    };
}

template <typename ModelT>
Ensemble<int>::ModelFactory lattice_model(std::vector<int> dims) {
    return [dims](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        std::size_t n = 1;
        for (int d : dims) n *= static_cast<std::size_t>(d);
        auto g = std::make_unique<RegularLattice<int>>(create_random_lattice(n, 0.5, rng), dims);
        return std::make_unique<ModelT>(std::move(g), &rng);
    };
}

// Star with the hub up and all leaves down, under a given voter update scheme.
Ensemble<int>::ModelFactory star_voter(std::size_t leaves, VoterUpdate update) {
    return [leaves, update](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        std::vector<int> states(leaves + 1, -1);
        states[0] = 1;
        auto g = std::make_unique<AdjacencyGraph<int>>(states, star_edges(leaves));
        return std::make_unique<VoterModel<int>>(std::move(g), &rng, update);
    };
}

} // namespace

// ------------------------------------------------------------ exact results

TEST_CASE(exact_consensus_time_reference_is_sane) {
    // Two nodes, one up: each update flips one of them => exactly 1 update.
    CHECK_NEAR(exact_mean_updates_to_consensus(2, 1), 1.0, 1e-12);
    // Symmetric in n <-> N - n, and zero at the absorbing states.
    CHECK_NEAR(exact_mean_updates_to_consensus(50, 15), exact_mean_updates_to_consensus(50, 35), 1e-6);
    CHECK_NEAR(exact_mean_updates_to_consensus(50, 50), 0.0, 1e-6);
    // Large-N limit: T ~ -N (N-1) [f ln f + (1-f) ln(1-f)] updates.
    const int N = 2000;
    const double mf = -static_cast<double>(N) * (N - 1) * std::log(0.5);
    CHECK_NEAR(exact_mean_updates_to_consensus(N, N / 2) / mf, 1.0, 0.01);
}

TEST_CASE(voter_exit_probability_equals_initial_fraction) {
    // On a regular graph the voter magnetization is a martingale, so the
    // probability of ending at +1 equals the initial up-fraction exactly.
    const int replicas = 3000;
    for (double f0 : {0.3, 0.5, 0.8}) {
        Ensemble<int> ens(complete_voter(50, f0), /*seed=*/100);
        const auto res = ens.run(replicas, /*max_steps=*/100000);
        CHECK(res.consensus_count == replicas);
        const double se = std::sqrt(f0 * (1.0 - f0) / replicas);
        CHECK_NEAR(res.exit_prob_up, f0, 4.0 * se);
    }
}

TEST_CASE(voter_mean_consensus_time_matches_exact_value) {
    // Measured in single updates, so the comparison with the birth-death chain
    // is exact (no rounding up to whole sweeps).
    const int N = 64, replicas = 3000;
    Ensemble<int> ens(complete_voter(N, 0.5), /*seed=*/200);
    const auto res = ens.run(replicas, /*max_steps=*/100000000LL, /*use_mc_steps=*/false);
    CHECK(res.consensus_count == replicas);

    const double exact = exact_mean_updates_to_consensus(N, N / 2);
    const double se = res.stddev_consensus_time / std::sqrt(static_cast<double>(replicas));
    CHECK_NEAR(res.mean_consensus_time, exact, 4.0 * se);
}

// --------------------------------------------------------------- mechanics

TEST_CASE(single_update_consensus_times_are_recorded) {
    // Regression: with use_mc_steps = false the times used to be read from the
    // sweep counter, which stays 0, so every recorded time was 0.
    Ensemble<int> ens(complete_voter(20, 0.5), /*seed=*/300);
    const auto res = ens.run(50, /*max_steps=*/1000000, /*use_mc_steps=*/false);
    CHECK(res.consensus_count == 50);
    CHECK(res.consensus_times.size() == 50);
    for (long long t : res.consensus_times) CHECK(t > 0);
    CHECK(res.mean_consensus_time > 20.0);   // at least ~N updates on average
}

TEST_CASE(sweep_consensus_times_are_recorded) {
    Ensemble<int> ens(complete_voter(20, 0.5), /*seed=*/301);
    const auto res = ens.run(50, /*max_steps=*/100000, /*use_mc_steps=*/true);
    CHECK(res.consensus_count == 50);
    for (long long t : res.consensus_times) CHECK(t > 0);
}

TEST_CASE(ensemble_is_reproducible_from_its_seed) {
    Ensemble<int> a(complete_voter(30, 0.4), /*seed=*/400);
    Ensemble<int> b(complete_voter(30, 0.4), /*seed=*/400);
    Ensemble<int> c(complete_voter(30, 0.4), /*seed=*/401);
    const auto ra = a.run(40, 100000), rb = b.run(40, 100000), rc = c.run(40, 100000);
    CHECK(ra.consensus_times == rb.consensus_times);
    CHECK(ra.exit_prob_up == rb.exit_prob_up);
    CHECK(ra.consensus_times != rc.consensus_times);
}

TEST_CASE(ensemble_counts_censored_runs) {
    // A cap of 1 sweep is far too short for N = 200 to reach consensus.
    Ensemble<int> ens(complete_voter(200, 0.5), /*seed=*/500);
    const auto res = ens.run(20, /*max_steps=*/1);
    CHECK(res.consensus_count == 0);
    CHECK(res.survival_fraction == 1.0);
    CHECK(res.consensus_times.empty());
    CHECK(res.mean_consensus_time == 0.0);
}

TEST_CASE(ensemble_with_no_replicas_is_empty) {
    Ensemble<int> ens(complete_voter(10, 0.5), /*seed=*/600);
    const auto res = ens.run(0, 100);
    CHECK(res.replicas == 0);
    CHECK(res.consensus_count == 0);
}

// ------------------------------------------------------------ update schemes

TEST_CASE(voter_update_schemes_conserve_their_weighted_magnetization) {
    // Star with n leaves, hub up, leaves down. Each scheme conserves a
    // different weighted magnetization M, and P(+1) = (1 + M_0) / 2:
    //   node update:    M = sum k s / sum k        -> P = n / 2n          = 1/2
    //   link update:    M = sum s / N              -> P = 1 / (n + 1)
    //   reverse update: M = sum (s/k) / sum (1/k)  -> P = (1/n) / (1/n + n) = 1 / (1 + n^2)
    const std::size_t n = 10;
    const int replicas = 4000;
    const struct { VoterUpdate update; double expected; } cases[] = {
        {VoterUpdate::Node,    0.5},
        {VoterUpdate::Link,    1.0 / (n + 1)},
        {VoterUpdate::Reverse, 1.0 / (1.0 + n * n)},
    };
    for (const auto& c : cases) {
        Ensemble<int> ens(star_voter(n, c.update), /*seed=*/700);
        const auto res = ens.run(replicas, /*max_steps=*/1000000, true, /*threads=*/0);
        CHECK(res.consensus_count == replicas);
        const double se = std::sqrt(c.expected * (1.0 - c.expected) / replicas);
        CHECK_NEAR(res.exit_prob_up, c.expected, 4.0 * se);
    }

    // The rejection-free model simulates node update, so it must give 1/2 too.
    // (On a regular graph a wrong node weight a_i instead of a_i / k_i would go
    // unnoticed; on the star it would not.)
    auto rf_star = [n](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        std::vector<int> states(n + 1, -1);
        states[0] = 1;
        return std::make_unique<RejectionFreeVoterModel<int>>(
            std::make_unique<AdjacencyGraph<int>>(states, star_edges(n)), &rng);
    };
    Ensemble<int> ens(rf_star, /*seed=*/701);
    const auto res = ens.run(replicas, 1000000, true, /*threads=*/0);
    CHECK(res.consensus_count == replicas);
    CHECK_NEAR(res.exit_prob_up, 0.5, 4.0 * std::sqrt(0.25 / replicas));
}

// ------------------------------------------------------ rejection-free voter

TEST_CASE(rejection_free_voter_matches_exact_consensus_time) {
    // Same exact birth-death-chain value as the plain model, in single updates.
    const int N = 64, replicas = 3000;
    Ensemble<int> ens(complete_rejection_free_voter(N, 0.5), /*seed=*/800);
    const auto res = ens.run(replicas, /*max_steps=*/100000000LL, /*use_mc_steps=*/false, /*threads=*/0);
    CHECK(res.consensus_count == replicas);
    const double exact = exact_mean_updates_to_consensus(N, N / 2);
    const double se = res.stddev_consensus_time / std::sqrt(static_cast<double>(replicas));
    CHECK_NEAR(res.mean_consensus_time, exact, 4.0 * se);
    CHECK_NEAR(res.exit_prob_up, 0.5, 4.0 * std::sqrt(0.25 / replicas));
}

TEST_CASE(rejection_free_voter_matches_plain_voter_on_lattice) {
    // No closed form on a lattice: compare the two algorithms statistically.
    const int replicas = 2000;
    Ensemble<int> plain(lattice_model<VoterModel<int>>({8, 8}), /*seed=*/900);
    Ensemble<int> fast (lattice_model<RejectionFreeVoterModel<int>>({8, 8}), /*seed=*/901);
    const auto a = plain.run(replicas, 1000000, true, /*threads=*/0);
    const auto b = fast.run(replicas, 1000000, true, /*threads=*/0);
    CHECK(a.consensus_count == replicas);
    CHECK(b.consensus_count == replicas);
    const double se = std::sqrt((a.stddev_consensus_time * a.stddev_consensus_time
                               + b.stddev_consensus_time * b.stddev_consensus_time) / replicas);
    CHECK_NEAR(b.mean_consensus_time, a.mean_consensus_time, 4.0 * se);
    CHECK_NEAR(b.exit_prob_up, a.exit_prob_up, 4.0 * std::sqrt(0.5 / replicas));
}

// --------------------------------------------------------------- threading

TEST_CASE(ensemble_result_is_independent_of_thread_count) {
    Ensemble<int> one (complete_voter(40, 0.5), /*seed=*/1000);
    Ensemble<int> many(complete_voter(40, 0.5), /*seed=*/1000);
    const auto a = one.run(64, 100000, true, /*threads=*/1);
    const auto b = many.run(64, 100000, true, /*threads=*/4);
    CHECK(a.consensus_times == b.consensus_times);
    CHECK(a.exit_prob_up == b.exit_prob_up);
    CHECK(a.m1 == b.m1 && a.m2 == b.m2 && a.m4 == b.m4);
    CHECK(a.mean_final_rho == b.mean_final_rho);
}

TEST_CASE(ensemble_propagates_factory_exceptions) {
    for (int threads : {1, 4}) {
        auto calls = std::make_shared<std::atomic<int>>(0);
        auto factory = [calls](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            if (++*calls == 3) throw std::runtime_error("factory failure");
            return std::make_unique<VoterModel<int>>(
                std::make_unique<CompleteGraph<int>>(create_random_lattice(10, 0.5, rng)), &rng);
        };
        Ensemble<int> ens(factory, 1100);
        CHECK_THROWS_AS(ens.run(20, 1000, true, threads), std::runtime_error);
    }
}

// ---------------------------------------------------------------- censoring

TEST_CASE(survival_and_median_without_censoring) {
    Ensemble<int> ens(complete_voter(30, 0.5), /*seed=*/1200);
    const auto res = ens.run(101, /*max_steps=*/100000);
    CHECK(res.consensus_count == 101);
    CHECK(res.max_steps == 100000);

    std::vector<long long> sorted = res.consensus_times;
    std::sort(sorted.begin(), sorted.end());
    CHECK(res.median_consensus_time == static_cast<double>(sorted[50]));   // 51st of 101
    CHECK(res.mean_consensus_time_lower_bound == res.mean_consensus_time); // nothing censored

    CHECK(res.survival(-1) == 1.0);
    CHECK(res.survival(sorted.front() - 1) == 1.0);
    CHECK(res.survival(sorted.back()) == 0.0);
    CHECK_NEAR(res.survival(sorted[50]), 50.0 / 101.0, 1e-12);
    double prev = 1.0;
    for (long long t = 0; t <= sorted.back(); t += 5) {   // non-increasing
        CHECK(res.survival(t) <= prev);
        prev = res.survival(t);
    }
}

TEST_CASE(censoring_is_reported_not_hidden) {
    // Cap at roughly the median consensus time: about half the runs are censored.
    Ensemble<int> probe(complete_voter(30, 0.5), /*seed=*/1300);
    const auto full = probe.run(200, 100000);
    const long long cap = static_cast<long long>(full.median_consensus_time);

    Ensemble<int> capped(complete_voter(30, 0.5), /*seed=*/1300);   // same replicas
    const auto res = capped.run(200, cap);
    CHECK(res.consensus_count > 0 && res.consensus_count < 200);
    CHECK_NEAR(res.survival(cap), res.survival_fraction, 1e-12);
    // The naive mean over finished runs is biased low; the lower bound counts
    // censored runs at the cap, and both stay below the uncensored mean.
    CHECK(res.mean_consensus_time < full.mean_consensus_time);
    CHECK(res.mean_consensus_time <= res.mean_consensus_time_lower_bound);
    CHECK(res.mean_consensus_time_lower_bound <= full.mean_consensus_time);

    // Fewer than half finished => no median.
    Ensemble<int> short_run(complete_voter(30, 0.5), /*seed=*/1300);
    const auto none = short_run.run(200, 2);
    CHECK(none.consensus_count < 100);
    CHECK(std::isnan(none.median_consensus_time));
    CHECK(none.mean_consensus_time_lower_bound <= 2.0);
}
