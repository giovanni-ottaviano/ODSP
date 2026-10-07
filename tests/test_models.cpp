#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <stdexcept>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "RegularLattice.hpp"
#include "CompleteGraph.hpp"
#include "VoterModel.hpp"
#include "NoisyVoterModel.hpp"
#include "MajorityModel.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "Utils.hpp"
#include "AdjacencyGraph.hpp"
#include "Generators.hpp"

namespace {

std::unique_ptr<BaseGraph<int>> make_lattice(const std::vector<int>& dims, double up, std::mt19937& rng) {
    const std::size_t n = std::accumulate(dims.begin(), dims.end(), std::size_t{1},
                                          [](std::size_t a, int b) { return a * static_cast<std::size_t>(b); });
    return std::make_unique<RegularLattice<int>>(create_random_lattice(n, up, rng), dims);
}

std::unique_ptr<BaseGraph<int>> make_complete(std::size_t n, double up, std::mt19937& rng) {
    return std::make_unique<CompleteGraph<int>>(create_random_lattice(n, up, rng));
}

// Random graph with heterogeneous degrees: a ring (so it is connected) plus
// random chords, without self-loops or repeated edges.
std::unique_ptr<BaseGraph<int>> make_random_graph(std::size_t n, std::size_t chords, double up, std::mt19937& rng) {
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::set<std::pair<std::size_t, std::size_t>> seen;
    auto add = [&](std::size_t a, std::size_t b) {
        if (a == b) return;
        const auto key = std::minmax(a, b);
        if (seen.insert(key).second) edges.emplace_back(a, b);
    };
    for (std::size_t i = 0; i < n; ++i) add(i, (i + 1) % n);
    std::uniform_int_distribution<std::size_t> node(0, n - 1);
    for (std::size_t c = 0; c < chords; ++c) add(node(rng), node(rng));
    return std::make_unique<AdjacencyGraph<int>>(create_random_lattice(n, up, rng), edges);
}

} // namespace

// -------------------------------------------------- incremental bookkeeping

TEST_CASE(voter_bookkeeping_matches_recount) {
    std::mt19937 rng(11);
    VoterModel<int> on_lattice(make_lattice({6, 7}, 0.5, rng), &rng);
    check_bookkeeping(on_lattice, 200);
    VoterModel<int> on_complete(make_complete(40, 0.3, rng), &rng);
    check_bookkeeping(on_complete, 50);
}

TEST_CASE(majority_bookkeeping_matches_recount) {
    std::mt19937 rng(12);
    MajorityModel<int> on_lattice(make_lattice({4, 5, 3}, 0.5, rng), &rng);
    check_bookkeeping(on_lattice, 50);
    MajorityModel<int> on_ring(make_lattice({30}, 0.5, rng), &rng);   // 2 neighbours: frequent ties
    check_bookkeeping(on_ring, 50);
}

TEST_CASE(noisy_voter_bookkeeping_matches_recount) {
    std::mt19937 rng(13);
    NoisyVoterModel<int> on_lattice(make_lattice({8, 8}, 0.5, rng), &rng, /*noise=*/0.1);
    check_bookkeeping(on_lattice, 100);
    NoisyVoterModel<int> on_complete(make_complete(30, 0.5, rng), &rng, /*noise=*/0.05);
    check_bookkeeping(on_complete, 100);
}

TEST_CASE(voter_update_schemes_bookkeeping_on_heterogeneous_graph) {
    for (VoterUpdate u : {VoterUpdate::Node, VoterUpdate::Link, VoterUpdate::Reverse}) {
        std::mt19937 rng(15);
        VoterModel<int> m(make_random_graph(60, 40, 0.5, rng), &rng, u);
        CHECK(m.update_scheme() == u);
        check_bookkeeping(m, 100);
        VoterModel<int> star(std::make_unique<AdjacencyGraph<int>>(create_random_lattice(11, 0.5, rng), star_edges(10)),
                             &rng, u);
        check_bookkeeping(star, 50);
    }
}

TEST_CASE(majority_bookkeeping_on_heterogeneous_graph) {
    std::mt19937 rng(16);
    MajorityModel<int> m(make_random_graph(60, 40, 0.5, rng), &rng);
    check_bookkeeping(m, 50);
}

TEST_CASE(untracked_active_links_match_tracked) {
    std::mt19937 rng_a(14), rng_b(14);
    VoterModel<int> tracked  (make_lattice({8, 8}, 0.5, rng_a), &rng_a, /*track_active_links=*/true);
    VoterModel<int> untracked(make_lattice({8, 8}, 0.5, rng_b), &rng_b, /*track_active_links=*/false);
    for (int s = 0; s < 50; ++s) {
        tracked.advance(1, true, false);
        untracked.advance(1, true, false);
        // Same seed => identical trajectories; only the bookkeeping path differs.
        CHECK(tracked.get_graph().states() == untracked.get_graph().states());
        CHECK(tracked.get_active_links() == untracked.get_active_links());
    }
}

// ------------------------------------------------------------ Model driver

TEST_CASE(step_counters_follow_the_update_mode) {
    std::mt19937 rng(21);
    NoisyVoterModel<int> m(make_lattice({5, 5}, 0.5, rng), &rng, /*noise=*/0.5);   // never absorbs

    m.advance(3, /*use_mc_steps=*/true);
    CHECK(m.get_mc_steps() == 3);
    CHECK(m.get_updates() == 3 * 25);

    m.advance(10, /*use_mc_steps=*/false);
    CHECK(m.get_mc_steps() == 3);          // single updates do not count as sweeps
    CHECK(m.get_updates() == 3 * 25 + 10);
}

TEST_CASE(advance_stops_at_absorbing_consensus) {
    std::mt19937 rng(22);
    VoterModel<int> m(make_complete(20, 0.5, rng), &rng);
    const bool reached = m.advance(100000, /*use_mc_steps=*/true);
    CHECK(reached);
    CHECK(m.is_consensus_reached());
    CHECK(m.get_mc_steps() < 100000);      // stopped early
    CHECK(m.get_active_links() == 0);
    CHECK(std::fabs(m.get_magnetization_per_site()) == 1.0);

    // Consensus is absorbing for the voter model: nothing changes afterwards.
    const std::vector<int> frozen = m.get_graph().states();
    m.advance(20, true, /*stop_when_absorbed=*/false);
    CHECK(m.get_graph().states() == frozen);
}

TEST_CASE(initial_observables_are_computed_on_construction) {
    std::mt19937 rng(23);
    VoterModel<int> m(make_complete(100, 0.7, rng), &rng);
    CHECK(m.get_magnetization() == 70.0 - 30.0);
    CHECK_NEAR(m.get_magnetization_per_site(), 0.4, 1e-12);
    // Complete graph: active links = n_up * n_down out of N(N-1)/2 edges.
    CHECK(m.get_active_links() == 70 * 30);
    CHECK_NEAR(m.get_active_link_density(), (70.0 * 30.0) / (100.0 * 99.0 / 2.0), 1e-12);
    CHECK(m.get_mc_steps() == 0);
    CHECK(m.get_updates() == 0);
}

// ----------------------------------------------------------- model specifics

TEST_CASE(noisy_voter_validates_noise_and_absorption) {
    std::mt19937 rng(31);
    CHECK_THROWS_AS(NoisyVoterModel<int>(make_complete(5, 0.5, rng), &rng, -0.01), std::invalid_argument);
    CHECK_THROWS_AS(NoisyVoterModel<int>(make_complete(5, 0.5, rng), &rng, 1.01), std::invalid_argument);

    NoisyVoterModel<int> silent(make_complete(5, 0.5, rng), &rng, 0.0);
    NoisyVoterModel<int> noisy (make_complete(5, 0.5, rng), &rng, 0.2);
    CHECK(silent.consensus_is_absorbing());
    CHECK(!noisy.consensus_is_absorbing());
    CHECK(noisy.noise() == 0.2);
}

TEST_CASE(noisy_voter_leaves_consensus) {
    std::mt19937 rng(32);
    NoisyVoterModel<int> m(make_complete(50, 1.0, rng), &rng, /*noise=*/0.2);
    CHECK(m.is_consensus_reached());
    m.advance(5, true, /*stop_when_absorbed=*/true);   // must ignore consensus: not absorbing
    CHECK(m.get_mc_steps() == 5);
    CHECK(!m.is_consensus_reached());
}

TEST_CASE(noisy_voter_at_full_noise_has_independent_spins) {
    // noise = 1: every update draws a fresh +/-1 spin, so in the stationary
    // state the spins are i.i.d. and N <m^2> = 1, i.e. chi = 1 and <m> = 0.
    std::mt19937 rng(33);
    NoisyVoterModel<int> m(make_complete(100, 0.5, rng), &rng, /*noise=*/1.0);
    const auto s = m.sample_stationary(/*burn_in=*/20, /*n_samples=*/4000, /*interval=*/1);
    CHECK(s.m.size() == 4000);
    CHECK_NEAR(s.mean, 0.0, 0.01);
    CHECK_NEAR(s.susceptibility, 1.0, 0.15);
    CHECK_NEAR(s.binder, 0.0, 0.1);    // Gaussian => Binder cumulant ~ 0
    // |m| is half-normal: N (<m^2> - <|m|>^2) = 1 - 2/pi.
    CHECK_NEAR(s.abs_susceptibility, 1.0 - 2.0 / std::acos(-1.0), 0.06);

    // Each sweep refreshes a spin with probability 1 - 1/e, so successive
    // samples have correlation 1/e: tau_int = 1/2 + 1/(e - 1) ~ 1.08.
    const double tau = 0.5 + 1.0 / (std::exp(1.0) - 1.0);
    CHECK_NEAR(s.tau_m, tau, 0.3);
    // Error of <m>: sigma sqrt(2 tau / n), with sigma = 1/sqrt(N) = 0.1.
    const double mean_err = 0.1 * std::sqrt(2.0 * tau / 4000.0);
    CHECK_NEAR(s.mean_err, mean_err, 0.35 * mean_err);
    CHECK(s.susceptibility_err > 0.0 && s.susceptibility_err < 0.1);
    CHECK(s.binder_err > 0.0 && s.binder_err < 0.1);
}

TEST_CASE(sample_stationary_with_no_samples_is_empty) {
    std::mt19937 rng(34);
    NoisyVoterModel<int> m(make_complete(10, 0.5, rng), &rng, 0.1);
    const auto s = m.sample_stationary(5, 0);
    CHECK(s.m.empty());
    CHECK(m.get_mc_steps() == 0);      // nothing was run
}

TEST_CASE(majority_on_complete_graph_follows_the_majority) {
    // With 80% up, every node's neighbourhood has an up majority, so the
    // dynamics can only flip down spins up: deterministic +1 consensus.
    std::mt19937 rng(41);
    MajorityModel<int> m(make_complete(101, 0.8, rng), &rng);
    CHECK(m.advance(1000, true));
    CHECK(m.get_magnetization_per_site() == 1.0);
}

TEST_CASE(majority_keep_current_tie_rule_freezes_ring_domains) {
    // On a ring each node has two neighbours, so a node at a domain wall sees a
    // 1-1 tie. With KeepCurrent nobody ever changes in ++--++--...; with
    // Random the walls move.
    std::vector<int> states(16);
    for (std::size_t i = 0; i < states.size(); ++i) states[i] = (i / 2) % 2 == 0 ? 1 : -1;
    std::mt19937 rng(43);

    MajorityModel<int> keep(std::make_unique<RegularLattice<int>>(states, std::vector<int>{16}), &rng,
                            TieRule::KeepCurrent);
    CHECK(keep.tie_rule() == TieRule::KeepCurrent);
    keep.advance(200, true, false);
    CHECK(keep.get_graph().states() == states);

    MajorityModel<int> random(std::make_unique<RegularLattice<int>>(states, std::vector<int>{16}), &rng,
                              TieRule::Random);
    random.advance(5, true, false);
    CHECK(random.get_graph().states() != states);
}

TEST_CASE(majority_adopts_plurality_with_three_opinions) {
    // 5 x opinion 0, 3 x opinion 1, 2 x opinion 2, fully connected: every node
    // sees opinion 0 as the strict plurality (a 0-holder sees 4/3/2), so the
    // final state is all 0 -- on the fast CompleteGraph and on a generic graph.
    const std::vector<int> states = {0, 1, 2, 0, 1, 0, 2, 0, 1, 0};
    std::vector<std::pair<std::size_t, std::size_t>> all_pairs;
    for (std::size_t i = 0; i < states.size(); ++i)
        for (std::size_t j = i + 1; j < states.size(); ++j) all_pairs.emplace_back(i, j);

    std::mt19937 rng(44);
    MajorityModel<int> fast(std::make_unique<CompleteGraph<int>>(states), &rng);
    MajorityModel<int> generic(std::make_unique<AdjacencyGraph<int>>(states, all_pairs), &rng);
    // Magnetization-based consensus detection assumes +/-1, so do not stop on it.
    fast.advance(50, true, /*stop_when_absorbed=*/false);
    generic.advance(50, true, /*stop_when_absorbed=*/false);
    CHECK(fast.get_graph().states() == std::vector<int>(states.size(), 0));
    CHECK(generic.get_graph().states() == std::vector<int>(states.size(), 0));
    CHECK(fast.get_active_links() == 0);
}

TEST_CASE(majority_flips_a_lone_minority_spin) {
    // A single -1 on a 2D lattice sees four +1 neighbours and must flip, while
    // each +1 node sees at most one -1 among four neighbours and never flips.
    // (On a ring the -1's neighbours would see a 1-1 tie and could flip.)
    std::vector<int> states(25, 1);
    states[12] = -1;
    std::mt19937 rng(42);
    MajorityModel<int> m(std::make_unique<RegularLattice<int>>(states, std::vector<int>{5, 5}), &rng);
    CHECK(m.advance(100, true));
    CHECK(m.get_magnetization() == 25.0);
}

// ------------------------------------------------------ rejection-free voter

namespace {

// W / N recomputed from scratch: the probability that a plain voter update
// changes something.
double effective_probability(const BaseGraph<int>& g) {
    double W = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const std::size_t k = g.degree(i);
        if (k == 0) continue;
        W += static_cast<double>(k - g.count_neighbours_in_state(i, g.get_state(i))) / static_cast<double>(k);
    }
    return W / static_cast<double>(g.size());
}

} // namespace

TEST_CASE(rejection_free_voter_bookkeeping_matches_recount) {
    std::mt19937 rng(51);
    RejectionFreeVoterModel<int> lat(make_lattice({7, 6}, 0.5, rng), &rng);
    RejectionFreeVoterModel<int> net(make_random_graph(50, 30, 0.4, rng), &rng);
    for (Model<int>* m : {static_cast<Model<int>*>(&lat), static_cast<Model<int>*>(&net)}) {
        auto& rf = static_cast<RejectionFreeVoterModel<int>&>(*m);
        CHECK_NEAR(rf.effective_update_probability(), effective_probability(m->get_graph()), 1e-12);
        for (int s = 0; s < 60; ++s) {
            check_bookkeeping(*m, 1);
            CHECK_NEAR(rf.effective_update_probability(), effective_probability(m->get_graph()), 1e-12);
        }
    }
}

TEST_CASE(rejection_free_voter_counts_time_like_the_plain_model) {
    std::mt19937 rng(52);
    RejectionFreeVoterModel<int> m(make_lattice({5, 5}, 0.5, rng), &rng);
    m.advance(3, /*use_mc_steps=*/true, /*stop_when_absorbed=*/false);
    CHECK(m.get_mc_steps() == 3);
    CHECK(m.get_updates() == 75);
    m.advance(7, /*use_mc_steps=*/false, false);
    CHECK(m.get_updates() == 82);

    // At consensus nothing can happen, but time still advances.
    CHECK(m.advance(1000000, true));
    CHECK(m.effective_update_probability() == 0.0);
    const long long sweeps = m.get_mc_steps();
    const std::vector<int> frozen = m.get_graph().states();
    m.advance(10, true, /*stop_when_absorbed=*/false);
    CHECK(m.get_mc_steps() == sweeps + 10);
    CHECK(m.get_graph().states() == frozen);
}
