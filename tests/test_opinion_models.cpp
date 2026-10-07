#include <algorithm>
#include <cmath>
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
#include "VoterModel.hpp"
#include "RejectionFreeVoterModel.hpp"
#include "NoisyVoterModel.hpp"
#include "MajorityModel.hpp"
#include "QVoterModel.hpp"
#include "NonlinearVoterModel.hpp"
#include "MajorityVoteModel.hpp"
#include "IsingModel.hpp"
#include "SznajdModel.hpp"
#include "Statistics.hpp"
#include "Utils.hpp"

namespace {

std::unique_ptr<BaseGraph<int>> lattice(std::vector<int> dims, double up, std::mt19937& rng) {
    std::size_t n = 1;
    for (int d : dims) n *= static_cast<std::size_t>(d);
    return std::make_unique<RegularLattice<int>>(create_random_lattice(n, up, rng), dims);
}

std::unique_ptr<BaseGraph<int>> complete(std::size_t n, double up, std::mt19937& rng) {
    return std::make_unique<CompleteGraph<int>>(create_random_lattice(n, up, rng));
}

std::unique_ptr<BaseGraph<int>> all_up_lattice(int L) {
    return std::make_unique<RegularLattice<int>>(std::vector<int>(static_cast<std::size_t>(L * L), 1),
                                                 std::vector<int>{L, L});
}

// Mean single updates to consensus over an ensemble, compared with the exact
// voter value on a complete graph: for models that must reduce to the voter.
void check_reduces_to_voter(const Ensemble<int>::ModelFactory& factory, int N, unsigned seed) {
    const int replicas = 3000;
    Ensemble<int> ens(factory, seed);
    const auto res = ens.run(replicas, 100000000LL, /*use_mc_steps=*/false, /*threads=*/0);
    CHECK(res.consensus_count == replicas);
    const double se = res.stddev_consensus_time / std::sqrt(static_cast<double>(replicas));
    CHECK_NEAR(res.mean_consensus_time, exact_mean_updates_to_consensus(N, N / 2), 4.0 * se);
    CHECK_NEAR(res.exit_prob_up, 0.5, 4.0 * std::sqrt(0.25 / replicas));
}

} // namespace

// =================================================================== zealots

TEST_CASE(frozen_nodes_never_change_in_any_model) {
    std::mt19937 rng(1);
    std::vector<std::unique_ptr<Model<int>>> models;
    models.push_back(std::make_unique<VoterModel<int>>(lattice({8, 8}, 0.5, rng), &rng));
    models.push_back(std::make_unique<RejectionFreeVoterModel<int>>(lattice({8, 8}, 0.5, rng), &rng));
    models.push_back(std::make_unique<NoisyVoterModel<int>>(lattice({8, 8}, 0.5, rng), &rng, 0.3));
    models.push_back(std::make_unique<MajorityModel<int>>(lattice({8, 8}, 0.5, rng), &rng));
    models.push_back(std::make_unique<QVoterModel<int>>(lattice({8, 8}, 0.5, rng), &rng, 2, 0.2));
    models.push_back(std::make_unique<NonlinearVoterModel<int>>(lattice({8, 8}, 0.5, rng), &rng, 0.5));
    models.push_back(std::make_unique<MajorityVoteModel<int>>(lattice({8, 8}, 0.5, rng), &rng, 0.2));
    models.push_back(std::make_unique<IsingModel<int>>(lattice({8, 8}, 0.5, rng), &rng, 5.0));
    models.push_back(std::make_unique<SznajdModel<int>>(lattice({8, 8}, 0.5, rng), &rng));

    const std::vector<std::size_t> zealots = {0, 9, 18, 27, 36, 45, 54, 63};   // the diagonal
    for (auto& m : models) {
        m->set_frozen(zealots);
        CHECK(m->frozen_count() == zealots.size());
        std::vector<int> frozen_states;
        for (std::size_t z : zealots) frozen_states.push_back(m->get_graph().get_state(z));
        check_bookkeeping(*m, 40);   // observables stay consistent
        for (std::size_t k = 0; k < zealots.size(); ++k)
            CHECK(m->get_graph().get_state(zealots[k]) == frozen_states[k]);
    }
}

TEST_CASE(frozen_set_can_be_changed) {
    std::mt19937 rng(2);
    VoterModel<int> m(lattice({5, 5}, 0.5, rng), &rng);
    CHECK(m.frozen_count() == 0 && !m.is_frozen(3));
    m.set_frozen(3);
    m.set_frozen(std::vector<std::size_t>{3, 4, 7});
    CHECK(m.frozen_count() == 3 && m.is_frozen(3) && m.is_frozen(7));
    m.set_frozen(4, false);
    CHECK(m.frozen_count() == 2 && !m.is_frozen(4));
    m.set_frozen(std::vector<std::size_t>{3, 7}, false);
    CHECK(m.frozen_count() == 0);
}

TEST_CASE(rejection_free_voter_rates_exclude_zealots) {
    // A ring coarsens until few links are active, which drives the model into
    // its jump regime, where frozen nodes must carry zero rate. Its internal
    // total must match a recount (frozen nodes excluded) at every sweep.
    std::mt19937 rng(6);
    RejectionFreeVoterModel<int> m(lattice({200}, 0.5, rng), &rng);
    m.set_frozen(std::vector<std::size_t>{0, 100});
    const int z0 = m.get_graph().get_state(0), z1 = m.get_graph().get_state(100);
    double lowest = 1.0;
    for (int sweep = 0; sweep < 3000; ++sweep) {
        m.advance(1, true, false);
        const BaseGraph<int>& g = m.get_graph();
        double W = 0.0;
        for (std::size_t i = 0; i < g.size(); ++i)
            if (!m.is_frozen(i))
                W += static_cast<double>(g.degree(i) - g.count_neighbours_in_state(i, g.get_state(i))) / g.degree(i);
        CHECK_NEAR(m.effective_update_probability(), W / g.size(), 1e-9);
        lowest = std::min(lowest, m.effective_update_probability());
    }
    CHECK(lowest < 0.05);   // the jump regime (below 0.1) was really exercised
    CHECK(m.get_graph().get_state(0) == z0 && m.get_graph().get_state(100) == z1);
}

TEST_CASE(single_zealot_imposes_its_opinion) {
    // One up zealot among down voters: all-up is the only absorbing state.
    std::vector<int> states(25, -1);
    states[12] = 1;
    std::mt19937 rng(3);
    VoterModel<int> m(std::make_unique<RegularLattice<int>>(states, std::vector<int>{5, 5}), &rng);
    m.set_frozen(12);
    CHECK(m.advance(10000000));
    CHECK(m.get_magnetization() == 25.0);
}

TEST_CASE(voter_with_zealots_has_beta_binomial_stationary_state) {
    // Complete graph with N free voters, Z+ up and Z- down zealots. The number
    // n of up free voters is a birth-death chain with
    //   pi(n+1) / pi(n) = (N-n)(n+Z+) / ((n+1)(N-n-1+Z-)),
    // which is exactly the Beta-Binomial(N, Z+, Z-) distribution:
    //   <n> = N Z+ / Z,  Var n = N Z+ Z- (Z + N) / (Z^2 (Z + 1)),  Z = Z+ + Z-.
    const int N = 60, Zp = 4, Zm = 2, Z = Zp + Zm, total = N + Z;
    const double mean_n = static_cast<double>(N) * Zp / Z;
    const double var_n = static_cast<double>(N) * Zp * Zm * (Z + N) / (static_cast<double>(Z) * Z * (Z + 1));
    // In terms of m/site = (2 (n + Z+) - total) / total:
    const double mean_m = (2.0 * (mean_n + Zp) - total) / total;
    const double chi = total * 4.0 * var_n / (static_cast<double>(total) * total);   // total * Var(m)

    for (int algorithm = 0; algorithm < 2; ++algorithm) {
        std::mt19937 rng(4 + algorithm);
        std::vector<int> states = create_random_lattice(total, 0.5, rng);
        std::vector<std::size_t> zealots;
        for (int z = 0; z < Z; ++z) {
            states[z] = z < Zp ? 1 : -1;
            zealots.push_back(static_cast<std::size_t>(z));
        }
        std::unique_ptr<Model<int>> m;
        if (algorithm == 0) m = std::make_unique<VoterModel<int>>(std::make_unique<CompleteGraph<int>>(states), &rng);
        else m = std::make_unique<RejectionFreeVoterModel<int>>(std::make_unique<CompleteGraph<int>>(states), &rng);
        m->set_frozen(zealots);

        CHECK(!m->advance(10, true));   // zealots of both kinds: no consensus
        const auto s = m->sample_stationary(/*burn_in=*/500, /*n_samples=*/40000, /*interval=*/1);
        CHECK_NEAR(s.mean, mean_m, 4.0 * s.mean_err);
        CHECK_NEAR(s.susceptibility, chi, 4.0 * s.susceptibility_err);
    }
}

// ================================================================= q-voter

TEST_CASE(qvoter_with_q1_is_the_voter_model) {
    for (bool repetition : {true, false}) {
        check_reduces_to_voter([repetition](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<QVoterModel<int>>(
                std::make_unique<CompleteGraph<int>>(create_random_lattice(64, 0.5, rng)), &rng,
                /*q=*/1, /*p=*/0.0, Nonconformity::None, repetition, /*track=*/false);
        }, 64, repetition ? 10 : 11);
    }
}

TEST_CASE(qvoter_mean_field_stationary_state) {
    // Choose the stationary up-fraction x* = 0.8 and invert the mean-field
    // equation for p (see QVoterModel.hpp); a large complete graph must then
    // settle at |m| = 2 x* - 1 = 0.6.
    const int q = 3;
    const double x = 0.8;
    const double A = (1 - x) * std::pow(x, q) - x * std::pow(1 - x, q);          // conformity drift
    const double p_ind  = A / (A + x - 0.5);                                      // = 0.2424
    const double B = std::pow(1 - x, q + 1) - std::pow(x, q + 1);                 // anticonformity drift
    const double p_anti = A / (A - B);                                            // = 0.1905

    const struct { Nonconformity type; double p; } cases[] = {
        {Nonconformity::Independence, p_ind}, {Nonconformity::Anticonformity, p_anti}};
    for (const auto& c : cases) {
        std::mt19937 rng(12);
        QVoterModel<int> m(complete(4000, 0.9, rng), &rng, q, c.p, c.type);
        CHECK(!m.consensus_is_absorbing());
        const auto s = m.sample_stationary(/*burn_in=*/300, /*n_samples=*/2000, /*interval=*/1);
        CHECK_NEAR(s.abs_mean, 2 * x - 1, 0.02);
    }
}

TEST_CASE(qvoter_disordered_above_critical_noise) {
    // Mean-field critical points for q = 3: independence (q-1)/(q-1+2^(q-1)) = 1/3,
    // anticonformity (q-1)/(2q) = 1/3. Well above them |m| ~ 1/sqrt(N).
    for (Nonconformity type : {Nonconformity::Independence, Nonconformity::Anticonformity}) {
        std::mt19937 rng(13);
        QVoterModel<int> m(complete(4000, 0.9, rng), &rng, 3, 0.5, type);
        const auto s = m.sample_stationary(300, 1000, 1);
        CHECK(s.abs_mean < 0.06);
    }
}

TEST_CASE(qvoter_distinct_panel_uses_different_neighbours) {
    // Four fully connected nodes, q = 3, the -1 node frozen. A distinct panel
    // of a +1 node is always {+1, +1, -1}: never unanimous, so nothing can
    // ever change. A panel drawn WITH repetition can be {-1, -1, -1}, and a
    // +1 node then flips. (The -1 is node 0, the first entry of every other
    // node's neighbour list, so a sampler that repeats early indices is caught.)
    const std::vector<int> start = {-1, 1, 1, 1};
    for (bool repetition : {false, true}) {
        std::mt19937 rng(100);
        QVoterModel<int> m(std::make_unique<CompleteGraph<int>>(start), &rng,
                           3, 0.0, Nonconformity::None, repetition);
        m.set_frozen(0);
        bool changed = false;
        m.advance(2000, true, false, [&](const Model<int>& model) {
            if (model.get_graph().states() != start) changed = true;
        });
        CHECK(changed == repetition);
    }
    // Nodes with fewer than q neighbours are never influenced.
    std::mt19937 rng(14);
    std::vector<int> states = {1, -1, -1, -1, -1};   // star: hub up, leaves down
    QVoterModel<int> star(std::make_unique<AdjacencyGraph<int>>(states, star_edges(4)), &rng,
                          2, 0.0, Nonconformity::None, false);
    star.advance(200, true, false);
    for (std::size_t leaf = 1; leaf <= 4; ++leaf) CHECK(star.get_graph().get_state(leaf) == -1);
}

TEST_CASE(qvoter_validates_parameters) {
    std::mt19937 rng(15);
    CHECK_THROWS_AS(QVoterModel<int>(complete(5, 0.5, rng), &rng, 0), std::invalid_argument);
    CHECK_THROWS_AS(QVoterModel<int>(complete(5, 0.5, rng), &rng, 2, 1.5), std::invalid_argument);
    QVoterModel<int> none(complete(5, 0.5, rng), &rng, 2, 0.3, Nonconformity::None);
    CHECK(none.p() == 0.0 && none.consensus_is_absorbing());
}

// ========================================================== nonlinear voter

TEST_CASE(nonlinear_voter_with_alpha_1_is_the_voter_model) {
    check_reduces_to_voter([](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        return std::make_unique<NonlinearVoterModel<int>>(
            std::make_unique<CompleteGraph<int>>(create_random_lattice(64, 0.5, rng)), &rng, 1.0, false);
    }, 64, 20);
}

TEST_CASE(nonlinear_voter_alpha_controls_ordering) {
    const std::size_t N = 400;
    auto mean_time = [N](double alpha, unsigned seed) {
        Ensemble<int> ens([N, alpha](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<NonlinearVoterModel<int>>(complete(N, 0.5, rng), &rng, alpha);
        }, seed);
        return ens.run(200, 100000, true, 0).mean_consensus_time;
    };
    // alpha > 1 orders in ~ln N sweeps, far faster than the voter's ~N ln 2.
    CHECK(mean_time(2.0, 21) < 0.2 * mean_time(1.0, 22));

    // alpha < 1: the mixed state is stable; no consensus, |m| stays small.
    std::mt19937 rng(23);
    NonlinearVoterModel<int> m(complete(200, 0.5, rng), &rng, 0.5);
    CHECK(!m.advance(2000));
    const auto s = m.sample_stationary(0, 1000, 1);
    CHECK(s.abs_mean < 0.15);

    CHECK_THROWS_AS(NonlinearVoterModel<int>(complete(5, 0.5, rng), &rng, 0.0), std::invalid_argument);
}

// ===================================================== majority vote + noise

TEST_CASE(majority_vote_mean_field_magnetization) {
    // On a complete graph every node sees the global majority, so the
    // stationary up-fraction of the majority side is 1 - noise: |m| = 1 - 2 noise.
    for (double noise : {0.1, 0.3}) {
        std::mt19937 rng(31);
        MajorityVoteModel<int> m(complete(4000, 0.5, rng), &rng, noise);
        const auto s = m.sample_stationary(200, 1000, 1);
        CHECK_NEAR(s.abs_mean, 1.0 - 2.0 * noise, 0.02);
    }
}

TEST_CASE(majority_vote_transition_on_square_lattice) {
    // noise_c = 0.075 on the square lattice: ordered well below, disordered well above.
    std::mt19937 rng(32);
    MajorityVoteModel<int> ordered(all_up_lattice(32), &rng, 0.03);
    const auto lo = ordered.sample_stationary(200, 400, 1);
    CHECK(lo.abs_mean > 0.85);
    CHECK_NEAR(lo.binder, 2.0 / 3.0, 0.02);

    MajorityVoteModel<int> disordered(all_up_lattice(32), &rng, 0.15);
    const auto hi = disordered.sample_stationary(200, 400, 1);
    CHECK(hi.abs_mean < 0.1);
    CHECK(hi.binder < 0.3);

    CHECK(MajorityVoteModel<int>(all_up_lattice(4), &rng, 0.0).consensus_is_absorbing());
    CHECK_THROWS_AS(MajorityVoteModel<int>(all_up_lattice(4), &rng, -0.1), std::invalid_argument);
}

// ===================================================================== Ising

TEST_CASE(ising_matches_exact_enumeration_on_3x3) {
    // All 2^9 states of a periodic 3x3 lattice: exact <E>/N and <|m|>.
    const double T = 2.5;
    RegularLattice<int> geometry(std::vector<int>(9, 1), {3, 3});
    double Z = 0.0, sumE = 0.0, sumAbsM = 0.0;
    for (int config = 0; config < 512; ++config) {
        auto spin = [config](std::size_t i) { return (config >> i) & 1 ? 1 : -1; };
        double E = 0.0, M = 0.0;
        for (std::size_t i = 0; i < 9; ++i) {
            M += spin(i);
            for (std::size_t j : geometry.neighbours(i))
                if (j > i) E -= spin(i) * spin(j);
        }
        const double w = std::exp(-E / T);
        Z += w; sumE += w * E / 9.0; sumAbsM += w * std::fabs(M) / 9.0;
    }
    const double exact_e = sumE / Z, exact_abs_m = sumAbsM / Z;

    for (IsingDynamics dyn : {IsingDynamics::Metropolis, IsingDynamics::Glauber}) {
        std::mt19937 rng(41);
        IsingModel<int> m(lattice({3, 3}, 0.5, rng), &rng, T, 1.0, 0.0, dyn);
        m.advance(1000, true, false);
        std::vector<double> e;
        std::vector<double> abs_m;
        m.advance(200000, true, false, [&](const Model<int>&) {
            e.push_back(m.energy_per_site());
            abs_m.push_back(std::fabs(m.get_magnetization_per_site()));
        });
        auto mean = [](const stats::Moments& mo) { return mo.m1; };
        const double e_err = stats::jackknife_error(e, 20, mean);
        const double m_err = stats::jackknife_error(abs_m, 20, mean);
        CHECK_NEAR(stats::moments(e).m1, exact_e, 4.0 * e_err);
        CHECK_NEAR(stats::moments(abs_m).m1, exact_abs_m, 4.0 * m_err);
    }
}

TEST_CASE(ising_2d_spontaneous_magnetization_matches_onsager) {
    // T = 2.0 < T_c = 2.269: m = (1 - sinh(2/T)^-4)^(1/8) = 0.9113. Start
    // ordered (from random the system can stay stuck in striped states).
    const double T = 2.0;
    const double onsager = std::pow(1.0 - std::pow(std::sinh(2.0 / T), -4.0), 0.125);
    for (IsingDynamics dyn : {IsingDynamics::Metropolis, IsingDynamics::Glauber}) {
        std::mt19937 rng(42);
        IsingModel<int> m(all_up_lattice(32), &rng, T, 1.0, 0.0, dyn);
        const auto s = m.sample_stationary(300, 6000, 1);
        CHECK_NEAR(s.abs_mean, onsager, 4.0 * s.abs_mean_err);   // finite-size shift << error
    }
    // Above T_c the order melts.
    std::mt19937 rng(43);
    IsingModel<int> hot(all_up_lattice(32), &rng, 3.5);
    CHECK(hot.sample_stationary(300, 500, 1).abs_mean < 0.1);
}

TEST_CASE(ising_curie_weiss_on_complete_graph) {
    // J = 1/(N-1): mean field with T_c = 1 and m = tanh(m / T).
    const std::size_t N = 2000;
    const double T = 0.8;
    double m_star = 1.0;
    for (int it = 0; it < 200; ++it) m_star = std::tanh(m_star / T);   // = 0.710

    std::mt19937 rng(44);
    IsingModel<int> m(std::make_unique<CompleteGraph<int>>(std::vector<int>(N, 1)), &rng, T,
                      1.0 / static_cast<double>(N - 1));
    const auto s = m.sample_stationary(200, 1000, 1);
    CHECK_NEAR(s.abs_mean, m_star, 0.02);
}

TEST_CASE(ising_energy_and_parameters) {
    std::mt19937 rng(45);
    IsingModel<int> m(lattice({6, 6}, 0.5, rng), &rng, 2.0, 1.0, 0.3);
    m.advance(50, true, false);
    const BaseGraph<int>& g = m.get_graph();
    double E = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        E -= 0.3 * g.get_state(i);
        for (std::size_t j : g.neighbours(i))
            if (j > i) E -= g.get_state(i) * g.get_state(j);
    }
    CHECK_NEAR(m.energy(), E, 1e-9);
    CHECK(!m.consensus_is_absorbing());

    IsingModel<int> cold(all_up_lattice(5), &rng, 0.0);
    CHECK(cold.consensus_is_absorbing());
    cold.advance(10, true, false);
    CHECK(cold.get_magnetization() == 25.0);   // T = 0: no flip out of the ground state
    CHECK_THROWS_AS(IsingModel<int>(all_up_lattice(3), &rng, -1.0), std::invalid_argument);
}

// ==================================================================== Sznajd

TEST_CASE(sznajd_agreeing_pair_persuades_its_neighbours) {
    // Path a-b-c-d with states (-, +, +, -): the only agreeing pair is (b, c),
    // which converts a and d. The run must end at +1 consensus, every time.
    for (int r = 0; r < 100; ++r) {
        std::mt19937 rng(200 + r);
        SznajdModel<int> m(std::make_unique<AdjacencyGraph<int>>(std::vector<int>{-1, 1, 1, -1},
                                                                 EdgeList{{0, 1}, {1, 2}, {2, 3}}), &rng);
        CHECK(m.advance(1000));
        CHECK(m.get_magnetization() == 4.0);
    }
}

TEST_CASE(sznajd_bookkeeping_with_multi_node_updates) {
    std::mt19937 rng(51);
    SznajdModel<int> on_lattice(lattice({10, 10}, 0.5, rng), &rng);
    check_bookkeeping(on_lattice, 30);
    const EdgeList ba = barabasi_albert_edges(200, 2, rng);
    SznajdModel<int> on_network(std::make_unique<AdjacencyGraph<int>>(create_random_lattice(200, 0.5, rng), ba), &rng);
    check_bookkeeping(on_network, 30);
}

TEST_CASE(sznajd_exit_probability_is_steeper_than_voter) {
    auto exit_prob = [](double f0, unsigned seed) {
        Ensemble<int> ens([f0](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
            return std::make_unique<SznajdModel<int>>(lattice({12, 12}, f0, rng), &rng);
        }, seed);
        const auto res = ens.run(1000, 1000000, true, 0);
        CHECK(res.consensus_count == res.replicas);
        return res.exit_prob_up;
    };
    CHECK_NEAR(exit_prob(0.5, 52), 0.5, 4.0 * std::sqrt(0.25 / 1000));   // symmetry
    CHECK(exit_prob(0.35, 53) < 0.2);   // voter would give 0.35
}
