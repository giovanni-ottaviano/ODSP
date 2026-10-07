#include <cmath>
#include <random>
#include <vector>

#include "TestFramework.hpp"
#include "Correlations.hpp"
#include "RegularLattice.hpp"
#include "Statistics.hpp"
#include "Utils.hpp"

// Spatial correlation, coarsening length, structure factor / second-moment
// correlation length, and two-time autocorrelation (library level).

namespace {

const double kPi = std::acos(-1.0);

std::vector<int> repeat(const std::vector<int>& pattern, std::size_t times) {
    std::vector<int> v;
    for (std::size_t t = 0; t < times; ++t) v.insert(v.end(), pattern.begin(), pattern.end());
    return v;
}

} // namespace

// ------------------------------------------------------- spatial correlation

TEST_CASE(correlation_of_a_period_four_ring) {
    // ++--++--...: G(1) = 0, G(2) = -1, G(4) = 1 exactly (m = 0).
    RegularLattice<int> ring(repeat({1, 1, -1, -1}, 4), {16});
    const std::vector<double> G = corr::spatial_correlation(ring);
    CHECK(G.size() == 9);   // r = 0..L/2
    const double expected[] = {1, 0, -1, 0, 1, 0, -1, 0, 1};
    for (std::size_t r = 0; r < G.size(); ++r) CHECK_NEAR(G[r], expected[r], 1e-12);
}

TEST_CASE(correlation_at_consensus_is_one) {
    RegularLattice<int> lat(std::vector<int>(36, 1), {6, 6});
    for (double g : corr::spatial_correlation(lat)) CHECK(g == 1.0);
}

TEST_CASE(correlation_of_independent_spins_vanishes) {
    std::mt19937 rng(1);
    RegularLattice<int> lat(create_random_lattice(64 * 64, 0.5, rng), {64, 64});
    const std::vector<double> G = corr::spatial_correlation(lat);
    CHECK(G[0] == 1.0);
    for (std::size_t r = 1; r < G.size(); ++r) CHECK_NEAR(G[r], 0.0, 0.05);   // ~1/sqrt(2N) noise
}

TEST_CASE(correlation_is_label_independent_and_handles_q_opinions) {
    std::mt19937 rng(2);
    std::vector<int> spins = create_random_lattice(20 * 20, 0.3, rng);
    std::vector<int> labels(spins.size());
    for (std::size_t i = 0; i < spins.size(); ++i) labels[i] = spins[i] > 0 ? 7 : 3;   // relabel +1 -> 7, -1 -> 3
    const auto a = corr::spatial_correlation(RegularLattice<int>(spins, {20, 20}));
    const auto b = corr::spatial_correlation(RegularLattice<int>(labels, {20, 20}));
    for (std::size_t r = 0; r < a.size(); ++r) CHECK_NEAR(a[r], b[r], 1e-12);

    // Three opinions on a ring, against a brute-force evaluation of
    // (P_same(r) - sum p^2) / (1 - sum p^2).
    const std::vector<int> x = {0, 0, 1, 2, 2, 2, 1, 0, 1, 1, 2, 0};
    const auto G = corr::spatial_correlation(RegularLattice<int>(x, {12}));
    double p2 = 0.0;
    for (int a2 : {0, 1, 2}) {
        const double p = std::count(x.begin(), x.end(), a2) / 12.0;
        p2 += p * p;
    }
    for (std::size_t r = 1; r <= 6; ++r) {
        int same = 0;
        for (std::size_t i = 0; i < 12; ++i) same += x[i] == x[(i + r) % 12];
        CHECK_NEAR(G[r], (same / 12.0 - p2) / (1.0 - p2), 1e-12);
    }
}

TEST_CASE(correlation_of_continuous_opinions_is_pearson) {
    // x_i = cos(2 pi i / L): mean 0, variance 1/2, covariance at r = cos(2 pi r / L) / 2.
    const int L = 24;
    std::vector<double> x(L);
    for (int i = 0; i < L; ++i) x[static_cast<std::size_t>(i)] = std::cos(2.0 * kPi * i / L);
    const auto G = corr::spatial_correlation(RegularLattice<double>(x, {L}));
    for (int r = 0; r <= L / 2; ++r) CHECK_NEAR(G[static_cast<std::size_t>(r)], std::cos(2.0 * kPi * r / L), 1e-12);
}

TEST_CASE(correlation_with_open_boundaries_uses_existing_pairs) {
    // ++++---- on an open chain: at r = 1 there are 7 pairs, 6 agreeing:
    // G(1) = (6/7 - 1/2) / (1/2) = 5/7.
    RegularLattice<int> chain(std::vector<int>{1, 1, 1, 1, -1, -1, -1, -1}, {8}, Boundary::Open);
    CHECK_NEAR(corr::spatial_correlation(chain)[1], 5.0 / 7.0, 1e-12);
}

TEST_CASE(coarsening_length_interpolates_the_half_crossing) {
    CHECK_NEAR(corr::coarsening_length({1.0, 0.8, 0.4, 0.1}), 1.75, 1e-12);
    CHECK_NEAR(corr::coarsening_length({1.0, 0.25, 0.1}), 2.0 / 3.0, 1e-12);
    CHECK(corr::coarsening_length({1.0, 0.9, 0.7}) == 2.0);   // never below 1/2: the largest distance
}

// ------------------------------------------------------- structure factor

TEST_CASE(structure_factor_of_simple_configurations) {
    // All up on a ring of 8, labels {-1, +1}: phi_a = delta - 1/2, so
    // S(0) = ((8 - 4)^2 + (0 - 4)^2) / 8 = 4 = N m^2 / 2; S(k_min) = 0.
    const auto all_up = corr::structure_factor(RegularLattice<int>(std::vector<int>(8, 1), {8}), std::vector<int>{-1, 1});
    CHECK_NEAR(all_up.s0, 4.0, 1e-12);
    CHECK_NEAR(all_up.s_min, 0.0, 1e-12);
    // One cosine wave of spins (++++----): S(0) = 0 and S(k_min) > 0.
    const auto wave = corr::structure_factor(RegularLattice<int>(std::vector<int>{1, 1, 1, 1, -1, -1, -1, -1}, {8}),
                                             std::vector<int>{-1, 1});
    CHECK_NEAR(wave.s0, 0.0, 1e-12);
    CHECK(wave.s_min > 1.0);
}

TEST_CASE(second_moment_length_formula) {
    // xi = sqrt(S0/Smin - 1) / (2 sin(pi/L)).
    CHECK_NEAR(corr::second_moment_length(10.0, 2.0, 32), 2.0 / (2.0 * std::sin(kPi / 32)), 1e-12);
    CHECK(std::isnan(corr::second_moment_length(1.0, 2.0, 32)));   // S0 < Smin: undefined
}

TEST_CASE(jackknife_of_a_function_of_two_means) {
    // Brute-force leave-one-block-out for f(<x>, <y>) = <x>/<y>.
    std::mt19937 rng(3);
    std::normal_distribution<double> g(5.0, 1.0);
    std::vector<double> x(100), y(100);
    for (std::size_t i = 0; i < 100; ++i) { x[i] = g(rng); y[i] = g(rng) + 3.0; }
    auto f = [](double mx, double my) { return mx / my; };
    const int B = 10;
    std::vector<double> theta;
    for (int b = 0; b < B; ++b) {
        double sx = 0, sy = 0;
        for (int i = 0; i < 100; ++i) if (i / 10 != b) { sx += x[static_cast<std::size_t>(i)]; sy += y[static_cast<std::size_t>(i)]; }
        theta.push_back(f(sx / 90, sy / 90));
    }
    double mean = 0, var = 0;
    for (double t : theta) mean += t / B;
    for (double t : theta) var += (t - mean) * (t - mean);
    CHECK_NEAR(stats::jackknife_error2(x, y, B, f), std::sqrt(var * (B - 1) / B), 1e-12);
}

// ---------------------------------------------------- two-time autocorrelation

TEST_CASE(two_time_autocorrelation_and_overlap) {
    const std::vector<int> a = {1, -1, 1, 1, -1, -1};
    std::vector<int> flipped(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) flipped[i] = -a[i];
    CHECK(corr::autocorrelation(a, a) == 1.0);
    CHECK(corr::autocorrelation(a, flipped) == -1.0);
    CHECK_NEAR(corr::autocorrelation(a, std::vector<int>{1, 1, 1, 1, 1, 1}), 0.0, 1e-15);
    CHECK(corr::overlap(a, a) == 1.0);
    CHECK_NEAR(corr::overlap(std::vector<int>{0, 1, 2, 2}, std::vector<int>{0, 2, 2, 1}), 0.5, 1e-15);
    // Continuous: Pearson correlation of the two snapshots.
    const std::vector<double> x = {0.1, 0.5, 0.9, 0.3}, y = {0.3, 1.1, 1.9, 0.7};   // y = 2x + 0.1
    CHECK_NEAR(corr::autocorrelation(x, y), 1.0, 1e-12);
}
