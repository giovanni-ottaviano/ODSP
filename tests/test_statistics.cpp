#include <cmath>
#include <random>
#include <vector>

#include "TestFramework.hpp"
#include "Statistics.hpp"

namespace {

std::vector<double> iid_normal(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> x(n);
    for (double& v : x) v = d(rng);
    return x;
}

// AR(1): x_t = phi x_{t-1} + eps_t with unit-variance noise. Exactly known:
// variance 1/(1-phi^2), tau_int = 1/2 + phi/(1-phi).
std::vector<double> ar1(std::size_t n, double phi, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> x(n);
    double v = d(rng) / std::sqrt(1.0 - phi * phi);   // start in the stationary state
    for (double& out : x) { out = v; v = phi * v + d(rng); }
    return x;
}

} // namespace

TEST_CASE(moments_of_a_small_series) {
    const stats::Moments mo = stats::moments(std::vector<double>{1.0, -1.0, 2.0, 0.0});
    CHECK_NEAR(mo.m1, 0.5, 1e-15);
    CHECK_NEAR(mo.abs_m1, 1.0, 1e-15);
    CHECK_NEAR(mo.m2, 1.5, 1e-15);
    CHECK_NEAR(mo.m4, 4.5, 1e-15);
}

TEST_CASE(autocorrelation_time_of_uncorrelated_series) {
    CHECK_NEAR(stats::integrated_autocorrelation_time(iid_normal(50000, 1)), 0.5, 0.05);
    CHECK(stats::integrated_autocorrelation_time(std::vector<double>(100, 3.0)) == 0.5);   // constant
    CHECK(stats::integrated_autocorrelation_time(std::vector<double>{1.0}) == 0.5);
}

TEST_CASE(autocorrelation_time_of_ar1_series) {
    const double phi = 0.8, exact = 0.5 + phi / (1.0 - phi);   // 4.5
    CHECK_NEAR(stats::integrated_autocorrelation_time(ar1(200000, phi, 2)), exact, 0.1 * exact);
}

TEST_CASE(jackknife_error_of_the_mean_uncorrelated) {
    const std::vector<double> x = iid_normal(40000, 3);
    const double err = stats::jackknife_error(x, 50, [](const stats::Moments& m) { return m.m1; });
    CHECK_NEAR(err, 1.0 / std::sqrt(40000.0), 0.3 / std::sqrt(40000.0));
}

TEST_CASE(jackknife_error_of_the_mean_includes_autocorrelation) {
    // Blocks of 4000 samples >> tau = 4.5, so the jackknife must recover the
    // inflated error sigma * sqrt(2 tau / n), ~3x the naive sigma / sqrt(n).
    const double phi = 0.8, n = 200000.0;
    const double sigma2 = 1.0 / (1.0 - phi * phi), tau = 0.5 + phi / (1.0 - phi);
    const double expected = std::sqrt(sigma2 * 2.0 * tau / n);
    const double err = stats::jackknife_error(ar1(200000, phi, 4), 50, [](const stats::Moments& m) { return m.m1; });
    CHECK_NEAR(err, expected, 0.35 * expected);
}

TEST_CASE(jackknife_matches_brute_force_leave_one_out) {
    // Nonlinear estimator (Binder-like), compared with an explicit
    // leave-one-block-out recomputation.
    const std::vector<double> x = iid_normal(103, 5);   // remainder of 3 samples is ignored
    auto estimator = [](const stats::Moments& m) { return 1.0 - m.m4 / (3.0 * m.m2 * m.m2) + m.abs_m1; };
    const int B = 10;
    const std::size_t len = x.size() / B;

    std::vector<double> theta;
    for (int b = 0; b < B; ++b) {
        std::vector<double> rest;
        for (int c = 0; c < B; ++c)
            if (c != b) rest.insert(rest.end(), x.begin() + c * len, x.begin() + (c + 1) * len);
        theta.push_back(estimator(stats::moments(rest)));
    }
    double mean = 0.0, var = 0.0;
    for (double t : theta) mean += t / B;
    for (double t : theta) var += (t - mean) * (t - mean);
    const double brute = std::sqrt(var * (B - 1) / B);

    CHECK_NEAR(stats::jackknife_error(x, B, estimator), brute, 1e-12);
}

TEST_CASE(jackknife_needs_enough_samples) {
    auto mean = [](const stats::Moments& m) { return m.m1; };
    CHECK(stats::jackknife_error(std::vector<double>(19, 1.0), 10, mean) == 0.0);
    CHECK(stats::jackknife_error(std::vector<double>(100, 1.0), 1, mean) == 0.0);
    CHECK(stats::jackknife_error(std::vector<double>(100, 1.0), 10, mean) == 0.0);   // constant
}
