#ifndef ODSP_STATISTICS_HPP
#define ODSP_STATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Error analysis for correlated Monte Carlo time series.
//
// Successive samples of a Markov chain are correlated, so the naive standard
// error sigma / sqrt(n) underestimates the true error by a factor
// sqrt(2 tau_int). Two standard tools handle this:
//
//   * integrated_autocorrelation_time(): estimates tau_int directly, which
//     tells you how many samples are effectively independent
//     (n_eff = n / (2 tau_int)) and whether burn-in/sampling are long enough;
//   * jackknife_error(): block jackknife. Split the series into B contiguous
//     blocks, recompute the estimator leaving out one block at a time, and take
//     the spread. It gives errors for NONLINEAR estimators (susceptibility,
//     Binder cumulant) and is valid when each block is much longer than
//     tau_int -- check that with the autocorrelation time.
namespace stats {

// Moments of a series of order-parameter samples, the input of the usual
// estimators (susceptibility, Binder cumulant).
struct Moments {
    double m1     = 0.0;   // <m>
    double abs_m1 = 0.0;   // <|m|>
    double m2     = 0.0;   // <m^2>
    double m4     = 0.0;   // <m^4>
};

inline Moments moments(const double* x, std::size_t n) {
    Moments mo;
    if (n == 0) return mo;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = x[i], v2 = v * v;
        mo.m1 += v; mo.abs_m1 += std::fabs(v); mo.m2 += v2; mo.m4 += v2 * v2;
    }
    const double inv = 1.0 / static_cast<double>(n);
    mo.m1 *= inv; mo.abs_m1 *= inv; mo.m2 *= inv; mo.m4 *= inv;
    return mo;
}

inline Moments moments(const std::vector<double>& x) { return moments(x.data(), x.size()); }

// Block-jackknife standard error of estimator(Moments). Uses `blocks`
// contiguous blocks of equal length (a remainder of fewer than `blocks` samples
// at the end is ignored). Returns 0 if there are fewer than 2 blocks' worth of
// samples.
template <typename Estimator>
double jackknife_error(const std::vector<double>& x, int blocks, Estimator estimator) {
    if (blocks < 2 || x.size() < 2 * static_cast<std::size_t>(blocks)) return 0.0;
    const std::size_t B = static_cast<std::size_t>(blocks);
    const std::size_t len = x.size() / B;
    const std::size_t used = len * B;

    // Per-block sums of m, |m|, m^2, m^4, so each leave-one-out mean is O(1).
    std::vector<Moments> block_sums(B);
    Moments total;
    for (std::size_t b = 0; b < B; ++b) {
        Moments mo = moments(x.data() + b * len, len);
        mo.m1 *= len; mo.abs_m1 *= len; mo.m2 *= len; mo.m4 *= len;   // means -> sums
        block_sums[b] = mo;
        total.m1 += mo.m1; total.abs_m1 += mo.abs_m1; total.m2 += mo.m2; total.m4 += mo.m4;
    }

    std::vector<double> theta(B);
    const double inv = 1.0 / static_cast<double>(used - len);
    for (std::size_t b = 0; b < B; ++b) {
        Moments loo;
        loo.m1     = (total.m1     - block_sums[b].m1)     * inv;
        loo.abs_m1 = (total.abs_m1 - block_sums[b].abs_m1) * inv;
        loo.m2     = (total.m2     - block_sums[b].m2)     * inv;
        loo.m4     = (total.m4     - block_sums[b].m4)     * inv;
        theta[b] = estimator(loo);
    }

    double mean = 0.0;
    for (double t : theta) mean += t;
    mean /= static_cast<double>(B);
    double var = 0.0;
    for (double t : theta) var += (t - mean) * (t - mean);
    return std::sqrt(var * static_cast<double>(B - 1) / static_cast<double>(B));
}

// Integrated autocorrelation time, in units of the sample spacing, with the
// convention tau_int = 1/2 + sum_{t>=1} rho(t): an uncorrelated series gives
// 0.5 and the error of the mean is sqrt(2 tau_int / n) * sigma. The sum is
// cut at the smallest window W >= c * tau_int(W) (Sokal's automatic
// windowing; c ~ 5-10 trades bias against noise). A constant series returns 0.5.
inline double integrated_autocorrelation_time(const std::vector<double>& x, double c = 6.0) {
    const std::size_t n = x.size();
    if (n < 2) return 0.5;

    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(n);

    auto autocov = [&](std::size_t t) {
        double s = 0.0;
        for (std::size_t i = 0; i + t < n; ++i) s += (x[i] - mean) * (x[i + t] - mean);
        return s / static_cast<double>(n - t);
    };

    const double c0 = autocov(0);
    if (c0 <= 0.0) return 0.5;

    double tau = 0.5;
    for (std::size_t t = 1; t < n / 2; ++t) {
        tau += autocov(t) / c0;
        if (static_cast<double>(t) >= c * tau) break;
    }
    return std::max(tau, 0.5);
}

// Sarle's bimodality coefficient BC = (G1^2 + 1) / (G2 + 3 (n-1)^2 / ((n-2)(n-3))),
// with G1, G2 the bias-corrected sample skewness and excess kurtosis. Uniform
// data give 5/9; larger values suggest a bimodal distribution, a normal one
// gives 1/3. NaN for fewer than 4 values or zero variance.
inline double bimodality_coefficient(const std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    if (x.size() < 4) return std::nan("");
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= n;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double v : x) {
        const double d = v - mean, d2 = d * d;
        m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m2 /= n; m3 /= n; m4 /= n;
    if (m2 <= 0.0) return std::nan("");
    const double g = m3 / std::pow(m2, 1.5), k = m4 / (m2 * m2) - 3.0;
    const double G1 = g * std::sqrt(n * (n - 1.0)) / (n - 2.0);
    const double G2 = ((n + 1.0) * k + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
    return (G1 * G1 + 1.0) / (G2 + 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0)));
}

// Counts of x in `bins` equal bins over [lo, hi]; the upper edge belongs to the
// last bin, and values outside the range are clamped to the end bins. With
// hi <= lo everything lands in the first bin.
inline std::vector<std::size_t> histogram(const std::vector<double>& x, int bins, double lo, double hi) {
    std::vector<std::size_t> counts(static_cast<std::size_t>(std::max(bins, 1)), 0);
    const double width = (hi - lo) / static_cast<double>(counts.size());
    for (double v : x) {
        long long idx = width > 0.0 ? static_cast<long long>(std::floor((v - lo) / width)) : 0;
        idx = std::max(0LL, std::min(idx, static_cast<long long>(counts.size()) - 1));
        ++counts[static_cast<std::size_t>(idx)];
    }
    return counts;
}

// Block-jackknife error of f(<x>, <y>), a function of the means of two series
// sampled together (e.g. a ratio of averages, as in the second-moment
// correlation length). Same blocks as jackknife_error; 0 if too few samples.
template <typename F>
double jackknife_error2(const std::vector<double>& x, const std::vector<double>& y, int blocks, F f) {
    if (blocks < 2 || x.size() != y.size() || x.size() < 2 * static_cast<std::size_t>(blocks)) return 0.0;
    const std::size_t B = static_cast<std::size_t>(blocks), len = x.size() / B, used = len * B;
    std::vector<double> bx(B, 0.0), by(B, 0.0);
    double tx = 0.0, ty = 0.0;
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t i = b * len; i < (b + 1) * len; ++i) { bx[b] += x[i]; by[b] += y[i]; }
        tx += bx[b]; ty += by[b];
    }
    const double inv = 1.0 / static_cast<double>(used - len);
    std::vector<double> theta(B);
    for (std::size_t b = 0; b < B; ++b) theta[b] = f((tx - bx[b]) * inv, (ty - by[b]) * inv);
    double mean = 0.0;
    for (double t : theta) mean += t;
    mean /= static_cast<double>(B);
    double var = 0.0;
    for (double t : theta) var += (t - mean) * (t - mean);
    return std::sqrt(var * static_cast<double>(B - 1) / static_cast<double>(B));
}

} // namespace stats

#endif // ODSP_STATISTICS_HPP
