#ifndef ODSP_UTILS_HPP
#define ODSP_UTILS_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <cmath>
#include <utility>

// Build a flat (linearised) array of +1 / -1 spins with the requested fraction
// of +1, then shuffle. Length must equal the product of the lattice
// dimensions you intend to use.
inline std::vector<int> create_random_lattice(std::size_t length,
                                              double up_fraction,
                                              std::mt19937& rng) {
    if (up_fraction < 0.0 || up_fraction > 1.0)
        throw std::invalid_argument(
            "create_random_lattice: up_fraction must be in [0, 1].");

    std::vector<int> lattice(length, -1);
    const std::size_t n_up = static_cast<std::size_t>(up_fraction * length);
    for (std::size_t i = 0; i < n_up; ++i)
        lattice[i] = 1;

    std::shuffle(lattice.begin(), lattice.end(), rng);
    return lattice;
}

// Opinions 0..q-1 in prescribed proportions: fractions[k] of the nodes hold
// opinion k (fractions must be >= 0 and sum to 1 within 1e-9). Counts are
// floor(fraction * length), and the few nodes left over go to the opinions with
// the largest remainders, so the counts are exact and sum to length. Shuffled.
inline std::vector<int> create_opinions_with_fractions(std::size_t length,
                                                       const std::vector<double>& fractions,
                                                       std::mt19937& rng) {
    if (fractions.empty())
        throw std::invalid_argument("create_opinions_with_fractions: no fractions given.");
    double total = 0.0;
    for (double f : fractions) {
        if (f < 0.0) throw std::invalid_argument("create_opinions_with_fractions: fractions must be >= 0.");
        total += f;
    }
    if (std::fabs(total - 1.0) > 1e-9)
        throw std::invalid_argument("create_opinions_with_fractions: fractions must sum to 1.");

    const std::size_t q = fractions.size();
    std::vector<std::size_t> count(q);
    std::vector<std::pair<double, std::size_t>> remainder(q);   // (remainder, opinion)
    std::size_t assigned = 0;
    for (std::size_t k = 0; k < q; ++k) {
        const double exact = fractions[k] * static_cast<double>(length);
        count[k] = static_cast<std::size_t>(exact);
        remainder[k] = {exact - static_cast<double>(count[k]), k};
        assigned += count[k];
    }
    std::sort(remainder.begin(), remainder.end(),
              [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (std::size_t r = 0; assigned < length; ++r, ++assigned) ++count[remainder[r % q].second];

    std::vector<int> opinions;
    opinions.reserve(length);
    for (std::size_t k = 0; k < q; ++k) opinions.insert(opinions.end(), count[k], static_cast<int>(k));
    std::shuffle(opinions.begin(), opinions.end(), rng);
    return opinions;
}

// Opinions 0..q-1, as evenly split as possible (counts differ by at most 1).
inline std::vector<int> create_balanced_opinions(std::size_t length, int q, std::mt19937& rng) {
    if (q < 1) throw std::invalid_argument("create_balanced_opinions: q must be >= 1.");
    std::vector<int> opinions(length);
    for (std::size_t i = 0; i < length; ++i) opinions[i] = static_cast<int>(i % static_cast<std::size_t>(q));
    std::shuffle(opinions.begin(), opinions.end(), rng);
    return opinions;
}

// Every node holds its own opinion: node i holds opinion i. The usual start
// for studying how the number of opinions decays.
inline std::vector<int> create_distinct_opinions(std::size_t length) {
    std::vector<int> opinions(length);
    for (std::size_t i = 0; i < length; ++i) opinions[i] = static_cast<int>(i);
    return opinions;
}

// Continuous opinions drawn independently and uniformly from [lo, hi): the
// standard start for bounded-confidence models.
inline std::vector<double> create_uniform_opinions(std::size_t length, std::mt19937& rng,
                                                   double lo = 0.0, double hi = 1.0) {
    if (!(hi > lo)) throw std::invalid_argument("create_uniform_opinions: need hi > lo.");
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<double> opinions(length);
    for (double& x : opinions) x = d(rng);
    return opinions;
}

#endif // ODSP_UTILS_HPP
