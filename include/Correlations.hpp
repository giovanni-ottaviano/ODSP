#ifndef ODSP_CORRELATIONS_HPP
#define ODSP_CORRELATIONS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <type_traits>
#include <vector>

#include "RegularLattice.hpp"

// Spatial and temporal correlations of opinion configurations on lattices.
//
//   spatial_correlation(lattice)      G(r), r = 0..L/2, along the lattice axes
//   coarsening_length(G)              where G first drops to 1/2: the domain size L(t)
//   structure_factor(lattice, labels) S(0) and S(k_min), for the second-moment
//   second_moment_length(S0, Smin, L) correlation length xi at critical points
//   autocorrelation(now, then)        two-time C(t, t_w)
//   overlap(now, then)                fraction of nodes with the same opinion
//
// One definition of G(r) covers all opinion types:
//   * discrete opinions: G(r) = (P_same(r) - sum_k p_k^2) / (1 - sum_k p_k^2),
//     with P_same(r) the probability that two nodes r apart agree and p_k the
//     opinion shares. For +/-1 spins this is exactly the connected correlation
//     (<s_i s_{i+r}> - m^2) / (1 - m^2); it is unchanged by relabelling opinions;
//   * continuous opinions: the Pearson correlation of the deviations from the
//     mean opinion.
// G(0) = 1, and G = 1 at consensus (perfect correlation). Distances run along
// each axis (periodic wrap-around, or only existing pairs for open boundaries)
// and are averaged over the axes, up to half the shortest side. The cost is
// O(N D L/2) per call: fine up to ~512^2 sites.
namespace corr {

namespace detail {

// Row-major strides of a lattice.
inline std::vector<std::size_t> strides(const std::vector<int>& dims) {
    std::vector<std::size_t> s(dims.size(), 1);
    for (std::size_t k = dims.size(); k-- > 1;) s[k - 1] = s[k] * static_cast<std::size_t>(dims[k]);
    return s;
}

// Calls pair(i, j) for every pair of sites at distance r along `axis`.
template <typename F>
void for_pairs(const std::vector<int>& dims, bool periodic, std::size_t axis, int r, F pair) {
    const std::vector<std::size_t> st = strides(dims);
    std::size_t n = 1;
    for (int d : dims) n *= static_cast<std::size_t>(d);
    const int L = dims[axis];
    for (std::size_t i = 0; i < n; ++i) {
        const int c = static_cast<int>((i / st[axis]) % static_cast<std::size_t>(L));
        int c2 = c + r;
        if (c2 >= L) {
            if (!periodic) continue;
            c2 -= L;
        }
        const std::size_t j = i + static_cast<std::size_t>(c2) * st[axis] - static_cast<std::size_t>(c) * st[axis];
        pair(i, j);
    }
}

} // namespace detail

inline int max_distance(const std::vector<int>& dims) {
    return *std::min_element(dims.begin(), dims.end()) / 2;
}

template <typename T>
std::vector<double> spatial_correlation(const RegularLattice<T>& lattice) {
    const std::vector<T>& x = lattice.states();
    const std::vector<int>& dims = lattice.get_dimensions();
    const bool periodic = lattice.get_boundary() == Boundary::Periodic;
    const int r_max = max_distance(dims);
    std::vector<double> G(static_cast<std::size_t>(r_max) + 1, 1.0);
    const double n = static_cast<double>(x.size());

    if constexpr (std::is_integral<T>::value) {
        std::map<T, double> count;
        for (const T& v : x) count[v] += 1.0;
        double p2 = 0.0;
        for (const auto& kv : count) p2 += (kv.second / n) * (kv.second / n);
        if (1.0 - p2 <= 1e-15) return G;   // consensus
        for (int r = 1; r <= r_max; ++r) {
            double same = 0.0, pairs = 0.0;
            for (std::size_t axis = 0; axis < dims.size(); ++axis)
                detail::for_pairs(dims, periodic, axis, r, [&](std::size_t i, std::size_t j) {
                    same += x[i] == x[j];
                    pairs += 1.0;
                });
            G[static_cast<std::size_t>(r)] = (same / pairs - p2) / (1.0 - p2);
        }
    } else {
        double mean = 0.0, var = 0.0;
        for (const T& v : x) mean += static_cast<double>(v);
        mean /= n;
        for (const T& v : x) var += (static_cast<double>(v) - mean) * (static_cast<double>(v) - mean);
        var /= n;
        if (var <= 0.0) return G;   // all opinions equal
        for (int r = 1; r <= r_max; ++r) {
            double cov = 0.0, pairs = 0.0;
            for (std::size_t axis = 0; axis < dims.size(); ++axis)
                detail::for_pairs(dims, periodic, axis, r, [&](std::size_t i, std::size_t j) {
                    cov += (static_cast<double>(x[i]) - mean) * (static_cast<double>(x[j]) - mean);
                    pairs += 1.0;
                });
            G[static_cast<std::size_t>(r)] = cov / pairs / var;
        }
    }
    return G;
}

// The distance at which G first drops to 1/2, interpolated linearly between
// integer distances; proportional to the typical domain size during
// coarsening. If G never drops to 1/2, the largest distance available.
inline double coarsening_length(const std::vector<double>& G) {
    for (std::size_t r = 1; r < G.size(); ++r)
        if (G[r] <= 0.5) {
            const double a = G[r - 1], b = G[r];
            return static_cast<double>(r - 1) + (a - 0.5) / (a - b);
        }
    return G.empty() ? 0.0 : static_cast<double>(G.size() - 1);
}

// Structure factor of a discrete configuration at k = 0 and at the smallest
// wavevector k_min = 2 pi / L (averaged over the axes), with the opinion
// fields phi_a(i) = delta(s_i, a) - 1/q over the q labels of the model:
//     S(k) = (1/N) sum_a |sum_i phi_a(i) exp(i k x_i)|^2.
// For +/-1 spins this is half the usual spin structure factor (S(0) = N m^2 / 2),
// a factor that cancels in the correlation length. Periodic hypercubic
// lattices only (k_min must be a lattice wavevector).
struct StructureFactor {
    double s0 = 0.0, s_min = 0.0;
};

template <typename T>
StructureFactor structure_factor(const RegularLattice<T>& lattice, const std::vector<T>& labels) {
    const std::vector<T>& x = lattice.states();
    const std::vector<int>& dims = lattice.get_dimensions();
    const std::vector<std::size_t> st = detail::strides(dims);
    const double n = static_cast<double>(x.size()), q = static_cast<double>(labels.size());
    StructureFactor out;
    for (const T& a : labels) {
        double count = 0.0;
        for (const T& v : x) count += v == a;
        out.s0 += (count - n / q) * (count - n / q) / n;
    }
    const double pi = std::acos(-1.0);
    for (std::size_t axis = 0; axis < dims.size(); ++axis) {
        const double k = 2.0 * pi / dims[axis];
        for (const T& a : labels) {
            double re = 0.0, im = 0.0;   // the -1/q part sums to zero over a full period
            for (std::size_t i = 0; i < x.size(); ++i) {
                if (x[i] != a) continue;
                const double c = static_cast<double>((i / st[axis]) % static_cast<std::size_t>(dims[axis]));
                re += std::cos(k * c);
                im += std::sin(k * c);
            }
            out.s_min += (re * re + im * im) / n / static_cast<double>(dims.size());
        }
    }
    return out;
}

// Second-moment correlation length xi = sqrt(S0/Smin - 1) / (2 sin(pi/L)),
// from AVERAGED structure factors <S(0)>, <S(k_min)> (not from single
// snapshots). NaN when S0 <= Smin.
inline double second_moment_length(double s0, double s_min, int L) {
    if (!(s0 > s_min) || !(s_min > 0.0)) return std::nan("");
    return std::sqrt(s0 / s_min - 1.0) / (2.0 * std::sin(std::acos(-1.0) / L));
}

// Two-time autocorrelation between two snapshots of the same nodes:
//   +/-1 spins: C = (1/N) sum_i s_i(t) s_i(t_w)  (1 = unchanged, 0 = uncorrelated);
//   continuous opinions: the Pearson correlation of the two snapshots.
template <typename T>
double autocorrelation(const std::vector<T>& now, const std::vector<T>& then) {
    const double n = static_cast<double>(now.size());
    if constexpr (std::is_integral<T>::value) {
        double sum = 0.0;
        for (std::size_t i = 0; i < now.size(); ++i) sum += static_cast<double>(now[i]) * static_cast<double>(then[i]);
        return sum / n;
    } else {
        double ma = 0.0, mb = 0.0;
        for (std::size_t i = 0; i < now.size(); ++i) { ma += now[i]; mb += then[i]; }
        ma /= n; mb /= n;
        double cab = 0.0, va = 0.0, vb = 0.0;
        for (std::size_t i = 0; i < now.size(); ++i) {
            cab += (now[i] - ma) * (then[i] - mb);
            va += (now[i] - ma) * (now[i] - ma);
            vb += (then[i] - mb) * (then[i] - mb);
        }
        return (va > 0.0 && vb > 0.0) ? cab / std::sqrt(va * vb) : 1.0;
    }
}

// Fraction of nodes holding the same opinion in both snapshots.
template <typename T>
double overlap(const std::vector<T>& now, const std::vector<T>& then) {
    double same = 0.0;
    for (std::size_t i = 0; i < now.size(); ++i) same += now[i] == then[i];
    return now.empty() ? 1.0 : same / static_cast<double>(now.size());
}

} // namespace corr

#endif // ODSP_CORRELATIONS_HPP
