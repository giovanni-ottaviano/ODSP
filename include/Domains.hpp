#ifndef ODSP_DOMAINS_HPP
#define ODSP_DOMAINS_HPP

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <type_traits>
#include <vector>

#include "BaseGraph.hpp"
#include "RegularLattice.hpp"

// Domains of a discrete opinion state: maximal groups of nodes connected
// through edges whose two ends hold the same opinion. Every node is in exactly
// one domain (an isolated node is a domain of size 1). Found by union-find in
// O(N * degree), or O(N) on a complete graph (one domain per distinct opinion).
//
// On a periodic RegularLattice each domain also gets a WRAP COUNT: the number
// of independent directions in which it winds around the torus (the rank of the
// winding vectors of its cycles). A droplet has 0; a horizontal, vertical or
// diagonal stripe in 2D has 1; the sea around droplets, or consensus, has D. The
// union-find tracks each node's unwrapped displacement to its root (as in
// Newman & Ziff's percolation algorithm): an edge closing a cycle inside one
// domain whose displacements disagree by a multiple of the sides winds around
// the torus, and that multiple is a winding vector.

struct Domain {
    std::size_t size = 0;
    int wraps = -1;   // independent winding directions on a periodic lattice; -1 otherwise
};

struct DomainStats {
    std::vector<Domain> domains;   // sorted by decreasing size
    std::size_t nodes = 0;         // N
    int dimensions = 0;            // D of a periodic RegularLattice; 0 otherwise (wraps unknown)

    std::size_t count() const { return domains.size(); }
    // Number of domains / N.
    double density() const { return nodes == 0 ? 0.0 : static_cast<double>(count()) / static_cast<double>(nodes); }
    // Largest domain / N.
    double largest_fraction() const {
        return nodes == 0 ? 0.0 : static_cast<double>(domains.front().size) / static_cast<double>(nodes);
    }
    // Mean domain size N / count, in nodes.
    double mean_size() const { return nodes == 0 ? 0.0 : static_cast<double>(nodes) / static_cast<double>(count()); }
    // Size-weighted mean domain size sum s^2 / N, in nodes (includes the largest domain).
    double weighted_mean_size() const {
        if (nodes == 0) return 0.0;
        double sum = 0.0;
        for (const Domain& d : domains) sum += static_cast<double>(d.size) * static_cast<double>(d.size);
        return sum / static_cast<double>(nodes);
    }
    // Some domain winds around the torus in some but not all directions.
    bool stripes() const {
        for (const Domain& d : domains)
            if (d.wraps > 0 && d.wraps < dimensions) return true;
        return false;
    }
};

namespace domains_detail {

// Union-find with path halving and union by size.
struct UnionFind {
    std::vector<std::size_t> parent, size;
    explicit UnionFind(std::size_t n) : parent(n), size(n, 1) { std::iota(parent.begin(), parent.end(), 0); }
    std::size_t find(std::size_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    }
    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

inline void sort_by_size(std::vector<Domain>& d);

// Linearly independent integer vectors in echelon form (distinct pivots, kept in
// ascending pivot order); its size is the rank of everything inserted.
class Basis {
public:
    void insert(std::vector<long long> v) {
        for (const auto& r : _rows) {
            const std::size_t p = pivot(r);
            if (v[p] == 0) continue;
            const long long a = r[p], b = v[p];
            long long g = 0;
            for (std::size_t k = 0; k < v.size(); ++k) {
                v[k] = v[k] * a - r[k] * b;   // fraction-free elimination: v[p] becomes 0
                g = std::gcd(g, v[k] < 0 ? -v[k] : v[k]);
            }
            if (g > 1)
                for (long long& x : v) x /= g;
        }
        if (std::all_of(v.begin(), v.end(), [](long long x) { return x == 0; })) return;
        const std::size_t p = pivot(v);
        auto at = std::find_if(_rows.begin(), _rows.end(), [&](const auto& r) { return pivot(r) > p; });
        _rows.insert(at, std::move(v));
    }
    void merge(const Basis& other) { for (const auto& r : other._rows) insert(r); }
    int rank() const { return static_cast<int>(_rows.size()); }

private:
    static std::size_t pivot(const std::vector<long long>& v) {
        std::size_t p = 0;
        while (v[p] == 0) ++p;
        return p;
    }
    std::vector<std::vector<long long>> _rows;
};

// Union-find that also tracks each node's displacement to its root on a
// periodic lattice, and the winding vectors of each domain.
template <typename T>
DomainStats periodic_domains(const RegularLattice<T>& g) {
    const std::size_t n = g.size();
    const std::vector<int>& L = g.get_dimensions();
    const std::size_t D = L.size();
    std::vector<std::size_t> stride(D, 1);
    for (std::size_t k = D - 1; k-- > 0;) stride[k] = stride[k + 1] * static_cast<std::size_t>(L[k + 1]);

    std::vector<std::size_t> parent(n), size(n, 1), path;
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<long long> off(n * D, 0);   // displacement to parent (to the root after find)
    std::vector<Basis> basis(n);
    auto find = [&](std::size_t x) {
        path.clear();
        while (parent[x] != x) { path.push_back(x); x = parent[x]; }
        // Closest to the root first: its parent's offset is already relative to the root.
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const std::size_t p = parent[*it];
            if (p != x)
                for (std::size_t k = 0; k < D; ++k) off[*it * D + k] += off[p * D + k];
            parent[*it] = x;
        }
        return x;
    };

    std::vector<long long> e(D), w(D);
    for (std::size_t i = 0; i < n; ++i) {
        const T si = g.get_state(i);
        for (std::size_t j : g.neighbours(i)) {
            if (j <= i || g.get_state(j) != si) continue;
            // The unit step from i to j: the one axis where they differ; across
            // the boundary a difference of -(L - 1) is a +1 step (L >= 3).
            for (std::size_t k = 0; k < D; ++k) {
                const long long ci = static_cast<long long>((i / stride[k]) % static_cast<std::size_t>(L[k]));
                const long long cj = static_cast<long long>((j / stride[k]) % static_cast<std::size_t>(L[k]));
                const long long dc = cj - ci;
                e[k] = dc == 0 ? 0 : (dc == 1 || dc == -(L[k] - 1)) ? 1 : -1;
            }
            const std::size_t ri = find(i), rj = find(j);
            if (ri != rj) {   // keep d(j) = d(i) + e after the union
                std::size_t big = ri, small = rj;
                long long sign = 1;   // small = rj: off[rj] = d(i) + e - d(j)
                if (size[ri] < size[rj]) { big = rj; small = ri; sign = -1; }
                for (std::size_t k = 0; k < D; ++k)
                    off[small * D + k] = sign * (off[i * D + k] + e[k] - off[j * D + k]);
                parent[small] = big;
                size[big] += size[small];
                basis[big].merge(basis[small]);
                basis[small] = Basis();
            } else if (basis[ri].rank() < static_cast<int>(D)) {
                bool winds = false;
                for (std::size_t k = 0; k < D; ++k) {
                    w[k] = (off[i * D + k] + e[k] - off[j * D + k]) / L[k];   // an exact multiple of the side
                    winds = winds || w[k] != 0;
                }
                if (winds) basis[ri].insert(w);
            }
        }
    }

    DomainStats out;
    out.nodes = n;
    out.dimensions = static_cast<int>(D);
    for (std::size_t i = 0; i < n; ++i)
        if (parent[i] == i) out.domains.push_back({size[i], basis[i].rank()});
    sort_by_size(out.domains);
    return out;
}

inline void sort_by_size(std::vector<Domain>& d) {
    std::sort(d.begin(), d.end(), [](const Domain& a, const Domain& b) { return a.size > b.size; });
}

} // namespace domains_detail

template <typename T>
DomainStats find_domains(const BaseGraph<T>& g) {
    static_assert(std::is_integral<T>::value, "find_domains: opinions must be discrete (an integer type).");
    DomainStats out;
    const std::size_t n = g.size();
    out.nodes = n;
    if (n == 0) return out;

    if (const auto* lat = dynamic_cast<const RegularLattice<T>*>(&g))
        if (lat->get_boundary() == Boundary::Periodic) return domains_detail::periodic_domains(*lat);

    if (g.is_complete()) {   // every pair is linked: one domain per distinct opinion
        std::map<T, std::size_t> counts;
        for (const T& s : g.states()) ++counts[s];
        for (const auto& c : counts) out.domains.push_back({c.second, -1});
        domains_detail::sort_by_size(out.domains);
        return out;
    }

    domains_detail::UnionFind uf(n);
    for (std::size_t i = 0; i < n; ++i) {
        const T si = g.get_state(i);
        for (std::size_t j : g.neighbours(i))
            if (j > i && g.get_state(j) == si) uf.unite(i, j);
    }
    for (std::size_t i = 0; i < n; ++i)
        if (uf.find(i) == i) out.domains.push_back({uf.size[i], -1});
    domains_detail::sort_by_size(out.domains);
    return out;
}

#endif // ODSP_DOMAINS_HPP
