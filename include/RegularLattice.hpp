#ifndef ODSP_REGULARLATTICE_HPP
#define ODSP_REGULARLATTICE_HPP

#include <vector>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <utility>

#include "BaseGraph.hpp"

// Boundary conditions of a RegularLattice.
//   Periodic : opposite faces are joined, every site has 2*D neighbours.
//   Open     : no wrap-around; sites on faces, edges and corners have fewer
//              neighbours (a 2D corner has 2). Degrees are then heterogeneous,
//              which matters e.g. for which magnetization the voter model
//              conserves (see VoterModel.hpp).
enum class Boundary { Periodic, Open };

// D-dimensional regular lattice (hypercubic, nearest neighbours).
//
// Works for any number of dimensions D >= 1: dimensions.size() sets D and each
// node has up to 2*D nearest neighbours (+/-1 along every axis), all of them
// with periodic boundaries (the default).
//
// Sites live in a flat row-major array. The lattice exposes both:
//   * a lattice-specific coordinate API (get_site/set_site by (x0,x1,...),
//     get_dimensions) which is handy for setup and visualisation, and
//   * the generic node-indexed BaseGraph API used by the models.
// The coordinate->index mapping is row-major; the neighbour structure is
// precomputed once so the model never recomputes it in the hot loop. The lists
// live in ONE flat array, node i's neighbours being a contiguous slice: with
// periodic boundaries every node has exactly 2*D, so the slice is found by
// arithmetic; with open boundaries degrees vary and an offsets array locates it.
//
// CONSTRAINT (periodic only): every side length must be >= 3. With periodic
// boundaries a side of length 1 or 2 makes the +1 and -1 shifts along that axis
// land on the same site, so that neighbour would be listed twice
// (multiplicity-2 coupling) -- which silently double-counts in
// neighbour-averaging models such as MajorityModel. Rather than let that happen
// quietly, the constructors reject such sides with std::invalid_argument. Open
// boundaries have no such problem and accept any side >= 1.
template <typename T>
class RegularLattice : public BaseGraph<T> {
public:
    RegularLattice() = default;
    RegularLattice(const std::vector<T>& lattice, const std::vector<int>& dimensions,
                   Boundary boundary = Boundary::Periodic);
    RegularLattice(std::vector<T>&& lattice, std::vector<int>&& dimensions,
                   Boundary boundary = Boundary::Periodic);

    // --- BaseGraph interface (node-indexed) ---
    std::size_t size() const override { return _lattice.size(); }
    NeighbourView neighbours(std::size_t node) const override {
        if (_offsets.empty())   // periodic: fixed 2*D neighbours per node
            return NeighbourView(_adjacency.data() + node * _coordination, _coordination);
        return NeighbourView(_adjacency.data() + _offsets[node], _offsets[node + 1] - _offsets[node]);
    }
    T    get_state(std::size_t node) const override { return _lattice[node]; }
    void set_state(std::size_t node, T value) override { _lattice[node] = value; }
    const std::vector<T>& states() const override { return _lattice; }

    // --- lattice-specific coordinate API ---
    T    get_site(const std::vector<int>& position) const { return _lattice[_linear_index(position)]; }
    void set_site(const std::vector<int>& position, T value) { _lattice[_linear_index(position)] = value; }
    const std::vector<int>& get_dimensions() const { return _dimensions; }
    Boundary                get_boundary()   const { return _boundary; }
    std::size_t linear_index(const std::vector<int>& position) const { return _linear_index(position); }

    // Smallest side length for which periodic neighbours are all distinct.
    static constexpr int MIN_PERIODIC_SIDE = 3;

private:
    void        _validate_dimensions(const std::vector<int>& dimensions) const;
    void        _build_adjacency();
    std::size_t _linear_index(const std::vector<int>& position) const;
    bool        _is_valid_position(const std::vector<int>& position) const;
    std::size_t _expected_size(const std::vector<int>& dimensions) const;

    std::vector<T>           _lattice;
    std::vector<int>         _dimensions;
    Boundary                 _boundary = Boundary::Periodic;
    std::size_t              _coordination = 0;   // 2*D (periodic)
    std::vector<std::size_t> _offsets;     // open only: node i's neighbours at [_offsets[i], _offsets[i+1])
    std::vector<std::size_t> _adjacency;
};

// ---------------------------------------------------------------------------
// Definitions (header-only)
// ---------------------------------------------------------------------------

template <typename T>
std::size_t RegularLattice<T>::_expected_size(const std::vector<int>& dimensions) const {
    const long long prod =
        std::accumulate(dimensions.begin(), dimensions.end(), 1LL, std::multiplies<long long>());
    return static_cast<std::size_t>(prod);
}

template <typename T>
void RegularLattice<T>::_validate_dimensions(const std::vector<int>& dimensions) const {
    if (dimensions.empty())
        throw std::invalid_argument("RegularLattice: dimensions must not be empty.");
    for (std::size_t k = 0; k < dimensions.size(); ++k) {
        if (dimensions[k] < 1)
            throw std::invalid_argument(
                "RegularLattice: every side length must be >= 1 (axis " + std::to_string(k) +
                " has length " + std::to_string(dimensions[k]) + ").");
        if (_boundary == Boundary::Periodic && dimensions[k] < MIN_PERIODIC_SIDE)
            throw std::invalid_argument(
                "RegularLattice: with periodic boundaries every side length must be >= " +
                std::to_string(MIN_PERIODIC_SIDE) + " (axis " + std::to_string(k) + " has length " +
                std::to_string(dimensions[k]) + "); a shorter side would produce duplicate neighbours. "
                "Open boundaries allow it.");
    }
}

template <typename T>
RegularLattice<T>::RegularLattice(const std::vector<T>& lattice, const std::vector<int>& dimensions,
                                  Boundary boundary)
    : _lattice(lattice), _dimensions(dimensions), _boundary(boundary)
{
    _validate_dimensions(_dimensions);
    if (_lattice.size() != _expected_size(_dimensions))
        throw std::invalid_argument("RegularLattice: lattice size does not match the product of dimensions.");
    _build_adjacency();
}

template <typename T>
RegularLattice<T>::RegularLattice(std::vector<T>&& lattice, std::vector<int>&& dimensions,
                                  Boundary boundary)
    : _lattice(std::move(lattice)), _dimensions(std::move(dimensions)), _boundary(boundary)
{
    _validate_dimensions(_dimensions);
    if (_lattice.size() != _expected_size(_dimensions))
        throw std::invalid_argument("RegularLattice: lattice size does not match the product of dimensions.");
    _build_adjacency();
}

template <typename T>
void RegularLattice<T>::_build_adjacency() {
    const std::size_t n = _lattice.size();
    const std::size_t d = _dimensions.size();

    // Row-major strides: stride[k] = product of dims after k.
    std::vector<std::size_t> stride(d, 1);
    for (std::size_t k = d - 1; k-- > 0; )
        stride[k] = stride[k + 1] * static_cast<std::size_t>(_dimensions[k + 1]);

    _coordination = 2 * d;
    _offsets.clear();
    if (_boundary == Boundary::Open) _offsets.assign(n + 1, 0);
    _adjacency.clear();
    _adjacency.reserve(n * 2 * d);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < d; ++k) {
            const int dim   = _dimensions[k];
            const int coord = static_cast<int>((i / stride[k]) % static_cast<std::size_t>(dim));
            for (int s : {-1, 1}) {
                int nc = coord + s;
                if (nc < 0 || nc >= dim) {
                    if (_boundary == Boundary::Open) continue;   // no neighbour beyond the face
                    nc = (nc + dim) % dim;                       // periodic wrap-around
                }
                const long long nb =
                    static_cast<long long>(i) +
                    static_cast<long long>(nc - coord) * static_cast<long long>(stride[k]);
                _adjacency.push_back(static_cast<std::size_t>(nb));
            }
        }
        if (!_offsets.empty()) _offsets[i + 1] = _adjacency.size();
    }
}

template <typename T>
std::size_t RegularLattice<T>::_linear_index(const std::vector<int>& position) const {
    if (!_is_valid_position(position))
        throw std::out_of_range("RegularLattice: position out of range.");

    std::size_t linear_index = 0;
    for (std::size_t ix = 0; ix < position.size(); ++ix) {
        std::size_t stride = 1;
        for (std::size_t j = ix + 1; j < _dimensions.size(); ++j)
            stride *= static_cast<std::size_t>(_dimensions[j]);
        linear_index += static_cast<std::size_t>(position[ix]) * stride;
    }
    return linear_index;
}

template <typename T>
bool RegularLattice<T>::_is_valid_position(const std::vector<int>& position) const {
    if (position.size() != _dimensions.size())
        return false;
    for (std::size_t i = 0; i < position.size(); ++i)
        if (position[i] < 0 || position[i] >= _dimensions[i])
            return false;
    return true;
}

#endif // ODSP_REGULARLATTICE_HPP
