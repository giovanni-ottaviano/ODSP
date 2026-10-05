#ifndef ODSP_BASEGRAPH_HPP
#define ODSP_BASEGRAPH_HPP

#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

// Read-only view of one node's neighbours, returned by value from
// BaseGraph::neighbours(). It is either
//   * EXPLICIT: a window onto a stored adjacency list (lattices, sparse graphs), or
//   * IMPLICIT: "every node in [0, n) except `self`" (complete graph), which
//     needs no storage at all -- a complete graph would otherwise need O(N^2)
//     memory for its lists.
// Both behave like a random-access sequence: range-for, size(), operator[],
// begin()/end(). Picking a random neighbour is just nb[uniform(0, size-1)].
//
// The view is only valid while the graph is alive and its topology unchanged.
class NeighbourView {
public:
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = std::size_t;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const std::size_t*;
        using reference         = std::size_t;

        iterator(const NeighbourView* view, std::size_t k) : _view(view), _k(k) {}
        std::size_t operator*() const { return (*_view)[_k]; }
        iterator&   operator++() { ++_k; return *this; }
        iterator    operator++(int) { iterator old = *this; ++_k; return old; }
        bool operator==(const iterator& o) const { return _k == o._k; }
        bool operator!=(const iterator& o) const { return _k != o._k; }

    private:
        const NeighbourView* _view;
        std::size_t          _k;
    };

    // Explicit list: `size` node ids starting at `data`, or a whole vector.
    NeighbourView(const std::size_t* data, std::size_t size) : _data(data), _size(size), _self(0) {}
    explicit NeighbourView(const std::vector<std::size_t>& list) : NeighbourView(list.data(), list.size()) {}

    // Implicit: all nodes in [0, n_nodes) except `self`.
    static NeighbourView all_except(std::size_t n_nodes, std::size_t self) {
        return NeighbourView(nullptr, n_nodes == 0 ? 0 : n_nodes - 1, self);
    }

    std::size_t size()  const { return _size; }
    bool        empty() const { return _size == 0; }
    std::size_t operator[](std::size_t k) const {
        return _data ? _data[k] : (k < _self ? k : k + 1);
    }
    iterator begin() const { return iterator(this, 0); }
    iterator end()   const { return iterator(this, _size); }

    std::vector<std::size_t> to_vector() const { return std::vector<std::size_t>(begin(), end()); }

private:
    NeighbourView(const std::size_t* data, std::size_t size, std::size_t self)
        : _data(data), _size(size), _self(self) {}

    const std::size_t* _data;   // nullptr => implicit "all except _self"
    std::size_t        _size;
    std::size_t        _self;
};

// Abstract topology + state container that any model runs on.
//
// The interface is expressed in terms of *linear node indices* in [0, size()),
// NOT lattice coordinates: coordinates only make sense for a regular lattice,
// whereas a node index is meaningful for any topology (complete graph,
// small-world, empirical network, ...). A concrete graph is responsible for
// mapping its own structure onto this flat indexing.
//
// Besides the five required methods, two neighbourhood queries have generic
// defaults that scan neighbours(). A graph that can answer them faster (the
// complete graph does it in O(1) from global state counts) overrides them, and
// every model and observable that uses them speeds up automatically.
template <typename T>
class BaseGraph {
public:
    virtual ~BaseGraph() = default;

    // --- topology ---
    virtual std::size_t   size() const = 0;                     // number of nodes
    virtual NeighbourView neighbours(std::size_t node) const = 0;
    std::size_t degree(std::size_t node) const { return neighbours(node).size(); }
    // True if every node is linked to every other one. Lets algorithms that
    // only depend on "everyone sees everyone" use faster methods (e.g. sorting
    // opinions instead of scanning N^2 pairs).
    virtual bool is_complete() const { return false; }

    // --- state (the opinion carried by each node) ---
    virtual T    get_state(std::size_t node) const = 0;
    virtual void set_state(std::size_t node, T value) = 0;
    virtual const std::vector<T>& states() const = 0;   // flat view, e.g. for observables

    // --- neighbourhood queries (override when a faster answer exists) ---

    // Number of neighbours of `node` whose state equals `s`.
    virtual std::size_t count_neighbours_in_state(std::size_t node, T s) const {
        std::size_t c = 0;
        for (std::size_t j : neighbours(node))
            if (get_state(j) == s) ++c;
        return c;
    }

    // Histogram of the neighbours' states as (state, count) pairs with count > 0,
    // in no particular order. `out` is cleared first; pass the same vector on
    // every call to avoid reallocating. The linear search is cheap because the
    // number of distinct opinions in a neighbourhood is small.
    virtual void neighbour_state_counts(std::size_t node,
                                        std::vector<std::pair<T, std::size_t>>& out) const {
        out.clear();
        for (std::size_t j : neighbours(node)) {
            const T s = get_state(j);
            bool found = false;
            for (auto& entry : out)
                if (entry.first == s) { ++entry.second; found = true; break; }
            if (!found) out.emplace_back(s, 1);
        }
    }
};

#endif // ODSP_BASEGRAPH_HPP
