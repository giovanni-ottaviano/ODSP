#ifndef ODSP_OPINIONCOUNTS_HPP
#define ODSP_OPINIONCOUNTS_HPP

#include <algorithm>
#include <cstddef>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

// Number of nodes holding each opinion, updated in O(1) as nodes change
// opinion. Used by Model for discrete (integer) opinions: it gives the number
// of surviving opinions, consensus detection for any number of opinions, and
// the size of the largest opinion group.
//
// Storage is a plain array indexed by opinion value (offset by the smallest
// value seen), which covers the usual labelings -- +/-1 spins, 0..q-1, or one
// distinct label per node -- in O(range) memory. The range grows as new values
// appear; if the labels are too sparse for that (range much larger than the
// number of nodes) it switches to an ordered map.
template <typename T>
class OpinionCounts {
    static_assert(std::is_integral<T>::value, "OpinionCounts: opinions must be an integer type.");

public:
    // Recount from scratch.
    void reset(const std::vector<T>& states) {
        _nodes = states.size();
        _dense.clear();
        _map.clear();
        _use_map = false;
        _distinct = 0;
        if (states.empty()) return;
        const auto [lo, hi] = std::minmax_element(states.begin(), states.end());
        if (!_fits(static_cast<long long>(*lo), static_cast<long long>(*hi))) _use_map = true;
        else { _lo = static_cast<long long>(*lo); _dense.assign(static_cast<std::size_t>(*hi - *lo) + 1, 0); }
        for (const T& s : states) _add(s);
    }

    // One node changes from opinion `from` to opinion `to`.
    void move(T from, T to) {
        _remove(from);
        _add(to);
    }

    std::size_t count(T s) const {
        if (_use_map) {
            const auto it = _map.find(s);
            return it == _map.end() ? 0 : it->second;
        }
        const long long idx = static_cast<long long>(s) - _lo;
        return (idx < 0 || idx >= static_cast<long long>(_dense.size())) ? 0 : _dense[static_cast<std::size_t>(idx)];
    }

    // Number of opinions held by at least one node.
    std::size_t distinct() const { return _distinct; }

    // Size of the largest opinion group (O(range) scan).
    std::size_t max_count() const {
        std::size_t best = 0;
        if (_use_map) for (const auto& kv : _map) best = std::max(best, kv.second);
        else          for (std::size_t c : _dense) best = std::max(best, c);
        return best;
    }

    // (opinion, count) for every opinion held, in increasing opinion order.
    std::vector<std::pair<T, std::size_t>> sorted() const {
        std::vector<std::pair<T, std::size_t>> out;
        if (_use_map) {
            for (const auto& kv : _map) out.push_back(kv);
        } else {
            for (std::size_t k = 0; k < _dense.size(); ++k)
                if (_dense[k] > 0) out.emplace_back(static_cast<T>(_lo + static_cast<long long>(k)), _dense[k]);
        }
        return out;
    }

private:
    // Dense storage is used while the value range stays within a few times
    // the number of nodes.
    bool _fits(long long lo, long long hi) const {
        const long long limit = std::max<long long>(1024, 4 * static_cast<long long>(_nodes));
        return hi - lo < limit;
    }

    void _add(T s) {
        if (_use_map) {
            if (_map[s]++ == 0) ++_distinct;
            return;
        }
        const long long v = static_cast<long long>(s);
        if (_dense.empty()) { _lo = v; _dense.assign(1, 0); }
        if (v < _lo || v >= _lo + static_cast<long long>(_dense.size())) _grow(v);
        if (_use_map) { _add(s); return; }   // _grow switched representation
        if (_dense[static_cast<std::size_t>(v - _lo)]++ == 0) ++_distinct;
    }

    void _remove(T s) {
        if (_use_map) {
            const auto it = _map.find(s);
            if (--it->second == 0) { _map.erase(it); --_distinct; }
            return;
        }
        if (--_dense[static_cast<std::size_t>(static_cast<long long>(s) - _lo)] == 0) --_distinct;
    }

    // Extend the dense range to include v, or switch to the map.
    void _grow(long long v) {
        const long long new_lo = std::min(_lo, v);
        const long long new_hi = std::max(_lo + static_cast<long long>(_dense.size()) - 1, v);
        if (!_fits(new_lo, new_hi)) {
            for (std::size_t k = 0; k < _dense.size(); ++k)
                if (_dense[k] > 0) _map[static_cast<T>(_lo + static_cast<long long>(k))] = _dense[k];
            _dense.clear();
            _use_map = true;
            return;
        }
        std::vector<std::size_t> grown(static_cast<std::size_t>(new_hi - new_lo) + 1, 0);
        std::copy(_dense.begin(), _dense.end(), grown.begin() + (_lo - new_lo));
        _dense.swap(grown);
        _lo = new_lo;
    }

    std::size_t              _nodes = 0;
    long long                _lo = 0;
    std::vector<std::size_t> _dense;      // _dense[v - _lo] = nodes holding opinion v
    bool                     _use_map = false;
    std::map<T, std::size_t> _map;
    std::size_t              _distinct = 0;
};

#endif // ODSP_OPINIONCOUNTS_HPP
