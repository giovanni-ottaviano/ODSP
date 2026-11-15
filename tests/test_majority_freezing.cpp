#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "TestFramework.hpp"
#include "AdjacencyGraph.hpp"
#include "EdgeList.hpp"
#include "MajorityModel.hpp"
#include "RegularLattice.hpp"

// Majority rule: exact detection of frozen states (no unfrozen node can change)
// behind a cheap gate, and the exact time of the last opinion change.

namespace {

// Periodic 8 x 8: rows 0-4 hold +1, rows 5-7 hold -1 (every node sees 3 of its
// own kind: frozen under both tie rules).
std::vector<int> bands() {
    std::vector<int> s(64);
    for (int i = 0; i < 64; ++i) s[static_cast<std::size_t>(i)] = i / 8 < 5 ? 1 : -1;
    return s;
}
constexpr std::size_t FLIPPABLE = 2 * 8 + 3;   // row 2, column 3: inside the +1 band

std::vector<int> diagonal() {
    std::vector<int> s(64);
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) s[static_cast<std::size_t>(x * 8 + y)] = (x + y) % 8 < 4 ? 1 : -1;
    return s;
}

std::unique_ptr<MajorityModel<int>> on_lattice(std::vector<int> s, int L, TieRule tie, std::mt19937& rng) {
    return std::make_unique<MajorityModel<int>>(std::make_unique<RegularLattice<int>>(s, std::vector<int>{L, L}), &rng, tie);
}

} // namespace

TEST_CASE(majority_frozen_flat_stripes_both_rules) {
    std::mt19937 rng(1);
    for (TieRule tie : {TieRule::Random, TieRule::KeepCurrent})
        CHECK(on_lattice(bands(), 8, tie, rng)->is_frozen_state());
}

TEST_CASE(majority_not_frozen_with_flippable_node) {
    std::mt19937 rng(1);
    std::vector<int> s = bands();
    s[FLIPPABLE] = -1;
    for (TieRule tie : {TieRule::Random, TieRule::KeepCurrent})
        CHECK(!on_lattice(s, 8, tie, rng)->is_frozen_state());
}

TEST_CASE(majority_diagonal_stripes_depend_on_tie_rule) {
    std::mt19937 rng(1);
    CHECK(on_lattice(diagonal(), 8, TieRule::KeepCurrent, rng)->is_frozen_state());
    CHECK(!on_lattice(diagonal(), 8, TieRule::Random, rng)->is_frozen_state());
}

TEST_CASE(majority_plurality_tie_not_frozen) {
    // Star: centre 0 holds opinion 0, its leaves {0, 0, 1, 1, 2} are zealots, so
    // only the centre matters; it sees a 2-2 tie between 0 and 1.
    std::mt19937 rng(1);
    const EdgeList star = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}};
    for (TieRule tie : {TieRule::Random, TieRule::KeepCurrent}) {
        MajorityModel<int> m(std::make_unique<AdjacencyGraph<int>>(std::vector<int>{0, 0, 0, 1, 1, 2}, star), &rng, tie);
        m.set_frozen(std::vector<std::size_t>{1, 2, 3, 4, 5});
        CHECK(m.is_frozen_state() == (tie == TieRule::KeepCurrent));
    }
}

TEST_CASE(majority_zealots_do_not_block_freezing) {
    std::mt19937 rng(1);
    std::vector<int> s = bands();
    s[FLIPPABLE] = -1;
    for (TieRule tie : {TieRule::Random, TieRule::KeepCurrent}) {
        auto m = on_lattice(s, 8, tie, rng);
        m->set_frozen(FLIPPABLE);
        CHECK(m->is_frozen_state());
    }
}

TEST_CASE(majority_gate_waits_a_full_sweep) {
    std::mt19937 rng(2);
    auto m = on_lattice(bands(), 8, TieRule::Random, rng);
    CHECK(!m->is_absorbing_state());   // frozen, but no sweep has shown it yet
    m->advance(1, true, false);
    CHECK(m->get_changes() == 0);
    CHECK(m->is_absorbing_state());
}

TEST_CASE(majority_last_change_update_exact) {
    std::mt19937 rng(3);
    std::vector<int> s = bands();
    s[FLIPPABLE] = -1;
    auto m = on_lattice(s, 8, TieRule::Random, rng);
    CHECK(m->last_change_update() == 0);
    while (m->get_changes() == 0) m->advance(1, false, false);
    const long long u = m->get_updates();
    CHECK(m->last_change_update() == u);
    m->advance(3, true, false);
    CHECK(m->get_changes() == 1);
    CHECK(m->last_change_update() == u);
    CHECK(m->is_frozen_state() && m->is_absorbing_state());
}

TEST_CASE(majority_advance_stops_at_freezing) {
    std::mt19937 rng(4);
    std::bernoulli_distribution coin(0.5);
    std::vector<int> s(256);
    for (int& x : s) x = coin(rng) ? 1 : -1;
    auto m = on_lattice(s, 16, TieRule::KeepCurrent, rng);
    m->advance(100000, true, true);
    CHECK(m->get_mc_steps() < 100000);
    CHECK(m->is_frozen_state());
    const long long froze = static_cast<long long>(std::ceil(static_cast<double>(m->last_change_update()) / 256.0));
    CHECK(m->get_mc_steps() - froze == (m->is_consensus_reached() ? 0 : 1));
}

namespace {
// KeepCurrent on the bands with one flippable node, after a first sweep that
// happened not to pick it: the gate is open, but the state is not frozen.
std::unique_ptr<MajorityModel<int>> quiet_but_not_frozen(std::mt19937& rng) {
    std::vector<int> s = bands();
    s[FLIPPABLE] = -1;
    for (;;) {
        auto m = on_lattice(s, 8, TieRule::KeepCurrent, rng);
        m->advance(1, true, false);
        if (m->get_changes() == 0) return m;
    }
}
} // namespace

TEST_CASE(majority_failed_check_not_repeated_until_a_change) {
    std::mt19937 rng(5);
    auto m = quiet_but_not_frozen(rng);
    CHECK(m->frozen_checks() == 0);
    CHECK(!m->is_absorbing_state());
    CHECK(m->frozen_checks() == 1);
    for (int k = 0; k < 10; ++k) CHECK(!m->is_absorbing_state());
    CHECK(m->frozen_checks() == 1);   // same state: the failed check is remembered
    while (m->get_changes() == 0) m->advance(1, false, false);
    m->advance(1, true, false);   // a sweep after the change (frozen now)
    CHECK(m->is_absorbing_state());
    CHECK(m->frozen_checks() == 2);
}

TEST_CASE(majority_new_zealot_clears_the_failed_check) {
    std::mt19937 rng(6);
    auto m = quiet_but_not_frozen(rng);
    CHECK(!m->is_absorbing_state());
    m->set_frozen(FLIPPABLE);   // no opinion changes, yet the state is now frozen
    CHECK(m->is_absorbing_state());
}
