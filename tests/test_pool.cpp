#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "TestFramework.hpp"
#include "CompleteGraph.hpp"
#include "Ensemble.hpp"
#include "Parallel.hpp"
#include "VoterModel.hpp"

// Shared thread pool: nested parallel loops borrow the idle workers of the
// enclosing loop instead of creating threads of their own.

namespace {

// Keeps track of how many items run at once, and of the threads they ran on.
struct Activity {
    std::atomic<int> active{0}, peak{0};
    std::mutex mutex;
    std::set<std::thread::id> threads;

    void enter() {
        const int now = ++active;
        for (int p = peak; now > p && !peak.compare_exchange_weak(p, now);) {}
        std::lock_guard<std::mutex> lock(mutex);
        threads.insert(std::this_thread::get_id());
    }
    void leave() { --active; }

    // Enters, then waits (at most `timeout`) until `target` items are running
    // at once; true if they were. Once reached, later callers do not wait.
    bool wait_for(int target, std::chrono::milliseconds timeout) {
        enter();
        const auto end = std::chrono::steady_clock::now() + timeout;
        while (peak < target && std::chrono::steady_clock::now() < end)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        leave();
        return peak >= target;
    }
};

} // namespace

TEST_CASE(pool_nested_loop_borrows_idle_threads) {
    // Outer item 0 returns at once; item 1 runs an inner loop that asks for one
    // thread. The three idle workers must join it.
    Activity inner;
    std::atomic<bool> reached{false};
    par::parallel_for(2, 4, [&](int i) {
        if (i == 0) return;
        par::parallel_for(8, 1, [&](int) {
            if (inner.wait_for(3, std::chrono::milliseconds(300))) reached = true;
        });
    });
    CHECK(reached);
    CHECK(inner.threads.size() >= 3);
}

TEST_CASE(pool_never_exceeds_the_thread_budget) {
    Activity act;
    par::parallel_for(4, 3, [&](int) {
        par::parallel_for(10, 3, [&](int) {
            act.enter();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            act.leave();
        });
    });
    CHECK(act.peak <= 3);
    CHECK(act.threads.size() <= 3);
}

TEST_CASE(pool_runs_every_nested_index_once) {
    // Three levels; results land in storage owned by each caller, so every loop
    // must wait for its items running on other threads before it returns.
    // (CHECK is not thread-safe: results are checked after the loops.)
    for (int threads : {1, 2, 5}) {
        std::vector<std::atomic<int>> hits(4 * 6 * 7);
        for (auto& h : hits) h = 0;
        std::vector<int> complete(4, 0);
        par::parallel_for(4, threads, [&](int a) {
            std::vector<int> mid(6, 0);
            par::parallel_for(6, threads, [&](int b) {
                std::vector<int> low(7, 0);
                par::parallel_for(7, threads, [&](int c) {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));   // still running when the caller runs out
                    low[static_cast<std::size_t>(c)] = 1;
                    ++hits[static_cast<std::size_t>((a * 6 + b) * 7 + c)];
                });
                mid[static_cast<std::size_t>(b)] = static_cast<int>(std::count(low.begin(), low.end(), 1));
            });
            complete[static_cast<std::size_t>(a)] = std::count(mid.begin(), mid.end(), 7) == 6;
        });
        for (int c : complete) CHECK(c == 1);
        for (const auto& h : hits) CHECK(h == 1);
    }
}

TEST_CASE(pool_nested_exceptions_reach_their_caller) {
    std::atomic<int> caught{0}, after{0};
    par::parallel_for(3, 4, [&](int i) {
        if (i != 1) return;
        try {
            par::parallel_for(10, 4, [](int j) { if (j == 5) throw std::runtime_error("inner"); });
        } catch (const std::runtime_error&) {
            ++caught;
        }
        par::parallel_for(10, 4, [&](int) { ++after; });   // the pool still works
    });
    CHECK(caught == 1);
    CHECK(after == 10);
    // Uncaught, a nested exception propagates through the outer loop.
    CHECK_THROWS_AS(par::parallel_for(3, 4, [](int i) {
                        par::parallel_for(5, 2, [i](int j) { if (i == 2 && j == 3) throw std::logic_error("deep"); });
                    }),
                    std::logic_error);
}

TEST_CASE(pool_ensemble_inside_a_loop_borrows_idle_threads) {
    // An Ensemble run with threads = 1 inside a pool item uses the idle workers.
    Activity act;
    std::atomic<bool> reached{false};
    auto factory = [&](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        if (act.wait_for(3, std::chrono::milliseconds(300))) reached = true;
        auto g = std::make_unique<CompleteGraph<int>>(std::vector<int>{1, -1, 1, -1});
        return std::make_unique<VoterModel<int>>(std::move(g), &rng);
    };
    int replicas = 0;
    par::parallel_for(2, 4, [&](int i) {
        if (i == 0) return;
        Ensemble<int> ens(factory, 7);
        replicas = ens.run(8, 1000, true, 1).replicas;
    });
    CHECK(replicas == 8);
    CHECK(reached);
}
