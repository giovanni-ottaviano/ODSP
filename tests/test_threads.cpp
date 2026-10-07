#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>

#include "TestFramework.hpp"
#include "CompleteGraph.hpp"
#include "Ensemble.hpp"
#include "VoterModel.hpp"
#include "Utils.hpp"

// ODSP_MAX_THREADS caps what "all cores" (threads <= 0) means.

namespace {

// Sets an environment variable for the lifetime of the object, then restores it.
struct ScopedEnv {
    std::string name, old;
    bool had = false;
    ScopedEnv(const std::string& n, const char* value) : name(n) {
        if (const char* v = std::getenv(n.c_str())) { had = true; old = v; }
        if (value) setenv(n.c_str(), value, 1);
        else       unsetenv(n.c_str());
    }
    ~ScopedEnv() {
        if (had) setenv(name.c_str(), old.c_str(), 1);
        else     unsetenv(name.c_str());
    }
};

unsigned hardware() { return std::max(1u, std::thread::hardware_concurrency()); }

} // namespace

TEST_CASE(threads_resolve_respects_max_threads_variable) {
    {
        ScopedEnv env("ODSP_MAX_THREADS", nullptr);
        CHECK(resolve_threads(0) == static_cast<int>(hardware()));
        CHECK(resolve_threads(5) == 5);                              // explicit counts are honoured
    }
    {
        ScopedEnv env("ODSP_MAX_THREADS", "3");
        CHECK(resolve_threads(0) == std::min(3, static_cast<int>(hardware())));
        CHECK(resolve_threads(-1) == std::min(3, static_cast<int>(hardware())));
        CHECK(resolve_threads(5) == 5);
    }
    for (const char* bad : {"abc", "0", "-2", ""}) {               // ignored: back to all cores
        ScopedEnv env("ODSP_MAX_THREADS", bad);
        CHECK(resolve_threads(0) == static_cast<int>(hardware()));
    }
}

TEST_CASE(threads_ensemble_uses_at_most_the_cap) {
    ScopedEnv env("ODSP_MAX_THREADS", "2");
    std::mutex mutex;
    std::set<std::thread::id> ids;
    Ensemble<int> ens([&](std::mt19937& rng) -> std::unique_ptr<Model<int>> {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ids.insert(std::this_thread::get_id());
        }
        return std::make_unique<VoterModel<int>>(
            std::make_unique<CompleteGraph<int>>(create_random_lattice(50, 0.5, rng)), &rng);
    }, 1);
    ens.run(40, 100000, true, /*threads=*/0);
    CHECK(!ids.empty() && ids.size() <= 2);
}

TEST_CASE(threads_test_suite_runs_capped) {
    // test_main sets ODSP_MAX_THREADS=8 unless the environment already caps lower.
    const char* v = std::getenv("ODSP_MAX_THREADS");
    CHECK(v != nullptr);
    CHECK(std::atoi(v) >= 1 && std::atoi(v) <= 8);
}
