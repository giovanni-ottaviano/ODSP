#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// Parallel sweeps and parallel stationary chains (driver).

using namespace driver;
namespace fs = std::filesystem;

namespace {

struct DriverRun {
    int status = 0;
    std::string out;
};

DriverRun run(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    DriverRun r;
    r.status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    r.out = out.str();
    return r;
}

struct TempPrefix {
    std::string prefix;
    TempPrefix() {
        std::random_device rd;
        prefix = (fs::temp_directory_path() / ("odsp_par_" + std::to_string(rd()))).string();
    }
    ~TempPrefix() {
        std::error_code ec;
        const fs::path dir = fs::path(prefix).parent_path();
        const std::string stem = fs::path(prefix).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(stem, 0) == 0) fs::remove(f.path(), ec);
    }
    std::string file(const std::string& suffix) const { return read_without_comments(prefix + suffix); }
};

double row_value(const Row& row, const std::string& name) {
    for (const auto& c : row) if (c.first == name) return c.second;
    odsp_test::fail(__FILE__, __LINE__, "column '" + name + "' missing");
}

Plan<int> plan_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    return read_plan<int>(o);
}

} // namespace

// ------------------------------------------------------------ parallel_for

TEST_CASE(parallel_for_runs_every_index_once) {
    for (int threads : {1, 3, 8}) {
        std::vector<std::atomic<int>> hits(50);
        for (auto& h : hits) h = 0;
        parallel_for(50, threads, [&hits](int i) { ++hits[static_cast<std::size_t>(i)]; });
        for (const auto& h : hits) CHECK(h == 1);
    }
    parallel_for(0, 4, [](int) { odsp_test::fail(__FILE__, __LINE__, "called for an empty range"); });
}

TEST_CASE(parallel_for_uses_at_most_the_given_threads) {
    std::mutex mutex;
    std::set<std::thread::id> ids;
    parallel_for(40, 3, [&](int) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::lock_guard<std::mutex> lock(mutex);
        ids.insert(std::this_thread::get_id());
    });
    CHECK(!ids.empty() && ids.size() <= 3);
}

TEST_CASE(parallel_for_propagates_exceptions) {
    CHECK_THROWS_AS(parallel_for(20, 4, [](int i) { if (i == 7) throw std::runtime_error("boom"); }), std::runtime_error);
    CHECK_THROWS_AS(parallel_for(20, 1, [](int i) { if (i == 7) throw std::runtime_error("boom"); }), std::runtime_error);
}

// ------------------------------------------------------------ parallel sweeps

TEST_CASE(parallel_sweep_is_independent_of_threads) {
    TempPrefix a, b;
    for (auto* t : {&a, &b})
        CHECK(run({"--model", "majority-vote", "--init", "up", "--dims", "8x8", "--task", "stationary", "--samples", "300",
                   "--burn-in", "50", "--sweep", "noise=0.05:0.3:0.05", "--seed", "1", "--threads", t == &a ? "1" : "4",
                   "--csv", t->prefix}).status == 0);
    CHECK(!a.file("_sweep.csv").empty() && a.file("_sweep.csv") == b.file("_sweep.csv"));

    TempPrefix c, d;
    for (auto* t : {&c, &d})
        CHECK(run({"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas", "20", "--observables", "rho",
                   "--sweep", "n=10,20,30", "--log-times", "--seed", "2", "--threads", t == &c ? "1" : "5",
                   "--csv", t->prefix}).status == 0);
    for (const char* f : {"_sweep.csv", "_point0_replicas.csv", "_point2_replicas.csv", "_point1_timeseries.csv"})
        CHECK(!c.file(f).empty() && c.file(f) == d.file(f));
}

TEST_CASE(parallel_sweep_prints_rows_in_point_order) {
    const DriverRun r = run({"--model", "majority-vote", "--init", "up", "--dims", "8x8", "--task", "stationary",
                             "--samples", "100", "--burn-in", "10", "--sweep", "noise=0.3,0.1,0.2", "--seed", "3",
                             "--threads", "3"});
    CHECK(r.status == 0);
    const std::size_t p3 = r.out.find("\n         0.3"), p1 = r.out.find("\n         0.1"), p2 = r.out.find("\n         0.2");
    CHECK(p3 != std::string::npos && p1 != std::string::npos && p2 != std::string::npos);
    CHECK(p3 < p1 && p1 < p2);   // the order given, not completion order
}

TEST_CASE(parallel_sweep_reports_errors) {
    // The model rejects noise > 1 when the plans are validated, before any point runs.
    CHECK(run({"--model", "majority-vote", "--task", "stationary", "--sweep", "noise=0.1,1.5", "--threads", "2"}).status == 2);
}

// ------------------------------------------------------- stationary chains

TEST_CASE(stationary_chains_are_independent_of_threads) {
    Plan<int> p = plan_of({"--model", "noisy-voter", "--noise", "0.2", "--graph", "complete", "--n", "50", "--task",
                           "stationary", "--replicas", "4", "--samples", "200", "--burn-in", "50"});
    const Row one = task_stationary<int>(p.params, p.setup, 7, "", false, 1);
    const Row four = task_stationary<int>(p.params, p.setup, 7, "", false, 4);
    CHECK(one == four);
    CHECK(row_value(one, "chains") == 4.0);
}

TEST_CASE(stationary_single_chain_is_unchanged) {
    // --replicas 1 (the default) must reproduce the single-chain run exactly.
    Plan<int> p = plan_of({"--model", "noisy-voter", "--noise", "0.2", "--graph", "complete", "--n", "50", "--task",
                           "stationary", "--samples", "200", "--burn-in", "50"});
    std::mt19937 rng(8);
    auto model = p.setup.factory(rng);
    const auto direct = model->sample_stationary(p.params.burn_in, p.params.samples, p.params.interval, true, p.params.blocks);
    const Row row = task_stationary<int>(p.params, p.setup, 8, "", false, 4);
    CHECK(row_value(row, "mean") == direct.mean);
    CHECK(row_value(row, "binder") == direct.binder);
    CHECK(row_value(row, "binder_err") == direct.binder_err);
}

TEST_CASE(stationary_chains_pool_samples_correctly) {
    // Noisy voter at full noise: independent spins, so <m> = 0 and N var(m) = 1.
    Plan<int> p = plan_of({"--model", "noisy-voter", "--noise", "1", "--graph", "complete", "--n", "100", "--task",
                           "stationary", "--replicas", "6", "--samples", "2000", "--burn-in", "20"});
    const Row row = task_stationary<int>(p.params, p.setup, 9, "", false, 6);
    CHECK_NEAR(row_value(row, "mean"), 0.0, 4.0 * row_value(row, "mean_err"));
    CHECK_NEAR(row_value(row, "susceptibility"), 1.0, 4.0 * row_value(row, "susceptibility_err"));
    // Equilibrated, independent chains: the spread between chains agrees with
    // the errors within them.
    const double ratio = row_value(row, "susceptibility_err") / row_value(row, "susceptibility_err_within");
    CHECK(ratio > 0.4 && ratio < 2.5);
}

TEST_CASE(stationary_chains_specific_heat_matches_enumeration) {
    const double T = 2.5;
    RegularLattice<int> geometry(std::vector<int>(9, 1), {3, 3});
    double Z = 0.0, s1 = 0.0, s2 = 0.0;
    for (int config = 0; config < 512; ++config) {
        auto spin = [config](std::size_t i) { return (config >> i) & 1 ? 1 : -1; };
        double E = 0.0;
        for (std::size_t i = 0; i < 9; ++i)
            for (std::size_t j : geometry.neighbours(i))
                if (j > i) E -= spin(i) * spin(j);
        const double w = std::exp(-E / T), e = E / 9.0;
        Z += w; s1 += w * e; s2 += w * e * e;
    }
    const double exact = 9.0 * (s2 / Z - (s1 / Z) * (s1 / Z)) / (T * T);
    Plan<int> p = plan_of({"--model", "ising", "--temperature", "2.5", "--dims", "3x3", "--task", "stationary",
                           "--replicas", "5", "--burn-in", "500", "--samples", "20000"});
    const Row row = task_stationary<int>(p.params, p.setup, 10, "", false, 5);
    CHECK_NEAR(row_value(row, "specific_heat"), exact, 4.0 * row_value(row, "specific_heat_err"));
}

TEST_CASE(stationary_chains_warn_when_they_disagree) {
    // Ising well below T_c from random starts with no burn-in: chains order
    // into opposite magnetizations, so they disagree far beyond their own errors.
    const DriverRun r = run({"--model", "ising", "--temperature", "1.2", "--dims", "16x16", "--task", "stationary",
                             "--replicas", "6", "--burn-in", "0", "--samples", "300", "--seed", "11", "--threads", "6"});
    CHECK(r.status == 0);
    CHECK(r.out.find("chains disagree") != std::string::npos);
}

TEST_CASE(stationary_chains_samples_file_has_chain_column) {
    TempPrefix t;
    CHECK(run({"--model", "noisy-voter", "--noise", "0.5", "--graph", "complete", "--n", "30", "--task", "stationary",
               "--replicas", "3", "--samples", "50", "--burn-in", "5", "--seed", "12", "--csv", t.prefix}).status == 0);
    const std::string samples = t.file("_samples.csv");
    CHECK(samples.rfind("sample,chain,sweep,m\n", 0) == 0);
    CHECK(std::count(samples.begin(), samples.end(), '\n') == 151);   // header + 3 x 50
}
