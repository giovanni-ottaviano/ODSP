#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// Replica-averaged time series and log-spaced recording (driver).

using namespace driver;
namespace fs = std::filesystem;

namespace {

int run_quiet(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    return status;
}

struct TempPrefix {
    std::string prefix;
    TempPrefix() {
        std::random_device rd;
        prefix = (fs::temp_directory_path() / ("odsp_ts_" + std::to_string(rd()))).string();
    }
    ~TempPrefix() {
        std::error_code ec;
        const fs::path dir = fs::path(prefix).parent_path();
        const std::string stem = fs::path(prefix).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(stem, 0) == 0) fs::remove(f.path(), ec);
    }
};

// A CSV as named columns of numbers.
std::map<std::string, std::vector<double>> read_csv(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    std::string cell;
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    std::map<std::string, std::vector<double>> cols;
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (std::size_t k = 0; std::getline(ls, cell, ','); ++k) cols[names[k]].push_back(std::stod(cell));
    }
    return cols;
}

} // namespace

TEST_CASE(timeseries_recording_grids) {
    const std::vector<long long> log = recording_times(1000, 0, 10);
    CHECK(log.front() == 0 && log.back() == 1000);
    for (std::size_t i = 1; i < log.size(); ++i) CHECK(log[i] > log[i - 1]);   // distinct, increasing
    for (long long t : {1LL, 10LL, 100LL}) CHECK(std::find(log.begin(), log.end(), t) != log.end());
    CHECK(log.size() <= 32);                                                  // at most 10 per decade
    CHECK(recording_times(20, 7, 0) == (std::vector<long long>{0, 7, 14, 20}));
    CHECK(recording_times(5, 1, 0) == (std::vector<long long>{0, 1, 2, 3, 4, 5}));
    CHECK(recording_times(10, 0, 3).back() == 10);
}

TEST_CASE(timeseries_voter_rho_decay_is_exact) {
    // Complete graph, n up spins: E[n(N-n)] shrinks by the factor
    // 1 - 2/(N(N-1)) at every update, so <rho>(t) = rho0 (1 - 2/(N(N-1)))^(N t)
    // for t in sweeps -- averaging over ALL runs, absorbed ones (rho = 0) included.
    TempPrefix tmp;
    const int N = 20;
    CHECK(run_quiet({"--model", "voter", "--graph", "complete", "--n", "20", "--task", "ensemble", "--replicas", "3000",
                     "--steps", "300", "--log-times", "--observables", "rho", "--seed", "1", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    CHECK(ts.at("time").front() == 0.0 && ts.at("time").back() == 300.0);
    const double rho0 = 2.0 * 10 * 10 / (N * (N - 1.0)), factor = 1.0 - 2.0 / (N * (N - 1.0));
    for (std::size_t k = 0; k < ts.at("time").size(); ++k) {
        const double expected = rho0 * std::pow(factor, N * ts.at("time")[k]);
        // Late on, almost every run has reached rho = 0; if all of them have,
        // the sample standard error is 0 too. Since 0 <= rho <= rho0, the
        // variance is at most E[rho] rho0, which bounds the error from the model.
        const double err = std::max(ts.at("rho_err")[k], std::sqrt(expected * rho0 / 3000.0));
        CHECK_NEAR(ts.at("rho_mean")[k], expected, 4.0 * err);
    }
}

TEST_CASE(timeseries_survival_and_running_columns) {
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "voter", "--graph", "complete", "--n", "30", "--task", "ensemble", "--replicas", "200",
                     "--steps", "200", "--log-times", "5", "--observables", "rho", "--seed", "2", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    const auto& s = ts.at("survival");
    CHECK(s.front() == 1.0);
    for (std::size_t k = 1; k < s.size(); ++k) CHECK(s[k] <= s[k - 1]);
    // Voter: a run stops exactly when it reaches consensus.
    CHECK(ts.at("running_fraction") == s);
    // Survivors still disagree somewhere; the all-run mean includes the zeros.
    for (std::size_t k = 0; k < s.size(); ++k)
        if (s[k] > 0) {
            CHECK(ts.at("rho_mean_running")[k] > 0.0);
            CHECK(ts.at("rho_mean")[k] <= ts.at("rho_mean_running")[k] + 1e-12);
        }
    // Final survival = 1 - fraction of runs that reached consensus.
    const auto reps = read_csv(tmp.prefix + "_replicas.csv");
    double reached = 0;
    for (double v : reps.at("reached_consensus")) reached += v;
    CHECK_NEAR(s.back(), 1.0 - reached / 200.0, 1e-12);
}

TEST_CASE(timeseries_activity_per_interval) {
    // Noisy voter at full noise: half of all updates change the opinion, in
    // every interval (a model that can leave consensus: all runs go to the cap).
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "noisy-voter", "--noise", "1", "--graph", "complete", "--n", "100", "--task", "ensemble",
                     "--replicas", "20", "--steps", "100", "--every", "10", "--observables", "activity", "--seed", "3",
                     "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    CHECK(ts.at("time").size() == 11);
    CHECK(ts.at("running_fraction").back() == 1.0);
    for (std::size_t k = 1; k < ts.at("time").size(); ++k) CHECK_NEAR(ts.at("activity_mean")[k], 0.5, 0.02);
}

TEST_CASE(timeseries_independent_of_threads) {
    TempPrefix a, b;
    for (auto* t : {&a, &b})
        CHECK(run_quiet({"--model", "voter", "--graph", "ba", "--n", "200", "--m", "2", "--task", "ensemble", "--replicas",
                         "30", "--log-times", "--observables", "rho,persistence", "--seed", "4", "--threads",
                         t == &a ? "1" : "5", "--csv", t->prefix}) == 0);
    const std::string sa = read_without_comments(a.prefix + "_timeseries.csv");
    const std::string sb = read_without_comments(b.prefix + "_timeseries.csv");
    CHECK(!sa.empty() && sa == sb);   // the data; the provenance differs (command line, time)
}

TEST_CASE(timeseries_run_task_log_times) {
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--steps", "1000", "--stop", "false", "--log-times",
                     "--seed", "5", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    std::vector<double> grid;
    for (long long t : recording_times(1000, 0, 10)) grid.push_back(static_cast<double>(t));
    CHECK(ts.at("sweep") == grid);
}

TEST_CASE(timeseries_in_sweeps_and_option_errors) {
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas", "10", "--sweep",
                     "n=10,20", "--log-times", "--observables", "rho", "--seed", "6", "--csv", tmp.prefix}) == 0);
    CHECK(fs::exists(tmp.prefix + "_point0_timeseries.csv") && fs::exists(tmp.prefix + "_point1_timeseries.csv"));
    CHECK(run_quiet({"--model", "voter", "--task", "ensemble", "--log-times", "0"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--task", "ensemble", "--log-times", "--every", "5"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--task", "stationary", "--log-times"}) == 2);   // does not apply
}
