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

// Correlation observables in the driver: coarsening length, two-time
// autocorrelation / overlap with --tw, second-moment xi in stationary runs, and
// G(r) curves with --correlation.

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
        prefix = (fs::temp_directory_path() / ("odsp_corr_" + std::to_string(rd()))).string();
    }
    ~TempPrefix() {
        std::error_code ec;
        const fs::path dir = fs::path(prefix).parent_path();
        const std::string stem = fs::path(prefix).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(stem, 0) == 0) fs::remove(f.path(), ec);
    }
};

// Columns of a CSV (comments skipped); "nan" cells become NaN.
std::map<std::string, std::vector<double>> read_csv(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    std::map<std::string, std::vector<double>> cols;
    for (const auto& n : names) cols[n];
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (std::size_t k = 0; std::getline(ls, cell, ','); ++k)
            cols[names[k]].push_back(cell == "nan" || cell == "-nan" ? std::nan("") : std::stod(cell));
    }
    return cols;
}

double row_value(const Row& row, const std::string& name) {
    for (const auto& c : row) if (c.first == name) return c.second;
    odsp_test::fail(__FILE__, __LINE__, "column '" + name + "' missing");
}

bool has_column(const Row& row, const std::string& name) {
    for (const auto& c : row) if (c.first == name) return true;
    return false;
}

Plan<int> plan_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    Options o = Options::from_args(static_cast<int>(argv.size()), argv.data());
    return read_plan<int>(o);
}

} // namespace

// ------------------------------------------------- second-moment length xi

TEST_CASE(xi_of_the_one_dimensional_ising_chain_is_exact) {
    // 1D Ising: G(r) = t^r with t = tanh(1/T), so S(k) = (1-t^2)/(1+t^2-2t cos k)
    // and the second-moment length is exactly sqrt(t)/(1-t) (3.66 at T = 1).
    const double t = std::tanh(1.0), exact = std::sqrt(t) / (1.0 - t);
    Plan<int> p = plan_of({"--model", "ising", "--temperature", "1", "--dims", "200", "--init", "up", "--task",
                           "stationary", "--replicas", "4", "--burn-in", "500", "--samples", "20000", "--interval", "2"});
    const Row row = task_stationary<int>(p.params, p.setup, 1, "", false, 4);
    CHECK_NEAR(row_value(row, "xi"), exact, 4.0 * row_value(row, "xi_err"));
    CHECK_NEAR(row_value(row, "xi_over_L"), row_value(row, "xi") / 200.0, 1e-12);
}

TEST_CASE(xi_only_on_periodic_hypercubic_lattices) {
    const auto lattice = plan_of({"--model", "ising", "--temperature", "3", "--dims", "8x8", "--task", "stationary",
                                  "--samples", "50", "--burn-in", "10"});
    CHECK(has_column(task_stationary<int>(lattice.params, lattice.setup, 2, "", false), "xi"));
    for (std::vector<std::string> args : {
             std::vector<std::string>{"--model", "ising", "--temperature", "3", "--graph", "complete", "--n", "50"},
             std::vector<std::string>{"--model", "ising", "--temperature", "3", "--dims", "8x8", "--boundary", "open"},
             std::vector<std::string>{"--model", "ising", "--temperature", "3", "--dims", "8x6"}}) {
        args.insert(args.end(), {"--task", "stationary", "--samples", "50", "--burn-in", "10"});
        const auto p = plan_of(args);
        CHECK(!has_column(task_stationary<int>(p.params, p.setup, 2, "", false), "xi"));
    }
}

// --------------------------------------------------------- coarsening length

TEST_CASE(coarsening_length_grows_in_the_voter_model) {
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "voter", "--dims", "32x32", "--task", "ensemble", "--replicas", "8", "--steps", "300",
                     "--log-times", "--observables", "length", "--seed", "3", "--threads", "4", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    CHECK(ts.at("length_mean").front() < 1.0);           // random start: correlated over less than a site
    CHECK(ts.at("length_mean").back() > 3.0);            // domains have grown
    CHECK(run_quiet({"--model", "voter", "--graph", "complete", "--n", "50", "--observables", "length"}) == 2);
}

// ------------------------------------------------ two-time autocorrelation

TEST_CASE(autocorrelation_of_the_noisy_voter_at_full_noise_is_exact) {
    // Every update gives a node a fresh random opinion: s_i(t) s_i(t_w) averages
    // to the probability that i was never updated, (1 - 1/N)^(N (t - t_w)).
    TempPrefix tmp;
    const int N = 100, tw = 5;
    CHECK(run_quiet({"--model", "noisy-voter", "--noise", "1", "--graph", "complete", "--n", "100", "--task",
                     "ensemble", "--replicas", "300", "--steps", "12", "--every", "1", "--tw", "5", "--observables",
                     "autocorrelation", "--seed", "4", "--threads", "4", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    const auto& time = ts.at("time");
    const auto& c = ts.at("autocorrelation_tw5_mean");
    const auto& err = ts.at("autocorrelation_tw5_err");
    for (std::size_t k = 0; k < time.size(); ++k) {
        if (time[k] < tw) { CHECK(std::isnan(c[k])); continue; }
        const double expected = std::pow(1.0 - 1.0 / N, N * (time[k] - tw));
        CHECK_NEAR(c[k], expected, 4.0 * err[k] + 1e-12);
    }
}

TEST_CASE(tw_times_join_the_recording_grid) {
    TempPrefix tmp;
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--steps", "100", "--stop", "false", "--log-times", "--tw",
                     "7,33", "--observables", "m,autocorrelation,overlap", "--seed", "5", "--csv", tmp.prefix}) == 0);
    const auto ts = read_csv(tmp.prefix + "_timeseries.csv");
    const auto& sweep = ts.at("sweep");
    for (double tw : {7.0, 33.0}) {
        const auto it = std::find(sweep.begin(), sweep.end(), tw);
        CHECK(it != sweep.end());
        const std::size_t k = static_cast<std::size_t>(it - sweep.begin());
        const std::string suffix = "_tw" + std::to_string(static_cast<int>(tw));
        CHECK(ts.at("autocorrelation" + suffix)[k] == 1.0);
        CHECK(ts.at("overlap" + suffix)[k] == 1.0);
        CHECK(std::isnan(ts.at("autocorrelation" + suffix)[0]));
    }
}

TEST_CASE(autocorrelation_option_errors) {
    // Needs --tw; needs a time series in ensembles; +/-1 or continuous opinions only.
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--observables", "autocorrelation"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--tw", "5", "--observables",
                     "autocorrelation"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--init", "balanced", "--opinions", "3", "--tw", "5",
                     "--observables", "autocorrelation"}) == 2);
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--init", "balanced", "--opinions", "3", "--tw", "5",
                     "--steps", "10", "--observables", "overlap"}) == 0);
    CHECK(run_quiet({"--model", "deffuant", "--epsilon", "0.3", "--dims", "8x8", "--tw", "5", "--steps", "10",
                     "--observables", "autocorrelation"}) == 0);
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--task", "stationary", "--tw", "5"}) == 2);   // unused
}

// --------------------------------------------------------------- G(r) curves

TEST_CASE(correlation_curves_for_each_task) {
    TempPrefix a, b, c;
    CHECK(run_quiet({"--model", "voter", "--dims", "16x16", "--steps", "20", "--stop", "false", "--every", "5",
                     "--correlation", "--seed", "6", "--csv", a.prefix}) == 0);
    const auto run = read_csv(a.prefix + "_correlation.csv");
    CHECK(run.at("sweep").size() == 5 * 9);   // times 0,5,10,15,20 x r = 0..8
    CHECK(run.at("G").front() == 1.0);        // G(0)

    CHECK(run_quiet({"--model", "voter", "--dims", "16x16", "--task", "ensemble", "--replicas", "6", "--steps", "50",
                     "--log-times", "--correlation", "--seed", "7", "--threads", "3", "--csv", b.prefix}) == 0);
    const auto ens = read_csv(b.prefix + "_correlation.csv");
    CHECK(ens.count("G_mean") && ens.count("G_err") && ens.count("time") && ens.count("r"));

    CHECK(run_quiet({"--model", "ising", "--temperature", "2.5", "--dims", "16x16", "--task", "stationary", "--samples",
                     "200", "--burn-in", "50", "--correlation", "--seed", "8", "--csv", c.prefix}) == 0);
    const auto st = read_csv(c.prefix + "_correlation.csv");
    CHECK(st.at("r").size() == 9);
    CHECK(st.at("G_mean")[0] == 1.0);
    for (std::size_t r = 1; r < 9; ++r) CHECK(st.at("G_mean")[r] < st.at("G_mean")[r - 1] + 3.0 * st.at("G_err")[r]);

    CHECK(run_quiet({"--model", "voter", "--graph", "complete", "--n", "30", "--correlation"}) == 2);       // lattices only
    CHECK(run_quiet({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--correlation"}) == 2);  // needs a time series
}

TEST_CASE(ensemble_correlation_independent_of_threads) {
    TempPrefix a, b;
    for (auto* t : {&a, &b})
        CHECK(run_quiet({"--model", "voter", "--dims", "12x12", "--task", "ensemble", "--replicas", "10", "--steps", "40",
                         "--log-times", "--correlation", "--observables", "length", "--seed", "9", "--threads",
                         t == &a ? "1" : "4", "--csv", t->prefix}) == 0);
    CHECK(read_without_comments(a.prefix + "_correlation.csv") == read_without_comments(b.prefix + "_correlation.csv"));
    CHECK(read_without_comments(a.prefix + "_timeseries.csv") == read_without_comments(b.prefix + "_timeseries.csv"));
}
