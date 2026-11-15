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

// --save-trajectories: every replica's time series in prefix_trajectories.csv.

using namespace driver;
namespace fs = std::filesystem;

namespace {

int run(std::vector<std::string> args, std::string* output = nullptr) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    if (output) *output = out.str() + err.str();
    return status;
}

struct TempFiles {
    std::string stem;
    TempFiles() {
        std::random_device rd;
        stem = (fs::temp_directory_path() / ("odsp_traj_" + std::to_string(rd()))).string();
    }
    ~TempFiles() {
        std::error_code ec;
        const fs::path dir = fs::path(stem).parent_path();
        const std::string name = fs::path(stem).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(name, 0) == 0) fs::remove(f.path(), ec);
    }
    std::string path(const std::string& suffix) const { return stem + suffix; }
};

// Header names of a CSV (after the '#' preamble).
std::vector<std::string> header_of(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    return names;
}

// A cell as a number; empty, "nan" or text cells (e.g. a swept "tie" value) as NaN.
double number(const std::string& cell) {
    try {
        std::size_t used = 0;
        const double v = std::stod(cell, &used);
        return used == cell.size() ? v : std::nan("");
    } catch (const std::exception&) {
        return std::nan("");
    }
}

// Columns by name; non-numeric cells parse as NaN.
std::map<std::string, std::vector<double>> read_csv(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    std::map<std::string, std::vector<double>> cols;
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (std::size_t k = 0; std::getline(ls, cell, ','); ++k)
            cols[names[k]].push_back(number(cell));
    }
    return cols;
}

const std::vector<std::string> kEnsemble = {"--model", "voter", "--dims", "8x8", "--task", "ensemble",
    "--replicas", "5", "--steps", "40", "--log-times", "5", "--observables", "rho,abs_m", "--seed", "1"};

std::vector<std::string> with(std::vector<std::string> a, std::vector<std::string> extra) {
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

} // namespace

TEST_CASE(trajectories_columns_and_rows) {
    TempFiles f;
    CHECK(run(with(kEnsemble, {"--save-trajectories", "true", "--csv", f.stem})) == 0);
    const auto names = header_of(f.path("_trajectories.csv"));
    CHECK((names == std::vector<std::string>{"replica", "time", "rho", "abs_m"}));
    const auto traj = read_csv(f.path("_trajectories.csv"));
    const auto ts = read_csv(f.path("_timeseries.csv"));
    const std::size_t K = ts.at("time").size();
    CHECK(traj.at("time").size() == 5 * K);
    for (std::size_t r = 0; r < 5; ++r) {
        CHECK(traj.at("replica")[r * K] == static_cast<double>(r));
        for (std::size_t k = 0; k < K; ++k) CHECK(traj.at("time")[r * K + k] == ts.at("time")[k]);
    }
}

TEST_CASE(trajectories_means_match_timeseries) {
    TempFiles f;
    CHECK(run(with(kEnsemble, {"--save-trajectories", "true", "--csv", f.stem})) == 0);
    const auto traj = read_csv(f.path("_trajectories.csv"));
    const auto ts = read_csv(f.path("_timeseries.csv"));
    const std::size_t K = ts.at("time").size();
    for (const std::string name : {"rho", "abs_m"})
        for (std::size_t k = 0; k < K; ++k) {
            double sum = 0.0;
            for (std::size_t r = 0; r < 5; ++r) sum += traj.at(name)[r * K + k];
            CHECK_NEAR(sum / 5, ts.at(name + "_mean")[k], 1e-12);
        }
}

TEST_CASE(trajectories_option_errors) {
    const std::string msg = "--save-trajectories is for --task ensemble with a time series (--every or --log-times)";
    TempFiles f;
    std::string out;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "run", "--steps", "10", "--save-trajectories", "true",
               "--csv", f.stem}, &out) == 2);
    CHECK(out.find(msg) != std::string::npos);
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--replicas", "2", "--steps", "10",
               "--save-trajectories", "true", "--csv", f.stem}, &out) == 2);
    CHECK(out.find(msg) != std::string::npos);
    CHECK(run(with(kEnsemble, {"--save-trajectories", "true"}), &out) == 2);
    CHECK(out.find("--save-trajectories writes <prefix>_trajectories.csv: add --csv <prefix>") != std::string::npos);
}

TEST_CASE(trajectories_off_by_default) {
    TempFiles f;
    CHECK(run(with(kEnsemble, {"--csv", f.stem})) == 0);
    CHECK(fs::exists(f.path("_timeseries.csv")));
    CHECK(!fs::exists(f.path("_trajectories.csv")));
}
