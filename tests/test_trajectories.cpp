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
#include "DriverTestHelpers.hpp"
#include "driver/Driver.hpp"

// --save-trajectories: every replica's time series in prefix_trajectories.csv.

using namespace driver;
namespace fs = std::filesystem;

using namespace driver_test;

namespace {

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
