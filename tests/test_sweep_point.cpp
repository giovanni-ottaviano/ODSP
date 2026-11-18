#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "DriverTestHelpers.hpp"

// --sweep-point i: run only point i of a sweep, with the seed and the output
// files it gets in the full run, so a sweep split across jobs merges into it.

using namespace driver_test;

namespace {

std::vector<std::string> with(std::vector<std::string> a, const std::vector<std::string>& extra) {
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

// The data lines of a CSV (no preamble, no header).
std::vector<std::string> data_lines(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line;
    std::getline(in, line);
    std::vector<std::string> lines;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

bool exists(const std::string& path) { return std::filesystem::exists(path); }

}  // namespace

TEST_CASE(sweep_point_row_equals_the_full_runs_row) {
    const std::vector<std::string> base = {"--model", "voter", "--task", "ensemble", "--sweep", "dims=4x4,6x6,8x8",
                                           "--replicas", "6", "--steps", "300", "--seed", "9"};
    TempFiles full, one;
    CHECK(run(with(base, {"--csv", full.stem})) == 0);
    std::string out;
    CHECK(run(with(base, {"--sweep-point", "1", "--csv", one.stem}), &out) == 0);
    CHECK(out.find("point 1 of 3") != std::string::npos);
    CHECK(header_of(one.path("_sweep.csv")) == header_of(full.path("_sweep.csv")));
    const auto all = data_lines(full.path("_sweep.csv"));
    const auto got = data_lines(one.path("_sweep.csv"));
    if (!(all.size() == 3 && got.size() == 1)) { CHECK(all.size() == 3 && got.size() == 1); return; }
    CHECK(got[0] == all[1]);
    CHECK(data_lines(one.path("_point1_replicas.csv")) == data_lines(full.path("_point1_replicas.csv")));
    CHECK(!exists(one.path("_point0_replicas.csv")));
    CHECK(!exists(one.path("_point2_replicas.csv")));
    std::ifstream cfg(one.path("_config.txt"));
    std::stringstream text;
    text << cfg.rdbuf();
    CHECK(text.str().find("sweep-point = 1") != std::string::npos);
}

TEST_CASE(sweep_point_with_sweep2_uses_flat_order) {
    const std::vector<std::string> base = {"--model", "noisy-voter", "--task", "ensemble", "--sweep", "dims=4x4,6x6",
                                           "--sweep2", "noise=0.1,0.2", "--replicas", "4", "--steps", "200",
                                           "--seed", "5"};
    TempFiles full, one;
    CHECK(run(with(base, {"--csv", full.stem})) == 0);
    CHECK(run(with(base, {"--sweep-point", "3", "--csv", one.stem})) == 0);
    const auto all = data_lines(full.path("_sweep.csv"));
    const auto got = data_lines(one.path("_sweep.csv"));
    if (!(all.size() == 4 && got.size() == 1)) { CHECK(all.size() == 4 && got.size() == 1); return; }
    CHECK(got[0] == all[3]);
    CHECK(got[0].rfind("6x6,0.2,", 0) == 0);
}

TEST_CASE(sweep_point_stationary) {
    const std::vector<std::string> base = {"--model", "ising", "--dims", "6x6", "--task", "stationary",
                                           "--samples", "200", "--burn-in", "50", "--sweep", "temperature=2.2,2.4",
                                           "--seed", "3"};
    TempFiles full, one;
    CHECK(run(with(base, {"--csv", full.stem})) == 0);
    CHECK(run(with(base, {"--sweep-point", "0", "--csv", one.stem})) == 0);
    const auto all = data_lines(full.path("_sweep.csv"));
    const auto got = data_lines(one.path("_sweep.csv"));
    if (!(all.size() == 2 && got.size() == 1)) { CHECK(all.size() == 2 && got.size() == 1); return; }
    CHECK(got[0] == all[0]);
}

TEST_CASE(sweep_point_with_first_replica) {
    const std::vector<std::string> base = {"--model", "voter", "--task", "ensemble", "--sweep", "dims=4x4,6x6",
                                           "--steps", "300", "--seed", "7"};
    TempFiles full, part;
    CHECK(run(with(base, {"--replicas", "5", "--csv", full.stem})) == 0);
    CHECK(run(with(base, {"--replicas", "3", "--first-replica", "2", "--sweep-point", "1", "--csv", part.stem})) == 0);
    const auto all = data_lines(full.path("_point1_replicas.csv"));
    const auto got = data_lines(part.path("_point1_replicas.csv"));
    if (!(all.size() == 5 && got.size() == 3)) { CHECK(all.size() == 5 && got.size() == 3); return; }
    for (std::size_t k = 0; k < 3; ++k) CHECK(got[k] == all[2 + k]);
    CHECK(got[0].rfind("2,", 0) == 0);
}

TEST_CASE(sweep_point_validation) {
    const std::vector<std::string> base = {"--model", "voter", "--dims", "4x4", "--task", "ensemble",
                                           "--replicas", "2", "--steps", "50"};
    std::string out;
    CHECK(run(with(base, {"--sweep-point", "0"}), &out) == 2);   // no --sweep
    CHECK(out.find("--sweep-point") != std::string::npos);
    CHECK(run({"--model", "voter", "--dims", "4x4", "--sweep-point", "0"}, &out) == 2);   // task run
    CHECK(out.find("--sweep-point") != std::string::npos);
    const auto sw = with(base, {"--sweep", "dims=4x4,6x6"});
    CHECK(run(with(sw, {"--sweep-point", "-1"}), &out) == 2);
    CHECK(out.find("--sweep-point") != std::string::npos);
    CHECK(run(with(sw, {"--sweep-point", "2"}), &out) == 2);
    CHECK(out.find("--sweep-point") != std::string::npos);
    CHECK(out.find("2 points") != std::string::npos);
    CHECK(run(with(sw, {"--sweep-point", "x"}), &out) == 2);
    CHECK(out.find("--sweep-point") != std::string::npos);
}
