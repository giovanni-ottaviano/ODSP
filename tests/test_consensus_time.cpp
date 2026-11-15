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

// consensus_time in prefix_replicas.csv: written only for models that can leave
// consensus (first passage), where stop_time is the cap.

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
        stem = (fs::temp_directory_path() / ("odsp_ctime_" + std::to_string(rd()))).string();
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

} // namespace

TEST_CASE(consensus_time_written_for_first_passage_models) {
    TempFiles tmp;
    std::string out;
    CHECK(run({"--model", "noisy-voter", "--noise", "0.001", "--dims", "6x6", "--task", "ensemble",
               "--replicas", "30", "--steps", "2000", "--seed", "5", "--csv", tmp.stem}, &out) == 0);
    const std::vector<std::string> h = header_of(tmp.stem + "_replicas.csv");
    CHECK(h.size() > 4 && h[3] == "stop_time" && h[4] == "consensus_time");
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    double sum = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i) {
        CHECK(rep.at("stop_time")[i] == 2000.0);                     // the cap, as before
        if (rep.at("reached_consensus")[i] == 1.0) {
            CHECK(rep.at("consensus_time")[i] >= 0.0 && rep.at("consensus_time")[i] <= 2000.0);
            sum += rep.at("consensus_time")[i];
            ++n;
        } else {
            CHECK(std::isnan(rep.at("consensus_time")[i]));
        }
    }
    CHECK(n > 0);
    // the driver's own first-passage mean, from its summary line
    const std::size_t at = out.find("First-passage time (sweeps): mean ");
    CHECK(at != std::string::npos);
    const double printed = std::stod(out.substr(at + std::string("First-passage time (sweeps): mean ").size()));
    CHECK_NEAR(sum / n, printed, 1e-3 * printed + 1e-9);
    // and the median (ceil(R/2)-th smallest, unreached last): order-sensitive per replica
    std::vector<double> times;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i)
        times.push_back(rep.at("reached_consensus")[i] == 1.0 ? rep.at("consensus_time")[i] : 1e300);
    std::sort(times.begin(), times.end());
    const std::size_t m = out.find(", median ", at);
    CHECK(m != std::string::npos);
    CHECK_NEAR(times[(times.size() + 1) / 2 - 1], std::stod(out.substr(m + 9)), 1e-9);
}

TEST_CASE(consensus_time_capped_runs_are_nan) {
    TempFiles tmp;
    CHECK(run({"--model", "noisy-voter", "--noise", "0.5", "--dims", "8x8", "--task", "ensemble",
               "--replicas", "6", "--steps", "3", "--seed", "2", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.stem + "_replicas.csv");
    int capped = 0;
    for (std::size_t i = 0; i < rep.at("replica").size(); ++i)
        if (rep.at("reached_consensus")[i] == 0.0) {
            CHECK(std::isnan(rep.at("consensus_time")[i]));
            ++capped;
        }
    CHECK(capped > 0);                      // the fixture really has runs that hit the cap
}

TEST_CASE(consensus_time_absent_for_absorbing_models) {
    for (std::vector<std::string> model : {std::vector<std::string>{"--model", "voter"},
                                           std::vector<std::string>{"--model", "majority"}}) {
        TempFiles tmp;
        std::vector<std::string> args = model;
        args.insert(args.end(), {"--dims", "6x6", "--task", "ensemble", "--replicas", "4", "--steps", "500",
                                 "--seed", "3", "--csv", tmp.stem});
        CHECK(run(args) == 0);
        const std::vector<std::string> h = header_of(tmp.stem + "_replicas.csv");
        for (const auto& name : h) CHECK(name != "consensus_time");
    }
}
