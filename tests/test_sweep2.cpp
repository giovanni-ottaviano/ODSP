#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// Two-parameter sweeps: --sweep KEY=... --sweep2 KEY2=... runs the grid of all
// combinations (--sweep outer) and writes one table with both key columns.

using namespace driver;
namespace fs = std::filesystem;

namespace {

struct DriverRun {
    int status = 0;
    std::string out, err;
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
    r.err = err.str();
    return r;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

struct TempPrefix {
    std::string prefix;
    TempPrefix() {
        std::random_device rd;
        prefix = (fs::temp_directory_path() / ("odsp_sweep2_" + std::to_string(rd()))).string();
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

std::vector<std::vector<std::string>> rows_of(const std::string& csv) {
    std::vector<std::vector<std::string>> rows;
    std::istringstream in(csv);
    std::string line, cell;
    while (std::getline(in, line)) {
        std::vector<std::string> r;
        std::stringstream ls(line);
        while (std::getline(ls, cell, ',')) r.push_back(cell);
        rows.push_back(r);
    }
    return rows;
}

const std::vector<std::string> kEnsemble = {"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas",
                                            "8", "--sweep", "n=10,20", "--sweep2", "update=node,link,reverse",
                                            "--seed", "5"};

} // namespace

TEST_CASE(sweep2_runs_the_grid_in_order) {
    TempPrefix t;
    std::vector<std::string> args = kEnsemble;
    args.insert(args.end(), {"--threads", "4", "--csv", t.prefix});
    const DriverRun r = run(args);
    CHECK(r.status == 0);
    const auto rows = rows_of(t.file("_sweep.csv"));
    CHECK(rows.size() == 7);   // header + 2 x 3 points
    CHECK(rows[0][0] == "n" && rows[0][1] == "update" && rows[0][2] == "replicas");
    const std::vector<std::pair<std::string, std::string>> grid = {
        {"10", "node"}, {"10", "link"}, {"10", "reverse"}, {"20", "node"}, {"20", "link"}, {"20", "reverse"}};
    for (std::size_t i = 0; i < grid.size(); ++i) {
        CHECK(rows[i + 1][0] == grid[i].first && rows[i + 1][1] == grid[i].second);
        CHECK(rows[i + 1][2] == "8");
    }
    for (int i = 0; i < 6; ++i) CHECK(fs::exists(t.prefix + "_point" + std::to_string(i) + "_replicas.csv"));
    CHECK(rows_of(t.file("_point5_replicas.csv")).size() == 9);
    CHECK(contains(r.out, "for 2 x 3 values of --n, --update"));
}

TEST_CASE(sweep2_is_independent_of_threads_and_replays) {
    TempPrefix a, b, c;
    for (auto* t : {&a, &b}) {
        std::vector<std::string> args = {"--model", "ising", "--dims", "6x6", "--task", "stationary", "--samples", "100",
                                         "--burn-in", "20", "--replicas", "2", "--sweep", "dims=6x6,8x8", "--sweep2",
                                         "temperature=2,2.5,3", "--seed", "6", "--threads", t == &a ? "1" : "4",
                                         "--csv", t->prefix};
        CHECK(run(args).status == 0);
    }
    CHECK(!a.file("_sweep.csv").empty() && a.file("_sweep.csv") == b.file("_sweep.csv"));
    CHECK(rows_of(a.file("_sweep.csv")).size() == 7);
    CHECK(run({"--config", a.prefix + "_config.txt", "--csv", c.prefix}).status == 0);
    CHECK(a.file("_sweep.csv") == c.file("_sweep.csv"));
}

TEST_CASE(sweep2_errors) {
    auto status_and = [](const DriverRun& r, const std::string& part) { return r.status == 2 && contains(r.err, part); };
    const std::vector<std::string> base = {"--model", "voter", "--graph", "complete", "--n", "10", "--task", "ensemble",
                                           "--replicas", "2"};
    auto with = [&](std::vector<std::string> extra) {
        std::vector<std::string> a = base;
        a.insert(a.end(), extra.begin(), extra.end());
        return run(a);
    };
    CHECK(status_and(with({"--sweep2", "n=10,20"}), "--sweep2 needs --sweep"));
    CHECK(status_and(with({"--sweep", "n=10,20", "--sweep2", "n=30,40"}), "--sweep2: --n is already swept"));
    CHECK(status_and(with({"--sweep", "n=10,20", "--sweep2", "seed=1,2"}), "--sweep2: cannot sweep --seed"));
    CHECK(status_and(with({"--sweep", "n=10,20", "--sweep2", "temperature=1,2"}), "--sweep2: --temperature does not apply"));
    CHECK(status_and(with({"--sweep", "n=10,20", "--sweep2", "update=node:link"}), "option --sweep2"));
    CHECK(status_and(with({"--sweep", "n=10:60000:1", "--sweep2", "update=node,link"}), "more than 100000 points"));
}
