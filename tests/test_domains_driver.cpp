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

// Domain observables in the driver, prefix_domains.csv, and the majority
// model's freeze_time.

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
        stem = (fs::temp_directory_path() / ("odsp_domains_" + std::to_string(rd()))).string();
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

bool has(const std::vector<std::string>& v, const std::string& x) {
    for (const auto& s : v)
        if (s == x) return true;
    return false;
}

} // namespace

// ------------------------------------------------------------- observables

TEST_CASE(domains_driver_run_time_series) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "16x16", "--steps", "20", "--stop", "false", "--observables",
               "domains,largest_domain,mean_domain,weighted_domain,stripes", "--seed", "1", "--csv", tmp.stem}) == 0);
    const auto ts = read_csv(tmp.path("_timeseries.csv"));
    for (const char* name : {"domains", "largest_domain", "mean_domain", "weighted_domain", "stripes"})
        CHECK(ts.count(name) == 1);
    const std::size_t rows = ts.at("domains").size();
    CHECK(rows > 0);
    for (std::size_t k = 0; k < rows; ++k) {
        CHECK_NEAR(ts.at("domains")[k] * ts.at("mean_domain")[k], 1.0, 1e-12);
        CHECK(ts.at("largest_domain")[k] > 0.0 && ts.at("largest_domain")[k] <= 1.0);
        CHECK(ts.at("weighted_domain")[k] >= ts.at("mean_domain")[k] - 1e-12);
        CHECK(ts.at("stripes")[k] == 0.0 || ts.at("stripes")[k] == 1.0);
    }
}

TEST_CASE(domains_driver_continuous_refused) {
    std::string out;
    CHECK(run({"--model", "deffuant", "--epsilon", "0.3", "--dims", "8x8", "--steps", "5", "--observables", "domains"}, &out) != 0);
    CHECK(out.find("is for discrete opinions") != std::string::npos);
}

TEST_CASE(domains_driver_stripes_needs_periodic_lattice) {
    std::string out;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--boundary", "open", "--steps", "5", "--observables", "stripes"}, &out) != 0);
    CHECK(out.find("periodic lattice") != std::string::npos);
    CHECK(run({"--model", "voter", "--graph", "regular", "--n", "50", "--k", "4", "--steps", "5", "--observables", "stripes"}) != 0);
    CHECK(run({"--model", "voter", "--dims", "8x8", "--boundary", "open", "--steps", "5", "--observables", "domains"}) == 0);
}

TEST_CASE(domains_driver_help_lists_observables) {
    std::string out;
    run({"--help", "observables"}, &out);
    for (const char* name : {"domains", "largest_domain", "mean_domain", "weighted_domain", "stripes"})
        CHECK(out.find(std::string("  ") + name + " ") != std::string::npos);
}

// ------------------------------------------------------------- ensemble outputs

namespace {
// Rows of prefix_domains.csv grouped by replica: sizes (and wraps, if present).
struct DomainRows {
    std::map<int, std::vector<double>> sizes, wraps;
};
DomainRows read_domains(const std::string& path) {
    const auto c = read_csv(path);
    DomainRows d;
    for (std::size_t k = 0; k < c.at("replica").size(); ++k) {
        const int r = static_cast<int>(c.at("replica")[k]);
        d.sizes[r].push_back(c.at("size")[k]);
        if (c.count("wraps")) d.wraps[r].push_back(c.at("wraps")[k]);
    }
    return d;
}

// Horizontal bands on an 8 x 8 lattice (rows 0-4: +1), frozen under majority rule.
std::string write_bands(const TempFiles& tmp) {
    const std::string f = tmp.path("_bands.txt");
    std::ofstream out(f);
    for (int i = 0; i < 64; ++i) out << i << ' ' << (i / 8 < 5 ? 1 : -1) << '\n';
    return f;
}
} // namespace

TEST_CASE(domains_driver_ensemble_domains_csv) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--replicas", "6", "--steps", "30",
               "--observables", "domains,stripes", "--seed", "2", "--csv", tmp.stem}) == 0);
    CHECK(header_of(tmp.path("_domains.csv")) == (std::vector<std::string>{"replica", "size", "wraps"}));
    const DomainRows d = read_domains(tmp.path("_domains.csv"));
    const auto rep = read_csv(tmp.path("_replicas.csv"));
    CHECK(d.sizes.size() == 6);
    for (int r = 0; r < 6; ++r) {
        const std::vector<double>& s = d.sizes.at(r);
        double total = 0.0;
        for (std::size_t k = 0; k < s.size(); ++k) {
            total += s[k];
            if (k > 0) CHECK(s[k] <= s[k - 1]);   // decreasing size
        }
        CHECK(total == 64.0);
        CHECK_NEAR(static_cast<double>(s.size()), rep.at("domains")[static_cast<std::size_t>(r)] * 64.0, 1e-9);
        for (double w : d.wraps.at(r)) CHECK(w >= 0.0 && w <= 2.0);
    }
}

TEST_CASE(domains_driver_no_domains_csv_unless_requested) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--replicas", "3", "--steps", "10",
               "--observables", "m", "--seed", "2", "--csv", tmp.stem}) == 0);
    CHECK(fs::exists(tmp.path("_replicas.csv")));
    CHECK(!fs::exists(tmp.path("_domains.csv")));
}

TEST_CASE(domains_driver_open_lattice_no_wraps_column) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--boundary", "open", "--task", "ensemble", "--replicas", "3",
               "--steps", "10", "--observables", "domains", "--seed", "2", "--csv", tmp.stem}) == 0);
    CHECK(header_of(tmp.path("_domains.csv")) == (std::vector<std::string>{"replica", "size"}));
}

TEST_CASE(domains_driver_majority_freezes) {
    TempFiles tmp;
    CHECK(run({"--model", "majority", "--tie", "keep", "--dims", "16x16", "--task", "ensemble", "--replicas", "10",
               "--steps", "100000", "--observables", "stripes,domains", "--seed", "3", "--csv", tmp.stem}) == 0);
    const std::vector<std::string> h = header_of(tmp.path("_replicas.csv"));
    CHECK(h.size() > 4 && h[3] == "stop_time" && h[4] == "freeze_time");
    const auto rep = read_csv(tmp.path("_replicas.csv"));
    for (std::size_t r = 0; r < 10; ++r) {
        CHECK(rep.at("absorbed")[r] == 1.0);
        CHECK(rep.at("stop_time")[r] < 100000);
        const double lag = rep.at("stop_time")[r] - rep.at("freeze_time")[r];
        CHECK(lag == (rep.at("reached_consensus")[r] == 1.0 ? 0.0 : 1.0));
    }
}

TEST_CASE(domains_driver_freeze_time_summary_in_sweep_table) {
    TempFiles tmp;
    CHECK(run({"--model", "majority", "--dims", "8x8", "--task", "ensemble", "--replicas", "4", "--steps", "100000",
               "--observables", "domains", "--sweep", "tie=random,keep", "--seed", "3", "--csv", tmp.stem}) == 0);
    const auto sw = read_csv(tmp.path("_sweep.csv"));
    CHECK(sw.count("mean_freeze_time") == 1 && sw.count("sd_freeze_time") == 1);
    CHECK(sw.at("absorbed_fraction") == (std::vector<double>{1.0, 1.0}));
    for (double v : sw.at("mean_freeze_time")) CHECK(v >= 0.0);
    CHECK(fs::exists(tmp.path("_point0_domains.csv")) && fs::exists(tmp.path("_point1_domains.csv")));
}

TEST_CASE(domains_driver_freeze_time_already_frozen) {
    TempFiles tmp;
    const std::string bands = write_bands(tmp);
    CHECK(run({"--model", "majority", "--dims", "8x8", "--init", "file", "--init-path", bands, "--task", "ensemble",
               "--replicas", "2", "--steps", "50", "--observables", "stripes", "--seed", "1", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.path("_replicas.csv"));
    CHECK(rep.at("absorbed") == (std::vector<double>{1.0, 1.0}));
    CHECK(rep.at("stop_time") == (std::vector<double>{1.0, 1.0}));
    CHECK(rep.at("freeze_time") == (std::vector<double>{0.0, 0.0}));
    CHECK(rep.at("stripes") == (std::vector<double>{1.0, 1.0}));
}

TEST_CASE(domains_driver_freeze_time_updates_unit) {
    TempFiles tmp;
    CHECK(run({"--model", "majority", "--tie", "keep", "--dims", "8x8", "--task", "ensemble", "--replicas", "4",
               "--steps", "1000000", "--time-unit", "updates", "--seed", "5", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.path("_replicas.csv"));
    for (std::size_t r = 0; r < 4; ++r) {
        CHECK(rep.at("absorbed")[r] == 1.0);
        const double lag = rep.at("stop_time")[r] - rep.at("freeze_time")[r];
        CHECK(lag == (rep.at("reached_consensus")[r] == 1.0 ? 0.0 : 64.0));
    }
}

TEST_CASE(domains_driver_freeze_time_capped_replica_is_nan) {
    TempFiles tmp;
    CHECK(run({"--model", "majority", "--dims", "16x16", "--task", "ensemble", "--replicas", "2", "--steps", "1",
               "--seed", "5", "--csv", tmp.stem}) == 0);
    const auto rep = read_csv(tmp.path("_replicas.csv"));
    for (std::size_t r = 0; r < 2; ++r) {
        CHECK(rep.at("absorbed")[r] == 0.0);
        CHECK(std::isnan(rep.at("freeze_time")[r]));
    }
}

TEST_CASE(domains_driver_freeze_time_only_for_majority) {
    TempFiles tmp;
    CHECK(run({"--model", "voter", "--dims", "8x8", "--task", "ensemble", "--replicas", "3", "--steps", "10",
               "--sweep", "dims=6x6,8x8", "--seed", "2", "--csv", tmp.stem}) == 0);
    CHECK(!has(header_of(tmp.path("_point0_replicas.csv")), "freeze_time"));
    CHECK(!has(header_of(tmp.path("_sweep.csv")), "mean_freeze_time"));
}
