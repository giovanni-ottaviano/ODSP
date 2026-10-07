#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "CsvWriter.hpp"
#include "Version.hpp"
#include "driver/Driver.hpp"

// Provenance: every output file records the ODSP version, git commit, date,
// command line and full configuration as '#' comment lines.

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
        prefix = (fs::temp_directory_path() / ("odsp_prov_" + std::to_string(rd()))).string();
    }
    ~TempPrefix() {
        std::error_code ec;
        const fs::path dir = fs::path(prefix).parent_path();
        const std::string stem = fs::path(prefix).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(stem, 0) == 0) fs::remove(f.path(), ec);
    }
    std::string file(const std::string& suffix) const {
        std::ifstream in(prefix + suffix);
        std::ostringstream os;
        os << in.rdbuf();
        return os.str();
    }
};

// The file without its '#' lines.
std::string body(const std::string& text) {
    std::istringstream in(text);
    std::string line, out;
    while (std::getline(in, line))
        if (line.empty() || line[0] != '#') out += line + "\n";
    return out;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

std::string git_describe() {
    FILE* pipe = popen("git describe --always --dirty 2>/dev/null", "r");
    if (!pipe) return "unknown";
    char buf[256] = {0};
    std::string out = fgets(buf, sizeof(buf), pipe) ? buf : "";
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? "unknown" : out;
}

} // namespace

TEST_CASE(provenance_csv_writer_preamble) {
    const std::string saved = CsvWriter::preamble();
    TempPrefix t;
    CsvWriter::set_preamble("first line\nsecond\n\nlast");
    {
        CsvWriter csv(t.prefix + "_a.csv", {"x", "y"});
        csv.row(1, 2);
    }
    CHECK(t.file("_a.csv") == "# first line\n# second\n#\n# last\nx,y\n1,2\n");
    CsvWriter::set_preamble("");
    {
        CsvWriter csv(t.prefix + "_b.csv", {"x"});
        csv.row(3);
    }
    CHECK(t.file("_b.csv") == "x\n3\n");
    CsvWriter::set_preamble(saved);
}

TEST_CASE(provenance_version_matches_git) {
    CHECK(std::string(ODSP_VERSION) == "0.1.0");
    CHECK(std::string(ODSP_GIT_VERSION) == git_describe());   // embedded by the makefile
    CHECK(contains(odsp_version_string(), "ODSP 0.1.0, git "));
    const DriverRun v = run({"--version"});
    CHECK(v.status == 0 && contains(v.out, odsp_version_string()));
}

TEST_CASE(provenance_in_driver_outputs) {
    TempPrefix t;
    CHECK(run({"--model", "voter", "--graph", "complete", "--n", "20", "--task", "ensemble", "--replicas", "5",
               "--log-times", "--seed", "5", "--csv", t.prefix}).status == 0);
    for (const char* f : {"_replicas.csv", "_timeseries.csv", "_fixation.csv"}) {
        const std::string text = t.file(f);
        CHECK(text.rfind("# " + odsp_version_string() + ", written ", 0) == 0);
        CHECK(contains(text, "\n# command: odsp --model voter --graph complete"));
        CHECK(contains(text, "\n#   seed = 5\n"));
        CHECK(contains(text, "\n#   model = voter\n"));
        CHECK(body(text).rfind(f == std::string("_fixation.csv") ? "opinion," : (f == std::string("_replicas.csv") ? "replica," : "time,"), 0) == 0);
    }
    CHECK(t.file("_config.txt").rfind("# " + odsp_version_string(), 0) == 0);
}

TEST_CASE(provenance_can_be_turned_off) {
    TempPrefix t;
    CHECK(run({"--model", "voter", "--dims", "6x6", "--steps", "3", "--stop", "false", "--seed", "6", "--provenance",
               "false", "--csv", t.prefix}).status == 0);
    CHECK(t.file("_timeseries.csv").rfind("sweep,", 0) == 0);
}

TEST_CASE(provenance_config_replay_still_exact) {
    TempPrefix a, b;
    CHECK(run({"--model", "voter", "--graph", "ba", "--n", "100", "--m", "2", "--task", "ensemble", "--replicas", "8",
               "--seed", "7", "--csv", a.prefix}).status == 0);
    CHECK(run({"--config", a.prefix + "_config.txt", "--csv", b.prefix}).status == 0);
    CHECK(!body(a.file("_replicas.csv")).empty() && body(a.file("_replicas.csv")) == body(b.file("_replicas.csv")));
}

TEST_CASE(provenance_in_sweep_outputs) {
    TempPrefix t;
    CHECK(run({"--model", "voter", "--graph", "complete", "--task", "ensemble", "--replicas", "4", "--sweep", "n=10,20",
               "--seed", "8", "--threads", "2", "--csv", t.prefix}).status == 0);
    for (const char* f : {"_sweep.csv", "_point0_replicas.csv", "_point1_replicas.csv"}) {
        CHECK(t.file(f).rfind("# ODSP ", 0) == 0);
        CHECK(contains(t.file(f), "\n#   sweep = n=10,20\n"));
    }
}

TEST_CASE(provenance_preamble_is_restored_after_a_run) {
    const std::string before = CsvWriter::preamble();
    TempPrefix t;
    CHECK(run({"--model", "voter", "--dims", "6x6", "--steps", "2", "--seed", "9", "--csv", t.prefix}).status == 0);
    CHECK(CsvWriter::preamble() == before);
}
