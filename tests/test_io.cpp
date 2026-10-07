#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "TestFramework.hpp"
#include "CsvWriter.hpp"
#include "Cli.hpp"

namespace fs = std::filesystem;

namespace {

// Fresh scratch directory under the system temp dir, removed on destruction.
struct ScratchDir {
    fs::path path;
    ScratchDir() {
        std::random_device rd;
        path = fs::temp_directory_path() / ("odsp_test_" + std::to_string(rd()));
        fs::create_directories(path);
    }
    ~ScratchDir() { std::error_code ec; fs::remove_all(path, ec); }
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

} // namespace

TEST_CASE(csv_writer_writes_header_and_rows) {
    ScratchDir dir;
    const fs::path file = dir.path / "nested" / "sub" / "out.csv";   // parents are created
    {
        CsvWriter csv(file.string(), {"model", "sweep", "rho"});
        CHECK(csv.enabled());
        csv.row("voter", 1, 0.5);
        csv.row("majority", 20, 0.25);
    }
    CHECK(read_file(file) == "model,sweep,rho\nvoter,1,0.5\nmajority,20,0.25\n");
}

TEST_CASE(csv_writer_keeps_full_double_precision) {
    ScratchDir dir;
    const fs::path file = dir.path / "precision.csv";
    const double x = 0.1 + 0.2;   // 0.30000000000000004
    {
        CsvWriter csv(file.string(), {"x"});
        csv.row(x);
    }
    std::ifstream in(file);
    std::string header, value;
    std::getline(in, header);
    std::getline(in, value);
    CHECK(std::stod(value) == x);   // round-trips exactly
}

TEST_CASE(csv_writer_uses_shortest_float_form) {
    ScratchDir dir;
    const fs::path file = dir.path / "short.csv";
    {
        CsvWriter csv(file.string(), {"a", "b", "c", "d"});
        csv.row(0.3, 1e-20, 2.0, 0.1f);
    }
    CHECK(read_file(file) == "a,b,c,d\n0.3,1e-20,2,0.1\n");
}

TEST_CASE(csv_writer_with_empty_path_is_disabled) {
    CsvWriter csv("", {"a", "b"});
    CHECK(!csv.enabled());
    csv.row(1, 2);   // must be a harmless no-op
}

TEST_CASE(csv_writer_reports_unwritable_path) {
    ScratchDir dir;
    // A directory cannot be opened as a file.
    CHECK_THROWS_AS(CsvWriter(dir.path.string(), {"a"}), std::runtime_error);
}

TEST_CASE(cli_csv_path_follows_the_prefix) {
    CliOptions opt;
    CHECK(!opt.csv_enabled());
    CHECK(opt.csv_path("rho").empty());

    opt.csv_prefix = "results/run1";
    CHECK(opt.csv_enabled());
    CHECK(opt.csv_path("rho") == "results/run1_rho.csv");
}

TEST_CASE(cli_extra_option_values) {
    CliOptions opt;
    CHECK(opt.get("--edges").empty());
    opt.extra["--edges"] = "net.txt";
    CHECK(opt.get("--edges") == "net.txt");
}
