#ifndef ODSP_DRIVERTESTHELPERS_HPP
#define ODSP_DRIVERTESTHELPERS_HPP

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "TestHelpers.hpp"
#include "driver/Driver.hpp"

// Helpers for tests that run the odsp driver and read the files it writes.

namespace driver_test {

// Runs the driver with `args` (without the program name), capturing stdout and
// stderr into *output when given. Returns the exit status.
inline int run(std::vector<std::string> args, std::string* output = nullptr) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = driver::run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    if (output) *output = out.str() + err.str();
    return status;
}

inline int run_quiet(std::vector<std::string> args) { return run(std::move(args)); }

// A unique output prefix in the temporary directory; every file starting with
// it is removed when the object goes out of scope.
struct TempFiles {
    std::string stem;
    explicit TempFiles(const std::string& tag = "odsp_test_") {
        std::random_device rd;
        stem = (std::filesystem::temp_directory_path() / (tag + std::to_string(rd()) + "_" + std::to_string(rd()))).string();
    }
    TempFiles(const TempFiles&) = delete;
    TempFiles& operator=(const TempFiles&) = delete;
    ~TempFiles() {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path dir = fs::path(stem).parent_path();
        const std::string name = fs::path(stem).filename().string();
        for (const auto& f : fs::directory_iterator(dir, ec))
            if (f.path().filename().string().rfind(name, 0) == 0) fs::remove(f.path(), ec);
    }
    std::string path(const std::string& suffix) const { return stem + suffix; }
    // Writes `text` to stem + suffix and returns that path.
    std::string write(const std::string& suffix, const std::string& text) const {
        std::ofstream(stem + suffix) << text;
        return stem + suffix;
    }
};

// Header names of a CSV (after the '#' preamble).
inline std::vector<std::string> header_of(const std::string& path) {
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::vector<std::string> names;
    std::stringstream hs(line);
    while (std::getline(hs, cell, ',')) names.push_back(cell);
    return names;
}

// A cell as a number; empty, "nan" or text cells (e.g. a swept "tie" value) as NaN.
inline double number(const std::string& cell) {
    try {
        std::size_t used = 0;
        const double v = std::stod(cell, &used);
        return used == cell.size() ? v : std::nan("");
    } catch (const std::exception&) {
        return std::nan("");
    }
}

// Columns by name (every header name present, possibly empty); non-numeric
// cells parse as NaN.
inline std::map<std::string, std::vector<double>> read_csv(const std::string& path) {
    const std::vector<std::string> names = header_of(path);
    std::istringstream in(read_without_comments(path));
    std::string line, cell;
    std::getline(in, line);
    std::map<std::string, std::vector<double>> cols;
    for (const auto& n : names) cols[n];
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        for (std::size_t k = 0; std::getline(ls, cell, ',') && k < names.size(); ++k)
            cols[names[k]].push_back(number(cell));
    }
    return cols;
}

}  // namespace driver_test

#endif
