#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TestFramework.hpp"
#include "driver/Driver.hpp"

// Stationary runs state how uncertain their error bars are: an error estimated
// from k independent groups (chains, or jackknife blocks of one chain) is
// itself uncertain by about 1/sqrt(2 (k - 1)).

using namespace driver;

namespace {

std::string output_of(std::vector<std::string> args) {
    args.insert(args.begin(), "odsp");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::ostringstream out, err;
    std::streambuf* o = std::cout.rdbuf(out.rdbuf());
    std::streambuf* e = std::cerr.rdbuf(err.rdbuf());
    const int status = run_driver(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    CHECK(status == 0);
    return out.str();
}

std::size_t count(const std::string& text, const std::string& part) {
    std::size_t n = 0;
    for (std::size_t p = text.find(part); p != std::string::npos; p = text.find(part, p + 1)) ++n;
    return n;
}

} // namespace

TEST_CASE(error_bar_uncertainty_formula) {
    CHECK(std::abs(error_bar_uncertainty(10) - 1.0 / std::sqrt(18.0)) < 1e-15);   // 24%
    CHECK(std::abs(error_bar_uncertainty(20) - 1.0 / std::sqrt(38.0)) < 1e-15);   // 16%
    CHECK(std::isnan(error_bar_uncertainty(1)));
}

TEST_CASE(error_bar_uncertainty_is_printed) {
    const std::vector<std::string> base = {"--model", "ising", "--temperature", "3", "--dims", "8x8", "--task", "stationary",
                                           "--burn-in", "10", "--samples", "200", "--seed", "1"};
    std::vector<std::string> chains = base;
    chains.insert(chains.end(), {"--replicas", "10"});
    const std::string c = output_of(chains);
    CHECK(count(c, "error bars from 10 chains are themselves uncertain by about +-24%") == 1);
    CHECK(count(c, "more --replicas") == 1);

    const std::string one = output_of(base);   // one chain, 20 jackknife blocks
    CHECK(count(one, "error bars from 20 jackknife blocks are themselves uncertain by about +-16%") == 1);

    std::vector<std::string> sweep = chains;
    sweep.insert(sweep.end(), {"--sweep", "temperature=2.5,3,3.5", "--threads", "3"});
    CHECK(count(output_of(sweep), "error bars from 10 chains are themselves uncertain by about +-24%") == 1);   // once
}
